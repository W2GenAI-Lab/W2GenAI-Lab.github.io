## 视频扩散模型的主要开销在注意力

以 Wan2.1-14B 生成 720p、81 帧视频为例：每层自注意力处理 75,600 个视频 token，40 层，每个去噪步有条件和无条件两次前向，共 50 步。在 B200 上，FA4 稠密计算一次这样的自注意力约 79 ms，整段视频约 470 s，大部分时间花在注意力上。HunyuanVideo 720p、129 帧的序列更长，有 118,800 个视频 token。

块稀疏注意力只计算一部分 (query tile, key 块) 组合，其余跳过。Sol-Attn、PISA 等免训练方法在 Wan 上只计算 12.5–18% 的块。这些方法仍有三项开销：

1. **选块需要单独计算一遍**：通常先把 Q、K 按块平均，算一张粗略的打分图，再挑块。
2. **kernel 内部仍是在线 softmax**：每处理一个块，都要更新每行的最大值，并重新缩放累加器。
3. **被跳过的块要么直接丢掉，要么用很粗的近似补回**：丢掉会让 softmax 分母偏小，粗近似又会系统性地低估这部分的贡献。

## 观察：相邻两个去噪步之间，每行注意力的 LSE 几乎不变

LSE（log-sum-exp）指每一行 softmax 分母的对数：

$$L_i = \log \sum_j e^{s_{ij}}$$

我们在 Wan2.1-1.3B 的稠密运行里，把每次自注意力的 LSE 与上一步同一层、同一位置的 LSE 相减（自然对数）：

| 去噪步 | 变化量中位数 | 99 分位 |
|---|---|---|
| 1 | 0.761 | 2.88 |
| 10 | 0.100 | 0.56 |
| 20 | 0.058 | 0.31 |
| 30 | 0.049 | 0.26 |
| 40 | 0.064 | 0.31 |

第 10 步以后，中位数在 0.05–0.10 之间。也就是说，在计算这一步的注意力之前，每一行 softmax 的分母已经可以从上一步得到，误差约 5–10%。

EchoSparse 为每一层、每个 CFG 分支保存上一步的 LSE，并在同一个注意力 kernel 里用在三个地方。名字里的 Echo 指上一步的统计量在下一步被再次使用。

<figure class="wide">
  <img src="assets/echosparse-explainer.gif" alt="动画：EchoSparse 的一层注意力，从沿用上一步的 LSE、选块、按聚类得到的代表，到统一的归一化和 B200 上的融合 kernel" loading="lazy">
  <figcaption>一层注意力的玩具示例：沿用上一步的 LSE、选块、聚类代表、统一归一化、B200 上的融合 kernel。</figcaption>
</figure>

## 用法一：上一步的 LSE 作为固定的 softmax 偏移量

FlashAttention 一类的 kernel 用“在线 softmax”保证数值稳定：每处理一个 key 块，先更新每行的最大值 \(m\)，再用 \(e^{m_\text{old} - m_\text{new}}\) 缩放已经累加的分子和分母。

EchoSparse 直接用上一步的 LSE \(a_i\) 作偏移：

$$O_i = \frac{\sum_j e^{s_{ij} - a_i}\, v_j}{\sum_j e^{s_{ij} - a_i}}$$

\(a_i\) 在分子分母里同时出现，会被约掉，所以无论它准不准，结果都是精确的。它只需要让指数值不溢出、不下溢。kernel 因此不求行最大值，也不重新缩放。

如果某个 tile 的行和超出安全范围（以 2 为底的对数超过 ±60），这个 tile 改用在线 softmax 重算。实测约 0.5% 的 tile 需要重算。在相同块列表下，这一项把主 kernel 的耗时从 22.0 ms 降到 11.7 ms（B200，Wan2.1-14B 的形状）。

## 用法二：计算选中块时顺带得到下一步的块概率

偏移量接近真实 LSE 时，\(e^{s_{ij} - a_i}\) 已经接近 softmax 概率。kernel 在计算选中块时，把这些值按 64 token 的 key 块、128 行的 query tile 累加，就得到这一步每个块实际占的注意力概率。没有算到的块，用块平均 key 在每个 tile 的 16 行上估计，并用同一 tile 里精确计算过的块校准。

下一步，每个 128 行的 query tile 按这张概率图取前 10% 的块。选块不需要单独的打分计算。

这种选法比池化打分准确得多。离线测试中（Wan2.1-1.3B 的真实 Q/K/V，每个 tile 取 10% 的块，有补偿），单次调用的相对误差为：

| 选块方式 | 相对误差 |
|---|---|
| 上一步的真实块概率 | 0.110 |
| 本步的真实块概率 | 0.110 |
| 池化打分取前 10% | 0.188 |
| Sol-Attn 的阈值规则（实际取 12% 的块） | 0.158 |

用上一步的概率选块，与用本步的概率选块误差相同。

还有一个结果：在 Wan 上，每 2 步、4 步甚至 20 步才刷新一次选块，与每步刷新相比，视频 PSNR 相差不超过 0.55 dB。说明在双向生成里，选块本身跨步就很稳定。EchoSparse 每步都刷新，是因为在这个设计里刷新不需要额外计算。

## 用法三：被跳过的块仍然进入同一个 softmax

每个被跳过的块，用它的平均 key \(\bar k_b\) 和平均 value \(\bar v_b\) 作代表，权重是块内 token 数 \(n_b\)：

$$n_b\, e^{\,q \cdot \bar k_b / \sqrt{d} \,-\, a_i}\; \bar v_b$$

因为所有项共用同一个偏移 \(a_i\)，精确块和补偿项可以直接加在同一个分子分母里，最后统一做一次除法。

这一项对画质影响最大：去掉补偿后，14B 的 PSNR 下降 1.19 dB，1.3B 下降 5.37 dB。

但“先平均再取指数”总是小于“先取指数再平均”，块内 key 越分散，低估越多。所以按块概率估计，落选块中最重要的三分之一，每块改用 4 个代表。这 4 个代表用 k-means 求得，距离在“查询度量”下计算：

- 用本次调用的 query 求 \(G = Q^\top Q / (L d) + \lambda I\)，Cholesky 分解为 \(G = M M^\top\)；
- 在 \(X = K M\) 的坐标里聚类。两个 key 在这个坐标下的距离平方，约等于所有 query 看到的 logit 差的平方均值；
- 每个 token 的权重来自 64 个探针 query（均匀取的 query 块均值）在块内的 softmax；
- 从 4 段连续子块初始化，最多 6 轮 Lloyd 迭代。代表仍是簇内的普通均值，权重是簇的大小。

4 个代表比 1 个代表在 14B 和 1.3B 上分别提高 0.72 dB 和 0.98 dB；改用查询度量后，又分别提高 0.35 dB 和 0.39 dB，在 HunyuanVideo 上提高 1.14 dB。

## 试过但没有采用的做法

- **用已知 LSE 校准补偿的分母。** 思路是：被跳过部分的概率约等于 1 减去已算部分的概率，分母可以直接算出来。实测在所有密度下都比上面的补偿差。原因是 LSE 的步间变化与被跳过的概率是同一量级，用它来推算被跳过部分，误差太大。
- **按误差预算 ε 决定每行算多少块。** 两个问题：池化 key 的估计严重低估块概率（名义上跳过 5%，实际跳过 30–36%）；统一的 ε 会把更多块分给注意力分散的头，而这些头恰好是补偿近似得最好的头。每个 tile 按概率图取相同块数，误差与按头全局分配相同，负载也天然均衡。
- **二阶方差修正。** 在补偿项上加块内方差的修正，平均误差更大，个别头上会估计过大。

## 完整流程

- 前 10 个去噪步用 FA4 稠密计算，与 Sol-Attn 论文的设置相同。第 10 步（最后一个稠密步）让所有块都经过 EchoSparse 的 kernel，得到精确输出和第一张块概率图。
- 之后每一步，每个 128 行的 query tile 精确计算 10% 的 key 块，其余块按上面的方式补偿。
- 有 CFG 时，无条件分支沿用有条件分支同一步的选块，但保存自己的 LSE。
- Wan 的视频 token 按三维 Morton 顺序排列，一个 64 token 的块对应一个 4×4×4 的时空小块。HunyuanVideo 的文本 token 放在序列末尾，每个 tile 都精确计算；文本 token 自己的 query 行用 FA4。MiniMax-H3 的文本、条件和音频前缀按同样方式处理。

## 实现

**B200（SM100）**：kernel 用 CuTe DSL 编写，使用 tcgen05 MMA 和 tensor memory，按 FA4 的方式做 warp 分工，每个 SM 放两个 CTA。精确块、块平均代表和 k-means 代表在同一个 kernel 里、同一个 softmax 中计算。Wan2.1-14B 形状下，每次自注意力调用的耗时：

| 方法 | 每次调用 |
|---|---|
| FA4 稠密 | 79.1 ms |
| Sol-Attn（τ = 1.0，论文的阈值估计） | 35.4 ms |
| PISA（12.5%） | 34.6 ms |
| EchoSparse（10%） | 18.1 ms |

**RTX 5090（SM120）**：没有 tcgen05 和 tensor memory，kernel 用 Triton 编写，使用普通的 tensor core 点积，计算的量和精度与 B200 相同。精确块按两个 32 key 的半块计算，补偿每次 32 列。显存只有 32 GB，所以每层只保留各分支的 LSE 和下一步的选块（两张位图），块概率图等作为各层共用的临时空间；14B、HunyuanVideo、H3 的权重放在 CPU 内存里，按组拷到 GPU。

## 结果

### RTX 5090：完整视频生成

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-rtx5090-dark.png">
  <img src="assets/results-rtx5090.png" alt="RTX 5090：EchoSparse 相对 FlashAttention-4 的加速（整段视频和单次稀疏步前向）以及对 FA4 的 PSNR，并列 B200 上同一提示词的 PSNR" loading="lazy">
</picture>

每个模型取 B200 评测里的第 0 个提示词，种子相同，EchoSparse 和 FA4 稠密各在一张 GPU 上跑一遍完整的 50 步生成。Wan2.1-14B、HunyuanVideo、MiniMax-H3 的权重放在 CPU 内存里，按组拷到 GPU；1.3B 直接放在 GPU 上。5090 上每个提示词的 PSNR 与 B200 上同一提示词相当或更高，换用 Triton kernel 后画质没有损失。HunyuanVideo 的单次稀疏步前向比 FA4 快 3.02 倍，整段视频快 1.99 倍。

### B200：完整视频生成

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-b200-dark.png">
  <img src="assets/results-b200.png" alt="B200：四个模型上 EchoSparse、PISA、Sol-Attn 相对 FlashAttention-4 的端到端加速和对 FA4 的 PSNR" loading="lazy">
</picture>

50 个去噪步，含文本编码和 VAE 解码，各方法的前 10 步都用稠密注意力。三种方法都用各自公开的设置：EchoSparse 计算 10% 的块，Sol-Attn 用论文的阈值估计（τ = 1.0，实际 15–18%），PISA 计算 12.5%。PSNR 以同一提示词、同一种子的 FA4 稠密输出为参考，每个模型 10–25 个提示词。EchoSparse 算的块最少，四个模型上加速都最高；相对 Sol-Attn，PSNR 在 Wan2.1-14B、Wan2.1-1.3B、HunyuanVideo 上分别高 1.10、1.19、2.44 dB（逐提示词配对的置信区间都不含 0）。MiniMax-H3 上三种方法的差别在置信区间之内。

## 局限

- **前 10 个稠密步限制了端到端加速。** Wan2.1-14B 在 B200 上，前 10 步稠密约 94 s，后 40 步里注意力以外的计算约 109 s，两者合计已超过 FA4 总时间的 40%。所以稀疏步本身快 2.11 倍，整段只快 1.70 倍。
- **MiniMax-H3 的收益小。** H3 的注意力更分散，每次前向里注意力以外的部分约 1.6 s，整段只快 1.25 倍，画质与 Sol-Attn、PISA 没有可测的差别。
- **PSNR 衡量的是与稠密输出的接近程度。** VBench 子集上，稀疏方法与 FA4 的差别都很小。
- **适用范围**：B200 或 RTX 5090；batch size 1；head dim 128；4096 个 token 以下的序列直接走稠密。

## 使用

```python
import echosparse

pipe = ...                 # diffusers 的 WanPipeline / HunyuanVideoPipeline / MiniMax-H3 流水线
echosparse.enable(pipe)    # 在设置好 scheduler 之后调用
video = pipe(prompt="A cat surfing at sunset", ...).frames[0]
```

`enable` 识别模型并套用对应的预设（密度 0.10、前 10 步稠密）。完整脚本在 `examples/`，RTX 5090 上加 `--offload` 把大模型的权重放在 CPU 内存里。
