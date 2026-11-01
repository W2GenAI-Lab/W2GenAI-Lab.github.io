视频扩散模型的 self-attention 序列很长。Wan2.1-14B 生成 720p、81 帧的视频，单层 attention 要处理约 7.56 万个 token，一次前向要过几十层，一段视频要去噪 50 步。对某个 query 来说，大部分 KV 块的贡献都很小，看上去大可以丢掉。

在 LLM 里，这条路已经有不少成功的设计：滑动窗口（SWA）、各种稀疏访问，还有 GDN、KDA 这类线性 attention 的状态更新。不过它们大多是训练进模型结构里的。我们面对的是另一个更挑剔的设定：拿一个已经用 dense softmax 训练好的视频 DiT，不重训，直接把 attention 换掉。视频里一个 query 的概率质量常常散在空间和时间方向的很多块上，单独看每一块都很弱，合起来却不小；把它们直接清零，变的不只是分子，还有分母，也就是整行的加权方式。在这种 training-free 替换里，硬丢弃往往先在画面上露馅。这并不是说 trainable 的 SWA 或线性 attention 在视频上行不通，不少主流视频模型已经加入了局部窗口一类的机制，只是那是另一条路线。

所以我们选择把不重要的块**算得粗一点，而不是一律清零**。这个思路不是我们首先提出的。EchoSparse 是在 Haopeng Li 等人的 PISA（[论文](https://arxiv.org/abs/2602.01077)，[代码](https://github.com/xie-lab-ml/piecewise-sparse-attention)）和 Sol-Attn（[项目页](https://nvlabs.github.io/Sana/Sol-Attn/)，[论文](https://arxiv.org/abs/2607.24027)）的基础上继续做的。PISA 把常见的 keep-or-drop 换成 exact-or-approximate：重要块精确计算，其余块用块内 Taylor 展开近似。Sol-Attn 同样给未选中的块加零阶补偿，并且复用同一个 token 对池化块的 score tile 来做路由。两者都把补偿项放进 attention 的分子和分母，也都已经把多个块的代表向量拼起来做 GEMM。

EchoSparse 在这条线上又往前走了三步：

1. **时间上的统计量。** 去噪相邻两步里，每个 attention 行的统计量变化不大，所以上一步留下的 LSE（log-sum-exp）和各块的质量（block mass）可以在本步一起用：LSE 当作 softmax 的固定偏移，块质量给出“哪些块相对重要”的排序。
2. **多层级近似，而不是丢弃。** 一个块的代表数 $m_b$ 不必固定为 1。从 1、2、4、8 一直到 64（也就是原始 token），这是一组由粗到细的分辨率档位。当前实现用了其中 1、4、64 三档，把列预算花在更值得的块上。
3. **执行与布局。** 代表项天然零碎：一个块 1 列，另一个块 4 列。它们可以拼成更宽的 GEMM，但拼接、搬运和同步本身也有成本，需要按 GPU 的方式组织数据和流水线。

说得更直白一点。第一，FlashAttention 内循环里每个 KV tile 都要做的“行最大值归约、缩放系数、旧累积量重缩放”，有了固定偏移可以整段删掉。第二，代表数从 1 到 4 再到 64，是一个按块调节精度和开销的旋钮。第三，许多 $(B_q\times d)(d\times1)$、$(B_q\times d)(d\times4)$ 这样的小乘法要聚成足够宽的 tile，再让搬运和计算重叠起来。代码在这里：**[W2GenAI-Lab/EchoSparse-Attn](https://github.com/W2GenAI-Lab/EchoSparse-Attn)**。

![EchoSparse 单层 attention 讲解动画](assets/echosparse-explainer.gif)

先交代范围。主线目前支持 B200（SM100）和 RTX 5090（SM120），H100（SM90）和 RTX 4090 的正式适配还在 TODO 里。文中 grouped 代表描述符、TMA 与三级流水的内容来自 H100 **实验分支**，没有合入主线，涉及的地方我会标出来。

## 1. 一个 KV 块可以被算成什么样

先统一符号。全序列长度为 $N$，key 维度为 $d$，value 维度为 $d_v$。query 按 $B_q$ 行切成 tile，KV 按 $B_k$ 个 token 切成块，块 $b$ 用 $m_b$ 个代表行参与计算。本文的实现里 $B_q=128$、$B_k=64$、$d=d_v=128$。对一个 query tile $Q\in\mathbb R^{B_q\times d}$ 和一个块 $K_b\in\mathbb R^{B_k\times d}$、$V_b\in\mathbb R^{B_k\times d_v}$，精确计算是

$$
S_b=\frac{QK_b^\top}{\sqrt d}\in\mathbb R^{B_q\times B_k},\qquad
P_b=\exp\bigl(S_b-c\,\mathbf 1^\top\bigr),\qquad
\mathcal N\mathrel{+}=P_bV_b,\qquad
\mathcal D\mathrel{+}=P_b\mathbf 1 .
$$

这里 $c\in\mathbb R^{B_q}$ 是每行一个的偏移，$\mathcal N\in\mathbb R^{B_q\times d_v}$ 是未归一化的分子累积，$\mathcal D\in\mathbb R^{B_q}$ 是分母；所有块扫完后输出 $O=\mathrm{diag}(\mathcal D)^{-1}\mathcal N$。分子分母用花体，是为了不和序列长度 $N$ 混淆。

常说的 $(B_q\times d)(d\times B_k)(B_k\times d_v)$ 只是一种**尺寸简写**。两次矩阵乘之间夹着逐元素指数和整行共享的归一化，它不是一个可以随意结合的三矩阵乘积。按这个简写，三种处理方式是：

- **exact**：$(B_q\times d)(d\times B_k)$ 得到 $B_qB_k$ 个 logit，做 $B_qB_k$ 次指数，再乘 $(B_q\times B_k)(B_k\times d_v)$。读 $B_k(d+d_v)$ 个元素，约 $2B_qB_k(d+d_v)$ FLOPs。
- **drop**（SWA、块稀疏里的落选块）：相当于 $B_k\to0$，这个块对 $\mathcal N$ 和 $\mathcal D$ 都没有贡献。不花钱，但分母里也少了它那一份。
- **一个摘要**：$(B_q\times d)(d\times1)$ 加上 $(B_q\times1)(1\times d_v)$。读 $d+d_v$ 个元素，约 $2B_q(d+d_v)$ FLOPs，是 exact 的 $1/B_k$。

### 零阶摘要为什么必须带 count

只看一行 query $q$。块里有 $n_b$ 个有效 token（通常 $n_b=B_k$，尾块可能更少），令

$$
\bar k=\frac1{n_b}\sum_{j}k_j,\qquad \bar v=\frac1{n_b}\sum_j v_j,\qquad
\bar s=\frac{q^\top\bar k}{\sqrt d},\qquad
\delta_j=\frac{q^\top(k_j-\bar k)}{\sqrt d}.
$$

于是 $s_j=\bar s+\delta_j$，而且 $\sum_j\delta_j=0$。块的精确贡献可以写成

$$
\sum_j e^{s_j}=e^{\bar s}\sum_j e^{\delta_j},\qquad
\sum_j e^{s_j}v_j=e^{\bar s}\sum_j e^{\delta_j}v_j .
$$

零阶近似取 $e^{\delta_j}\approx1$：

$$
\mathcal D_b^{(0)}=n_b\,e^{\bar s}=e^{\bar s+\ln n_b},\qquad
\mathcal N_b^{(0)}=e^{\bar s}\sum_j v_j=n_b\,e^{\bar s}\,\bar v .
$$

所以一个代表不是“随便挑一个 token”，而是三元组 $(\bar k,\bar v,n_b)$。它替 $n_b$ 个 token 说话，就要在分子和分母里各占 $n_b$ 份。和精确 token 混在一起时，最方便的写法是把 count 并进 logit：代表的 logit 是 $\bar s+\ln n_b$。实现里用 $\exp_2$，加的是 $\log_2 n_b$，满块就是 $\log_2 64=6$。漏掉这一项，整块的权重就被压低 64 倍，相当于 logit 差了 $\ln 64\approx4.16$ nats，归一化也就错了。

### 一阶、二阶：多存了什么，又贵在哪里

零阶把 $e^{\delta_j}$ 的变化全部扔掉了，自然会想到往下展开 $e^{\delta}=1+\delta+\tfrac12\delta^2+\cdots$。

到一阶，分母的新项是 $\sum_j\delta_j=0$，消失了；分子却不消失：

$$
\sum_j\delta_j v_j=\frac{H^\top q}{\sqrt d},\qquad
H=\sum_j (k_j-\bar k)\,v_j^\top\in\mathbb R^{d\times d_v},
$$

$$
\mathcal N_b^{(1)}=e^{\bar s}\Bigl(\sum_j v_j+\frac{H^\top q}{\sqrt d}\Bigr),\qquad \mathcal D_b^{(1)}=n_b\,e^{\bar s}.
$$

$H$ 记录的是块内 key 的偏移和 value 之间的相关性，这正是一个均值代表丢掉的东西。到二阶，又多出

$$
C=\sum_j (k_j-\bar k)(k_j-\bar k)^\top\in\mathbb R^{d\times d},\qquad
T=\sum_j (k_j-\bar k)\otimes(k_j-\bar k)\otimes v_j\in\mathbb R^{d\times d\times d_v},
$$

$$
\mathcal D_b^{(2)}=e^{\bar s}\Bigl(n_b+\frac{q^\top Cq}{2d}\Bigr),\qquad
\mathcal N_b^{(2)}=\mathcal N_b^{(1)}+\frac{e^{\bar s}}{2d}\,T(q,q,\cdot).
$$

问题出在大小上。下表是**为每个块显式缓存这些矩**的直接做法，取 $B_k=64$、$d=d_v=128$：

| 每块的摘要 | 缓存内容（元素） | 数值 | 每个 query 的应用代价 |
|---|---|---:|---|
| 原始 $K_b,V_b$（exact） | $B_k(d+d_v)$ | 16,384 | $O(B_k(d+d_v))$ |
| 零阶 $\bar k,\bar v,n_b$ | $d+d_v+1$ | 257 | $O(d+d_v)$ |
| 一阶 $H$ | $dd_v$ | 16,384 | $O(dd_v)$ |
| 二阶 $C,T$ | $d^2+d^2d_v$ | 2,113,536 | $O(d^2d_v)$ |

一阶这一行有点讽刺：$dd_v=128\times128$ 恰好等于 $B_k(d+d_v)=64\times256$。在这个形状下，逐块的一阶矩和原始块一样大，对一个 query tile 应用它，也就是 $(B_q\times d)(d\times d_v)$，FLOPs 也和精算这个块相当。二阶的 $T$ 未压缩时有 $d^2d_v=2{,}097{,}152$ 个元素，是原始块的 128 倍；利用两个 key 下标的对称性，也只能降到 $d(d+1)d_v/2\approx106$ 万，仍约是原始块的 64.5 倍。当然这只是直接实现的账，不是所有可能实现的下界：也可以不缓存、回头重读 token，或者用低秩、结构化的近似，但那又要付出新的计算，或者引入新的假设。

PISA 的处理很聪明。它实际的 hybrid 是**逐块零阶加全局一阶**。记 $\mathcal U$ 为某个 query tile 的落选块集合。示意地写（满块、不考虑尾块），理想的逐块一阶项 $\sum_{b\in\mathcal U} e^{\bar s_b}H_b^\top q/\sqrt d$ 被换成 $\bigl(\sum_{b\in\mathcal U} e^{\bar s_b}\bigr)\bar H^\top q/\sqrt d$。求和只遍历 $\mathcal U$，但 $\bar H$ 是在整个序列上预先汇总好的全局一阶统计量，不随每个 query 的 $\mathcal U$ 重新计算；满块等长时可以把它理解成各块 $H_b$ 的平均，生产代码对尾块另有按 count 的缩放。预处理时会先形成每块的 $H_b$ 再归约，但 attention 内循环只需要加载这一个全局矩阵，不必流式搬运每块的 $H_b$。这当然也是近似。至于二阶，PISA 在相关工作里结合 Based、power attention 讨论过 Taylor 特征维度的膨胀；上面那张表是我们自己的整理，用来说明为什么不直接往高阶走，并不代表 PISA 的 kernel 存了逐块的二阶张量。

EchoSparse 走的是另一个方向：**不升阶，而是把块切细**。每一组仍然只用零阶摘要，但组数 $m_b$ 可以变，这是第 3 节的内容。在那之前，先把 softmax 本身的内循环理顺。

## 2. LSE 复用：内循环里到底删掉了什么

FlashAttention 一类的 kernel 按 KV tile 扫描，用 online softmax 维护每行的运行最大值。写成 base-2 的形式（实现里都是 $\exp_2$，本节的 $s_{ij}$ 都已乘过 $\log_2 e$），每来一个 KV tile，每行 $i$ 要做：

$$
\begin{aligned}
m_i^{\text{new}}&=\max\bigl(m_i,\ \max_j s_{ij}\bigr),\qquad
\alpha_i=2^{\,m_i-m_i^{\text{new}}},\qquad
p_{ij}=2^{\,s_{ij}-m_i^{\text{new}}},\\
\ell_i&\leftarrow\alpha_i\,\ell_i+\sum_j p_{ij},\qquad
O_i\leftarrow\alpha_i\,O_i+\sum_j p_{ij}v_j .
\end{aligned}
$$

如果事先有一个每行固定的偏移 $\hat a_i$，循环就只剩

$$
p_{ij}=2^{\,s_{ij}-\hat a_i},\qquad \ell_i\leftarrow\ell_i+\sum_jp_{ij},\qquad O_i\leftarrow O_i+\sum_jp_{ij}v_j .
$$

EchoSparse 取 $\hat a_i$ 为**上一去噪步同层、同 CFG 分支、同一行的 log2 LSE**。下面是去掉索引、mask 和块质量写回之后的 Triton 风格对比（对应主线的[固定偏移路径](https://github.com/W2GenAI-Lab/EchoSparse-Attn/blob/fe8c7452916c74fe9243d65be070b4d9c378e080/echosparse/kernels/sm120.py#L35-L59)和 [online 回退路径](https://github.com/W2GenAI-Lab/EchoSparse-Attn/blob/fe8c7452916c74fe9243d65be070b4d9c378e080/echosparse/kernels/sm120.py#L146-L166)），形状取 `BM=128`、`BN=64`、`D=Dv=128`。主线 SM120 实际把一个 64-token 块拆成两个 32-key 的半块来走，H100 实验分支是 64 宽。exact token 的 `log_count` 为 0，代表项为 $\log_2 n$，同一段代码都适用。这不是可以直接编译的完整 kernel。

```python
# q: [BM, D]  k: [BN, D]  v: [BN, Dv]  s2, p: [BM, BN]  m, l, prior2: [BM]  acc: [BM, Dv]
# scale2 = log2(e) / sqrt(D); prior2 is a log2 LSE, in the same units as exp2

# online softmax: m = -inf, l = 0, acc = 0
for k, v, log_count in kv_tiles:
    s2 = tl.dot(q, tl.trans(k)) * scale2 + log_count[None, :]
    block_max = tl.max(s2, axis=1)            # [BM] row reduction
    m_new = tl.maximum(m, block_max)          # [BM]
    alpha = tl.exp2(m - m_new)                # [BM] extra exp2
    p = tl.exp2(s2 - m_new[:, None])          # [BM, BN] waits for m_new
    l = l * alpha + tl.sum(p, axis=1)         # rescale old denominator
    acc = acc * alpha[:, None]                # rescale old O: BM*Dv FP32
    acc = tl.dot(p.to(tl.bfloat16), v, acc)
    m = m_new
out = acc / l[:, None]
new_lse2 = m + tl.log2(l)

# fixed offset: one load of the previous LSE per query row, before the KV loop
prior2 = tl.load(PREV_LSE2 + rows * H + head)  # [BM]
l = tl.zeros((BM,), tl.float32)
acc = tl.zeros((BM, Dv), tl.float32)
for k, v, log_count in kv_tiles:
    s2 = tl.dot(q, tl.trans(k)) * scale2 + log_count[None, :]
    p = tl.exp2(s2 - prior2[:, None])         # [BM, BN]
    l += tl.sum(p, axis=1)
    acc = tl.dot(p.to(tl.bfloat16), v, acc)   # accumulate straight into O
out = acc / l[:, None]                        # current denominator
new_lse2 = prior2 + tl.log2(l)
```

对照着数一下，每个 KV tile 删掉的是：

- **行最大值归约**：128 行、每行长度 64 的 max，逻辑上约 $128\times63$ 次比较和相应的 lane 间归约，外加 128 次运行最大值的比较。
- **缩放系数**：128 个 $\alpha_i=2^{m_i-m_i^{\text{new}}}$。注意，$128\times64=8192$ 个 $p_{ij}$ 的 $\exp_2$ 一个也没少。
- **旧累积量的重缩放**：分母 128 个元素，输出累积器 $128\times128=16{,}384$ 个 FP32 元素。

这些是逻辑上的工作量，不是机器指令数，编译器可能会向量化或合并。16,384 个 FP32 占 64 KiB，但累积器在片上，这也不意味着每轮省下 64 KiB 显存流量。只是它会随 KV tile 的数量反复发生：密度 0.10 时每个 query tile 平均约 118 个 exact 块，单 exact 循环就是约 190 万次累积器乘法，压缩列的循环还要另算。新增的成本只有循环开始前每行读一个 FP32 的历史 LSE，一个 tile 512 字节。仍然要算的也写清楚：QK、逐 score 的 $\exp_2$、行和、PV、最后用**本步**分母做的归一化，以及下一步要用的 $\log_2\ell$；循环结束后还有一次标记坏 tile 的范围检查。

比工作量更要紧的是依赖。online 版本里有两条串行链：

```text
online:  S ──> row max ──> m_new ──> exp2(S - m_new) ──> P·V ──┐
                             └──> alpha ──> scale old O ───────┴──> O
fixed:   S ──────────────────────> exp2(S - prior) ──> P·V ──────> O
```

一条是 $p_{ij}$ 要等本 tile 的行最大值归约完才能算；另一条是新的 PV 累加进 $O$ 之前，旧 $O$ 必须先乘完 $\alpha$。QK 和 PV 交给 Tensor Core，max、归约和逐行缩放走普通执行单元和 SFU，吞吐和并行方式都不同，夹在两次矩阵乘之间就会拉长关键路径。固定偏移之后，链只剩 $S\to\exp_2\to PV$，更容易和搬运、下一轮 QK 重叠。（这不是说下一轮 QK 一定要等 $O$ 缩放完，有的 kernel 会提前发 QK；能确定的是上面这两条依赖。）

在 B200 上，这个差别更具体。当前固定偏移的 CuTe DSL 实现把 $O$ 放在 TMEM（tensor memory）里，所有 PV 的 MMA 都直接累加到同一块 `tO` 上，只在第一次清零。循环里 score 从 TMEM 读出、减去固定 LSE、做 $\exp_2$、转成 BF16 写回，然后只加分母；整个循环结束后，才把 $O$ 从 TMEM 读到寄存器，除以本步的行和，写出输出和新的 LSE（[源码](https://github.com/W2GenAI-Lab/EchoSparse-Attn/blob/fe8c7452916c74fe9243d65be070b4d9c378e080/echosparse/kernels/sparse_attention.py#L354-L393)）。中间没有逐轮的 $O$ 行缩放路径。至于和 FA4 相比到底少了多少次 TMEM 读写，需要反汇编或 NCU 归因才能回答，我们没有做，也就不给数字；score 的 TMEM 读取、P 的写回和相应的 barrier 照常存在。

### 为什么固定偏移是精确的

这件事一句话就够：同一组项的分子分母乘上同一个 $2^{-\hat a_i}$，比值不变。

$$
\frac{O_i}{\ell_i}=\frac{\sum_j 2^{s_{ij}-\hat a_i}v_j}{\sum_j 2^{s_{ij}-\hat a_i}}=\frac{\sum_j 2^{s_{ij}}v_j}{\sum_j 2^{s_{ij}}},\qquad
\hat a_i+\log_2\ell_i=\log_2\sum_j2^{s_{ij}} .
$$

关键是最后除的是**本步**累出来的 $\ell_i$，而不是拿上一步的分母替代它。所以 $\hat a_i$ 取什么值都不影响结果，正确性并不依赖前后两步的 LSE 接近。接近影响的是数值范围：记本步的 log2 LSE 为 $\mathrm{lse}^{(2)}_i$，则 $\ell_i=2^{\,\mathrm{lse}^{(2)}_i-\hat a_i}$，偏移越准，$\ell_i$ 越靠近 1，$p_{ij}$ 越不容易上溢或下溢。那它们实际有多接近？

![相邻去噪步的 LSE 变化](assets/fig5_lse_drift_zh.png)

*图 1　Wan2.1-1.3B、稠密 FA4、5 个提示词，同层、同 CFG 分支、同一行的 $|\mathrm{LSE}(t)-\mathrm{LSE}(t-1)|$，自然对数单位。图中是来源报告列出的 7 个汇总点，并不表示只测了 7 步；连线只是引导视线，不是拟合或插值。数据：[docs/REPORT.md §3](https://github.com/Owen718/lse-sparse/blob/05e2dab310bf7c710d8b5276aa588e433823db15/docs/REPORT.md#3-相邻去噪步的-lse-变化13b5-个提示词)（B200 历史开发实验，仓库 Owen718/lse-sparse，commit 05e2dab）。*

中间阶段很安静，中位数只有 0.05 到 0.1 nats；开头（第 1 步中位数 0.76）和最后几步（第 49 步 0.28）明显更大，p99 到了 2.88 和 1.11。报告里还记着个别（层, 头）的极端行，LSE 一步变化上百，原因没查明。所以“LSE 几乎不变”并不是每一步都成立，kernel 也没有假设它成立：循环结束后检查 $\log_2\ell_i$，超出约 $\pm60$ 的 tile 会被标记（README 给的比例约 0.5%），对它选中的 exact 块用上面的 online 递推重跑，再叠加跳过块的均值补偿。这不是重算整个 dense attention，也不会保留四代表那种压缩形式。仓库在相同块列表下测得 kernel 时间约少 47%；这是那一组受控实测，不是说每个 GPU、每个形状都会省 47%。

最后提醒一句：这个恒等式只管“一组固定的项怎么做 softmax”。哪些块精算、哪些块用代表，那是近似，下面才开始。

## 3. 多层级代表：表示、分配、聚类

### 3.1 表示：$m_b$ 是分辨率档位

把第 1 节的零阶摘要推广一步：把块 $b$ 的 token 分成 $m_b$ 组，每组用它自己的 K/V 均值和真实 token 数做零阶摘要。所有参与计算的项，不管是原始 token 还是代表，都写成同一种三元组 $(\tilde k_a,\tilde v_a,n_a)$，进入**同一个**分子和分母：

$$
\tilde o=\frac{\sum_a n_a\,e^{\,q^\top\tilde k_a/\sqrt d}\,\tilde v_a}{\sum_a n_a\,e^{\,q^\top\tilde k_a/\sqrt d}} .
$$

- $m_b=1$：整块一个均值，$n=n_b$；
- $m_b=4$：四个 cluster，各自的 K/V 均值和 count；
- $m_b=64$：每个 token 自成一组，$n=1$，就是 exact。

中间的 2、8、16、32 都是合理的设计点，从整块均值到原始 token，是一串离散的分辨率档位。**当前代码只实现了 1、4、64 三档**，没有通用的可调代表数接口；构造四代表的 kernel 把 cluster 补到 16 行，只是为了满足 `tl.dot` 的最小尺寸。

有两个地方特别容易写错。一是 $n_a$ 必须同时出现在分子和分母里，空 cluster 和 padding 的 $\ln n_a=-\infty$，权重自然为零。二是 exact 项和压缩项共用同一个分子、同一个分母；如果各自 softmax 再相加，每部分都被归一化成 1，块之间的相对质量就全丢了。

$m_b$ 变大，意味着**分组更细，而不是 Taylor 阶数更高**。每一组仍然只是零阶摘要，但组与组之间的 logit 差异由各自的指数精确表达，被近似掉的只有组内偏移。看一个只有 4 个 token 的玩具例子：logits $s=(0,0,2,2)$，$v=(1,1,0,0)$。精确输出是 $2/(2+2e^2)\approx0.119$。整块一个均值时，平均 logit 是 1，$n=4$，输出就是平均 $v$，等于 0.5，差得不是一点半点。分成 $(0,0)$ 和 $(2,2)$ 两组、各 $n=2$，结果和精确值完全一样，因为组内的 logit 相同。一般地，对固定的 query，只要一组内所有 key 的 logit 相同，count 加权的代表就能精确还原这一组；key 完全相同只是一个充分条件。分到每组一个 token，就回到了 exact 块。

代价随 $m_b$ 线性增长：每块读 $m_b(d+d_v)$ 个元素，约 $2B_qm_b(d+d_v)$ FLOPs。误差却不是线性的。直觉上，前几刀最值钱：把块里明显不同的两群 key 分开，误差往往掉得最多；再往后切，组内已经差不多了，收益递减，直到每组一个 token，误差为零。这只是直觉。下面的局部方差分析也推不出一条通用的 $\log m$ 收敛定律。它真正给出的是一个工程判断：**把有限的列预算花在误差大、又被关注得多的块上**。怎么花，是下一小节；为什么组内差异决定误差，是再下一小节。

### 3.2 分配：谁拿 64，谁拿 4，谁拿 1

分配需要一张“块有多重要”的图。对 query tile $a$ 和块 $b$，定义

$$
M_{ab}=\sum_{i\in a}\sum_{j\in b}e^{\,s_{ij}-c_i},\qquad c_i=(\ln2)\,\hat a_i ,
$$

这里回到自然对数单位：$s_{ij}=q_i^\top k_j/\sqrt d$ 是普通 logit，$c_i$ 是把第 2 节的 log2 LSE $\hat a_i$ 换算成自然对数的上一步 LSE；kernel 里的 $p_{ij}=2^{\,s^{(2)}_{ij}-\hat a_i}$（$s^{(2)}_{ij}=s_{ij}\log_2 e$，即第 2 节里的 base-2 logit）与 $e^{s_{ij}-c_i}$ 是同一个数。这是**用历史 LSE 缩放过的、未归一化的块质量**。先验接近真实 LSE 时，它接近 tile $a$ 的各行放在块 $b$ 上的概率之和，但并不自动等于本步精确的归一化概率。

如果每步都单独拿全部块精算打分来得到这张图，省下的又吃回去了。EchoSparse 把它拆成两部分，都在本步顺带产生：

- **exact 块的精确统计**：固定偏移让 $e^{s_{ij}-c_i}$ 本来就处在概率量级，kernel 计算选中块时按 64-token 块累加 $p_{ij}$，得到的就是这些块的 $M_{ab}$，几乎不需要额外工作。
- **落选块的轻量估计**：它们没有精确的 $s_{ij}$。一个 `estimate_skipped` 过程在每个 query tile 只抽样若干行，用块均值估计落选块的质量，再用同一 tile 里已精算的块做校准，校准系数夹在 $[1,64]$ 内。

两部分合起来，再按 tile 汇总 $\exp(c_i-\mathrm{LSE}^{\text{new}}_i)$ 重新缩放，其中 $\mathrm{LSE}^{\text{new}}_i$ 是本步算出的自然对数 LSE（log2 实现里就是 `2^(prior2 - new_lse2)`；LSE 是对数量，要在指数里相减，不能直接相除），就是下一步使用的质量图。

下一步的分配是一个有限预算策略。设共有 $n_{\text{blk}}$ 个块、密度为 $\rho$：质量最大的 $\mathrm{round}(\rho\,n_{\text{blk}})$ 个块加上局部邻居用 exact；接下来一批块，按 README 的说法是落选块里质量最高的约三分之一，用 4 个代表；其余用 1 个均值。**没有任何块被丢掉**，落选块只是降了分辨率。它也不是从某个误差公式逐块解出来的最优分配，只是按质量排序，把更细的分辨率给更重的块。

![时间反馈与统一 attention](assets/fig1_feedback_zh.png)

*图 2　同一份统计量在三处起作用：LSE 做固定偏移，块质量排序决定每块的分辨率，落选块以 $(\tilde k,\tilde v,n)$ 参与同一个 softmax；融合 kernel 的输出又产生下一步的 LSE 和块质量。*

这个回路需要一个起点。每个（CFG 分支，层）各自保存状态，CFG 的后续分支可以复用分支 0 的块选择。默认前 **9 步**直接用 FA4 稠密计算；**第 10 步**让所有块都经过固定偏移的 EchoSparse kernel，输出是精确的，同时建立第一张质量图；从第 11 步开始才进入稀疏步。

![每个去噪步的两路 CFG forward 耗时（历史 B200 实验）](assets/fig6_step_latency_zh.png)

*图 3　历史开发实验（Wan2.1-1.3B，B200，480×832、81 帧、50 步、CFG 6），纵轴是两路 CFG transformer forward 的耗时之和，不是 attention 单独耗时，也不是整段视频。三轮的中位数，浅色带是三轮的最小到最大（n=3），不是置信区间。稀疏组是旧的单均值 LGSF（density 0.10，refine=0，每个分支独立选块），不是当前的四代表或 packed 配置。第 10 步是全块初始化，稀疏阶段从第 11 步开始；第 11 步三轮里有一轮出现了 513 ms，所以那里的色带特别宽。数据：[outputs/results/curve13.json](https://github.com/Owen718/lse-sparse/blob/05e2dab310bf7c710d8b5276aa588e433823db15/outputs/results/curve13.json)。*

### 3.3 聚类：误差从哪里来，怎样让它小

现在回到一组 $C$ 内部，看零阶摘要到底错在哪。设这组有 $n$ 个 token，沿用第 1 节的记号：组均值 $\bar k,\bar v$，$\bar s=q^\top\bar k/\sqrt d$，$\delta_j=q^\top(k_j-\bar k)/\sqrt d$，组内平均 $\mathbb E_C\delta=0$。精确分母和代表分母分别是

$$
Z_C=n\,e^{\bar s}\,\mathbb E_Ce^{\delta},\qquad \widehat Z_C=n\,e^{\bar s},\qquad
\mathbb E_Ce^{\delta}=1+\tfrac12\mathbb E_C\delta^2+O\bigl(\mathbb E_C|\delta|^3\bigr).
$$

一阶项因为用的是普通均值而抵消，分母误差的主项是**组内 logit 方差**。若 $|\delta_j|\le r$，由 Taylor 余项 $e^\delta-1-\delta=\tfrac12\delta^2e^{\xi}$ 可以得到一个明确的局部界：

$$
0\le Z_C-\widehat Z_C\le\frac{n\,e^{\bar s+r}}{2}\,\mathbb E_C\delta^2 .
$$

非负性就是 Jensen 不等式：代表总是低估一组的分母。分子还多了 value：

$$
\mathcal N_C-\widehat{\mathcal N}_C=n\,e^{\bar s}\Bigl[\mathrm{Cov}_C(\delta,v)+\tfrac12\,\mathbb E_C\bigl(\delta^2v\bigr)+\cdots\Bigr].
$$

只看 K 的方差还不够，K 的偏移和 V 的协方差也在里面。而第一项正是组内一阶矩 $H_C^\top q/(n\sqrt d)$，这里有一个很干净的分解。整块的一阶矩可以拆成组内部分加组间部分：

$$
H_b=\sum_C H_C+\sum_C n_C\,(\bar k_C-\bar k_b)(\bar v_C-\bar v_b)^\top .
$$

分成 $m_b$ 组以后，组间那部分不再需要任何 $d\times d_v$ 的矩阵：每组有自己的中心，组间差异由各自的指数在所有阶上精确表达，存储只是 $m_b(d+d_v)$。剩下的误差只来自组内的 $H_C$、组内方差和更高阶项；每组一个 token 时，它们全部为零。这也给了第 1 节那个问题另一个答案：与其花 $dd_v$ 存一个逐块一阶矩，不如花几列把块切开。

这些式子是逐组的局部描述。整行输出还要经过共享的分母，各组的误差会互相牵连，所以推不出整层输出随 $m$ 的某条收敛律。但它说清了聚类该优化什么：**让同一组内 query 看到的 logit 尽量一致**。

第一反应是对 K 直接做欧氏 k-means……可是 attention 只通过 $q^\top k$ 看 key，K 空间里离得远的两个方向，query 未必分得出来。令 $\tilde q=q/\sqrt d$，$G=\mathbb E[\tilde q\tilde q^\top]$，对一个 key 偏移 $\Delta k$（列向量），

$$
\mathbb E_q\bigl[(\tilde q^\top\Delta k)^2\bigr]=\Delta k^\top G\,\Delta k=\|M^\top\Delta k\|^2,\qquad G=MM^\top .
$$

于是对 query 平均之后的组内 logit 方差是

$$
\mathbb E_q\,\mathbb E_C\,\delta^2=\frac1n\sum_{j\in C}\bigl\|M^\top(k_j-\bar k)\bigr\|^2,
$$

恰好是变换坐标 $x_j=M^\top k_j$ 里的 k-means 目标（key 按行堆成矩阵 $K$ 时，就是 $X=KM$）。线性变换把均值映到均值：**对任何固定的分组**，$X$ 里的普通均值就是普通 K 均值的变换。所以在这个理想的方差目标下，“在 $X$ 里聚类”和“写回普通 K 均值”是自洽的。

实现上还有几处细节。$G$ 按 head 用当前的 query 分布构造，不是每个 query tile 单独一个。Cholesky 分解的是 $G_\epsilon=G+\epsilon I$，$\epsilon$ 取平均对角的 $10^{-3}$ 倍，于是

$$
\|M_\epsilon^\top\Delta k\|^2=\Delta k^\top G\,\Delta k+\epsilon\,\|\Delta k\|^2,
$$

在 query 度量上轻轻掺了一点欧氏距离。每个 64-token 块分成 4 组：从四个连续的 16-token 组开始，用 64 个 probe query 给每个 token 一个权重，做加权 Lloyd 迭代，默认最多 6 次。权重只影响**谁分到哪一组**；写回的代表 K、V 是组内的普通均值，$n_a$ 是真实的 token 数，统一公式才成立。实际的加权 Lloyd 在分配时用的是加权中心，它不等于最后写回的普通均值；也就是说，加权 Lloyd 只是借用了上面这个度量来做分组的启发式，找到的是局部解，既不保证最终 attention 误差被精确最小化，也不保证 $m$ 变大时误差逐点单调下降。

## 4. Packing：把 1+1+4 拼成 6 列

先数一数一个 query tile 实际要处理多少列。实验形状 $N=75600$ 有 $n_{\text{blk}}=1182$ 个 64-token 块（padding 后容量 75648）。密度 0.10 时，exact 块约 118 个，约 $118\times64\approx7.6$ 千列；其余约 1064 个块里约三分之一用四代表，约 355 块、共约 1420 列；剩下约 709 块各 1 列。压缩列合计 $1420+709=2129$，与实验记录的平均值吻合。每个 tile 面对约 9.7 千列，不到 dense 的 13%，压缩列只占约五分之一，却是最零碎的：一个块 1 列，另一个块 4 列。

按块各算各的，就是一串 $(B_q\times d)(d\times1)$、$(B_q\times d)(d\times4)$ 的小乘法，离 Tensor Core 一次 MMA 想要的列数差得很远。那为什么不把它们聚成 $(B_q\times d)(d\times[1+1+4])$？用最小的例子推一遍。块 1、块 2 各有一个均值代表，块 3 有四个代表，拼成

$$
K_*=\begin{bmatrix}\bar k_1^\top\\ \bar k_2^\top\\ K_3^{(4)}\end{bmatrix}\in\mathbb R^{6\times d},\qquad
V_*=\begin{bmatrix}\bar v_1^\top\\ \bar v_2^\top\\ V_3^{(4)}\end{bmatrix}\in\mathbb R^{6\times d_v},\qquad
n_*=(n_1,\,n_2,\,n_{3,1},\,\dots,\,n_{3,4}) .
$$

然后

$$
S=\frac{QK_*^\top}{\sqrt d}\in\mathbb R^{B_q\times6},\qquad
W=\exp\bigl(S-c\,\mathbf 1^\top+\mathbf 1\,(\ln n_*)^\top\bigr),
$$

$$
\mathcal N\mathrel{+}=WV_*,\qquad \mathcal D\mathrel{+}=W\mathbf 1,\qquad O=\mathrm{diag}(\mathcal D)^{-1}\mathcal N .
$$

$c$ 仍是第 1 节那个每行的偏移，在 EchoSparse 里是自然对数单位的历史 LSE，即 $c_i=(\ln2)\,\hat a_i$；不过对任何固定偏移，下面的等价性都成立。为什么和分开算一样？$QK_*^\top$ 的 6 列就是三个小矩阵乘各自的列，逐列不变；指数和 count 修正都是逐元素的；把 $W$ 按列写成 $[W_1\ W_2\ W_3]$，就有

$$
WV_*=W_1\bar v_1^\top+W_2\bar v_2^\top+W_3V_3^{(4)},\qquad W\mathbf 1=W_1\mathbf 1+W_2\mathbf 1+W_3\mathbf 1 ,
$$

也就是把三个块的分子、分母贡献加进同一个 $\mathcal N$、$\mathcal D$。这里有两件事缺一不可：$n_*$ 必须跟着列走，同时进入分子和分母；exact token 也加进同一对 $\mathcal N$、$\mathcal D$，全部扫完只除一次。如果把三块各自 softmax 后的输出相加，每块都被归一化成 1，那就完全是另一回事了。

所以，**在相同的稀疏 plan、代表 K/V、multiplicity 和统一归一化下，packing 是一次等价重组**，实数算术下输出相同，不引入新的算法近似；EchoSparse 相对 dense 的近似全部来自上一节的代表压缩。浮点求和的顺序变了，舍入会有细微差别，所以并不是 bitwise 相同。

当然，6 列远不是一个 Tensor Core tile。真正有意义的是跨很多块持续地凑：实验分支每次迭代处理 128 列，均值流一次拼 128 个块的均值，四代表流一次拼 32 个块、每块 4 列。这样每次 MMA 的列基本是满的，只有每条流的最后一次需要 mask。

对照一下已发布的 nonpacked 路径。它是三个循环：一个遍历 exact 块列表；一个以 128 列为单位扫描**全部**块的均值，其中 exact 块和四代表块对应的列被 mask 掉；一个遍历四代表块列表，每块一个 id、展开成 4 列，列表补到固定宽度。在这次测试的计划里，每个 tile 按这个布局要走 1280 个均值列加 1536 个代表列，共 2816 列，而真正有用的平均只有 2129 列，约 76%。被 mask 的列可以不搬数据，但 MMA 按固定形状发射，这些列的发射和计算开销照付。

![packing 的等价重组](assets/fig2_packing_zh.png)

*图 4　示意图，列数不代表真实规模。左右两边是同一组项：5 个均值、8 个四代表列、10 个 exact 块。nonpacked 用三个循环，均值循环里大量列被 mask；grouped packing 把它们整理成两条紧凑的流。*

实验分支的 grouped packing 就是这种重组的一个实现。一个小 kernel 为每个（query tile, head）生成两条紧凑的 descriptor 流：均值流一个块 id 对应一列；四代表流一个块 id 对应 4 列，在片上按 $4\cdot\text{block}+\text{sub}$ 展开地址。和更早的“逐列 packed 流”相比，后者给每个代表列各存一个索引，一个四代表块占 4 个 id，grouped 只存 1 个；nonpacked 的四代表列表本来就是每块一个 id，不在这个比较里。两类来源分在两个循环里，混合 tile 也就不必对两个源张量都做 masked load。

要强调的是，**被打包的是 descriptor 和索引，代表 K/V 本身并没有被复制成一个连续的大张量**，每次迭代仍按索引 gather。为什么不干脆复制过去？这要看搬运的账。

## 5. 搬运与流水：拼接本身并不免费

直觉上，把每个 tile 要用的代表 K/V 复制成一个连续张量，就可以像 exact 块一样整块搬，问题好像就解决了。可复制本身是一次读加一次写，之后 kernel 还得再读一遍。更麻烦的是，选择对每个（query tile, head）都不一样。粗算一下：每个 tile 平均约 2129 列，每列 $d+d_v=256$ 个 BF16，约 1.09 MB；591 个 query tile 乘 40 个 head，按 tile 物化一遍要写约 26 GB。而同一 plan 下 NCU 记录的主 kernel DRAM 读取总共只有约 9.7 GB。这只是量级估计，但足以说明“先拷成连续张量”不能天真地做。

即使数据就在手边，小块搬运也有固定开销：每次发出搬运要准备 descriptor 和地址，要满足对齐，要用 barrier 通知消费者数据到了，还要在 shared memory 里留出缓冲区。负载越零碎，这些固定成本越难摊销。所以要先把有效列凑成像样的 tile，再让搬运和计算重叠，两件事缺一不可。

重叠的收益可以用一个简单模型看。每次迭代装载耗时 $t_\ell$、计算耗时 $t_c$，共 $n$ 次迭代。串行是 $n(t_\ell+t_c)$；假设每次迭代耗时恒定、搬运与计算可以理想重叠，两级流水的稳态每次接近 $\max(t_\ell,t_c)$，总时间大约是

$$
t_\ell+(n-1)\,\max(t_\ell,t_c)+t_c+t_{\text{sync}},
$$

首尾两项是填充和排空，$n=1$ 时正好退回串行的 $t_\ell+t_c$；$t_{\text{sync}}$ 代表没能被隐藏的等待、barrier 等开销。$n$ 小的时候，填充和排空占比就大。按上面的计数，每个 tile 的 exact 循环约 118 次迭代，均值流只有约 6 次（709/128），四代表流约 12 次（355/32）。这也是 exact 循环被先优化的原因：在 Wan2.1-14B 的形状下，按 nonpacked 基线的执行列计，它约占 73%（$7552/(7552+2816)$；按 grouped 的有用列计约 78%），流水线在那里最有用武之地。这只是性能模型，不是实测时序，也不是什么必然的加速定律。

H100 有 TMA（Tensor Memory Accelerator）：用一个 descriptor 描述张量形状，由专门的硬件异步把整块数据搬进 shared memory。exact 块恰好适合。K、V 的布局是 `[N,H,D]`，一个 64-token 块是按原张量步长描述的一块矩形 tile（$H>1$ 时它在 token 方向并不连续），用 block shape 为 `[64,1,D]` 的 descriptor，在行偏移 $64\cdot\text{block}$ 处一次取走。

![三级软件流水的时间线](assets/fig3_pipeline_zh.png)

*图 5　exact 块循环的三级软件流水，纯示意，非实测。第 0 块被消费完、槽位释放之后，第 3 块才能装入同一个槽位；消费者等“数据到了”，装载者等“槽位空了”。图中的 TMA 只用于 exact 块，代表列仍是指针 gather。*

实验分支现在实现的是：**exact 块经 TMA 加载，三级软件流水让搬运与 dot 重叠，代表列按 grouped descriptor 做指针 gather**。三级流水由 Triton 的软件流水调度生成，三个 stage 是环形缓冲区的三个槽位，不是三组分工不同的 warp，也不是三个 kernel。单独打开 TMA 而不加流水几乎没有收益（57.15 ms 对 57.22 ms），两者组合才快。流水再加深也没用，3 到 6 级的时间几乎一样，剩下的停顿主要来自同步，而不是预取不够。

NCU 给了一点机制上的证据。在同一个 Wan2.1-14B plan 上，从三级流水到 TMA 加三级流水，执行指令从 10.42 G 降到 9.79 G，tensor pipe active 从 38.95% 升到 42.69%，barrier 停顿从 17.86% 升到 22.57%，wait 从 21.13% 降到 18.63%，主 kernel 在这次 NCU profiling 里的耗时从 39.06 ms 降到 35.64 ms，这和表里 71.65→44.51 ms 的完整 GPU 流程计时不是同一种测量。地址计算少了，同步依然显著，barrier 和 wait 加起来仍占约四成 warp 周期。tensor pipe active 是 Tensor Core 管线忙碌的时间比例，不是 MFU，我们也没有测 MFU。

下一步想做的是代表流本身的 producer–consumer。设想是这样的：专门的 producer warp 按 grouped descriptor 去 gather 后续的代表列，在 shared memory 的三个槽位里整理成满宽的 tile；consumer warp 只管对已就绪的槽位做 QK、$\exp_2$ 和 PV，累加进同一对 $\mathcal N$、$\mathcal D$；用完的槽位交还 producer，继续装下一批。依赖还是那两类：consumer 等“数据到了”，producer 等“槽位空了”，不能覆盖还在被读的槽位。本质上是一个异步的环形队列，图 5 那条时间线换成两组独立的 warp 来跑。

这一步**还没有实现，也没有 release**。在 Triton 3.6 的 SM90 路径上，`warp_specialize=True` 被接受但没有效果，寄存器（254）、shared memory（131,104 B）和时间都和普通软件流水一样；这个版本也没有 `tl.async_task`，写不出真正的 warp 分工。我们还写过一个显式三槽 cp.async 环形缓冲加 WMMA 的手写原型，8 个 warp 既搬也算，结果是 18.0 ms 对 Triton 的 2.48 ms，而且有正确性 bug，只能算工具链验证。WMMA 的累积器是逐 warp 的片段，表达不了 FlashAttention-3 那种一个 warpgroup 发整块 MMA 的结构，需要手写 wgmma 或 CuTe 内核，那是后续工作。

## 6. H100 与 RTX 5090 的实测

### H100 单次 attention 测试

![H100 单次 attention GPU 流程耗时](assets/fig4_h100_bars_zh.png)

<div class="results">

| 方案 | 耗时 | 相对 dense FA4 | 相对 nonpacked |
|---|---:|---:|---:|
| 稠密 FlashAttention-4 | 163.48 ms | 1.00× | — |
| nonpacked EchoSparse（已发布版） | 71.65 ms | 2.28× | 1.00× |
| grouped descriptor + TMA + 三级流水 | 44.51 ms | 3.67× | 1.61× |

</div>

测量口径：合成 BF16 的 Q/K/V，形状 `[B,N,H,D]=[1,75600,40,128]`，非因果，稀疏密度 0.10，稀疏计划冻结，由两个稀疏方案共享，20 次 CUDA event 计时的中位数；对照是 FlashAttention-4（入口 `flash_attn.cute.interface.flash_attn_func`）。稀疏方案的计时指 GPU 上的整个稀疏流程，包含代表向量构造和 packing，不含 host 端 descriptor 构造、scratch 分配与编译。这是单次 attention，不是整层 Transformer，更不是端到端视频生成；dense 与稀疏计算的项不同（稀疏平均约 118 个 exact 块加 2129 个压缩列），表的后两行是同一份稀疏计划在不同实现下的开销，dense 一行则计算全部原始 token；所以 1.61× 只比较实现，3.67× 还包含稀疏本身的收益；这些倍数也不代表视频生成会有同样的加速。原始记录见 [evidence/h100_attention_benchmark.json](evidence/h100_attention_benchmark.json)（实验分支结果的脱敏副本）。

1.61× 来自 packing 与流水化这一系统优化，3.67× 里还包含稀疏本身相对 dense 的收益。仅供参考：Wan2.1-14B 在 packing 路径上完整生成一次，从 818.60 s 降到 764.57 s（1.07×），这个比较只相对 nonpacked，每个方案各跑一次，也没有 dense 的完整基线。

### 最新 RTX 5090 完整生成结果

RTX 5090 的已发布结果从另一个角度看系统：整段视频生成，attention 之外的耗时也算进去。最新图补上了 Wan2.1-14B 的**整视频 1.39× 加速**，同一模型的单次稀疏步前向则为 1.61×。

![最新 RTX 5090 完整生成结果](assets/results-rtx5090-dark.png)

*最新发布的 RTX 5090 结果。每个模型取 B200 评测的第 0 个提示词及相同 seed，EchoSparse 与稠密 FA4 各在单张 GPU 上跑一次 50 步生成。Wan2.1-14B、HunyuanVideo 和 MiniMax-H3 使用按组 CPU 权重 offload，Wan2.1-1.3B 权重驻留 GPU。左侧分别是整段视频和单次稀疏步前向的加速比，右侧是相对 FA4 的 PSNR，并列同一提示词的 B200 数值。这组数据与上面的 H100 合成单 attention 测试口径不同。*

## 7. 现状与后续

已经在主线里的：固定偏移的 LSE 复用、块质量反馈、1/4/64 三档代表和 query 度量聚类，B200 与 RTX 5090 的结果（含端到端生成）在主线 README 里。只在 H100 实验分支里的：grouped descriptor、exact 块的 TMA 加载和三级软件流水，需要显式打开，尚待集成进主线并补齐多模型 benchmark。还没有实现的：真正独立的 producer–consumer 代表流，以及 2、8、16、32 这些中间档位的代表数。最新主线也还没有正式的 RTX 4090（SM89）后端，需要单独适配与验收，H100 的 TMA 路径不能直接搬到 Ada 上；动画里的历史 RTX 4090 数据属于另一批实验，和本文的 H100 测试不是同一个口径。

回头看，三件事其实是同一个公式的三个侧面。上一步的统计量让 softmax 免去 max 与 rescale，也给分辨率分配提供了排序；count 加权的统一 softmax 让一个块可以用 1、4 或 64 列参与，而不必被清零；也正因为所有项共用一对分子分母，packing 只需要关心“同一组项怎么排”。数学上的等价，是我们敢大胆重排数据和流水线的底气。
