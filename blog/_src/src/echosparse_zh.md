视频扩散模型的 self-attention 序列非常长。比如 Wan2.1-14B 生成 720p、81 帧的视频，单层 attention 要处理约 7.56 万个 token。对一个 query 来说，很多 KV 块的贡献很小。

LLM 上有不少高稀疏度的做法，比如滑动窗口（SWA），以及 GDN、KDA 这类线性 attention，但它们基本是在训练时就进入模型结构的。我们关心的是另一种情况：已经用 dense softmax 训好的视频 DiT，不重新训练，直接替换 attention。视频里一个 query 的概率质量往往分散在时空方向的很多块上，单个块很小，合起来不小。把这些块直接丢掉，分子和分母都会变，整行的加权方式也随之改变，在 training-free 替换中这通常会体现在生成画面上。训练阶段引入 SWA 或线性 attention 的视频模型属于另一类方法。

我们的工作是在 Haopeng Li 等人的 PISA（[论文](https://arxiv.org/abs/2602.01077)，[代码](https://github.com/xie-lab-ml/piecewise-sparse-attention)）和 Sol-Attn（[项目页](https://nvlabs.github.io/Sana/Sol-Attn/)，[论文](https://arxiv.org/abs/2607.24027)）的基础上继续做的。两者都是近似而不是丢弃：PISA 把 keep-or-drop 改成 exact-or-approximate，重要块精确计算，其余块用块内 Taylor 展开近似；Sol-Attn 给未选中的块加零阶补偿，并用同一个 token 对池化块的 score tile 做路由。补偿项都进入 attention 的分子和分母，多个块的代表向量也都已经拼起来做 GEMM。

EchoSparse 在此基础上做了三件事：

1. **复用上一步的统计量。** 去噪相邻两步里，每个 attention 行的统计量变化很小，所以上一步的 LSE（log-sum-exp）和各块的质量（block mass）可以在本步使用。LSE 作为 softmax 的固定偏移，FlashAttention 内循环里每个 KV tile 都要做的行最大值归约、缩放系数计算和旧累积量重缩放因此可以去掉；块质量给出块之间的相对重要性。
2. **多层级代表块。** 一个块的代表数 $m_b$ 不必固定为 1，可以取 1、2、4、8 直到 64（即原始 token）这些档位，代表越多近似越细，开销也越大。当前实现只用了 1、4、64 三档。
3. **代表块的 packing。** 代表项的计算很零碎，例如 $(B_q\times d)(d\times1)$、$(B_q\times d)(d\times1)$、$(B_q\times d)(d\times4)$。它们可以合成一个 $(B_q\times d)(d\times[1+1+4])$ 的计算。但拼接、搬运和同步也有成本，需要把数据组织成足够宽的 tile，并让搬运和计算重叠。

代码：**[W2GenAI-Lab/EchoSparse-Attn](https://github.com/W2GenAI-Lab/EchoSparse-Attn)**。

[![EchoSparse 单层 attention 讲解动画](assets/echosparse-video-poster.png)](assets/echosparse-v13-1080p.mp4)

## 1. 一个 KV 块可以被算成什么样（baseline PISA介绍）

记全序列长度为 $N$，key 维度为 $d$，value 维度为 $d_v$。query 按 $B_q$ 行切成 tile，KV 按 $B_k$ 个 token 切成块，块 $b$ 用 $m_b$ 个代表行参与计算。本文的实现里 $B_q=128$、$B_k=64$、$d=d_v=128$。对一个 query tile $Q\in\mathbb R^{B_q\times d}$ 和一个块 $K_b\in\mathbb R^{B_k\times d}$、$V_b\in\mathbb R^{B_k\times d_v}$，精确计算是

$$
S_b=\frac{QK_b^\top}{\sqrt d}\in\mathbb R^{B_q\times B_k},\qquad
P_b=\exp\bigl(S_b-c\,\mathbf 1^\top\bigr),\qquad
\mathcal N\mathrel{+}=P_bV_b,\qquad
\mathcal D\mathrel{+}=P_b\mathbf 1 .
$$

其中 $c\in\mathbb R^{B_q}$ 是每行的偏移，$\mathcal N\in\mathbb R^{B_q\times d_v}$ 是未归一化的分子，$\mathcal D\in\mathbb R^{B_q}$ 是分母，所有块累加完后 $O=\mathrm{diag}(\mathcal D)^{-1}\mathcal N$。

$(B_q\times d)(d\times B_k)(B_k\times d_v)$ 只是尺寸上的简写，两次矩阵乘之间还有逐元素指数和整行共享的归一化，不能当成普通的三矩阵连乘。按这个简写比较三种处理：

- **exact**：$(B_q\times d)(d\times B_k)$ 得到 $B_qB_k$ 个 logit，做 $B_qB_k$ 次指数，再乘 $(B_q\times B_k)(B_k\times d_v)$。读 $B_k(d+d_v)$ 个元素，约 $2B_qB_k(d+d_v)$ FLOPs。
- **drop**（SWA、块稀疏中的落选块）：相当于 $B_k\to0$，这个块对 $\mathcal N$ 和 $\mathcal D$ 都没有贡献，分母里也少了这一部分。
- **一个代表**：$(B_q\times d)(d\times1)$ 和 $(B_q\times1)(1\times d_v)$。读 $d+d_v$ 个元素，约 $2B_q(d+d_v)$ FLOPs，是 exact 的 $1/B_k$。

### 零阶近似与 count

取一行 query $q$，块里有 $n_b$ 个有效 token（通常 $n_b=B_k$，尾块可能更少）。令

$$
\bar k=\frac1{n_b}\sum_{j}k_j,\qquad \bar v=\frac1{n_b}\sum_j v_j,\qquad
\bar s=\frac{q^\top\bar k}{\sqrt d},\qquad
\delta_j=\frac{q^\top(k_j-\bar k)}{\sqrt d}.
$$

则 $s_j=\bar s+\delta_j$，且 $\sum_j\delta_j=0$。块的精确贡献是

$$
\sum_j e^{s_j}=e^{\bar s}\sum_j e^{\delta_j},\qquad
\sum_j e^{s_j}v_j=e^{\bar s}\sum_j e^{\delta_j}v_j .
$$

零阶近似取 $e^{\delta_j}\approx1$：

$$
\mathcal D_b^{(0)}=n_b\,e^{\bar s}=e^{\bar s+\ln n_b},\qquad
\mathcal N_b^{(0)}=e^{\bar s}\sum_j v_j=n_b\,e^{\bar s}\,\bar v .
$$

所以一个代表是三元组 $(\bar k,\bar v,n_b)$，在分子和分母里都按 $n_b$ 个 token 计。和精确 token 一起计算时，可以把 count 并进 logit，代表的 logit 为 $\bar s+\ln n_b$。实现用 $\exp_2$，加的是 $\log_2 n_b$，满块为 $\log_2 64=6$。如果漏掉这一项，整块权重会小 64 倍，相当于 logit 少了 $\ln 64\approx4.16$。

### 一阶和二阶

继续展开 $e^{\delta}=1+\delta+\tfrac12\delta^2+\cdots$。一阶时分母的新项 $\sum_j\delta_j=0$，分子的新项不为零：

$$
\sum_j\delta_j v_j=\frac{H^\top q}{\sqrt d},\qquad
H=\sum_j (k_j-\bar k)\,v_j^\top\in\mathbb R^{d\times d_v},
$$

$$
\mathcal N_b^{(1)}=e^{\bar s}\Bigl(\sum_j v_j+\frac{H^\top q}{\sqrt d}\Bigr),\qquad \mathcal D_b^{(1)}=n_b\,e^{\bar s}.
$$

$H$ 是块内 key 偏移与 value 的相关项，均值代表不包含这部分信息。二阶还需要

$$
C=\sum_j (k_j-\bar k)(k_j-\bar k)^\top\in\mathbb R^{d\times d},\qquad
T=\sum_j (k_j-\bar k)\otimes(k_j-\bar k)\otimes v_j\in\mathbb R^{d\times d\times d_v},
$$

$$
\mathcal D_b^{(2)}=e^{\bar s}\Bigl(n_b+\frac{q^\top Cq}{2d}\Bigr),\qquad
\mathcal N_b^{(2)}=\mathcal N_b^{(1)}+\frac{e^{\bar s}}{2d}\,T(q,q,\cdot).
$$

如果为每个块显式缓存这些矩（$B_k=64$、$d=d_v=128$）：

| 每块的摘要 | 缓存内容（元素） | 数值 | 每个 query 的应用代价 |
|---|---|---:|---|
| 原始 $K_b,V_b$（exact） | $B_k(d+d_v)$ | 16,384 | $O(B_k(d+d_v))$ |
| 零阶 $\bar k,\bar v,n_b$ | $d+d_v+1$ | 257 | $O(d+d_v)$ |
| 一阶 $H$ | $dd_v$ | 16,384 | $O(dd_v)$ |
| 二阶 $C,T$ | $d^2+d^2d_v$ | 2,113,536 | $O(d^2d_v)$ |

在这个形状下 $dd_v=128\times128=B_k(d+d_v)=64\times256$，逐块一阶矩和原始块一样大，对一个 query tile 应用它（$(B_q\times d)(d\times d_v)$）的 FLOPs 也和精算这个块相当。二阶的 $T$ 未压缩时有 $d^2d_v=2{,}097{,}152$ 个元素，是原始块的 128 倍；利用两个 key 下标的对称性降到 $d(d+1)d_v/2\approx106$ 万，仍约为原始块的 64.5 倍。这是直接缓存的代价；不缓存而回读 token，或者用低秩、结构化近似，会换成额外的计算或新的近似假设。

PISA 实际使用的是**逐块零阶加全局一阶**。记 $\mathcal U$ 为某个 query tile 的落选块集合。示意地写（满块、不考虑尾块），理想的逐块一阶项 $\sum_{b\in\mathcal U} e^{\bar s_b}H_b^\top q/\sqrt d$ 被换成 $\bigl(\sum_{b\in\mathcal U} e^{\bar s_b}\bigr)\bar H^\top q/\sqrt d$。求和只遍历 $\mathcal U$，而 $\bar H$ 是在整个序列上预先汇总的全局一阶统计量，不随每个 query 的 $\mathcal U$ 重新计算；满块等长时可以理解为各块 $H_b$ 的平均，生产代码对尾块另有按 count 的缩放。预处理时先形成每块的 $H_b$ 再归约，attention 内循环只加载这一个全局矩阵。这也是一种近似。二阶方面，PISA 在相关工作中结合 Based、power attention 讨论过 Taylor 特征维度的膨胀；上表是我们按直接缓存方式算的代价，不代表 PISA 的 kernel 存储了逐块二阶张量。

EchoSparse 不提高展开阶数，而是把块分成更多组，每组仍用零阶近似，组数 $m_b$ 可变（第 3 节）。

## 2. LSE 复用

FlashAttention 类 kernel 按 KV tile 扫描，用 online softmax 维护每行的运行最大值。实现中统一使用 $\exp_2$，本节的 $s_{ij}$ 都已乘过 $\log_2 e$。每个 KV tile，每行 $i$ 要做：

$$
\begin{aligned}
m_i^{\text{new}}&=\max\bigl(m_i,\ \max_j s_{ij}\bigr),\qquad
\alpha_i=2^{\,m_i-m_i^{\text{new}}},\qquad
p_{ij}=2^{\,s_{ij}-m_i^{\text{new}}},\\
\ell_i&\leftarrow\alpha_i\,\ell_i+\sum_j p_{ij},\qquad
O_i\leftarrow\alpha_i\,O_i+\sum_j p_{ij}v_j .
\end{aligned}
$$

如果每行有一个事先给定的固定偏移 $\hat a_i$，循环变成

$$
p_{ij}=2^{\,s_{ij}-\hat a_i},\qquad \ell_i\leftarrow\ell_i+\sum_jp_{ij},\qquad O_i\leftarrow O_i+\sum_jp_{ij}v_j .
$$

EchoSparse 取 $\hat a_i$ 为上一去噪步同层、同 CFG 分支、同一行的 log2 LSE。下面比较[固定偏移路径](https://github.com/W2GenAI-Lab/EchoSparse-Attn/blob/fe8c7452916c74fe9243d65be070b4d9c378e080/echosparse/kernels/sm120.py#L35-L59)与 [online 回退路径](https://github.com/W2GenAI-Lab/EchoSparse-Attn/blob/fe8c7452916c74fe9243d65be070b4d9c378e080/echosparse/kernels/sm120.py#L146-L166)。

取 `BM=128`、`BN=64`、`D=Dv=128`，`scale2=log2(e)/sqrt(D)`；原始 token 的 `log_count` 为 0，代表项为 $\log_2 n$。主线 SM120 将 64-token 块分成两个 32-key 半块，H100 实验路径使用 64 宽 tile。

```python
# Online softmax
m = tl.full((BM,), -float("inf"), tl.float32)
l = tl.zeros((BM,), tl.float32)
acc = tl.zeros((BM, Dv), tl.float32)
for k, v, log_count in kv_tiles:
    s2 = tl.dot(q, tl.trans(k)) * scale2 + log_count[None, :]
    block_max = tl.max(s2, axis=1)
    m_new = tl.maximum(m, block_max)
    alpha = tl.exp2(m - m_new)
    p = tl.exp2(s2 - m_new[:, None])
    l = l * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None]
    acc = tl.dot(p.to(tl.bfloat16), v, acc)
    m = m_new
out = acc / l[:, None]
new_lse2 = m + tl.log2(l)

# Fixed offset
prior2 = tl.load(PREV_LSE2 + rows * H + head)
l = tl.zeros((BM,), tl.float32)
acc = tl.zeros((BM, Dv), tl.float32)
for k, v, log_count in kv_tiles:
    s2 = tl.dot(q, tl.trans(k)) * scale2 + log_count[None, :]
    p = tl.exp2(s2 - prior2[:, None])
    l += tl.sum(p, axis=1)
    acc = tl.dot(p.to(tl.bfloat16), v, acc)
out = acc / l[:, None]
new_lse2 = prior2 + tl.log2(l)
```

每个 KV tile 省掉的是：

- **行最大值归约**：128 行、每行 64 个元素的 max，逻辑上约 $128\times63$ 次比较及 lane 间归约，另有 128 次运行最大值比较。
- **缩放系数**：128 个 $\alpha_i=2^{m_i-m_i^{\text{new}}}$。$128\times64=8192$ 个 $p_{ij}$ 的 $\exp_2$ 仍然要算。
- **旧累积量重缩放**：分母 128 个元素，输出累积器 $128\times128=16{,}384$ 个 FP32 元素。

这些是逻辑工作量，不等于机器指令数；累积器在片上，也不对应显存流量。它们随 KV tile 数重复发生：密度 0.10 时每个 query tile 平均约 118 个 exact 块，仅 exact 循环就有约 190 万次累积器乘法。新增的只是循环前每行读一个 FP32 的历史 LSE，每个 tile 512 字节。QK、逐 score 的 $\exp_2$、行和、PV、用本步分母做的最终归一化、新 LSE 的 $\log_2\ell$ 以及循环后的范围检查都保留。

另一个收益来自依赖关系：

```text
online:  S ──> row max ──> m_new ──> exp2(S - m_new) ──> P·V ──┐
                             └──> alpha ──> scale old O ───────┴──> O
fixed:   S ──────────────────────> exp2(S - prior) ──> P·V ──────> O
```

online 版本中，$p_{ij}$ 要等本 tile 的行最大值归约完成；新的 PV 累加进 $O$ 之前，旧 $O$ 要先乘 $\alpha$。QK、PV 在 Tensor Core 上执行，max、归约和逐行缩放在普通执行单元和 SFU 上执行，这两段夹在矩阵乘之间会拉长关键路径。固定偏移后只剩 $S\to\exp_2\to PV$，更容易与搬运和下一轮 QK 重叠。

B200 上，当前固定偏移的 CuTe DSL 实现把 $O$ 放在 TMEM 中，所有 PV 的 MMA 都直接累加到同一块 `tO`，只在第一次清零；循环中从 TMEM 读出 score，减去固定 LSE，做 $\exp_2$，转为 BF16 写回，再累加分母。循环结束后才把 $O$ 读到寄存器、除以本步行和并写出输出和新 LSE（[源码](https://github.com/W2GenAI-Lab/EchoSparse-Attn/blob/fe8c7452916c74fe9243d65be070b4d9c378e080/echosparse/kernels/sparse_attention.py#L354-L393)），中间没有逐轮的 $O$ 行缩放。

### 固定偏移为什么不改变结果

对同一组项，分子分母乘同一个 $2^{-\hat a_i}$，比值不变：

$$
\frac{O_i}{\ell_i}=\frac{\sum_j 2^{s_{ij}-\hat a_i}v_j}{\sum_j 2^{s_{ij}-\hat a_i}}=\frac{\sum_j 2^{s_{ij}}v_j}{\sum_j 2^{s_{ij}}},\qquad
\hat a_i+\log_2\ell_i=\log_2\sum_j2^{s_{ij}} .
$$

最后除的是本步累加的 $\ell_i$，不是上一步的分母，所以数学上 $\hat a_i$ 取任何值结果都一样，只是浮点求和与舍入会有细微差别。前后两步 LSE 是否接近，影响的是数值范围：记本步的 log2 LSE 为 $\mathrm{lse}^{(2)}_i$，则 $\ell_i=2^{\,\mathrm{lse}^{(2)}_i-\hat a_i}$，偏移越准，$\ell_i$ 越接近 1，$p_{ij}$ 越不容易上溢或下溢。

![相邻去噪步的 LSE 变化](assets/fig5_lse_drift_zh.png)

*图 1　相邻去噪步同一行 LSE 的变化（Wan2.1-1.3B，稠密 FA4，5 个提示词，自然对数单位，B200 历史实验）。点是报告给出的 7 个汇总值。数据：[REPORT.md §3](https://github.com/Owen718/lse-sparse/blob/05e2dab310bf7c710d8b5276aa588e433823db15/docs/REPORT.md#3-相邻去噪步的-lse-变化13b5-个提示词)。*

去噪中段的中位数变化只有 0.05 到 0.1；第 1 步中位数是 0.76，第 49 步是 0.28，对应的 p99 是 2.88 和 1.11，个别（层, 头）上的变化达到上百。因此 kernel 在循环结束后检查 $\log_2\ell_i$，超出约 $\pm60$ 的 tile 会被标记（README 给出的比例约 0.5%），对它选中的 exact 块用 online 递推重算，再加上跳过块的均值补偿；这一步不重算整个 dense attention，也不保留四代表形式。在相同块列表下，仓库测得 kernel 时间约减少 47%。

## 3. 多层级代表块

### 3.1 多层级表示

把块 $b$ 的 token 分成 $m_b$ 组，每组用自己的 K/V 均值和真实 token 数做零阶近似。原始 token 和代表都写成三元组 $(\tilde k_a,\tilde v_a,n_a)$，进入同一个分子和分母：

$$
\tilde o=\frac{\sum_a n_a\,e^{\,q^\top\tilde k_a/\sqrt d}\,\tilde v_a}{\sum_a n_a\,e^{\,q^\top\tilde k_a/\sqrt d}} .
$$

- $m_b=1$：整块一个均值，$n=n_b$；
- $m_b=4$：四个 cluster，各自的 K/V 均值和 count；
- $m_b=64$：每个 token 一组，$n=1$，即 exact。

2、8、16、32 也都是可选的档位，当前代码只实现了 1、4、64。构造四代表的 kernel 把 cluster 补到 16 行，是为了满足 `tl.dot` 的最小尺寸。

$n_a$ 必须同时出现在分子和分母里；空 cluster 和 padding 的 $\ln n_a=-\infty$，权重为零。exact 项和压缩项共用一个分子和一个分母，如果分别 softmax 再相加，每部分都会被归一化成 1，块之间的相对质量就丢了。

$m_b$ 变大是分组变细，不是 Taylor 阶数变高。每组仍是零阶近似，组与组之间的 logit 差异由各自的指数精确表示，被近似掉的只有组内偏移。例如一个 4-token 块，logits $s=(0,0,2,2)$，$v=(1,1,0,0)$，精确输出是 $2/(2+2e^2)\approx0.119$。用一个均值时平均 logit 为 1、$n=4$，输出等于平均 $v=0.5$。分成 $(0,0)$ 和 $(2,2)$ 两组、各 $n=2$，结果和精确值相同，因为组内 logit 相同。一般地，对固定的 query，只要一组内所有 key 的 logit 相同，count 加权的代表就能精确还原这一组；key 完全相同只是一个充分条件。每组一个 token 时就回到 exact 块。

每块读取 $m_b(d+d_v)$ 个元素，GEMM 开销约 $2B_qm_b(d+d_v)$ FLOPs。最终输出误差还取决于组内 logit 方差和 K/V 相关性，不能只由代表数确定。

### 3.2 如何选择 $m_b=1$、$4$、$64$

对 query tile $a$ 和块 $b$，定义

$$
M_{ab}=\sum_{i\in a}\sum_{j\in b}e^{\,s_{ij}-c_i},\qquad c_i=(\ln2)\,\hat a_i ,
$$

这里用自然对数单位：$s_{ij}=q_i^\top k_j/\sqrt d$ 是普通 logit，$c_i$ 是把第 2 节的 log2 LSE $\hat a_i$ 换算成自然对数后的上一步 LSE；kernel 里的 $p_{ij}=2^{\,s^{(2)}_{ij}-\hat a_i}$（$s^{(2)}_{ij}=s_{ij}\log_2 e$）与 $e^{s_{ij}-c_i}$ 是同一个数。$M_{ab}$ 是用历史 LSE 缩放的未归一化块质量。先验接近真实 LSE 时，它接近 tile $a$ 各行在块 $b$ 上的概率之和，但不等于本步精确的归一化概率。

这张质量图不单独做一遍全量打分，而是在本步计算中得到：

- **exact 块**：$e^{s_{ij}-c_i}$ 本来就在概率量级，kernel 计算选中块时按 64-token 块累加 $p_{ij}$，就得到这些块的 $M_{ab}$。
- **落选块**：没有精确的 $s_{ij}$。`estimate_skipped` 在每个 query tile 抽样若干行，用块均值估计落选块的质量，再用同一 tile 已精算的块校准，校准系数限制在 $[1,64]$。

两部分再按 tile 汇总 $\exp(c_i-\mathrm{LSE}^{\text{new}}_i)$ 重新缩放，$\mathrm{LSE}^{\text{new}}_i$ 是本步算出的自然对数 LSE（log2 实现中为 `2^(prior2 - new_lse2)`），得到下一步使用的质量图。

下一步按质量排序分配：共有 $n_{\text{blk}}$ 个块、密度为 $\rho$ 时，质量最大的 $\mathrm{round}(\rho\,n_{\text{blk}})$ 个块和局部邻居用 exact；落选块中质量最高的约三分之一用 4 个代表；其余用 1 个均值。没有块被丢掉。这个分配是按质量排序的启发式，不是从误差公式解出的最优解。

![时间反馈与统一 attention](assets/fig1_feedback_zh.png)

*图 2　LSE 作为固定偏移，块质量决定每块的分辨率；kernel 的输出给出下一步的 LSE 和块质量。*

每个（CFG 分支，层）各自保存状态，CFG 的后续分支可以复用分支 0 的块选择。默认前 9 步用 FA4 稠密计算，第 10 步让所有块都经过固定偏移的 EchoSparse kernel，输出精确并建立第一张质量图，第 11 步起进入稀疏计算。

![每个去噪步的两路 CFG forward 耗时（历史 B200 实验）](assets/fig6_step_latency_zh.png)

*图 3　B200 历史实验：Wan2.1-1.3B 每个去噪步两路 CFG forward 的耗时（三次运行的中位数和范围）。稀疏组是早期的单均值版本，不是当前的四代表或 packed 配置。数据：[curve13.json](https://github.com/Owen718/lse-sparse/blob/05e2dab310bf7c710d8b5276aa588e433823db15/outputs/results/curve13.json)。*

### 3.3 聚类与误差

考虑一组 $C$，有 $n$ 个 token，记号同第 1 节：组均值 $\bar k,\bar v$，$\bar s=q^\top\bar k/\sqrt d$，$\delta_j=q^\top(k_j-\bar k)/\sqrt d$，$\mathbb E_C\delta=0$。精确分母和代表分母分别为

$$
Z_C=n\,e^{\bar s}\,\mathbb E_Ce^{\delta},\qquad \widehat Z_C=n\,e^{\bar s},\qquad
\mathbb E_Ce^{\delta}=1+\tfrac12\mathbb E_C\delta^2+O\bigl(\mathbb E_C|\delta|^3\bigr).
$$

一阶项因使用普通均值而抵消，分母误差的主项是组内 logit 方差。若 $|\delta_j|\le r$，由 $e^\delta-1-\delta=\tfrac12\delta^2e^{\xi}$ 得

$$
0\le Z_C-\widehat Z_C\le\frac{n\,e^{\bar s+r}}{2}\,\mathbb E_C\delta^2 .
$$

左边非负来自 Jensen 不等式，即代表总是低估这一组的分母。分子误差还和 value 有关：

$$
\mathcal N_C-\widehat{\mathcal N}_C=n\,e^{\bar s}\Bigl[\mathrm{Cov}_C(\delta,v)+\tfrac12\,\mathbb E_C\bigl(\delta^2v\bigr)+\cdots\Bigr].
$$

第一项就是组内一阶矩 $H_C^\top q/(n\sqrt d)$。整块的一阶矩可以分成组内和组间两部分：

$$
H_b=\sum_C H_C+\sum_C n_C\,(\bar k_C-\bar k_b)(\bar v_C-\bar v_b)^\top .
$$

分成 $m_b$ 组后，组间部分由各组自己的中心和指数在所有阶上精确表示，只需 $m_b(d+d_v)$ 的存储，不需要 $d\times d_v$ 的矩阵；剩下的误差来自组内的 $H_C$、组内方差和更高阶项，每组一个 token 时全部为零。和第 1 节对比：与其为每块存一个 $dd_v$ 的一阶矩，不如用几列把块分开。

分母误差的主项由组内 logit 方差控制，因此可以用平均平方 logit 偏差作为分组目标。把当前 head 的所有 query 按行组成 $Q\in\mathbb R^{N\times d}$。对一个 $B_k$-token 块 $b$，记它的分组为 $\mathcal C_b$，组内偏移为 $\Delta k_j=k_j-\bar k_C$，目标写成

$$
J_b=\frac1{NB_k}\sum_{i=1}^N\sum_{C\in\mathcal C_b}\sum_{j\in C}
\left(\frac{q_i^\top\Delta k_j}{\sqrt d}\right)^2 .
$$

直接按这个式子计算，每次调整分组都要遍历所有 query。但 $q_i$ 与分组无关，可以先把 query 这一维的求和提出来：

$$
\begin{aligned}
G&=\frac1{Nd}\sum_{i=1}^N q_iq_i^\top=\frac{Q^\top Q}{Nd},\\
J_b&=\frac1{B_k}\sum_{C\in\mathcal C_b}\sum_{j\in C}\Delta k_j^\top G\,\Delta k_j .
\end{aligned}
$$

这样所有 query 对 key 偏差的影响都汇总进了一个 $d\times d$ 矩阵。$Q^\top Q$ 是一个可用 Tensor Core 完成的 Gram 矩阵乘：输入是 $N\times d$，输出只有 $d\times d$；$d=128$ 时就是 $128\times128$。它的计算量为 $O(Nd^2)$，而完整 $QK^\top$ 为 $O(N^2d)$。每次 attention 调用、每个 head 只计算一份 $G$，该 head 的所有 KV 块和后续聚类迭代都复用它，无须为每个 query tile 重算。

接下来需要高效计算 $\Delta k^\top G\Delta k$。直接计算这个二次型，每个点与候选中心的距离比较需要 $O(d^2)$ 的工作。Cholesky 分解把对称正定矩阵写成下三角矩阵与其转置的乘积。$G$ 可能存在 query 未覆盖的零方向，因而先加一个小的对角正则项，使分解更稳定：

$$
G_\epsilon=G+\epsilon I=MM^\top,\qquad
\epsilon=10^{-3}\frac{\operatorname{tr}(G)}d .
$$

于是二次型变成普通的平方长度：

$$
\begin{aligned}
\Delta k^\top G_\epsilon\Delta k
&=\Delta k^\top MM^\top\Delta k=\|M^\top\Delta k\|^2,\\
&=\Delta k^\top G\Delta k+\epsilon\|\Delta k\|^2 .
\end{aligned}
$$

因此只要预先变换 key，令 $x_j=M^\top k_j$，后续就能直接计算欧氏距离。key 在代码里按行存储，所以矩阵形式是 $X=KM$。线性变换保留均值，$\bar x_C=M^\top\bar k_C$，正则化后的目标便是

$$
\begin{aligned}
J_{b,\epsilon}
&=J_b+\frac\epsilon{B_k}\sum_{C\in\mathcal C_b}\sum_{j\in C}\|\Delta k_j\|^2,\\
&=\frac1{B_k}\sum_{C\in\mathcal C_b}\sum_{j\in C}\|x_j-\bar x_C\|^2 .
\end{aligned}
$$

这就是变换后坐标中的 k-means 平方距离目标。Cholesky 的计算量为 $O(d^3)$，只分解这个小矩阵，得到的 $M$ 由所有块共享；$KM$ 也只需在聚类前计算一次。之后每轮迭代只比较一个块的 64 个点与 4 个中心，复杂度为 $O(B_km_bd)$，不再包含 query 数量 $N$。因此，前面由误差分析得到的分组目标，最终可以用一次 query Gram 统计、一次 key 变换和小规模 k-means 来处理。

实现还用 64 个 probe query 给 token 加权，让分组更重视经常被关注的 token。从四个连续的 16-token 组开始，最多进行 6 次加权 Lloyd 迭代，中心按权重更新。分组确定后，写回的 K、V 仍是组内普通均值，$n_a$ 是真实 token 数。加权分组借用了上述度量，是对这个方差目标的启发式调整；最终 attention 输出误差还受 K/V 相关性和统一归一化影响。

## 4. Packing：把 1+1+4 拼成 6 列

实验形状 $N=75600$ 有 $n_{\text{blk}}=1182$ 个 64-token 块（padding 后容量 75648）。密度 0.10 时，exact 块约 118 个，约 $118\times64\approx7.6$ 千列；其余约 1064 个块中约三分之一用四代表，约 355 块、约 1420 列；剩下约 709 块各 1 列。压缩列合计 $1420+709=2129$，与实验记录的平均值一致。每个 tile 约 9.7 千列，不到 dense 的 13%，其中压缩列约占五分之一，但分布最零碎：有的块 1 列，有的块 4 列。

逐块计算时，这些是一串 $(B_q\times d)(d\times1)$、$(B_q\times d)(d\times4)$ 的小乘法，列数远小于一次 Tensor Core MMA 的宽度。以最小的例子说明如何合成 $(B_q\times d)(d\times[1+1+4])$：块 1、块 2 各有一个均值代表，块 3 有四个代表，拼成

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

$c$ 是第 1 节的每行偏移，在 EchoSparse 中是自然对数单位的历史 LSE，$c_i=(\ln2)\,\hat a_i$；下面的等价性对任何固定偏移都成立。$QK_*^\top$ 的 6 列就是三个小矩阵乘各自的列；指数和 count 修正是逐元素的；把 $W$ 按列写成 $[W_1\ W_2\ W_3]$，有

$$
WV_*=W_1\bar v_1^\top+W_2\bar v_2^\top+W_3V_3^{(4)},\qquad W\mathbf 1=W_1\mathbf 1+W_2\mathbf 1+W_3\mathbf 1 ,
$$

也就是三个块的分子和分母贡献加进同一个 $\mathcal N$、$\mathcal D$。$n_*$ 必须随列一起进入分子和分母；exact token 也累加进同一对 $\mathcal N$、$\mathcal D$，全部扫完后只除一次。把三个块各自 softmax 后的输出相加则是另一种计算，每块都被归一化成 1。

因此，在相同的稀疏计划、代表 K/V、multiplicity 和统一归一化下，packing 只是重新分组，数学上输出相同，不引入新的近似；EchoSparse 相对 dense 的近似全部来自上一节的代表压缩。浮点求和顺序改变会带来细微的舍入差别。

6 列远不够一个 Tensor Core tile，实际要跨很多块填满。实验分支每次迭代处理 128 列：均值流一次拼 128 个块的均值，四代表流一次拼 32 个块、每块 4 列，只有每条流的最后一次需要 mask。

已发布的 nonpacked 路径有三个循环：exact 块列表；以 128 列为单位扫描全部块的均值，并 mask 掉 exact 块和四代表块对应的列；四代表块列表，每块一个 id、展开成 4 列，列表补到固定宽度。在这次测试的计划里，每个 tile 要走 1280 个均值列和 1536 个代表列，共 2816 列，平均有用的只有 2129 列，约 76%。被 mask 的列可以不搬数据，但 MMA 按固定形状发射，这部分开销仍然存在。

![packing 的等价重组](assets/fig2_packing_zh.png)

*图 4　示意图：同一组项在 nonpacked 三个循环与 grouped 两条流中的排布，列数不按比例。*

实验分支的 grouped packing 为每个（query tile, head）生成两条紧凑的 descriptor 流：均值流一个块 id 对应一列；四代表流一个块 id 对应 4 列，在片上按 $4\cdot\text{block}+\text{sub}$ 展开地址。更早的逐列 packed 流给每个代表列存一个索引，一个四代表块占 4 个 id，grouped 只存 1 个；nonpacked 的四代表列表本来就是每块一个 id。两类来源分在两个循环里，混合 tile 也不必对两个源张量都做 masked load。

打包的是 descriptor 和索引，代表 K/V 本身没有复制成连续张量，每次迭代仍按索引 gather。

## 5. 搬运与流水

把每个 tile 用到的代表 K/V 复制成连续张量，可以像 exact 块一样整块搬运，但复制本身要读一次、写一次，kernel 还要再读一次；而且每个（query tile, head）的选择都不同。按平均每个 tile 2129 列、每列 $d+d_v=256$ 个 BF16 估算，每个 tile 约 1.09 MB，591 个 query tile 乘 40 个 head，逐 tile 物化一遍要写约 26 GB，而同一计划下 NCU 记录的主 kernel DRAM 读取总量约 9.7 GB。

不复制时，小块搬运也有固定开销：descriptor 和地址计算、对齐、通知数据到达的 barrier，以及 shared memory 缓冲区。负载越零碎，这些开销越难摊销。所以需要先把有效列凑成足够宽的 tile，再让搬运和计算重叠。

设每次迭代装载耗时 $t_\ell$、计算耗时 $t_c$，共 $n$ 次迭代。串行是 $n(t_\ell+t_c)$。假设每次迭代耗时恒定、搬运与计算理想重叠，两级流水的总时间约为

$$
t_\ell+(n-1)\,\max(t_\ell,t_c)+t_c+t_{\text{sync}},
$$

首尾两项是填充和排空，不计 $t_{\text{sync}}$ 时，$n=1$ 退化为串行的 $t_\ell+t_c$；$t_{\text{sync}}$ 表示未被隐藏的等待和 barrier 开销。$n$ 小时填充和排空占比大。按前面的计数，每个 tile 的 exact 循环约 118 次迭代，均值流约 6 次（709/128），四代表流约 12 次（355/32）。在 Wan2.1-14B 的形状下，按 nonpacked 基线的执行列计，exact 循环约占 73%（$7552/(7552+2816)$；按 grouped 的有用列计约 78%），所以先在 exact 循环上做流水。

H100 的 TMA（Tensor Memory Accelerator）用一个 descriptor 描述张量形状，由专用硬件把整块数据异步搬进 shared memory。K、V 的布局是 `[N,H,D]`，一个 64-token 块是按原张量步长描述的矩形 tile（$H>1$ 时在 token 方向不连续），用 block shape 为 `[64,1,D]` 的 descriptor 在行偏移 $64\cdot\text{block}$ 处取出。

![三级软件流水的时间线](assets/fig3_pipeline_zh.png)

*图 5　exact 块循环三级软件流水的示意时间线（非实测），TMA 只用于 exact 块。*

实验分支目前的做法是：exact 块经 TMA 加载，三级软件流水让搬运与 dot 重叠，代表列按 grouped descriptor 做指针 gather。三级流水由 Triton 的软件流水调度生成，三个 stage 是环形缓冲区的三个槽位：第 0 块被消费、槽位释放后，第 3 块才能装入同一槽位；计算等数据到达，装载等槽位空出。只开 TMA 不加流水时几乎没有收益（主 kernel 57.15 ms 对 57.22 ms），两者组合才有效；流水级数从 3 加到 6，时间基本不变。

同一 Wan2.1-14B 计划的 NCU 测量中，加入 TMA 后，指令数从 10.42 G 降到 9.79 G，主 kernel 耗时从 39.06 ms 降到 35.64 ms。barrier 和 wait 仍占约四成 warp 周期，搬运指令减少后，同步仍是主要开销。

代表流可以组织为 producer–consumer：producer warp 按 grouped descriptor gather 后续代表列，在 shared memory 的三个槽位里整理成满宽 tile；consumer warp 对已就绪的槽位做 QK、$\exp_2$ 和 PV，累加进同一对 $\mathcal N$、$\mathcal D$，用完后交还槽位。consumer 等数据到达，producer 等槽位空出。

## 6. 实验结果

### H100 单次 attention（实验分支）

![H100 单次 attention GPU 流程耗时](assets/fig4_h100_bars_zh.png)

<div class="results">

| 方案 | 耗时 | 相对 dense FA4 | 相对 nonpacked |
|---|---:|---:|---:|
| 稠密 FlashAttention-4 | 163.48 ms | 1.00× | — |
| nonpacked EchoSparse（已发布版） | 71.65 ms | 2.28× | 1.00× |
| grouped descriptor + TMA + 三级流水（实验） | 44.51 ms | 3.67× | 1.61× |

</div>

合成 BF16 输入，形状 `[B,N,H,D]=[1,75600,40,128]`，非因果，密度 0.10，20 次计时取中位数；稀疏方案计时包括代表构造和 packing 在内的整个 GPU 流程。后两行共享同一份冻结的稀疏计划，比较的是实现方式，1.61× 来自 packing 和流水；dense 一行计算全部原始 token，3.67× 还包含稀疏本身的收益。这是单次 attention 的结果，不代表整段视频生成有同样的加速。原始记录：[evidence/h100_attention_benchmark.json](evidence/h100_attention_benchmark.json)。

### B200 与 RTX 5090 完整视频生成

![B200 完整视频生成结果](assets/results-b200-dark.png)

*B200 发布结果：50 步完整视频生成，与 Sol-Attn、PISA 对比，PSNR 相对稠密 FA4。*

![最新 RTX 5090 完整生成结果](assets/results-rtx5090-dark.png)

*RTX 5090 发布结果：单 GPU，每个模型一个固定提示词和 seed 的 50 步生成；右侧 PSNR 并列同一提示词的 B200 数值。*

Wan2.1-14B 的整视频加速在 B200 上是 1.70×，在 RTX 5090 上是 1.39×（同一模型单次稀疏步 forward 为 1.61×）。两组评测设置不同：RTX 5090 每个模型只跑一个提示词，Wan2.1-14B、HunyuanVideo 和 MiniMax-H3 还使用了 CPU 权重 offload，所以这两个数不是同一提示词下的硬件对比。

## 7. 总结

视频扩散模型的 attention 更适合近似而不是丢弃：低权重的块数量多，合起来仍影响分子和分母，用 count 加权的代表把它们留在同一个 softmax 里，而不是清零。近似也不必带来零碎计算，1 列、4 列的代表可以拼成完整的 GEMM tile，与 exact token 共用分子和分母，结果不变。相邻去噪步的 LSE 通常接近，上一步的 LSE 可以作为固定偏移，去掉 online softmax 的 max 和 rescale，块质量则决定每个块用多少代表。

完整的 packed producer–consumer 路径还在探索，尚未实现和 release；H100 与 RTX 4090 的支持也尚未完成。
