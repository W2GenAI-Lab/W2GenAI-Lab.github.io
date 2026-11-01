Self-attention in video diffusion models runs over very long sequences. Wan2.1-14B generating a 720p, 81-frame video processes about 75,600 tokens in each attention layer. For a given query, many KV blocks contribute very little.

LLMs have several high-sparsity designs, such as sliding-window attention (SWA) and linear attention like GDN and KDA, but these are mostly built into the architecture at training time. The case we care about is different: a video DiT already trained with dense softmax, whose attention is replaced without retraining. In video, a query's probability mass is often spread over many blocks across space and time; each block is small, but together they are not. Dropping these blocks changes both the numerator and the denominator, and therefore the weighting of the whole row; in training-free replacement this usually shows up in the generated frames. Video models that introduce SWA or linear attention during training are a separate class of methods.

Our work builds on PISA by Haopeng Li et al. ([paper](https://arxiv.org/abs/2602.01077), [code](https://github.com/xie-lab-ml/piecewise-sparse-attention)) and Sol-Attn ([project page](https://nvlabs.github.io/Sana/Sol-Attn/), [paper](https://arxiv.org/abs/2607.24027)). Both approximate rather than drop. PISA replaces keep-or-drop with exact-or-approximate: critical blocks are computed exactly and the rest are approximated with a blockwise Taylor expansion. Sol-Attn adds a zeroth-order compensation for unselected blocks and routes with a single token-to-pooled-block score tile. In both, the compensation enters the numerator and denominator of attention, and the representatives of many blocks are already concatenated into GEMMs.

EchoSparse adds three things on top of this:

1. **Reusing the previous step's statistics.** Between adjacent denoising steps the statistics of each attention row change very little, so the previous step's LSE (log-sum-exp) and per-block mass can be used in the current step. With the LSE as a fixed softmax offset, the row-max reduction, scale-factor computation and rescaling of old accumulators that a FlashAttention inner loop performs for every KV tile can be removed; the block mass gives the relative importance of blocks.
2. **Multi-level representatives.** The number of representatives of a block, $m_b$, need not be 1. It can take the levels 1, 2, 4, 8 up to 64 (the original tokens); more representatives give a finer approximation at a higher cost. The current implementation uses only 1, 4 and 64.
3. **Packing the representatives.** Representative computations are fragmented, for example $(B_q\times d)(d\times1)$, $(B_q\times d)(d\times1)$ and $(B_q\times d)(d\times4)$. They can be combined into one $(B_q\times d)(d\times[1+1+4])$ computation. Concatenation, data movement and synchronization have costs, though, so the data has to be organized into tiles wide enough for efficient GEMMs, with data movement overlapping compute.

Code: **[W2GenAI-Lab/EchoSparse-Attn](https://github.com/W2GenAI-Lab/EchoSparse-Attn)**.

[![Animated explainer of one EchoSparse attention layer](assets/echosparse-video-poster.png)](assets/echosparse-v13-1080p.mp4)

## 1. Approximating a KV block (the PISA baseline)

Let $N$ be the full sequence length, $d$ the key dimension and $d_v$ the value dimension. Queries are split into tiles of $B_q$ rows and keys/values into blocks of $B_k$ tokens; block $b$ takes part with $m_b$ representative rows. In this implementation $B_q=128$, $B_k=64$ and $d=d_v=128$. For a query tile $Q\in\mathbb R^{B_q\times d}$ and a block $K_b\in\mathbb R^{B_k\times d}$, $V_b\in\mathbb R^{B_k\times d_v}$, the exact computation is

$$
S_b=\frac{QK_b^\top}{\sqrt d}\in\mathbb R^{B_q\times B_k},\qquad
P_b=\exp\bigl(S_b-c\,\mathbf 1^\top\bigr),\qquad
\mathcal N\mathrel{+}=P_bV_b,\qquad
\mathcal D\mathrel{+}=P_b\mathbf 1 .
$$

Here $c\in\mathbb R^{B_q}$ is a per-row offset, $\mathcal N\in\mathbb R^{B_q\times d_v}$ the unnormalized numerator and $\mathcal D\in\mathbb R^{B_q}$ the denominator; after all blocks, $O=\mathrm{diag}(\mathcal D)^{-1}\mathcal N$.

$(B_q\times d)(d\times B_k)(B_k\times d_v)$ is only shape shorthand: between the two matrix products there is an elementwise exponential and a normalization shared by the whole row, so it is not an ordinary three-matrix product. In that shorthand, the three options compare as follows:

- **exact**: $(B_q\times d)(d\times B_k)$ gives $B_qB_k$ logits, then $B_qB_k$ exponentials, then $(B_q\times B_k)(B_k\times d_v)$. It reads $B_k(d+d_v)$ elements and costs about $2B_qB_k(d+d_v)$ FLOPs.
- **drop** (SWA, unselected blocks in block sparsity): effectively $B_k\to0$; the block contributes nothing to $\mathcal N$ or $\mathcal D$, and its share of the denominator is missing.
- **one representative**: $(B_q\times d)(d\times1)$ and $(B_q\times1)(1\times d_v)$. It reads $d+d_v$ elements and costs about $2B_q(d+d_v)$ FLOPs, $1/B_k$ of exact.

### Zeroth order and the count

Take one query row $q$ and a block with $n_b$ valid tokens (usually $n_b=B_k$; a tail block may have fewer). Let

$$
\bar k=\frac1{n_b}\sum_{j}k_j,\qquad \bar v=\frac1{n_b}\sum_j v_j,\qquad
\bar s=\frac{q^\top\bar k}{\sqrt d},\qquad
\delta_j=\frac{q^\top(k_j-\bar k)}{\sqrt d}.
$$

Then $s_j=\bar s+\delta_j$ and $\sum_j\delta_j=0$. The exact contribution of the block is

$$
\sum_j e^{s_j}=e^{\bar s}\sum_j e^{\delta_j},\qquad
\sum_j e^{s_j}v_j=e^{\bar s}\sum_j e^{\delta_j}v_j .
$$

The zeroth-order approximation sets $e^{\delta_j}\approx1$:

$$
\mathcal D_b^{(0)}=n_b\,e^{\bar s}=e^{\bar s+\ln n_b},\qquad
\mathcal N_b^{(0)}=e^{\bar s}\sum_j v_j=n_b\,e^{\bar s}\,\bar v .
$$

A representative is therefore the triple $(\bar k,\bar v,n_b)$, counted as $n_b$ tokens in both numerator and denominator. Mixed with exact tokens, the count can be folded into the logit, giving the representative the logit $\bar s+\ln n_b$. The implementation uses $\exp_2$ and adds $\log_2 n_b$, which is $\log_2 64=6$ for a full block. Without this term the block's weight is 64 times too small, equivalent to a logit error of $\ln 64\approx4.16$.

### First and second order

Expanding further, $e^{\delta}=1+\delta+\tfrac12\delta^2+\cdots$. At first order the new denominator term $\sum_j\delta_j$ is zero, while the numerator term is not:

$$
\sum_j\delta_j v_j=\frac{H^\top q}{\sqrt d},\qquad
H=\sum_j (k_j-\bar k)\,v_j^\top\in\mathbb R^{d\times d_v},
$$

$$
\mathcal N_b^{(1)}=e^{\bar s}\Bigl(\sum_j v_j+\frac{H^\top q}{\sqrt d}\Bigr),\qquad \mathcal D_b^{(1)}=n_b\,e^{\bar s}.
$$

$H$ is the correlation between key offsets and values inside the block, which a mean representative does not contain. Second order also needs

$$
C=\sum_j (k_j-\bar k)(k_j-\bar k)^\top\in\mathbb R^{d\times d},\qquad
T=\sum_j (k_j-\bar k)\otimes(k_j-\bar k)\otimes v_j\in\mathbb R^{d\times d\times d_v},
$$

$$
\mathcal D_b^{(2)}=e^{\bar s}\Bigl(n_b+\frac{q^\top Cq}{2d}\Bigr),\qquad
\mathcal N_b^{(2)}=\mathcal N_b^{(1)}+\frac{e^{\bar s}}{2d}\,T(q,q,\cdot).
$$

If these moments are cached explicitly for every block ($B_k=64$, $d=d_v=128$):

| Per-block summary | Cached content (elements) | Value | Per-query application cost |
|---|---|---:|---|
| Raw $K_b,V_b$ (exact) | $B_k(d+d_v)$ | 16,384 | $O(B_k(d+d_v))$ |
| Zeroth order $\bar k,\bar v,n_b$ | $d+d_v+1$ | 257 | $O(d+d_v)$ |
| First order $H$ | $dd_v$ | 16,384 | $O(dd_v)$ |
| Second order $C,T$ | $d^2+d^2d_v$ | 2,113,536 | $O(d^2d_v)$ |

At this shape $dd_v=128\times128=B_k(d+d_v)=64\times256$: a per-block first-order moment is as large as the raw block, and applying it to a query tile, a $(B_q\times d)(d\times d_v)$ product, costs about as many FLOPs as computing the block exactly. Uncompressed, the second-order $T$ has $d^2d_v=2{,}097{,}152$ elements, 128 times the raw block; exploiting the symmetry of its two key indices brings it to $d(d+1)d_v/2\approx1.06$ million, still about 64.5 times the raw block. That is the cost of direct caching; rereading tokens instead, or using low-rank or structured approximations, trades it for extra computation or new approximating assumptions.

What PISA actually uses is **blockwise zeroth order plus global first order**. Let $\mathcal U$ be the set of unselected blocks for a query tile. Schematically (full blocks, ignoring tails), the ideal blockwise first-order term $\sum_{b\in\mathcal U} e^{\bar s_b}H_b^\top q/\sqrt d$ becomes $\bigl(\sum_{b\in\mathcal U} e^{\bar s_b}\bigr)\bar H^\top q/\sqrt d$. The sum runs over $\mathcal U$ only, while $\bar H$ is a global first-order statistic precomputed over the whole sequence and not recomputed for each query's $\mathcal U$; with equal full blocks it can be read as the average of the $H_b$, and the production code applies a count-based scale for tail blocks. Preprocessing forms each $H_b$ and then reduces them, and the attention inner loop loads only that one global matrix. This is also an approximation. On second order, PISA's related work discusses the growth of Taylor feature dimensions in connection with Based and power attention; the table above is our accounting for direct caching and does not mean that PISA's kernel stores per-block second-order tensors.

EchoSparse does not raise the expansion order. It splits a block into more groups, each still approximated at zeroth order, with a variable number of groups $m_b$ (Section 3).

## 2. Reusing the LSE

A FlashAttention-style kernel scans KV tiles and keeps a running row maximum with online softmax. The implementation uses $\exp_2$ throughout, and every $s_{ij}$ in this section has already been multiplied by $\log_2 e$. For each KV tile, each row $i$ does:

$$
\begin{aligned}
m_i^{\text{new}}&=\max\bigl(m_i,\ \max_j s_{ij}\bigr),\qquad
\alpha_i=2^{\,m_i-m_i^{\text{new}}},\qquad
p_{ij}=2^{\,s_{ij}-m_i^{\text{new}}},\\
\ell_i&\leftarrow\alpha_i\,\ell_i+\sum_j p_{ij},\qquad
O_i\leftarrow\alpha_i\,O_i+\sum_j p_{ij}v_j .
\end{aligned}
$$

With a fixed per-row offset $\hat a_i$ given in advance, the loop becomes

$$
p_{ij}=2^{\,s_{ij}-\hat a_i},\qquad \ell_i\leftarrow\ell_i+\sum_jp_{ij},\qquad O_i\leftarrow O_i+\sum_jp_{ij}v_j .
$$

EchoSparse sets $\hat a_i$ to the log2 LSE of the same layer, CFG branch and row from the previous denoising step. The comparison below follows the [fixed-offset path](https://github.com/W2GenAI-Lab/EchoSparse-Attn/blob/fe8c7452916c74fe9243d65be070b4d9c378e080/echosparse/kernels/sm120.py#L35-L59) and [online fallback](https://github.com/W2GenAI-Lab/EchoSparse-Attn/blob/fe8c7452916c74fe9243d65be070b4d9c378e080/echosparse/kernels/sm120.py#L146-L166).

Use `BM=128`, `BN=64`, `D=Dv=128` and `scale2=log2(e)/sqrt(D)`; `log_count` is 0 for original tokens and $\log_2 n$ for representatives. The main SM120 kernel splits a 64-token block into two 32-key halves; the H100 experimental path uses a 64-wide tile.

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

Per KV tile, the fixed offset removes:

- **The row-max reduction**: a max over 64 entries for each of 128 rows, logically about $128\times63$ comparisons plus the cross-lane reduction, and 128 more comparisons for the running maximum.
- **The scale factors**: 128 values $\alpha_i=2^{m_i-m_i^{\text{new}}}$. The $128\times64=8192$ $\exp_2$ calls for $p_{ij}$ are still computed.
- **The rescale of old accumulators**: 128 denominator elements and $128\times128=16{,}384$ FP32 elements of the output accumulator.

These are logical amounts of work, not machine instruction counts, and since the accumulator is on chip they do not correspond to memory traffic. They recur with every KV tile: at density 0.10 a query tile has about 118 exact blocks on average, so the exact loop alone accounts for about 1.9 million accumulator multiplies. The only addition is one FP32 history LSE per row loaded before the loop, 512 bytes per tile. QK, the per-score $\exp_2$, the row sum, PV, the final normalization by the current denominator, the $\log_2\ell$ for the new LSE and the range check after the loop all remain.

The dependencies change as well:

```text
online:  S ──> row max ──> m_new ──> exp2(S - m_new) ──> P·V ──┐
                             └──> alpha ──> scale old O ───────┴──> O
fixed:   S ──────────────────────> exp2(S - prior) ──> P·V ──────> O
```

In the online version $p_{ij}$ waits for the tile's row-max reduction, and the old $O$ must be multiplied by $\alpha$ before the new PV is accumulated into it. QK and PV run on the Tensor Cores, while max, reductions and row scaling run on the ordinary execution units and the SFU; sitting between the matrix products, they lengthen the critical path. With a fixed offset only $S\to\exp_2\to PV$ remains, which overlaps more easily with data movement and the next QK.

On B200, the current fixed-offset CuTe DSL implementation keeps $O$ in TMEM: every PV MMA accumulates directly into the same `tO`, zeroed only on the first iteration. In the loop, scores are read from TMEM, the fixed LSE is subtracted, $\exp_2$ is applied, and P is converted to BF16 and written back before the denominator is accumulated. Only after the loop is $O$ read into registers, divided by the current row sum, and written out with the new LSE ([source](https://github.com/W2GenAI-Lab/EchoSparse-Attn/blob/fe8c7452916c74fe9243d65be070b4d9c378e080/echosparse/kernels/sparse_attention.py#L354-L393)); there is no per-iteration row rescale of $O$.

### Why the fixed offset does not change the result

Multiplying numerator and denominator by the same $2^{-\hat a_i}$ leaves the ratio unchanged:

$$
\frac{O_i}{\ell_i}=\frac{\sum_j 2^{s_{ij}-\hat a_i}v_j}{\sum_j 2^{s_{ij}-\hat a_i}}=\frac{\sum_j 2^{s_{ij}}v_j}{\sum_j 2^{s_{ij}}},\qquad
\hat a_i+\log_2\ell_i=\log_2\sum_j2^{s_{ij}} .
$$

The final division uses the $\ell_i$ accumulated in the current step, not the previous step's denominator, so mathematically any value of $\hat a_i$ gives the same result; floating-point summation and rounding differ slightly. Whether the two steps' LSEs are close affects numerical range: with $\mathrm{lse}^{(2)}_i$ the current step's log2 LSE, $\ell_i=2^{\,\mathrm{lse}^{(2)}_i-\hat a_i}$, so the better the offset, the closer $\ell_i$ is to 1 and the less likely $p_{ij}$ is to overflow or underflow.

![Adjacent-step LSE change](assets/fig5_lse_drift_en.png)

*Figure 1. Change of the same row's LSE between adjacent denoising steps (Wan2.1-1.3B, dense FA4, 5 prompts, natural-log units, historical B200 experiment). The points are the seven summaries given in the report; the line is only a guide. Data: [REPORT.md §3](https://github.com/Owen718/lse-sparse/blob/05e2dab310bf7c710d8b5276aa588e433823db15/docs/REPORT.md#3-相邻去噪步的-lse-变化13b5-个提示词).*

In the middle of denoising the median change is only 0.05–0.1. At step 1 the median is 0.76 and at step 49 it is 0.28, with p99 values of 2.88 and 1.11, and a few (layer, head) rows change by hundreds. The kernel therefore checks $\log_2\ell_i$ after the loop. Tiles outside roughly $\pm60$ are flagged (about 0.5% of tiles according to the README); their selected exact blocks are recomputed with the online recurrence and the skipped-block mean compensation is added back. This does not recompute full dense attention, and it does not keep the four-representative form. On identical block lists, the repository measures about 47% less kernel time.

## 3. Multi-level representatives

### 3.1 Multi-level representation

Split the tokens of block $b$ into $m_b$ groups and approximate each group at zeroth order with its own K/V means and true token count. Original tokens and representatives are all triples $(\tilde k_a,\tilde v_a,n_a)$ in one numerator and one denominator:

$$
\tilde o=\frac{\sum_a n_a\,e^{\,q^\top\tilde k_a/\sqrt d}\,\tilde v_a}{\sum_a n_a\,e^{\,q^\top\tilde k_a/\sqrt d}} .
$$

- $m_b=1$: one mean for the whole block, $n=n_b$;
- $m_b=4$: four clusters, each with its own K/V means and count;
- $m_b=64$: one token per group, $n=1$, which is exact.

2, 8, 16 and 32 are also possible levels; the current code implements only 1, 4 and 64. The kernel that builds the four representatives pads the clusters to 16 rows to meet the minimum size of `tl.dot`.

$n_a$ must appear in both the numerator and the denominator; empty clusters and padding have $\ln n_a=-\infty$ and weight zero. Exact and compressed terms share one numerator and one denominator. Softmaxing them separately and adding would normalize each part to 1 and lose the relative mass between blocks.

A larger $m_b$ means finer groups, not a higher Taylor order. Each group is still a zeroth-order approximation; the logit differences between groups are represented exactly by their own exponentials, and only within-group offsets are approximated. For example, take a 4-token block with logits $s=(0,0,2,2)$ and values $v=(1,1,0,0)$. The exact output is $2/(2+2e^2)\approx0.119$. With one mean, the mean logit is 1 with $n=4$, and the output equals the mean $v$, 0.5. With two groups $(0,0)$ and $(2,2)$, $n=2$ each, the result equals the exact value because the logits within each group are equal. In general, for a fixed query, if all keys in a group have the same logit, the count-weighted representative reproduces that group exactly; identical keys are only a sufficient condition. One token per group is the exact block again.

Each block reads $m_b(d+d_v)$ elements and costs about $2B_qm_b(d+d_v)$ GEMM FLOPs. The final output error also depends on within-group logit variance and K/V correlation, so the representative count alone does not determine it.

### 3.2 Choosing $m_b=1$, $4$ or $64$

For query tile $a$ and block $b$ define

$$
M_{ab}=\sum_{i\in a}\sum_{j\in b}e^{\,s_{ij}-c_i},\qquad c_i=(\ln2)\,\hat a_i ,
$$

in natural-log units: $s_{ij}=q_i^\top k_j/\sqrt d$ is the ordinary logit, and $c_i$ is the previous step's LSE obtained by converting Section 2's log2 LSE $\hat a_i$ to natural log; the kernel's $p_{ij}=2^{\,s^{(2)}_{ij}-\hat a_i}$ (with $s^{(2)}_{ij}=s_{ij}\log_2 e$) is the same number as $e^{s_{ij}-c_i}$. $M_{ab}$ is an unnormalized block mass scaled by the historical LSE. When the prior is close to the true LSE it is close to the total probability that tile $a$'s rows put on block $b$, but it is not the exact normalized probability of the current step.

The mass map is not produced by a separate full scoring pass; it comes out of the current step's computation:

- **Exact blocks**: $e^{s_{ij}-c_i}$ is already on a probability scale, so while computing the selected blocks the kernel sums $p_{ij}$ per 64-token block, which gives $M_{ab}$ for those blocks.
- **Skipped blocks**: they have no exact $s_{ij}$. `estimate_skipped` samples a few rows per query tile, estimates the skipped blocks' mass from block means, and calibrates the estimate on the blocks computed exactly in the same tile, with the calibration factor clamped to $[1,64]$.

The two parts are then rescaled per tile by aggregating $\exp(c_i-\mathrm{LSE}^{\text{new}}_i)$, where $\mathrm{LSE}^{\text{new}}_i$ is the natural-log LSE computed in the current step (`2^(prior2 - new_lse2)` in the log2 implementation), giving the mass map for the next step.

The next step allocates by mass ranking. With $n_{\text{blk}}$ blocks and density $\rho$, the $\mathrm{round}(\rho\,n_{\text{blk}})$ highest-mass blocks and their local neighbors are exact; the most probable third of the remaining blocks get four representatives; the rest get one mean. No block is dropped. This allocation is a ranking heuristic, not an optimum derived from an error formula.

![Temporal feedback and unified attention](assets/fig1_feedback_en.png)

*Figure 2. The LSE serves as the fixed offset and the block mass sets each block's resolution; the kernel's output gives the next step's LSE and block masses.*

Each (CFG branch, layer) keeps its own state, and later CFG branches can reuse branch 0's block selection. By default the first 9 steps run dense FA4; step 10 sends every block through the fixed-offset EchoSparse kernel, which gives an exact output and builds the first mass map; sparse computation starts at step 11.

![Two-branch CFG forward time per denoising step (historical B200 experiment)](assets/fig6_step_latency_en.png)

*Figure 3. Historical B200 experiment: two-branch CFG forward time per denoising step for Wan2.1-1.3B (median and range of three runs). The sparse arm is the early single-mean version, not the current four-representative or packed configuration. Data: [curve13.json](https://github.com/Owen718/lse-sparse/blob/05e2dab310bf7c710d8b5276aa588e433823db15/outputs/results/curve13.json).*

### 3.3 Clustering and error

Consider a group $C$ of $n$ tokens with the notation of Section 1: group means $\bar k,\bar v$, $\bar s=q^\top\bar k/\sqrt d$, $\delta_j=q^\top(k_j-\bar k)/\sqrt d$, and $\mathbb E_C\delta=0$. The exact and representative denominators are

$$
Z_C=n\,e^{\bar s}\,\mathbb E_Ce^{\delta},\qquad \widehat Z_C=n\,e^{\bar s},\qquad
\mathbb E_Ce^{\delta}=1+\tfrac12\mathbb E_C\delta^2+O\bigl(\mathbb E_C|\delta|^3\bigr).
$$

The first-order term cancels because the plain mean is used, so the leading denominator error is the within-group logit variance. If $|\delta_j|\le r$, then $e^\delta-1-\delta=\tfrac12\delta^2e^{\xi}$ gives

$$
0\le Z_C-\widehat Z_C\le\frac{n\,e^{\bar s+r}}{2}\,\mathbb E_C\delta^2 .
$$

The lower bound is Jensen's inequality: a representative always underestimates its group's denominator. The numerator error also depends on the values:

$$
\mathcal N_C-\widehat{\mathcal N}_C=n\,e^{\bar s}\Bigl[\mathrm{Cov}_C(\delta,v)+\tfrac12\,\mathbb E_C\bigl(\delta^2v\bigr)+\cdots\Bigr].
$$

The first term is the within-group first-order moment $H_C^\top q/(n\sqrt d)$. The block's first-order moment splits into within-group and between-group parts:

$$
H_b=\sum_C H_C+\sum_C n_C\,(\bar k_C-\bar k_b)(\bar v_C-\bar v_b)^\top .
$$

With $m_b$ groups, the between-group part is represented exactly, at every order, by each group's own center and exponential, using $m_b(d+d_v)$ storage and no $d\times d_v$ matrix. The remaining error comes from the within-group $H_C$, the within-group variance and higher-order terms, all of which vanish with one token per group. Compared with Section 1: instead of storing a $dd_v$ first-order moment per block, a few columns can split the block.

The leading denominator error depends on within-group logit variance, so the mean squared logit deviation provides a grouping objective. Stack all queries of the current head as rows of $Q\in\mathbb R^{N\times d}$. For a block $b$ of $B_k$ tokens, let $\mathcal C_b$ denote its groups and $\Delta k_j=k_j-\bar k_C$ the within-group offset. The objective is

$$
J_b=\frac1{NB_k}\sum_{i=1}^N\sum_{C\in\mathcal C_b}\sum_{j\in C}
\left(\frac{q_i^\top\Delta k_j}{\sqrt d}\right)^2 .
$$

Evaluating this expression directly would revisit every query whenever the grouping changes. But the queries do not depend on the grouping, so their contribution can be summed in advance:

$$
\begin{aligned}
G&=\frac1{Nd}\sum_{i=1}^N q_iq_i^\top=\frac{Q^\top Q}{Nd},\\
J_b&=\frac1{B_k}\sum_{C\in\mathcal C_b}\sum_{j\in C}\Delta k_j^\top G\,\Delta k_j .
\end{aligned}
$$

The effect of all queries on a key deviation is now contained in a $d\times d$ matrix. Computing $Q^\top Q$ is a Gram matrix multiplication that can use Tensor Cores: the input is $N\times d$, while the output is only $d\times d$, or $128\times128$ when $d=128$. It costs $O(Nd^2)$, compared with $O(N^2d)$ for the full $QK^\top$. Each attention call builds one $G$ per head, shared by all KV blocks and subsequent clustering iterations of that head; it is not recomputed for each query tile.

The next step is to evaluate $\Delta k^\top G\Delta k$ efficiently. Evaluating this quadratic form directly costs $O(d^2)$ for each point-to-center comparison. Cholesky factorization expresses a symmetric positive-definite matrix as a lower-triangular matrix times its transpose. Some directions may not be covered by the queries, leaving $G$ singular, so the implementation first adds a small diagonal regularizer to stabilize the factorization:

$$
G_\epsilon=G+\epsilon I=MM^\top,\qquad
\epsilon=10^{-3}\frac{\operatorname{tr}(G)}d .
$$

The quadratic form then becomes an ordinary squared length:

$$
\begin{aligned}
\Delta k^\top G_\epsilon\Delta k
&=\Delta k^\top MM^\top\Delta k=\|M^\top\Delta k\|^2,\\
&=\Delta k^\top G\Delta k+\epsilon\|\Delta k\|^2 .
\end{aligned}
$$

Transforming each key once, $x_j=M^\top k_j$, therefore makes ordinary Euclidean distances sufficient for subsequent iterations. Keys are stored as rows in the code, giving $X=KM$. A linear transform preserves means, so $\bar x_C=M^\top\bar k_C$, and the regularized objective becomes

$$
\begin{aligned}
J_{b,\epsilon}
&=J_b+\frac\epsilon{B_k}\sum_{C\in\mathcal C_b}\sum_{j\in C}\|\Delta k_j\|^2,\\
&=\frac1{B_k}\sum_{C\in\mathcal C_b}\sum_{j\in C}\|x_j-\bar x_C\|^2 .
\end{aligned}
$$

This is the k-means squared-distance objective in the transformed coordinates. Cholesky costs $O(d^3)$ and operates only on this small matrix; the resulting factor $M$ is shared by all blocks. $KM$ is computed once before clustering. Each subsequent iteration compares only the block's 64 points with 4 centers, costing $O(B_km_bd)$ without a factor of $N$. The grouping objective derived from the error analysis can thus be handled by one query Gram computation, one key transform, and small k-means problems.

The implementation also weights tokens using 64 probe queries, giving more influence to tokens that frequently receive attention. It starts from four contiguous 16-token groups and runs at most 6 weighted Lloyd iterations, updating the centers with those weights. Once the groups are fixed, the K and V written back are still plain within-group means, and $n_a$ is the true token count. Weighted grouping uses the metric above as a heuristic adjustment to the variance objective; the final attention output error also depends on K/V correlation and the shared normalization.

## 4. Packing: turning 1+1+4 into six columns

The experimental shape $N=75600$ has $n_{\text{blk}}=1182$ blocks of 64 tokens (padded capacity 75,648). At density 0.10 there are about 118 exact blocks, about $118\times64\approx7.6$ thousand columns. Of the other ~1064 blocks, about a third use four representatives, roughly 355 blocks and 1420 columns; the remaining ~709 blocks contribute one column each. The compressed columns total $1420+709=2129$, matching the recorded average. Each tile handles about 9.7 thousand columns, under 13% of dense; the compressed columns are about a fifth of these but the most fragmented, with 1 column for some blocks and 4 for others.

Computed block by block, they form a series of small $(B_q\times d)(d\times1)$ and $(B_q\times d)(d\times4)$ products, far narrower than one Tensor Core MMA. The smallest example shows how to form $(B_q\times d)(d\times[1+1+4])$. Blocks 1 and 2 each have one mean representative and block 3 has four. Stack them:

$$
K_*=\begin{bmatrix}\bar k_1^\top\\ \bar k_2^\top\\ K_3^{(4)}\end{bmatrix}\in\mathbb R^{6\times d},\qquad
V_*=\begin{bmatrix}\bar v_1^\top\\ \bar v_2^\top\\ V_3^{(4)}\end{bmatrix}\in\mathbb R^{6\times d_v},\qquad
n_*=(n_1,\,n_2,\,n_{3,1},\,\dots,\,n_{3,4}) .
$$

Then

$$
S=\frac{QK_*^\top}{\sqrt d}\in\mathbb R^{B_q\times6},\qquad
W=\exp\bigl(S-c\,\mathbf 1^\top+\mathbf 1\,(\ln n_*)^\top\bigr),
$$

$$
\mathcal N\mathrel{+}=WV_*,\qquad \mathcal D\mathrel{+}=W\mathbf 1,\qquad O=\mathrm{diag}(\mathcal D)^{-1}\mathcal N .
$$

$c$ is the per-row offset of Section 1, which in EchoSparse is the historical LSE in natural-log units, $c_i=(\ln2)\,\hat a_i$; the equivalence below holds for any fixed offset. The six columns of $QK_*^\top$ are the columns of the three small products; the exponential and the count correction are elementwise; and writing $W$ by columns as $[W_1\ W_2\ W_3]$,

$$
WV_*=W_1\bar v_1^\top+W_2\bar v_2^\top+W_3V_3^{(4)},\qquad W\mathbf 1=W_1\mathbf 1+W_2\mathbf 1+W_3\mathbf 1 ,
$$

so the three blocks' numerator and denominator contributions are added into the same $\mathcal N$ and $\mathcal D$. $n_*$ must travel with the columns into both numerator and denominator, and exact tokens accumulate into the same $\mathcal N$ and $\mathcal D$, with one division after everything is scanned. Adding the outputs of three separately softmaxed blocks is a different computation, in which each block is normalized to 1.

Under the same sparse plan, representative K/V, multiplicities and unified normalization, packing is therefore only a regrouping: the output is mathematically the same and no new approximation is introduced. EchoSparse's approximation relative to dense comes entirely from the representative compression of the previous section. The changed summation order causes slight rounding differences.

Six columns are far from a Tensor Core tile, so tiles have to be filled across many blocks. The experimental branch processes 128 columns per iteration: the mean stream gathers the means of 128 blocks at a time and the four-representative stream 32 blocks of 4 columns each, and only the last iteration of each stream needs a mask.

The released nonpacked path has three loops: the exact block list; a scan over the means of all blocks, 128 columns at a time, masking the columns of exact and four-representative blocks; and the four-representative block list, one id per block expanded to 4 columns and padded to a fixed width. In the plan used here, each tile walks 1280 mean columns and 1536 representative columns, 2816 in total, of which 2129 are useful on average, about 76%. Masked columns need not move data, but MMAs issue at a fixed shape, so their cost remains.

![The packing regrouping](assets/fig2_packing_en.png)

*Figure 4. Schematic: the same terms arranged as three nonpacked loops and as two grouped streams; column counts not to scale.*

The experimental grouped packing builds two compact descriptor streams for each (query tile, head). In the mean stream one block id maps to one column; in the four-representative stream one block id maps to 4 columns, with addresses expanded on chip as $4\cdot\text{block}+\text{sub}$. The earlier per-column packed stream stored one index per representative column, 4 ids per four-representative block, whereas grouped stores 1; the nonpacked four-representative list already used one id per block. Keeping the two sources in separate loops also spares mixed tiles a masked load on both source tensors.

What is packed is descriptors and indices. The representative K/V are not copied into a contiguous tensor, and each iteration still gathers them by index.

## 5. Data movement and pipelining

Copying the representative K/V each tile needs into a contiguous tensor would allow whole-block transfers like those for exact blocks, but the copy costs a read and a write, and the kernel then reads the copy again; moreover, the selection differs for every (query tile, head). At an average of 2129 columns per tile and $d+d_v=256$ BF16 values per column, a tile needs about 1.09 MB, so materializing once per tile for 591 query tiles and 40 heads writes about 26 GB. The main kernel's total DRAM read recorded by NCU on the same plan is about 9.7 GB.

Without a copy, small transfers still have fixed costs: descriptors and address arithmetic, alignment, barriers that signal data arrival, and shared-memory buffers. The more fragmented the work, the harder these are to amortize. Useful columns therefore have to be gathered into wide enough tiles first, and data movement then has to overlap compute.

Let each iteration take $t_\ell$ to load and $t_c$ to compute, over $n$ iterations. Serial execution takes $n(t_\ell+t_c)$. Assuming constant per-iteration times and ideal overlap, a two-stage pipeline takes about

$$
t_\ell+(n-1)\,\max(t_\ell,t_c)+t_c+t_{\text{sync}},
$$

where the first and last terms are fill and drain (excluding $t_{\text{sync}}$, $n=1$ reduces to the serial $t_\ell+t_c$), and $t_{\text{sync}}$ stands for waits and barriers that are not hidden. When $n$ is small, fill and drain take a large share. By the counts above, a tile's exact loop runs about 118 iterations, the mean stream about 6 (709/128) and the four-representative stream about 12 (355/32). At the Wan2.1-14B shape the exact loop accounts for about 73% of the columns executed by the nonpacked baseline ($7552/(7552+2816)$; about 78% of grouped useful columns), so pipelining is applied to the exact loop first.

H100's TMA (Tensor Memory Accelerator) describes a tensor shape with a descriptor and moves a whole tile into shared memory asynchronously in dedicated hardware. K and V are laid out as `[N,H,D]`; a 64-token block is a rectangular tile described by the original tensor strides (not contiguous along the token dimension when $H>1$), fetched by a descriptor with block shape `[64,1,D]` at row offset $64\cdot\text{block}$.

![Timeline of the three-stage software pipeline](assets/fig3_pipeline_en.png)

*Figure 5. Schematic timeline of the three-stage software pipeline in the exact block loop (not measured); TMA applies only to exact blocks.*

The experimental branch currently loads exact blocks through TMA, overlaps loads with dots in a three-stage software pipeline, and gathers representative columns by pointer through the grouped descriptors. The three stages are generated by Triton's software-pipelining scheduler and are three slots of a ring buffer: block 3 can enter a slot only after block 0 has been consumed and released it; compute waits for data to arrive, and loading waits for a free slot. TMA without pipelining barely helps (main kernel 57.15 ms versus 57.22 ms); only the combination is effective, and going from 3 to 6 stages leaves the time essentially unchanged.

In NCU measurements of the same Wan2.1-14B plan, adding TMA reduces the instruction count from 10.42 G to 9.79 G and the main-kernel time from 39.06 ms to 35.64 ms. Barrier and wait still account for about 40% of warp cycles: synchronization remains a major cost after reducing the load instructions.

The representative stream can use a producer–consumer design: producer warps gather upcoming columns through the grouped descriptors and arrange full-width tiles in three shared-memory slots; consumer warps run QK, $\exp_2$ and PV on ready slots, accumulating into the same $\mathcal N$ and $\mathcal D$, then return each slot. The consumer waits for data; the producer waits for a free slot.

## 6. Results

### A single attention call on H100 (experimental branch)

![Complete GPU time of one attention call on H100](assets/fig4_h100_bars_en.png)

<div class="results">

| Configuration | Time | vs dense FA4 | vs nonpacked |
|---|---:|---:|---:|
| Dense FlashAttention-4 | 163.48 ms | 1.00× | — |
| Nonpacked EchoSparse (released) | 71.65 ms | 2.28× | 1.00× |
| Grouped descriptors + TMA + 3 stages (experimental) | 44.51 ms | 3.67× | 1.61× |

</div>

Synthetic BF16 inputs of shape `[B,N,H,D]=[1,75600,40,128]`, non-causal, density 0.10, median of 20 timings; the sparse timings cover the whole GPU flow, including representative construction and packing. The last two rows share one frozen sparse plan and compare implementations, so the 1.61× comes from packing and pipelining; the dense row computes all original tokens, so the 3.67× also includes the gain of sparsity itself. This is a single attention call and does not imply the same speedup for whole-video generation. Raw record: [evidence/h100_attention_benchmark.json](evidence/h100_attention_benchmark.json).

### Whole-video generation on B200 and RTX 5090

![B200 whole-video generation results](assets/results-b200-dark.png)

*B200 released results: 50-step whole-video generation compared with Sol-Attn and PISA, with PSNR against dense FA4.*

![Latest RTX 5090 full-generation results](assets/results-rtx5090-dark.png)

*RTX 5090 released results: single GPU, one fixed prompt and seed per model, 50-step generation; the PSNR panel lists the B200 value for the same prompt alongside.*

For Wan2.1-14B the whole-video speedup is 1.70× on B200 and 1.39× on RTX 5090 (1.61× for a single sparse-step forward of the same model). The two evaluations are set up differently: on RTX 5090 each model runs a single prompt, and Wan2.1-14B, HunyuanVideo and MiniMax-H3 use CPU weight offload, so the two numbers are not a same-prompt hardware comparison.

## 7. Summary

Attention in video diffusion models is better approximated than dropped: the low-weight blocks are numerous and together still affect the numerator and denominator, and count-weighted representatives keep them in the same softmax instead of zeroing them. Approximation also need not mean fragmented computation: 1- and 4-column representatives can be packed into full GEMM tiles that share the numerator and denominator with exact tokens, without changing the result. LSE values are usually close between adjacent denoising steps, so the previous step's LSE can serve as a fixed offset that removes the max and rescale of online softmax, and the block mass decides how many representatives each block gets.

The full packed producer–consumer path is still under exploration and has not been implemented or released. H100 and RTX 4090 support is also pending.
