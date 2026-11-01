Self-attention in a video diffusion transformer runs over a huge sequence. Wan2.1-14B at 720p and 81 frames has roughly 75,000 tokens per layer, dozens of layers per forward pass, and 50 denoising steps per video. Plenty of KV blocks contribute almost nothing to a given query, so skipping them looks like an easy win. Then you try it and hit three questions. How do you know what to skip without scanning every block first? What exactly is lost when a block is skipped, and can it be put back? And once you have a sparse structure, does the GPU actually run it fast?

EchoSparse answers all three with one design. **Between adjacent denoising steps the statistics of each attention row change little, so the previous step's LSE (log-sum-exp) and its per-block mass can both be reused in the current step. The LSE supplies the softmax scaling; the block mass supplies the relative importance of blocks.** This post walks through the math in the order it is derived, then looks at how it lands on data layout and GPU pipelining, and ends with one set of measurements from an H100 experimental branch.

The code lives at **[W2GenAI-Lab/EchoSparse-Attn](https://github.com/W2GenAI-Lab/EchoSparse-Attn)**. Below is the animation from the repository; the post follows its order. The web version of this article also embeds the [original Canvas animation](assets/echosparse-canvas-animation.html) with a replay button.

![Animated explainer of one EchoSparse attention layer](assets/echosparse-explainer.gif)

A word on scope. The main branch README currently supports B200 (SM100) and RTX 5090 (SM120); proper H100 (SM90) and RTX 4090 support is still on the TODO list. Everything here about packed representatives, TMA and the three-stage pipeline comes from an **experimental H100 branch** that has not been merged, and I will say so as I go.

## 1. One row of attention is a weighted average

Fix a query $q\in\mathbb R^d$ and key/value pairs $\{(k_j,v_j)\}_{j=1}^{N}$, and let $s_j=q^\top k_j/\sqrt d$. The output of this row is

$$
o=\frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}} ,
$$

a weighted average of the $v_j$ with normalized weights. Every attention implementation leans on one small fact: for any constant $c$ independent of $j$,

$$
o=\frac{\sum_j e^{s_j-c}v_j}{\sum_j e^{s_j-c}} ,
$$

since the factor $e^{-c}$ cancels. FlashAttention takes $c$ to be a running row maximum. Each time the maximum grows, everything accumulated so far has to be rescaled, and there is a row-wise max reduction on top of that.

## 2. Last step's LSE as a fixed offset

Let $a=\log\sum_j e^{s_j}$ be the row's LSE. With $c=a$, $e^{s_j-a}$ is exactly the attention probability. We obviously do not know the current $a$ before computing the row. But hold on: adjacent steps see similar inputs, so how far can $a$ move?

![Adjacent-step LSE change](assets/fig5_lse_drift_en.png)

*Figure 1. Wan2.1-1.3B, dense FA4, 5 prompts. $|\mathrm{LSE}(t)-\mathrm{LSE}(t-1)|$ for the same layer, CFG branch and row, in natural-log units. Seven step summaries reported in the source are shown; the line is a visual guide, not a fit or interpolation. Data: [docs/REPORT.md §3](https://github.com/Owen718/lse-sparse/blob/05e2dab310bf7c710d8b5276aa588e433823db15/docs/REPORT.md#3-相邻去噪步的-lse-变化13b5-个提示词) (historical B200 development experiment, repository Owen718/lse-sparse, commit 05e2dab).*

Two things to read honestly from this plot. In the middle of denoising the median change is only 0.05–0.1 nats, which is quiet. But the start (median 0.76 at step 1) and the last steps (0.28 at step 49) are clearly larger, and the p99 reaches 2.88 and 1.11. So "the LSE barely changes" is not true for every step. The report also records a few extreme (layer, head) rows with |δ| in the hundreds, cause not found.

Can the previous LSE still serve as an offset? Yes, and the reason matters: **the offset is an identity, and correctness does not depend on the two LSEs being close.** For any given set of terms, subtracting any fixed $c$ leaves the output unchanged. So taking the previous step's LSE for the same layer, CFG branch and row as the fixed offset $a_i$ introduces no error by itself. Closeness buys something else: scaled row sums stay near 1, away from overflow and underflow, so the kernel can drop the max reduction and the rescaling. In the repository this cuts kernel time by about 47% on identical block lists. Tiles whose row sums wander too far (about 0.5% according to the README) are flagged: their selected exact blocks are rerun with online softmax, and the skipped-block mean compensation is added back on top. That is not a recomputation of full dense attention, and it does not automatically keep the four-representative form. The implementation works in $\log_2$/$\exp_2$, which changes nothing mathematically.

The identity covers only a softmax over a *fixed set of terms*. The sparse selection and the representative compression that follow are a different matter: they are approximations.

## 3. Which resolution does each block get?

Split the sequence into 64-token KV blocks and the queries into 128-row tiles. For a query tile $a$ and a KV block $b$, the block participates at one of three resolutions. The animation calls the number of rows used $m_b$:

- $m_b=64$, exact: all original tokens of the block;
- $m_b=4$, refined: four clustered representatives for the whole block;
- $m_b=1$, mean: the block's mean key and mean value.

Who decides? A natural quantity is

$$
M_{ab}=\sum_{i\in a}\sum_{j\in b}e^{\,s_{ij}-a_i},
$$

with $a_i$ the previous step's LSE. This is a **historical-LSE-scaled, unnormalized block mass**. When the prior is close to the true LSE it is close to the probability that tile $a$'s rows put on block $b$, but it is not automatically the exact normalized probability of the current step. The code rescales and calibrates before selection (the masses are rescaled by $\exp(a_i-\mathrm{LSE}^{\text{new}}_i)$ summarized per tile, or $2^{a_i-\mathrm{LSE}^{\text{new}}_i}$ in the log2 implementation, since LSEs are logarithms and are never divided directly; the skipped-block estimates are then calibrated against the exactly computed blocks), so do not read $M_{ab}$ as an exact probability.

With $B$ KV blocks and density $\rho$, the $k=\mathrm{round}(\rho B)$ blocks of highest mass (plus local neighbours) are exact; the next batch under a refined budget $R$ gets four representatives (the README describes it as roughly the most probable third of the skipped blocks); the rest get one. The key point: **no block is dropped.** A skipped block merely lowers its resolution and still takes part in the single softmax below.

## 4. One softmax over $(\tilde k_a,\tilde v_a,n_a)$

All three resolutions fit one shape of term. An exact term is an original token with $n_a=1$. A mean term has the block's mean K and V, and $n_a$ is the number of valid tokens in the block (the last block may hold fewer than 64). A refined term has a cluster's mean K and V, and $n_a$ is the cluster's true token count. The output is

$$
\tilde o=\frac{\sum_a n_a\,e^{\,q^\top\tilde k_a/\sqrt d}\,\tilde v_a}{\sum_a n_a\,e^{\,q^\top\tilde k_a/\sqrt d}} .
$$

Two details are easy to get wrong. First, $n_a$ must appear in both numerator and denominator: a representative speaking for $n_a$ tokens carries $n_a$ shares of weight in the denominator and the same shares in the numerator. Writing $n_a e^{x}=e^{x+\ln n_a}$ shows it is a $\ln n_a$ correction on the logit, and the implementation adds $\log_2 n_a$. An empty cluster has $\ln n_a=-\infty$, so zero weight, and padding works the same way. Second, exact and compressed terms share one numerator and one denominator. Running a separate softmax per group and adding the results would normalize each group to 1 and erase their relative masses.

A small example. A block has four tokens with logits $s=(0,0,2,2)$ and values $v=(1,1,0,0)$. Exactly: numerator $2$, denominator $2+2e^2\approx16.78$, output about $0.119$. Represent the whole block by its mean: mean logit $1$, $n=4$, denominator $4e\approx10.87$, mean value $0.5$, output $0.5$. Hmm, that is not a small miss. Now cluster the first two tokens and the last two separately: logits $0$ and $2$, $n=2$ each, and the result equals the exact value.

That is where the approximation lives. For generally different $k_j$, $\sum_j e^{s_j}\ne n\,e^{\bar s}$ (Jensen), and the mean value is not the attention-weighted value. Relative to dense attention, representative compression is therefore an **approximation**, not an identity. When is it exact? For a fixed query, it suffices that all keys in a cluster produce the same logit; then the count-weighted representative reproduces the cluster's contribution exactly. Identical keys are one sufficient case, not the only one. Clustering is about making keys in one cluster look as alike as possible *to the queries*.

## 5. Clustering in the query metric

How do we measure "as alike as possible"? My first instinct is plain Euclidean k-means on K... but attention only sees a key through $q^\top k$, and two keys far apart in K-space may be indistinguishable to the queries.

Let $\delta=k-k'$ be a column vector (fold $1/\sqrt d$ into $q$). For query $q_i$ the logit difference is $q_i^\top\delta$, and the mean squared difference over queries is

$$
\frac1L\sum_i\bigl(q_i^\top\delta\bigr)^2=\delta^\top G\,\delta,\qquad G=\mathbb E[qq^\top]=MM^\top ,
$$

with $M$ the Cholesky factor of $G$, so this equals $\|M^\top\delta\|^2$. If keys are stored as rows of a matrix $K$, the transformed coordinates are $X=KM$, and ordinary Euclidean clustering in $X$ minimizes the logit spread each cluster presents to the queries.

One refinement. The implementation does not factor $G$ itself but $G_\epsilon=G+\epsilon I$ ($\epsilon$ is $10^{-3}$ times the mean diagonal, for a stable Cholesky). The transformed squared distance is then

$$
\|M_\epsilon^\top\delta\|^2=\delta^\top G\,\delta+\epsilon\,\|\delta\|^2 ,
$$

the mean squared logit difference **plus** a small Euclidean term. The unregularized equality is exact; the regularized one is what the code does, with a light dose of Euclidean distance mixed into the query metric.

Each 64-token block is split into four clusters. Sixty-four probe queries provide per-token weights, and a weighted Lloyd iteration (started from four contiguous 16-token groups, at most six iterations by default) does the assignment. The weights only decide **who joins which cluster**. The written-back representatives are plain cluster means of K and V, and $n_a$ is the real token count, which is what the unified formula of section 4 needs.

## 6. Time feedback: measure while computing

Section 3 needs the mass map. If every step scored all blocks in a separate exact pass, we would pay back the savings. EchoSparse lets the exact blocks that are computed anyway produce the mass: the fixed offset puts $e^{s_{ij}-a_i}$ on a probability-like scale, and the kernel accumulates it per 64-token block while processing the selected ones, giving $M_{ab}$ for those blocks.

Skipped blocks have no exact $s_{ij}$, so they need an estimate. EchoSparse does not run an additional full exact QK scoring sweep; it runs a lightweight `estimate_skipped` pass. For each query tile it samples a few rows, estimates the skipped blocks' masses from their block means, and calibrates against the exactly computed blocks (the calibration factor is clamped to $[1,64]$), which also yields the scales for the next step.

![Temporal feedback and unified attention](assets/fig1_feedback_en.png)

*Figure 2. One set of statistics, used three ways: the LSE is the fixed offset, the block-mass ranking sets each block's resolution, and skipped blocks join the same softmax as $(\tilde k,\tilde v,n)$. The fused kernel's output then produces the next step's LSE and block masses.*

The loop needs a start. Each (CFG branch, layer) keeps its own state, and later CFG branches may reuse the block selection of branch 0. By default the first **9 steps** run dense FA4, and the **10th step** sends every block through the fixed-offset EchoSparse kernel, producing an exact output and the first mass map; sparse steps follow.

![Per-step two-CFG forward time (historical B200 experiment)](assets/fig6_step_latency_en.png)

*Figure 3. Historical development experiment (Wan2.1-1.3B, B200, 480×832, 81 frames, 50 steps, CFG 6). The vertical axis is the sum of the two CFG transformer forwards, not attention-only time and not a full video. Median of 3 rounds; the shaded band is the observed min–max (n=3), not a confidence interval. The sparse arm is the old single-mean LGSF (density 0.10, refine=0, separate block selection per branch), not the current four-representative or packed configuration. Step 10 is the all-block initialization and the sparse phase starts at step 11; at step 11 one of the three rounds shows 513 ms, hence the wide band. Data: [outputs/results/curve13.json](https://github.com/Owen718/lse-sparse/blob/05e2dab310bf7c710d8b5276aa588e433823db15/outputs/results/curve13.json).*

## 7. A quick count: how many columns does 10% density leave?

Before layout, count what a query tile handles. In the experimental shape, $L=75600$ gives $B=1182$ blocks (padded capacity 75,648). At density 0.10 about 118 blocks are exact, roughly $118\times64\approx7.6$k columns. Of the remaining $\approx1064$ blocks about a third go refined: about 355 blocks of 4 columns, $\approx1420$ columns. The other $\approx709$ blocks give one column each. That is $1420+709=2129$ compressed columns, matching the 2129 average in the experiment record.

A tile thus faces about $9.7$k columns, under 13% of dense, with compressed columns about a fifth of those. They are few but fragmented: a mean block is 1 column, a refined block is 4, neither contiguous nor close to the 64 columns a tensor-core loop likes to eat at once. The column count drops below 13%, yet the measured speedups in section 10 are far smaller than that ratio, so layout deserves attention of its own.

## 8. Packing: the same computation, laid out tighter

The algorithm is complete; the question left is how efficiently a GPU runs it. First, what nonpacked (the released version) actually does. It is not one mixed list of exact, mean and refined entries. The source has three loops: one over the exact block list; one that sweeps the means of **all** blocks 128 columns at a time, with the columns of exact and refined blocks masked out; and one over the refined list, with one id per block and four representatives each. So the mean loop carries many wasted masked columns, and the refined list is padded to a fixed width. In this benchmark's plan the baseline layout walks 1280 mean columns plus 1536 representative columns per tile, 2816 in total, while on average only 2129 carry real entries, about 76% utilization.

![Packing as an equivalent reorganization](assets/fig2_packing_en.png)

*Figure 4. Schematic; column counts are illustrative. Both sides hold the same items: 5 means, 8 refined representatives, 10 exact blocks. Nonpacked uses three loops with many masked columns in the mean loop; grouped packing arranges them into two compact streams.*

Packing reorganizes exactly these columns. Write the computed terms as a set $\mathcal A$ with weights $w_a=n_a e^{s_a-c}$; the output is $\sum_{a\in\mathcal A}w_a\tilde v_a/\sum_{a\in\mathcal A}w_a$. Permute $\mathcal A$, regroup it, or delete zero-weight padding terms ($n_a=0$), and numerator and denominator are unchanged as real sums, because addition is commutative and associative and the deleted terms contributed zero. So **given the same sparse plan, representative K/V, multiplicities and common normalization, packing is an equivalent reorganization: it introduces no additional approximation, and its output is identical in real arithmetic.** EchoSparse's approximation relative to dense comes from the representative compression of section 4, and packing neither adds to nor removes any of it. Rounding can differ slightly because the reduction order changes; that is an implementation detail.

The grouped packing of the experimental branch splits descriptors into two compact streams. In the mean stream one block id is one column; in the refined stream one block id is four columns, expanded on chip as $4\cdot\text{block}+\text{sub}$. The $4n_g\to n_g$ reduction compares grouped packing with the earlier per-column packed stream, which stored one index per representative column and so needed four ids per refined block; nonpacked already keeps one id per refined block and is not part of that comparison. Keeping the two sources in separate loops also avoids masked loads on both tensors in a mixed tile. What is packed is the **descriptors and indices**; the representative K/V data is *not* copied into one big contiguous tensor.

## 9. Moving and computing on the GPU: tiles, TMA and a three-stage pipeline

Picture the main loop as data flowing through three levels: HBM, shared memory, registers. The 128-row Q tile is loaded **before** the KV loop and reused throughout; it is not re-read from HBM each round. Each round brings in the K and V of one KV block (or a group of representative columns), the tensor cores compute $S=QK^\top$, and $S$ with $V$ updates the accumulator. If moving and computing strictly alternate, the tensor cores spend much of their time waiting.

The cure is pipelining: while the current tile is consumed, transfers for later tiles are already in flight. Conceptually a producer fetches and a consumer computes, coordinated by a ring buffer with dependencies and waits. H100's TMA (Tensor Memory Accelerator) lets a descriptor describe a tensor's shape and has dedicated hardware copy blocks into shared memory asynchronously. Exact blocks fit it well: K and V are laid out as `[L,H,D]`, a 64-token block is a contiguous strided tile of that tensor, and a descriptor with block shape `[64,1,D]` fetches it at row offset $64\cdot\text{block}$ in one request.

This connects to packing. A masked global load can indeed save some traffic, yet the MMA issues at a fixed tile shape, so a tile with few valid columns still pays much of the same issue and compute cost, and the more wasted columns, the more waste. Fewer descriptor columns and fewer masked loads raise the useful fraction of each round and give the pipeline something worth overlapping. The two optimizations can be combined, and the benefit of the combination is whatever the measurement shows; I do not claim any arithmetic law for it.

![Load/compute timeline of a three-stage software pipeline](assets/fig3_pipeline_en.png)

*Figure 5. Pure schematic, not measured. Tile 3 can enter slot 0 only after tile 0 has been consumed and the slot released. The consumer waits for "data arrived", the loader for "slot free"; these are the dependencies mentioned in the text.*

The experimental branch implements this combination: **exact blocks are loaded through TMA, and a three-stage software pipeline overlaps the loads with the dot products**, while grouped representative columns still use pointer gathers and do not go through TMA. The pipeline comes from Triton's software-pipelining scheduler. The three stages are three slots of a ring buffer, not three groups of warps with different roles, and not three kernels. In the experiment TMA alone gave almost no gain; only together with the multi-stage pipeline was it fast. One more distinction: main's SM120 kernel walks a 64-token block as two 32-key halves, whereas the H100 candidate uses the 64-wide path. They should not be conflated.

A genuinely independent producer/consumer architecture, with dedicated warps that only load and others that only issue MMAs, is **not implemented**. On the Triton 3.6 SM90 path we tried the warp-specialization switch; registers, spills, shared memory and time were identical to the ordinary software pipeline, so it had no effect. A hand-written WMMA plus cp.async prototype was slower. Reaching it takes handwritten wgmma or CuTe kernels, which is future work.

## 10. Measurements on H100 and RTX 5090

### H100 single-attention fixture

![H100 single-attention GPU pipeline time](assets/fig4_h100_bars_en.png)

| Configuration | Time | vs dense FA4 | vs nonpacked |
|---|---:|---:|---:|
| Dense FlashAttention-4 | 163.48 ms | 1.00× | — |
| Released nonpacked EchoSparse | 71.65 ms | 2.28× | 1.00× |
| Grouped descriptors + TMA + 3 stages | 44.51 ms | 3.67× | 1.61× |

Measurement scope: synthetic BF16 Q/K/V of shape `[B,L,H,D]=[1,75600,40,128]`, noncausal, density 0.10, a frozen sparse plan shared by all configurations, median of 20 CUDA-event timings; the baseline is FlashAttention-4 through `flash_attn.cute.interface.flash_attn_func`. "Complete" for the sparse rows means the whole GPU sparse pipeline, including representative construction and packing, and excluding host-side descriptor construction, scratch allocation and compilation. This is one attention call, not a transformer layer and certainly not end-to-end video generation. Dense and sparse also compute different terms (on average about 118 exact blocks plus 2129 compressed columns for the sparse ones), so the table shows what the same sparse plan costs under different implementations; it does not imply a video-generation speedup of the same size. The raw record is [evidence/h100_attention_benchmark.json](evidence/h100_attention_benchmark.json), a sanitized copy of the experimental-branch result.

The 1.61× is the packing-and-pipelining system work; the 3.67× also contains the gain of sparsity itself over dense. For reference only: one full Wan2.1-14B generation on the packing path went from 818.60 s to 764.57 s (1.07×), against nonpacked, one run per arm, with no full dense baseline.

### Current RTX 5090 full-generation results

The released RTX 5090 measurements give a different view of the system: complete video generation, including the time outside attention. The latest chart adds a **1.39× whole-video speedup for Wan2.1-14B**, alongside its 1.61× sparse-step-forward speedup.

![Latest RTX 5090 full-generation results](assets/results-rtx5090-dark.png)

*Latest released RTX 5090 results. Each model uses prompt 0 and the same seed as the B200 evaluation, with one 50-step generation per method on one GPU. Wan2.1-14B, HunyuanVideo and MiniMax-H3 use grouped CPU weight offload; Wan2.1-1.3B stays on the GPU. The left panel separates whole-video and single sparse-step-forward speedup. The right panel reports PSNR against FA4, alongside the B200 result for the same prompt. This cohort is separate from the synthetic H100 single-attention fixture above.*

## 11. Scope and what comes next

The B200 and RTX 5090 results, including end-to-end generation, are in the main README. The H100 path lives only on the experimental branch; it needs grouped representatives, TMA and the stage count switched on explicitly, and still has to be integrated into main with multi-model benchmarks. The latest main has no formal RTX 4090 (SM89) backend yet, which needs its own adaptation and acceptance, and the H100 TMA path does not carry over directly to Ada. The historical RTX 4090 data in the animation belongs to a different cohort of experiments and is not the same scope as the H100 fixture here. An independent producer/consumer schedule is also still open.

Looking back, one idea runs through the design. Last step's statistics spare softmax its max and rescale, rank the blocks for selection, and give compression a yardstick aligned with probabilities. The unified softmax puts exact, refined and mean terms in one numerator and denominator, so packing only has to decide how the same terms are arranged. That mathematical equivalence is what lets the infrastructure rearrange data and pipelines boldly.
