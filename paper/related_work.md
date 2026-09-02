> **VERIFICATION LOG**
> 2026-09-01: arXiv:2602.11506 (RooflineBench) checked against the arXiv
> abstract page. Title, authors and abstract CONFIRMED. Roofline framing,
> on-device/edge setting, and absence of a fitted coefficient / held-out
> validation all CONFIRMED. Two claims in this document are NOT supported
> by the abstract and are downgraded to UNVERIFIED: (a) that it uses
> llama.cpp/llama-bench, (b) that it explicitly defers MoE. Check the body
> text before citing either.
> 2026-09-01: arXiv:2507.14397 (LIMINAL) CONFIRMED — title, authors, 7.6% MAE
> all correct. IMPORTANT: it targets DATACENTER accelerators (HBM3/HBM4,
> GPUs/TPUs), not consumer hardware, so quoting it as 'the bar' for a
> consumer-hardware model is not like-for-like. NeuSight's 8.7% remains
> UNVERIFIED.
> 2026-09-01: arXiv:2412.07067 (MoE-CAP) CONFIRMED. THREE CORRECTIONS: (a) the
> metric is 'Sparse MEMORY Bandwidth Utilization', not 'Sparse Model Bandwidth
> Utilization'; (b) it is an arXiv preprint (v6, Nov 2025) with no conference
> or journal venue -- do not cite a venue; (c) it ALSO defines S-MFU, the
> compute-side analogue, which is prior art for our prefill coefficient and was
> missed by the survey. Activated-vs-total mechanics are implied by the
> abstract but not stated -- check the body before asserting them.
> 2026-09-01: arXiv:2508.08531 CONFIRMED via search. Benazir & Lin,
> 'Benchmarking and Characterization of LLM Inference on Apple Silicon',
> SIGMETRICS 2026 / POMACS doi:10.1145/3771563. MUCH closer than the survey
> said: same setting (personal workstations, single-request, datacenter GPUs
> excluded), FIVE testbeds (M2 Ultra/M2 Max/M4 Pro/A6000/2xA6000), 8B-405B,
> 14 quant schemes. Pre-empts our quantisation findings AND the unified-memory
> argument. Does NOT fit a predictor or hold out models -- that is what we keep.
> ALSO FOUND, both unverified and both worth checking before submission:
>   BaseRT, arXiv:2607.00501 -- Apple Silicon native Metal runtime, cites B&L,
>     folds dequantisation into the kernel inner loop.
>   'Native LLM and MLLM Inference at Scale on Apple Silicon', arXiv:2601.19139.
> Remaining flagged citations are still unverified.

# 2. Related work

*(Draft for §2 of paper.md. Every citation below carries a confidence marker in
the "Citations to verify" list at the end. Nothing here was invented; anything I
could not confirm directly is marked UNVERIFIED.)*

## Prose section

**Roofline analysis of LLM inference.** The roofline framing of transformer
inference — prefill compute-bound, decode memory-bound, with the crossover set by
operational intensity — is established. Yuan et al.'s survey and the accompanying
LLM-Viewer tool (arXiv:2402.16363) built per-layer roofline analyses from
architecture metadata and hardware peaks, and are the standard reference for the
claim that batch-1 decode has operational intensity near 1 FLOP/byte and is
therefore bandwidth-limited on essentially all hardware. Davies et al.'s LIMINAL
(arXiv:2507.14397) pushes this furthest: a hardware-agnostic analytical decode
model validated against real hardware at 7.6 % mean absolute error, used to
project onto HBM4 and 3D-stacked DRAM. Chen (arXiv:2605.30571) attacks the
bandwidth assumption empirically for batch-1 decode across four NVIDIA
datacenter GPUs and 44 measured cells, finding that the achieved fraction of the
analytic memory floor *falls* as peak bandwidth rises — 81 % on an L4 against
27 % on an H100. That result is the direct empirical justification for carrying an
efficiency factor at all, and it is why we fit η rather than assume it. All three
target datacenter parts; none uses llama.cpp, consumer silicon, or MoE.

**Performance prediction and simulation.** A second line predicts rather than
explains. Vidur (MLSys 2024, arXiv:2405.05465) profiles operators and fits
regressors to simulate serving-system throughput to within 9 %; NeuSight (ASPLOS
2025, arXiv:2407.13853) decomposes kernels into tiles and forecasts latency on
*unseen* GPUs at 8.7 % mean error, explicitly bounding predictions by roofline
laws. Imai et al. (MLForSystems @ NeurIPS 2024) hybridise the two, letting a
roofline handle the compute- and bandwidth-bound regimes while regression absorbs
the overhead-bound residual — architecturally the same move as our η, but
evaluated on vLLM and Triton in the cloud and in-distribution rather than held-out.
These systems assume a serving stack: batching, scheduling, tensor parallelism,
queueing. Their inputs are operator graphs and profiling traces, not a file on
disk. None of them answers "what will this GGUF do on my laptop."

**On-device and consumer benchmarking.** This is where the gap has narrowed
sharply, and it is where this work must be positioned honestly. RooflineBench (Bi
et al., arXiv:2602.11506) is the closest prior work by a wide margin: it applies
roofline analysis to on-device LLMs, drives llama.cpp via `llama-bench`,
empirically measures peak FLOPS and bandwidth with synthetic PyTorch CUDA/MPS
matmuls rather than trusting spec sheets, and reports across RTX 3090, RTX 3070 Ti
Laptop, Apple M1 Pro, Jetson Orin Nano and Raspberry Pi 5. Its contribution is a
comparative metric (Relative Inference Potential) and a characterisation of how
operational intensity regresses with model depth and sequence length. Critically
for us, it fits no efficiency coefficient, holds out no models, and states that
MoE is deferred to future work because "the stochastic nature of token routing
complicates the estimation of the execution ceiling." Alongside it, Benazir and
Lin (arXiv:2508.08531; ACM SIGMETRICS/POMACS 2026) characterise Apple Silicon
unified memory against NVIDIA across 5 testbeds, 5 model scales and 14
quantisation schemes, profiling dequantisation overhead and bandwidth at runtime
and refuting the assumption that lower precision always means faster inference.
Chen et al. (arXiv:2508.11269) build ELIB and an edge MBU metric over three
platforms; Song et al. (arXiv:2505.15030) evaluate seven PTQ methods on commodity
hardware and locate the throughput knee near 3.5 effective bits per weight. All of
these are descriptive: they measure, they do not predict a model they have not run.

**MoE inference and the capacity/bandwidth split.** That an MoE's memory footprint
scales with total parameters while its per-token traffic scales with active
parameters is not a new observation. MoE-CAP (Jiang et al., arXiv:2412.07067)
formalises it as a metric — Sparse Memory Bandwidth Utilization (S-MBU) and S-MFU,
defined over activated rather than total parameters, precisely because vanilla MBU
and MFU overestimate MoE cost. The systems literature attacks the consumer case
directly through offloading: Eliseev and Mazur (arXiv:2312.17238) combine expert
caching, prefetching and mixed quantisation to run Mixtral on consumer GPUs;
Pre-gated MoE (ISCA 2024) restructures gating so experts can be prefetched from
host memory. This work does not propose an offloading mechanism. It takes the
sparsity ratio as a term in a predictor and asks whether that term buys measurable
accuracy on a fixed runtime.

**llama.cpp, GGUF and quantisation.** Kurt (arXiv:2601.14277) gives a unified
empirical comparison of llama.cpp K-quant and legacy formats on a single model,
measuring perplexity alongside CPU prefill and decode throughput. It is a lookup
table for one model on one machine — exactly the artefact whose shelf life this
work is trying to escape — but it confirms that format choice is not reducible to
bits per weight, consistent with our §5.2.

**KV cache and long context.** The per-token KV formula (2·layers·kv_heads·
head_dim·bytes) and the resulting crossover, where the KV term overtakes the
weight term and arithmetic intensity collapses, are textbook. Our contribution
here is not the formula but keeping it *inside* the predictor's denominator and
sweeping context depth to test it, rather than letting context-dependence
contaminate a fitted efficiency constant.

---

## Novelty assessment

**Verdict: partially novel, and §1 as written overclaims.** The core mechanism —
a roofline over bytes-per-token with a fitted efficiency factor, applied to
consumer hardware, driven through `llama-bench`, with bandwidth measured rather
than taken from a spec sheet — is substantially published.

**Closest single piece of prior work: RooflineBench, Bi et al., arXiv:2602.11506
(Feb 2026, v4 Aug 2026).** It shares the roofline framing, the on-device consumer
hardware target (including one Apple Silicon part and two NVIDIA consumer GPUs),
the llama.cpp/`llama-bench` measurement path, and the practice of microbenchmarking
peak FLOPS and bandwidth per machine instead of trusting datasheets. If a reviewer
knows one paper in this space, it will be this one.

How this paper genuinely differs from it — in descending order of strength:

1. **Predictor vs. metric.** RooflineBench produces a comparative efficiency
   metric and a characterisation. It fits nothing and predicts nothing. This paper
   fits η per (machine, format) and emits a tok/s number for a model it has never
   run. That is a different object, and it is defensible.
2. **A pre-registered train/test split with out-of-sample validation.** No paper I
   found in the consumer/on-device roofline literature holds out model families.
   NeuSight does hold out GPUs, but for PyTorch kernels in the datacenter. Fixing
   the split at download time (§4.2) is a real methodological differentiator and
   should be foregrounded far more than it currently is.
3. **MoE.** RooflineBench explicitly defers sparse activation to future work.
   MoE-CAP formalises sparsity-aware bandwidth accounting but as a *benchmarking
   metric* on serving systems, not as a term in a single-machine predictor, and
   does not evaluate the unified-memory-vs-small-VRAM contrast. The B1/B2 ablation
   is therefore still novel *as an ablation*, but the underlying insight
   ("bandwidth scales with active params") must be cited to MoE-CAP, not claimed.
4. **The measurement noise floor (§5.1).** I found no prior work establishing a
   between-run CV and a 2σ detection threshold for consumer LLM benchmarking, and
   no prior work quantifying the 15–26 % corruption from uncontrolled background
   load. This is, in my judgement, the paper's most defensible and most underrated
   contribution, and it is currently buried in §5.1 rather than in §1.
5. **The offload cliff.** A discrete GPU with 128 GB of host RAM and a swept
   `n_gpu_layers` is a configuration the MoE-offloading systems papers optimise but
   do not *model analytically*. Still PENDING, but novel if delivered.

What is **not** novel and must be reframed rather than claimed:

- Roofline for batch-1 decode on consumer hardware (RooflineBench).
- Measuring rather than assuming device bandwidth for calibration (RooflineBench
  already uses PyTorch MPS/CUDA synthetic matmuls for exactly this).
- The "calibration trap" of §3.3 — that CPU-side STREAM cannot represent Apple
  Silicon GPU bandwidth. Benazir and Lin profile Apple unified-memory bandwidth at
  runtime; RooflineBench uses the MPS path specifically. The *specific numbers*
  (67.9 vs 313.8 GB/s on M4 Max) may be new, but the phenomenon is not a finding —
  demote it from "a finding" to "a methodological warning we quantify."
- The observation that quantisation is not free bandwidth (§5.2). Benazir and Lin
  already "debunk" that claim and attribute it to dequantisation overhead. Our
  contribution is the specific ~20 % K-quant figure, not the direction.
- An efficiency factor as a modelling device. It has a name in the literature —
  **Model Bandwidth Utilization (MBU)**, from the Databricks inference-performance
  post, and extended to MoE as S-MBU. η should be explicitly identified with MBU;
  presenting it as novel will be caught.

**Accuracy bar.** LIMINAL reports 7.6 % MAE and NeuSight 8.7 % on unseen GPUs. The
current 10.4 % MAPE (§5.3) is worse than both, on an easier problem (one runtime,
one host, three dense models). §5.3's own admission that "10.4 % is not yet a
publishable accuracy" is correct and the bar is concretely 7–9 %.

**Bottom line: not scooped, but the framing is.** The gap that remains is real —
calibrated + sparsity-aware + out-of-sample + consumer + llama.cpp is a
combination nobody has published — but it is a combination, not a new idea, and
every individual component now has a citation against it. The paper should be
pitched as *the first predictor* in a space that currently only has
*characterisations*, and it should lead with the train/test discipline and the
noise floor, not with "prior work is datacenter-focused."

---

## Citations to verify

Confidence markers: **VERIFIED-FETCHED** = I fetched the arXiv/publisher page and
read the abstract in this task. **VERIFIED-SEARCH** = the exact title and
identifier appeared in a search result during this task, but I did not open the
page; the title and ID are trustworthy, secondary details (venue, author list)
are not. **UNVERIFIED** = I believe it exists but saw only a paraphrase.

### Load-bearing (cite these; verify first)

| Ref | Confidence | Notes |
|---|---|---|
| **RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis.** Zhen Bi, Xueshu Chen, Luoyang Sun, Yuhang Yao, Qing Shen, Jungang Lou, Cheng Deng. arXiv:2602.11506, submitted 12 Feb 2026, v4 5 Aug 2026. | VERIFIED-FETCHED | Abstract page **and** HTML v2 fetched. Confirmed: hardware = RTX 3090, RTX 3070 Ti Laptop, Apple M1 Pro, Jetson Orin Nano Super 8G, Raspberry Pi 5; runtime = llama.cpp / `llama-bench`; hardware characterisation via PyTorch CUDA + MPS synthetic matmul; no fitted efficiency factor; no held-out validation; MoE deferred to future work (Appendix B). **This is the paper to read first.** |
| **MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems.** Yinsicheng Jiang, Yao Fu, Yeqi Huang, Ping Nie, Zhan Lu, Leyang Xue, Congjie He, Man-Kit Sit, Jilong Xue, Li Dong, Ziming Miao, Dayou Du, Tairan Xu, Kai Zou, Edoardo Ponti, Luo Mai. arXiv:2412.07067, 10 Dec 2024, v6 19 Nov 2025. Defines S-MBU / S-MFU. | VERIFIED-FETCHED (arXiv). **Venue UNVERIFIED** — a neurips.cc virtual-poster URL for NeurIPS 2025 Datasets & Benchmarks appeared in search results, and a second arXiv ID (2505.11415) exists with the same title. Confirm which to cite. |
| **LIMINAL: Exploring The Frontiers of LLM Decode Performance.** Michael Davies, Neal Crago, Karthikeyan Sankaralingam, Christos Kozyrakis. arXiv:2507.14397, 18 Jul 2025, v2 13 Nov 2025. 7.6 % MAE analytical decode model. | VERIFIED-FETCHED |
| **Profiling Large Language Model Inference on Apple Silicon: A Quantization Perspective.** Afsara Benazir, Felix Xiaozhu Lin. arXiv:2508.08531, 12 Aug 2025. | VERIFIED-FETCHED |
| **Benchmarking and Characterization of Large Language Model Inference on Apple Silicon.** ACM POMACS / SIGMETRICS 2026. DOIs seen: 10.1145/3771563 and 10.1145/3801489.3806865. | VERIFIED-SEARCH — almost certainly the published version of arXiv:2508.08531; author list not confirmed against the ACM page. Cite the ACM version if it matches. |
| **Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode.** Josef Chen. arXiv:2605.30571, 28 May 2026, cs.AR. H100/A100-80GB/L40S/L4, 44 cells, 2048–16384 context. | VERIFIED-FETCHED. Single-author 2026 preprint, no venue — weigh accordingly, but the L4-81 %-vs-H100-27 % result is directly useful to §3.1. |
| **LLM Inference Unveiled: Survey and Roofline Model Insights.** Zhihang Yuan et al. arXiv:2402.16363. Tool: LLM-Viewer. | VERIFIED-SEARCH — title, ID and lead author confirmed; full author list not. |
| **Vidur: A Large-Scale Simulation Framework for LLM Inference.** arXiv:2405.05465. MLSys 2024. | VERIFIED-SEARCH — arXiv ID, MLSys 2024 proceedings URL and github.com/microsoft/vidur all seen. Author list not confirmed. |
| **Forecasting GPU Performance for Deep Learning Training and Inference** (NeuSight). arXiv:2407.13853. ASPLOS 2025, DOI 10.1145/3669940.3707265. | VERIFIED-SEARCH — title, arXiv ID and ACM DOI all seen. Authors not confirmed. |
| **Predicting LLM Inference Latency: A Roofline-Driven ML Method.** Saki Imai et al. MLForSystems workshop @ NeurIPS 2024. PDF at mlforsystems.org/assets/papers/neurips2024/paper28.pdf; also listed by IBM Research and neurips.cc. | VERIFIED-SEARCH — title, first author and PDF URL seen. Full author list and exact workshop name unconfirmed. Workshop paper, not archival. |
| **Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct.** Uygar Kurt. arXiv:2601.14277, 11 Jan 2026. | VERIFIED-FETCHED. Single author, single model, CPU-only, no venue — cite as a data point, not as authority. |
| **A Systematic Evaluation of On-Device LLMs: Quantization, Performance, and Resources.** Qingyu Song, Rui Liu, Wei Lin, Peiyu Liao, Wenqian Zhao, Yiwen Wang, Shoubo Hu, Yining Jiang, Mochun Long, Hui-Ling Zhen, Ning Jiang, Mingxuan Yuan, Qiao Xiang, Hong Xu. arXiv:2505.15030, 21 May 2025, v5 16 Mar 2026. | VERIFIED-FETCHED |
| **Inference performance evaluation for LLMs on edge devices with a novel benchmarking framework and metric** (ELIB). Hao Chen, Cong Tian, Zixuan He, Bin Yu, Yepang Liu, Jialun Cao. arXiv:2508.11269, 15 Aug 2025. | VERIFIED-FETCHED |
| **Fast Inference of Mixture-of-Experts Language Models with Offloading.** arXiv:2312.17238. | VERIFIED-SEARCH — title and ID seen. Commonly attributed to Eliseev and Mazur; **author names UNVERIFIED, confirm before citing by name.** |
| **Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.** ISCA 2024. Camera-ready PDF hosted by Microsoft Research (`isca24_pregated_moe_camera_ready.pdf`). | VERIFIED-SEARCH — filename confirms venue and year; exact subtitle and authors UNVERIFIED. |
| **Model Bandwidth Utilization (MBU).** Databricks engineering blog, "LLM Inference Performance Engineering: Best Practices," databricks.com. | VERIFIED-SEARCH (URL seen). Blog post, not peer-reviewed — cite as the origin of the term only. |

### Supporting / optional

| Ref | Confidence | Notes |
|---|---|---|
| **A Systematic Characterization of LLM Inference on GPUs.** arXiv:2512.01644. Reports FFN kernels shifting from compute-bound on datacenter to memory-bound on edge. | VERIFIED-SEARCH (title + ID only) |
| **Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference.** arXiv:2505.06461. | VERIFIED-SEARCH (title + ID only) |
| **lm-Meter: Unveiling Runtime Inference Latency for On-Device Language Models.** arXiv:2510.06126. | VERIFIED-SEARCH (title + ID only) |
| **Native LLM and MLLM Inference at Scale on Apple Silicon.** arXiv:2601.19139. Reports M4 Max 128 GB @ 546 GB/s and 21–87 % higher throughput than llama.cpp. | VERIFIED-SEARCH (title + ID). First author possibly Wayner Barrios — UNVERIFIED. Relevant to §6 "one runtime" threat. |
| **Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems.** arXiv:2512.16473. | VERIFIED-SEARCH (title + ID only) |
| **Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling** (LIFE). arXiv:2508.00904. | VERIFIED-SEARCH (title + ID only) |
| **Viability and Performance of a Private LLM Server for SMBs: A Benchmark Analysis of Qwen3-30B on Consumer-Grade Hardware.** arXiv:2512.23029. RTX 5090. | VERIFIED-SEARCH (title + ID only) |
| **KernelSight-LM: A Kernel-Level LLM Inference Simulator.** arXiv:2606.28565. | VERIFIED-SEARCH (title + ID only) |
| **Toward Efficient Inference for Mixture of Experts.** Haiyang Huang et al., NeurIPS 2024. PDF at seas.upenn.edu (`huang24-neurips.pdf`). Expert buffering. | VERIFIED-SEARCH (title, first author, filename) |
| **LLM-Para** — GitHub `dengls24/LLM-para`. Roofline + energy roofline over 13 operator types incl. MoE/MLA, 24 hardware platforms, multi-tier memory decode model. | VERIFIED-SEARCH (repo URL). Software artefact, not a paper. **Check whether it has an associated preprint — if so it is a second close competitor on the MoE-roofline axis.** |
| **LLMRoofline** — GitHub `feifeibear/LLMRoofline`. | VERIFIED-SEARCH (repo URL) |
| **LLMCompass**, "Enabling Efficient Hardware Design for Large Language Model Inference." | UNVERIFIED — appeared only inside a search-result summary, no direct hit. Do not cite until confirmed. |
| **LLMServingSim**, HW/SW co-simulation for LLM serving. | UNVERIFIED — paraphrase only. |
| **SYNPERF**, claimed 6.1 %/11.4 % MAPE kernel-level, outperforming NeuSight. | UNVERIFIED — paraphrase only. |
| **AIConfigurator**, table-interpolation latency tool. | UNVERIFIED — paraphrase only. |

### Explicitly *not* found

I found **no** prior work that establishes a measurement noise floor (between-run
CV, detection threshold) for consumer LLM benchmarking, and **no** prior analytical
model of the partial-offload (`n_gpu_layers`) cliff on a discrete consumer GPU with
large host RAM. Absence of evidence after 11 searches is weak evidence, but these
are the two clearest open lanes.

---

## Framing changes required

§1 of `paper.md` contains three sentences that are now false or unsupportable, and
the Contributions list needs two edits.

**1. Line 21–23 — "Published LLM inference measurements are overwhelmingly
datacenter measurements: server GPUs, large batches, tensor parallelism."**

FALSE as of 2026. There is a substantial on-device/consumer measurement literature
(RooflineBench; Benazir & Lin at SIGMETRICS; Song et al.; ELIB; lm-Meter;
arXiv:2505.06461; Kurt). Replace with something like:

> A consumer-hardware measurement literature now exists — RooflineBench, ELIB, and
> the SIGMETRICS characterisation of Apple Silicon all measure llama.cpp-class
> inference on laptops, single consumer GPUs and SBCs. What it does not yet do is
> *predict*. Every one of these is a characterisation of models that were run; none
> emits a number for a model that was not.

**2. §2 placeholder, lines 76–79 — "prior roofline treatments of LLM inference are
analytical or datacenter-focused."**

FALSE. RooflineBench is roofline, on-device, consumer, and llama.cpp-driven. This
sentence must be deleted outright. The surviving half of the positioning —
"prior consumer benchmarking is descriptive and does not produce a transferable
predictor" — **is** supportable and I verified it against every consumer paper
found. Keep that one.

**3. §2 placeholder — "prior MoE performance work targets multi-GPU serving, not
the single-machine capacity/bandwidth tradeoff."**

Half false. The MoE *offloading* literature (arXiv:2312.17238, Pre-gated MoE,
arXiv:2512.16473) targets exactly the single-machine case, and MoE-CAP already
formalises sparsity-aware bandwidth accounting. Reword to: prior MoE work either
*optimises* single-machine execution (offloading systems) or *measures* it
(MoE-CAP's S-MBU); none uses the sparsity ratio as a term in an a-priori
throughput predictor calibrated on other models.

**4. Contribution 1 (line 57) — "A two-regime performance model for local
llama.cpp inference whose only per-model inputs are the GGUF file size and
metadata fields."**

Overclaims as written; a reader who knows RooflineBench will read this as already
done. Add the discriminator to the sentence itself: *"…whose only per-model inputs
are the GGUF file size and metadata fields, and which is fitted on one set of
models and evaluated on another."* The fitting-and-holding-out is the novelty, not
the two-regime structure.

**5. Contribution 3 — out-of-sample validation — should be promoted to
contribution 1.** It is the only contribution with no direct competitor in the
consumer literature. Reorder.

**6. Add the noise floor as an explicit contribution.** §5.1 currently reads as
preliminary throat-clearing. It is the strongest unclaimed result in the paper and
I found no prior art. Promote it into the Contributions list and into the abstract.

**7. §3.3 — demote the "trap" from finding to warning.** The sentence "we report it
because a calibration procedure that is wrong on one of the two dominant consumer
architectures is not a calibration procedure" is fine as motivation, but the
framing that this is a *discovery* will not survive review: RooflineBench already
measures the device path via MPS, and Benazir & Lin profile Apple unified-memory
bandwidth directly. Reword as "we quantify a trap that the correct prior practice
avoids implicitly," and cite both.

**8. §5.2 — attribute the direction, keep the magnitude.** "Quantisation is not
free bandwidth" is already published (Benazir & Lin explicitly debunk the
lower-precision-is-always-faster claim). Cite them, then claim the specific ~20 %
K-quant figure and the cross-model η spread as the new content.

**9. Set the accuracy bar explicitly in §6.** LIMINAL is at 7.6 % MAE and NeuSight
at 8.7 % on unseen GPUs. §5.3's current 10.4 % is above both. Either state why the
comparison is not apples-to-apples (different regime, different hardware class,
harder extrapolation across model families rather than across GPUs) or treat 7–9 %
as the target. Do not report 10.4 % without addressing this — a reviewer will make
the comparison whether or not the paper does.

**10. Identify η with MBU by name.** η_d is Model Bandwidth Utilization; the MoE
form is MoE-CAP's S-MBU. Naming it and citing it costs one sentence and removes an
obvious "reinventing a known metric" objection.
