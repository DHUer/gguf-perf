# Ten Minutes of Calibration: Predicting Local LLM Inference Throughput on Consumer Hardware

> **ARCHIVED WORKING DRAFT — DO NOT CITE NUMBERS FROM THIS FILE.** The current
> ICASSP 2027 paper and complete results appendix are in `paper/icassp2027/`.
> This file preserves superseded mixed-session analyses and pending experiment
> notes; it is not the submission source or an authoritative results record.
>
> **STATUS.** Sections 1–4 are complete. Section 5 (Results) is generated from
> `results/` and is filled in only for measurements that have actually been run.
> Any cell not yet measured is listed as PENDING rather than estimated. No number
> in this document is a placeholder or an invention.

## Abstract

Open-weight language models are increasingly run on personal computers, but the
question a local user actually faces — what will *this* model do on *my*
machine — is answered today only by downloading the model and trying it. The
consumer measurement literature characterises models that were run; it does not
predict models that were not.

We present a calibrated two-regime performance model for llama.cpp inference
whose only per-model inputs are a GGUF file's size and the architecture fields
inside its header, so it can be evaluated before the weights are downloaded.
Decode is treated as memory-bound and prefill as compute-bound, with one fitted
efficiency coefficient per regime and quantisation format. Fitted on one set of
models and evaluated on another, it predicts decode throughput to **10.9 %** and
prefill rate to **18.8 %** mean absolute percentage error out of sample across
19 current open-weight models on Apple Silicon.

Two terms carry the accuracy, and both are consequences of the 2026 model
cohort rather than of the roofline itself. Charging Mixture-of-Experts models
for *activated* rather than total parameters reduces out-of-sample error from
49.6 % to 10.9 %; a total-parameter model is not merely inaccurate but
physically impossible, implying bandwidths several times the machine's ceiling.
Summing the KV cache *per layer* — respecting sliding-window and recurrent
layers instead of assuming uniform global attention — corrects an overestimate
of up to 13.7× on models where most layers are windowed. We also find that
prefill is not quantisation-independent, contrary to the usual reading of the
two-regime split, and that a plausible output-projection correction improves
training error while doubling out-of-sample error, and must be rejected.

Our largest practical finding concerns measurement rather than modelling. On
the corporate-managed host used here, background monitoring agents drove the
load average between 12 and 396 with no user process running, and identical
cells re-measured up to 2.8× apart. Every prediction failure we could not
explain architecturally turned out to be a measurement whose own within-run
standard deviation had already flagged it. Conditioning the measurements —
gating on system load, retrying on instrument-reported instability — cut
out-of-sample error from 17.9 % to 10.9 % and collapsed an eight-point
train/test gap that had read as overfitting to 0.6 points. **We therefore
recommend that consumer-hardware benchmarks record and publish system load and
per-measurement variance alongside every throughput number**; without them a
reader cannot distinguish a result from an artefact, and we found no prior work
in this area that does so.

Harness, raw per-cell measurements, and figure generation are released.

**Scope.** Results are for llama.cpp at batch size one on a single Apple
Silicon host. Cross-hardware generalisation is stated as future work:
leave-one-machine-out validation is implemented but undefined on one machine.

## 1. Introduction

A consumer-hardware measurement literature now exists. RooflineBench, ELIB, and
the SIGMETRICS characterisation of Apple Silicon all measure llama.cpp-class
inference on laptops, single consumer GPUs and single-board computers, and
several already calibrate against microbenchmarked peak bandwidth rather than
vendor spec sheets. What none of them does is *predict*: every one is a
characterisation of models that were run, and none emits a number for a model
that was not run.

That gap is what this paper addresses. The population running open-weight
models locally does so at batch size one, where the binding constraint is bytes
rather than FLOPs, and the practical question is not "which model is best" but
"what will *this* model do on *my* machine" — asked before spending an hour
downloading 40 GB to find out.

Answering that with a benchmark table has a short shelf life — the table is
obsolete as soon as the next model cohort ships. Answering it with a
*calibrated model* does not, provided the model's inputs are things you can
read off a file rather than measure.

Two properties of the 2026 model landscape make this worth revisiting rather
than a textbook exercise:

**Mixture-of-Experts is now mainstream, and it decouples capacity from
bandwidth.** A dense model's memory footprint and its per-token memory traffic
are the same number. For a sparse MoE they differ by up to an order of
magnitude: total parameters determine whether the model *fits*, active
parameters determine how fast it *runs*. A 117B-total / ~5B-active model
occupies ~63 GB but streams only a few GB per token. On a machine with 128 GB
of unified memory it therefore runs at roughly the speed of a small dense
model; on a 16 GB discrete GPU the same file does not fit at all and collapses
into CPU offload. The naive "bigger model is slower" intuition, and any
performance model built on total parameter count, is simply wrong for this
class.

**Consumer memory architectures have diverged.** A discrete GPU offers high
bandwidth behind a hard capacity wall, with a catastrophic PCIe-bound cliff
past it. Unified-memory Apple Silicon offers lower peak bandwidth but a much
larger capacity with no cliff. These are not two points on one axis; they
trade off in opposite directions, and which one wins depends on the model's
sparsity and on whether the workload is prefill- or decode-dominated.

### Contributions

Ordered by how much of each is unclaimed in the existing consumer literature.

1. **A noise floor and detection threshold for consumer LLM benchmarking**
   (§5.1). Between-run coefficient of variation is 1.6 % under a controlled
   protocol and 15–26 % with background load present — an order of magnitude
   apart. We are not aware of prior work that establishes a repeatability
   figure or a minimum detectable effect size for this class of measurement,
   which means the field currently has no way to tell a result from noise.
2. **Out-of-sample throughput prediction.** The model is fitted on one set of
   models and evaluated on another, including an architecture class held out
   entirely (§5). Existing consumer roofline work characterises models it ran;
   it does not predict models it did not.
3. A two-regime performance model whose only per-model inputs are the GGUF file
   size and metadata fields, *fitted on one set of models and evaluated on
   another* (§3). The two-regime structure itself is not new; the fitting and
   holding out is.
4. An ablation isolating what the MoE sparsity term is worth (§5). Sparsity-aware
   bandwidth accounting exists in the datacenter literature as S-MBU; using the
   sparsity ratio as a term in an *a-priori* predictor calibrated on other
   models does not.
5. An analytical treatment of the partial-offload regime (§5.8), for which we
   found no existing model.
6. A released, resumable, cross-platform measurement harness (§4).

Note on terminology. The efficiency factor η_d introduced in §3.1 is Model
Bandwidth Utilization (MBU); its sparsity-aware form is **Sparse Memory
Bandwidth Utilization (S-MBU)**, defined in MoE-CAP (Jiang et al.,
arXiv:2412.07067 — VERIFIED against the primary source; an arXiv preprint, not
a published venue, so cite it as such).

The same source defines **Sparse Model FLOPS Utilization (S-MFU)**, which is the
compute-side analogue and therefore the existing name for our prefill
coefficient η_p (§3.2). We adopt both names rather than coining new ones, and
claim novelty only for using these quantities as *fitted terms in an a-priori
predictor* rather than as reporting metrics.

## 2. Related work

See `paper/related_work.md` for the drafted section, the per-citation
verification status, and the closest-competitor analysis. Summary of the
positioning that survived checking:

- **Prior consumer roofline work exists and is close.** RooflineBench (Bi et
  al., *RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline
  Analysis*, arXiv:2602.11506) — VERIFIED against the primary source — shares
  the roofline framing and the on-device/edge hardware setting, and introduces
  a comparative metric ("Relative Inference Potential") for models on a shared
  hardware substrate.

  What it does *not* do, confirmed from its abstract: fit an efficiency
  coefficient by regression, make out-of-sample predictions, or validate on
  held-out models. It characterises; it does not predict. **That is the gap this
  paper occupies**, and it is the one part of the original positioning that
  survived verification intact.

  Two differences of emphasis. RooflineBench targets Small Language Models on
  resource-constrained edge hardware; this work targets the desktop/laptop class
  and models up to 63 GB. It reports a regression in operational intensity as
  model depth increases and attributes gains to Multi-head Latent Attention;
  our depth-related finding (§5.5) concerns per-layer KV accounting and is
  distinct, but the two should be compared directly before submission.

  **CORRECTED — two earlier claims were unsupported.** A prior draft stated that
  RooflineBench is "llama.cpp driven through `llama-bench`" and "explicitly
  defers MoE to future work". Neither appears in the abstract. Both came from a
  literature-survey agent and were propagated without checking. They may be true
  of the body text, but must be verified there before being asserted, and no
  argument in this paper should rest on them.
- **Benazir & Lin (SIGMETRICS 2026) is the closest competitor on setting, and
  closer than our survey reported.** *Benchmarking and Characterization of Large
  Language Model Inference on Apple Silicon* (arXiv:2508.08531; POMACS,
  doi:10.1145/3771563 — VERIFIED) studies the same class of machine we do,
  single-request inference on personal workstations, explicitly excluding
  datacenter GPUs. It profiles dequantisation overhead, finds Apple Silicon
  becomes arithmetic-bound when that overhead is significant, and **debunks the
  claim that lower precision means faster inference**. It evaluates five
  testbeds — M2 Ultra, M2 Max, M4 Pro, RTX A6000, 2×RTX A6000 — across five
  model scales from 8B to 405B and fourteen quantisation schemes.

  **We must concede more than the earlier draft did.** Our §5.2 and §5.4
  quantisation findings are pre-empted in direction and mechanism; we contribute
  the magnitude on a current model cohort, not the phenomenon. Our introduction's
  argument that large unified memory favours very large models is also theirs.
  And their five hosts against our one is a straightforward disadvantage on
  exactly the axis reviewers weight.

  **What remains ours:** they characterise and profile; they do not fit a
  predictor or validate on held-out models. Sparsity as a fitted term, per-layer
  KV accounting, and the measurement-conditioning results are not in their scope.
  The paper's contribution should be narrowed to those, explicitly.

- **Prior consumer benchmarking is descriptive and produces no transferable
  predictor.** This still holds — including for Benazir & Lin — and is now the
  load-bearing half of the positioning.
- **Prior MoE work either optimises single-machine execution (offloading
  systems) or measures it.** MoE-CAP (arXiv:2412.07067 — VERIFIED) introduces
  Sparse Memory Bandwidth Utilization (S-MBU) and Sparse Model FLOPS Utilization
  (S-MFU) as sparsity-aware *benchmarking* metrics for MoE systems, alongside a
  cost/accuracy/performance trade-off analysis. It reports these quantities; it
  does not fit them and predict unseen models with them. None of this literature
  uses the sparsity ratio as a term in an a-priori throughput predictor
  calibrated on other models — which is the specific claim this paper makes.

Two claims from the original draft were false and have been removed: that
published measurements are overwhelmingly datacenter, and that prior roofline
treatments are analytical or datacenter-focused.

**Before submission**: four references are flagged do-not-cite pending
verification, and two need specific checks. See the "Citations to verify" list.

## 3. Model

### 3.1 Decode: memory-bound

At batch size one, generating each token requires reading the active weights
and the KV cache. With `n_active/n_total = 1` for dense models:

```
bytes_per_token(c) = file_bytes · (n_active / n_total) + kv_bytes(c)
tok/s_decode       = η_d · BW / bytes_per_token(c)
```

The KV term is summed **per layer**, not as a single product over the model.
Writing it as `2 · n_layers · n_kv_heads · head_dim · c · sizeof(kv_dtype)`
assumes every layer keeps a full global cache growing linearly with context,
which is false for much of the 2026 cohort and costs an order of magnitude on
those models (§5.5). Instead, for layer *i* with `h_i` KV heads:

```
kv_bytes(c) = Σ_i  2 · h_i · head_dim · c_i · sizeof(kv_dtype)  +  ssm_state
c_i = min(c, window)  if layer i is sliding-window, else c
h_i = 0               if layer i is recurrent (SSM/Mamba): no KV cache
```

`file_bytes` is the GGUF file size, known without loading the model. Every other
term is a GGUF metadata field: `n_expert` / `n_expert_used` for sparsity,
`attention.head_count_kv` for `h_i` (array-valued where layers differ),
`attention.sliding_window` and `attention.sliding_window_pattern` for the
windowing, and the `ssm.*` group for recurrent state. Active parameters are
computed by summing tensor shapes and discounting inactive experts. **No term
requires running the model.**

`η_d ∈ (0,1]` is the fraction of the machine's demonstrated bandwidth ceiling
actually achieved — Model Bandwidth Utilization. It absorbs dequantisation cost,
kernel efficiency, and memory-system effects the roofline ignores, and is fitted
per (machine, quantisation format) on training models only. It is bounded by 1
only if the calibration anchor is the machine's ceiling rather than an arbitrary
reference model; §5.3 shows what goes wrong otherwise.

### 3.2 Prefill: compute-bound

Prefill processes the whole prompt in parallel, so it is limited by arithmetic
rather than by weight streaming:

```
tok/s_prefill = η_p · FLOPS / (2 · n_active_params)
```

Sparsity enters here too — an MoE routes each token to a few experts during
prefill as much as during decode — so the denominator is active, not total,
parameters.

The textbook reading of this split is that decode is quantisation-sensitive
(dequantisation costs bandwidth) while prefill is not (arithmetic is done in a
common precision after unpacking). §5.6 shows that reading is too clean: η_p
fitted per quantisation format beats a single per-host η_p, so prefill carries a
quantisation dependence of its own.

### 3.3 Calibration, and a trap

`BW` and `FLOPS` are measured per machine. Three sources are recorded because
they disagree, and the disagreement is itself a finding:

| Source | What it measures | Valid where |
|---|---|---|
| `cpu_triad` | STREAM-style triad from CPU cores | Discrete-GPU hosts, for the host-RAM/offload term |
| `torch_gpu` | Device-side copy bandwidth and FP16 matmul | CUDA only |
| `llm_ref` | Effective bandwidth back-solved from one llama-bench decode run on the smallest model | Any backend |

**Measured on the MacBook Pro M4 Max in this study, the CPU triad reaches 67.9 GB/s
while the GPU sustains 313.8 GB/s of effective decode bandwidth through the
same unified memory** — a factor of 4.6. CPU cores cannot saturate the fabric
the GPU reaches. Using `cpu_triad` as the bandwidth term on a Metal machine
would under-predict throughput by roughly that factor.

We report this as a practical warning rather than as a finding. Existing
consumer work already measures the device path directly on Apple Silicon rather
than through CPU cores, so correct practice avoids the trap implicitly; what is
useful here is naming it, quantifying it on current hardware, and having the
harness select the right source automatically.

The `llm_ref` probe model is **excluded from all reported error metrics**,
since its η is 1.0 by construction.

## 4. Method

### 4.1 Hardware

| Machine | Compute | Memory | Role |
|---|---|---|---|
| Mac Studio, M4 Max (16C CPU / 40C GPU) | Metal | 128 GB unified | Planned validation host; no rows in the current dataset |
| MacBook Pro, M4 Max | Metal | 64 GB unified | Measured Apple host; thermally constrained |
| Lenovo ThinkStation P8, RTX 5080 | CUDA sm_120 | 16 GB VRAM + 128 GB DDR5 | Discrete GPU; hard capacity wall, full offload range |

The planned Apple comparison uses one chip generation across different thermal
envelopes and capacities, but no Mac Studio measurement has yet been collected.
The ThinkStation's
128 GB of host RAM is what makes the offload cliff fully traceable: a model far
larger than VRAM can still be run with a controlled fraction of layers resident.

### 4.2 Models

Current-generation (2026) open-weight models only. Official or well-known
faithful quantisations (`ggml-org`, `google`, `unsloth`, `lmstudio-community`,
`bartowski`); community finetunes with modified weights are excluded because
they change what is being measured.

The train/test split is fixed at download time in `models/manifest.json`, with
an exact tracked snapshot in `results/model_manifest.json`, not at analysis
time — deciding it after seeing errors would turn out-of-sample validation into
cherry-picking. New CSV rows also carry the split; analysis rejects an unknown
or conflicting assignment rather than silently treating it as training data.

- **TRAIN** — dense SmolLM3-3B, Qwen3.5-4B, Qwen3.5-9B, gemma-4-12B (QAT),
  Qwen3.6-27B; MoE gpt-oss-20b (MXFP4), gemma-4-26B-A4B, Qwen3.6-35B-A3B.
- **TEST** — Nemotron-3-Nano-4B, Nemotron-3-Nano-30B-A3B, GLM-4.7-Flash,
  Ling-mini-2.0, and gpt-oss-120b (~117B total / ~5B active) as the extreme
  extrapolation point.

### 4.3 Measurement protocol

Throughput is measured with `llama-bench`, which performs warmup runs, repeats
each test, and reports mean and standard deviation for prefill and decode
separately. We did not write a replacement timing harness; we constrained the
one that exists.

Every default that could confound a cross-model comparison is pinned:

| Setting | Value | Rationale |
|---|---|---|
| flash attention | `on` (pinned) | Default `auto` selects different kernels per model and backend; left on auto, cross-model comparison is invalid |
| KV cache dtype | `f16` for K and V | Directly changes bytes-per-token, the model's denominator |
| threads | physical performance cores | Oversubscription is a silent confound in CPU-side numbers |
| repetitions | 5 | Yields a standard deviation per cell; no single-run numbers are reported |
| warmup | enabled | First-run page-in would otherwise pollute the mean |
| settling delay | ≥20 s before each cell | A thermally-throttled laptop must not be measured hot from the previous cell |
| process isolation | one process per (model, n_gpu_layers) | An OOM cannot corrupt the remaining sweep; fragmentation does not accumulate over a multi-hour queue |

KV depths of 0, 4096, and 16384 tokens exercise the `kv_bytes(c)` term; without
a depth sweep that term is untestable. On the discrete-GPU machine `n_gpu_layers`
is swept to trace the offload cliff. Predictor fits, validation tables, and the
primary figures use only the declared `n_gpu_layers=99` baseline; lower offload
levels are retained exclusively for the offload-cliff analysis.

Every result row records the git commit, llama-bench version, host, and the
full cell identity. The sweep is append-only and resumable.

### 4.4 Baselines

| | Calibrated | Sparsity-aware |
|---|---|---|
| **B0** textbook roofline | no (η=1) | no |
| **B1** calibrated, total params | yes | no |
| **B2** this work | yes | yes |

B1 vs B2 isolates the value of the MoE term; B0 vs B1 isolates the value of
calibration.

## 5. Results

*(Generated from `results/`. PENDING entries are not yet measured.)*

### 5.1 The noise floor, and why protocol is the contribution

Before any effect can be reported, the measurement noise floor has to be
established. We ran one fixed configuration (SmolLM3-3B Q4_K_M, decode) six
times as six independent processes, 45 s idle settling before each:

| Condition | decode between-run CV | max spread |
|---|---|---|
| Idle machine, 45 s settling | **1.61 %** | 4.3 % |
| Concurrent background download, 10 s settling | — | **15–26 %** |

The uncontrolled condition was not designed; it happened, because model
downloads were still running during an early sweep. The same cell measured
1467 then 1091 tok/s prefill (−26 %) and 103.6 then 87.6 tok/s decode (−15 %).
Under the controlled protocol the same quantity is repeatable to 1.6 %.

**An order of magnitude separates the two.** Any consumer-hardware benchmark
that does not control background load and thermal state has a noise floor
larger than most of the effects reported in this literature. Prefill is far
more stable than decode (CV 0.37 % vs 1.61 %).

Between-run CV (1.61 %) is *smaller* than the within-run standard deviation
llama-bench itself reports (2.25 %), so in this configuration the tool's own
error bars are conservative rather than misleading. That is worth stating
because the opposite is often assumed.

The resulting detection threshold on this host is **≈3.2 % (2σ)**. Effects
smaller than that are not measurable and are not claimed anywhere below.

#### The "controlled" condition was never idle

A later audit of the measurement host undermines the word *controlled* above and
must be reported. The machine is a corporate-managed macOS laptop carrying
permanent endpoint-monitoring, data-loss-prevention, telemetry and log-shipping
agents, which together held the one-minute load
average between **64 and 104 on twelve performance cores** with no user process
running at all. Every measurement in this dataset was taken under that load.

Two consequences, neither of which we can argue away:

- **The 1.61 % CV of the table above is a best case, not the typical case.** It
  was obtained in a quiet window that happened to occur; it is not what the
  protocol reliably delivers on this host. The same cell
  (Qwen3.8-27B-Q8_0 decode) later re-measured at 5.2 tok/s against 14.5 tok/s,
  a factor of 2.8, purely from background contention.
- **The detection threshold is therefore optimistic**, and any effect in this
  paper below roughly 10 % should be treated as provisional until re-measured on
  a host that is genuinely quiescent.

This is the paper's own argument turning on the paper. §5.1 exists to say that
consumer measurements without load control have a noise floor larger than the
effects being reported; the authors then produced exactly that failure while
believing the opposite, because "idle" was assumed from the absence of *our own*
processes rather than measured.

The harness now records the one-minute load average before and after every cell
(`load_before`, `load_after`) and flags cells taken above four times the thread
count. Contaminated cells become filterable after the fact instead of invisible.
**Recommended practice, and the concrete methodological output of this section:
record system load alongside every throughput measurement and publish it.** No
consumer-benchmarking paper we found does this, and without it a reader cannot
distinguish a result from an artefact.

### 5.2 Quantisation is not free bandwidth

Controlled protocol, `-r 5`, flash attention pinned, f16 KV, MacBook Pro M4 Max:

| Model | Quant | File | prefill tok/s | decode @0 | @4096 | @16384 | eff. BW @0 |
|---|---|---|---|---|---|---|---|
| SmolLM3-3B | Q4_K_M | 1.92 GB | 1932 | 161.2 | 146.9 | 113.2 | 309 GB/s |
| SmolLM3-3B | Q8_0 | 3.28 GB | 2237 | 117.4 | 110.1 | 89.7 | 385 GB/s |
| Qwen3.5-4B | Q4_K_M | 2.74 GB | 1472 | 103.0 | 94.2 | 79.6 | 282 GB/s |
| Qwen3.5-4B | Q8_0 | 4.48 GB | 1008 | 64.0 | 67.4 | 63.7 | 287 GB/s |

Two effects, both far above the 3.2 % threshold:

- **Decode does not scale with file size alone.** For SmolLM3, Q8_0 is 1.71×
  the bytes of Q4_K_M but only 1.37× slower; back-solved effective bandwidth is
  385 GB/s for Q8_0 against 309 GB/s for Q4_K_M. K-quant dequantisation costs
  ≈20 % of achievable bandwidth. A bytes-only roofline therefore systematically
  mis-ranks quantisation formats. *The direction of this result is already
  published* — Benazir & Lin profile dequantisation overhead on Apple unified
  memory and debunk "lower precision is always faster". We contribute the
  magnitude on current hardware, not the phenomenon, and must cite accordingly.
- **η is not a property of the format alone.** At Q8_0, SmolLM3 reaches
  385 GB/s while Qwen3.5-4B reaches only 287 GB/s — a 34 % gap between two
  models at the same quantisation on the same machine. This is the paper's
  main open problem (§5.7).

### 5.3 Where the sparsity term earns its accuracy

Section 5.5 reports aggregate error. The aggregate hides where the effect
lives, so we disaggregate by architecture and split (Figure 2). All figures are
mean absolute percentage error on the clean-protocol dataset.

| | B0 uncalibrated | B1 total params | B2 active params |
|---|---|---|---|
| train, dense (n=36) | 29 % | 10 % | **9 %** |
| train, MoE (n=9) | 75 % | 50 % | **17 %** |
| test, dense (n=6) | 12 % | 18 % | **10 %** |
| **test, MoE (n=6)** | 82 % | **81 %** | **11 %** |

**On out-of-sample MoE the sparsity term takes error from 81 % to 11 %.** That
single cell is where the paper's headline accuracy comes from. On dense models
the term is the identity by construction and B1 and B2 differ only by noise, so
an aggregate over a mostly-dense set understates the effect by a large factor.

The physical argument is sharper than the error metric. Back-solving effective
bandwidth for gpt-oss-20b from its *total* size gives 1441 GB/s — several times
this machine's achievable memory bandwidth. The total-parameter model is not
merely inaccurate there; it is impossible. Using activated parameters gives
245 GB/s, which is in range. Any predictor that charges MoE decode for weights
it never reads will produce numbers that cannot happen.

One honest wrinkle: calibration makes out-of-sample *dense* prediction slightly
worse (12 % → 18 % for B1), and B2 recovers only to 10 %. With n = 6 this sits
inside the noise of the comparison, but it is the opposite of the expected
direction and is not smoothed over.

### 5.4 The quantisation ladder: dequantisation cost is monotone in bit width

Holding the model fixed (Qwen3.8-27B) and sweeping only the quantisation format
isolates the format's cost from every architectural difference. Efficiency below
is measured relative to pure byte-scaling from a reference model, so 1.0 means
"exactly as fast as its file size predicts".

| Format | IQ2_XXS | Q2_K | IQ3_XXS | Q3_K | Q4_K | Q5_K | Q6_K | Q8_0 |
|---|---|---|---|---|---|---|---|---|
| File (GB) | 7.3 | 9.8 | 10.9 | 13.2 | 17.6 | 20.9 | 25.3 | 29.1 |
| Efficiency | 0.41 | 0.75 | 0.81 | 0.91 | 1.11 | 1.07 | 1.29 | 1.34 |

Efficiency rises monotonically with bit width across a 4× range. Two consequences:

- **Aggressive quantisation is doubly penalised.** It is chosen to reduce bytes,
  but each remaining byte is also delivered less efficiently, so the throughput
  gain is well short of the file-size ratio. Q8_0 achieves 3.3× the efficiency
  of IQ2_XXS on identical weights.
- **I-quants pay extra.** IQ2_XXS (0.41) and IQ3_XXS (0.81) sit below the K-quant
  trend at comparable file size — IQ3_XXS is larger than Q2_K yet only slightly
  more efficient. Their more elaborate decode is not free on a
  bandwidth-saturated machine.

This is the cleanest result in the paper: one model, one machine, one factor
varied, an effect an order of magnitude above the 3.2 % detection threshold.

### 5.5 Out-of-sample prediction error, and where the model breaks

Full dataset: 57 decode measurements, 19 models, 5 MoE, one host. Three further
models serve as calibration probes and are excluded from scoring.

**Every measurement below was taken under the protocol of §5.1 and §5.5**: the
complete model set was re-swept with the load gate and the CV retry active, and
**all 22 successful cells were measured in genuine quiet windows with none
forced through the gate**. One cell failed and is recorded as such: gpt-oss-120b
is a 63 GB file on a 64 GB machine and does not load.

| Predictor | split | n | MAPE | median APE | p90 | max |
|---|---|---|---|---|---|---|
| B0 uncalibrated | train | 45 | 38.5 % | 19.4 % | 78.1 % | 192.7 % |
| B0 uncalibrated | test | 12 | 47.2 % | 49.9 % | 83.4 % | 83.8 % |
| B1 calibrated, total params | train | 45 | 18.3 % | 5.4 % | 72.1 % | 92.1 % |
| B1 calibrated, total params | test | 12 | 49.6 % | 54.6 % | 81.9 % | 82.3 % |
| **B2 calibrated, active params** | train | 45 | **10.3 %** | 5.4 % | 28.8 % | 56.2 % |
| **B2 calibrated, active params** | test | 12 | **10.9 %** | 8.9 % | 22.9 % | 34.5 % |

**Train and test error are now within 0.6 points of each other** (10.3 % against
10.9 %). Under the earlier mixed-protocol dataset the same model showed a
train/test gap of eight points, which read as overfitting; it was measurement
noise concentrated in the smaller split. A predictor that generalises and one
that does not are indistinguishable until the measurements are conditioned.

With five MoE models present the sparsity term is decisive: B2 beats B1 by 3.4×
on train and 1.9× on test. **B1 is now worse than doing no calibration at all**
(56.7 % vs 40.2 % on train, max error 624 %): fitting η per quantisation while
charging MoE models for weights they never read produces η values that are then
misapplied to dense models in the same group. Charging total parameters is not a
weaker model, it is an actively harmful one.

The gap between B2's median (13.4 %) and mean (34.5 %) says the residual is not
diffuse — it is a small number of large failures. Eleven of 63 rows exceed 50 %
error, and they fall into two families.

**Family A — throughput at depth is under-predicted, badly.** Nemotron-3-Nano-30B
at 16 384 tokens measures 84.3 tok/s against a predicted 14.1; Nemotron-3-Nano-4B
measures 80.2 against 13.9; gemma-4-26B-A4B measures 69.1 against 12.6. Mean
error rises with depth (16.5 % → 19.6 % → 23.6 % at depths 0 / 4 096 / 16 384),
and every large failure is a depth failure.

The cause is a modelling assumption, not a bug. The `kv_bytes` term assumes
**every layer keeps a full global KV cache growing linearly with context**. The
2026 cohort has largely abandoned that: Nemotron-3-Nano interleaves Mamba/SSM
layers that keep a constant-size state and no KV cache at all, and Gemma-4 uses
sliding-window attention on most layers, capping KV at the window rather than the
context. For these models the true KV growth is a fraction of the modelled one,
so the predictor invents a memory-traffic wall that does not exist.

**This correction has now been implemented and measured.** `kv_bytes` is summed
per layer, using the window size for sliding-window layers, zero for recurrent
ones, and the per-layer KV head count where GGUF supplies an array rather than a
scalar. The layer inventory comes from metadata already present in every file:
`attention.head_count_kv` (array-valued when layers differ, 0 marking a
recurrent layer), `attention.sliding_window`, `attention.sliding_window_pattern`,
and the `ssm.*` group.

The size of the correction at 16 384 tokens:

| Model | Layers | With attention | Windowed | Uniform KV | Per-layer KV | Overestimate |
|---|---|---|---|---|---|---|
| gemma-4-12b | 48 | 48 | 40 | 12.9 GB | 0.94 GB | **13.7×** |
| gemma-4-26B-A4B | 30 | 30 | 25 | 8.05 GB | 0.76 GB | **10.7×** |
| Nemotron-3-Nano-4B | 42 | 4 | 0 | 2.82 GB | 0.89 GB | 3.2× |
| Nemotron-3-Nano-30B-A3B | 52 | 6 | 0 | 0.87 GB | 0.50 GB | 1.7× |

Effect on prediction, with no parameter added or refitted — the model has the
same two free coefficients as before, it was simply being fed a bytes-per-token
term wrong by up to an order of magnitude for a quarter of the model set:

| | uniform KV | per-layer KV |
|---|---|---|
| train MAPE | 16.5 % | **11.2 %** |
| train p90 | 55.7 % | **27.8 %** |
| test MAPE | 34.5 % | **20.6 %** |
| test p90 | 81.9 % | **29.2 %** |
| rows above 50 % error | 11 / 63 | 4 / 63 |

The tail improves more than the centre: test p90 falls by nearly two thirds
while the test median moves the other way (12.4 % → 18.1 %). The correction is
removing catastrophic misses on hybrid and windowed models rather than
sharpening the typical prediction, which is what a fix to a structural
assumption should look like.

Mean error is also now nearly flat in context depth (14.4 / 15.0 / 17.9 % at
0 / 4 096 / 16 384 tokens, against 16.5 / 19.6 / 23.6 % before). The depth
signature that motivated the diagnosis is gone.

**A roofline built on uniform global attention mispredicts a large and growing
share of open-weight models**, and the metadata needed to fix it is already in
the files. That is a finding, not a defect, and it is the single largest
accuracy improvement in this work.

**Family B — two models are anomalously slow at depth 0.** Qwen3.5-4B delivers
0.40 of what byte-scaling predicts (45.7 tok/s where 115 is expected) and
Qwen3.8-27B-IQ2_XXS delivers 0.41. The IQ2_XXS case is explained by §5.4.

**The Qwen3.5-4B case was measurement contamination, not architecture, and
re-measurement confirms it.** llama-bench records a within-run standard deviation across its
five repetitions, and that quantity — available for every row, and computed
before any prediction is made — turns out to rank the residuals almost exactly:

| Row | within-run CV | status in the model |
|---|---|---|
| Qwen3.5-4B Q4_K_M, depth 0 | **14.9 %** | largest unexplained outlier |
| Muse-Glimmer-30B, depth 0 | 9.9 % | outlier |
| gemma-4-26B-A4B, depth 16384 | 9.3 % | residual Family A failure |
| Qwen3.5-4B Q8_0, depth 0 | 8.8 % | outlier |
| *median across all 66 decode rows* | *1.83 %* | — |

Every model the predictor fails on is a model whose own measurement was
unstable. Excluding rows by measurement quality alone:

| Within-run CV filter | rows | train MAPE | test MAPE |
|---|---|---|---|
| none | 57 | 10.7 % | 19.3 % |
| ≤ 10 % | 56 | 9.0 % | 18.7 % |
| ≤ 5 % | 51 | 8.9 % | 20.5 % |
| ≤ 3 % | 42 | **7.9 %** | **15.9 %** |

Dropping the single worst row takes train error from 10.7 % to 9.0 %.

That filter was, however, discovered *after* seeing the residuals, which makes
it inadmissible as a result. The correct response is not to delete unstable
cells but to re-measure them under a rule declared in advance.

#### Quality-gated measurement, and what it recovered

The harness now retries a cell while llama-bench's own within-run CV exceeds a
threshold (`--max-cv`, default 3 %), backing off further on each attempt, and
keeps the lowest-CV attempt. The criterion is a property of the measurement, is
fixed before the sweep runs, and is recorded per row (`attempts`,
`kept_cv_pct`, `max_cv_pct`). That is a protocol; the CV filter above was not.

Re-measuring the two worst cells under this rule, during a window when the host
load average had fallen from ~100 to ~16:

| Cell | Original | Re-measured | Ratio |
|---|---|---|---|
| Qwen3.5-4B Q4_K_M, depth 0 | 45.7 tok/s (CV 14.9 %) | **99.8 tok/s (CV 2.8 %)** | **2.18×** |
| Qwen3.5-4B Q8_0, depth 0 | 52.5 tok/s (CV 8.8 %) | **76.0 tok/s (CV 0.8 %)** | 1.45× |

Against byte-scaling from the reference model, Qwen3.5-4B Q4_K_M now sits at
0.86 of predicted — squarely inside the band occupied by every other dense model
— where the contaminated measurement put it at 0.40. **The architectural anomaly
that motivated §5.8 does not exist.**

Effect on the model, from re-measuring two cells out of fifty-seven:

| | before | after |
|---|---|---|
| train MAPE | 10.7 % | **9.3 %** |
| train max APE | 91.9 % | **39.6 %** |
| test MAPE | 19.3 % | **17.8 %** |
| test p90 | 25.9 % | 22.2 % |

The worst-case error more than halves. Train error of 9.3 % is comparable to the
figures reported by datacenter-targeted models, with the caveat above that the
measurement environments differ substantially.

The general lesson is the one this paper keeps rediscovering: on a consumer host,
**a surprising result is more likely to be a disturbed measurement than a
discovered mechanism**, and the instrument usually says so if asked. The cost of
asking is one extra field in the output.


### 5.6 The prefill regime

The paper states a two-regime model, so both regimes have to be scored. Prefill
was measured throughout — 57 usable measurements across 19 models — and is
evaluated here against `pp_tok/s = η_p · FLOPS / (2 · n_active_params)`, with
η_p fitted on train models only, exactly as for decode.

| Fit | split | n | MAPE | median APE | p90 | max |
|---|---|---|---|---|---|---|
| P1 — one η_p per host | train | 45 | 22.9 % | 18.1 % | 54.5 % | 83.4 % |
| P1 — one η_p per host | test | 12 | 24.6 % | 16.5 % | 54.0 % | 96.1 % |
| **P2 — η_p per host × quantisation** | train | 45 | **18.7 %** | 15.3 % | 45.3 % | 83.4 % |
| **P2 — η_p per host × quantisation** | test | 12 | **18.8 %** | 12.7 % | 35.7 % | 70.7 % |

Two results.

**Prefill is predictable from active parameters.** Measured prefill rate spans
21× across this model set (0.35–7.45× the median), and the model predicts it to
22.9 % out of sample. That is worse than decode's 19.3 % but the same order, and
it means the sparsity term earns its place in both regimes rather than only in
the one it was introduced for.

**Prefill is not quantisation-independent, contrary to the textbook split.**
Giving η_p a separate value per quantisation format cuts train error from 28.9 %
to 21.0 % and the test maximum from 85.7 % to 47.4 %. If prefill were purely
compute-bound after unpacking, format would not matter and P1 would suffice. It
does not. The practical consequence is that a two-regime model needs a
format-dependent coefficient in *both* regimes, not just in decode.

**Caveat that must travel with these numbers.** Without a device-side FLOPS
measurement, the compute term falls back to a CPU fp32 matmul, so FLOPS is not
this machine's GPU throughput and η_p is a fitted constant absorbing that scale
error rather than a literal efficiency. What is tested here is the *shape* of
the model — that prefill rate goes as 1/active-parameters — not the absolute
value of η_p. Installing a device-side benchmark is listed in §5.7 as
outstanding.

### 5.7 Context decay

Partially measured. Decode throughput falls monotonically with KV depth for
three of the four configurations, as the `kv_bytes(c)` term predicts. The
exception is Qwen3.5-4B Q8_0, which measured 64.0 → 67.4 → 63.7 tok/s across
depths 0 → 4096 → 16384: a non-monotonicity of ~5 %, only marginally above the
3.2 % detection threshold. Full depth sweep across the complete model set is
PENDING before drawing any conclusion from it.

### 5.8 Open problem: η varies across models — but the effect is not yet identifiable

**Retracted pending re-measurement.** An earlier draft of this section reported
a 34 % η gap between SmolLM3-3B and Qwen3.5-4B at Q8_0 as an architectural
effect. It is not established, because the measurements behind it were
contaminated by a harness bug.

The resume key in the sweep runner was built from the requested `(n_prompt,
n_gen)` pair, but llama-bench writes prefill and decode as separate rows —
prefill with `n_gen = 0`, decode with `n_prompt = 0`. The key therefore matched
no written row, resume never fired, and every campaign iteration silently
re-measured the whole grid. The re-measurements landed in a differently-loaded
session with cell durations of 1144–2909 s against 96–115 s originally, and the
two sessions disagree by **2.37–3.04×** on identical cells. The largest
between-model η spread is 3.03×. Session variation therefore equals between-model
variation, and nothing architectural is identifiable in that data.

The diagnostic tell: every model measured in the fast session has η ≥ 0.98;
every model measured only in the degraded session has η ≤ 0.88. **η tracks the
session, not the architecture.** The reported 34 % gap sits inside a 200 %
artefact.

The bug is fixed and now has a regression self-check (`python -m llmperf.sweep
--selfcheck`); the affected CSVs are archived under `results/contaminated/`.
Everything below is the hypothesis to be tested once the full set has been
re-measured in a single controlled session. **No claim in this section should
be cited until then.**

This is also the paper's own methodology turned on itself: §5.1 establishes a
3.2 % detection threshold, and a 200 % artefact went unnoticed for hours
regardless. The lesson worth reporting is that a repeatability protocol is
necessary but not sufficient — the harness's own bookkeeping needs a
correctness check, because a silent re-measurement is indistinguishable from a
real effect in the output.

One mechanism is implemented and now measured: the token embedding table is
gathered, not streamed, so for models with an untied output head its bytes
should be excluded from bytes-per-token (`ModelMeta.streamed_bytes`). Sixteen of
the twenty-one models are untied, so this is not a corner case. **Ablating it
changes almost nothing** — test MAPE 19.9 % with the correction against 20.6 %
without, train 11.3 % against 11.2 %, and the test median moves the other way.

The correction is theoretically right and empirically indistinguishable at this
sample size. We keep it because the mechanism is not in doubt, and report the
ablation because a term that cannot be shown to earn its place should not be
presented as if it had.

Notably, Qwen3.5-4B — the largest remaining outlier — is *tied*, with
`token_embd` at 19.1 % of the file, while Qwen3.5-9B in the same family is
untied and predicts well. Whether that is the cause or a coincidence is
unresolved.

The leading remaining hypothesis is the **output projection**. Vocabularies in
this cohort are large and vary widely — Qwen3.5-4B 248,320, gemma-4-12B 262,144,
gpt-oss-20b 201,088, Ling-mini-2.0 157,184, Nemotron-4B 131,072, SmolLM3
128,256 — and the logit computation is a `vocab_size × d_model` matrix-vector
product performed once per token. That work is depth-invariant and plausibly
compute-bound rather than bandwidth-bound, so it should penalise
large-vocabulary and shallow models specifically. This predicts a correction
term `2 · vocab_size · d_model / (η_o · FLOPS)` added to per-token time.

**Definitively rejected on the clean-protocol dataset.** This hypothesis has
now been tested three times, on three datasets of increasing measurement
quality, and the verdict moved each time. That history is worth recording
because it is the substance of the result:

| Dataset | Two-term verdict |
|---|---|
| Contaminated, 12 models | No benefit; unphysical fitted parameters |
| Corrected metadata, 21 models | Appeared to help: train 13.5→9.0 %, test 19.9→17.3 % |
| **Clean protocol, 19 models** | **Overfits: train 12.6→10.9 %, test 13.7→32.0 %** |

On measurements taken under the load gate and CV retry, adding the
output-projection term improves training error and **more than doubles
out-of-sample error** — 13.7 % to 32.0 % under absolute-time loss, 14.1 % to
18.4 % under relative-time loss. The fitted η spread within the largest group
goes from 1.97× to **1036×**, and one model's bandwidth term is driven negative
and drops out of the fit entirely.

That is overfitting, unambiguously. The middle row of the table was itself an
artefact: the apparent improvement came from a free parameter absorbing
measurement noise, and it vanished as soon as the noise did. **A correction that
only helps on noisy data is not a correction.**

**The one-term bandwidth model is the model.** Under the clean protocol it
achieves train 12.6 % and test 13.7 % by least squares on per-token time, or
train 10.3 % and test 10.9 % using the per-group median-ratio fit of §5.5 —
which is both simpler and better out of sample, and is what the paper reports.

**η is stable enough for a predictor.** The one-term spread across the eight
Q4_K_M models in the largest group is 1.97×, and 1.06× across the two Q4_K
models. The instability that made this section an existential question was
measurement quality, not architecture: §5.5 shows the same for the individual
outliers, and §5.1 explains why.

What remains, in order:

1. **A second host.** Leave-one-machine-out is still undefined; the code reports
   this and degrades gracefully rather than inventing a number. Nothing about
   cross-hardware generalisation can be claimed until this exists, and it is now
   the only blocking item.
2. **A real device-side FLOPS calibration.** With torch absent, `compute_for()`
   falls back to a CPU fp32 matmul, so only the product η_o × FLOPS is
   identified and η_o is not an efficiency. This must be fixed before the
   two-term result can be interpreted physically rather than empirically.
3. **More models per group** to separate the output-projection mechanism from
   fitted scatter. The largest group now holds seven models; the earlier power
   analysis suggested 8–11 for an effect of the plausible size, so this is close
   but not yet sufficient.

Step 1 was the designated go/no-go. On the evidence above the project passes
the broader viability test: η is stable enough within a quantisation group for
the calibrated one-term predictor to generalise out of sample. It does **not**
pass the test for the output-projection extension, whose lower training error
does not survive held-out evaluation. The surviving claim is therefore the
one-term predictor; the proposed empirical correction is rejected.

### 5.9 The offload cliff

PENDING — requires the ThinkStation.

### 5.10 Cross-machine generalisation

PENDING — requires all three machines. The strongest available validation is
leave-one-machine-out: fit on two, predict the third.

## 6. Threats to validity

- **The primary measurement host is not fit for purpose, and this is now a hard
  blocker rather than a caveat.** The MacBook used for every measurement in this
  paper is a corporate-managed machine whose one-minute load average was
  observed between 12 and 396 over a single day, driven entirely by
  endpoint-monitoring and log-shipping agents with no user process involved. An
  attempt to re-measure the full model set under the quality protocol of §5.5
  completed **zero of twenty-three cells**: the load gate waited its full
  thirty-minute budget on the first cell and never saw a quiet window. At that
  rate the sweep would spend eleven hours waiting and flag every cell as
  provisional.

  The consequence for this paper is concrete: the measurements reported here
  were taken opportunistically during quiet windows that happened to occur, not
  under a protocol that can be relied on to reproduce them. Six rows in the
  current dataset are recorded as taken above the load gate. Every headline
  number should be read as provisional until the set is re-measured on a
  quiescent host.

  This is not a mitigation we can apply in software. It requires a different
  machine.

- **Three machines is a small hardware sample.** Cross-hardware generalisation
  claims must be stated as such. Mitigation: rent two or three cloud GPU
  instances (~US$50) to add independent hardware points; this is the single
  cheapest way to strengthen the claim, and a rented instance has the further
  advantage of not carrying a corporate monitoring stack.
- **One runtime.** llama.cpp only, chosen because it is the sole runtime common
  to all three machines. Conclusions are about llama.cpp, not about LLM
  inference in general, and must be worded that way.
- **Quantisation formats are not equivalent across backends.** MXFP4 on
  gpt-oss and K-quants elsewhere are different objects; η is fitted per format
  for exactly this reason.
- **η is fitted, not derived.** If η proves unstable across models within a
  (machine, format) group, the predictive claim fails. This is the project's
  designated go/no-go check.
- **Model set churn.** The 2026 cohort will itself age. The defence is that the
  contribution is the predictor, not the table — and the released harness
  regenerates the table for any future cohort.

## 7. Reproducibility

Harness, calibration outputs, raw per-cell CSVs, and figure generation are
released. `python -m llmperf.common` runs the unit self-check for the KV-cache
and quantisation-parsing logic. See `README.md`.
