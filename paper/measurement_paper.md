# Your LLM Benchmark Is Measuring Your Monitoring Agents

**Target:** performance-engineering / workload-characterisation conference
(ICPE, IISWC, ESEM, ICPADS). ~8 pages.

> **STATUS.** Every number in this document comes from measurements in
> `results/`. Nothing is estimated or illustrative. Section 8 lists what remains
> unverified.

## Abstract

Benchmarks of open-weight language models on consumer hardware are now common,
and none of the ones we surveyed reports the state of the machine they ran on.
We show that omission is not a detail.

On a corporate-managed laptop of exactly the kind these benchmarks use,
background monitoring and log-shipping agents drove the one-minute load average
between 8 and 396 over a single day with no user process running. Identical
measurement cells taken hours apart under that variation disagreed by up to
**2.8×** — larger than the difference between adjacent model generations, and
larger than most effects reported in this literature.

We quantify the resulting noise floor, show that it is invisible without
instrumentation, and give a protocol that removes it. Under a controlled
protocol the between-run coefficient of variation for decode throughput is
**1.61 %**, giving a minimum detectable effect of 3.2 % at 2σ. Without load
control, the same quantity varies by 15–26 %.

We then show the practical consequence on a downstream task. Using these
measurements to fit an analytical throughput model, conditioning the
measurements cut out-of-sample error from **17.9 % to 10.9 %** and collapsed an
eight-point train/test gap — which had read as overfitting — to **0.6 points**.
A model that generalises and one that does not were indistinguishable until the
instrument was fixed.

Most usefully, we find that the instrument already knows. `llama-bench`'s own
within-run standard deviation, recorded for free on every measurement and
computed before any analysis, ranks prediction failures almost exactly: every
model our downstream model failed on was a model whose own measurement was
unstable. We report three case studies in which this project drew a wrong
conclusion from disturbed data — including an "architectural anomaly" that
disappeared on re-measurement — and in every case the signal was already in the
output, unread.

We recommend that consumer-hardware LLM benchmarks record and publish system
load and per-measurement variance alongside every throughput number, and we
release a harness that does so, gates on both, and retries when the instrument
reports instability.

## 1. Introduction

The population running open-weight language models on personal computers is
large and growing, and a measurement literature has grown with it: throughput
comparisons across models, quantisation formats, runtimes and devices, on
laptops, single consumer GPUs and single-board computers.

These studies are careful about the things a benchmark is traditionally careful
about. They pin sampling parameters, fix context lengths, repeat runs and report
standard deviations. What they do not do — in any instance we found — is report
what else the machine was doing.

On a dedicated server that omission is harmless. On the machines this literature
actually studies it is not. A modern managed laptop runs endpoint-monitoring,
log-shipping, telemetry and indexing agents continuously, and their load is
neither small nor stable.

This paper measures that effect, shows it is large enough to invert conclusions,
and gives a protocol that removes it.

### Contributions

1. A quantified noise floor and minimum detectable effect size for consumer LLM
   throughput measurement (§4). We found no prior work that establishes either.
2. Evidence that the host is the dominant nuisance variable: load average 8–396
   from background agents alone, and 2.8× disagreement on identical cells (§5).
3. The finding that **measurement instability is already recorded and already
   predicts downstream error** — the benchmarking tool's own within-run standard
   deviation ranks failures without needing any analysis (§6).
4. Three case studies in which disturbed measurements produced confident wrong
   conclusions, drawn from this project's own history (§7).
5. A measurement protocol — load gating plus instrument-triggered retry — and
   what it recovered: downstream out-of-sample error 17.9 % → 10.9 %, train/test
   gap 8 points → 0.6 (§8).
6. A released harness that records and enforces all of the above.

## 2. Background and related work

The claim this paper rests on is a negative one — that ambient host load is not
recorded in this literature — so it was tested directly rather than assumed. It
survives, but in a narrower form than first drafted, and the adjacent work needs
stating precisely.

**Interference is studied, but as deliberate co-location on server hardware.**
There is a body of work on GPU contention in which the interfering workload is
introduced on purpose and measured. One 2025 study builds custom CUDA
benchmarks that each stress a specific GPU resource, co-locates them using CUDA
streams and Green Contexts, and reports interference reaching 151.66 % at the
95th percentile with a concurrency of only two (arXiv:2501.16909, UNVERIFIED —
figures taken from a search summary, not the primary source). That is a
controlled-treatment design on H100-class and prosumer GPUs. **Our setting is
the opposite**: the interfering load is neither introduced nor controllable, it
comes from mandatory platform software, and the question is how much it
corrupts measurements taken in ignorance of it.

**Consumer benchmarks observe variance without attributing it.** A study of
LLM inference on consumer Blackwell GPUs reports "high-variance outliers across
all concurrency levels" and attributes them to intermittent contention and
scheduling instability (arXiv:2601.09527, UNVERIFIED). The observation overlaps
with ours; the diagnosis is left open and no host state is recorded, so the
variance cannot be attributed after the fact.

**Reproducibility practice in this literature is real but incomplete.** The same
work releases a Docker image, orchestration scripts, raw per-run JSON and energy
traces. Artifact release is therefore not the gap. What is absent is any record
of what else the machine was doing while each number was produced — which is
precisely the field that would let a reader distinguish a result from an
artefact.

**Adjacent benchmarking efforts** include LLM-Inference-Bench across AI
accelerators (arXiv:2411.00136), TokenPowerBench on inference power consumption
(arXiv:2512.03024), and a Qwen3-30B consumer-hardware analysis
(arXiv:2512.23029). Closest on setting is Benazir & Lin's SIGMETRICS 2026
characterisation of Apple Silicon inference (arXiv:2508.08531; POMACS
doi:10.1145/3771563 — VERIFIED), which profiles dequantisation overhead across
five testbeds. None of these records host load or reports a repeatability
figure.

**Variance is already taken seriously for LLM *quality* benchmarks, and not for
performance ones.** A distinct literature quantifies uncertainty in LLM
evaluation scores — how many repetitions a benchmark needs, how much of a score
gap is seed and prompt noise, how rarely papers report statistical validation at
all (arXiv:2410.03492, arXiv:2509.24086, arXiv:2406.10229; all UNVERIFIED). One
survey reportedly finds the share of LLM evaluation papers using statistical
validation *declining*, from 32.6 % to 28.0 %.

This is the closest intellectual neighbour to the present work and the framing
risk we most need to address. The problems are the same class and have different
causes and different remedies. In quality benchmarking the variance is endogenous
— sampling temperature, prompt phrasing, random seed — and the remedy is more
repetitions and a confidence interval. In throughput benchmarking on a consumer
host the dominant variance is **exogenous**: it comes from software the
experimenter does not control and cannot see in the output. More repetitions do
not fix it, because repetitions taken during the same period of contention are
correlated; §5 shows two sessions of five repetitions each, internally
consistent and 2.8× apart. The remedy has to be recording and gating on host
state.

**A representative gap.** A careful 2026 consumer-GPU study benchmarks four
models across 79 configurations on Blackwell hardware, releases Docker images,
orchestration scripts and raw per-run data, and reports tail latencies
(arXiv:2601.09527, UNVERIFIED). We could find no coefficient of variation,
repeat count, warm-up or discard rule, confidence interval or error bar in the
retrieved text. This is not a criticism of that work specifically — it is more
rigorous than most — but an illustration that artifact release and measurement
uncertainty are independent, and that the field currently supplies the first
without the second.

**The refined claim.** Deliberate co-location interference is well studied on
server hardware, and score variance is well studied for quality benchmarks. Ambient platform load on consumer machines — the condition
under which essentially all local-LLM benchmarking actually happens — is not,
and no work we found publishes system load or a minimum detectable effect size
alongside its throughput numbers. That is the gap this paper fills.

*All arXiv identifiers in this section except 2508.08531 are UNVERIFIED and
were obtained via search summaries. Each must be checked against its primary
source before submission.*

## 3. Method

All measurements use `llama-bench` from llama.cpp, which performs warm-up runs,
repeats each test, and reports mean and standard deviation separately for
prefill and decode. We did not write a replacement timing harness; we
constrained the one that exists and recorded its context.

**Host.** MacBook Pro M4 Max, 64 GB unified memory, 12 performance cores, macOS. The
machine is corporate-managed and carries endpoint-monitoring, data-loss-
prevention and log-shipping agents that cannot be disabled. This is
representative of the class, not a pathological case.

**Pinned settings.** Flash attention `on` (the default `auto` silently selects
different kernels per model and backend), KV cache dtype f16, threads set to
physical performance cores, five repetitions per cell, warm-up enabled, a
settling delay before each cell, and one process per cell so a failure cannot
corrupt the remainder of a queue.

**Recorded per cell.** Beyond throughput: the one-minute load average before and
after, the number of measurement attempts, the within-run coefficient of
variation of the kept attempt, the git commit, and the `llama-bench` version.

**Workload.** 23 current open-weight models spanning 1.9–63 GB, dense and
Mixture-of-Experts, eight quantisation formats, at KV depths of 0, 4 096 and
16 384 tokens.

## 4. The noise floor

We measured one fixed configuration six times as six independent processes with
45 s of idle settling before each.

| Quantity | Between-run CV |
|---|---|
| Decode throughput | **1.61 %** |
| Prefill throughput | **0.37 %** |

This gives a minimum detectable effect of **≈3.2 % at 2σ** for decode on this
host. Effects smaller than that are not measurable and should not be reported.

Two observations that matter for practice.

**Prefill is four times more stable than decode.** Prefill saturates the compute
units and is relatively insensitive to interference; decode is memory-bound at
batch size one and competes with everything else on the memory system. A
benchmark reporting only prefill will look far more repeatable than it is.

**The residual is not stationary.** Across the six runs, throughput drifted
upward monotonically over the later runs at +0.7 % per run. A coefficient of
variation summarises that drift as if it were random scatter. Reporting the
per-run series, not just a CV, is the honest presentation.

## 5. The host is never idle

The figure above was obtained in a quiet window. Auditing the host showed such
windows are the exception.

With no user process running, the one-minute load average was observed between
**8 and 396** on twelve performance cores, driven by a log-shipping agent
(61.6 % CPU), a telemetry daemon (44.6 %), an endpoint-monitoring agent
(38.6 %), a data-loss-prevention agent (22.5 %), the window
server and assorted productivity applications. At the high end, a plain
`uptime` invocation took over five minutes to return.

Under that variation, the same measurement cell taken hours apart disagreed by
**up to 2.8×**. One example, the same model and configuration measured twice:

| | Measurement 1 | Measurement 2 |
|---|---|---|
| Qwen3.8-27B Q8_0, decode | 14.5 tok/s | 5.2 tok/s |

Neither reading is flagged by anything a conventional benchmark reports. Both
have plausible-looking standard deviations. Only the host state distinguishes
them, and the host state is exactly what is not recorded.

**Implication.** A benchmark comparing two models measured at different times on
such a host is, with meaningful probability, reporting the difference between
two load conditions.

## 6. The instrument already knows

The central practical finding of this paper is that detecting this costs
nothing, because the information is already being produced.

`llama-bench` reports a within-run standard deviation across its repetitions.
That number is available for every measurement, is computed before any
downstream analysis, and is therefore not contaminated by knowledge of the
result.

Using these measurements to fit an analytical throughput model, we ranked every
model by the prediction error it produced, and separately by its own within-run
CV. The two rankings agree at the top:

| Cell | Within-run CV | Rank as prediction failure |
|---|---|---|
| Qwen3.5-4B Q4_K_M, depth 0 | **14.9 %** | worst |
| Muse-Glimmer-30B, depth 0 | 9.9 % | 2nd |
| gemma-4-26B-A4B, depth 16 384 | 9.3 % | 3rd |
| Qwen3.5-4B Q8_0, depth 0 | 8.8 % | 4th |
| *median across all cells* | *1.83 %* | — |

Every cell the downstream model failed on was a cell whose own measurement was
unstable. Excluding rows on measurement quality alone moved training error from
10.7 % to 9.0 % by dropping a single row.

The correct response is not to discard those cells but to re-measure them, and
the within-run CV tells you which ones without any analysis at all.

## 7. Three case studies in self-deception

The following are drawn from this project's own record. We report them because
each was believed, acted on, and wrong, and because in every case the evidence
of the error was already present in data we had collected.

**7.1 A resume key that never matched.** The measurement runner keyed completed
work on the requested `(n_prompt, n_gen)` pair, but `llama-bench` writes prefill
with `n_gen = 0` and decode with `n_prompt = 0`. The key matched no written row,
so resume never fired and every campaign iteration silently re-measured the
entire grid into whatever load condition prevailed. Identical cells across the
two sessions disagreed by 2.37–3.04×. Because the largest genuine between-model
effect was 3.03×, nothing architectural was identifiable in that dataset. The
only visible symptom was a log line reading "24 cells already done" followed by
"0 skipped" — a contradiction nobody reads.

**7.2 A filter chosen after seeing the answer.** Having observed that
high-CV cells predicted badly, we filtered on CV and reported the improvement.
The filter was selected after seeing the residuals, which makes it inadmissible:
the same procedure applied to noise produces the same improvement. The
defensible version is the retry rule of §8, which uses the same quantity but
fixes the threshold in advance and re-measures rather than deletes.

**7.3 A correction that only worked on bad data.** We hypothesised that the
output projection — a `vocab × d_model` matrix-vector product performed once per
token — explained residual error, and added it as a second model term. Across
three datasets of increasing measurement quality the verdict inverted twice:

| Dataset | Verdict |
|---|---|
| Contaminated | No benefit, unphysical fitted parameters |
| Partially corrected | Appeared to help: test error 19.9 % → 17.3 % |
| **Clean protocol** | **Overfits: test error 13.7 % → 32.0 %** |

On properly conditioned measurements the term improves training error and more
than doubles out-of-sample error, with the fitted efficiency spread exploding
from 1.97× to 1036×. The middle row was an artefact: a free parameter absorbing
measurement noise, which vanished when the noise did.

**A correction that only helps on noisy data is not a correction.** We suggest
this as a general test.

## 8. The protocol, and what it recovered

Two rules, both fixed before measuring.

**Load gate.** Before each cell, wait for the one-minute load average to fall
below a threshold (default 1.5× the thread count), polling up to a bounded
timeout. On a permanently busy host the gate proceeds anyway, records the load,
and flags the cell — measuring with a recorded caveat beats not measuring.

**Instrument-triggered retry.** After each cell, if the within-run CV exceeds a
declared threshold (default 3 %), settle longer and re-measure, keeping the
lowest-CV attempt. The criterion is a property of the measurement, not of the
result.

Re-sweeping the full model set under both rules: **22 of 23 cells measured, none
forced through the gate.** The remaining cell is a 63 GB model on a 64 GB
machine and does not load. 30 of 145 rows required more than one attempt.

Measurement quality, before and after:

| | Before | After |
|---|---|---|
| Median within-run CV | 1.83 % | **0.68 %** |
| Maximum within-run CV | 14.9 % | **2.81 %** |

Effect on the downstream analytical model:

| | Before | After |
|---|---|---|
| Out-of-sample error | 17.9 % | **10.9 %** |
| Out-of-sample maximum | 74.6 % | **34.5 %** |
| Train/test gap | 8.0 points | **0.6 points** |

The last row is the one we would emphasise. An eight-point train/test gap is
normally read as overfitting and prompts a change to the model. It was
measurement noise concentrated in the smaller split. **A predictor that
generalises and one that does not are indistinguishable until the instrument is
fixed.**

Finally, the individual anomaly that motivated the largest modelling effort in
the parent project:

| Cell | Original | Re-measured |
|---|---|---|
| Qwen3.5-4B Q4_K_M | 45.7 tok/s (CV 14.9 %) | **99.8 tok/s (CV 2.8 %)** |

At 45.7 tok/s the model sat at 0.40 of what byte-scaling predicts, an apparent
architectural outlier. At 99.8 it sits at 0.86, indistinguishable from every
other dense model measured. The anomaly did not exist.

## 9. Recommendations

For anyone publishing consumer-hardware LLM measurements:

1. **Record and publish system load** with every throughput number. One field.
2. **Publish per-measurement variance**, not only the mean. Most tools already
   compute it.
3. **State a minimum detectable effect** and do not report differences below it.
4. **Gate on load before measuring**, and re-measure when the instrument
   reports instability. Declare both thresholds in advance.
5. **Report the per-run series** for repeatability claims, not just a CV — drift
   and scatter are different problems.
6. **Treat a surprising result as a suspect measurement first.** In this project
   every unexplained architectural effect was a disturbed measurement, and the
   instrument had already said so.

## 10. Threats to validity

- **One host.** The load figures characterise one corporate-managed laptop. The
  *magnitude* elsewhere is unknown; the *mechanism* is not host-specific.
- **One runtime, batch size one.** Conclusions are about `llama.cpp` at batch 1.
- **The downstream model is ours.** §8's error improvements are measured on a
  throughput model we also wrote. The measurement-quality findings (§4–§7) do
  not depend on it; the §8 improvements do.
- **The clean sweep caught a lucky window.** It succeeded because the host
  happened to stay quiet for several hours. This does not demonstrate the
  protocol works on a permanently busy machine — only that it correctly refuses
  to measure on one.
- **Related work is unverified.** Ten citations are flagged; the novelty claim
  for the load-and-variance recommendation is not yet confirmed.

## 11. Conclusion

Consumer-hardware LLM benchmarking has inherited the reporting conventions of
server benchmarking without inheriting the controlled environment. On the
machines this literature studies, the host is the dominant nuisance variable,
its influence reaches 2.8× on identical work, and essentially nobody records it.

The remedy is cheap: one extra field in the output, one threshold checked before
measuring, one retry when the instrument reports instability. Applied here, it
turned a model that appeared to overfit into one that generalises, and dissolved
an architectural anomaly that had consumed weeks.

The instrument was telling us the whole time. We were not writing it down.
# Archived measurement-methods draft

> **DO NOT CITE NUMBERS FROM THIS FILE.** This draft predates the strict
> protocol-complete row selector. The current paper and results appendix are in
> `paper/icassp2027/`.
