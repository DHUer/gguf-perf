# Handoff

Everything an agent needs to pick this project up cold. Read this, then
`README.md` (how to run) and `paper/paper.md` (findings). This file holds only
what is *not* already in those — decisions, retractions, and running state.

Last updated: 2026-08-30.

---

## What this is

A first paper for an author with **no prior LLM/ML research experience**,
targeting **CCF-C minimum, CCF-B preferred**. That constraint is load-bearing:
no model training, no framework surgery, no custom kernels. The contribution
must come from experimental design and analysis. Everything runs off
`llama-bench`; not one line of model code is written.

**Paper:** a calibrated two-regime performance model for llama.cpp inference on
consumer PCs — decode memory-bound, prefill compute-bound — whose per-model
inputs are only a GGUF file size and metadata fields, validated out-of-sample
on models never used for fitting. Chosen over a benchmark-table paper because a
predictor stays useful for models released after publication.

**Target venues:** empirical-SE journals (EMSE, IST, JSS — CCF-B), *not* EMNLP.
Journal-first for a first-time author: major-revision cycles instead of binary
reject. CCF-C fallback should be systems/HPC journals (Journal of
Supercomputing, CCPE), **not** Neurocomputing/KBS/ESWA — those are ML-methods
journals and a pure performance study risks desk rejection on scope.

## Hardware

| Machine | Compute | Memory | Status |
|---|---|---|---|
| MacBook Pro M4 Max | Metal | 64 GB unified | **all work so far is from here** |
| Mac Studio M4 Max 40C | Metal | 128 GB unified | **not yet run** |
| Lenovo ThinkStation P8 + RTX 5080 | CUDA sm_120 | 16 GB VRAM + 128 GB DDR5 | **not yet run** |

The two Macs share a chip generation and differ mainly in thermal envelope and
capacity, which isolates throttling from architecture. The ThinkStation's
128 GB host RAM is what makes the offload cliff fully traceable.

## Where it stands

The stage modules have been validated end to end on the MacBook:
`doctor → fetch → calibrate → sweep → analyze`, plus `repeatability` and
`refine`. The new cross-platform `campaign` driver is the intended entry point,
but it has not yet completed a clean run because legacy shell drivers are still
active; see "Currently running" below.

14 of 23 models are complete (105.1 GB; the full manifest is 336.4 GB).
The clean CSV has 84 rows from 14 files. Analysis excludes the calibration
probe, leaving 39 decode measurements from 13 models, including 2 MoE, on
**one host**.

### Results that hold

- **Noise floor (§5.1).** Between-run CV is **1.61 %** idle with 45 s settling,
  versus **15–26 %** with background load. Prefill is far more stable
  (CV 0.37 %). Detection threshold **3.2 % (2σ)**. No prior work establishes a
  repeatability figure for consumer LLM benchmarking — this is the strongest
  unclaimed result in the paper and is now contribution #1.
- **MoE sparsity ablation (§5.3).** Out-of-sample MAPE **70.9 % → 31.0 %** when
  bytes-per-token uses activated rather than total parameters. The physical
  argument is sharper than the metric: back-solving gpt-oss-20b from total size
  gives 1441 GB/s, several times the machine's achievable bandwidth — not merely
  inaccurate but impossible. Activated parameters give 245 GB/s.
- **Quantisation is not free bandwidth (§5.2).** K-quant dequantisation costs
  ≈20 % of achievable bandwidth. *Direction already published* (Benazir & Lin,
  SIGMETRICS 2026); we contribute the magnitude, and must cite them.
- **Calibration trap (§3.3).** The last clean calibration cited in the
  manuscript reads 67.9 GB/s from the CPU triad while the GPU-backed reference
  sustains 313.8 GB/s through the same unified memory — 4.6×. Framed as a
  warning, not a discovery; correct prior practice already avoids it.

### Results that were retracted

**Everything measured before 2026-08-30 12:30 is contaminated.** `sweep.py`'s
resume key was built from the requested `(n_prompt, n_gen)` pair, but
llama-bench writes prefill with `n_gen=0` and decode with `n_prompt=0`, so the
key matched no written row, resume never fired, and every campaign iteration
silently re-measured the whole grid into a differently-loaded session. Identical
cells disagreed by **2.37–3.04×**; the largest between-model η spread is 3.03×.
η tracked the session, not the architecture.

Fixed, with a regression guard: `python -m llmperf.sweep --selfcheck`.
Contaminated CSVs are archived under `results/contaminated/` — keep them, they
are the evidence for the methodology point that a repeatability protocol is
necessary but not sufficient.

An earlier "10.4 % MAPE" and a "34 % architectural η gap" both came from this
data. Neither is real. Do not cite them.

### The go/no-go (§5.7)

**η is not stable across models within a quantisation format.** If it cannot be
predicted from GGUF metadata, the model degrades from a predictor into a
lookup table and the central claim fails.

The leading hypothesis — a compute-bound output-projection term
`2·vocab_size·d_model / (η_o·FLOPS)` — was tested and **is not supported**.
Fitting it lowers MAPE but produces unphysical parameters (η_o = 0.0195 makes
the output term 23–215 % of per-token time; η_d reaches 5.00, i.e. 500 % of the
roofline) and the cross-model η spread does not shrink (Q8_0 gets *worse*,
2.12 → 9.79). Fit-free Spearman ρ = −0.40 (p = 0.60) and −0.50 (p = 0.67): right
sign, nowhere near significant. The MAPE gain is one free parameter absorbing
scatter.

To settle it: 8–11 models per (host, quant) group (currently 3–4), output-load
leverage ≥5× within a group (Q8_0 currently spans 1.7× and cannot resolve
anything at any sample size), real device-side FLOPS calibration, and a second
host.

Also unresolved: **η exceeds 1 for several formats** (Q4_0 2.07, Q8_0 1.21),
which is unphysical. The calibration anchor is a single reference model that is
evidently not the machine's ceiling.

## Literature position

`paper/related_work.md` has the full analysis, per-citation verification status,
and 10 concrete framing fixes. Summary:

**Not scooped, but the original framing was wrong.** Closest prior work is
**RooflineBench (arXiv:2602.11506)** — same roofline framing, consumer hardware
including Apple Silicon, llama.cpp via `llama-bench`, microbenchmark
calibration. It fits no efficiency coefficient, predicts nothing, holds out no
models, and **explicitly defers MoE**. That is the gap this paper occupies.

Three things that must be respected:
1. **η already has a name**: Model Bandwidth Utilization (MBU); the MoE form is
   S-MBU (MoE-CAP). Use the existing terms, do not coin new ones.
2. **The accuracy bar is 7–9 %**, not 10 %. LIMINAL 7.6 % MAE; NeuSight 8.7 % on
   *unseen GPUs*. Current 31 % is far off.
3. **Open lanes with no competitor found**: the noise floor, and any analytical
   model of the partial-offload (`n_gpu_layers`) cliff.

Four citations are flagged do-not-cite pending verification; two need specific
checks. See the "Citations to verify" list. **Verify every reference before
submission** — they were gathered by an agent under a search budget.

## Currently running

**Incident RESOLVED 2026-08-30.** Two `run_campaign.sh` processes (pids 17685
and 44845) both appended to `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf.part`, growing it
to 19.0 GB against an expected 16.9 GB. Nothing corrupt reached the dataset: the
size check in `http_download` refuses to promote a `.part` whose length does not
match the manifest, so the file was never benchmarked. The 14 completed models
are intact.

Root cause was mine, and it was not the lock failing — it was the lock being
bypassed. Two drivers existed (`run_campaign.sh` with a lock directory,
`llmperf.campaign` with a lock file) so they did not exclude each other, and I
then `rm -rf`'d a lock that a surviving process still held.

Three fixes, all in place:

1. `run_campaign.sh` is **deleted**. One driver, one lock.
2. `http_download` now discards a `.part` at or beyond the expected size instead
   of resuming from it, so a doubled partial cannot loop forever.
3. `llmperf.campaign`'s lock is a heartbeat file, not a pid liveness probe.
   `os.kill(pid, 0)` is not portable — on Windows it calls TerminateProcess and
   would kill the process it is checking for.

**Never delete `results/campaign.lock.json` to start a second campaign.** If it
looks stale, it expires by itself after five minutes of missed heartbeats. A
background thread refreshes it continuously and SIGTERM/SIGINT release it, so a
lock that is genuinely present means a campaign is genuinely running.

### Run the campaign detached, not as a tracked background task

The campaign takes hours. An agent-harness background task does not survive
that — two attempts here were killed at roughly 40 minutes. Start it detached:

```bash
nohup python -m llmperf.campaign > results/campaign.nohup.log 2>&1 < /dev/null &
```

The tradeoff is real and worth stating: a detached job survives but sends no
completion callback, so progress has to be read from `results/campaign.log` and
`results/STATUS.md`. There is no configuration that gives both.

Being killed costs nothing but time. Every stage is resumable — across the two
kills above the model count still advanced from 14 to 20, no partial file was
corrupted, and the lock released cleanly each time.

## Next, in priority order

1. **Recover from the duplicate legacy downloads described above.** The shared
   Gemma partial is oversized and untrustworthy. Stop both process trees,
   remove that one `.part`, regenerate calibration and analysis while idle, and
   resume with `python -m llmperf.campaign` only.
2. **Run the Mac Studio and the ThinkStation.** This is the main scientific
   blocker.
   One host means leave-one-machine-out is undefined, no cross-architecture
   claim is admissible, and the MoE capacity/bandwidth story — the paper's
   differentiator against RooflineBench — is entirely untested. Follow
   `README.md`; run `python -m llmperf.doctor` first.
3. **Offload-cliff sweep on the ThinkStation**: `--ngl 0 8 16 24 32 48 99`.
   Open ground; only the 16 GB-VRAM + 128 GB-RAM machine can produce it.
4. **Install torch cu128 on the ThinkStation** for real device FLOPS/bandwidth.
   Without it `compute_for()` falls back to a CPU fp32 matmul and η_o is not an
   efficiency.
5. **Fix the calibration anchor** so η ≤ 1 (device-side measurement, or
   best-of-N across reference models).
6. **Re-select remaining models for vocabulary × depth spread**, not parameter
   count. Otherwise §5.7 stays statistically unresolvable even at 22 models.
7. **Verify the flagged citations.**

**Do not pick the paper's headline yet.** The predictor line is at 31 % against
a 7–9 % bar, but the MoE ablation direction is clearly right. The noise-floor
line has solid evidence and no competitor. Cross-machine data decides which is
viable; choosing before it arrives is guesswork.

## Traps already paid for

- **`huggingface_hub` stalls at zero bytes.** Its Xet path hangs on
  unauthenticated connections while a plain ranged GET to the same CDN runs at
  full rate. `fetch.py` uses stdlib urllib with Range resume and retries. Set
  `HF_TOKEN` anyway to lift the rate limit.
- **Never run two downloaders against one models directory.** They both append
  to the same `.part`; each process tracks only its own byte count, so a shared
  file can exceed the expected size and still be renamed by one process. The
  shell and Python campaign drivers use different lock files. Use only
  `python -m llmperf.campaign` for new runs.
- **`llama-bench -fa` defaults to `auto`** and silently picks a different
  attention kernel per model and backend. Pinned to `on`. Same reasoning for
  `-ctk`/`-ctv` and `-t`.
- **Calibrating with a model that is also evaluated is circular** — its η is 1.0
  by construction. `analyze.py` excludes the probe from all error metrics.
- **GGUF repos contain non-LM files.** `mmproj-*` (vision projector) and `mtp-*`
  (multi-token-prediction head) sit beside the weights; a shortest-filename
  heuristic picks them and benchmarks the wrong thing.
- **Token embeddings are gathered, not streamed.** For untied output heads their
  bytes must be excluded from bytes-per-token. Implemented in
  `ModelMeta.streamed_bytes`; does not apply to tied-embedding models.
- **Community finetunes are excluded on purpose** ("uncensored", "abliterated",
  "heretic", merges). They change the weights and would confound the study.
- **Agents told to "verify via web search" without a budget hang indefinitely.**
  A 21-agent workflow burned 4 hours and 945k tokens for zero output this way.
  Cap it: "at most N searches, then answer with what you have."

## Layout

```
llmperf/common.py         paths, llama-bench discovery, GGUF metadata, KV maths
llmperf/fetch.py          download + fixed train/test split manifest
llmperf/calibrate.py      hardware microbenchmarks (three bandwidth sources)
llmperf/doctor.py         environment, backend, dependency, disk, self-checks
llmperf/campaign.py       canonical cross-platform unattended driver
llmperf/repeatability.py  noise floor — run before trusting any effect
llmperf/sweep.py          drives llama-bench, append-only resumable CSV
llmperf/analyze.py        fit, out-of-sample validation, figures
llmperf/refine.py         two-term model, LOMO, eta-spread diagnostics
llmperf/doctor.py         environment check — run this first on a new machine
llmperf/campaign.py       the driver: fetch → sweep idle → analyze → repeat
llmperf/figures.py        publication figures (PDF + PNG, print-safe)
paper/paper.md            manuscript
paper/related_work.md     literature + novelty assessment + framing fixes
results/                  per-machine CSV + calibration JSON (commit these)
results/contaminated/     archived bad data, kept as evidence
```

Start a new machine with `python -m llmperf.doctor`. Fast self-checks are
`python -m llmperf.doctor --quick`, `python -m llmperf.common`,
`python -m llmperf.sweep --selfcheck`, and
`python -m llmperf.refine --selfcheck`.

## House rules

- **Never fabricate a number, citation, or result.** The paper marks unmeasured
  cells PENDING rather than estimating them. Keep it that way — the author acts
  on what is written.
- Report negative results plainly. The output-projection hypothesis failing is a
  finding, not an embarrassment.
- Reuse `llama-bench`; do not write a replacement timing harness.
