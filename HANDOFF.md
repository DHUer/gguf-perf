# Handoff

Read this first, then `README.md` (how to run) and `paper/paper.md` (findings).
This file holds what those don't: decisions, retractions, traps, and the job
that's actually next.

Last updated 2026-09-01, on handoff from the macOS host to the Windows
ThinkStation.

---

## Your job, in one paragraph

Everything that can be done on one machine is done. The project is blocked on a
second host. You are on the ThinkStation (RTX 5080, 16 GB VRAM, 128 GB RAM,
Windows). **Produce `results/measurements_<host>.csv` and
`results/calibration_<host>.json` for this machine, plus an offload-cliff sweep.**
That unblocks leave-one-machine-out validation, §5.9 and §5.10 of the paper, and
turns a single-host study into a cross-architecture one. Nothing else is more
valuable; resist polishing prose until the data exists.

## What this is

A first paper for an author with **no prior LLM/ML research background**,
targeting a **conference** (CCF-C realistic, CCF-B a stretch). That constraint
is load-bearing: no model training, no framework surgery. Everything runs off
`llama-bench`; not one line of model code is written.

Public repo: <https://github.com/DHUer/gguf-perf>

**Two papers**, deliberately split:

- **`paper/paper.md`** — the predictor. Calibrated two-regime throughput model,
  inputs are GGUF file size and header metadata only. Blocked on your data.
- **`paper/measurement_paper.md`** — the measurement-methodology paper. Complete
  and submittable *now*; needs no new hardware. Likely the better conference bet.

## Where it stands

**Data: 22 of 23 models measured on one host** (Apple M4 Max, 64 GB), under a
declared quality protocol, all in genuine quiet windows. One model
(gpt-oss-120b, 63 GB) doesn't fit that machine.

**Current headline numbers** (regenerate with `python -m llmperf.analyze`;
do not trust numbers typed in prose without checking):

| | train | test |
|---|---|---|
| B0 uncalibrated | 38.5 % | 47.2 % |
| B1 calibrated, total params | 18.3 % | 49.6 % |
| **B2 calibrated, active params** | **10.3 %** | **10.9 %** |
| Prefill, η_p per host × quant | 18.7 % | 18.8 % |

The train/test gap is 0.6 points. That only appeared after the measurements were
conditioned; before that it was 8 points and read as overfitting.

**Findings that hold:** MoE activated-parameter accounting (out-of-sample MoE
error 81 % → 11 %); per-layer KV summation (corrects up to 13.7× overestimate on
sliding-window models); the noise floor and load-contamination results; the
quantisation ladder.

**Findings rejected, honestly, in the paper:** the output-projection correction
(improves train, doubles test error — rejected); prefill is *not*
quantisation-independent, contrary to the textbook split.

## What you must NOT redo

Four bugs cost this project days each. All are fixed and have regression
guards. If you see behaviour resembling them, check the guard before assuming a
new bug.

1. **Resume key never matched.** Keyed on requested `(n_prompt, n_gen)`, but
   llama-bench writes prefill with `n_gen=0` and decode with `n_prompt=0`. Resume
   silently never fired; every run re-measured everything into a different load
   condition. Guard: `python -m llmperf.sweep --selfcheck`.
2. **MoE expert tensors unmatched for fused architectures.** gemma-4 ships a
   combined `ffn_gate_up_exps`; an enumerated name list missed it and counted
   most expert weight as always-active. Now matches `_exps` generically, and
   warns if a declared-sparse model resolves as >50 % active.
3. **CSV schema drift.** Adding a column made the writer emit more values than
   the on-disk header had, silently corrupting the file. Now migrates on append.
4. **Load gate no-op on Windows** — relevant to you. `os.getloadavg()` is
   Unix-only; NaN compares false against any threshold, so the gate silently
   disabled itself. Fixed via psutil. **Verify `python -m llmperf.doctor` reports
   a real load number before trusting any measurement you take.**

## Traps already paid for

- **`llama-bench -fa` defaults to `auto`** and silently picks different attention
  kernels per model and backend. Pinned to `on`. Same reasoning for `-ctk`/`-ctv`
  and `-t`.
- **Calibrating on a model you also evaluate is circular.** All calibration
  probes are excluded from scoring.
- **The calibration anchor must be a large, high-precision model.** Anchoring on
  the smallest one reports a floor, not a ceiling, and inflated every η by 1.49×.
- **`huggingface_hub` stalls at zero bytes** on unauthenticated connections;
  `fetch.py` uses plain ranged GETs. Set `HF_TOKEN` anyway.
- **GGUF repos contain non-LM files** (`mmproj-*` vision projectors, `mtp-*`
  heads). Filtered.
- **Community finetunes are excluded on purpose** — modified weights confound.
- **Agents told to "verify via web search" without a budget hang indefinitely.**
  A 21-agent workflow burned 4 hours for zero output. Cap it.

## Your actual runbook

```powershell
git clone https://github.com/DHUer/gguf-perf.git ; cd gguf-perf
py -3.13 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128

$env:LLMPERF_HOST = "thinkstation-5080"
$env:HF_TOKEN     = "hf_..."

.venv\Scripts\python -m llmperf.doctor      # DO NOT PROCEED ON A FAIL
.venv\Scripts\python -m llmperf.campaign
```

**llama.cpp must be a CUDA 12.8+ build.** The 5080 is Blackwell, sm_120. Older
builds have no kernels for it and fall back to CPU silently — which presents as
"the GPU is slow", not as an error. `doctor` checks this explicitly.

Then the offload cliff, which is the contribution nobody else has:

```powershell
.venv\Scripts\python -m llmperf.sweep --only Qwen3.8-27B  --force --ngl 0 8 16 24 32 48 99
.venv\Scripts\python -m llmperf.sweep --only gpt-oss-20b  --force --ngl 0 8 16 24 32 48 99
```

Pick three or four models straddling the 16 GB VRAM boundary. Do **not** run the
ngl sweep across all 23 — 7× the cells for little extra signal.

**gpt-oss-120b (63 GB) is the single most valuable cell you can produce.** It
does not fit 16 GB VRAM and runs via offload into your 128 GB system RAM. The
Mac Studio runs the same file fast in unified memory. That contrast is the
paper's central argument and only these two machines together can show it.

When done: `results/measurements_thinkstation-5080.csv` and
`calibration_thinkstation-5080.json` are the deliverable. A few hundred KB.
Commit and push.

Then, and only then:

```powershell
.venv\Scripts\python -m llmperf.analyze   # leave-one-machine-out becomes defined
.venv\Scripts\python -m llmperf.refine
.venv\Scripts\python -m llmperf.figures   # fig9_offload_cliff will now generate
```

## Open issues you inherit

1. **Citations.** Four verified against primary sources (RooflineBench
   arXiv:2602.11506, LIMINAL arXiv:2507.14397, MoE-CAP arXiv:2412.07067,
   Benazir & Lin arXiv:2508.08531 / POMACS 10.1145/3771563). **Six remain
   unverified and are marked do-not-cite.** Every verification so far found the
   survey agent had got something wrong. Do not trust `related_work.md` entries
   that lack a VERIFIED marker, and do not treat a web-search summary as a
   source.
2. **Benazir & Lin is a closer competitor than first reported** — SIGMETRICS
   2026, same setting, five testbeds, pre-empts our quantisation findings and the
   unified-memory argument. The concession is written into §2. Our surviving
   claim is prediction and out-of-sample validation, which they do not do.
3. **A parallel Codex workspace exists** at `.workspaces/codex/` (gitignored,
   31k files). It contains work not in the public repo: `llmperf/audit.py`,
   `llmperf/render_paper.py`, `paper/DATA_COLLECTION.md`, and a different
   `related_work.md`. **Two divergent versions of this paper exist.** Reconcile
   before building further.
4. **Stale numbers in prose.** Fossil tables have twice survived several
   revisions (one said "31 %" when the value was 10.9 %; §5.3 carried a whole
   table from a two-generation-old dataset). Before submission, cross-check every
   percentage in the papers against live `analyze` output.
5. **No LICENSE.** Empirical-SE venues expect one on artifacts.

## House rules

- **Never fabricate a number, citation, or result.** Unmeasured cells are marked
  PENDING, never estimated. The author acts on what is written.
- **Report negative results plainly.** Three hypotheses were rejected here and
  saying so is part of the contribution.
- **Reuse `llama-bench`.** Do not write a replacement timing harness.
- **A surprising result is a suspect measurement first.** Every unexplained
  architectural effect in this project turned out to be a disturbed measurement,
  and the instrument had already flagged it.

## Layout

```
llmperf/common.py         paths, llama-bench discovery, GGUF metadata, KV maths
llmperf/doctor.py         environment check — run first on a new machine
llmperf/fetch.py          model download + fixed train/test split manifest
llmperf/calibrate.py      hardware microbenchmarks (CPU triad, device, LLM probe)
llmperf/repeatability.py  noise floor — run before trusting any effect
llmperf/sweep.py          drives llama-bench; load gate, CV retry, resumable CSV
llmperf/campaign.py       unattended driver: fetch → sweep → analyze → repeat
llmperf/analyze.py        fit, out-of-sample validation, diagnostics
llmperf/refine.py         two-term model, leave-one-machine-out, eta spread
llmperf/figures.py        publication figures (PDF + PNG, print-safe)
paper/paper.md            the predictor paper
paper/measurement_paper.md  the measurement paper (submittable now)
paper/related_work.md     citation audit + verification log
results/                  per-host CSV + calibration JSON (commit these)
results/contaminated/     archived bad data, kept as evidence
```

Self-checks: `python -m llmperf.common`, `python -m llmperf.sweep --selfcheck`,
`python -m llmperf.refine`.
