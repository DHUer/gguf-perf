# Project handoff

Last updated 2026-09-04 after completing the two-host analysis and ICASSP 2027
paper build.

## State in one paragraph

The M4 Max and RTX 5080 campaigns needed for the paper are complete. The strict
selector retains 216 measurements, and the scored data contain 99 decode plus
99 prefill rows across 33 host--file configurations and 21 unique GGUF files.
Target-fitted B2 reaches 13.11% held-out MAPE on the Mac and 36.15% on RTX.
B2 median-ratio transfer gives 13.94% and 36.70% on those same test rows, so
target adaptation is not shown to be necessary within this narrow Q4 test. Its
all-target MAPE is 20.82%/21.69%, versus target-fitted all-row MAPE of
10.96%/13.84%; the 57.1%/116.8% values belong only to a rejected two-term
extension. The main PDF has technical content on pages 1--4
and references only on page 5, and the supplement is 16 pages. The only
non-computational submission blocker is the real author name, affiliation, and
email required by the non-blind ICASSP submission.

## What is complete

- Mac: 132 selected rows from 22 measured files. After three Q8 calibration
  probes are excluded, 15 training and four held-out configurations contribute
  57 decode and 57 prefill rows.
- RTX: 84 selected rows from 14 files, all scored: 12 training and two held-out
  configurations contribute 42 rows per phase.
- All selected RTX cells completed with CUDA, five repetitions, a 45-second
  settle, requested `n_gpu_layers=99`, and depths 0/4096/16384.
- Analysis, host-separated figures, the ICASSP source, and the complete
  supplement have been regenerated for both hosts.
- The local RTX model directory contains the 14-file, 100.7-GB cohort. There is
  no interrupted `.part` download to resume.

Headline MAPE values are generated in
`results/error_table_by_host.csv` and
`results/error_table_prefill_by_host.csv`:

| Host | Split | B0 | B1 | B2 | P2 depth 0 |
|---|---|---:|---:|---:|---:|
| M4 Max | train | 33.57 | 15.80 | **10.39** | **4.13** |
| M4 Max | test | 46.89 | 49.36 | **13.11** | **18.68** |
| RTX 5080 | train | 20.27 | **9.85** | 10.12 | **5.86** |
| RTX 5080 | test | 41.65 | 51.85 | **36.15** | 108.18 |

## Claims that survived

1. Activated-parameter accounting materially improves held-out decode error on
   both hosts relative to charging all stored weights.
2. Per-layer GGUF metadata prevents large KV-byte overestimates for hybrid and
   sliding-window architectures.
3. A B2 median ratio learned on the other host transfers usefully to the fixed
   held-out Q4 rows: 13.94% Mac and 36.70% RTX MAPE, close to target-fitted
   13.11% and 36.15%.
4. Quantization efficiency and even low-bit ordering are backend-specific.

## Negative results and hard boundaries

- **Cross-host transfer is useful but narrowly tested.** Across all target rows,
  B2 median-ratio transfer gives 20.82% MAPE on 57 Mac rows and 21.69% on 42
  RTX rows (median 15.41%/18.75%, maximum 75.81%/71.53%), versus target-fitted
  all-row MAPE of 10.96%/13.84%. On target test rows alone it gives
  13.94%/36.70% MAPE (median 8.63%/26.18%), nearly matching the target-fitted
  13.11%/36.15%. All six test configurations are Q4 variants, and RTX has only
  two, so this does not establish universal transfer.
- **The prefill baseline is unsupported on RTX under this protocol.** P2
  held-out depth-zero MAPE is 108.18% on RTX, and it worsens outside the
  equation's zero-prefix scope.
- **No residency claim.** `n_gpu_layers=99` is a request, not physical VRAM
  telemetry. Windows shared-memory spill was not measured.
- **No offload-cliff claim.** No `n_gpu_layers < 99` sweep was collected.
- **No universal quantization claim.** Many RTX quantization groups have only
  one training family, and each host's held-out set contains only Q4 variants.
- **No accepted output-projection extension.** Its leave-one-host-out errors
  are 57.1% on Mac and 116.8% on RTX, far worse than B2, and it did not deliver
  a material, physically credible held-out improvement.

The RTX prefill protocol is an explicit limitation: 29 of 42 selected prefill
rows exceed 3% within-cell CV, whereas the selected RTX decode maximum is
1.652%. Do not filter those prefill rows after looking at their errors.

`results/host_transfer.csv` is the generated transfer source: it contains B2
over all target rows, B2 over target test rows, and the rejected two-term
absolute-time extension.

## Submission action

Replace this placeholder in `paper/icassp2027/main.tex`:

```text
Author Name
Affiliation
author@example.com
```

with the verified author identity, then rebuild and recheck the PDFs. ICASSP
2027 is not blind. Do not invent these fields.

The current generated artifacts are:

- `paper/icassp2027/gguf-throughput-icassp2027.pdf` — five pages: technical
  content on pages 1--4 and references only on page 5
- `paper/icassp2027/gguf-throughput-supplement.pdf` — 16 pages

Their SHA-256 values are `80C5F60A9050E6E1184195C388987CD5F4CDD578D8C0C7D36D26BE6838F2CE4A`
and `38D8BA99504B0E0B9CC5AA659496343EE802C4C725CF79B76185C3C0454AEDDF`,
respectively.

The main-paper title is `GGUF-METADATA PREDICTION OF SINGLE-SEQUENCE LLAMA.CPP THROUGHPUT ACROSS TWO SYSTEMS`.

## Reproduce and validate

From `C:\Users\lun\Papers\gguf-perf`:

```powershell
.\.venv\Scripts\python.exe -m llmperf.analyze
.\.venv\Scripts\python.exe -m llmperf.refine
.\.venv\Scripts\python.exe -m llmperf.figures
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe paper\icassp2027\audit_submission.py
git diff --check
```

Build instructions and the local Tectonic command are in
`paper/icassp2027/README.md`. The analysis does not require another model
download.

## If the study is extended

The highest-value extension is not another uninstrumented resident baseline.
It is a planned discrete-GPU experiment with physical residency/spill
telemetry, randomized execution order, additional held-out quantization
families, and a declared partial-offload grid. Keep every lower-`ngl` row out of
the current primary fit.

For any new host:

```powershell
$env:LLMPERF_HOST = 'stable-host-name'
$env:LLAMA_BENCH = 'C:\path\to\llama-bench.exe'
.\.venv\Scripts\python.exe -m llmperf.doctor
.\.venv\Scripts\python.exe -m llmperf.calibrate
.\.venv\Scripts\python.exe -m llmperf.sweep --dry-run
```

Do not run a download and benchmark concurrently. Preserve failed rows, raw
logs, runtime identification, and model hashes; never relabel a prompt-batch
failure as an out-of-memory failure without direct evidence.

## Known traps with regression guards

1. Resume keys must use the phase-specific values written by `llama-bench`.
   Guard: `python -m llmperf.sweep --selfcheck`.
2. Fused MoE tensors such as `ffn_gate_up_exps` must be included in activated
   parameter accounting.
3. CSV append must migrate schema before writing new columns.
4. Windows load gating uses `psutil`; `os.getloadavg()` is unavailable.
5. Final-protocol row selection is atomic. Never splice favorable fields from
   different attempts.
6. Flash attention and F16 K/V must remain pinned; backend-dependent `auto`
   settings change the experiment.

## Source of truth

```text
llmperf/analyze.py                      fitting and host-separated validation
llmperf/refine.py                       transfer and rejected extensions
llmperf/figures.py                      publication figures
results/measurements_lun-mac.csv        Mac raw measurements
results/measurements_rtx5080.csv        RTX raw measurements
results/host_transfer.csv               all-target/test B2 and two-term transfer
results/STATUS.md                       audited cohort summary
paper/icassp2027/main.tex               submission manuscript
paper/icassp2027/generate_supplement.py supplement generator
SESSION_CHECKPOINT.md                   restart-specific local state
```

The older Markdown manuscripts under `paper/` are working history, not the
submission source. Do not restore their superseded Mac-only numbers.
