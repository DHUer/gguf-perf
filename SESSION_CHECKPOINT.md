# Restart checkpoint — 2026-09-04

This is the completed two-host checkpoint for the ICASSP 2027 paper. Model
acquisition is finished; there is no partial download or measurement sweep to
resume.

## Local state

- Repository: `C:\Users\lun\Papers\gguf-perf`
- Git HEAD underlying the working tree:
  `1bdce5e0d917dce152cba4809d05fc1922a98797`
- CUDA runner:
  `C:\Users\lun\AppData\Local\gguf-perf\llama-b10794\llama-bench.exe`
- Tectonic:
  `C:\Users\lun\AppData\Local\gguf-perf\tectonic-0.17.0\tectonic.exe`
- GPU: NVIDIA GeForce RTX 5080, compute capability 12.0, driver 572.70,
  16,302 MiB reported by `llama-bench`
- CPU/RAM: AMD Threadripper PRO 7975WX, 32 cores/64 threads, 127.4 GiB RAM
- RTX calibration: CUDA copy bandwidth 801.06 GB/s and FP16 throughput
  118.83 TFLOP/s with PyTorch 2.11.0+cu128
- Mac calibration: 380.05 GB/s device copy, 377.24 GB/s selected LLM reference,
  and 13.95 TFLOP/s FP16 matrix throughput

The local `models/` directory contains 14 complete GGUF files totaling
100,662,965,376 bytes (about 100.7 GB). No `.part` file remains. Do not start a
new fetch merely to continue the paper.

The working tree intentionally contains the paper, frozen manifests, figures,
tests, RTX results, and submission requirements. No cleanup or reset should be
performed just because these paths are modified or untracked.

## Completed data

- `results/measurements_lun-mac.csv`: 132 selected successful rows from 22
  measured files; 19 scored configurations after three calibration probes are
  excluded.
- `results/measurements_rtx5080.csv`: 84 selected successful rows from 14
  scored files; 12 training and two held-out configurations.
- Combined selected cohort: 216 successful rows (132 Mac and 84 RTX).
- Combined scored cohort: 99 decode and 99 prefill rows across 33 host--file
  configurations and 21 unique GGUF files.
- Every selected RTX cell uses CUDA, requested `n_gpu_layers=99`, five
  repetitions, 45-second settling, and depths 0/4096/16384.
- No partial-offload sweep was run. Requested layer offload does not prove
  physical VRAM residency; no residency/spill telemetry exists.

RTX decode gating succeeded: the maximum selected within-cell CV is 1.652%.
Prefill was not gated, and 29 of 42 RTX prefill rows exceed 3% CV. Preserve
those rows as part of the reported negative result.

## Checked results

| Host | B2 decode train/test MAPE | P2 prefill train/test MAPE |
|---|---:|---:|
| M4 Max | 10.39% / 13.11% | 4.13% / 18.68% |
| RTX 5080 | 10.12% / 36.15% | 5.86% / 108.18% |

B2 median-ratio leave-one-host-out MAPE over all target rows is 20.82% on Mac
and 21.69% on RTX; median APE is 15.41%/18.75% and maximum APE is
75.81%/71.53%, versus target-fitted all-row MAPE of 10.96%/13.84%. On target
test rows only, transfer gives 13.94%/36.70% MAPE (median 8.63%/26.18%), close
to target-fitted 13.11%/36.15%. This narrow,
all-Q4 result does not show target adaptation is necessary, but neither does it
establish universal transfer. The rejected two-term output-projection model,
not B2, has 57.1%/116.8% transfer MAPE. Prefill prediction still fails on RTX.

The generated `results/host_transfer.csv` records the all-target B2,
target-test B2, and rejected two-term absolute-time variants.

## Paper state

- Title: `GGUF-METADATA PREDICTION OF SINGLE-SEQUENCE LLAMA.CPP THROUGHPUT ACROSS TWO SYSTEMS`
- Source: `paper/icassp2027/main.tex`
- Main PDF: `paper/icassp2027/gguf-throughput-icassp2027.pdf`
- Supplement: `paper/icassp2027/gguf-throughput-supplement.pdf`
- Main PDF: five pages; pages 1--4 are technical content and page 5 is
  references only
- Supplement: 16 pages
- Final verification: 30 unit tests pass; the latest quick `llmperf.doctor`
  reports 13 pass, 0 warn, 0 fail, and an earlier full run reported 14 pass,
  including a live RTX decode. The submission audit passes every
  machine-verifiable check and blocks only on author identity. Both PDFs are
  unencrypted Letter documents with embedded fonts, and the LaTeX logs have no
  undefined references, overfull boxes, or balance-package warnings.
- Packaged PDF SHA-256 values:
  `80C5F60A9050E6E1184195C388987CD5F4CDD578D8C0C7D36D26BE6838F2CE4A`
  (main) and
  `38D8BA99504B0E0B9CC5AA659496343EE802C4C725CF79B76185C3C0454AEDDF`
  (supplement).

The only non-computational submission blocker is the placeholder author block
(`Author Name`, `Affiliation`, `author@example.com`). ICASSP 2027 is non-blind;
obtain the verified identity rather than inventing it.

## Continue after restart

Open PowerShell:

```powershell
Set-Location 'C:\Users\lun\Papers\gguf-perf'
$env:LLMPERF_HOST = 'rtx5080'
$env:LLAMA_BENCH = 'C:\Users\lun\AppData\Local\gguf-perf\llama-b10794\llama-bench.exe'
```

`llama-bench` is also auto-discovered from this managed install path after a
restart; the explicit variable pins the measured binary in the b10794-labeled
directory, although its version query did not independently report a revision.
These variables are only needed for hardware checks or new measurements.
Analysis uses the saved per-host files.

Validate and regenerate analysis artifacts:

```powershell
.\.venv\Scripts\python.exe -m llmperf.doctor --quick
.\.venv\Scripts\python.exe -m llmperf.analyze
.\.venv\Scripts\python.exe -m llmperf.refine
.\.venv\Scripts\python.exe -m llmperf.figures
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe paper\icassp2027\audit_submission.py
git diff --check
```

Rebuild the paper from `paper\icassp2027`:

```powershell
Set-Location 'C:\Users\lun\Papers\gguf-perf\paper\icassp2027'
$tectonic = 'C:\Users\lun\AppData\Local\gguf-perf\tectonic-0.17.0\tectonic.exe'

New-Item -ItemType Directory -Force build, supplement-build | Out-Null
& $tectonic --keep-logs --keep-intermediates --outdir build main.tex
Copy-Item build\main.pdf gguf-throughput-icassp2027.pdf
..\..\.venv\Scripts\python.exe generate_supplement.py
& $tectonic --keep-logs --keep-intermediates --outdir supplement-build supplement.tex
Copy-Item supplement-build\supplement.pdf gguf-throughput-supplement.pdf
```

After inserting author details, repeat the build and page/font/reference audit.
Do not restore an earlier Mac-only draft or acquisition checkpoint.

## Guardrails

- Do not pool hosts as the primary score; report each host's cohort.
- Do not describe `n_gpu_layers=99` as proof of full physical residency.
- Do not claim an offload cliff without lower-`ngl` measurements.
- Do not filter prefill cells after observing their residuals.
- Do not restore the invalid 10.9% Mac headline; the atomically selected value
  is 13.11% held out.
- The 63.39-GB gpt-oss-120B Mac run initialized but failed prompt-batch
  execution (`res=-3`); it was not proven to be an out-of-memory failure.
- Keep raw CSVs append-only and preserve the frozen manifest/metadata files.
