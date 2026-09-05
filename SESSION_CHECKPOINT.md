# Restart checkpoint — 2026-09-04

This is the completed two-host checkpoint for the ICASSP 2027 paper. Model
acquisition is finished; there is no partial download or measurement sweep to
resume.

## Local state

- Repository: `C:\Users\lun\Papers\gguf-perf`
- Git parent commit for this Mac Studio handoff:
  `d1b707c2ed4d0b483b0edc6221f1894676e8149e`
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

- `results/measurements_lun-mac.csv`: 132 selected successful rows from the
  64-GB MacBook Pro M4 Max across 22
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
| MacBook M4 Max | 10.39% / 13.11% | 4.13% / 18.68% |
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
  `DC67DDFC94E2566186D0472990F9EE08E947A0C7FD4F43A7ABA209E8CF20D86D`
  (main) and
  `1429702520E9B2AE3351533C5D7F75D723F451D4F763D98F584B5E3198AFF96C`
  (supplement).

The only non-computational submission blocker is the placeholder author block
(`Author Name`, `Affiliation`, `author@example.com`). ICASSP 2027 is non-blind;
obtain the verified identity rather than inventing it.

## Next campaign: 128-GB Mac Studio M4 Max

The Mac Studio is a planned third host and has **no rows in the frozen paper
cohort yet**. The existing `lun-mac` rows are from the 64-GB MacBook Pro M4 Max.
Do not append Studio measurements to `measurements_lun-mac.csv`.

Use the stable host ID `mac-studio-m4-max`. On the Studio, clone or update the
repository, install `llama.cpp`, and create a native macOS environment:

```bash
git pull --ff-only
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export LLMPERF_HOST=mac-studio-m4-max
export LLAMA_BENCH="$(command -v llama-bench)"
```

Prefer transferring the exact GGUF files used on the MacBook. If they are not
available, inspect and fetch the declared cohort before measuring:

```bash
python -m llmperf.fetch --set all --dry-run
python -m llmperf.fetch --set all
```

Record hardware, OS, source, package, and runner identity before the sweep:

```bash
system_profiler SPHardwareDataType SPDisplaysDataType > results/system_profile_mac-studio-m4-max.txt
sw_vers > results/os_mac-studio-m4-max.txt
git rev-parse HEAD > results/source_commit_mac-studio-m4-max.txt
brew list --versions llama.cpp > results/llama_cpp_package_mac-studio-m4-max.txt
shasum -a 256 "$LLAMA_BENCH" > results/llama_bench_mac-studio-m4-max.sha256
```

Then run the same declared protocol as the completed hosts:

```bash
python -m llmperf.doctor
python -m llmperf.calibrate
python -m llmperf.sweep --dry-run --repetitions 5 --settle 45 --depths 0 4096 16384
python -m llmperf.sweep --repetitions 5 --settle 45 --depths 0 4096 16384
```

Expected new files are `results/calibration_mac-studio-m4-max.json`,
`results/env_mac-studio-m4-max.json`, and
`results/measurements_mac-studio-m4-max.csv`, plus the five provenance files
above. The sweep is append-only and resumes completed cells. Do not use
`llmperf.campaign` for this run: it regenerates analysis and overwrites the
curated `results/STATUS.md` before the three-host paper update is ready.

After collection, return these files to the main workspace. The next analysis
must keep all three hosts separate, recompute leave-one-host-out transfer, and
revise the title, tables, figures, supplement, and claims before submission.

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
- Do not attribute any row to the planned 128-GB Mac Studio; it has not yet
  contributed measurements to the frozen cohort.
- Do not filter prefill cells after observing their residuals.
- Do not restore the invalid 10.9% Mac headline; the atomically selected value
  is 13.11% held out.
- The 63.39-GB gpt-oss-120B Mac run initialized but failed prompt-batch
  execution (`res=-3`); it was not proven to be an out-of-memory failure.
- Keep raw CSVs append-only and preserve the frozen manifest/metadata files.
