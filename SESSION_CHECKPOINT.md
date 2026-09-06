# Restart checkpoint — 2026-09-05

## Current state

The three declared measurement campaigns are complete. Do not resume model
fetching or the Mac Studio sweep unless a later integrity check finds damage.
The generated three-host analysis tables, diagnostic figures, publication
figures, manuscript, supplement, and packaged PDFs are current. Every
machine-verifiable submission check passes; verified author identity is the
only remaining blocker.

Repository: `/Users/lun/Projects/O1A/gguf-perf`

```bash
export LLMPERF_HOST=mac-studio-m4-max
export LLAMA_BENCH=/Users/lun/.local/share/gguf-perf/llama-b10794/llama-bench
export SSL_CERT_FILE=/etc/ssl/cert.pem
```

The native `.venv` uses Python 3.12.14 and PyTorch 2.14.0 with working MPS.
The pinned Tectonic binary and offline cache are:

```text
/Users/lun/.local/share/gguf-perf/tectonic-0.17.0/tectonic
/Users/lun/.local/share/gguf-perf/tectonic-cache
```

The current unit suite passes 96/96 tests. The last full doctor check reported
14 pass, 0 warnings, and 0 failures, including a live Metal decode.

## Completed Mac Studio campaign

The host is a Mac Studio Mac16,9 with an M4 Max, 12 performance plus four
efficiency CPU cores, 40 GPU cores, 128 GB unified memory, and macOS 26.6.2.
Calibration recorded 396.73 GB/s MPS device-copy bandwidth and 15.00 TFLOP/s
FP16 matrix throughput. The official arm64 llama.cpp b10794 runner is pinned to
the same release label as the RTX runner.

All 23 declared GGUFs, totaling 336,367,242,336 bytes, are present. There are
no extra GGUFs or `.part` files. The Studio completed all six declared cells
for every file, including gpt-oss-120B, with no terminal failures:

- 138 selected rows from 23 files;
- 69 decode and 69 prefill observations;
- 18 observations from three Q8 calibration probes excluded from fitting and
  scoring;
- 20 scored configurations, split 15 train and five test;
- 60 scored rows per phase.

The fetch history is no longer operationally relevant, but the frozen
pre-fetch manifest remains at
`/Users/lun/.local/share/gguf-perf/provenance/model_manifest.prefetch.json`
with SHA-256
`2a672d622a30fd2107a01735744de9ba24e97025c3b9b973b5df442b2b662ec6`.

`results/model_integrity_mac-studio-m4-max.json` records a passing verification
of all 23 exact sizes and LFS SHA-256 values, uncached header parsing, and scalar
plus per-depth KV metadata agreement with `results/model_metadata.json`.

## Final cohort

| | MacBook (`lun-mac`) | Studio (`mac-studio-m4-max`) | RTX (`rtx5080`) | Total |
|---|---:|---:|---:|---:|
| selected successful files | 22 | 23 | 14 | 59 host--file configurations |
| selected rows | 132 | 138 | 84 | **354** |
| excluded probe configurations | 3 | 3 | 0 | 6 |
| scored configurations | 19 | 20 | 14 | **53** |
| scored train / test configurations | 15 / 4 | 15 / 5 | 12 / 2 | 42 / 11 |
| scored decode rows | 57 | 60 | 42 | **159** |
| scored prefill rows | 57 | 60 | 42 | **159** |

The scored configurations represent 22 unique GGUF files. Every selected row
uses flash attention, F16 K/V, five repetitions, 45-second settling, requested
`n_gpu_layers=99`, and depths 0/4096/16384. No lower-`ngl` campaign exists.

The MacBook raw CSV retains legacy observations and two explicit records for
the gpt-oss-120B prompt-batch failure. The atomic selector keeps complete
protocol rows first and never splices fields between attempts. The failure was
not proven to be an out-of-memory event.

## Checked prediction results

| Host | B2 decode train/test MAPE | P2 prefill train/test MAPE |
|---|---:|---:|
| MacBook M4 Max | 10.39% / 13.11% | 4.13% / 18.68% |
| Mac Studio M4 Max | 12.40% / 14.37% | 4.27% / 22.23% |
| RTX 5080 | 10.12% / 36.15% | 5.86% / 108.18% |

B2 leave-one-host-out transfer, with coefficients learned from the remaining
host pair, gives:

| Target | All-target MAPE / median / max | Test MAPE / median / max |
|---|---:|---:|
| MacBook | 14.47% / 8.02% / 73.61% | 11.59% / 6.00% / 45.82% |
| Mac Studio | 15.42% / 10.78% / 84.30% | 16.76% / 14.36% / 57.74% |
| RTX 5080 | 20.80% / 18.18% / 68.13% | 35.97% / 25.66% / 68.13% |

Target-fitted all-row B2 MAPE is 10.96%, 12.89%, and 13.84%; target-fitted
test MAPE is 13.11%, 14.37%, and 36.15%. The held-out cohort is still narrow,
spanning MXFP4, Q4_K, and Q4_K_M, so transfer is not evidence for a universal
coefficient.

The rejected two-term absolute-time extension has all-target transfer MAPE of
17.88%, 16.93%, and 149.33% on MacBook, Studio, and RTX. Do not confuse those
values with B2.

At 16,384 tokens of existing prefix, P2 held-out MAPE is 76.18%, 68.11%, and
133.34%. The zero-prefix prefill equation has no context-dependent work term;
these values are a scope diagnostic, and RTX prefill is already unsupported at
depth zero.

## Measurement quality

| Host | Decode CV median/max; rows over 3% | Prefill CV median/max; rows over 3% |
|---|---:|---:|
| MacBook | 0.827% / 3.035%; 1/66 | 1.375% / 4.840%; 8/66 |
| Mac Studio | 0.424% / 2.304%; 0/69 | 0.117% / 0.963%; 0/69 |
| RTX 5080 | 0.433% / 1.652%; 0/42 | 3.558% / 39.192%; 29/42 |

The retry gate applies to decode, not prefill. Preserve every selected prefill
row rather than filtering it after observing residuals.

## Provenance state

- Studio system, OS, Python packages, official b10794 archive/binary, Tectonic
  0.17.0 archive/binary, model repositories/revisions/LFS hashes, and local
  model integrity all have persistent records under `results/`.
- `results/measurement_source_tree_mac-studio-m4-max.sha256` covers the exact
  direct `llmperf/*.py` inventory plus `requirements.txt` used for measurement.
  Do not refresh it after source edits.
- Repository `.git` metadata and Apple command-line tools are absent. The
  source handoff records parent commit
  `d1b707c2ed4d0b483b0edc6221f1894676e8149e`; retain the limitation.
- Historical MacBook executable provenance remains incomplete. The RTX runner
  directory is labelled b10794, but its version query did not independently
  return a revision. Do not generalize runtime comparisons beyond those facts.

## Paper state

The final title is
`GGUF-METADATA PREDICTION OF SINGLE-SEQUENCE LLAMA.CPP THROUGHPUT ACROSS THREE SYSTEMS`.
The current three-host result CSVs, figures, `main.tex`, `supplement.tex`, and
packaged PDFs are regenerated and audited.

- `paper/icassp2027/gguf-throughput-icassp2027.pdf`: five pages, with technical
  content on pages 1--4 and references only on page 5; SHA-256
  `303d94f4612d73a68d164d4236b564b84d97338f172b444639681f32a821e64d`.
- `paper/icassp2027/gguf-throughput-supplement.pdf`: 22 pages; SHA-256
  `03341a1748383ce0e3e70054acf86519595f1c80ffb857cb3ab299c52728808f`.
- Unit tests: 96/96 pass.
- Submission audit: exit status 2 with every machine-verifiable check PASS and
  `author identity` as the sole blocker.

The remaining external blocker is the placeholder author block:

```text
Author Name
Affiliation
author@example.com
```

ICASSP 2027 is non-blind. Obtain verified values; do not invent them.

## Continue after restart

Do not run another fetch or sweep. Regenerate and validate derived artifacts
only if their inputs or paper-side source changed:

```bash
cd /Users/lun/Projects/O1A/gguf-perf
export LLMPERF_HOST=mac-studio-m4-max
export LLAMA_BENCH=/Users/lun/.local/share/gguf-perf/llama-b10794/llama-bench
export SSL_CERT_FILE=/etc/ssl/cert.pem

.venv/bin/python -m llmperf.doctor --quick
.venv/bin/python -m llmperf.analyze
.venv/bin/python -m llmperf.refine
.venv/bin/python paper/icassp2027/generate_main_figures.py
.venv/bin/python paper/icassp2027/generate_supplement.py
.venv/bin/python -m unittest discover -s tests -v
```

Build after the manuscript edit:

```bash
cd paper/icassp2027
export TECTONIC_CACHE_DIR=/Users/lun/.local/share/gguf-perf/tectonic-cache
TECTONIC_BIN=/Users/lun/.local/share/gguf-perf/tectonic-0.17.0/tectonic

mkdir -p build supplement-build
"$TECTONIC_BIN" --only-cached --keep-logs --keep-intermediates \
  --outdir build main.tex
cp build/main.pdf gguf-throughput-icassp2027.pdf

"$TECTONIC_BIN" --only-cached --keep-logs --keep-intermediates \
  --outdir supplement-build supplement.tex
cp supplement-build/supplement.pdf gguf-throughput-supplement.pdf

cd ../..
.venv/bin/python paper/icassp2027/audit_submission.py
```

Before author details are supplied, exit status 2 is acceptable only when
`author identity` is the sole blocker and every machine-verifiable check passes.

## Guardrails

- Do not pool hosts as the primary score.
- Do not describe requested `n_gpu_layers=99` as proof of physical residency.
- Do not claim an offload cliff without lower-`ngl` measurements.
- Do not filter prefill cells after observing residuals.
- Keep raw measurement CSVs append-only.
- Preserve frozen manifests, metadata, model hashes, integrity report, binary
  records, and measurement-source inventory.
- Do not run `llmperf.campaign`; it overwrites curated `results/STATUS.md`.
- Do not restore superseded Mac-only or earlier cohort numbers.
