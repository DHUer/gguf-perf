# Campaign status

Audited 2026-09-05 for the completed MacBook M4 Max, Mac Studio M4 Max, and RTX
5080 campaigns. This is the curated result snapshot; do not overwrite it with
the generic `llmperf.campaign` status writer.

## Coverage

| | MacBook (`lun-mac`) | Studio (`mac-studio-m4-max`) | RTX (`rtx5080`) | Combined |
|---|---:|---:|---:|---:|
| selected successful files | 22 | 23 | 14 | 59 host--file configurations |
| selected successful rows | 132 | 138 | 84 | **354** |
| excluded probe configurations | 3 | 3 | 0 | 6 |
| scored configurations | 19 | 20 | 14 | **53** |
| scored train / test configurations | 15 / 4 | 15 / 5 | 12 / 2 | 42 / 11 |
| scored decode rows | 57 | 60 | 42 | **159** |
| scored prefill rows | 57 | 60 | 42 | **159** |

The 53 scored host--file configurations represent 22 unique GGUF files. Three
Q8 files on each Apple host are selected for provenance but excluded as
calibration probes, accounting for 36 additional observations. RTX has no
probe exclusion. Every selected row satisfies the complete declared protocol,
and no partial-offload row enters the primary cohort.

Studio completed the full 23-file manifest and all 138 phase--depth cells with
no terminal failures, including the 63.39-GB gpt-oss-120B file. The MacBook raw
CSV retains legacy observations and two error records for that file; the
selector keeps complete final-protocol rows atomically and never combines
fields across attempts.

## Host-separated prediction error

| Task/model | MacBook train | MacBook test | Studio train | Studio test | RTX train | RTX test |
|---|---:|---:|---:|---:|---:|---:|
| Decode B0, unfitted total bytes | 33.57% | 46.89% | 32.95% | 57.98% | 20.27% | 41.65% |
| Decode B1, fitted total parameters | 15.80% | 49.36% | 17.76% | 55.25% | **9.85%** | 51.85% |
| Decode B2, fitted active parameters | **10.39%** | **13.11%** | **12.40%** | **14.37%** | 10.12% | **36.15%** |
| Prefill P1, depth 0 | 6.81% | 22.68% | 7.00% | 28.04% | 14.46% | 124.51% |
| Prefill P2, depth 0 | **4.13%** | **18.68%** | **4.27%** | **22.23%** | **5.86%** | 108.18% |

The target-fitted results are reported separately because the host cohorts
differ. A cohort-size-weighted pooled score is not a primary result.

At 16,384 tokens of existing prefix, held-out P2 MAPE is 76.18% on MacBook,
68.11% on Studio, and 133.34% on RTX. The equation has no prefix-dependent work
term, so these are scope failures. RTX depth-zero prefill is already a failed
replication under the measured protocol.

## Cross-host transfer

The B2 median-ratio leave-one-host-out evaluation fits each available source
host independently and gives each source host one vote before applying the
coefficient to the target.

| Held-out target | Source training rows | Target rows | MAPE | Median APE | Maximum APE | Target-fitted all-row MAPE |
|---|---:|---:|---:|---:|---:|---:|
| MacBook M4 Max | 81 | 57 | 14.47% | 8.02% | 73.61% | 10.96% |
| Mac Studio M4 Max | 81 | 60 | 15.42% | 10.78% | 84.30% | 12.89% |
| RTX 5080 | 90 | 42 | 20.80% | 18.18% | 68.13% | 13.84% |

Restricted to each target's fixed test rows:

| Target | Test rows | Transferred MAPE | Median APE | Maximum APE | Target-fitted MAPE |
|---|---:|---:|---:|---:|---:|
| MacBook M4 Max | 12 | 11.59% | 6.00% | 45.82% | 13.11% |
| Mac Studio M4 Max | 15 | 16.76% | 14.36% | 57.74% | 14.37% |
| RTX 5080 | 6 | 35.97% | 25.66% | 68.13% | 36.15% |

Transfer remains close to target fitting on these fixed tests, but the 11 test
configurations span only MXFP4, Q4_K, and Q4_K_M, and RTX contributes only two.
Long tails remain, so this does not establish universal hardware-independent
accuracy.

The rejected two-term absolute-time extension has all-target transfer MAPE of
17.88%, 16.93%, and 149.33% for MacBook, Studio, and RTX. These are not B2
values. `host_transfer.csv` exports all three transfer variants.

## Measurement quality

| Host/phase | Rows | Median CV | Maximum CV | Rows over 3% |
|---|---:|---:|---:|---:|
| MacBook decode | 66 | 0.827% | 3.035% | 1 |
| MacBook prefill | 66 | 1.375% | 4.840% | 8 |
| Studio decode | 69 | 0.424% | 2.304% | 0 |
| Studio prefill | 69 | 0.117% | 0.963% | 0 |
| RTX decode | 42 | 0.433% | 1.652% | 0 |
| RTX prefill | 42 | 3.558% | 39.192% | 29 |

The retry gate applies to decode, not prefill. The noisy prefill observations
are retained as a declared protocol limitation and were not filtered using
prediction error. RTX acquisition spans two sessions and has no independent
between-run repeatability campaign.

## Calibration

| Host | Accelerator/backend | Device-copy bandwidth | FP16 matrix throughput | Threads |
|---|---|---:|---:|---:|
| MacBook M4 Max | Apple Metal/MPS | 380.05 GB/s | 13.95 TFLOP/s | 12 |
| Mac Studio M4 Max | Apple Metal/MPS | 396.73 GB/s | 15.00 TFLOP/s | 12 |
| RTX 5080 | NVIDIA CUDA 12.8 | 801.06 GB/s | 118.83 TFLOP/s | 16 |

The prediction exports use accelerator device-copy bandwidth on both Apple
hosts and CUDA copy bandwidth on RTX. LLM-reference measurements remain a
diagnostic and fallback rather than the primary bandwidth scale.

## Scope and provenance

All rows requested `n_gpu_layers=99`, but physical residency was not
instrumented. Windows shared-memory spill cannot be excluded, and no lower-`ngl`
sweep exists. There is no offload-cliff claim.

The Studio workspace contains exactly 23 GGUF files totaling
336,367,242,336 bytes, no `.part` files, and no undeclared GGUFs.
`model_sources_mac-studio-m4-max.json` records immutable repository revisions
for 13 repositories and LFS SHA-256 values for all 23 files.
`model_integrity_mac-studio-m4-max.json` records a passing full-file hash,
uncached header, and frozen metadata check for every model. The official arm64
llama.cpp b10794 and Tectonic 0.17.0 packages, their extracted binaries,
system/OS details, Python packages, source handoff, and the validated 12-entry
measurement-source inventory are also recorded.

Historical provenance remains less complete: the original MacBook executable
revision is unknown, and the RTX executable's version query did not
independently confirm the b10794 label. Full-file model hashes were added for
the Studio acquisition, not retroactively captured on the original machines.
These distinctions limit runtime and cross-hardware generalization.

## Paper state

The three-host result CSVs, publication figures, generated supplement source,
supplement summary, main manuscript, and packaged PDFs are final and current.

The title is
`GGUF-METADATA PREDICTION OF SINGLE-SEQUENCE LLAMA.CPP THROUGHPUT ACROSS THREE SYSTEMS`.
The main PDF has five pages, with technical content on pages 1--4 and references
only on page 5; its SHA-256 is
`303d94f4612d73a68d164d4236b564b84d97338f172b444639681f32a821e64d`.
The 22-page supplement SHA-256 is
`03341a1748383ce0e3e70054acf86519595f1c80ffb857cb3ab299c52728808f`.

All 96 unit tests pass. The submission audit exits 2 with every
machine-verifiable check PASS and `author identity` as its sole blocker. The
unverified `Author Name`, `Affiliation`, and `author@example.com` block must be
replaced with real details; ICASSP 2027 is non-blind, so do not fabricate it.
