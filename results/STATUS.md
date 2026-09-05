# Campaign status

Audited 2026-09-04 for the two-host ICASSP 2027 manuscript. This snapshot
supersedes the earlier Mac-only status and RTX download checkpoints.

## Coverage

| | M4 Max (`lun-mac`) | RTX 5080 (`rtx5080`) | Combined |
|---|---:|---:|---:|
| measured files | 22 | 14 | 36 host--file configurations |
| selected successful rows | 132 | 84 | **216** |
| scored configurations | 19 | 14 | **33** |
| scored train / test configurations | 15 / 4 | 12 / 2 | 27 / 6 |
| scored decode rows | 57 | 42 | **99** |
| scored prefill rows | 57 | 42 | **99** |

The 33 scored host--file configurations correspond to 21 unique GGUF files.
Three Mac Q8 files used as calibration probes are selected for provenance but
excluded from scoring. All RTX files are scored. Every selected row uses the
complete quality protocol and requested `n_gpu_layers=99`; there are no
partial-offload rows.

The archived Mac CSV also retains earlier pre-protocol rows and a prompt-batch
failure. The atomic selector excludes archived attempts whenever a complete
final-protocol row exists and never combines fields from different attempts.

## Host-separated prediction error

| Task/model | Mac train | Mac test | RTX train | RTX test |
|---|---:|---:|---:|---:|
| Decode B0, unfitted total bytes | 33.57% | 46.89% | 20.27% | 41.65% |
| Decode B1, fitted total parameters | 15.80% | 49.36% | **9.85%** | 51.85% |
| Decode B2, fitted active parameters | **10.39%** | **13.11%** | 10.12% | **36.15%** |
| Prefill P1, depth 0 | 6.81% | 22.68% | 14.46% | 124.51% |
| Prefill P2, depth 0 | **4.13%** | **18.68%** | **5.86%** | 108.18% |

Results are reported by host because the cohorts differ and the Mac contributes
more held-out rows. A pooled score is not the primary result.

At 16,384 tokens of existing prefix, held-out P2 MAPE is 76.18% on Mac and
133.34% on RTX. The equation has no context-dependent work term, so this is a
scope failure. RTX depth-zero prefill is already a failed replication.

## Cross-host transfer

The B2 median-ratio leave-one-host-out evaluation gives:

| Held-out host | Rows | MAPE | Median APE | Maximum APE |
|---|---:|---:|---:|---:|
| M4 Max | 57 | 20.82% | 15.41% | 75.81% |
| RTX 5080 | 42 | 21.69% | 18.75% | 71.53% |

For context, target-fitted B2 has all-row MAPE of 10.96% on Mac and 13.84% on
RTX. Unlike leave-one-host-out transfer, that comparator mixes training rows
used to fit the target coefficients with held-out rows.

Restricted to each target's fixed test rows:

| Target host | Test rows | Transferred MAPE | Median APE | Target-fitted MAPE |
|---|---:|---:|---:|---:|
| M4 Max | 12 | 13.94% | 8.63% | 13.11% |
| RTX 5080 | 6 | 36.70% | 26.18% | 36.15% |

The transferred and target-fitted errors are close on this narrow test, so the
data do not show that target-host adaptation is necessary. However, all six
test configurations are Q4 variants, RTX contributes only two, and the
all-target maxima remain above 70%; this does not establish universal
hardware-independent accuracy.

The 57.1% Mac and 116.8% RTX transfer errors belong to the rejected two-term
output-projection extension, not to B2.

`results/host_transfer.csv` exports all three evaluated variants: B2 over all
target rows, B2 restricted to target test rows, and the rejected two-term
absolute-time extension.

## Measurement quality

- RTX decode: 42 selected rows, maximum within-cell CV 1.652%.
- RTX prefill: 29 of 42 selected rows exceed 3% CV; prefill was not gated.
- All selected RTX rows use CUDA, five repetitions, 45-second settling, and
  prefix depths 0/4096/16384.
- The RTX cohort was acquired across two sessions and has no independent
  between-run repeatability campaign.

The noisy prefill observations are retained as a declared protocol limitation;
they were not removed based on prediction error.

## Scope and provenance

The RTX run requested full layer offload, but no physical VRAM-residency or
Windows shared-memory-spill telemetry was recorded. The largest analytical
working sets may exceed reported VRAM once runtime buffers are included. There
is no physical-residency proof and no offload sweep.

Both held-out cohorts contain only Q4 variants, and RTX has only two held-out
files. The Mac executable revision was not recovered; the RTX executable came
from a managed directory labeled `llama-b10794`, but its version query did not
independently report a revision. Full historical model hashes were not frozen for every source
file. These limitations preclude universal runtime, quantization, or
cross-hardware accuracy claims.

See `paper/icassp2027/gguf-throughput-supplement.pdf` for the 16-page row-level
tables, protocol diagnostics, calibration details, negative results, and source
hash audit.

The generated paper is titled `GGUF-METADATA PREDICTION OF SINGLE-SEQUENCE LLAMA.CPP THROUGHPUT ACROSS TWO SYSTEMS`. It has five pages: technical content on
pages 1--4 and references only on page 5, using ICASSP's permitted fifth-page
allowance. Its sole non-computational submission blocker is the unverified
placeholder author name, affiliation, and email; ICASSP 2027 is non-blind.
