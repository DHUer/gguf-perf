# Project handoff

Last updated 2026-09-05 after completing the MacBook M4 Max, Mac Studio M4
Max, and RTX 5080 campaigns, analysis, and final paper build.

## State in one paragraph

The strict selector retains 354 protocol-complete rows: 132 MacBook, 138 Mac
Studio, and 84 RTX. After six Apple calibration-probe configurations are
excluded, the scored data contain 159 decode plus 159 prefill rows across 53
host--file configurations and 22 unique GGUF files. Target-fitted B2 held-out
MAPE is 13.11%, 14.37%, and 36.15% on MacBook, Studio, and RTX. Test-only B2
transfer from the remaining host pair gives 11.59%, 16.76%, and 35.97%. The
derived result tables, figures, manuscript, and packaged PDFs are current. The
audit passes every machine-verifiable check; verified author identity remains
the only non-computational blocker.

## Completed campaigns

- MacBook Pro M4 Max (`lun-mac`): 132 selected rows from 22 successful files;
  19 scored configurations after three Q8 probes are excluded, split 15 train
  and four test, yielding 57 rows per phase.
- Mac Studio M4 Max (`mac-studio-m4-max`): the complete 23-file manifest
  produced 138 selected rows with no failures; 20 scored configurations after
  three Q8 probes are excluded, split 15 train and five test, yielding 60 rows
  per phase. The 63.39-GB gpt-oss-120B file completed all six cells.
- RTX 5080 (`rtx5080`): 84 selected rows from 14 scored files, split 12 train
  and two test, yielding 42 rows per phase.
- Combined: 354 selected rows, including 36 Apple probe observations, and 318
  scored rows across 53 host--file configurations.

Every selected row uses flash attention, F16 K/V, five repetitions, 45-second
settling, requested `n_gpu_layers=99`, and depths 0/4096/16384. No
partial-offload sweep was collected.

## Checked results

| Host | Split | B0 | B1 | B2 | P2 depth 0 |
|---|---|---:|---:|---:|---:|
| MacBook M4 Max | train | 33.57% | 15.80% | **10.39%** | **4.13%** |
| MacBook M4 Max | test | 46.89% | 49.36% | **13.11%** | **18.68%** |
| Mac Studio M4 Max | train | 32.95% | 17.76% | **12.40%** | **4.27%** |
| Mac Studio M4 Max | test | 57.98% | 55.25% | **14.37%** | **22.23%** |
| RTX 5080 | train | 20.27% | **9.85%** | 10.12% | **5.86%** |
| RTX 5080 | test | 41.65% | 51.85% | **36.15%** | 108.18% |

The three-host B2 transfer results are:

| Target | All-target MAPE / median / max | Test MAPE / median / max | Target-fitted all/test MAPE |
|---|---:|---:|---:|
| MacBook | 14.47% / 8.02% / 73.61% | 11.59% / 6.00% / 45.82% | 10.96% / 13.11% |
| Mac Studio | 15.42% / 10.78% / 84.30% | 16.76% / 14.36% / 57.74% | 12.89% / 14.37% |
| RTX 5080 | 20.80% / 18.18% / 68.13% | 35.97% / 25.66% / 68.13% | 13.84% / 36.15% |

The fixed test sets span MXFP4, Q4_K, and Q4_K_M, and the RTX target has only
two configurations. Transfer is informative but does not establish a universal
coefficient. The rejected two-term absolute-time extension has all-target MAPE
of 17.88%, 16.93%, and 149.33%; those values are not B2 results.

The prefill model remains a negative result on RTX. At 16K existing-prefix
depth, held-out P2 MAPE is 76.18% on MacBook, 68.11% on Studio, and 133.34% on
RTX, outside the zero-prefix equation's intended scope.

## Measurement quality

- MacBook decode median/max CV: 0.827%/3.035%; one of 66 rows exceeds 3%.
  Prefill median/max: 1.375%/4.840%; 8/66 exceed 3%.
- Studio decode median/max CV: 0.424%/2.304%. Prefill median/max:
  0.117%/0.963%. No Studio selected row exceeds 3%.
- RTX decode median/max CV: 0.433%/1.652%. Prefill median/max:
  3.558%/39.192%; 29/42 exceed 3%.

Prefill was not gated. Retain its noisy observations; never remove cells after
examining prediction residuals.

## Studio environment and provenance

Repository root: `/Users/lun/Projects/O1A/gguf-perf`.

```bash
export LLMPERF_HOST=mac-studio-m4-max
export LLAMA_BENCH=/Users/lun/.local/share/gguf-perf/llama-b10794/llama-bench
export SSL_CERT_FILE=/etc/ssl/cert.pem
```

- Mac Studio Mac16,9: M4 Max, 12 performance plus four efficiency CPU cores,
  40 GPU cores, 128 GB unified memory, macOS 26.6.2.
- Calibration: 396.73 GB/s MPS device-copy bandwidth and 15.00 TFLOP/s FP16.
- Official arm64 llama.cpp b10794 binary and archive hashes are recorded under
  `results/`; this tag matches the RTX runner directory.
- Tectonic 0.17.0:
  `/Users/lun/.local/share/gguf-perf/tectonic-0.17.0/tectonic`.
- Offline cache: `/Users/lun/.local/share/gguf-perf/tectonic-cache`.
- Exactly 23 GGUF files total 336,367,242,336 bytes. There are no extra GGUFs
  or `.part` files.
- `results/model_integrity_mac-studio-m4-max.json` records successful full-file
  SHA-256, uncached header parsing, and frozen metadata checks for every file.
- `results/measurement_source_tree_mac-studio-m4-max.sha256` freezes all direct
  `llmperf/*.py` modules and `requirements.txt` used for measurement.
- Repository `.git` metadata and Apple command-line tools are absent. The
  recorded handoff commit is `d1b707c2ed4d0b483b0edc6221f1894676e8149e`;
  preserve the limitation instead of manufacturing a local commit identity.

Do not refresh the measurement-source checksum after editing analysis code: it
attests what ran. Paper, tests, and documentation are outside that inventory.

## Paper state

The regenerated CSV exports, host-separated publication figures,
`paper/icassp2027/supplement.tex`, and `supplement_summary.pdf` contain the
three-host results. The final main and supplement PDFs are packaged and match
their build products byte-for-byte.

The title is
`GGUF-METADATA PREDICTION OF SINGLE-SEQUENCE LLAMA.CPP THROUGHPUT ACROSS THREE SYSTEMS`.

- Main PDF: five pages, with technical content on pages 1--4 and references
  only on page 5; SHA-256
  `303d94f4612d73a68d164d4236b564b84d97338f172b444639681f32a821e64d`.
- Supplement PDF: 22 pages; SHA-256
  `03341a1748383ce0e3e70054acf86519595f1c80ffb857cb3ab299c52728808f`.
- Unit tests: 96/96 pass.
- Submission audit: exit status 2, every machine-verifiable check PASS, with
  `author identity` as the sole blocker.

Replace `Author Name`, `Affiliation`, and `author@example.com` with verified
details, then rebuild and rerun the audit. ICASSP 2027 is non-blind; never
invent them.

Regenerate derived artifacts from the repository root with:

```bash
.venv/bin/python -m llmperf.analyze
.venv/bin/python -m llmperf.refine
.venv/bin/python paper/icassp2027/generate_main_figures.py
.venv/bin/python paper/icassp2027/generate_supplement.py
.venv/bin/python -m unittest discover -s tests -v
```

Do not run `llmperf.campaign`: its generic status writer would replace the
curated `results/STATUS.md`.

## Guardrails

1. Keep host results separate; pooled values are not primary claims.
2. Treat `n_gpu_layers=99` as a request, not physical-residency telemetry.
3. Make no offload-cliff claim without lower-`ngl` measurements.
4. Preserve the append-only raw CSVs and frozen manifest/metadata/provenance.
5. Keep the three Q8 probe exclusions host-specific.
6. Preserve explicit failures. The MacBook gpt-oss-120B prompt-batch failure
   was not proven to be out of memory; Studio completed the same file.
7. Do not filter prefill observations post hoc.
8. Do not restore the superseded 10.9% Mac headline; the fixed MacBook held-out
   B2 value is 13.11%.

## Source of truth

```text
results/measurements_lun-mac.csv               MacBook raw measurements
results/measurements_mac-studio-m4-max.csv     Studio raw measurements
results/measurements_rtx5080.csv               RTX raw measurements
results/model_manifest.json                    frozen cohort/splits/sizes
results/model_metadata.json                    frozen GGUF metadata
results/error_table_by_host.csv                decode headline values
results/error_table_prefill_by_host.csv        prefill headline values
results/host_transfer.csv                      all-target/test transfer
results/STATUS.md                              audited campaign summary
paper/icassp2027/main.tex                      submission manuscript
paper/icassp2027/generate_supplement.py        supplement generator
SESSION_CHECKPOINT.md                          restart-specific state
```

The other Markdown manuscripts under `paper/` are historical working drafts.
