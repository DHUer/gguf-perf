# GGUF throughput prediction on consumer hardware

This repository measures and predicts single-sequence `llama.cpp` throughput
from GGUF metadata. The completed study covers three systems: a 64-GB MacBook
Pro M4 Max, a 128-GB Mac Studio M4 Max, and an NVIDIA RTX 5080.

The model is deliberately small and interpretable:

```text
decode:  tok/s = eta_d(host, quant) * bandwidth
                 / (active weight bytes + KV bytes)
prefill: tok/s = eta_p(host, quant) * FLOPS / (2 * active parameters)
```

GGUF tensor shapes and routing metadata estimate activated parameters for
Mixture-of-Experts (MoE) models. Per-layer attention metadata accounts for
global, sliding-window, and recurrent state instead of treating every layer as
global attention.

## Completed evidence

The strict selector retains 354 protocol-complete phase--depth rows: 132 from
the MacBook, 138 from the Mac Studio, and 84 from the RTX 5080. The Studio
completed the full 23-file manifest, including the 63.39-GB gpt-oss-120B file.
After three Q8 calibration probes on each Apple host are excluded, the scored
cohort contains 159 decode and 159 prefill rows from 53 host--file
configurations and 22 unique GGUF files.

| Host | Scored train/test configurations | Decode B2 train/test MAPE | Prefill P2 train/test MAPE |
|---|---:|---:|---:|
| MacBook Pro M4 Max | 15 / 4 | 10.39% / 13.11% | 4.13% / 18.68% |
| Mac Studio M4 Max | 15 / 5 | 12.40% / 14.37% | 4.27% / 22.23% |
| NVIDIA RTX 5080 | 12 / 2 | 10.12% / 36.15% | 5.86% / 108.18% |

These are target-host fits and are always reported by host. In the separate B2
median-ratio leave-one-host-out diagnostic, all-target MAPE is 14.47%, 15.42%,
and 20.80% for MacBook, Studio, and RTX respectively, versus target-fitted
all-row MAPE of 10.96%, 12.89%, and 13.84%. On fixed target test rows, transfer
MAPE is 11.59%, 16.76%, and 35.97%, compared with target-fitted 13.11%, 14.37%,
and 36.15%.

The transfer result is useful but narrow: held-out formats span MXFP4, Q4_K,
and Q4_K_M, cohorts differ by host, and the RTX target has only two held-out
configurations. It does not establish a universal hardware-independent
coefficient. The simple prefill equation remains unsupported on RTX.

`results/host_transfer.csv` contains the all-target and target-test B2 transfer
rows plus the rejected two-term output-projection experiment. See
[`results/STATUS.md`](results/STATUS.md) for the complete audited snapshot and
[`SESSION_CHECKPOINT.md`](SESSION_CHECKPOINT.md) for local restart state.

## Paper status

The final packaged ICASSP manuscript is
`GGUF-METADATA PREDICTION OF SINGLE-SEQUENCE LLAMA.CPP THROUGHPUT ACROSS THREE SYSTEMS`.
The main PDF has five pages: pages 1--4 contain technical content and page 5
contains references only. The companion supplement has 22 pages. Their
SHA-256 values are:

- main: `303d94f4612d73a68d164d4236b564b84d97338f172b444639681f32a821e64d`
- supplement: `03341a1748383ce0e3e70054acf86519595f1c80ffb857cb3ab299c52728808f`

All 96 unit tests pass. The submission audit returns status 2 with every
machine-verifiable check passing and `author identity` as its sole blocker.

The placeholder author name, affiliation, and email in
`paper/icassp2027/main.tex` must still be replaced. ICASSP 2027 is non-blind,
so those fields must be supplied by the author and must not be invented.

Historical Markdown drafts elsewhere under `paper/` are not submission sources.

## Environment

- Python 3.9+
- `llama-bench` from `llama.cpp`
- macOS/Metal or Windows/NVIDIA CUDA for measurement
- Tectonic or a conventional LaTeX/BibTeX toolchain for the paper

The Studio run used Python 3.12.14, PyTorch 2.14.0 with MPS, and the official
arm64 `llama.cpp` b10794 release. The RTX run used PyTorch 2.11.0+cu128 and a
managed b10794-labelled runner. Exact package and binary records are under
`results/`.

Create an environment with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `py -m venv .venv` and `.venv\Scripts\python.exe`.

## Reproduce the checked results

The Studio workspace already contains the exact 23-file manifest, so no model
download is needed for analysis regeneration.

```bash
.venv/bin/python -m llmperf.analyze
.venv/bin/python -m llmperf.refine
.venv/bin/python paper/icassp2027/generate_main_figures.py
.venv/bin/python paper/icassp2027/generate_supplement.py
.venv/bin/python -m unittest discover -s tests -v
```

Analysis and publication generation use the frozen metadata snapshot rather
than whichever GGUFs happen to be installed. Paper build and audit commands are
in [`paper/icassp2027/README.md`](paper/icassp2027/README.md).

## Extending the study

Give every new machine a stable host ID, retain its measurements in a distinct
`results/measurements_<host>.csv`, and preserve calibration, environment,
binary, model-source, and model-integrity provenance. Inspect the fixed model
set before acquiring anything:

```bash
.venv/bin/python -m llmperf.fetch --set all --dry-run
.venv/bin/python -m llmperf.doctor
.venv/bin/python -m llmperf.calibrate
.venv/bin/python -m llmperf.sweep --dry-run --repetitions 5 --settle 45 \
  --depths 0 4096 16384
```

Only one fetch or measurement campaign may run at a time. Never delete a live
campaign lock to force another process through it.

## Measurement controls

| Setting | Value |
|---|---|
| flash attention | `-fa on` |
| KV cache | `-ctk f16 -ctv f16` |
| repetitions | 5 |
| settling | 45 seconds |
| prefix depths | 0, 4096, 16384 |
| process isolation | one process per model/offload cell |
| selector | complete protocol first, then CV/load; never residual error |

Decode maximum within-cell CV is 3.035% on MacBook, 2.304% on Studio, and
1.652% on RTX. Prefill was not gated: 8/66 MacBook and 29/42 RTX rows exceed
3% CV, while all 69 Studio prefill rows are below 1%. Those observations remain
in the fixed cohort rather than being filtered after seeing prediction error.

## Provenance and important outputs

The Studio workspace contains exactly 23 final GGUFs totaling
336,367,242,336 bytes, with no extra `.gguf` or `.part` files. The persisted
integrity report verifies every full-file SHA-256, header parse, and frozen
metadata comparison. The official b10794 runner archive/binary, Tectonic
0.17.0 binary, Python environment, hardware/OS, source handoff, and exact
measurement-source checksum inventory are also recorded under `results/`.

```text
results/model_manifest.json                         frozen split/size provenance
results/model_metadata.json                         frozen GGUF-derived metadata
results/model_sources_mac-studio-m4-max.json        repository revisions and LFS hashes
results/model_integrity_mac-studio-m4-max.json      local 23-file verification report
results/calibration_<host>.json                     per-host calibration
results/env_<host>.json                             per-host runtime/protocol record
results/measurements_<host>.csv                     append-only raw measurements
results/error_table_by_host.csv                     host-separated decode errors
results/error_table_prefill_by_host.csv             host-separated prefill errors
results/host_transfer.csv                           B2 and rejected-extension transfer
results/predictions.csv                             row-level decode predictions
results/predictions_prefill.csv                     row-level prefill predictions
figures/paper/                                      generated publication figures
paper/icassp2027/                                   submission source and PDFs
```

Never pool hosts as the primary score, substitute estimates for missing runs,
or filter cells using prediction residuals. `n_gpu_layers=99` records a request,
not proof of physical accelerator residency. No lower-`ngl` sweep exists, so no
offload-cliff claim is supported.
