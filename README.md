# GGUF throughput prediction on consumer hardware

This repository measures and predicts single-sequence `llama.cpp` throughput from
GGUF metadata. The completed study compares a 64-GB MacBook Pro M4 Max with an NVIDIA RTX
5080 and is the source for the ICASSP 2027 manuscript in
[`paper/icassp2027/`](paper/icassp2027/).

The model is deliberately small and interpretable:

```text
decode:  tok/s = eta_d(host, quant) * bandwidth
                 / (active weight bytes + KV bytes)
prefill: tok/s = eta_p(host, quant) * FLOPS / (2 * active parameters)
```

GGUF tensor shapes and routing metadata estimate activated parameters for
Mixture-of-Experts (MoE) models. Per-layer attention metadata accounts for
global, sliding-window, and recurrent state instead of pretending that every
layer has a global KV cache.

## Completed evidence

The strict selector retains 216 successful phase--depth rows: 132 from the
MacBook Pro M4 Max and 84 from the RTX 5080. After the MacBook calibration probes are excluded,
the scored cohort contains 99 decode and 99 prefill rows from 33 host--file
configurations and 21 unique GGUF files. RTX contributes 14 measured files,
split into 12 training and two held-out configurations.

| Host | Decode B2 train/test MAPE | Prefill P2 train/test MAPE |
|---|---:|---:|
| MacBook Pro M4 Max | 10.39% / 13.11% | 4.13% / 18.68% |
| NVIDIA RTX 5080 | 10.12% / 36.15% | 5.86% / 108.18% |

The table reports target-host fits. In a separate B2 median-ratio
leave-one-host-out test, MAPE over all target rows is 20.82% on the Mac and
21.69% on RTX, compared with target-fitted all-row MAPE of 10.96%/13.84%
(median transfer APE 15.41%/18.75%, maximum 75.81%/71.53%). Restricted to
the fixed held-out rows, transfer MAPE is 13.94%/36.70% with medians
8.63%/26.18%, close to the target-fitted 13.11%/36.15%. B2 transfer is thus
useful in this narrow, all-Q4 test; the evidence does not show target-host
adaptation is necessary or establish a universal hardware-independent
predictor. The much larger 57.1%/116.8% errors belong to a rejected two-term
output-projection extension, not B2. The simple prefill equation is unsupported
on RTX under the measured protocol.

`results/host_transfer.csv` exports the all-target B2, target-test B2, and
rejected two-term transfer rows used for these comparisons.

The RTX measurements requested `n_gpu_layers=99`, but physical residency was
not instrumented. They must not be described as proven fully resident. No
partial-offload sweep was collected, so the repository makes no offload-cliff
claim.

For the exact cohort and caveats, see [`results/STATUS.md`](results/STATUS.md).
For restart state and local tool paths, see
[`SESSION_CHECKPOINT.md`](SESSION_CHECKPOINT.md).

## Paper status

The generated paper, `GGUF-METADATA PREDICTION OF SINGLE-SEQUENCE LLAMA.CPP THROUGHPUT ACROSS TWO SYSTEMS`, is
[`paper/icassp2027/gguf-throughput-icassp2027.pdf`](paper/icassp2027/gguf-throughput-icassp2027.pdf),
currently five pages: pages 1--4 contain the technical paper and page 5 contains
references only, as ICASSP permits. The companion supplement is 16 pages. The
only non-computational submission blocker is the placeholder author name,
affiliation, and email in `main.tex`; ICASSP 2027 is non-blind.

The Markdown files under `paper/` are historical working notes. The ICASSP
LaTeX source and generated result tables are the submission truth.

## Environment

- Python 3.9+
- `llama-bench` from `llama.cpp`
- macOS/Metal or Windows/NVIDIA CUDA for measurement
- Tectonic or a conventional LaTeX/BibTeX toolchain for the paper

Create the Python environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For RTX calibration, install a CUDA-enabled PyTorch build. The completed RTX
run used PyTorch 2.11.0+cu128 and the executable in the managed
`llama-b10794` directory; its version query did not independently report a
runtime revision.

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Reproduce the checked results

No additional model download is needed to regenerate the current analysis.

```powershell
.\.venv\Scripts\python.exe -m llmperf.analyze
.\.venv\Scripts\python.exe -m llmperf.refine
.\.venv\Scripts\python.exe -m llmperf.figures
```

Run the validation suite before rebuilding the paper:

```powershell
.\.venv\Scripts\python.exe -m llmperf.doctor --quick
.\.venv\Scripts\python.exe -m llmperf.common
.\.venv\Scripts\python.exe -m llmperf.analyze --selfcheck
.\.venv\Scripts\python.exe -m llmperf.sweep --selfcheck
.\.venv\Scripts\python.exe -m llmperf.refine --selfcheck
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

Paper build commands are documented in
[`paper/icassp2027/README.md`](paper/icassp2027/README.md).

## Collect data on another host

Only use this path to extend the study. Give every machine a stable host name,
run the environment check, calibrate it independently, and append measurements
to a host-specific CSV.

For the planned 128-GB Mac Studio M4 Max, use the stable host ID
`mac-studio-m4-max`. Its outputs must remain separate from the existing
64-GB MacBook data in `measurements_lun-mac.csv`.

On macOS:

```bash
export LLMPERF_HOST=mac-studio-m4-max
export LLAMA_BENCH="$(command -v llama-bench)"

python -m llmperf.doctor
python -m llmperf.calibrate
python -m llmperf.sweep --dry-run --repetitions 5 --settle 45 --depths 0 4096 16384
python -m llmperf.sweep --repetitions 5 --settle 45 --depths 0 4096 16384
```

This writes `calibration_mac-studio-m4-max.json`,
`env_mac-studio-m4-max.json`, and `measurements_mac-studio-m4-max.csv` under
`results/`. Prefer copying the exact GGUF cohort from the MacBook; if that is
not possible, use `python -m llmperf.fetch --set all` and verify the frozen
manifest before measuring.

```powershell
$env:LLMPERF_HOST = 'new-host-name'
$env:LLAMA_BENCH = 'C:\path\to\llama-bench.exe'

.\.venv\Scripts\python.exe -m llmperf.doctor
.\.venv\Scripts\python.exe -m llmperf.calibrate
.\.venv\Scripts\python.exe -m llmperf.sweep --dry-run
.\.venv\Scripts\python.exe -m llmperf.sweep --repetitions 5 --settle 45 --depths 0 4096 16384
```

If a clean machine lacks the frozen GGUF files, inspect the declared set before
fetching it:

```powershell
.\.venv\Scripts\python.exe -m llmperf.fetch --set all --dry-run
```

Only one fetch or measurement campaign may run at a time. The campaign lock at
`results/campaign.lock.json` prevents simultaneous I/O and benchmark activity;
do not delete a live lock to force a second process through it.

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

The decode gate worked on RTX: its 42 selected decode rows have at most 1.652%
within-cell CV. Prefill was not gated, and 29 of 42 RTX prefill rows exceed 3%
CV. Those noisy rows are retained and reported rather than silently filtered.

## Important outputs

```text
models/manifest.json                    live fixed split
results/model_manifest.json             frozen split provenance
results/model_metadata.json             frozen GGUF-derived metadata
results/calibration_<host>.json          per-host calibration
results/measurements_<host>.csv          append-only raw measurements
results/error_table_by_host.csv          host-separated decode errors
results/error_table_prefill_by_host.csv  host-separated prefill errors
results/host_transfer.csv                B2 and rejected two-term host transfer
results/predictions.csv                  row-level decode predictions
results/predictions_prefill.csv          row-level prefill predictions
figures/paper/                            generated publication figures
paper/icassp2027/                         submission source and PDFs
```

Never substitute estimated values for missing experiments. Keep partial-offload
rows separate from the `n_gpu_layers=99` baseline, and do not infer physical GPU
residency from the requested layer count alone.
