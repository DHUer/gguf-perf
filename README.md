# Local LLM inference performance study

This repository measures and models `llama.cpp` inference on consumer
hardware. It contains a reproducible benchmark pipeline, a fixed train/test
model split, analysis code, generated results, and the paper draft.

The central hypothesis is a calibrated two-regime model:

```text
decode:  tok/s = eta_d * bandwidth / (active weight bytes + KV bytes)
prefill: tok/s = eta_p * FLOPS / (2 * active parameters)
```

For Mixture-of-Experts (MoE) models, storage depends on total parameters while
decode traffic is approximated from activated parameters. This distinction is
the strongest result so far: on the current held-out data it reduces decode
MAPE from 70.9% to 31.0%.

## Current status

The predictor is **not yet validated for publication**. Current evidence is
from one MacBook Pro M4 Max only, and decode efficiency varies substantially
between models within the same quantisation format. Several fitted efficiencies
also exceed the physical roofline because the present calibration reference is
not the machine's true bandwidth ceiling.

Verified current coverage (2026-08-30):

- 14 of 23 model files downloaded (105.1 GB complete; full manifest 336.4 GB)
- 84 clean CSV rows from 14 files; analysis uses 39 decode measurements from
  13 models after excluding the calibration probe
- one host, including two MoE models
- last clean held-out comparison: B1 70.9% versus B2 31.0%; regenerate all
  calibration-derived outputs after resolving the live incident in `HANDOFF.md`
- between-run decode CV: 1.61% on an idle machine with 45 s settling, versus
  15-26% under background load

See [HANDOFF.md](HANDOFF.md) for decisions, retracted results, current running
state, and the next experiments. See [paper/paper.md](paper/paper.md) for the
manuscript and [paper/related_work.md](paper/related_work.md) for the literature
audit.

## Requirements

- Python 3.9+
- `llama-bench` from `llama.cpp`
- about 337 GB for the full model manifest, plus room for results
- macOS on Apple Silicon/Metal or Windows with an NVIDIA/CUDA build

Install the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install `llama-bench`:

- macOS: `brew install llama.cpp`
- Windows: download a release from
  <https://github.com/ggml-org/llama.cpp/releases> and add it to `PATH`.
  The RTX 5080 requires a CUDA 12.8+ build for `sm_120`.
- custom build: set `LLAMA_BENCH` to the executable path.

Set `LLMPERF_HOST` to a stable, unique machine name before collecting results.
Set `HF_TOKEN` to avoid severe unauthenticated Hugging Face rate limits. On the
CUDA host, install a CUDA 12.8 PyTorch build so calibration measures the device
rather than falling back to a CPU matrix multiply:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Run the campaign

Run the environment check first on every machine:

```bash
python -m llmperf.doctor
```

Preview the model set before downloading it:

```bash
python -m llmperf.fetch --set all --dry-run
```

The recommended entry point alternates one download with an idle measurement
sweep, then reruns analysis. It is resumable and writes `results/STATUS.md` when
it finishes:

```bash
# Apple Silicon
python -m llmperf.campaign

# Discrete GPU: include the partial-offload sweep
python -m llmperf.campaign --ngl 0 8 16 24 32 48 99
```

Use `--max-file-gb N` if a host cannot load the largest models.

**Only one campaign may run at a time.** It holds a heartbeat lock at
`results/campaign.lock.json`, and a second instance refuses to start: two
concurrent sweeps contaminate every timing number (~20% throughput change under
background I/O) and two concurrent fetches corrupt the shared `.part` file.
Do not delete the lock to force a second instance past it — that is exactly how
this project produced a 19 GB partial of a 17 GB file. Stale locks from dead
processes are cleared automatically after three hours.

The old `run_campaign.sh` has been removed. It used a different lock format from
`llmperf.campaign`, so the two did not exclude each other; a single Python
driver is the fix.

## Run stages manually

The campaign driver performs these stages in order:

```bash
# 1. Download. The split is written to models/manifest.json at this point.
python -m llmperf.fetch --set all

# 2. Calibrate bandwidth and compute.
python -m llmperf.calibrate

# 3. Measure. Resume skips completed phase/depth cells.
python -m llmperf.sweep --dry-run
python -m llmperf.sweep --repetitions 5 --settle 45 --depths 0 4096 16384

# On the discrete-GPU host, trace the offload cliff.
python -m llmperf.sweep --ngl 0 8 16 24 32 48 99 \
  --repetitions 5 --settle 45 --depths 0 4096 16384

# 4. Fit, validate, diagnose, and generate figures.
python -m llmperf.analyze
python -m llmperf.refine
```

For a separate repeatability experiment:

```bash
python -m llmperf.repeatability --runs 5 --settle 45
```

## Measurement controls

| Setting | Value | Reason |
|---|---|---|
| flash attention | `-fa on` | Avoids backend/model-dependent `auto` selection. |
| KV cache | `-ctk f16 -ctv f16` | Fixes the KV bytes in the model denominator. |
| threads | physical performance cores | Avoids CPU oversubscription. |
| repetitions | 5 | Records a mean and within-run standard deviation. |
| settling | 45 s for campaign data | Reduces thermal and session drift. |
| isolation | one process per model/offload cell | Contains OOMs and memory fragmentation. |
| resume key | model, offload, phase, depth | Correctly matches llama-bench's separate prefill/decode rows. |

The sweep is append-only. Do not merge data from an uncontrolled or concurrently
loaded session into the clean CSV. Known contaminated data is retained under
`results/contaminated/` as methodological evidence, not as input to analysis.

## Calibration caveat

The pipeline records three possible bandwidth sources:

- `cpu_triad`: CPU STREAM-style bandwidth
- `torch_gpu`: device bandwidth when CUDA PyTorch is available
- `llm_ref`: effective bandwidth back-solved from a reference model

On a clean M4 Max run, the CPU triad measured 67.9 GB/s while the GPU-backed
reference reached 313.8 GB/s through unified memory. The CPU result is therefore
not a valid Metal bandwidth ceiling. The current calibration JSON was overwritten
during the duplicate-download incident described in `HANDOFF.md`; rerun calibration
before treating generated analysis outputs as authoritative. `analyze.py` selects
the platform-appropriate source and excludes the reference model from error metrics
because its efficiency is 1.0 by construction.

## Outputs

```text
models/manifest.json       fixed train/test assignments
results/calibration_*.json per-host calibration
results/measurements_*.csv append-only measurements
results/error_table.csv    aggregate prediction errors
results/predictions.csv    row-level predictions
results/STATUS.md          generated campaign summary
figures/                   generated plots
paper/                     manuscript and literature audit
```

Each measurement row records the host, platform, protocol, model metadata,
`llama-bench` identification, and Git commit when available (`nogit` when the
working copy has no Git metadata).

## Self-checks

```bash
python -m llmperf.doctor --quick
python -m llmperf.common
python -m llmperf.sweep --selfcheck
python -m llmperf.refine --selfcheck
```

Never substitute estimated values for missing experiments. The manuscript marks
unmeasured results `PENDING` until the corresponding hardware run exists.
