# ICASSP 2027 paper

`gguf-throughput-icassp2027.pdf` is the generated ICASSP paper,
`GGUF-METADATA PREDICTION OF SINGLE-SEQUENCE LLAMA.CPP THROUGHPUT ACROSS TWO SYSTEMS`. It is five pages: pages 1--4 contain the technical paper and page
5 contains references only, using ICASSP's permitted fifth-page allowance.
It uses the supplied `spconf.sty` and `IEEEbib.bst` under
`.submission_reqiurement/`.

`gguf-throughput-supplement.pdf` is the generated 16-page companion appendix.
It contains the row-level decode and prefill results for both hosts, protocol
diagnostics, calibration details, negative results, limitations, and source
audit.

The manuscript's source of truth is `main.tex`, together with the generated
CSV tables under `../../results/`. Historical Markdown drafts elsewhere under
`paper/` contain superseded Mac-only analyses.

## Submission blocker

Before submission, replace the placeholder author name, affiliation, and email
near the top of `main.tex`. ICASSP 2027 explicitly requires a non-blind paper.
Do not fabricate the identity. This is the only non-computational submission
blocker in the current build.

## Data represented in the paper

- 216 selected rows: 132 from the 64-GB MacBook Pro M4 Max and 84 from RTX 5080
- 99 scored decode and 99 scored prefill rows
- 33 scored host--file configurations representing 21 unique GGUF files
- RTX cohort: 14 measured/scored files, split 12 train and two held out
- Decode B2 train/test MAPE: 10.39%/13.11% on Mac and 10.12%/36.15% on RTX
- Prefill P2 depth-zero train/test MAPE: 4.13%/18.68% on Mac and
  5.86%/108.18% on RTX
- B2 median-ratio leave-one-host-out MAPE: 20.82% on Mac and 21.69% on RTX,
  versus target-fitted all-row MAPE of 10.96%/13.84% (median transfer APE
  15.41%/18.75%, maximum 75.81%/71.53%)
- B2 transfer on target test rows: 13.94% on Mac and 36.70% on RTX (median
  8.63%/26.18%), close to target-fitted 13.11%/36.15%; all test files are Q4
- Rejected two-term output-projection transfer MAPE: 57.1% on Mac and 116.8%
  on RTX; these are not B2 values

The generated `../../results/host_transfer.csv` exports the all-target B2,
target-test B2, and rejected two-term absolute-time rows behind these numbers.

The manuscript does not claim physical RTX residency or an offload curve.
Every RTX row requested `n_gpu_layers=99`, but no VRAM-spill telemetry was
recorded and no lower-`ngl` sweep was collected. Prefill noise is also stated:
29 of 42 RTX prefill rows exceed 3% CV, while RTX decode has a 1.652% maximum.

## Regenerate analysis and figures

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m llmperf.analyze
.\.venv\Scripts\python.exe -m llmperf.refine
.\.venv\Scripts\python.exe -m llmperf.figures
```

No model download is needed for the current paper analysis.

## Build with the local Tectonic binary

From this directory:

```powershell
$tectonic = 'C:\Users\lun\AppData\Local\gguf-perf\tectonic-0.17.0\tectonic.exe'
New-Item -ItemType Directory -Force build, supplement-build | Out-Null

& $tectonic --keep-logs --keep-intermediates --outdir build main.tex
Copy-Item build\main.pdf gguf-throughput-icassp2027.pdf

..\..\.venv\Scripts\python.exe generate_supplement.py
& $tectonic --keep-logs --keep-intermediates --outdir supplement-build supplement.tex
Copy-Item supplement-build\supplement.pdf gguf-throughput-supplement.pdf
```

If `tectonic` is on `PATH`, the executable variable can be replaced with the
plain command. A conventional LaTeX/BibTeX sequence also works:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Pre-submission checks

1. Insert the verified author identity and rebuild both PDFs.
2. Confirm pages 1--4 contain technical content, page 5 contains references
   only, and the document uses Letter or A4.
3. Confirm all fonts are embedded and the PDF is unencrypted.
4. Confirm references are present and every numeric claim agrees with
   `error_table_by_host.csv` and `error_table_prefill_by_host.csv`.
5. Run the full test suite and `git diff --check` from the repository root.
6. Run `python paper/icassp2027/audit_submission.py`; before author details are
   inserted, its sole expected blocker is `author identity`.
7. Keep host-separated results; do not replace them with a pooled headline.

The supplement is supporting material and does not relax the main-paper page
limit. The paper reports host-adapted fits but does not claim they are necessary:
B2 transfer nearly matches them on the narrow held-out Q4 cohort. Long tails,
two machines, and only two RTX test files prevent a universal transfer claim.
