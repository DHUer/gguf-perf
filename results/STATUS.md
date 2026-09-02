# Campaign status

Generated 2026-08-30 21:28:49 on `lun-mac` (Darwin-arm64) by `llmperf.campaign`.
Regenerated on every campaign completion — do not hand-edit.

## Coverage

- models on disk: 23
- models still to fetch: 0
- measurement rows (this host): 133
- hosts with results: 1 (lun-mac)

## Prediction error

```
=== prediction error ===
                              model split  n  MAPE_%  median_APE_%  p90_APE_%  max_APE_%
     B0 uncalibrated (eta=1, total) train 51    40.2          29.7       85.0      115.0
     B0 uncalibrated (eta=1, total)  test 12    62.7          82.8       89.1       91.7
        B1 calibrated, total params train 51    56.7          11.1      206.7      624.0
        B1 calibrated, total params  test 12    64.7          58.7       74.3      242.0
B2 calibrated, active params (ours) train 51    18.9           6.5       62.7       96.1
B2 calibrated, active params (ours)  test 12    33.9          14.0       81.2       82.6
```

## Cross-model eta spread (go/no-go, paper section 5.7)

```
(not produced)
```

## Next

**Only one host has results.** Leave-one-machine-out is undefined and no cross-architecture claim is admissible. Run the same commands on the other machines — see HANDOFF.md, "Next, in priority order".
