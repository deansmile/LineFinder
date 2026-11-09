# Does the timing harness change any output?

The timed passes (`timed/`) were compared with the unmodified pipeline scripts run on the same node (`official/`). Full results: `compare_outputs.json`.

| Output | Result |
|---|---|
| JointBDOE boxes and orientations (854 files) | identical (max difference 0) |
| DepthPro focal lengths (427) | identical |
| DepthPro depth maps (427) | 421 bit-identical; 6 differ on ~3.7% of pixels, mean relative difference at most 5e-5 (worst pixel 0.91 m at ~10.5 m, IMG_6771) |
| LineFinder count, visibility, facing, start/end CSVs (timed logic on the official inputs) | identical |
| LineFinder outputs recomputed from the *timed* depth maps | all decisions identical; only the float alignment score in `start_end_selection.csv` differs in 3 images (IMG_5968, IMG_5972, IMG_6778) |

The 6 depth-map differences come from half-precision GPU kernels not being bit-reproducible between runs, not from the timing code; they don't change any LineFinder answer. The description pass made one set of API calls, which are the official descriptions, so there is no second run to compare.
