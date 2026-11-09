# Paper tables on the corrected 427-image GT (seeded rerun)

2026-11-09. Results folder: `results/2026-11-09/`, in the private HF dataset `deansmile123/LineFinder` (main) and on Torch. The commands are in `REPRODUCE.md`.

The paper text was not edited.

## Summary

- **Table 1:** LineFinder is still best on every row. Its overall accuracy is 65.74%, against the 65.67% given in the text; the table's 0.6283 is a typo.
- **Table 2:** with the paper's scoring, LineFinder gets 0.5328 overall (paper 0.5399). LineFinder was judged with a more lenient Gemini prompt than the VLMs. With one strict prompt for all four methods it drops to 0.3759, which is still the best. Kimi's recomputed value (0.2576) is well above the paper's 0.1957, and I couldn't reconcile them.
- **Table 3:** LineFinder's unreproducible counts (MAE 4.26, RMSE 10.52) are replaced by the rerun's MAE 3.84 and RMSE 7.43. LineFinder now has the lowest MAE and relative error, while GPT has the lowest RMSE. The VLM rows change only slightly.
- **Large count errors:** 36 of the 40 errors above 10 are mainly **missed detections**, not RANSAC rejections.
- **Facing:** Luna and Olivia's GT does distinguish sideways-left from sideways-right. The VLMs were scored without merging them on those images, while LineFinder was scored with the merge.
- **Timing:** LineFinder takes 1.37 s per image on average (0.38 s without the GPT-5.4 mini description call). The API baselines take 1.48 s (GPT-5.4 mini), 8.83 s (Kimi k2.6, standing in for K2.5) and 16.72 s (Gemini 3 Flash).

---

## Comparison with the paper (best new value per row in **bold**)

The new values use the corrected GT (427 images), LineFinder's seeded rerun and the saved VLM predictions. Table 1's new facing values use the sideways merge for all methods; see section 3 for the other variants.

### Table 1: line understanding (proportion correct)

| Row | Gemini paper → new | GPT paper → new | Kimi paper → new | LineFinder paper → new |
|---|---|---|---|---|
| Line facing direction | 0.5672 → 0.6628 | 0.3225 → 0.3536 | 0.3449 → 0.3607 | 0.8289 → **0.8478** |
| End of line visible | 0.3727 → 0.3794 | 0.3180 → 0.3255 | 0.4051 → 0.4075 | 0.6995 → **0.7002** |
| End location | 0.1690 → 0.1710 | 0.1381 → 0.1382 | 0.3171 → 0.3162 | 0.6549 → **0.6534** |
| Direction to turn (end) | 0.1088 → 0.1077 | 0.2947 → 0.3091 | 0.3473 → 0.3536 | 0.6502 → **0.6511** |
| Start of line visible | 0.5024 → 0.5059 | 0.4554 → 0.4684 | 0.5741 → 0.5761 | 0.6408 → **0.6323** |
| Start location | 0.1967 → 0.1991 | 0.2204 → 0.2225 | 0.4421 → 0.4450 | 0.5282 → **0.5293** |
| Direction to turn (start) | 0.2338 → 0.2365 | 0.4456 → 0.4543 | 0.4954 → 0.4965 | 0.5942 → **0.5878** |
| Overall (mean of 7) | 0.3072 → 0.3232 | 0.3135 → 0.3245 | 0.4180 → 0.4222 | 0.6283* → **0.6574** |

\* Typo in the paper's table; the mean of its seven values is 0.6567, matching the text's 65.67%.

### Table 2: person description (proportion correct, Gemini 3 Flash judge)

| Row | Gemini paper → new | GPT paper → new | Kimi paper → new | LineFinder paper → new (paper scoring) | LineFinder, same scoring as VLMs |
|---|---|---|---|---|---|
| Start person | 0.1527 → 0.1475 | 0.1666 → 0.1733 | 0.2000 → 0.2834 | 0.4601 → **0.4496** | 0.3091 |
| End person | 0.1155 → 0.1311 | 0.1343 → 0.1405 | 0.1913 → 0.2319 | 0.6197 → **0.6159** | 0.4426 |
| Overall | 0.1341 → 0.1393 | 0.1505 → 0.1569 | 0.1957 → 0.2576 | 0.5399 → **0.5328** | 0.3759 |

LineFinder is best in every row under either scoring.

### Table 3: counting

| Row | Gemini paper → new | GPT paper → new | Kimi paper → new | LineFinder paper → new |
|---|---|---|---|---|
| Images | 431 → 426 | 432 → 427 | 432 → 427 | 432 → 427 |
| MAE ↓ | 5.61 → 5.62 | 3.89 → 3.89 | 4.49 → 4.50 | 4.26 → **3.84** |
| RMSE ↓ | 13.72 → 13.55 | 7.14 → **7.13** | 11.16 → 11.10 | 10.52 → 7.43 |
| Mean signed error (→0) | 3.81 → 3.87 | 0.56 → **0.52** | 2.64 → 2.65 | 0.00 → −2.86 |
| Mean relative error ↓ | 0.426 → 0.426 | 0.320 → 0.320 | 0.345 → 0.345 | 0.285 → **0.252** |
| Exact match ↑ | 0.269 → **0.275** | 0.181 → 0.180 | 0.245 → 0.246 | 0.236 → 0.225 |
| Within ±1 ↑ | 0.464 → 0.472 | 0.426 → 0.424 | 0.505 → **0.504** | 0.516 → 0.468 |
| Within ±2 ↑ | 0.599 → 0.603 | 0.563 → 0.560 | 0.627 → **0.628** | 0.635 → 0.609 |
| Undercount rate ↓ | 0.292 → 0.284 | 0.354 → 0.356 | 0.273 → **0.272** | 0.420 → 0.553 |
| Overcount rate ↓ | 0.438 → 0.441 | 0.465 → 0.464 | 0.481 → 0.482 | 0.370 → **0.222** |

For the under- and overcount rows, "best" means lowest. The paper bolds by its own convention, so check these two rows.

### Table 4: qualitative examples

The seeded rerun gives exactly the paper's text for every candidate image.

| Paper example | Candidate images | Rerun output | Changed? |
|---|---|---|---|
| 9 people, facing away, neither end visible, turn left (start) / right (end) | IMG_6736, outdoor_140 | identical for both | no |
| 8 people, facing away, neither end visible, turn left (start) / right (end) | IMG_6732, supermarket_033 | identical for both | no |
| 6 people, facing right, start visible center right, end visible center left | outdoor_034 | identical | no |

Two different images match each of the first two descriptions; the figure shows which one the paper used.

---

## 1. Seed and rerun

- **Seed:** `ransac_line_3d` (`visual_3D.py`) now takes `seed=42` by default, and every call site uses it. It was the only unseeded random step; `top_down_direction.py` was already seeded with 42.
- **Rerun:** the full pipeline ran on all 427 images: JointBDOE, DepthPro, count, start/end selection, visibility and facing. The outputs are in `results/2026-11-09/official/`.
- **Change versus the saved, unseeded outputs:** the count changed for 19 images, facing for 9, and at least one visibility field for 15.
- **Start/end selection:** `identify_queue_start_end_batch.py` still contains an older selection algorithm. The rerun uses `identify_queue_start_end.py`, which the visibility and description scripts import.

## 2. Table 2 judgments

- **The judgments are automated, not manual.** Gemini 3 Flash (`gemini-3-flash-preview`) decides whether a predicted description and a GT description describe the same person. The earlier judgment files (`same_person_eval_*.jsonl`) cover only the 109 older images, and almost all of their entries are empty.
- **All 1,688 judgments from this run** are saved in `results/2026-11-09/table2/judgments_gemini3flash.jsonl`, with 0 errors. **None of the 427 images lacks a judgment or a prediction**, for any method.
- **The scoring differs between the methods**, in two ways:
  - **Judge prompt:** the VLM scripts use a strict prompt ("be conservative: if uncertain, return false", temperature 0). LineFinder's script uses a lenient one ("true if likely the same person", default temperature).
  - **Rule:** LineFinder's rule first checks that the visibility matches; the VLM rule looks only at whether descriptions are present.
  - The "same scoring as VLMs" column applies the strict prompt and the presence rule to LineFinder too.
- **Check against the paper (pre-correction GT, 433 images):** GPT gives 0.1559 against the paper's 0.1505, and Gemini 0.1397 against 0.1341.
  - The per-group values match the logged results within about 1–2 points: GPT Luna/Olivia 13.89% vs 13.73% logged, Gemini 11.27% vs 11.27%.
  - Kimi's older-image value matches (16.51% vs 16.20% logged). Its Luna/Olivia value (29.0%) can't be checked, because it was never logged; the paper's 0.1957 implies about 20.7%.
- **LineFinder's description model:** the code calls GPT-5.4 mini (`gpt-5.4-mini-2026-03-17`), not Gemini 3 Flash.

## 3. Facing labels

**Label set in each GT group** (427-image set):

| Group | Labels (count) |
|---|---|
| Older format (109) | Facing away 49, Facing sideways 29, Facing towards 29, other 2 |
| Luna (179) | away 62, sideways-left 49, sideways-right 49, towards 19 |
| Olivia (139) | away 62, sideways-left 23, sideways-right 24, towards 30 |

- **Luna and Olivia distinguish sideways-left from sideways-right; the older format doesn't.**
- **The scoring was not the same for all methods:**
  - VLMs on Luna/Olivia: exact match, no merge.
  - Gemini and Kimi on the older images: merged. GPT's older-image predictions use the same "Facing …" labels as the GT.
  - LineFinder: merged everywhere. `GT_json_canonical.csv` collapses left/right.

| Facing accuracy, 427 images | Gemini | GPT | Kimi | LineFinder |
|---|---:|---:|---:|---:|
| As the paper scored it | 0.5738 | 0.3255 | 0.3513 | 0.8478 |
| Merged for everyone | 0.6628 | 0.3536 | 0.3607 | **0.8478** |
| No merge for anyone | 0.5105 | 0.3255 | 0.3255 | **0.7752** |
| Overall (7 fields), no merge | 0.3014 | 0.3205 | 0.4172 | **0.6470** |

## 4. Count errors larger than 10

There are 40 such images, and together they account for **83.4%** of LineFinder's total squared error. Each image's shortfall splits into a detection miss (GT − detected boxes) and a RANSAC rejection (boxes − inliers):
- **36 are mainly detection misses:** JointBDOE finds 10–26 boxes where the GT has 18–70 people.
- **4 are mainly RANSAC rejections:** outdoor_086, IMG_6773, themepark_010, themepark_103.
- **By scene:** theme park 13, bank 10, outdoor 8, online 4, airport 4, supermarket 1.

The full table is in `results/2026-11-09/step4_large_errors.md`, and per-image values for all 427 images are in `linefinder_count_errors.csv`. Two of the worst cases:

| Image | Scene | GT | Boxes | Inliers | Error | Mainly | Share of squared error |
|---|---|---:|---:|---:|---:|---|---:|
| IMG_6787 | bank | 70 | 15 | 14 | −56 | detection miss | 13.3% |
| IMG_6782 | bank | 57 | 16 | 11 | −46 | detection miss | 9.0% |

(This corrects `paper_check_report.md`, which guessed that RANSAC was the cause.)

## 5–8. Reproduction and timing

- **Reproduction:** `REPRODUCE.md` lists every command.
- **Timing:** the method, hardware and the full table are in `timing_summary.md`.
  - The run was on one A100-80GB with the GPU allocated to the job alone; whole-node allocation isn't allowed on this cluster.
  - Timing changes no output. Boxes and all four LineFinder CSVs are identical. 6 of 427 depth maps show tiny fp16 differences, and those don't change any LineFinder decision.
  - There is no separate YOLO stage, because JointBDOE detects and estimates orientation in one network.
  - The VLM timing used 100 images (seed 42) with no failed calls. Kimi is k2.6 standing in for K2.5, which is no longer served, at temperature 0.6, the only value it accepts.

## Caveats for the paper

- **GPT's older-image results come from GPT-4o.** GPT's results on the 109 older images (in Tables 1–3) come from `gpt_line.py` with **GPT-4o** and an older 9-question prompt, not GPT-5.4 mini. They were kept by decision.
- **Kimi K2.5 can't be re-run:** it is no longer available, so step 7 used k2.6.
