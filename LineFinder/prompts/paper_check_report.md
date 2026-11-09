# LineFinder: checking the paper's numbers against the merged code and data

2026-11-09

## Summary

The project existed in two copies: one on the Torch cluster and one on the Shannon NAS machine. I merged them into a single repository and re-checked the paper's Table 1 (line visibility) and Table 3 (people counting) against it. The headline findings:

1. **The paper's tables were computed on the older ground truth.** The merged data includes 6 annotation corrections, which change the evaluation set from 432 images to 427. The paper's dataset statistics (517 images, 90 too hard, 427 evaluated) already use the corrected annotations, but Tables 1 and 3 use the older ones.
2. **Table 1 reproduces.** Its overall accuracy (65.67%) comes out exactly on the older ground truth. Its line-facing value (82.89%) is two images off what the same data gives (83.33%). It is also inconsistent with the overall value.
3. **Table 3 reproduces for the three VLMs but not for LineFinder.** GPT, Gemini and Kimi match on every metric except RMSE, which is slightly lower than reported. LineFinder's reported MAE 4.26 and RMSE 10.52 can't be produced from any prediction file in either copy. The current files give MAE 3.84 and RMSE 7.42.
4. **The visibility rule described in the paper is in the code.**
5. **One image, IMG_6728, had never been processed.** It now has a LineFinder count (6, which matches the ground truth), so all 427 images are covered.

The paper has not been edited. Recomputed values are in the table at the end.

---

## 1. Which counter produced `OLD_number_of_people.csv`

LineFinder's people counts come from two CSV files:
- `OL_number_of_people.csv`: the 408 newer images (outdoor, supermarket, theme park).
- `OLD_number_of_people.csv`: the 109 older images (airport, supermarket, bank, online).

**`OLD_number_of_people.csv` was produced by the RANSAC-inlier counter, not by counting detected boxes.** The evidence:

- **Script:** only `code/batch_number_of_people.py` writes a file with this name. It fits a 3D line to the detected people with RANSAC (1000 iterations, 0.8 m inlier distance, at least 30% inliers) and reports the number of inliers.
- **Columns:** the header, `image_path,image_id,OL_number_of_people,status`, is exactly the one this script writes.
- **Values:** all 108 counts are at or below the number of detected boxes for that image, and only 24 equal it. A box counter would report the box count every time.
- **History:** the file exists only in the Shannon snapshot. There is no earlier version to compare against.

The script's image list leaves out IMG_6728, which is why that image had no count (see section 4).

---

## 2. Line visibility (Table 1)

### What the code does

`code/batch_line_visibility_nogt.py` writes `line_visibility_new.csv`. 

**The rule described in the paper is implemented.** The code extends the fitted 3D line by the median spacing between people and projects the point into the image. It then marks the endpoint not visible if the point falls outside the image, the box touches the edge, or the depth there is more than 0.5 m closer.

  
### Reproducing Table 1

- **Inputs:** visibility and location come from `line_visibility_new.csv`. Facing direction comes from the two `queue_facing_results` files.
- **Label mapping:** the predictions say "sideways-left" or "sideways-right", while the ground truth just says "sideways". Both were counted as "sideways"; without that, facing accuracy drops to 49%.
- **Overall:** on the older ground truth (432 images), overall accuracy across the 7 fields is **65.67%, matching the paper exactly**.
- **Facing:** the same data gives **83.33%** (360 of 432), against the paper's 82.89% (358 of 432). The paper's two numbers can't both be right. With facing at 82.89%, the overall would be about 65.61%.

---

## 3. Which files produced Table 3

Each method's Table 3 numbers come from pooling two evaluations: one over the newer images (Luna and Olivia annotations) and one over the older images.

| Method | Prediction files | Images (older ground truth) |
|---|---|---|
| GPT | `gpt_line_luna_olivia.jsonl`, `gpt_line_old.jsonl` | 324 + 108 = 432 |
| Gemini | `gemini_line_luna_olivia.jsonl`, `gemini_line_old.jsonl` | 323 + 108 = 431 |
| Kimi | `kimi_line_luna_olivia.jsonl`, `kimi_line_old.jsonl` | 324 + 108 = 432 |
| LineFinder | recorded result only (`code/results_number_people.txt`) | 432 |

The scoring scripts are in `code/evaluation/eval_number_of_people*.py`.

- **Which ground truth:** the image counts identify the older ground truth. It marks 84 images too hard, so 517 − 84 = 433 are evaluated. IMG_6728 drops out because of a file-naming mismatch, leaving 432.
- **VLMs:** for all three, MAE, mean signed error, mean relative error, exact match, within ±1, within ±2 and the under- and overcount rates reproduce to about three decimals.
- **RMSE:** comes out 0.01 to 0.24 lower than reported for every VLM (for example, GPT 7.13 vs 7.14, Gemini 13.49 vs 13.72). Pooling all images together gives the closest match, but I couldn't find a method that reproduces RMSE exactly.
- **Gemini:** gives no usable count for outdoor_050, which is why it has one fewer image than the other methods.
- **LineFinder:** the reported MAE 4.26, RMSE 10.52 and mean signed error 0.00 can't be reproduced from any file in either copy. Every variant I tried lands between MAE 3.6 and 3.9 and RMSE 6.8 and 7.4:
  - the current RANSAC counts,
  - the earlier box counts,
  - box counts for every image,
  - with and without the too-hard images,
  - on both versions of the ground truth.

  The predictions behind the reported numbers appear not to have been kept.

---

## 4. Running LineFinder on IMG_6728

- **Why it was missing:** IMG_6728 (supermarket) is in the ground truth but had never been processed. It had no depth map and no detected people. Its annotation file is also named `IMG_6728.jpg.json`, which kept every evaluation script from matching it.
- **What was run:**
  - depth estimation (Depth Pro), giving a focal length of 634.7 px;
  - person detection with orientation (JointBDOE), using the same weights and thresholds as the main run, which found 7 people;
  - LineFinder's counting, visibility and facing steps.
- **Results:**
  - Count: **6, matching the ground truth.** GPT, Gemini and Kimi also predict 6.
  - Visibility: end of line not visible (move left); start of line visible at centre right.
  - Facing: sideways-right.

These outputs are in scratch space and have not yet been added to the data folder. The RANSAC step is not seeded, so a re-run could give a slightly different count.

---

## 5. Images where LineFinder's count is off by more than 10

These are the 427 images on the corrected ground truth. 39 have an error larger than 10, and every one is an undercount.

| Scene | Images |
|---|---:|
| Theme park | 13 |
| Bank | 9 |
| Outdoor | 8 |
| Online | 4 |
| Airport | 4 |
| Supermarket | 1 |

They are mostly long queues: the ground truth is 18 to 70 people, and LineFinder reports 7 to 21. This is consistent with RANSAC keeping only one straight stretch of a long or bending queue.

| Image | Scene | GT | LineFinder | Error |
|---|---|---:|---:|---:|
| IMG_6787 | bank | 70 | 14 | −56 |
| IMG_6782 | bank | 57 | 11 | −46 |
| IMG_5978 | online | 50 | 10 | −40 |
| themepark_017 | theme park | 45 | 10 | −35 |
| IMG_6704 | airport | 43 | 9 | −34 |
| IMG_5972 | online | 50 | 19 | −31 |
| IMG_6784 | bank | 47 | 19 | −28 |
| themepark_054 | theme park | 33 | 10 | −23 |
| themepark_042 | theme park | 35 | 12 | −23 |
| themepark_085 | theme park | 31 | 10 | −21 |
| outdoor_086 | outdoor | 33 | 12 | −21 |
| IMG_6777 | bank | 35 | 14 | −21 |
| IMG_5958 | online | 30 | 9 | −21 |
| themepark_098 | theme park | 35 | 15 | −20 |
| IMG_6767 | bank | 32 | 12 | −20 |
| IMG_6745 | supermarket | 28 | 10 | −18 |
| IMG_6711 | airport | 39 | 21 | −18 |
| IMG_5980 | online | 30 | 12 | −18 |
| themepark_035 | theme park | 27 | 10 | −17 |
| outdoor_070 | outdoor | 35 | 18 | −17 |
| themepark_093 | theme park | 28 | 12 | −16 |
| outdoor_130 | outdoor | 35 | 19 | −16 |
| outdoor_122 | outdoor | 30 | 14 | −16 |
| outdoor_043 | outdoor | 34 | 19 | −15 |
| themepark_032 | theme park | 27 | 13 | −14 |
| outdoor_185 | outdoor | 32 | 18 | −14 |
| outdoor_014 | outdoor | 26 | 12 | −14 |
| IMG_6786 | bank | 24 | 10 | −14 |
| IMG_6773 | bank | 29 | 15 | −14 |
| IMG_6692 | airport | 28 | 14 | −14 |
| themepark_034 | theme park | 21 | 8 | −13 |
| themepark_095 | theme park | 29 | 17 | −12 |
| themepark_010 | theme park | 23 | 11 | −12 |
| themepark_103 | theme park | 21 | 10 | −11 |
| themepark_055 | theme park | 23 | 12 | −11 |
| outdoor_006 | outdoor | 31 | 20 | −11 |
| IMG_6778 | bank | 27 | 16 | −11 |
| IMG_6766 | bank | 24 | 13 | −11 |
| IMG_6685 | airport | 18 | 7 | −11 |

IMG_5958 is listed in `code/exclude_eval.txt`, but its annotation doesn't mark it too hard, so it is included here.

---

## Annotation housekeeping

Two small fixes, applied but not yet committed:

- **`code/images_with_gt_json.txt`:** this list had 519 entries because it included two annotator-accuracy summary files as if they were images. Both entries are removed, leaving the 517 real images.
- **Luna's annotator-accuracy summary:** it still showed 16 hard images, from before the corrections. It is recomputed from the corrected annotations: 22 hard and 179 not hard, with the per-question accuracies updated. The same method reproduces both original summaries exactly from the older annotations.

---

## Old vs. recomputed values

- **Old:** the paper's values, or the values recorded in `code/results_number_people.txt` where only those were available.
- **New:** the corrected ground truth, 427 images including IMG_6728. Gemini has 426, since it gives no count for outdoor_050.
- **Blank cells:** only Table 1's overall and facing values were checked against the paper, so the other old cells are blank.

| Table | Metric | Method | Old | New |
|---|---|---|---:|---:|
| 1 | Overall (7 fields) | LineFinder | 65.67% | 65.81% |
| 1 | Line-facing direction | LineFinder | 82.89% | 83.84% |
| 1 | End visible | LineFinder | — | 70.02% |
| 1 | End location | LineFinder | — | 65.57% |
| 1 | End: direction to move | LineFinder | — | 65.11% |
| 1 | Start visible | LineFinder | — | 63.93% |
| 1 | Start location | LineFinder | — | 52.69% |
| 1 | Start: direction to move | LineFinder | — | 59.48% |
| 3 | Images | LF / GPT / Gemini / Kimi | 432 / 432 / 431 / 432 | 427 / 427 / 426 / 427 |
| 3 | MAE | LF / GPT / Gemini / Kimi | 4.26 / 3.89 / 5.61 / 4.49 | 3.84 / 3.90 / 5.62 / 4.50 |
| 3 | RMSE | LF / GPT / Gemini / Kimi | 10.52 / 7.14 / 13.72 / 11.16 | 7.42 / 7.13 / 13.55 / 11.10 |
| 3 | Mean signed error | LF / GPT / Gemini / Kimi | 0.00 / 0.56 / 3.81 / 2.64 | −2.85 / 0.52 / 3.87 / 2.65 |
| 3 | Mean relative error | LF / GPT / Gemini / Kimi | 0.285 / 0.320 / 0.426 / 0.345 | 0.253 / 0.320 / 0.426 / 0.345 |
| 3 | Exact match | LF / GPT / Gemini / Kimi | 0.236 / 0.181 / 0.269 / 0.245 | 0.225 / 0.180 / 0.275 / 0.246 |
| 3 | Within ±1 | LF / GPT / Gemini / Kimi | 0.516 / 0.426 / 0.464 / 0.505 | 0.466 / 0.424 / 0.472 / 0.504 |
| 3 | Within ±2 | LF / GPT / Gemini / Kimi | 0.635 / 0.563 / 0.599 / 0.627 | 0.609 / 0.560 / 0.603 / 0.628 |
| 3 | Undercount rate | LF / GPT / Gemini / Kimi | 0.42 / 0.354 / 0.292 / 0.273 | 0.550 / 0.356 / 0.284 / 0.272 |
| 3 | Overcount rate | LF / GPT / Gemini / Kimi | 0.37 / 0.465 / 0.438 / 0.481 | 0.225 / 0.464 / 0.441 / 0.482 |

On the corrected set, LineFinder's MAE (3.84) is level with GPT's (3.90). It has the lowest relative error of all four methods, but it undercounts in 55% of images.

---

## Open questions

1. Should Tables 1 and 3 be updated to the corrected 427-image set, so they agree with the dataset statistics?
2. Can the predictions behind LineFinder's reported 4.26 / 10.52 be recovered? If not, the recomputed values would replace them.
