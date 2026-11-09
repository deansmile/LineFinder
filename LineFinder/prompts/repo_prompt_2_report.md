# repo_prompt_2: merge and paper-check report

Working copy: Torch, `/scratch/ds5725/linefinder` (GitHub `deansmile/LineFinder`, branch `main`).
Date: 2026-11-09.

Status:
- **Code merge:** done. `main` is at `823f6ca`, a merge of Torch `70ed625` and Shannon `17a0454`.
- **GitHub:** the `shannon` branch is deleted; only `main` remains.
- **Data steps (prompt steps 4–9):** not started.
- **Uncommitted fixes:** the GT image list and the Luna summary (section 6) are changed on disk but not committed.

---

## 1. Code merge

### How Shannon was merged

Shannon's repo root is the same project as Torch's `LineFinder/` folder. A plain merge would have created a second `code/` and `GT_json/` next to `LineFinder/`. So Shannon's files were moved under `LineFinder/` before merging; `.gitignore` stayed at the repo root. The merge commit lists the original `origin/shannon` commit as its second parent, so Shannon's history is kept.

Note: the merge rewrote the conflicted Torch files on disk, so their original modification times were lost. "Newer" was judged from file content.

### Conflicts and how each was resolved

There were 565 conflicts:
- **538 differed only in file mode:** every Shannon file was marked executable (NAS filesystem); the contents matched. Torch's non-executable mode (100644) was kept, and Shannon's new files were also set to 100644.
- **27 differed in content:**

| File | Resolution |
|---|---|
| 9 `GT_json/GT_json_luna/outdoor_*.json` (004, 021, 095, 106, 111, 141, 159, 170, 174) | Shannon (GT corrections: 6 "too hard" no→yes; outdoor_021 people count 8→6; outdoor_170 and 174 endpoints changed to visible) |
| `identify_queue_start_end.py` | Shannon: newer endpoint ordering that projects the 3D queue axis into the image and orders people in 2D. Same API. |
| `identify_queue_start_end_batch.py` | Torch: the real batch runner. Shannon's was a single-image copy. |
| `batch_number_of_people.py` and `OL_number_of_people.csv` | Shannon: counts RANSAC inliers instead of all detected boxes. |
| `batch_line_visibility_nogt.py` | Shannon: horizontal-only edge rule. |
| `check_missing_start_end_descriptions.py` | Shannon: rewritten to check one combined CSV. |
| `gpt_line_luna_olivia.jsonl` | Shannon: one malformed JSONL record fixed. |
| `images_with_gt_json.txt` | Shannon: 519 lines, containing all 109 of Torch's. |
| 6 API scripts (`gpt_line*`, `describe_start_end_person_gpt5`, `qwenvl_*`, `rerun_missing_*`) | Torch: `.env`-based key loading. |
| `visualize_start_end_from_image.py` | Torch (only the default root path differed). |
| `gemini_line.py` | Shannon's logic with Torch's key loading, plus a new required `--image_set {old,new,full}` argument. Default outputs: `gemini_line_old.jsonl`, `gemini_line_luna_olivia.jsonl`, `gemini_line_full.jsonl`. |
| `openai_name_check.py` | Shannon's Kimi (Moonshot) model listing with `.env` key loading. `MOONSHOT_API_KEY` added to `.env.example`. |
| `olivia_luna_image_paths.txt` | Torch's paths with `themepark_081` removed (408 lines). |
| `.gitignore` | Torch's file plus Shannon's data folders, prefixed with `LineFinder/`. |

**Dropped:** `.gitattributes` (Hugging Face's default Git LFS file), `code/ds5725.code-workspace`, `github_push.md`.

**Added from Shannon without conflict:** 51 files, mainly the `evaluation/` scripts, the Kimi scripts and results, and helper scripts.

### LINEFINDER_ROOT

- **Where it's set:** one setting in `LineFinder/code/lf_paths.py`. It reads the `LINEFINDER_ROOT` environment variable and defaults to `/scratch/ds5725/linefinder/LineFinder`.
- **What changed:** 95 hardcoded paths across 48 Python files now use `LINEFINDER_ROOT` or `lf_path("...")`. Paths in comments became `$LINEFINDER_ROOT/...`.
- **Scripts outside `LineFinder/code`:** files in `evaluation/`, `JointBDOE/demos/image.py` and `ml-depth-pro/batched_depth.py` add `LineFinder/code` to `sys.path` before importing it.
- **Left as they were:** JointBDOE's own weights and `res_images` paths, and the paths in `upload.SBATCH` and `JointBDOE/test.SBATCH`.

### Compile check

`python -m py_compile` passes on 102 of 103 tracked `.py` files. `LineFinder/code/gui1.py` fails with a missing comma at line 68. It was already broken on Torch's `main` before the merge.

### Open items from the merge

- About 8 new Shannon scripts (the Kimi and `eval_start_end_person*` scripts) still read keys with `os.environ[...]`, so they fail if a key isn't exported.
- `MOONSHOT_API_KEY` must be added to `.env`.
- RANSAC in `visual_3D.py` is unseeded, so counts can change slightly between runs.

---

## 2. GT counts (paper: 517 / 90 / 427)

The merged GT matches the paper exactly: **517 images, 90 too hard, 427 remaining.**

- **Where the 90 come from:** Luna 22 and Olivia 68. The 109 older-format GT files (airports and supermarkets, banks, online) have no "too hard" field.
- **Why Shannon's GT is needed:** Torch's GT alone gives 84 too hard. The difference is Shannon's 6 corrected labels.
- **Why `images_with_gt_json.txt` had 519 lines:** it listed every JSON file under `GT_json/`, including two summary files that aren't images: `GT_json_olivia/annotation_accuracy.json` and `GT_json_luna/outdoor_annotation_accuracy.json`. Both entries are now removed (section 6).

---

## 3. Where `OLD_number_of_people.csv` came from

It was produced by the **RANSAC-inlier counter**, not by box counting.

- **Script:** only Shannon's `batch_number_of_people.py` writes it (`OUT_CSV = "OLD_number_of_people.csv"`). That script reads `old_image_paths.txt` and counts the people RANSAC keeps on the fitted 3D line (`number_of_people = int(inlier_mask.sum())`, with `RANSAC_NUM_ITERS=1000`, `DIST_THRESH=0.8`, `MIN_INLIER_RATIO=0.3`).
- **Columns:** the header `image_path,image_id,OL_number_of_people,status` is exactly what that script writes.
- **Values:** all 108 counts are at or below the image's box count (`bbox_orient/*_bboxes.npy`), and only 24 equal it. Box counting would make every value equal.
- **Git history:** the file exists only in the single commit `17a0454` ("Shannon snapshot"), so there are no earlier versions.
- **Why `IMG_6728` had no count:** `old_image_paths.txt` has 108 entries and leaves it out.

---

## 4. Visibility (Table 1)

### The current rule in `batch_line_visibility_nogt.py`

The script writes `line_visibility_new.csv`. An endpoint is marked **not visible** when its box is less than one box-width from the left or right image edge:

```
x1 < (x2 - x1)    or    (W - 1) - x2 < (x2 - x1)
```

- **Turn direction:** left if the box is near the left edge, right otherwise.
- **Location:** for visible endpoints, a five-way bucket from the box center's x position.
- **Replaced rule:** the old rule (10 px margin on all four edges) is still used by `batch_line_visibility.py` and `line_visibility_location.py`.

### Consistency check (the script was not re-run)

- **Row consistency:** every row is internally consistent with the rule. Visible endpoints have a location and "N/A" for turn; not-visible endpoints have "N/A" for location and a turn direction.
- **Versus Torch's old-rule output** (`OL_line_visibility.csv`, 408 shared images): 243 endpoints flip from visible to not visible, and 27 flip from not visible to visible (cases that touched only the top or bottom edge).
- **Limit:** this shows consistency, not proof that the CSV came from this exact code. That would need a full re-run.

### The paper's rule

The paper describes this rule: extend the fitted 3D line by the median spacing between people, project the point into the image, and mark the endpoint not visible if the point is outside the image, the box touches the edge, or the depth there is more than 0.5 m closer.

**No script implements it.** Nothing extrapolates the line or compares depth at a projected point. The only median-spacing code is in `identify_queue_start_end_batch.py`, which uses it to split the queue into segments.

### Reproducing the paper

- **Inputs:** visibility from `line_visibility_new.csv`. Facing direction from `OL_queue_facing_results.csv` (Luna/Olivia) plus `queue_facing_results.csv` (older images).
- **Label mapping:** the facing predictions say `sideways-left` / `sideways-right` while the GT says `sideways`, so both were counted as `sideways`. Without this, facing drops to 49%.
- **Result on the older GT (84 too hard, 432 images):** overall is 65.67%, matching the paper exactly.
- **Facing mismatch:** the same run gives facing 83.33% (360/432), not the paper's 82.89% (358/432). The paper's two numbers are inconsistent: a facing score of 82.89% would make the overall about 65.61%.

---

## 5. Counting (Table 3)

### Which files and GT produced the paper's numbers

| Method | Prediction files | Evaluator | GT |
|---|---|---|---|
| GPT | `gpt_line_luna_olivia.jsonl` + `gpt_line_old.jsonl` | `eval_number_of_people_gpt.py` + `_gpt_old.py` | older GT (84 too hard): 324 + 108 = 432 |
| Gemini | `gemini_line_luna_olivia.jsonl` + `gemini_line_old.jsonl` | `eval_number_of_people_gemini.py` + `_gemini_old.py` | older GT: 323 + 108 = 431 |
| Kimi | `kimi_line_luna_olivia.jsonl` + `kimi_line_old.jsonl` | `eval_number_of_people_gpt.py` (same format) + `_kimi_old.py` | older GT: 324 + 108 = 432 |
| LineFinder | recorded in `code/results_number_people.txt` (432 samples) | — | older GT |

- **VLMs:** pooled over both subsets on the older GT, MAE, signed error, relative error, exact match, ±1, ±2 and under/over rates match the recorded values to about 3 decimals. RMSE comes out 0.01–0.24 lower for all three models; I couldn't find a pooling method that reproduces it exactly.
- **Gemini:** has no usable count for `outdoor_050`.
- **LineFinder:** its recorded 4.26 / 10.52 doesn't come from any CSV in either copy.

| LineFinder variant tried | Images | MAE | RMSE |
|---|---:|---:|---:|
| Merged OL CSV, Luna/Olivia only | 318 | 2.90 | 5.31 |
| OLD CSV, older images only | 108 | 6.64 | 11.60 |
| Merged OL + OLD, corrected GT | 426 | 3.85 | 7.43 |
| Merged OL + OLD, older GT | 432 | 3.83 | 7.39 |
| Torch box-count OL + OLD, older GT | 432 | 3.81 | 7.23 |
| Box counts for all images, older GT | 432 | 3.64 | 6.82 |
| Including "too hard" images | 516 | 3.22–3.86 | 6.63–7.06 |

### Running LineFinder on IMG_6728

- **Why it was missing:** the image had never been processed (no depth map, no box files). Its GT file is also named `IMG_6728.jpg.json`, so the evaluators never matched it.
- **How it was run** (CPU, because the GPU queue was stuck behind the group limit):
  - Depth: ml-depth-pro, `depth-pro` environment. Focal length 634.68 px.
  - Person boxes and orientation: JointBDOE, `coco_s_1024_e500_t010_w005_best.pt`, conf 0.3, IoU 0.5. Found 7 boxes.
  - LineFinder count, visibility and facing: `annot` environment, which has the Python 3.10 that `visual_3D.py` needs.
- **Results:**
  - Count: 6 (GT 6). GPT, Gemini and Kimi also predict 6.
  - Visibility: end not visible (turn left); start visible at center right.
  - Facing: sideways-right.
- **Where the outputs are:** `/scratch/ds5725/tmp/claude-3077396/-scratch-ds5725/cc365bb3-3529-4b0a-8a7c-2e12254b9329/scratchpad/r6728/`. They are not yet in the repo or the data folder.

---

## 6. File fixes (not yet committed)

- **`LineFinder/code/images_with_gt_json.txt`:** removed `annotation_accuracy` and `outdoor_annotation_accuracy`, leaving 517 lines.
- **`LineFinder/GT_json/GT_json_luna/outdoor_annotation_accuracy.json`:** recomputed from the merged GT. It now says 201 total, 22 hard, 179 not hard, and the per-question stats are recomputed on the 179 (for example, people count 53/185 → 52/179). The method compares the prediction and annotation exactly, and it reproduces both existing summaries exactly from the older GT. The original Windows line endings were kept.
- **Olivia summary:** already correct (68 hard), so unchanged.

---

## 7. LineFinder count errors larger than 10 (427 images)

39 images have an error larger than 10, and every one is an undercount.

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
| IMG_6745 | supermarket (old) | 28 | 10 | −18 |
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

By scene: theme park 13, bank 9, outdoor 8, online 4, airport 4, supermarket 1.

`IMG_5958` is also listed in `code/exclude_eval.txt`, but its GT doesn't mark it too hard, so it was kept.

---

## 8. Old vs. new values for Tables 1 and 3

- **Old:** paper values where given; otherwise the values recorded in `code/results_number_people.txt`, which match the Table 3 numbers quoted from the paper.
- **New:** corrected GT, 427 images including `IMG_6728`. Gemini has 426 because `outdoor_050` has no count.
- **Paper not edited.**

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

The paper gives only the overall and facing values for Table 1's LineFinder row in the source used here, so the other six old cells are blank. The recorded Luna/Olivia-only values in `code/results_luna_olivia.txt` cover a different subset (324 images) and aren't comparable.

---

## 9. Decisions still open

1. Commit and push the two file fixes in section 6.
2. Add the `IMG_6728` outputs (depth map, box files, focal length, CSV rows) to the repo and data folder, possibly together with step 6 of the prompt (copying the Shannon data).
3. Re-run `batch_line_visibility_nogt.py` on all images into scratch, to confirm `line_visibility_new.csv` matches the current code.
4. Continue with prompt steps 4–9 (the Hugging Face data merge).
