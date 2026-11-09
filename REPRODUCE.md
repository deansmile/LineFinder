# Reproducing the LineFinder results (corrected 427-image GT, seeded rerun)

Reference run: `results/2026-11-09/` in the private Hugging Face dataset
`deansmile123/LineFinder` (main branch). Commands below are for the NYU Torch cluster.

## 0. Setup

- **Code:** this repository at `/scratch/ds5725/linefinder`. Data (images, weights) comes from the private HF dataset; see `DATA.md`.
- **API keys:** put them in `.env` at the repo root, which is git-ignored (template: `.env.example`). Jobs load it with `set -a; source .env; set +a`.
- **Environments** (Singularity overlays in `/scratch/ds5725/environments`):

  | Environment | Overlay | Python | Used for |
  |---|---|---|---|
  | `bdoe` | `line.ext3` | 3.8 | JointBDOE |
  | `depth-pro` | `annot.ext3` | 3.9 | DepthPro |
  | `annot` | `annot1.ext3` | 3.10 | LineFinder logic and evaluation |

- **openai 1.x:** `annot` has openai 0.28, which the GPT-5.4 benchmark script needs. Scripts that need openai 1.x (Kimi, the description step) add it from a separate folder:
  ```bash
  pip install --target /scratch/ds5725/environments/pydeps_openai1 "openai==1.109.1"
  # then run those scripts with PYTHONPATH=/scratch/ds5725/environments/pydeps_openai1
  ```
- **Data root:** every LineFinder script reads its data through `LINEFINDER_ROOT`, set in `LineFinder/code/lf_paths.py`. The default is `/scratch/ds5725/linefinder/LineFinder`.
- **Randomness:** `ransac_line_3d` in `visual_3D.py` uses `seed=42` by default, which makes counts, visibility and facing reproducible.

## 1. Evaluation image list (427 images)

These are the images in `LineFinder/GT_json_canonical.csv` with `too_hard != yes`, and the list is in `results/2026-11-09/image_list_427.txt`.

## 2. Full rerun + timing (GPU)

```bash
R=/scratch/ds5725/linefinder/LineFinder/results/$(date +%F)
sbatch --account=torch_pr_51_tandon_advanced --partition=a100_tandon --gres=gpu:1 \
  --cpus-per-task=16 --mem=128G --time=03:00:00 --output=$R/run_%j.log \
  --wrap="LineFinder/code/timing/run_rerun_and_timing.sh $R $R/image_list_427.txt 0 5"
```

`run_rerun_and_timing.sh` runs these steps in order:
1. **Official outputs** into `$R/official/`, using the unmodified scripts:
   - `ml-depth-pro/batched_depth.py`
   - `JointBDOE/demos/image.py` (weights `coco_s_1024_e500_t010_w005_best.pt`, conf 0.3, IoU 0.5)
   - `batch_number_of_people.py`, `batch_line_visibility_nogt.py`, `batch_queue_direction.py` and `export_start_end.py`
2. **Timed passes** into `$R/timed/` (`code/timing/time_{depthpro,jointbdoe,linefinder}.py`): models loaded once, 5 warm-up images, `torch.cuda.synchronize()` around the GPU stages.
3. **Comparison:** `code/timing/compare_outputs.py` checks the timed outputs against the official ones and writes `compare_outputs.json`.

## 3. Start/end person descriptions (GPT-5.4 mini, timed)

```bash
export LINEFINDER_ROOT=$R/official PYTHONPATH=/scratch/ds5725/environments/pydeps_openai1
python LineFinder/code/timing/time_describe.py --root $R/official \
  --input-csv $R/official/line_visibility_new.csv \
  --output-csv $R/official/start_end_person_descriptions.csv \
  --timing-csv $R/timed/timings_describe.csv --env-dir $R/timed --warmup 5
```

## 4. Tables (corrected GT)

Run these from `LineFinder/code/evaluation`:

```bash
# Table 1 (facing merged / unmerged / benchmark rule, all methods)
python eval_table1_all.py --gt-root ../.. --lf-vis $R/official/line_visibility_new.csv \
  --lf-facing $R/official/OL_queue_facing_results.csv --json-out $R/table1_427.json
# Table 2 (Gemini 3 Flash judge; judgments cached in the JSONL)
python eval_table2_all.py --gt-root ../.. --lf-desc $R/official/start_end_person_descriptions.csv \
  --lf-vis $R/official/line_visibility_new.csv --cache $R/table2/judgments_gemini3flash.jsonl \
  --json-out $R/table2/table2_427.json
# Table 3 + per-image LineFinder count errors (boxes vs RANSAC inliers)
python eval_table3_all.py --gt-root ../.. --lf-counts $R/official/OLD_number_of_people.csv \
  --bbox-dir $R/official/bbox_orient --out-dir $R
```

- **Check against the paper:** passing `--gt-root` the pre-correction GT (`results/2026-11-09/gt_precorrection_ref`) reproduces the paper-era evaluation sets of 432 and 433 images.
- **Table 2 judgments:** with the cached JSONL, re-scoring makes no new API calls.

## 5. VLM API timing (100 images, seed 42)

```bash
cd LineFinder/code/timing
python time_vlm_api.py --model gpt    --list $R/image_list_427.txt --n 100 --seed 42 --out-dir $R/api_timing
python time_vlm_api.py --model gemini --list $R/image_list_427.txt --n 100 --seed 42 --out-dir $R/api_timing
PYTHONPATH=/scratch/ds5725/environments/pydeps_openai1 python time_vlm_api.py --model kimi \
  --kimi-model kimi-k2.6 --kimi-temperature 0.6 --list $R/image_list_427.txt --n 100 --seed 42 --out-dir $R/api_timing
```

`code/timing/run_api_jobs.sh` runs the Table 2 judging and the Gemini/Kimi timing as one job.

## 6. Upload results (private dataset)

```bash
hf upload deansmile123/LineFinder $R results/$(basename $R) --repo-type=dataset \
  --exclude "timed/depth_map/*" --exclude "gt_precorrection_ref/*"
```
