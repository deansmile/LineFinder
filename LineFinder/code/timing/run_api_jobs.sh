#!/bin/bash
# Gemini/Kimi API work: Table 2 judging (VLMs) + step-7 API timing. Usage: run_api_jobs.sh <results_dir> <old_gt_root>
set -uo pipefail
D=$(realpath "$1"); OLDGT=$2; L=/scratch/ds5725/linefinder/LineFinder
singularity exec --overlay /scratch/ds5725/environments/annot1.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -lc "
  source /ext3/env.sh; conda activate annot; set -a; source /scratch/ds5725/linefinder/.env; set +a
  export PYTHONDONTWRITEBYTECODE=1 SSL_CERT_DIR=/etc/ssl/certs
  cd $L/code/evaluation
  echo '=== Table 2 judging, corrected GT'
  python eval_table2_all.py --gt-root $L --cache $D/table2/judgments_gemini3flash.jsonl --json-out $D/table2/table2_vlm_427.json 2>&1 | grep -v Warning
  echo '=== Table 2 judging, pre-correction GT (validation against paper)'
  python eval_table2_all.py --gt-root $OLDGT --cache $D/table2/judgments_gemini3flash.jsonl --json-out $D/table2/table2_vlm_oldgt.json 2>&1 | grep -v Warning
  cd $L/code/timing
  echo '=== API timing: Gemini'
  python time_vlm_api.py --model gemini --list $D/image_list_427.txt --n 100 --seed 42 --out-dir $D/api_timing | tail -3
  echo '=== API timing: Kimi k2.6 stand-in'
  PYTHONPATH=/scratch/ds5725/environments/pydeps_openai1 python time_vlm_api.py --model kimi --kimi-model kimi-k2.6 --kimi-temperature 0.6 --list $D/image_list_427.txt --n 100 --seed 42 --out-dir $D/api_timing | tail -3
"
echo '=== done'
