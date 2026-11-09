#!/bin/bash
# Upload a dated results folder to the private HF dataset (main). Usage: upload_results_hf.sh 2026-11-09
set -euo pipefail
DATE=$1; L=/scratch/ds5725/linefinder/LineFinder
export SSL_CERT_DIR=/etc/ssl/certs
singularity exec --overlay /scratch/ds5725/environments/line.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -lc "
source /ext3/env.sh; conda activate SegEarth; python - <<PY
from huggingface_hub import HfApi
api = HfApi(); repo = 'deansmile123/LineFinder'
info = api.dataset_info(repo)
assert info.private, 'dataset is PUBLIC - refusing to upload'
print('private:', info.private)
api.upload_large_folder(repo_id=repo, repo_type='dataset', revision='main', folder_path='$L',
    allow_patterns=['results/$DATE/**'],
    ignore_patterns=['results/$DATE/timed/depth_map/**', 'results/$DATE/timed/bbox_orient/**'])
files = [f for f in api.list_repo_files(repo, repo_type='dataset', revision='main') if f.startswith('results/$DATE/')]
print('files now on main under results/$DATE:', len(files))
PY"
