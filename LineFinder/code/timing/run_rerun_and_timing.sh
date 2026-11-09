#!/bin/bash
# Full LineFinder rerun (official outputs) + timed passes + output comparison.
#
# Usage: run_rerun_and_timing.sh <results_dir> <image_list> <device: 0|cpu> [warmup]
#   <results_dir>/official : outputs of the unmodified pipeline scripts (the results)
#   <results_dir>/timed    : outputs + per-image timings of the timing harness
# Stages run in their own environments (separate passes over the images).
set -euo pipefail

OUT=$(realpath "$1"); LIST=$(realpath "$2"); DEV=${3:-0}; WARMUP=${4:-5}
REPO=/scratch/ds5725/linefinder
CODE=$REPO/LineFinder/code
TIMING=$CODE/timing
OFF=$OUT/official; TIM=$OUT/timed
JBDOE_W=$REPO/JointBDOE/weights/coco_s_1024_e500_t010_w005_best.pt
SIF_NEW=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
SIF_OLD=/share/apps/images/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif
NV=""; [ "$DEV" != "cpu" ] && NV="--nv"
export SSL_CERT_DIR=/etc/ssl/certs PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg

mkdir -p "$OFF/code" "$TIM/lf"
for f in code/olivia_luna_image_paths.txt code/old_image_paths.txt image_paths.txt; do cp "$LIST" "$OFF/$f"; done
cp "$LIST" "$OUT/image_list.txt"
: > "$OFF/focal_length_px.txt"   # batched_depth.py appends; start empty
rm -f "$TIM/timings_per_image.csv"

depthpro() {  # run python in the depth-pro env, cwd ml-depth-pro
  singularity exec $NV --overlay /scratch/ds5725/environments/annot.ext3:ro $SIF_NEW /bin/bash -lc "
    export HOME=/scratch/ds5725 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 LINEFINDER_ROOT=$OFF; unset PYTHONPATH
    source /ext3/miniconda3/etc/profile.d/conda.sh; export CONDA_ENVS_PATH=/scratch/ds5725/.conda/envs
    conda activate depth-pro; cd $REPO/ml-depth-pro; $*"
}
bdoe() {      # run python in the bdoe env, cwd JointBDOE
  singularity exec $NV --overlay /scratch/ds5725/environments/line.ext3:ro $SIF_OLD /bin/bash -lc "
    source /ext3/env.sh; conda activate bdoe; export PYTHONDONTWRITEBYTECODE=1 LINEFINDER_ROOT=$OFF
    cd $REPO/JointBDOE; $*"
}
annot() {     # run python in the annot env (Python 3.10), cwd given as first arg
  local cwd=$1; shift
  singularity exec --overlay /scratch/ds5725/environments/annot1.ext3:ro $SIF_NEW /bin/bash -lc "
    source /ext3/env.sh; conda activate annot; export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg LINEFINDER_ROOT=$OFF
    cd $cwd; $*"
}

echo "=== hardware"; nvidia-smi || true; lscpu | grep -E 'Model name|^CPU\(s\)'; free -g | head -2
echo "=== [official] DepthPro";      depthpro python batched_depth.py
echo "=== [official] JointBDOE";     bdoe python demos/image.py --weights $JBDOE_W --device $DEV --conf-thres 0.3 --iou-thres 0.5 --vis-dir $OFF/vis
echo "=== [official] LineFinder";    annot $OFF "python $CODE/batch_number_of_people.py && python $CODE/batch_line_visibility_nogt.py && python $CODE/batch_queue_direction.py && python $CODE/export_start_end.py"
echo "=== [timed] DepthPro";         depthpro python $TIMING/time_depthpro.py --list $LIST --out-dir $TIM/depth_map --focal-txt $TIM/focal_length_px.txt --timing-csv $TIM/timings_per_image.csv --env-dir $TIM --warmup $WARMUP
echo "=== [timed] JointBDOE";        bdoe python $TIMING/time_jointbdoe.py --list $LIST --out-dir $TIM/bbox_orient --timing-csv $TIM/timings_per_image.csv --env-dir $TIM --weights $JBDOE_W --device $DEV --warmup $WARMUP
echo "=== [timed] LineFinder logic"; annot $TIM "python $TIMING/time_linefinder.py --list $LIST --out-dir $TIM/lf --timing-csv $TIM/timings_per_image.csv --env-dir $TIM --warmup $WARMUP"
echo "=== compare";                  annot $OUT "python $TIMING/compare_outputs.py --official $OFF --timed $TIM --out $OUT/compare_outputs.json" || echo "COMPARE: DIFFERENCES FOUND"
echo "=== done"
