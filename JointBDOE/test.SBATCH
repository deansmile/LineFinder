#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=res
#SBATCH --account=torch_pr_51_tandon_advanced

export SSL_CERT_DIR=/etc/ssl/certs

singularity exec --nv --fakeroot \
  --overlay /scratch/ds5725/environments/line.ext3:rw \
  /share/apps/images/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -lc '
    source /ext3/env.sh
    conda activate bdoe
    python demos/image.py --weights /scratch/ds5725/linefinder/JointBDOE/weights/coco_s_1024_e500_t010_w005_best.pt \
      --device 0 --conf-thres 0.3 --iou-thres 0.5
  '