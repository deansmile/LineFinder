#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=24:00:00
#SBATCH --output=out_%A.out
#SBATCH --gres=gpu
#SBATCH --job-name=install
#SBATCH --account=pr_113_tandon_advanced

export SSL_CERT_DIR=/etc/ssl/certs

singularity exec --nv \
  --overlay /scratch/ds5725/environments/line.ext3:rw \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -lc '
    source /ext3/env.sh
    conda activate bdoe
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  '