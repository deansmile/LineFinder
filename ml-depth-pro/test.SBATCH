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
	    --overlay /scratch/ds5725/environments/annot.ext3:rw \
	    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
	    /bin/bash -lc '
			export HOME=/scratch/ds5725
			export PYTHONNOUSERSITE=1
			export PIP_NO_USER=1
			unset PYTHONPATH

			source /ext3/miniconda3/etc/profile.d/conda.sh
			export CONDA_ENVS_PATH=/scratch/ds5725/.conda/envs
			export CONDA_PKGS_DIRS=/scratch/ds5725/.conda/pkgs
			export PIP_CACHE_DIR=/scratch/ds5725/.pip_cache
			export TMPDIR=/scratch/ds5725/.tmp

    		conda activate depth-pro
			python batched_depth.py
		'
