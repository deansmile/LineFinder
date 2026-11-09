#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --job-name=res
#SBATCH --account=torch_pr_51_tandon_advanced

export SSL_CERT_DIR=/etc/ssl/certs

singularity exec --nv --fakeroot \
	    --overlay /scratch/ds5725/environments/annot1.ext3:rw \
	    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
	    /bin/bash -lc '
			source /ext3/env.sh
		 	conda activate annot 
			pip install -U openai
			python qwenvl_line_reduced_questions.py
		'
