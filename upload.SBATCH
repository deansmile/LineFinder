#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --job-name=upload
#SBATCH --account=torch_pr_51_tandon_advanced

export SSL_CERT_DIR=/etc/ssl/certs

singularity exec --nv \
	--overlay /scratch/ds5725/environments/line.ext3:ro \
	/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
	/bin/bash -lc '
		source /ext3/env.sh
		conda activate SegEarth
		# hf upload ai4ce/NYC-CD /vast/ds5725/cd_datasets/quality_control /quality_control --repo-type=dataset
		# hf download ai4ce/NYC-CD VL-CMU-CD.zip --repo-type dataset  --local-dir /scratch/ds5725/cd_datasets/
		# hf download ai4ce/NYC-CD \
		# 	--repo-type dataset \
		# 	--include "grounded_sam_res/cmu_objects_gd15/**" \
		# 	--local-dir /scratch/ds5725/alvpr_code/GroundedSAM2/cmu_objects_gd15
		hf upload deansmile123/linefinder /scratch/ds5725/linefinder/LineFinder/GT_json GT_json
		hf upload deansmile123/linefinder /scratch/ds5725/linefinder/LineFinder/Images Images
	'
