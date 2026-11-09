#!/bin/bash
#SBATCH --job-name=rs_depthcorr
#SBATCH -o /workspace/drag1008/wilor/slurm/o/%x.o%j
#SBATCH -e /workspace/drag1008/wilor/slurm/e/%x.e%j
#SBATCH --partition=R3090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodelist=soda-02
#SBATCH --time=02:00:00

source /workspace/drag1008/anaconda3/etc/profile.d/conda.sh
conda activate wilor
export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl
cd /workspace/drag1008/wilor
python -u realsense_depth_correction.py "$@"
