#!/bin/bash
#SBATCH --job-name=export3d
#SBATCH -o /workspace/drag1008/wilor/slurm/o/%x.o%j
#SBATCH -e /workspace/drag1008/wilor/slurm/e/%x.e%j
#SBATCH --partition=R3090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00

source ~/anaconda3/etc/profile.d/conda.sh
conda activate hamer   # pytorch3d
export PYTHONUNBUFFERED=1
cd /workspace/drag1008/wilor

python -u export_3d_samples.py "$@"
