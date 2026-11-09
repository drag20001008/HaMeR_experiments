#!/bin/bash
#SBATCH --job-name=hamer_beta_adapter_smoke
#SBATCH -o /workspace/drag1008/hamer/slurm/o/%x.o%j
#SBATCH -e /workspace/drag1008/hamer/slurm/e/%x.e%j
#SBATCH --partition=R3090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=soda-01
#SBATCH --mem=32G
#SBATCH --time=00:15:00

mkdir -p /workspace/drag1008/hamer/slurm/o /workspace/drag1008/hamer/slurm/e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate hamer

export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl

cd /workspace/drag1008/hamer

echo "NODE: $(hostname)"
echo "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python smoke_test.py
