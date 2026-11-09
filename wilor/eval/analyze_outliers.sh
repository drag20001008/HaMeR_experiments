#!/bin/bash
#SBATCH --job-name=wilor_analyze_outliers
#SBATCH -o /workspace/drag1008/wilor/slurm/o/%x.o%j
#SBATCH -e /workspace/drag1008/wilor/slurm/e/%x.e%j
#SBATCH --partition=R3090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=soda-03
#SBATCH --mem=16G
#SBATCH --time=01:00:00

source ~/anaconda3/etc/profile.d/conda.sh
conda activate wilor

export PYTHONUNBUFFERED=1
cd /workspace/drag1008/wilor

python -u analyze_outliers.py \
    --checkpoint ./runs_baseline_v2/checkpoints/epoch=2-step=38142.ckpt \
    --batch_size 32 --num_workers 4 --max_batches 400 --top_k 30
