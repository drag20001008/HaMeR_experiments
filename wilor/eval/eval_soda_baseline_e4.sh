#!/bin/bash
#SBATCH --job-name=wilor_eval_baseline_e4
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
#SBATCH --time=12:00:00

source ~/anaconda3/etc/profile.d/conda.sh
conda activate wilor

export PYTHONUNBUFFERED=1
cd /workspace/drag1008/wilor

python -u eval_metric_scale.py \
    --dataset_npz /nas2/lait/1000_Members/jaehoon/DexYCB/wilor_annotations_v3.npz \
    --img_dir /nas2/lait/1000_Members/jaehoon/DexYCB \
    --checkpoint ./runs_baseline_v2/checkpoints/epoch=4-step=63570.ckpt \
    --split_mode official --official_split test \
    --batch_size 32 --num_workers 4
