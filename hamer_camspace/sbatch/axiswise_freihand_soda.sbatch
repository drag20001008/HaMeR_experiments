#!/bin/bash
#SBATCH --job-name=axiswise_freihand
#SBATCH -o /workspace/drag1008/hamer/slurm/o/%x.o%j
#SBATCH -e /workspace/drag1008/hamer/slurm/e/%x.e%j
#SBATCH --partition=R3090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --nodelist=soda-03
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

source /workspace/drag1008/anaconda3/etc/profile.d/conda.sh
conda activate hamer
export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl
cd /workspace/drag1008/hamer

python analyze_translation_axiswise_freihand.py \
    --checkpoint /nas2/lait/1000_Members/jaehoon/hamer_checkpoints/exp5_freeze_beta_2d/epoch=029.pt \
    --batch_size 64 \
    --num_workers 4 \
    --out_json /nas2/lait/1000_Members/jaehoon/hamer_eval_results/axiswise_freihand.json
