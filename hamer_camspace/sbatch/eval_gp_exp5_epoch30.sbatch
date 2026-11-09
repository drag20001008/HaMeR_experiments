#!/bin/bash
#SBATCH --job-name=hamer_eval_gp_exp5_epoch30
#SBATCH -o /workspace/drag1008/hamer/slurm/o/%x.o%j
#SBATCH -e /workspace/drag1008/hamer/slurm/e/%x.e%j
#SBATCH --partition=R4090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --nodelist=vegi-01
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

mkdir -p /workspace/drag1008/hamer/slurm/o /workspace/drag1008/hamer/slurm/e
source /workspace/drag1008/miniconda3/etc/profile.d/conda.sh
conda activate hamer
export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl
cd /workspace/drag1008/hamer

echo "NODE: $(hostname)"
echo "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

RESULT_DIR=/nas2/lait/1000_Members/jaehoon/hamer_eval_results
mkdir -p "$RESULT_DIR"

python eval_global_positioning.py \
    --backbone_mode freeze \
    --use_beta_adapter True \
    --use_2d_head True \
    --checkpoint /nas2/lait/1000_Members/jaehoon/hamer_checkpoints/exp5_freeze_beta_2d/epoch=029.pt \
    --batch_size 64 \
    --num_workers 4 \
    --tag gp_exp5_epoch30 \
    --out_json "$RESULT_DIR/gp_exp5_epoch30.json"
