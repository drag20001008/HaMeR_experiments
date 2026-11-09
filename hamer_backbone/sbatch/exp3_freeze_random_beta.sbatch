#!/bin/bash
#SBATCH --job-name=hamer_exp3_freeze_random_beta
#SBATCH -o /workspace/drag1008/hamer/slurm/o/%x.o%j
#SBATCH -e /workspace/drag1008/hamer/slurm/e/%x.e%j
#SBATCH --partition=R4090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --nodelist=vegi-02
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

mkdir -p /workspace/drag1008/hamer/slurm/o /workspace/drag1008/hamer/slurm/e

source /workspace/drag1008/miniconda3/etc/profile.d/conda.sh
conda activate hamer

export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl

cd /workspace/drag1008/hamer

echo "NODE: $(hostname)"
echo "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

CKPT_DIR=/nas2/lait/1000_Members/jaehoon/hamer_checkpoints/exp3_freeze_random_beta
mkdir -p "$CKPT_DIR"

RESUME_ARG=""
LATEST_CKPT=$(ls -v "$CKPT_DIR"/epoch=*.pt 2>/dev/null | tail -n 1)
if [ -n "$LATEST_CKPT" ]; then
    echo "found existing checkpoint, resuming from: $LATEST_CKPT"
    RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
fi

torchrun --standalone --nproc_per_node=2 train_backbone_experiments.py \
    --backbone_mode freeze \
    --decoder_init random \
    --use_beta_adapter True \
    --output_dir "$CKPT_DIR" \
    --max_epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_workers 8 \
    --wandb_project hamer-backbone-experiments \
    --wandb_run_name exp3_freeze_random_beta \
    $RESUME_ARG
