#!/bin/bash
#SBATCH --job-name=wilor_train_beta
#SBATCH -o /workspace/drag1008/wilor/slurm/o/%x.o%j
#SBATCH -e /workspace/drag1008/wilor/slurm/e/%x.e%j
#SBATCH --partition=R3090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4             # DDP: GPU당 1 task
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=soda-03
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

source ~/anaconda3/etc/profile.d/conda.sh
conda activate wilor

export PYTHONUNBUFFERED=1

# torch.save가 임시파일을 /tmp에 만든 뒤 os.rename으로 옮기는데,
# /tmp와 /workspace(네트워크 마운트)가 다른 파일시스템이라 EXDEV 에러로 크래시했음.
# 체크포인트 저장 경로와 같은 파일시스템에 TMPDIR을 둬서 rename이 성공하도록 함.
mkdir -p /workspace/drag1008/wilor/tmp_beta
export TMPDIR=/workspace/drag1008/wilor/tmp_beta

cd /workspace/drag1008/wilor

srun python -u train_wilor.py \
    --dataset_npz /nas2/lait/1000_Members/jaehoon/DexYCB/wilor_annotations_v3.npz \
    --img_dir /nas2/lait/1000_Members/jaehoon/DexYCB \
    --output_dir ./runs_beta_v2 \
    --split_mode official \
    --batch_size 16 \
    --max_epochs 10 \
    --num_workers 4 \
    --val_check_interval 300 \
    --devices 4 \
    --beta_conditioning \
    --wandb_project wilor-beta-conditioning \
    --wandb_run_name beta_v2_officialsplit
