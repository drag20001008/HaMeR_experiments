#!/bin/bash
#SBATCH --job-name=wilor_realsense
#SBATCH -o /workspace/drag1008/wilor/slurm/o/%x.o%j
#SBATCH -e /workspace/drag1008/wilor/slurm/e/%x.e%j
#SBATCH --partition=R3090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=soda-02
#SBATCH --mem=16G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate wilor

cd /workspace/drag1008/wilor

# RealSense D435i 녹화(2026-10-08)에서 10프레임마다 뽑은 color 이미지, pretrained wilor_final.ckpt로 추론
for name in outdoor_hand outdoor_glove indoor_hand2 indoor_glove; do
    echo "===== ${name} ====="
    python demo.py \
        --img_folder realsense_images/${name}_every10 \
        --out_folder realsense_demo_out/${name}_every10
done
