#!/bin/bash
#SBATCH --job-name=verify_pt3d_gt_mano
#SBATCH -o /workspace/drag1008/hamer/slurm/o/%x.o%j
#SBATCH -e /workspace/drag1008/hamer/slurm/e/%x.e%j
#SBATCH --partition=R3090
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --nodelist=soda-01
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

source /workspace/drag1008/anaconda3/etc/profile.d/conda.sh
conda activate hamer
export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl
cd /workspace/drag1008/hamer

python verify_pytorch3d_gt_mano.py \
    --out_json /nas2/lait/1000_Members/jaehoon/hamer_eval_results/verify_pt3d_gt_mano.json \
    --viz_dir /nas2/lait/1000_Members/jaehoon/hamer_eval_results/viz_gt_mano
