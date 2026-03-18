#!/bin/bash
#SBATCH --job-name=cpasync_v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

module purge
module load cuda
module load cmake

PROJECT_ROOT="/scratch/eecs570w26s001_class_root/eecs570w26s001_class/xxxchen/project2_cp_async_pipeline"
cd "${PROJECT_ROOT}"
mkdir -p outputs/v100 slurm

python3 scripts/run_cuda_benchmarks.py \
  --run-gemm --run-stencil --run-pointer \
  --gpu-name V100 \
  --run-v0 \
  --gemm-output outputs/v100/gemm_raw.csv \
  --stencil-output outputs/v100/stencil_raw.csv \
  --pointer-output outputs/v100/pointer_chase_raw.csv

python3 scripts/extract_l2_params.py --input outputs/v100/pointer_chase_raw.csv --output-json outputs/v100/l2_params.json
python3 scripts/run_model_validation.py \
  --gpu-list V100 \
  --gemm-raw outputs/v100/gemm_raw.csv \
  --stencil-raw outputs/v100/stencil_raw.csv \
  --measured-out outputs/v100/measured_speedup.csv
