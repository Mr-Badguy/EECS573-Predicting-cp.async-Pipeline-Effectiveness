#!/bin/bash
#SBATCH --job-name=cpasync_a40
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
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
mkdir -p outputs/a40 slurm

python3 scripts/run_cuda_benchmarks.py \
  --run-gemm --run-stencil --run-pointer \
  --gpu-name A40 \
  --run-v0 \
  --gemm-output outputs/a40/gemm_raw.csv \
  --stencil-output outputs/a40/stencil_raw.csv \
  --pointer-output outputs/a40/pointer_chase_raw.csv

python3 scripts/extract_l2_params.py --input outputs/a40/pointer_chase_raw.csv --output-json outputs/a40/l2_params.json
python3 scripts/run_model_validation.py \
  --gpu-list A40 \
  --gemm-raw outputs/a40/gemm_raw.csv \
  --stencil-raw outputs/a40/stencil_raw.csv \
  --measured-out outputs/a40/measured_speedup.csv \
  --compare-l2-modes \
  --leave-one-out
