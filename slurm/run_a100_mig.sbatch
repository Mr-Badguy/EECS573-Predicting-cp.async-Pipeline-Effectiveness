#!/bin/bash
#SBATCH --job-name=cpasync_a100_mig
#SBATCH --partition=gpu_mig40
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

module purge
module load cuda
module load cmake

# NOTE:
# This script assumes the allocated A100 is configured as MIG 3g.40gb by the cluster setup.
# If your cluster uses a dedicated MIG partition/constraint, adjust SBATCH directives accordingly.

PROJECT_ROOT="/scratch/eecs570w26s001_class_root/eecs570w26s001_class/xxxchen/project2_cp_async_pipeline"
cd "${PROJECT_ROOT}"
mkdir -p outputs/a100_mig slurm

python3 scripts/run_cuda_benchmarks.py \
  --run-gemm --run-stencil --run-pointer \
  --gpu-name A100_MIG_3g40gb \
  --run-v0 \
  --gemm-output outputs/a100_mig/gemm_raw.csv \
  --stencil-output outputs/a100_mig/stencil_raw.csv \
  --pointer-output outputs/a100_mig/pointer_chase_raw.csv

python3 scripts/extract_l2_params.py --input outputs/a100_mig/pointer_chase_raw.csv --output-json outputs/a100_mig/l2_params.json
python3 scripts/run_model_validation.py \
  --gpu-list A100_MIG_3g40gb \
  --gemm-raw outputs/a100_mig/gemm_raw.csv \
  --stencil-raw outputs/a100_mig/stencil_raw.csv \
  --measured-out outputs/a100_mig/measured_speedup.csv \
  --compare-l2-modes \
  --leave-one-out
