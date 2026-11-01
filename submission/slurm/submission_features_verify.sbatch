#!/bin/bash
#SBATCH --job-name=rs-feature-verify
#SBATCH --partition=gpu24
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=2560M
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
: "${PROJECT_ROOT:?}" "${SOURCE_ROOT:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
cd "$SOURCE_ROOT"
rs_common=(--plan "$SOURCE_ROOT/configs/hybrid_ensemble_submission.json" --data /home/minseok/data/semester/1/RS/data --project "$PROJECT_ROOT" --verify)
/home/minseok/.venvs/rs-hw2/bin/python -m rs_project.submission_features "${rs_common[@]}" --stage auxiliary --out "$PROJECT_ROOT/results/submission_aux_verify_${SLURM_JOB_ID}"
/home/minseok/.venvs/rs-hw2/bin/python -m rs_project.submission_features "${rs_common[@]}" --stage features --out "$PROJECT_ROOT/results/submission_features_verify_${SLURM_JOB_ID}" --aux "$PROJECT_ROOT/results/submission_aux_verify_${SLURM_JOB_ID}"
