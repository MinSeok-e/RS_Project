#!/bin/bash
#SBATCH --job-name=rs-submission-infer
#SBATCH --partition=gpu24
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=2560M
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
set -euo pipefail
: "${PROJECT_ROOT:?}" "${SOURCE_ROOT:?}" "${RS_SOURCE_COMMIT:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
cd "$SOURCE_ROOT"
/home/minseok/.venvs/rs-hw2/bin/python -m rs_project.submission_infer \
 --plan "$SOURCE_ROOT/configs/hybrid_ensemble_submission.json" \
 --data /home/minseok/data/semester/1/RS/data --project "$PROJECT_ROOT" \
 --out "$PROJECT_ROOT/results/submission_ensemble_${SLURM_JOB_ID}" "$@"
