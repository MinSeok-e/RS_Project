#!/bin/bash
#SBATCH --job-name=rs-project-refit
#SBATCH --partition=gpu24
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
set -euo pipefail
: "${PROJECT_ROOT:?}" "${SOURCE_ROOT:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
cd "$SOURCE_ROOT"
rs_python=/home/minseok/.venvs/rs-hw2/bin/python
rs_args=()
if [[ "${RS_SMOKE:-0}" == 1 ]]; then
    "$rs_python" -m unittest discover -s tests -v
    rs_args+=(--smoke)
fi
"$rs_python" -m rs_project.refit --data /home/minseok/data/semester/1/RS/data \
    --out "$PROJECT_ROOT/results/full_${SLURM_JOB_ID}" "${rs_args[@]}"
