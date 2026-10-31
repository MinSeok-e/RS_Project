#!/bin/bash
#SBATCH --job-name=rs-compare
#SBATCH --partition=gpu24
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --time=01:00:00
set -euo pipefail
: "${PROJECT_ROOT:?}" "${SOURCE_ROOT:?}" "${RS_SUITE:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
cd "$SOURCE_ROOT"
rs_python=/home/minseok/.venvs/rs-hw2/bin/python
if [[ "$RS_SUITE" == smoke ]]; then
    "$rs_python" -m unittest discover -s tests -v
fi
"$rs_python" -m rs_project.run --suite "$RS_SUITE" \
    --data /home/minseok/data/semester/1/RS/data \
    --config "${RS_CONFIG:-configs/best.json}" \
    --out "$PROJECT_ROOT/results/${RS_SUITE}_${SLURM_JOB_ID}"
