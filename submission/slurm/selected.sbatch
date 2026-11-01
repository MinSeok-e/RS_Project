#!/bin/bash
#SBATCH --job-name=rs-selected-export
#SBATCH --partition=gpu24
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=1536M
#SBATCH --time=00:05:00
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$SOURCE_ROOT"
/home/minseok/.venvs/rs-hw2/bin/python -m rs_project.selected_mix \
 --config "configs/$RS_SELECTED_CONFIG" --project "$PROJECT_ROOT" \
 --data /home/minseok/data/semester/1/RS/data --out "$PROJECT_ROOT/results/selected_${SLURM_JOB_ID}"
