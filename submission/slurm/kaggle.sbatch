#!/bin/bash
#SBATCH --job-name=rs-project-kaggle
#SBATCH --partition=gpu24
#SBATCH --qos=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/minseok/data/semester/1/RS/project
#SBATCH --output=/home/minseok/data/semester/1/RS/project/logs/kaggle-%j.out
#SBATCH --error=/home/minseok/data/semester/1/RS/project/logs/kaggle-%j.err
set -euo pipefail
/home/minseok/.venvs/rs-kaggle/bin/python submission/scripts/kaggle_project.py "$@"
