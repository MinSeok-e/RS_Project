#!/bin/bash
set -euo pipefail
rs_config=${1:?configuration filename in submission/configs}; shift
rs_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
rs_commit=$(git -C "$rs_root" rev-parse HEAD)
git -C "$rs_root" diff --quiet
git -C "$rs_root" diff --cached --quiet
rs_snapshot="$rs_root/snapshots/$(date -u +%Y%m%dT%H%M%S)_selected_$$"
mkdir -p "$rs_snapshot" "$rs_root/logs" "$rs_root/results"
git -C "$rs_root" archive "$rs_commit" submission | tar -x -C "$rs_snapshot"
sbatch --output="$rs_root/logs/selected-%j.out" --error="$rs_root/logs/selected-%j.err" \
 --export="ALL,PROJECT_ROOT=$rs_root,SOURCE_ROOT=$rs_snapshot/submission,RS_SOURCE_COMMIT=$rs_commit,RS_SELECTED_CONFIG=$rs_config" \
 "$@" "$rs_snapshot/submission/slurm/selected.sbatch"
