#!/bin/bash
# Snapshot uncommitted source before submission; no git commit or Kaggle upload.
set -euo pipefail
rs_root=$(cd "$(dirname "$0")/.." && pwd)
rs_suite=${1:-smoke}
case "$rs_suite" in
  cpu) rs_options=(--cpus-per-task=4 --mem=1G --time=01:00:00 --qos=normal);;
  gpu) rs_options=(--cpus-per-task=2 --mem=2G --gres=gpu:1 --time=00:10:00 --qos=normal);;
  smoke) rs_options=(--cpus-per-task=2 --mem=2G --gres=gpu:1 --time=00:05:00 --qos=debug);;
  *) echo 'Usage: bash scripts/submit_compare.sh {smoke|cpu|gpu}' >&2; exit 2;;
esac
rs_snapshot="$rs_root/snapshots/$(date -u +%Y%m%dT%H%M%S)_${rs_suite}_$$"
mkdir -p "$rs_snapshot" "$rs_root/logs" "$rs_root/results"
cp -r "$rs_root/rs_project" "$rs_root/configs" "$rs_root/tests" "$rs_root/slurm" "$rs_snapshot/"
sbatch "${rs_options[@]}" --job-name="rs-project-${rs_suite}" \
  --output="$rs_root/logs/${rs_suite}-%j.out" --error="$rs_root/logs/${rs_suite}-%j.err" \
  --export="ALL,PROJECT_ROOT=$rs_root,SOURCE_ROOT=$rs_snapshot,RS_SUITE=$rs_suite" \
  "$rs_snapshot/slurm/compare.sbatch"
