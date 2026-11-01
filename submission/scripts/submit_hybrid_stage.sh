#!/bin/bash
# Freeze committed code and run one stage; all computation stays on Slurm nodes.
set -euo pipefail
rs_stage=${1:?bases, auxiliary, features, verify, or infer}; shift
rs_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
rs_commit=$(git -C "$rs_root" rev-parse HEAD)
git -C "$rs_root" diff --quiet
git -C "$rs_root" diff --cached --quiet
rs_snapshot="$rs_root/snapshots/$(date -u +%Y%m%dT%H%M%S)_hybrid_${rs_stage}_$$"
mkdir -p "$rs_snapshot" "$rs_root/logs" "$rs_root/results"
git -C "$rs_root" archive "$rs_commit" submission | tar -x -C "$rs_snapshot"
rs_extra=()
case "$rs_stage" in
 bases) rs_script=submission_bases.sbatch ;;
 auxiliary|features) rs_script=submission_features.sbatch; rs_extra=(--stage "$rs_stage") ;;
 verify) rs_script=submission_features_verify.sbatch ;;
 infer) rs_script=submission_infer.sbatch ;;
 *) echo 'Unknown stage' >&2; exit 1 ;;
esac
sbatch --output="$rs_root/logs/hybrid-${rs_stage}-%j.out" --error="$rs_root/logs/hybrid-${rs_stage}-%j.err" \
 --export="ALL,PROJECT_ROOT=$rs_root,SOURCE_ROOT=$rs_snapshot/submission,RS_SOURCE_COMMIT=$rs_commit" \
 "$rs_snapshot/submission/slurm/$rs_script" "${rs_extra[@]}" "$@"
