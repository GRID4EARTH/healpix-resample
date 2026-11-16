#!/bin/bash
# SLURM job for the JSTARS revision experiments (see notebooks/hpc/README.md).
#
#   sbatch --export=ALL,STEP="e0 real"            notebooks/hpc/revision_slurm.sh
#   sbatch --export=ALL,STEP="e1e2 synthetic" --array=0-7 notebooks/hpc/revision_slurm.sh
#
# With --array, each task runs `STEP --shard <task>/<count>`.  Adapt the
# partition/account lines to your centre.
#SBATCH --job-name=hpxres-rev
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=hpxres-rev_%x_%A_%a.log
##SBATCH --partition=<cpu-partition>
##SBATCH --account=<project>
##SBATCH --gres=gpu:1            # optional: only with a CUDA build of PyTorch
set -euo pipefail

cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
STEP=${STEP:-selftest}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

ARGS=($STEP)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${ARGS[0]}" == "e1e2" ]]; then
    ARGS+=(--shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}")
fi

echo "host $(hostname)  step: ${ARGS[*]}  threads: $OMP_NUM_THREADS"
pixi run -e notebooks python notebooks/revision_run.py "${ARGS[@]}"
