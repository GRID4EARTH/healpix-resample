#!/bin/bash
# PBS Pro job for the JSTARS revision experiments (see notebooks/hpc/README.md).
#
#   qsub -v STEP="e0 real"                           notebooks/hpc/revision_pbs.sh
#   qsub -v STEP="e1e2 synthetic",NSHARD=8 -J 0-7    notebooks/hpc/revision_pbs.sh
#
# With -J, each sub-job runs `STEP --shard <index>/<NSHARD>`.  Adapt the
# queue/select lines to your centre.
#PBS -N hpxres-rev
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=06:00:00
#PBS -j oe
##PBS -q <queue>
set -euo pipefail

cd "${REPO_DIR:-$PBS_O_WORKDIR}"
STEP=${STEP:-selftest}
export OMP_NUM_THREADS=${NCPUS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

ARGS=($STEP)
if [[ -n "${PBS_ARRAY_INDEX:-}" && "${ARGS[0]}" == "e1e2" ]]; then
    ARGS+=(--shard "${PBS_ARRAY_INDEX}/${NSHARD:?set NSHARD to the array size}")
fi

echo "host $(hostname)  step: ${ARGS[*]}  threads: $OMP_NUM_THREADS"
pixi run -e notebooks python notebooks/revision_run.py "${ARGS[@]}"
