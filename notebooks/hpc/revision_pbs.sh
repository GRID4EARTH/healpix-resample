#!/bin/bash
# PBS Pro job for the JSTARS revision experiments (see notebooks/hpc/README.md).
#
#   qsub -v STEP="e0 real"                           notebooks/hpc/revision_pbs.sh
#   qsub -v STEP="e1e2 synthetic",NSHARD=8 -J 0-7    notebooks/hpc/revision_pbs.sh
#
# With -J, each sub-job runs `STEP --shard <index>/<NSHARD>`.  Adapt the
# queue/select lines to your centre.  Optional variables: PIXI (path to the
# pixi binary), PIXI_CACHE_DIR, and for hosts whose glibc is too old
# HPXREV_SIF (container image), HPXREV_CA (CA bundle), HPXREV_BINDS.
#PBS -N hpxres-rev
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=06:00:00
#PBS -j oe
##PBS -q <queue>
set -euo pipefail

cd "${REPO_DIR:-$PBS_O_WORKDIR}"
STEP=${STEP:-selftest}
NT=${NCPUS:-8}
export OMP_NUM_THREADS=$NT
export MKL_NUM_THREADS=$NT

ARGS=($STEP)
if [[ -n "${PBS_ARRAY_INDEX:-}" && ( "${ARGS[0]}" == "e1e2" || "${ARGS[0]}" == "e4" || "${ARGS[0]}" == "e3" || "${ARGS[0]}" == "e5" || "${ARGS[0]}" == "e6" || "${ARGS[0]}" == "e6x" || "${ARGS[0]}" == "e6y" || "${ARGS[0]}" == "e7" ) ]]; then
    ARGS+=(--shard "${PBS_ARRAY_INDEX}/${NSHARD:?set NSHARD to the array size}")
fi

# --as-is: use the installed env; concurrent array sub-jobs must not re-solve
# or rebuild it.
RUN=("${PIXI:-pixi}" run --as-is -e notebooks python notebooks/revision_run.py "${ARGS[@]}")
if [[ -n "${HPXREV_SIF:-}" ]]; then
    # Host glibc older than the locked environment needs (pixi.lock wants
    # glibc >= 2.28; e.g. SLES 12 has 2.19): run pixi inside a container
    # that only supplies a recent glibc (README, "glibc trop ancienne").
    set +u
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load "${SINGULARITY_MODULE:-singularity}"
    set -u
    RUN=(singularity exec --cleanenv -B "${HPXREV_BINDS:-/home1/datawork,/home1/scratch}"
         "$HPXREV_SIF" env OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
         ${PIXI_CACHE_DIR:+PIXI_CACHE_DIR=$PIXI_CACHE_DIR} PIXI_NO_TELEMETRY=1
         ${HPXREV_CA:+SSL_CERT_FILE=$HPXREV_CA} "${RUN[@]}")
fi

echo "host $(hostname)  step: ${ARGS[*]}  threads: $NT  container: ${HPXREV_SIF:-none}"
"${RUN[@]}"
