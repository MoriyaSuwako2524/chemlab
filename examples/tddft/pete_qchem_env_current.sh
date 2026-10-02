#!/bin/bash
# Source inside a scheduler allocation; configure installation/scratch paths first.
: "${QC:?Set QC to the Q-Chem installation directory}"
: "${QCAUX:?Set QCAUX to the Q-Chem auxiliary directory}"
: "${SCRATCH_ROOT:?Set SCRATCH_ROOT to a writable scratch directory}"
source "${MODULE_INIT:-/etc/profile.d/z00_lmod.sh}"
export QC QCAUX
source "$QC/bin/qchem.setup.sh"
module purge
module load "${QCHEM_COMPILER_MODULE:-intel/2021.2.0}"
export QCSCRATCH="$SCRATCH_ROOT/tddft_${SLURM_JOB_ID:-manual}"
mkdir -p "$QCSCRATCH"
