#!/bin/bash
set -e

# --- Core Count Handling ---
# This uses bash parameter expansion: if $1 is not set, default to 4 cores.
CORES=${1:-4}

echo "Selected Core Count: $CORES"
echo "---------------------------------------------------------------------"

# Define your slurm script file paths
SHELL_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Binaries-and-Shell-w.slurm"
STATS_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Shell-Cloud-Stats-Multicore.slurm"
SLAB_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Variable-Slab-Averages-Multicore.slurm"
TIME_AVG_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Time-Averaged-Slab-Means-Multicore.slurm"

# ---------------------------------------------------------------------
# STAGE 1: Shell Creation
# ---------------------------------------------------------------------
JOB_SHELL=$(sbatch --parsable --cpus-per-task=$CORES "$SHELL_SLURM")
echo "✅ Stage 1 Submitted: Shell Creation (Job ID: $JOB_SHELL using $CORES cores)"

# ---------------------------------------------------------------------
# STAGE 2: Cloud Stats
# ---------------------------------------------------------------------
JOB_STATS=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SHELL "$STATS_SLURM")
echo "✅ Stage 2 Queued: Cloud Stats (Job ID: $JOB_STATS, waits for $JOB_SHELL)"

# ---------------------------------------------------------------------
# STAGE 3: Slab Averages
# ---------------------------------------------------------------------
JOB_SLAB=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_STATS "$SLAB_SLURM")
echo "✅ Stage 3 Queued: Slab Averages (Job ID: $JOB_SLAB, waits for $JOB_STATS)"

# ---------------------------------------------------------------------
# STAGE 4: Time Averaged Slab Means
# ---------------------------------------------------------------------
JOB_TIME_AVG=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SLAB "$TIME_AVG_SLURM")
echo "✅ Stage 4 Queued: Time Averaged Slab Means (Job ID: $JOB_TIME_AVG, waits for $JOB_SLAB)"

echo "---------------------------------------------------------------------"
echo "Pipeline successfully linked! Use 'squeue -u \$USER' to monitor tracking status."