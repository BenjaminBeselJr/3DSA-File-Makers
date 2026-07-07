#!/bin/bash
set -e

# --- Command Line Argument Handling ---
# Usage: ./shell_stats_pipeline.sh [cores] [data_source]
CORES=${1:-4}
DATA_SRC=${2:-default}

echo "Shell & Stats Pipeline Initialized:"
echo " -> Allocated CPU Cores: $CORES"
echo " -> Target Dataset Key:  $DATA_SRC"
echo "---------------------------------------------------------------------"

SHELL_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Binaries-and-Shell-w.slurm"
STATS_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Shell-Cloud-Stats-Multicore.slurm"
SLAB_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Variable-Slab-Averages-Multicore.slurm"
TIME_AVG_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Time-Averaged-Slab-Means-Multicore.slurm"

# ---------------------------------------------------------------------
# Executions with Configuration Mapping Injection
# ---------------------------------------------------------------------
JOB_SHELL=$(sbatch --parsable --cpus-per-task=$CORES --export=ALL,data_source="$DATA_SRC" "$SHELL_SLURM")
echo "✅ Stage 1 Submitted: Shell Creation (Job ID: $JOB_SHELL)"

JOB_STATS=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SHELL --export=ALL,data_source="$DATA_SRC" "$STATS_SLURM")
echo "✅ Stage 2 Queued: Cloud Stats (Job ID: $JOB_STATS)"

JOB_SLAB=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_STATS --export=ALL,data_source="$DATA_SRC" "$SLAB_SLURM")
echo "✅ Stage 3 Queued: Slab Averages (Job ID: $JOB_SLAB)"

JOB_TIME_AVG=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SLAB --export=ALL,data_source="$DATA_SRC" "$TIME_AVG_SLURM")
echo "✅ Stage 4 Queued: Time Averaged Slab Means (Job ID: $JOB_TIME_AVG)"