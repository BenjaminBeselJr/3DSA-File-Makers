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
SLAB_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Body-Entrainment-Multicore.slurm"
SLAB_STATS="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Body-Entrainment-Stats-Multicore.slurm"

# ---------------------------------------------------------------------
# Executions with Configuration Mapping Injection
# ---------------------------------------------------------------------
JOB_SHELL=$(sbatch --parsable --cpus-per-task=$CORES --export=ALL,data_source="$DATA_SRC" "$SHELL_SLURM")
echo "✅ Stage 1 Submitted: Shell Creation (Job ID: $JOB_SHELL)"

JOB_STATS=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SHELL --export=ALL,data_source="$DATA_SRC" "$STATS_SLURM")
echo "✅ Stage 2A Queued: Cloud Stats (Job ID: $JOB_STATS)"

JOB_SLAB=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SHELL --export=ALL,data_source="$DATA_SRC" "$SLAB_SLURM")
echo "✅ Stage 2B Queued: Entrainment (Job ID: $JOB_SLAB)"

JOB_SLAB_STATS=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SLAB:$JOB_STATS --export=ALL,data_source="$DATA_SRC" "$SLAB_STATS")
echo "✅ Stage 3 Queued: Entrainment Stats (Job ID: $JOB_SLAB_STATS)"