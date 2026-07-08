#!/bin/bash
set -e

# --- Command Line Argument Handling ---
# Usage: ./master_pipeline.sh [cores] [data_source]
CORES=${1:-4}
DATA_SRC=${2:-default}  # Falls back to "default" if omitted

echo "Master Pipeline Initialized:"
echo " -> Allocated CPU Cores: $CORES"
echo " -> Target Dataset Key:  $DATA_SRC"
echo "---------------------------------------------------------------------"

# Script Paths
SHELL_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Binaries-and-Shell-w.slurm"
STATS_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Shell-Cloud-Stats-Multicore.slurm"
SLAB_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Variable-Slab-Averages-Multicore.slurm"
TIME_AVG_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Time-Averaged-Slab-Means-Multicore.slurm"

KE_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-KE-All-Multicore.slurm"
VPG_BB_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-VPG-and-VPGB-Multicore.slurm"
VPG_DN_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-VPG-DN-Multicore.slurm"
VPG_DL_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-VPG-DL-Multicore.slurm"

NEAREST_QL_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Nearest-Cloud-Shell-Distance-Multicore.slurm"
CONNECTED_QL_SLURM="/mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Slurm-Multicore/3DSA-Batch-Connected-Shell-Distance-Multicore.slurm"

# ---------------------------------------------------------------------
# STAGE 1: Root Tasks (No Dependencies)
# ---------------------------------------------------------------------
# Use --export=ALL,data_source="..." to forward configurations into Slurm
JOB_VPG_BB=$(sbatch --parsable --cpus-per-task=$CORES --export=ALL,data_source="$DATA_SRC" "$VPG_BB_SLURM")
echo "✅ Root Task Submitted: VPG B and B (Job ID: $JOB_VPG_BB)"

JOB_VPG_DN=$(sbatch --parsable --cpus-per-task=$CORES --export=ALL,data_source="$DATA_SRC" "$VPG_DN_SLURM")
echo "✅ Root Task Submitted: VPG DN     (Job ID: $JOB_VPG_DN)"

JOB_VPG_DL=$(sbatch --parsable --cpus-per-task=$CORES --export=ALL,data_source="$DATA_SRC" "$VPG_DL_SLURM")
echo "✅ Root Task Submitted: VPG DL     (Job ID: $JOB_VPG_DL)"

JOB_SHELL=$(sbatch --parsable --cpus-per-task=$CORES --export=ALL,data_source="$DATA_SRC" "$SHELL_SLURM")
echo "✅ Root Task Submitted: Shell Creation (Job ID: $JOB_SHELL)"

# ---------------------------------------------------------------------
# STAGE 2: Tasks dependent on Root Tasks
# ---------------------------------------------------------------------
JOB_STATS=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SHELL --export=ALL,data_source="$DATA_SRC" "$STATS_SLURM")
echo "✅ Stage 2A Queued: Cloud Stats (Job ID: $JOB_STATS, waits for $JOB_SHELL)"

JOB_KE=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SHELL:$JOB_VPG_BB:$JOB_VPG_DN:$JOB_VPG_DL --export=ALL,data_source="$DATA_SRC" "$KE_SLURM")
echo "✅ Stage 2B Queued: Kinetic Energy calculation (Job ID: $JOB_KE)"

JOB_NEAR_QL=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SHELL --export=ALL,data_source="$DATA_SRC" "$NEAREST_QL_SLURM")
echo "✅ Stage 2C Queued: Nearest ql shell distance calculation (Job ID: $JOB_NEAR_QL)"

JOB_CONN_QL=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SHELL --export=ALL,data_source="$DATA_SRC" "$CONNECTED_QL_SLURM")
echo "✅ Stage 2D Queued: Connected ql shell distance (Job ID: $JOB_CONN_QL)"

# ---------------------------------------------------------------------
# STAGE 3: Slab averages
# ---------------------------------------------------------------------
JOB_SLAB=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_STATS:$JOB_KE --export=ALL,data_source="$DATA_SRC" "$SLAB_SLURM")
echo "✅ Stage 3 Queued: Slab Averages (Job ID: $JOB_SLAB)"

# ---------------------------------------------------------------------
# STAGE 4: Time Averaged Slab Means
# ---------------------------------------------------------------------
JOB_TIME_AVG=$(sbatch --parsable --cpus-per-task=$CORES --dependency=afterok:$JOB_SLAB --export=ALL,data_source="$DATA_SRC" "$TIME_AVG_SLURM")
echo "✅ Stage 4 Queued: Time Averaged Slab Means (Job ID: $JOB_TIME_AVG)"
echo "---------------------------------------------------------------------"
echo "Pipeline successfully linked! Track progress using 'squeue -u \$USER'"