#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 #fallback to 4 cores if not specified by user
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --mem=128G

#SBATCH --job-name=body-entrainment-stats
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module list

# Resolve data source
DATA_SRC=${data_source:-Feb_cdnc70}

# Dynamically export whatever core count Slurm physically allocated for this task
export CORE_COUNT=$SLURM_CPUS_PER_TASK

# Force /usr/bin/python3 to check your personal pip installs first
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:/usr/local/lib64/python3.9/site-packages:/usr/local/lib/python3.9/site-packages:$PYTHONPATH

# Create log directory
LOG_DIR="$PWD/log-${DATA_SRC}"
mkdir -p "$LOG_DIR"

echo "Starting job execution for source: $DATA_SRC"
echo "---> Slurm allocated ${SLURM_CPUS_PER_TASK} cores. Forwarding CORE_COUNT=${CORE_COUNT} to Python."
echo "---> Runtime text tracking directed to: $LOG_DIR"

/usr/bin/time /usr/bin/python3 /mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Multicore-Python/3DSA-Body-Entrainment-Stats-Multicore.py --data_source "$DATA_SRC"\
  > "${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
  2> "${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

