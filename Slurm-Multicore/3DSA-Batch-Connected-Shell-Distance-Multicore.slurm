#!/bin/bash




#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 #fallback to 4 cores if not specified by user
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=mc-cql-job.err
#SBATCH --output=mc-cql-job.out
#SBATCH --mem=0
module list

# Dynamically export whatever core count Slurm physically allocated for this task
export CORE_COUNT=$SLURM_CPUS_PER_TASK

DATA_SRC=${data_source:-Feb_cdnc70}
echo "Starting job execution for source: $DATA_SRC"
echo "---> Slurm allocated ${SLURM_CPUS_PER_TASK} cores. Forwarding CORE_COUNT=${CORE_COUNT} to Python."

# Force /usr/bin/python3 to check your personal pip installs first
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

/usr/bin/time /usr/bin/python3 /mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Multicore-Python/3DSA-Batch-Connected-Shell-Distance-Multicore.py --data_source "$DATA_SRC"

