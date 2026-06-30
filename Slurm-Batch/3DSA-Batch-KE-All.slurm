#!/bin/bash




#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --partition=long
#SBATCH --error=job.err
#SBATCH --output=job.out
#SBATCH --mem=0
module list

# Force /usr/bin/python3 to check your personal pip installs first
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

/usr/bin/time /usr/bin/python3 /mnt/stor-pool-01/users/2821011/3DSA-File-Makers/Python-Code/3DSA-Batch-KE-All.py

