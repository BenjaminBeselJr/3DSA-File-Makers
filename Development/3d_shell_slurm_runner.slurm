#!/bin/bash




#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --partition=long
#SBATCH --error=job.err
#SBATCH --output=job.out
#SBATCH --mem=0
module list

/usr/bin/time /usr/bin/python3 /mnt/stor-pool-01/users/2821011/3d_shell_analysis_file_maker_6_9_2026.py

