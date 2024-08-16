#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=results/CHD_%a.out
#SBATCH --error=/dev/null
#SBATCH --array=0-100

module load julia
srun julia slurmjob_CHD.jl $SLURM_ARRAY_TASK_ID