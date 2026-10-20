#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node 128
#SBATCH --cpus-per-task 1
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --reservation=mpi_course
#SBATCH --exclusive

module load toolchain/foss
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for i in 1 2 4 8 16 32 64 128; do
    srun -n $i ./exercise1.exe > out_$i
done
