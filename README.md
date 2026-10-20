# MPI Exercise 1 — README

## Requirements
- C compiler with MPI support (`mpicc`).
- AION `toolchain/foss` module.
- Standard C and mathematical libraries (`-lm`).
- Provided headers: `Libraries/arrays.h` and `Libraries/linearalgebra.h`.
- Slurm for execution on AION.

## Compilation

On AION, request a compute node:

```bash
salloc -p batch --reservation=mpi_course -N 1 -n 1 --time=00:30:00
```

Load the required module and compile:

```bash
module load toolchain/foss
mpicc -O2 -Wall -o exercise1.exe exercise1.c -lm
```

## Execution

From the project directory, submit the Slurm script:

```bash
sbatch -J exercise1 script.sh
```

The script runs the program with 1, 2, 4, 8, 16, 32, 64, and 128 MPI processes on one AION node.

Results are saved in `out_1`, `out_2`, ..., `out_128`.

## Files

- `exercise1.c`: source code.
- `Libraries/`: required header files.
- `script.sh`: Slurm execution script.
- `out_*`: AION execution results.
