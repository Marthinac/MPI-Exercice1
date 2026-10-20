#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include "Libraries/arrays.h"
#include "Libraries/linearalgebra.h"


int main(int argc, char *argv[])
{
   // Initialize the MPI environment
   MPI_Init(NULL, NULL);
   int num_of_ranks;
   MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
   int mpi_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

   // Measure initial system time
   double t0 = MPI_Wtime();

   // Initialize Matrix A and B and vector c
   int rowsA = 128000;
   int colsA = 500;
   int seedA = 10;

   int rowsB = 500;
   int colsB = 1000;
   int seedB = 20;

   int elemC = 1000;
   int seedC = 30;

   // Compute the number of rows that must be distributed on each rank.
   int num_of_matrix_elements_per_rank = (int) rowsA / num_of_ranks;
   int num_of_residual_elements = (int) rowsA % num_of_ranks;

   double *vector_in = NULL; //c
   double *vector_interm; //v = Bc
   double **matrixA_in; //A
   double **matrixB_in = NULL; //B
   double *vector_out; //d = Av

   vector_interm = allocate_1d_double(&rowsB);

   // Initialize "local" and "initial" matrices. A, B and c are initialized by rank 0
   if (mpi_rank == 0)
   {
      vector_in = allocate_1d_double(&elemC);
      matrixA_in = allocate_2d_packed_double(&rowsA, &colsA);
      matrixB_in = allocate_2d_packed_double(&rowsB, &colsB);
      init_random_matrix(seedA, &rowsA, &colsA, matrixA_in);
      init_random_matrix(seedB, &rowsB, &colsB, matrixB_in);
      init_random_vector(seedC, &elemC, vector_in);
   } else {
      matrixA_in = allocate_2d_packed_double(&num_of_matrix_elements_per_rank, &colsA);
   }

   // Measure time before mathematical operations
   double t1 = MPI_Wtime();

   // Compute v = Bc on rank 0
   if (mpi_rank == 0)
   {
      dgemv(&rowsB, &colsB, matrixB_in, vector_in, vector_interm);
   }

   // Bcast intermediate vector v to all ranks
   MPI_Bcast(vector_interm, rowsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   // Distribute matrix A to all ranks.
   if (mpi_rank == 0) {
      for (int i = 1; i< num_of_ranks; i++){
         MPI_Send( &matrixA_in[i*num_of_matrix_elements_per_rank+num_of_residual_elements][0], num_of_matrix_elements_per_rank*colsA, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
      }
   } else {
      MPI_Recv( &matrixA_in[0][0], num_of_matrix_elements_per_rank*colsA, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   }

   // Rank 0 has more elements of the matrix
   if (mpi_rank == 0)
   {
      num_of_matrix_elements_per_rank += num_of_residual_elements;
   }

   // Allocate "local" output vector
   vector_out = allocate_1d_double(&num_of_matrix_elements_per_rank);

   // Compute "local" dgemv operation
   dgemv(&num_of_matrix_elements_per_rank, &colsA, matrixA_in, vector_interm, vector_out);

   // Compute "local dot product"
   double norm = ddot(&num_of_matrix_elements_per_rank, vector_out, vector_out);

   // Measure time after mathematical operations
   double t2 = MPI_Wtime();

   // Reduce valuse of the "norm" they contain the local dot products
   if (mpi_rank == 0)
   {
      MPI_Reduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   }
   else
   {
      MPI_Reduce(&norm, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   }

   // Rank 0 computes norm (sqrt of the dot product)
   if (mpi_rank==0){
      norm = sqrt(norm);
      printf("Norm of the output vector is : %4.10f\n", norm);
   }

   // Final time
   double t3 = MPI_Wtime();

   // Compute time differences and print them
   double dt0 = t1 - t0; // Time taken for initialization
   double dt1 = t2 - t1; // Time taken for calculation (Matrix Operations)
   double dt2 = t3 - t2; // Time taken for reduction (Reduce Operations)
   double dt3 = t3 - t0; // Total time

   if (mpi_rank == 0)
   {
      MPI_Reduce(MPI_IN_PLACE, &dt0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, &dt1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, &dt2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, &dt3, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   }
   else
   {
      MPI_Reduce(&dt0, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&dt1, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&dt2, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&dt3, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   }

   if (mpi_rank==0){
      printf("Average initialization time per rank: %f seconds\n", dt0/((double)num_of_ranks));
      printf("Average calculation time per rank   : %f seconds\n", dt1/((double)num_of_ranks));
      printf("Average reduction time per rank     : %f seconds\n", dt2/((double)num_of_ranks));
      printf("Average total time per rank         : %f seconds\n", dt3/((double)num_of_ranks));
   }

   MPI_Finalize();
   return 0;
}
