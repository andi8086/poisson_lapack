#include <stdio.h>
#include <lapacke.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
        int rr = 256;     // Rank.
        int kl = 1;     // Number of lower diagonals.
        int ku = 1;     // Number of upper diagonals.
        int nrhs = 1;   // Number of RHS.
/*
   [    1,     0,     0,     0,    0,    0]     [1]
    [27.50,   -50,  22.5,     0,    0,    0]     [1]
    [    0, 27.50,   -50,  22.5,    0,    0] x = [1]
    [    0,     0, 27.50,   -50, 22.5,    0]     [1]
    [    0,     0,     0, 27.50,  -50, 22.5]     [1]
    [    0,     0,     0,    -1,    4,-2.60]     [0]
    */

        /* vals is band storage,
         * rows KL + 1 to 2*KL + KU + 1   is the matrix
         * rows 1 to KL need not be set.
         * AB(KL+KU+1+i-j,j) = A(i,j)
         * max(1,j-KU) <= i <= min(N,j+KL)
         */

        /* so to transpose immediately, we have
         * cols KL + 1 to 2*KL + KU + 1 is the matrix
         * cols 1 to KL need not be set
         * AB(j,KL+KU+1+i-j) = A(i,j)
         */
        double *mat;
        mat = malloc(rr*rr*sizeof(double));
        memset(mat, 0, rr*rr*sizeof(double));
        for (int k = 0; k < rr; k++) {
                /* diagonal elements, invariant under transpose */
                if (k == 0 || k == rr - 1) {
                        mat[k * rr + (kl + ku)] = 1.0;
                } else
                        mat[k * rr + (kl + ku)] = -2.0;
                if (k > 0) { /* A[i,j-1] */
                        mat[(k-1)*rr + (kl+ku+1)] = 1.0;
                }
                if (k < rr - 1) {
                        mat[(k+1)*rr + (kl+ku-1)] = 1.0;
                }
        }

        double *rhs;

        rhs = malloc(rr*sizeof(double));
        memset(rhs,0,rr*sizeof(double));
        for (int i = 0; i < 100; i++) {
                rhs[i] = -1.0*1.0/(rr*rr);
                rhs[rr-i-1] = 1.0*1.0/(rr*rr);
        }
                                                                // */
      int lda = rr;   // Leading dimension of the matrix.
      int ipiv[rr];    // Information on pivoting array.
      int ldb = lda;  // Leading dimension of the RHS.
      int info = 0;   // Evaluation variable for solution process.
      int ii;         // Iterator.

      dgbsv_(&rr, &kl, &ku, &nrhs, mat, &lda, ipiv, rhs, &ldb, &info);
//      printf("info = %d\n", info);
      for (ii = 0; ii < rr; ii++) {
        printf("%.9f, %.9f\n", 1.0/rr*ii, rhs[ii]);
      }
      putchar('\n');return 0;
}
