#include <stdio.h>
#include <openblas64/lapacke.h>
#include <stdlib.h>
#include <string.h>

#define N 128

int *ipiv;
size_t rr = N*N;     // Rank.
int kl = 1;     // Number of lower diagonals.
int ku = 1;     // Number of upper diagonals.
int nrhs = 1;   // Number of RHS.
double *rhs;

int main(void)
{

        double *mat;
        mat = malloc(rr*rr*sizeof(double));
        if (!mat) {
                fprintf(stderr, "Out of memory\n");
                return -1;
        }
        ipiv = malloc(rr * sizeof(int));
        if (!ipiv) {
                fprintf(stderr, "Out of memory\n");
                free(mat);
                return -1;
        }
        memset(mat, 0, rr*rr*sizeof(double));
        for (int k = 0; k < rr; k+=N) {
                /* inner blocks */
                for (int l = 0; l < N; l++) {
                        /* diagonal of inner B matrix */
                        mat[(k+l)*rr + (k+l)] = 4.0;
                        /* tridiagonal elements of inner B matrix */
                        if (l > 0) {
                                mat[(k+l)*rr + (k+l-1)] = -1.0;
                        }
                        if (l < N - 1) {
                                mat[(k+l)*rr + (k+l+1)] = -1.0;
                        }
                }
                /* upper pentadiagonal band */
                for (int l = 0; l < N; l++) {
                        if (k > 0) {
                                mat[(k+l)*rr + (k+l-N)] = -1.0;
                        }
                        if (k < rr - N) {
                                mat[(k+l)*rr + (k+l+N)] = -1.0;
                        }
                }
        }
/*
        printf("\n");
        for (int i = 0; i < rr; i++) {
                for (int j = 0; j < rr; j++) {
                        printf("%1.0f ", mat[i*rr + j]);
                }
                printf("\n");
        }
*/

        rhs = malloc(rr*sizeof(double));
        if (!rhs) {
                fprintf(stderr, "Out of memory\n");
                free(ipiv);
                free(mat);
                return -1;
        }
        memset(rhs,0,rr*sizeof(double));
/*        for (int i = 0; i < 100; i++) {
                rhs[i] = -1.0*1.0/(rr*rr);
                rhs[rr-i-1] = 1.0*1.0/(rr*rr);
        }
        */

        rhs[(N/2-1)*N + N/4-1] = 1.0;
        rhs[(N/2-1)*N + 3*N/4-1] = 1.0;
                                                                // */
        int lda = rr;   // Leading dimension of the matrix.
        int ldb = lda;  // Leading dimension of the RHS.
        int info = 0;   // Evaluation variable for solution process.
        int ii;         // Iterator.

        int res = LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', rr, nrhs, mat, lda, ipiv, rhs, ldb);
//      printf("res = %d\n", res);
//      printf("info = %d\n", info);
        for (int iy = 0; iy < N; iy++) {
                if (iy == 0) {
                        printf("%d ", N);
                        for (int ix = 0; ix < N; ix++) {
                                        double x = 1.0 / N * ix;
                                        printf("%.6f ", x);

                        }
                        printf("\n");
                }
                double y = 1.0 / N * iy;
                printf("%.6f ", y);
                for (int ix = 0; ix < N; ix++) {
                        printf("%.6f ", rhs[iy*N+ix]);
                }
                printf("\n");
        }
        free(rhs);
        free(ipiv);
        free(mat);

        return 0;

}
