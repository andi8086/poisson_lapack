#include <stdio.h>
#include <openblas64/lapacke.h>
#include <openblas64/cblas.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 32 

int *ipiv;
size_t rr = N*N;     // Rank.
int kl = 1;     // Number of lower diagonals.
int ku = 1;     // Number of upper diagonals.
int nrhs = 1;   // Number of RHS.
lapack_complex_double *psi;    // complex wave function

void laplace(size_t rr, lapack_complex_double *mat)
{
        /* the matrix is complex, we only set real parts here */

        memset(mat, 0, rr*rr*sizeof(lapack_complex_double));
        for (int k = 0; k < rr; k+=N) {
                /* inner blocks */
                for (int l = 0; l < N; l++) {
                        /* diagonal of inner B matrix */
                        mat[(k+l)*rr + (k+l)] = 4.0;
                        /* upper tridiagonal elements of inner B matrix */
                        if (l < N - 1) {
                                mat[(k+l)*rr + (k+l+1)] = -1.0;
                        }
                }
                /* upper pentadiagonal band */
                for (int l = 0; l < N; l++) {
                        if (k < rr - N) {
                                mat[(k+l)*rr + (k+l+N)] = -1.0;
                        }
                }
        }
}


void gaussian_pulse(lapack_complex_double *psi,
        double x0, double y0, double A, double w,
        double k)
{

        for (int yi = 0; yi < N; yi++) {
                for (int xi = 0; xi < N; xi++) {
                        /* calculate distance from x,y */
                        double x = xi * 1.0/N;
                        double y = yi * 1.0/N;

                        double psi_e = A*exp(-(x-x0)*(x-x0)/w)*
                                        exp(-(y-y0)*(y-y0)/w);
                        double psi_ij = lapack_make_complex_double(
                                psi_e*cos(k*(x-x0)), psi_e*sin(k*(x-x0)));

                        psi[yi * N + xi] = psi_ij;
                }
        }
}


void psi_to_file(FILE *dat, lapack_complex_double *psi)
{
        for (int iy = 0; iy < N; iy++) {
                if (iy == 0) {
                        fprintf(dat, "%d ", N);
                        for (int ix = 0; ix < N; ix++) {
                                        double x = 1.0 / N * ix;
                                        fprintf(dat, "%.6f ", x);

                        }
                        fprintf(dat, "\n");
                }
                double y = 1.0 / N * iy;
                fprintf(dat, "%.6f ", y);
                for (int ix = 0; ix < N; ix++) {
                        fprintf(dat, "%.6f ", psi[iy*N+ix]*conj(psi[iy*N+ix]));
                }
                fprintf(dat, "\n");
        }
        fprintf(dat, "\n\n");
}


int main(void)
{

        lapack_complex_double *mat;
        mat = malloc(rr*rr*sizeof(lapack_complex_double));
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

        laplace(rr, mat);

        psi = malloc(rr*sizeof(lapack_complex_double));
        if (!psi) {
                fprintf(stderr, "Out of memory\n");
                free(ipiv);
                free(mat);
                return -1;
        }
        memset(psi,0,rr*sizeof(lapack_complex_double));

        psi[(N/2-1)*N + N/4-1] = 1.0;
        psi[(N/2-1)*N + 3*N/4-1] = 1.0;
                                                                // */
        int lda = rr;   // Leading dimension of the matrix.
        int ldb = lda;  // Leading dimension of the RHS.
        int info = 0;   // Evaluation variable for solution process.
        int ii;         // Iterator.

        /* everything seems to be invertedly ordered in memory, so we
         * have to tell LAPACK that we use the lower part of the
         * symmetric matrix, and also column major order, although
         * in C we have row major order */
/*        int res = LAPACKE_zsysv(
                LAPACK_COL_MAJOR, 'L', rr, nrhs, mat, lda, ipiv, psi, ldb); */
//      printf("res = %d\n", res);
//      printf("info = %d\n", info);

        gaussian_pulse(psi, 0.5, 0.5, 1.0, 0.005, -100.0);

        char filename[256];
/* d psi = laplace psi dt */
        /* normalize wave function with beta */

        lapack_complex_double beta = -1.0;
        lapack_complex_double dt = lapack_make_complex_double(0.0, 0.01);
        sprintf(filename, "out/psi.dat");
        FILE *dat = fopen(filename, "w");
        for (int i = 0; i < 10000; i++) {
                cblas_zgemv(CblasColMajor, CblasNoTrans, rr, rr, &dt, mat,
                      lda, psi, 1, &beta, psi, 1);
                double norm = 0.0;
                for (int i = 0; i < rr; i++) {
                        norm += conj(psi[i])*psi[i];
                }
                for (int i = 0; i < rr; i++) {
                        psi[i] /= norm;
                }
                if (i % 20 == 0) {
                        psi_to_file(dat, psi);
                }
                printf("time step %d\n", i);
        }
        fclose(dat);

        free(psi);
        free(ipiv);
        free(mat);

        return 0;

}
