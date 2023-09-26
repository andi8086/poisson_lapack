#include <stdio.h>
#include <openblas64/lapacke.h>
#include <openblas64/cblas.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 128 

int *ipiv;
size_t rr = N*N;     // Rank.
int kl = 1;     // Number of lower diagonals.
int ku = 1;     // Number of upper diagonals.
int nrhs = 1;   // Number of RHS.
lapack_complex_double *psi;    // complex wave function
lapack_complex_double *psi2;    // complex wave function

void laplace(size_t rr, lapack_complex_double *mat)
{
        /* the matrix is complex, we only set real parts here */

        memset(mat, 0, rr*rr*sizeof(lapack_complex_double));
        for (int k = 0; k < rr; k+=N) {
                /* inner blocks */
                for (int l = 0; l < N; l++) {
                        if (l > 0) {
                                mat[(k+l)*rr + (k+l-1)] = -1.0;
                        }
                        /* diagonal of inner B matrix */
                        mat[(k+l)*rr + (k+l)] = 4.0;
                        /* upper tridiagonal elements of inner B matrix */
                        if (l < N - 1) {
                                mat[(k+l)*rr + (k+l+1)] = -1.0;
                        }
                }
                /* upper pentadiagonal band */
                for (int l = 0; l < N; l++) {
                        if (k > N) {
                                mat[(k+l)*rr + (k+l-N)] = -1.0; 
                        }
                        if (k < rr - N) {
                                mat[(k+l)*rr + (k+l+N)] = -1.0;
                        }
                }
        }
}


void laplace_2d(lapack_complex_double *psi2, size_t rr, lapack_complex_double *psi)
{
        memset(psi2, 0, sizeof(lapack_complex_double)*rr);
        for (int k = 0; k < rr; k+=N) {
                /* inner blocks */
                for (int l = 0; l < N; l++) {
                        if (l > 0) {
                                psi2[k+l] += -1.0*psi[k+l-1];
                                //mat[(k+l)*rr + (k+l-1)] = -1.0;
                        }
                        /* diagonal of inner B matrix */
                        
                        psi2[k+l] += 4.0*psi[k+l];
//                        mat[(k+l)*rr + (k+l)] = 4.0;
                        /* upper tridiagonal elements of inner B matrix */
                        if (l < N - 1) {
                                psi2[k+l] += -1.0*psi[k+l+1];
                                //mat[(k+l)*rr + (k+l+1)] = -1.0;
                        }
                }
                /* upper pentadiagonal band */
                for (int l = 0; l < N; l++) {
                        if (k > N) {
                                psi2[k+l] += -1.0 * psi[k+l-N];
                                //mat[(k+l)*rr + (k+l-N)] = -1.0; 
                        }
                        if (k < rr - N) {
                                psi2[k+l] += -1.0 * psi[k+l+N];
                                //mat[(k+l)*rr + (k+l+N)] = -1.0;
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
                        fprintf(dat, "%.6f ", 1.0/rr *
                                lapack_complex_double_real(
                                        v[iy*N+ix]*conj(v[iy*N+ix])));
                }
                fprintf(dat, "\n");
        }
        fprintf(dat, "\n\n");
}


void potential(lapack_complex_double *psi, size_t rr)
{
        size_t idx;
        for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                        if (i == N/2) {
                                if ((j > N/3-N/16 && j < N/3+N/16)
                                        || (j > 2*N/3-N/16 && j < 2*N/3+N/16)) {
                                        idx = j*N + i;
                                        psi[idx] -= 1.0*psi[idx];
                                        //mat[idx * rr + idx] -= 1.0;
                                }
                        }
                }
        }
}


void timestep(lapack_complex_double *psi, lapack_complex_double *psi2,
              double dtr)
{
        lapack_complex_double dt = lapack_make_complex_double(0.0, dtr);
        for (size_t k = 0; k < rr; k++) {
                psi[k] += psi2[k]*dt;
        }
}

int main(void)
{

       // lapack_complex_double *mat;
       // mat = malloc(rr*rr*sizeof(lapack_complex_double));
       /* if (!mat) {
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
        potential(rr, mat);
        */
        psi2 = malloc(rr * sizeof(lapack_complex_double));
        psi = malloc(rr*sizeof(lapack_complex_double));
        if (!psi) {
                fprintf(stderr, "Out of memory\n");
                return -1;
        }
        memset(psi,0,rr*sizeof(lapack_complex_double));
        memset(psi2, 0, rr*sizeof(lapack_complex_double));
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

        gaussian_pulse(psi, 0.2, 0.5, 0.5, 0.02, -100000.0);

        char filename[256];
/* d psi = laplace psi dt */
        /* normalize wave function with beta */

        lapack_complex_double beta = 1;
        lapack_complex_double dt = lapack_make_complex_double(0, 0.001);
        sprintf(filename, "out/psi.dat");
        FILE *dat = fopen(filename, "w");
        for (int i = 0; i < 100000; i++) {
                laplace_2d(psi2, rr, psi);
                potential(psi2, rr);
                timestep(psi, psi2, 0.001);

                /*cblas_zgemv(CblasColMajor, CblasNoTrans, rr, rr, &dt, mat,
                      lda, psi, 1, &beta, psi, 1); */
                double norm = 0.0;
                for (int i = 0; i < rr; i++) {
                        /* we sum over  (psi*_ij psi_ij dx dy) */
                        norm += conj(psi[i])*psi[i]*1.0/rr;
                }
                for (int i = 0; i < rr; i++) {
                        psi[i] /= norm;
                }
                if (i % 200 == 0) {
                        psi_to_file(dat, psi);
                }
                printf("time step %d\n", i);
        }
        fclose(dat);

        free(psi2);
        free(psi);
        free(ipiv);

        return 0;

}
