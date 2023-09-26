#include <stdio.h>
#include <openblas64/lapacke.h>
#include <string.h>
#include <stdlib.h>

#define N 4

void print_matrix(lapack_complex_double *m)
{
        printf("\n");
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        printf("%.5f + %.5fi   ", creal(m[i*N + j]), cimag(m[i*N + j]));
                }
                printf("\n");
        }
        printf("\n");
}


int main(void)
{

        lapack_complex_double *m;

        m = malloc(sizeof(lapack_complex_double) * N * N);

        m[0] = 2.3;
        m[1] = lapack_make_complex_double(1.0, -3.3);
        m[2] = lapack_make_complex_double(0, -2.0);
        m[3] = 1.7;

        m[4] = lapack_make_complex_double(1.0, 3.3);
        m[5] = 0.5;
        m[6] = 7.0;
        m[7] = 0.0;

        m[8] = lapack_make_complex_double(0, 2.0);
        m[9] = 7.0;
        m[10] = 0.5;
        m[11] = -1.0;

        m[12] = 1.7;
        m[13] = 0.0;
        m[14] = -1.0;
        m[15] = 3.0;

        lapack_complex_double *m_old = malloc(sizeof(lapack_complex_double) * N * N);
        memcpy(m_old, m, sizeof(lapack_complex_double) * N * N);

        /* A is Hermitian */
        /* find T so that Q**H * A * Q = T and T is real sym tridiag */
        double *d = malloc(sizeof(double) * (N));
        double *e = malloc(sizeof(double) * (N-1));
        lapack_complex_double *tau = malloc(sizeof(lapack_complex_double) * (N-1));

        int info = LAPACKE_zhetrd(LAPACK_COL_MAJOR, 'L', N, m, N, d, e, tau);
        printf("zhetrd returned %d\n", info);

        /* calculate H(i) = I - tau * v * v**H */

        lapack_complex_double *v = malloc(sizeof(lapack_complex_double) * N);
        lapack_complex_double **H = malloc(sizeof(void *)*(N-1));
        for (int i = 0; i < N-1; i++) {
                H[i] = malloc(sizeof(lapack_complex_double) * N * N);
        }
        for (int i = 0; i < N-1; i++) {
                printf("H(%d): \n", i);
                for (int j = 0; j <= i; j++) {
                        v[j] = 0.0;
                }
                v[i+1] = 1.0;
                for (int j = i+2; j < N; j++) {
                        v[j] = m[j * N + i];
                }
                printf("v_%d = [%.5f + %.5fi, %.5f + %.5fi, %.5f + %.5fi, "
                                "%.5f + %.5fi]**T\n", i,
                     creal(v[0]), cimag(v[0]),
                     creal(v[1]), cimag(v[1]),
                     creal(v[2]), cimag(v[2]),
                     creal(v[3]), cimag(v[3]));

                /* Calculate I - tau * v * v**H */
                for (int j = 0; j < N; j++) {
                        for (int k = 0; k < N; k++) {
                                if (k == j) {
                                        H[i][j*N + k] = 1.0;
                                } else {
                                        H[i][j+N + k] = 0.0;
                                }
                                H[i][j*N + k] -= tau[i] * v[j] * conj(v[k]);
                        }
                }

                print_matrix(H[i]);
        }

        lapack_complex_double *Q = malloc(sizeof(lapack_complex_double) * N * N);
        lapack_complex_double *tmp = malloc(sizeof(lapack_complex_double) * N * N);
        /* create Q = H(1) H(2) H(3) */
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        tmp[i*N + j] = 0.0;
                        for (int k = 0; k < N; k++) {
                                tmp[i*N + j] += H[0][i*N + k]*H[1][(k*N)+j];
                        }
                }
        }
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        Q[i*N + j] = 0.0;
                        for (int k = 0; k < N; k++) {
                                Q[i*N + j] += tmp[i*N + k]*H[2][(k*N)+j];
                        }
                }
        }

        print_matrix(Q);

        /* calculate Q**H * A * Q */

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        tmp[i*N + j] = 0.0;
                        for (int k = 0; k < N; k++) {
                                tmp[i*N + j] += m_old[i*N + k]*Q[(k*N)+j];
                        }
                }
        }

        lapack_complex_double *T = malloc(sizeof(lapack_complex_double)*N*N);

        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        T[i*N + j] = 0.0;
                        for (int k = 0; k < N; k++) {
                                T[i*N + j] += conj(Q[k*N + i])*tmp[(k*N)+j];
                        }
                }
        }

        print_matrix(T);

        free(T);
        free(m_old);
        free(tmp);
        free(Q);

        for (int i = 0; i < N-1; i++) {
                free(H[i]);
        }
        free(H);
        free(v);
        free(tau);
        free(e);
        free(d);
        free(m);

        return 0;
}
