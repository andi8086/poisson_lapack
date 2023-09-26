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


void matrix_mul(lapack_complex_double *c, lapack_complex_double *a, lapack_complex_double *b)
{
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) { 
                        c[i*N + j] = 0.0;
                        for (int k = 0; k < N; k++) {
                                c[i*N + j] += a[i*N + k] * b[k*N + j];
                        }
                }
        }

}


void matrix_adj(lapack_complex_double *c, lapack_complex_double *a)
{
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        c[i*N + j] = conj(a[j*N + i]);
                }
        }
}


void mat_v(lapack_complex_double *b, lapack_complex_double *m, lapack_complex_double *a)
{
        for (int i = 0; i < N; i++) {
                b[i] = 0.0;

                for (int j = 0; j < N; j++) {
                        b[i] += m[i*N + j]*a[j];
                }
        }
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

        /* we need N-1 elements for zhetrd, but one additional for zstegr */
        double *e = malloc(sizeof(double) * (N));

        lapack_complex_double *tau = malloc(sizeof(lapack_complex_double) * (N-1));
        printf("******************");
        print_matrix(m);
        printf("******************");

        int info = LAPACKE_zhetrd(LAPACK_ROW_MAJOR, 'U', N, m, N, d, e, tau);
        printf("zhetrd returned %d\n", info);

        printf("******************");
        print_matrix(m);
        printf("******************");

        /* calculate H(i) = I - tau * v * v**H */

        lapack_complex_double *v = malloc(sizeof(lapack_complex_double) * N);
        lapack_complex_double **H = malloc(sizeof(void *)*(N-1));
        for (int i = 0; i < N-1; i++) {
                H[i] = malloc(sizeof(lapack_complex_double) * N * N);
                for (int j = i+1; j < N; j++) {
                        v[j] = 0.0;
                }
                for (int j = 0; j < i; j++) {
                        v[j] = m[j * N + i+1];
                }
                v[i] = 1.0;
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
                                        H[i][j*N + k] = 0.0;
                                }
                                H[i][j*N + k] -= tau[i] * v[j] * conj(v[k]);
                        }
                }
                printf("tau_%d = %.5f + %.5fi\n", i, creal(tau[i]), cimag(tau[i]));
                printf("H_%d:\n", i);
                print_matrix(H[i]);
        }

        lapack_complex_double *Q = malloc(sizeof(lapack_complex_double) * N * N);
        lapack_complex_double *tmp = malloc(sizeof(lapack_complex_double) * N * N);
        /* create Q = H(2) H(1) H(0) */
        matrix_mul(tmp, H[1], H[0]);
        matrix_mul(Q, H[2], tmp);
        print_matrix(Q);

        /* calculate Q * T * Q**H */

        lapack_complex_double *T = malloc(sizeof(lapack_complex_double)*N*N);
        memset(T, 0, sizeof(lapack_complex_double) * N * N);
        for (int i = 0; i < N; i++) {
                T[i*N + i] = d[i]; 
                if (i < N-1) {
                        T[i*N + i + 1] = e[i];
                        T[(i+1)*N + i] = e[i];
                }
        }

        printf("*****************************\n");
        printf("Tridiagonal matrix:\n");
        print_matrix(T);

        matrix_adj(H[0], Q);
        matrix_mul(tmp, T, H[0]);
        matrix_mul(T, Q, tmp);

        printf("*****************************\n");
        printf("Q T Q^H:\n");
        print_matrix(T);

        printf("*****************************\n");
        printf("Original matrix:\n");
        print_matrix(m_old);

        lapack_int num_eigenvals;
        double eigen_values[N];
        lapack_int issupz[2*N];
        lapack_complex_double *eigen_vecs = malloc(sizeof(lapack_complex_double) * N * N);
        /* Now we can calculate the eigen values and eigen vectors */
        
        int res = LAPACKE_zstegr(LAPACK_ROW_MAJOR, 'V', 'A', N, d, e, 0.0, 0.0,
        0, 0, 0.0, &num_eigenvals, eigen_values, eigen_vecs, N, issupz);
        printf("zstegr returned %d\n", res);

        printf("%d eigen values found.\n", num_eigenvals);

        /* Retransform eigen vectors with Q to match base of A */
        /* Q T Q^H Q v = A Q v = lambda Q v,
           hence  v' = Qv */


        for (int i = 0; i < N; i++) {
                printf("lambda_%d = %.5f\n", i, eigen_values[i]);

                /* store column i of eigen_vecs into row 2 of tmp */
                for (int j = 0; j < N; j++) {
                        tmp[N + j] = eigen_vecs[j*N + i];
                }
                mat_v(tmp, Q, &tmp[N]);

                printf("ev_%d = [ ", i);
                for (int j = 0; j < N; j++) {
                        printf("%.5f + %.5fi", creal(tmp[j]),
                                cimag(tmp[j]));
                        if (j < N-1) {
                                printf(", ");
                        }
                }
                printf(" ]\n");

                /* Test eigenvector */

                mat_v(&tmp[N], m_old, tmp);
                printf("A ev_%d / lambda_%d = [ ", i, i);
                for (int j = 0; j < N; j++) {
                        printf("%.5f + %.5fi", creal(tmp[N+j])/eigen_values[i],
                                cimag(tmp[N+j])/eigen_values[i]);
                        if (j < N-1) {
                                printf(", ");
                        }
                }
                printf(" ]\n");

        }


        free(eigen_vecs);

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
