#include <stdio.h>
#include <openblas64/lapacke.h>
#include <openblas64/cblas.h>
#include <string.h>
#include <stdlib.h>
#include <GL/glut.h>
#include <pthread.h>

pthread_mutex_t psi_mutex;

/* one side of the square */
#define X 64 
/* 2D laplace matrix has X^4 elements (N*N)*/
#define N (X*X)


#define CDOUBLE lapack_complex_double

void glcolor_heatmap(double min, double max, double val)
{
        float r, g, b;
        double scale = max - min;
        double rval = (val - min) / scale;
        if (rval <= 0) {
                glColor3f(0.0, 0.0, 0.0);
                return;
        }

        if (rval >= 1.0) {
                glColor3f(1.0, 1.0, 1.0);
        }

        /* intervals:
        0..1/4  black to blue
        1/4..2/4  blue down, green up
        2/4..3/4  keep green, red up
        3/4..4/4  keep green and red, blue up */
        if (rval < 0.25) {
                glColor3f(0.0, 0.0, rval*4);
        } else if (rval < 0.5) {
                glColor3f((rval - 0.25)*4, 0.0, 1.0-(rval - 0.25)*4);
        } else if (rval < 0.75) {
                glColor3f(1.0, (rval - 0.5)*4,  0.0);
        } else {
                glColor3f(1.0, 1.0, (rval - 0.75)*4);
        }
}

CDOUBLE *solution;

void display_frame(void)
{
//        glDrawBuffer(GL_BACK);
        pthread_mutex_lock(&psi_mutex);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        for (int i = 0; i < X-1; i++) {
                for (int j = 0; j < X-1; j++) {
                        glBegin(GL_QUADS);
                                glcolor_heatmap(-0.25, 0.25,
                                      lapack_complex_double_real(creal(solution[j*X+i])));
                                glVertex3f(1.0/X*(i+1), 1.0/X*j, 0.0);
                                glVertex3f(1.0/X*(i+1), 1.0/X*(j+1), 0.0);
                                glVertex3f(1.0/X*i, 1.0/X*(j+1), 0.0);
                                glVertex3f(1.0/X*i, 1.0/X*j, 0.0);
                        glEnd();
                }

        }
        glutSwapBuffers();
        pthread_mutex_unlock(&psi_mutex);
}



void print_matrix(lapack_complex_double *m, size_t rr )
{
        printf("\n");
        for (size_t i = 0; i < rr; i++) {
                for (size_t j = 0; j < rr; j++) {
                        printf("%.2f + %.2fi  ", creal(m[i*N + j]), cimag(m[i*rr + j]));
                }
                printf("\n");
        }
        printf("\n");
}


void print_matrix_real(lapack_complex_double *m, size_t rr )
{
        printf("\n");
        for (size_t i = 0; i < rr; i++) {
                for (size_t j = 0; j < rr; j++) {
                        printf("%+1.0f  ", creal(m[i*N + j]));
                }
                printf("\n");
        }
        printf("\n");
}


void matrix_mul(lapack_complex_double *c, lapack_complex_double *a, lapack_complex_double *b, size_t rr)
{
        for (size_t i = 0; i < rr; i++) {
                for (size_t j = 0; j < rr; j++) { 
                        c[i*rr + j] = 0.0;
                        for (size_t k = 0; k < rr; k++) {
                                c[i*rr + j] += a[i*rr + k] * b[k*rr + j];
                        }
                }
        }

}


void matrix_adj(lapack_complex_double *c, lapack_complex_double *a, size_t rr)
{
        for (size_t i = 0; i < rr; i++) {
                for (size_t j = 0; j < rr; j++) {
                        c[i*rr + j] = conj(a[j*rr + i]);
                }
        }
}


void mat_v(lapack_complex_double *b, lapack_complex_double *m, lapack_complex_double *a,
           size_t rr)
{
        for (size_t i = 0; i < rr; i++) {
                b[i] = 0.0;

                for (size_t j = 0; j < rr; j++) {
                        b[i] += m[i*rr + j]*a[j];
                }
        }
}


CDOUBLE *malloc_cmatrix(size_t dim)
{
        CDOUBLE *tmp = malloc(sizeof(CDOUBLE) * dim * dim);
        if (!tmp) {
                perror("Out of memory\n");
                exit(-1);
        }
        return tmp;
}


CDOUBLE *malloc_cvector(size_t dim)
{
        CDOUBLE *tmp = malloc(sizeof(CDOUBLE) * dim);
        if (!tmp) {
                perror("Out of memory\n");
                exit(-1);
        }
        return tmp;
}


/* input: dim: dimension of m
          m: hermitian matrix, with NxN elements
   output: Q: transformation matrix
           d: tridiagonal main diagonal elements, N elements
           e: tridiagonal extra diagonal elements, N elements
           ev: NxN matrix with columns = eigenvectors
           l: eigenvalues
           n_ev: found eigen values */
int eigenv(size_t dim, CDOUBLE *m, CDOUBLE *Q, double *d,
           double *e, CDOUBLE *ev, double *l,
           lapack_int *n_ev, lapack_int *issupz,
           lapack_int i_start, lapack_int i_end)
{
        const size_t mat_size = sizeof(lapack_complex_double) * dim * dim;

        CDOUBLE *tau = malloc_cvector(dim-1);
        CDOUBLE *v = malloc_cvector(dim);
        CDOUBLE *H = malloc_cmatrix(dim);
        CDOUBLE *tmp = malloc_cmatrix(dim);

        int info = LAPACKE_zhetrd(LAPACK_ROW_MAJOR, 'U', dim, m, dim, d, e, tau);
        if (info) {
                printf("Tridiagonalization failed.\n");
                free(tmp);
                free(H);
                free(v);
                free(tau);
                return info;
        }

        printf("Tridiagonalization completed.\n");

        memset(Q, 0, mat_size);
        for (size_t i = 0; i < dim; i++) {
                Q[i*dim + i] = 1.0;        
        }
        for (size_t i = 0; i < dim - 1; i++) {
                printf("Calculating Q: %02.0f%%\r", 100.0/(dim-1)*i);
                fflush(stdout);
                for (size_t j = i+1; j < dim; j++) {
                        v[j] = 0.0;
                }
                for (size_t j = 0; j < i; j++) {
                        v[j] = m[j * dim + i+1];
                }
                v[i] = 1.0;

                /* Calculate H_i = I - tau * v * v**H */
                for (size_t j = 0; j < dim; j++) {
                        for (size_t k = 0; k < dim; k++) {
                                if (k == j) {
                                        H[j*dim + k] = 1.0;
                                } else {
                                        H[j*dim + k] = 0.0;
                                }
                                H[j*dim + k] -= tau[i] * v[j] * conj(v[k]);
                        }
                }
                /* Calculate Q' = H_i * Q */
                // matrix_mul(tmp, H, Q, N);
                lapack_complex_double alpha = 1.0;
                lapack_complex_double beta = 0.0;
                memcpy(tmp, Q, mat_size);
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            dim, dim, dim, &alpha,
                            H, dim, tmp, dim, &beta, Q, dim);
        }
        printf("\n");
        info = LAPACKE_zstegr(LAPACK_ROW_MAJOR,
                              'V', 'A',
                              dim, d, e,
                              0.0, 0.0,
                              i_start, i_end,
                              0.0,
                              n_ev, l, ev,
                              dim, issupz);

        free(tmp);
        free(H);
        free(v);
        free(tau);
        return 0;
}


void laplace2d(size_t rr, size_t block_len, CDOUBLE *mat)
{
        /* the matrix is complex, we only set real parts here */

        memset(mat, 0, rr*rr*sizeof(CDOUBLE));
        for (size_t k = 0; k < rr; k+=block_len) {
                /* inner blocks */
                for (size_t l = 0; l < block_len; l++) {
                        if (l > 0) {
                                mat[(k+l)*rr + (k+l-1)] = -1.0;
                        }
                        /* diagonal of inner B matrix */
                        mat[(k+l)*rr + (k+l)] = 4.0;
                        /* upper tridiagonal elements of inner B matrix */
                        if (l < block_len - 1) {
                                mat[(k+l)*rr + (k+l+1)] = -1.0;
                        }
                }
                /* upper pentadiagonal band */
                for (size_t l = 0; l < block_len; l++) {
                        if (k >= block_len) {
                                mat[(k+l)*rr + (k+l-block_len)] = -1.0; 
                        }
                        if (k < rr - block_len) {
                                mat[(k+l)*rr + (k+l+block_len)] = -1.0;
                        }
                }
        }
}


CDOUBLE *eigen_vecs;
CDOUBLE *Q;

int frame_counter = -1;
int solution_i = 0;


void update_frame(int foo)
{

        glutPostRedisplay();

        frame_counter++;
        glutTimerFunc(50, update_frame, 0);
        if (frame_counter % 100) return;
        /* Evaluate time development */
        printf("New vector\n");
        int i = solution_i;
        pthread_mutex_lock(&psi_mutex);

        /* Retransform eigen vectors with Q to match base of A */
        /* Q T Q^H Q v = A Q v = lambda Q v,
           hence  v' = Qv */

           /* i goes from 0 to N-1 */
        //  store column i of eigen_vecs into row 2 of tmp
        for (int j = 0; j < N; j++) {
                solution[N + j] = eigen_vecs[j*N + i];
        }
        mat_v(solution, Q, &solution[N], N);

        pthread_mutex_unlock(&psi_mutex);
        solution_i++;
        if (solution_i == N) {
                solution_i = 0;
        }
}


int main(int argc, char **argv)
{

        CDOUBLE *m = malloc_cmatrix(N);
        laplace2d(N, X, m);


        double *d = malloc(sizeof(double) * N);
        /* we need N-1 elements for zhetrd, but one additional for zstegr */
        double *e = malloc(sizeof(double) * N);

        lapack_int *issupz = malloc(sizeof(lapack_int) * 2 * N);

        double *eigen_values = malloc(sizeof(double) * N);
        lapack_int num_eigenvals;

        eigen_vecs = malloc_cmatrix(N);
        /* Now we can calculate the eigen values and eigen vectors */
        Q = malloc_cmatrix(N); 

        eigenv(N, m, Q, d, e, eigen_vecs, eigen_values, &num_eigenvals, issupz,
               0, 0);
        
        printf("%d eigen values found.\n", num_eigenvals);

        pthread_mutex_init(&psi_mutex, NULL);
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
        glutInitWindowSize(640, 640);
        glutInitWindowPosition(100, 100);
        glutCreateWindow("Laplace Eigenvectors");

        glClearColor(0.0, 0.0, 0.0, 0.0);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
        glutDisplayFunc(display_frame);

        solution = malloc(sizeof(lapack_complex_double) * N * 2);

        update_frame(0);
        glutTimerFunc(50, update_frame, 0);
        glutMainLoop();


        free(solution);

        free(eigen_vecs);

        free(Q);
        free(eigen_values);
        free(issupz);
        free(e);
        free(d);
        free(m);

        return 0;
}

