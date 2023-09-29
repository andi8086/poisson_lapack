#include <stdio.h>
#include <openblas64/lapacke.h>
#include <openblas64/cblas.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <GL/glut.h>

#define DT 0.001
#define N 128

int *ipiv;
size_t rr = N*N;     // Rank.
int kl = 1;     // Number of lower diagonals.
int ku = 1;     // Number of upper diagonals.
int nrhs = 1;   // Number of RHS.
lapack_complex_double *psi;    // complex wave function
lapack_complex_double *psi2;    // complex wave function
pthread_mutex_t psi_mutex;
lapack_complex_double *V;


void laplace(size_t rr, lapack_complex_double *mat)
{
        /* the matrix is complex, we only set real parts here */

        memset(mat, 0, rr*rr*sizeof(lapack_complex_double));
        for (int k = 0; k < rr; k+=N) {
                /* inner blocks */
                for (int l = 0; l < N; l++) {
                        /* diagonal of inner B matrix */
                        mat[(k+l)*rr + (k+l)] = 4.0;
                        /* tridiagonal bands */ 
                        if (l > 0) {
                                mat[(k+l)*rr + (k+l-1)] = -1.0;
                        }
                        if (l < N - 1) {
                                mat[(k+l)*rr + (k+l+1)] = -1.0;
                        }
                }
                /* pentadiagonal bands */
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


void psi_to_file(FILE *dat, lapack_complex_double *v)
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


void potential(lapack_complex_double *V, size_t rr)
{
        size_t idx;
        for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                        if (i == N/2) {
                                if ((j > 0 && j < 2*N/5-N/32)
                                        ||
                                     (j > 2*N/5+N/32 && j < 3*N/5-N/32) ||
                                     (j > 3*N/5+N/32)) {
                                        idx = j*N + i;
                                        V[idx] = 1.0;
                                }
                        }
                }
        }
}


void visualize_potential(lapack_complex_double *vpsi, lapack_complex_double *psi, size_t rr)
{
        size_t idx;
        memcpy(vpsi, psi, rr*sizeof(lapack_complex_double));
        //memset(vpsi, 0, rr*sizeof(lapack_complex_double));
        int i = N/2;
        for (int j = 0; j < N; j++) {
                if ((j > 0 && j < 2*N/5-N/32)
                        ||
                     (j > 2*N/5+N/32 && j < 3*N/5-N/32) ||
                     (j > 3*N/5+N/32)) {
                        idx = j*N + i;
                        vpsi[idx] = rr;
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


void glcolor_heatmap(double min, double max, double val)
{
        float r, g, b;
        double scale = max - min;
        double rval = val / scale + min;
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


void display_frame(void)
{
//        glDrawBuffer(GL_BACK);
        pthread_mutex_lock(&psi_mutex);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        for (int i = 1; i < N; i++) {
                for (int j = 1; j < N; j++) {
                        glBegin(GL_QUADS);
                                glcolor_heatmap(0.0, 0.2,
                                      lapack_complex_double_real(conj(psi2[j*N+i])*psi2[j*N+i])*1.0/rr);
                                glVertex3f(1.0/N*(i+1), 1.0/N*j, 0.0);
                                glVertex3f(1.0/N*(i+1), 1.0/N*(j+1), 0.0);
                                glVertex3f(1.0/N*i, 1.0/N*(j+1), 0.0);
                                glVertex3f(1.0/N*i, 1.0/N*j, 0.0);
                        glEnd();
                                if (V[j*N+i] == 1.0) {
                                        glBegin(GL_QUADS);
                                        glColor3f(1.0, 1.0, 1.0);
                                        glVertex3f(1.0/N*(i+1), 1.0/N*j, 0.0);
                                        glVertex3f(1.0/N*(i+1), 1.0/N*(j+1), 0.0);
                                        glVertex3f(1.0/N*i, 1.0/N*(j+1), 0.0);
                                        glVertex3f(1.0/N*i, 1.0/N*j, 0.0);
                                        glEnd();
                                }
                }

        }
        glutSwapBuffers();
        pthread_mutex_unlock(&psi_mutex);
}


long frame_counter = 0;


void update_frame(void)
{
        /* Evaluate time development */
        pthread_mutex_lock(&psi_mutex);
        laplace_2d(psi2, rr, psi);
        for (size_t k = 0; k < rr; k++) {
                psi2[k] += V[k] * psi2[k];
        }
//        potential(psi2, rr);
        timestep(psi, psi2, DT);

        double norm = 0.0;
        for (int i = 0; i < rr; i++) {
                /* we sum over  (psi*_ij psi_ij dx dy) */
                norm += conj(psi[i])*psi[i]*1.0/rr;
        }
        for (int i = 0; i < rr; i++) {
                psi2[i] = psi[i]/norm;
        }
 //       visualize_potential(psi2, psi, rr);
        pthread_mutex_unlock(&psi_mutex);
        frame_counter++;
}


void update_display(int foo)
{
        glutTimerFunc(25, update_display, 0);
        glutPostRedisplay();
}


int main(int argc, char **argv)
{
        pthread_mutex_init(&psi_mutex, NULL);
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
        glutInitWindowSize(640, 640);
        glutInitWindowPosition(100, 100);
        glutCreateWindow("Psi");

        glClearColor(0.0, 0.0, 0.0, 0.0);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
//        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
        glutDisplayFunc(display_frame);
        glutIdleFunc(update_frame);

        V = malloc(rr* sizeof(lapack_complex_double));
        memset(V, 0, rr*sizeof(lapack_complex_double));
        potential(V, rr);
        psi2 = malloc(rr * sizeof(lapack_complex_double));
        psi = malloc(rr*sizeof(lapack_complex_double));
        if (!psi) {
                fprintf(stderr, "Out of memory\n");
                return -1;
        }
//        memset(psi,0,rr*sizeof(lapack_complex_double));
//        memset(psi2, 0, rr*sizeof(lapack_complex_double));

        gaussian_pulse(psi, 0.2, 0.5, 0.5, 0.02, 7777.0);

        glutTimerFunc(50, update_display, 0);
        glutMainLoop();

        free(V);
        free(psi2);
        free(psi);
        free(ipiv);


        return 0;

}
