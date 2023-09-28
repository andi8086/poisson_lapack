#include "petscsys.h"
#include <mpi.h>
#include <stdio.h>
#include <slepceps.h>
#include <petsc.h>
#include <petscdm.h>
// #include <petscdmda.h>
// #include <petscdmlabel.h>
// #include <petscds.h>

#define X 16
#define N (X*X)


typedef struct {
        uint64_t foo;
} user_ctx_t;


user_ctx_t app_ctx;

void fill_matrix(Mat *A)
{
        PetscInt rstart, rend;
        PetscInt i, j, ii, jj, col, row;
        PetscScalar v;
        MatGetOwnershipRange(*A, &rstart, &rend);

        for (i = 0; i < N; i += X) {
                for (ii = 0; ii < X; ii++) {
                        /* linear memory offset */
                        row = i + ii;
                        if (jj >= rstart && jj < rend) {
                                /* local data element */
                                v = 4.0;
                                col = row;
                                MatSetValues(*A, 1, &row, 1, &col, &v,
                                             INSERT_VALUES);
                                v = -1.0;
                                if ((row % X) > 0) {
                                       col = row - 1;
                                        MatSetValues(*A, 1, &row, 1, &col, &v,
                                                     INSERT_VALUES);
                                        MatSetValues(*A, 1, &col, 1, &row, &v,
                                                     INSERT_VALUES);
                                }

                                /* Pentadiagonals */
                                if (row >= X) {
                                        col = row - X;
                                        MatSetValues(*A, 1, &row, 1, &col, &v,
                                                     INSERT_VALUES);
                                        MatSetValues(*A, 1, &col, 1, &row, &v,
                                                     INSERT_VALUES);
                                }

                        }
                }
        }



        MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);

//        MatView(*A, PETSC_VIEWER_DRAW_WORLD);
}


int main(int argc, char **argv)
{
        DM da;
        Mat A;

        PetscCall(SlepcInitialize(&argc, &argv, NULL, "Blah"));
//        PetscCall(PetscInitialize(&argc, &argv, NULL, "Blah"));

        /*PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                               DMDA_STENCIL_STAR, 4, 4,
                               PETSC_DECIDE, PETSC_DECIDE,
                               1, 1, NULL, NULL, &da));
        PetscCall(DMSetFromOptions(da));
        PetscCall(DMSetUp(da));
        PetscCall(DMSetApplicationContext(da, &app_ctx));
        */

        PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
        PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE,
                              N, N));
        PetscCall(MatSetType(A, MATAIJ));

        fill_matrix(&A);

        EPS eps;

        PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));

        PetscCall(EPSSetOperators(eps, A, NULL));
        PetscCall(EPSSetProblemType(eps, EPS_HEP));
        PetscCall(EPSSetDimensions(eps, 4, 16, N));
        PetscCall(EPSSetFromOptions(eps));

        PetscCall(EPSSolve(eps));

        PetscInt its, nev;
        PetscCall(EPSGetIterationNumber(eps, &its));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterations: %" PetscInt_FMT "\n", its));
        PetscCall(EPSGetDimensions(eps, &nev, NULL, NULL));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Dimensions: %" PetscInt_FMT "\n", nev));

        PetscInt nconv;
        PetscCall(EPSGetConverged(eps, &nconv));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged EVs: %"
                                PetscInt_FMT "\n", nconv));

        Vec vr;

        PetscCall(MatCreateVecs(A, &vr, NULL));

        for (PetscInt i = 0; i < nconv; i++) {
                PetscCall(EPSGetEigenvector(eps, i, vr, NULL));
                VecView(vr, PETSC_VIEWER_DRAW_WORLD);
        }

        PetscCall(VecDestroy(&vr));

        PetscCall(EPSDestroy(&eps));

        PetscCall(MatDestroy(&A));
//        PetscCall(DMDestroy(&da));
//        PetscCall(PetscFinalize());
        PetscCall(SlepcFinalize());

        return 0;
}
