#include <stdio.h>
#include <gmshc.h>


int main(int argc, char **argv)
{
        int ierr;

        gmshInitialize(argc, argv, 1, 0, &ierr);
        gmshModelAdd("t1", &ierr);

        const double lc = 1E-2;

        gmshModelGeoAddPoint(0.0, 0.0, 0.0, lc, 1, &ierr);
        gmshModelGeoAddPoint(0.1, 0.0, 0.0, lc, 2, &ierr);
        gmshModelGeoAddPoint(0.1, 0.3, 0.0, lc, 3, &ierr);
        gmshModelGeoAddPoint(0.0, 0.3, 0.0, lc, 4, &ierr);

        gmshModelGeoAddLine(1, 2, 1, &ierr);
        gmshModelGeoAddLine(3, 2, 2, &ierr);
        gmshModelGeoAddLine(3, 4, 3, &ierr);
        gmshModelGeoAddLine(4, 1, 4, &ierr);

        const int cl1[] = {4, 1, -2, 3};
        gmshModelGeoAddCurveLoop(cl1, sizeof(cl1)/sizeof(cl1[0]), 1, 0, &ierr);

        const int s1[] = {1};
        gmshModelGeoAddPlaneSurface(s1, sizeof(s1)/sizeof(s1[0]), 1, &ierr);
        
        gmshModelGeoSynchronize(&ierr);

        /* 2 means 2D */
        gmshModelMeshGenerate(2, &ierr);
        gmshFltkRun(&ierr);

        gmshFinalize(&ierr);
        return 0;
}
