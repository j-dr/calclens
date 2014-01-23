#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <string.h>

#include "raytrace.h"
#include "fftpoissonsolve.h"

void threedpot_poissondriver(long planeNum)
{
  if(ThisTask == 0)
    fprintf(stderr,"FFT Poisson Driver is a stub!\n");
}
