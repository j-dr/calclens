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
#include "inthash.h"

static GridCell *GridCells = NULL;
static long NumGridCellsAlloc = 0;
static long NumGridCells = 0;

/* Notes for how to do this

1) compute for each bundle cell the range of grid cells needed
   for now do this step using an array of cells and inthash.c

2) sort cells by index

3) send/recv cells needed from other processors

4) do integral over the cells

*/

void threedpot_poissondriver(long planeNum)
{
  if(ThisTask == 0)
    fprintf(stderr,"FFT Poisson Driver is a stub!\n");

  //init hash table
  struct inthash *ih = new_inthash();


}

