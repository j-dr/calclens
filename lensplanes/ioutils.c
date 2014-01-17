#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>
#include <gsl/gsl_heapsort.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "raytrace.h"

/*gets the number of lines in a file*/
long fnumlines(FILE *fp)
{
  long i=-1;
  char c[5000];
  while(!feof(fp))
    {
      ++i;
      fgets(c,5000,fp);
    }
  rewind(fp);
  return i;
}

