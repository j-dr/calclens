#include <mpi.h>
#include <fftw3-mpi.h>

#ifdef MEMWATCH
#include "memwatch.h"
#endif

#ifdef USEMEMCHECK
#include <memcheck.h>
#endif

#ifdef DMALLOC
#include <dmalloc.h>
#endif

#ifndef _LGADGETIO_
#define _LGADGETIO_

long get_numparts_LGADGET(char fname[]);
int get_numfiles_LGADGET(char fname[]);
float get_omegam_LGADGET(char fname[]);
float get_scale_factor_LGADGET(char fname[]);
float get_period_length_LGADGET(char fname[]);
void read_LGADGET(char fname[], float **px, float **py, float **pz, long **id, int *Np);

#endif /* _LGADGETIO_ */
