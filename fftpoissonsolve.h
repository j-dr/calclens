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

#ifndef _FFTPSOLVE_
#define _FFTPSOLVE_

#define G 4.302113490418529e-9 //in Mpc (km/s)^2 / Msun
#define FOUR_PI_G 5.4061952545633574e-8 //4piG in  Mpc (km/s)^2 / Msun  

//global defs
extern ptrdiff_t NFFT;
extern ptrdiff_t AllocLocal,N0Local, N0LocalStart;
extern int *TaskN0Local;
extern int *TaskN0LocalStart;
extern int MaxN0Local;

#ifdef DOUBLEFFTW
extern fftw_plan fplan,bplan;
extern double *fftwrin;
extern fftw_complex *fftwcout;
#define FFT_TYPE double
#else
extern fftwf_plan fplan,bplan;
extern float *fftwrin;
extern fftwf_complex *fftwcout;
#define FFT_TYPE float
#endif

/* in fftpoissonsolve.c */
void comp_pot_snap(char *fbase);
void init_ffts(void);
void alloc_and_plan_ffts(void);
void cleanup_ffts(void);

#endif /* _FFTPSOLVE_ */
