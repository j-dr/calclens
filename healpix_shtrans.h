/*
 * does spherical harmonic transform on an healpix grid
 *  -Matthew R Becker, Univ. of Chicago 2010
 */

#ifdef MEMWATCH
#include "memwatch.h"
#endif

#ifdef USEMEMCHECK
#include <memcheck.h>
#endif

#ifdef DMALLOC
#include <dmalloc.h>
#endif

#ifndef HEALPIXSHT /* HEALPIXSHT */
#define HEALPIXSHT /* HEALPIXSHT */


#define STATIC_LOADBAL_SHT /* define to turn off adaptive load balance functions */
//#define OUTPUT_SHT_LOADBALANCE /* define to output a bunch of SHT load balance functions */

extern double *map2almRingTimesGlobal;
extern long Nmap2almRingTimesGlobal;
extern double *alm2mapMTimesGlobal;
extern long Nalm2mapMTimesGlobal;

typedef struct {
  long order;
  long *firstRingTasks;
  long *lastRingTasks;
  long Nmapvec;
  long *northStartIndMapvec;
  long *southStartIndMapvec;
  long *northStartIndGlobalMap;
  long *southStartIndGlobalMap;
  long lmax;
  long *firstMTasks;
  long *lastMTasks;
  long Nlm;
  double *ring_weights;
  double *window_function;
} HEALPixSHTPlan;

/* in healpix_shtrans.c */
long get_lmin_ylm(long m, double sintheta);
long lm2index(long l, long m, long lmax);
long num_lms(long lmax);
long order2lmax(long _order);
HEALPixSHTPlan healpixsht_plan(long order);
void healpixsht_destroy_plan(HEALPixSHTPlan plan);
void ring_synthesis(long Nphi, long shifted, float *ringvals);
void get_mrange_alm2map_healpix_mpi(int MyNTasks, long *firstRing, long *lastRing, long order);
void init_mrange_alm2map_healpix_mpi(long order);
void destroy_mrange_alm2map_healpix_mpi(void);
void ring_analysis(long Nphi, float *ringvals);
void init_ringrange_map2alm_healpix_mpi(long order);
void destroy_ringrange_map2alm_healpix_mpi(void);
void get_ringrange_map2alm_healpix_mpi(int MyNTasks, long *firstRing, long *lastRing, long order);
void healpixsht_destroy_internaldata(void);
void read_ring_weights(char *path, HEALPixSHTPlan *plan);
void read_window_function(char *path, HEALPixSHTPlan *plan);

/* in map2alm_transpose_mpi.c */
void map2alm_mpi(double *alm_real, double *alm_imag, float *mapvec, HEALPixSHTPlan plan);

/* in alm2allmaps_transpose_mpi.c */
void alm2allmaps_mpi(double *alm_real, double *alm_imag, float *mapvec, float *mapvec_gt, float *mapvec_gp,
			       float *mapvec_gtt, float *mapvec_gtp, float *mapvec_gpp,
			       HEALPixSHTPlan plan);

/* in alm2map_transpose_mpi.c */
void alm2map_mpi(double *alm_real, double *alm_imag, float *mapvec, HEALPixSHTPlan plan);

/* define a structure for info in and out of program to make it thread safe */
typedef struct
{
  double fsmall;
  double fbig;
  double eps;
  double cth_crit;
  long lmax;
  long mmax;
  long m_last;
  long m_crit;
  double *cf;
  double *recfac;
  double *mfac;
  double *t1fac;
  double *t2fac;
} plmgen_data;

/* in healpix_plmgen.c */
long plm2index(long l, long m);
void index2plm(long plmindex, long*l, long *m);
long num_plms(long lmax);
void plmgen(double cth, double sth, long m, double *vec, long *firstl, plmgen_data *plmdata);
plmgen_data *plmgen_init(long lmax, double eps);
void plmgen_destroy(plmgen_data *plmdata);
void plmgen_recalc_recfac(long m, plmgen_data *plmdata);

#endif /* HEALPIXSHT */
