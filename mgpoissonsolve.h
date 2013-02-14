/*
  routines to do full FAS multigrid on the sphere w/ BCs
  
  Matthew R. Becker, UofC 2012
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

#ifndef _MGPOISSONSOLVE_
#define _MGPOISSONSOLVE_

//use to only use float for poisson solver
//#define MGPOISSONSOLVE_FLOAT - does not work! DO NOT USE!
#ifdef MGPOISSONSOLVE_FLOAT
typedef float mgfloat;
#else
typedef double mgfloat;
#endif

typedef struct {
  mgfloat *grid;
  double *sinfacs;
  double *cosfacs;
  double *sintheta;
  double *costheta;
  double *sinphi;
  double *cosphi;
  double *diag;
  long N;
  double L;
  double dL;
  double thetaLoc;
  double phiLoc;
  double RmatSphereToPatch[3][3];
  double RmatPatchToSphere[3][3];
} *MGGrid,_MGGrid;

typedef struct {
  MGGrid u;
  MGGrid rho;
} MGGridSet;

// mgpoissonsolve_utils.c
double solve_fas_mggrid(MGGridSet *grids, long Nlev, long NumPreSmooth, long NumPostSmooth, long NumOuterCycles, long NumInnerCycles, double convFact);
void cycle_fas_mggrid(MGGridSet *grids, long lev, long NumPreSmooth, long NumPostSmooth, long NumInnerCycles);
void smooth_mggrid(MGGrid u, MGGrid rhs, long Nsmooth);
MGGrid lop_mggrid(MGGrid g);
void lop_mggrid_plusequal(MGGrid g, MGGrid pg);
MGGrid resid_mggrid(MGGrid u, MGGrid rhs);
double L1norm_mggrid(MGGrid u, MGGrid rhs);
double L2norm_mggrid(MGGrid u, MGGrid rhs);
double truncErr_mggrid(MGGrid uf, MGGrid uc, MGGrid lopc);
double fracErr_mggrid(MGGrid u, MGGrid rhs);
void interp_mggrid(MGGrid uf, MGGrid uc);
void interp_mggrid_plusequal(MGGrid uf, MGGrid uc);
void restrict_mggrid(MGGrid uc, MGGrid uf);
void restrict_mggrid_minusequal(MGGrid uc, MGGrid uf);
void resid_restrict_mggrid(MGGrid uc, MGGrid uf, MGGrid rhof);
void zero_mggrid(MGGrid u);
MGGrid alloc_mggrid(long N, double L);
MGGrid copy_mggrid(MGGrid u);
void free_mggrid(MGGrid u);
void write_mggrid(char fname[], MGGrid u);
void print_runtimes_mgsteps(void);
void reset_runtimes_mgsteps(void);

#endif /* _MGPOISSONSOLVE_ */

