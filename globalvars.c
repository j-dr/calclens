#include "raytrace.h"

/* global vars are defined here except those in 
   healpix_shtrans.c 
*/

const char *ProfileTagNames[] = {"TotalTime","StepTime","SHT","SHTSolve","MapShuffle",
				 "MG","MGSolve","RayIO","PartIO","RayProp",
				 "GridSearch","GalIO","RayBuff","Restart","InitEndLoadBal",
				 "GalMove","GalGridSearch","ImageGalIO","GridKappa","TreeBuild","TreeWalk"};

RayTraceData rayTraceData;                               /* global struct with all vars from config file */
long NbundleCells = 0;                                   /* the number of bundle cells used for overall domain decomp */
HEALPixBundleCell *bundleCells = NULL;                   /* the vector of bundle cells used for domain decomp */
long *bundleCellsNest2RestrictedPeanoInd = NULL;         /* a global "hash" of bundle nest indexes that defines the domain covered by rays in terms 
							    of a coniguous space-filling index set */
long *bundleCellsRestrictedPeanoInd2Nest = NULL;         /* inverse of bundleCellsNest2RestrictedPeanoInd */
long NrestrictedPeanoInd = 0;                            /* number of restricted peano inds == total number of bundle cells with rays */
long *firstRestrictedPeanoIndTasks = NULL;               /* array which holds first index of section of RestrictedPeanoInds assigned to each MPI task 
							    - this forms the domain decomp */
long *lastRestrictedPeanoIndTasks = NULL;                /* array which holds last index of section of RestrictedPeanoInds assigned to each MPI task 
							    - this forms the domain decomp */
HEALPixRay *AllRaysGlobal = NULL;                        /* pointer to memory location of all rays */
long MaxNumAllRaysGlobal;                                /* maximum number of rays that can be stored */
long NumAllRaysGlobal;                                   /* current number of rays stored */
int ThisTask;                                            /* this task's rank in MPI_COMM_WORLD */
int NTasks;                                              /* number of tasks in MPI_COMM_WORLD */
long NlensPlaneParts = 0;                                /* number of particles in lens plane for this task */
Part *lensPlaneParts = NULL;                             /* vector of particles for this task */
SourceGal *SourceGalsGlobal = NULL;                      /* source gals for task */
long NumSourceGalsGlobal = 0;                            /* # of gals in global vecs */
ImageGal *ImageGalsGlobal = NULL;                        /* images of source gals for task */
long NumImageGalsGlobal = 0;                             /* # of gals in global vecs */

long NmapCells = 0;
HEALPixMapCell *mapCells = NULL;
HEALPixMapCell *mapCellsGradTheta = NULL;
HEALPixMapCell *mapCellsGradPhi = NULL;
HEALPixMapCell *mapCellsGradThetaTheta = NULL;
HEALPixMapCell *mapCellsGradThetaPhi = NULL;
HEALPixMapCell *mapCellsGradPhiPhi = NULL;

const float HPIX_WINDOWFUNC_POW[HEALPIX_UTILS_MAXORDER+1] = {3.0,3.0,3.0,3.0,3.0,  // 0 - 4
							     3.0,3.0,3.0,3.0,3.0,  // 5 - 9
							     3.9,3.9,3.5,3.5,3.5,  //10 - 14
							     3.5,3.5,3.5,3.5,3.5,  //15 - 19
							     3.5,3.5,3.5,3.5,3.5,  //20 - 24
							     3.5,3.5,3.5,3.5,3.5}; //25 - 29
