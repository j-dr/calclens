#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>

#include "profile.h"
#include "healpix_utils.h"
#include "healpix_shtrans.h"

#ifdef MEMWATCH
#include "memwatch.h"
#endif

#ifdef USEMEMCHECK
#include <memcheck.h>
#endif

#ifdef DMALLOC
#include <dmalloc.h>
#endif

#ifndef _RAYTRACE_
#define _RAYTRACE_

#define RAYTRACEVERSION "CALCLENS v0.9c"

/* Some debugging macros
   undef DEBUG for no debugging
   DEBUG_LEVEL = 0 is for basic debugging
   DEBUG_LEVEL = 1 is for messages printed by a single task but not as critical
   DEBUG_LEVEL = 2 or above is used for messages that every task will print 
   
   define DEBUG_IO some output as follows
   1) activecells%d.dat - list of ring inds and nside vals for all active cells for each task (task # put into %d by printf)
   2) buffcells%d.dat - list of ring inds and nside vals for all buffer cells for each task (task # put into %d by printf)
   3) lists of inds from particle read: indget.dat are ring inds to read and indgetfile.dat are ring inds to read from file
*/

#ifdef NDEBUG
#undef DEBUG
#define DEBUG_LEVEL -1
#undef DEBUG_IO
#endif

#ifdef DEBUG
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif
#endif

#define MAX_FILENAME 1024

//constants
#define RHO_CRIT 2.77519737e11  /* Critial mass density in h^2 M_sun/Mpc^3 with H_{0} = h 100 km/s/Mpc*/
#define CSOL 299792.458         /* velocity of light in km/s */

typedef struct {
  //params in config file
  double OmegaM;
  double maxComvDistance;
  long NumLensPlanes;
  char LensPlanePath[MAX_FILENAME];
  char LensPlaneName[MAX_FILENAME];
  char OutputPath[MAX_FILENAME];
  long NumFilesIOInParallel;
  
  /* for making lens planes */
  char LightConeFileList[MAX_FILENAME]; 
  char LightConeFileType[MAX_FILENAME]; 
  double LightConeOriginX;  /* used to make lens planes from the light cone */
  double LightConeOriginY;
  double LightConeOriginZ;
  long LensPlaneOrder; 
  long NumDivLensPlane; /* not currently used */ 
  double memBuffSizeInMB; 
  long MaxNumLensPlaneInMem;
  double LightConePartChunkFactor;
  double partMass;            /* particle mass that can be used in writing lens planes - have to use it in read_lightcone.c file for it to take effect*/
  double MassConvFact;        /* conversion factors that can be used if needed in read_lightcone.c file */
  double LengthConvFact;      /* conversion factors that can be used if needed in read_lightcone.c file */
  double VelocityConvFact;    /* conversion factors that can be used if needed in read_lightcone.c file */
  
  /* for the point mass or NFW test */
  double raPointMass;
  double decPointMass;
  double radPointMass;
  double galRadPointNFWTest;
} RayTraceData;

//40 bytes 
#define NFIELDS_LCPARTICLE ((hsize_t) 8)
typedef struct {
  long partid;
  float px;
  float py;
  float pz;
  float vx;
  float vy;
  float vz;
  float mass;
} LCParticle;

typedef struct {
  long NumRayTracingPlanes;
  long HEALPixOrder;
  long NPix;
  long MaxTotNumLCParts;
  long ChunkSizeLCParts;
  long *NumLCParts;
  long *NumLCPartsUsed;
  LCParticle **LCParts;
  long *TotNumLCPartsInPlane;
  long *NumLCPartsInPix;
  long *NumLCPartsInPixUpdate;
  long NumLCPartWriteBuff;
  LCParticle *LCPartWriteBuff;
  long *PeanoInds;
  size_t *PeanoSortInds;
} WriteBuffData;

/* extern defs of global vars in globalvars.c */
extern const char *ProfileTagNames[];
extern RayTraceData rayTraceData;                        /* global struct with all vars from config file */
extern int ThisTask;                                     /* this task's rank in MPI_COMM_WORLD */
extern int NTasks;                                       /* number of tasks in MPI_COMM_WORLD */

/* in make_lensplanes_hdf5.c */
void makeRayTracingPlanesHDF5(void);

/* in make_lensplanes_pointmass_test.c */
void make_lensplanes_pointmass_test(void);

/* in lightconeio.c */
long getNumLCPartsFile(FILE *infp);
LCParticle getLCPartFromFile(long i, long Np, FILE *infp, int freeBuff);
long getNumLCPartsFile_ARTLC(FILE *infp);
LCParticle getLCPartFromFile_ARTLC(long i, long Np, FILE *infp, int freeBuff);
long getNumLCPartsFile_GADGET2(FILE *infp);
LCParticle getLCPartFromFile_GADGET2(long i, long Np, FILE *infp, int freeBuff);
long getNumLCPartsFile_LGADGET(FILE *infp);
LCParticle getLCPartFromFile_LGADGET(long i, long Np, FILE *infp, int freeBuff);

/* in cosmocalc.h - distances - assumes flat lambda */
void init_cosmocalc(void);
double comvdist_integ_funct(double a, void *p);
double angdist(double a);
double comvdist(double z);
double angdistdiff(double amin, double amax);
double acomvdist(double dist);

/* in config.c */
void read_config(char *filename);

/* in ioutils.c */
long fnumlines(FILE *fp);

/* in makemaps.c */
void mark_bundlecells(double mapbuffrad, int searchTag, int markTag);

#endif /* _RAYTRACE_ */
