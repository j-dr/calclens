#include <stdio.h>

#ifndef _RAYTRACE_
#define _RAYTRACE_

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
  long NFFT;
  long MaxNFFT;
  char ThreeDPotSnapList[MAX_FILENAME];
  double LengthConvFact;
  
  //internal params for code
  double planeRad;
} RayTraceData;

/* extern defs of global vars in globalvars.c */
extern const char *ProfileTagNames[];
extern RayTraceData rayTraceData;                        /* global struct with all vars from config file */
extern int ThisTask,NTakss;

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
FILE *fopen_retry(const char *filename, const char *mode);

#endif /* _RAYTRACE_ */
