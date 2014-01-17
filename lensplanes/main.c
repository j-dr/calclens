#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <assert.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_ieee_utils.h>

#include "raytrace.h"

int main(int argc, char **argv)
{
#ifdef DEF_GSL_IEEE_ENV
  gsl_ieee_env_setup();
#endif
  
  /* vars */
  char name[MAX_FILENAME];
  
  /* init MPI and get current tasks and number of tasks */
  int rc = MPI_Init(&argc,&argv);
  if(rc != MPI_SUCCESS)
    {
      fprintf(stderr,"Error starting MPI program. Terminating.\n");
      MPI_Abort(MPI_COMM_WORLD,rc);
    }
  MPI_Comm_size(MPI_COMM_WORLD,&NTasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&ThisTask);
  
  logProfileTag(PROFILETAG_TOTTIME);
  
#ifdef MEMWATCH
  mwDoFlush(1);
  //mwStatistics(MW_STAT_MODULE);
  mwStatistics(MW_STAT_LINE);
#endif
  
  /* init all vars for raytracing 
     1) read from config file
     2) set search radius
     3) set plane names
  */
  logProfileTag(PROFILETAG_INITEND_LOADBAL);
  if(ThisTask == 0)
    read_config(argv[1]);
  MPI_Bcast(&rayTraceData,(int) (sizeof(RayTraceData)),MPI_BYTE,0,MPI_COMM_WORLD); 
  logProfileTag(PROFILETAG_INITEND_LOADBAL);

  //even if the code is called with more than 1 MPI task, only task zero will make the lens planes
  if(ThisTask == 0)
    {
#ifdef POINTMASSTEST
      fprintf(stderr,"making lensing planes for a point mass or NFW test...\n");
      make_lensplanes_pointmass_test();
#else
      fprintf(stderr,"making lensing planes...\n");
      fprintf(stderr,"lens plane path/name: '%s/%s'\n",rayTraceData.LensPlanePath,rayTraceData.LensPlaneName);
      fprintf(stderr,"lens plane HEALPix order = %ld\n",rayTraceData.LensPlaneOrder);
      fprintf(stderr,"light cone file list: '%s'\n",rayTraceData.LightConeFileList);
      fprintf(stderr,"light cone file type: '%s'\n",rayTraceData.LightConeFileType);
      fprintf(stderr,"light cone origin: x,y,z = %f|%f|%f\n",rayTraceData.LightConeOriginX,rayTraceData.LightConeOriginY,rayTraceData.LightConeOriginZ);
      fprintf(stderr,"light cone unit conv. factors: mass = %le, length = %le, velocity  = %le\n",
	      rayTraceData.MassConvFact,rayTraceData.LengthConvFact,rayTraceData.VelocityConvFact);
      fprintf(stderr,"partMass (may not be used) = %le\n",rayTraceData.partMass);
      fprintf(stderr,"mem. buff. size = %lf MB, max. # of planes in mem = %ld, plane chunk alloc. factor = %lf\n",
	      rayTraceData.memBuffSizeInMB,rayTraceData.MaxNumLensPlaneInMem,rayTraceData.LightConePartChunkFactor);
      makeRayTracingPlanesHDF5();
#endif
    }
  
  ////////////////////////////
  MPI_Barrier(MPI_COMM_WORLD);
  ////////////////////////////
  
  //make maps of lens planes for debugging/viz
  long i;
  for(i=0;i<rayTraceData.NumLensPlanes;++i)
    make_lensplane_map(i);

  /* finish profiling info*/
  logProfileTag(PROFILETAG_TOTTIME);
  sprintf(name,"%s/timing",rayTraceData.OutputPath);
  printProfileInfo(name,ProfileTagNames);
  resetProfiler();
  
  MPI_Finalize();
  return 0;
}
