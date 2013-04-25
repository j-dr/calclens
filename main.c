#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>
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
    {
      read_config(argv[1]);
      
      if(argc >= 3)
	rayTraceData.Restart = atol(argv[2]);
      else
	rayTraceData.Restart = 0;
    }
  MPI_Bcast(&rayTraceData,(int) (sizeof(RayTraceData)),MPI_BYTE,0,MPI_COMM_WORLD); 
  logProfileTag(PROFILETAG_INITEND_LOADBAL);
  
  /* error check  */
  if(ThisTask == 0)
    {
      fprintf(stderr,"----------------------------------------------------------------------------------------------------------------------\n");
      fprintf(stderr,"running version '%s'.\n",RAYTRACEVERSION);
      fprintf(stderr,"code called with %d tasks.\n",NTasks);
      fprintf(stderr,"restart flag = %ld\n",rayTraceData.Restart);
      fprintf(stderr,"lensing plane path: %s\n",rayTraceData.LensPlanePath);
      fprintf(stderr,"Nplanes = %ld, omegam = %f, max_comvd = %f \n",rayTraceData.NumLensPlanes,rayTraceData.OmegaM,rayTraceData.maxComvDistance);
      fprintf(stderr,"galaxy image search radius = %f arcmin\n",rayTraceData.galImageSearchRad/M_PI*180.0*60.0);
      fprintf(stderr,"galaxy image ray buffer radius = %f arcmin\n",RAYBUFF_RADIUS_ARCMIN);
      fprintf(stderr,"SHT order = %ld, bundle order = %ld, ray order = %ld\n",rayTraceData.SHTOrder,rayTraceData.bundleOrder,rayTraceData.rayOrder);
      fprintf(stderr,"min,max comv. smoothing scale = %lg|%lg\n",rayTraceData.minComvSmoothingScale,rayTraceData.maxComvSmoothingScale);
      fprintf(stderr,"sizeof(long) = %lu, sizeof(long long) = %lu\n",sizeof(long),sizeof(long long));
      fprintf(stderr,"----------------------------------------------------------------------------------------------------------------------\n");
      fprintf(stderr,"\n");
    }
  
  /* code will either make the lens planes or do raytracing depending on configuration */
#ifdef MAKE_LENSPLANES
#ifdef POINTMASSTEST
  if(ThisTask == 0)
    {
      fprintf(stderr,"making lensing planes for a point mass or NFW test...\n");
      make_lensplanes_pointmass_test();
    }    
  
  ////////////////////////////
  MPI_Barrier(MPI_COMM_WORLD);
  ////////////////////////////
#else
  //even if the code is called with more than 1 MPI task, only task zero will make the lens planes
  if(ThisTask == 0)
    {
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
      
    }
  
  ////////////////////////////
  MPI_Barrier(MPI_COMM_WORLD);
  ////////////////////////////
  
  long i;
  for(i=0;i<rayTraceData.NumLensPlanes;++i)
    make_lensplane_map(i);
#endif /* POINTMASSTEST */
#else
  
#ifdef POINTMASSTEST
  if(ThisTask == 0)
    make_lensplanes_pointmass_test();
    
  ////////////////////////////
  MPI_Barrier(MPI_COMM_WORLD);
  ////////////////////////////
#endif
	
  /* do ray tracing */
  raytrace();
#endif /* MAKE_LENSPLANES */
  
  /* finish profiling info*/
  logProfileTag(PROFILETAG_TOTTIME);
  sprintf(name,"%s/timing",rayTraceData.OutputPath);
  printProfileInfo(name,ProfileTagNames);
  resetProfiler();
  
  //free HEALPix internal data
  healpixsht_destroy_internaldata();
  
  //free fftw data
  fftw_cleanup();
  
  MPI_Finalize();
  return 0;
}
