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
  
  /* do ray tracing */
  raytrace();
  
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
