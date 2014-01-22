#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <string.h>

#include "raytrace.h"

void fullsky_partdist_poissondriver(void)
{
  double time;
  long doPoissonSolve;
  long totNlensPlaneParts;
  
  //read parts
  if(!rayTraceData.UseHEALPixLensPlaneMaps)
    {
      time = -MPI_Wtime();
      logProfileTag(PROFILETAG_PARTIO);

      if(ThisTask == 0)
	fprintf(stderr,"reading parts from '%s/%s%04ld.h5'\n",
		rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,rayTraceData.CurrentPlaneNum);

      read_lcparts_at_planenum_fullsky_partdist(rayTraceData.CurrentPlaneNum);
      get_smoothing_lengths();

      logProfileTag(PROFILETAG_PARTIO);
      time += MPI_Wtime();

      if(ThisTask == 0)
	fprintf(stderr,"read %ld parts in %g seconds.\n",NlensPlaneParts,time);
    }

  /*
    if we read particles, then solve poisson eqn. Otherwise, even if 
    backdens is zero or non-zero, we do not need to solve the poisson eqn
  */
  MPI_Allreduce(&NlensPlaneParts,&totNlensPlaneParts,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  
  if(totNlensPlaneParts > 0)
    doPoissonSolve = 1;
  else
    doPoissonSolve = 0;
  
#ifdef DEBUG
#if DEBUG_LEVEL > 1
  if(NlensPlaneParts > 0)
    fprintf(stderr,"%d: pos,mass = %f|%f|%f|%e, total # of parts = %ld, # of parts on this task = %ld\n",
	    ThisTask,lensPlaneParts[0].pos[0],lensPlaneParts[0].pos[1],lensPlaneParts[0].pos[2],lensPlaneParts[0].mass,
	    totNlensPlaneParts,NlensPlaneParts);
  else
    fprintf(stderr,"%d: # of parts on this task = %ld\n",ThisTask,NlensPlaneParts);
#endif
#endif
  
  //run poisson solver
  if(doPoissonSolve)
    {
      do_healpix_sht_poisson_solve(rayTraceData.densfact,rayTraceData.backdens);
      
      //do MG step if needed
#ifndef SHTONLY
      logProfileTag(PROFILETAG_PARTIO);
      read_lcparts_at_planenum(rayTraceData.CurrentPlaneNum);
      get_smoothing_lengths();
      logProfileTag(PROFILETAG_PARTIO);
      
#ifdef DEBUG_IO_DD
      write_bundlecells2ascii("preMGPS");
#endif
      
      mgpoissonsolve(rayTraceData.densfact,rayTraceData.backdens);
#endif
    }
  
  //free parts
  destroy_parts();
}

void cutsky_partdist_poissondriver(void) 
{
  double time;
  long doPoissonSolve;
  
  //read parts
#ifdef SHTONLY
  if(!rayTraceData.UseHEALPixLensPlaneMaps)
    {
#endif
      time = -MPI_Wtime();
      logProfileTag(PROFILETAG_PARTIO);

      if(ThisTask == 0)
	fprintf(stderr,"reading parts from '%s/%s%04ld.h5'\n",
		rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,rayTraceData.CurrentPlaneNum);

      read_lcparts_at_planenum(rayTraceData.CurrentPlaneNum);
      get_smoothing_lengths();

      logProfileTag(PROFILETAG_PARTIO);
      time += MPI_Wtime();

      if(ThisTask == 0)
	fprintf(stderr,"read %ld parts in %g seconds.\n",NlensPlaneParts,time);
#ifdef SHTONLY
    }
#endif
  
  /* if there are particles read into memory, then we need to solve the poisson equation
     otherwise, there are two cases
     1) if backdens == 0, then rho == 0 everywhere and phi = 0 as well
     2) if backdens != 0, then rho is negative in the region being ray traced and zero elsewhere, so still need to solve the poisson equation
  */
#ifdef NOBACKDENS
  long totNlensPlaneParts;
  MPI_Allreduce(&NlensPlaneParts,&totNlensPlaneParts,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  
  if(totNlensPlaneParts > 0)
    doPoissonSolve = 1;
  else
    doPoissonSolve = 0;
#else
  doPoissonSolve = 1;
#endif
  
#ifdef DEBUG
#if DEBUG_LEVEL > 1
  if(NlensPlaneParts > 0)
    fprintf(stderr,"%d: pos,mass = %f|%f|%f|%e, total # of parts = %ld, # of parts on this task = %ld\n",
	    ThisTask,lensPlaneParts[0].pos[0],lensPlaneParts[0].pos[1],lensPlaneParts[0].pos[2],lensPlaneParts[0].mass,
	    totNlensPlaneParts,NlensPlaneParts);
  else
    fprintf(stderr,"%d: # of parts on this task = %ld\n",ThisTask,NlensPlaneParts);
#endif
#endif

  //run poisson solver
  if(doPoissonSolve)
    {
      do_healpix_sht_poisson_solve(rayTraceData.densfact,rayTraceData.backdens);
      
      //do MG step if needed
#ifndef SHTONLY

#ifdef DEBUG_IO_DD
      write_bundlecells2ascii("preMGPS");
#endif
      
      mgpoissonsolve(rayTraceData.densfact,rayTraceData.backdens);
#endif
    }

  //free parts since we do not need them anymore                                                                                                                                                                                                                                                                                                                 
  destroy_parts();
}
