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

static void set_plane_params(void);

void raytrace(void)
{
  long i,j;
  long doPoissonSolve;
  double time,minTime,maxTime;//,totTime,avgTime;
  double stepTime,restTime,startTime;
  int writeRestartFile;
  char pname[MAX_FILENAME];
  FILE *fpStepTime;
#ifdef NOBACKDENS  
  long totNlensPlaneParts;
#endif
  
  ////////////////////////////////////
  // init simulation params and I/O //
  ////////////////////////////////////
  if(ThisTask == 0)
    {
      logProfileTag(PROFILETAG_INITEND_LOADBAL);
      sprintf(pname,"%s/timing.%d",rayTraceData.OutputPath,ThisTask);
      
      if(rayTraceData.Restart > 0)
	fpStepTime = fopen(pname,"a");
      else
	fpStepTime = fopen(pname,"w");
      
      assert(fpStepTime != NULL);
      
      if(!(rayTraceData.Restart > 0))
	printStepTimesProfileTags(fpStepTime,(long) -1,ProfileTagNames);
      logProfileTag(PROFILETAG_INITEND_LOADBAL);
    }
  
  if(rayTraceData.Restart > 0)
    {
      if(ThisTask == 0)
	fprintf(stderr,"\nreading restart files in directory '%s'\n",rayTraceData.OutputPath);
      
      logProfileTag(PROFILETAG_RESTART);
      read_restart();
      logProfileTag(PROFILETAG_RESTART);
    }
  else
    {
      logProfileTag(PROFILETAG_INITEND_LOADBAL);
      if(ThisTask == 0)
	fprintf(stderr,"initializing domain decomposition.\n");
      init_bundlecells();
      
      if(ThisTask == 0)
	fprintf(stderr,"initializing rays.\n\n");
      alloc_rays();
      init_rays();
      logProfileTag(PROFILETAG_INITEND_LOADBAL);
    }
  
  //read gals
  if(strlen(rayTraceData.GalsFileList) > 0)
    {
      logProfileTag(PROFILETAG_GALIO);
      read_fits2gals();
      
      //remove extra gals not needed for a restart
      if(rayTraceData.Restart > 0)
	clean_gals_restart(); 
      logProfileTag(PROFILETAG_GALIO);
    }
  
  //timers for restart 
  restTime = MPI_Wtime();
  startTime = MPI_Wtime();
  stepTime = 0.0;
  
  /////////////////////////////////////
  // main driver loop for simulation //
  ///////////////////////////////////// 
  for(rayTraceData.CurrentPlaneNum=rayTraceData.Restart;rayTraceData.CurrentPlaneNum<rayTraceData.NumLensPlanes;++rayTraceData.CurrentPlaneNum)
    {
      //////////////////////////////////////////////
      // check for time limit or periodic restart //
      //////////////////////////////////////////////
      time = MPI_Wtime()-startTime;
      MPI_Reduce(&time,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); 
      MPI_Reduce(&stepTime,&minTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); 
      if(ThisTask == 0)
	{
	  time = MPI_Wtime()-restTime;
	  
	  if((rayTraceData.WallTimeLimit-maxTime) <= 5.0*minTime)
	    {
	      writeRestartFile = 2;
	      fprintf(stderr,"\nout of time! limit,curr,step = %lg|%lg|%lg, writing restart files at plane %ld.\n",
		    rayTraceData.WallTimeLimit,maxTime,minTime,rayTraceData.CurrentPlaneNum);
	    }
	  else if(time >= rayTraceData.WallTimeBetweenRestart)
	    {
	      restTime = MPI_Wtime();
	      writeRestartFile = 1;
	      fprintf(stderr,"\nwriting restart files at plane %ld.\n",rayTraceData.CurrentPlaneNum);
	    }
	  else
	    writeRestartFile = 0;
	}
      MPI_Bcast(&writeRestartFile,1,MPI_INT,0,MPI_COMM_WORLD);
      
      if(writeRestartFile)
	{
	  logProfileTag(PROFILETAG_RESTART);
	  write_restart();
	  logProfileTag(PROFILETAG_RESTART);
	  
	  sprintf(pname,"%s/timing",rayTraceData.OutputPath);
	  printProfileInfo(pname,ProfileTagNames);
	}
      
      if(writeRestartFile > 1)
	break;
      
      ///////////////////////////////////////
      //  START OF ACTUAL RAY TRACING STEP //
      ///////////////////////////////////////
      stepTime = -MPI_Wtime();
      logProfileTag(PROFILETAG_STEPTIME);
      
      if(ThisTask == 0)
	fprintf(stderr,"planeNum = %04ld (% 4ld of % 4ld) '%s/%s%04ld.h5'\n",rayTraceData.CurrentPlaneNum,rayTraceData.CurrentPlaneNum+1,rayTraceData.NumLensPlanes,
		rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,rayTraceData.CurrentPlaneNum);
      
      //set units and poisson solve type
      set_plane_params();
      
      //load balance the nodes
      logProfileTag(PROFILETAG_INITEND_LOADBAL);
      load_balance_tasks();
      logProfileTag(PROFILETAG_INITEND_LOADBAL);
      
      //do gals grid search
      if(strlen(rayTraceData.GalsFileList) > 0)
	{
	  logProfileTag(PROFILETAG_GRIDSEARCH);
	  gridsearch(rayTraceData.planeRad,rayTraceData.planeRadMinus1);
	  logProfileTag(PROFILETAG_GRIDSEARCH);
	}
      
      //read in new parts
      time = -MPI_Wtime();
      logProfileTag(PROFILETAG_PARTIO);
#ifdef USE_FULLSKY_PARTDIST
      read_lcparts_at_planenum_fullsky_partdist(rayTraceData.CurrentPlaneNum);
#else
      read_lcparts_at_planenum(rayTraceData.CurrentPlaneNum);
#endif
      get_smoothing_lengths();
      logProfileTag(PROFILETAG_PARTIO);
      time += MPI_Wtime();
      
      if(ThisTask == 0)
	fprintf(stderr,"read %ld parts in %g seconds.\n",NlensPlaneParts,time);

#ifdef NOBACKDENS      
      /* if there are particles read into memory, then we need to solve the poisson equation 
	 otherwise, there are two cases
	 1) if backdens == 0, then rho == 0 everywhere and phi = 0 as well
	 2) if backdens != 0, then rho is negative in the region being ray traced and zero elsewhere, so still need to solve the poisson equation
	 If we are ray tracing the entire sphere, then case 2 reduces to case 1 even if backdens != 0.  We do not account for that here.
      */
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
      
      //zero everything before force computation
      for(i=0;i<NbundleCells;++i)
	{
	  if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	    {
	      for(j=0;j<bundleCells[i].Nrays;++j)
		{
		  bundleCells[i].rays[j].alpha[0] = 0.0;
		  bundleCells[i].rays[j].alpha[1] = 0.0;
		  
		  bundleCells[i].rays[j].U[0] = 0.0;
		  bundleCells[i].rays[j].U[1] = 0.0;
		  bundleCells[i].rays[j].U[2] = 0.0;
		  bundleCells[i].rays[j].U[3] = 0.0;
		}
	    }
	}
      
      if(doPoissonSolve)
	{
	  /* This goes like this
	     1) prep the poisson solve - done in set_poisson_solve_params above
	        a) set splitting scale and poisson order - happens based on max smoothing length of parts 
		b) set the PARTBUFF_BUNDLECELLs based on the splitting scale

	     2) do the SHT long-range poisson solve
	        a) set the MAPBUFF_BUNDLECELLs based on the poisson order   
		b) grid parts
		c) solve poisson equation
		d) get defl angles and shear
		e) interp to rays

	     3) do the mg patch solver
	  */
	  do_healpix_sht_poisson_solve(rayTraceData.densfact,rayTraceData.backdens);
	  
#ifndef SHTONLY
#ifdef USE_FULLSKY_PARTDIST
	  logProfileTag(PROFILETAG_PARTIO);
	  read_lcparts_at_planenum(rayTraceData.CurrentPlaneNum);
	  get_smoothing_lengths();
	  logProfileTag(PROFILETAG_PARTIO);
#endif
#ifdef DEBUG_IO_DD
	  write_bundlecells2ascii("preMGPS");
#endif
#ifdef TREEPM
	  if(!rayTraceData.TreePMOnlyDoSHT)
	    do_tree_poisson_solve(rayTraceData.densfact);
	  gridkappadens(rayTraceData.densfact,rayTraceData.backdens);
#else
	  mgpoissonsolve(rayTraceData.densfact,rayTraceData.backdens);
#endif
#endif
	}
      
      //free parts since we do not need them anymore
      destroy_parts();
      
      //write rays
      if((rayTraceData.Restart == 0 || rayTraceData.CurrentPlaneNum >= rayTraceData.Restart) && strlen(rayTraceData.RayOutputName) > 0)
	{
	  logProfileTag(PROFILETAG_RAYIO);
	  write_rays();
	  logProfileTag(PROFILETAG_RAYIO);
	}
      
      //ray propagation is done for each active (bit 0 set) bundleCell
      for(i=0;i<NbundleCells;++i)
	{
	  if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	    {
	      //do ray propagation
	      logProfileTag(PROFILETAG_RAYPROP);
	      rayprop_sphere(rayTraceData.planeRadPlus1,rayTraceData.planeRad,rayTraceData.planeRadMinus1,i);
	      logProfileTag(PROFILETAG_RAYPROP);
	    }
	}
      
      logProfileTag(PROFILETAG_STEPTIME);
      stepTime += MPI_Wtime();
      
      if(ThisTask == 0)
	{
	  printStepTimesProfileTags(fpStepTime,rayTraceData.CurrentPlaneNum,NULL);
	  fprintf(stderr,"\n");
	}
      
    } // end of main driver loop for simulation
  
  ///////////////////////////
  // clean up and finalize //
  ///////////////////////////
  if(rayTraceData.CurrentPlaneNum == rayTraceData.NumLensPlanes)
    {
      if(ThisTask == 0)
	fprintf(stderr,"finished ray tracing for all lens planes.\n");
      
      //do last plane I/O
      if(strlen(rayTraceData.RayOutputName) > 0)
	{
	  logProfileTag(PROFILETAG_RAYIO);
	  write_rays();
	  logProfileTag(PROFILETAG_RAYIO);
	}
      
      //write a final set of restart files
      logProfileTag(PROFILETAG_RESTART);
      write_restart();
      logProfileTag(PROFILETAG_RESTART);

    }
  
  //clean up
  logProfileTag(PROFILETAG_INITEND_LOADBAL);
  if(ThisTask == 0)
    fclose(fpStepTime);
  destroy_rays();
  if(strlen(rayTraceData.GalsFileList) > 0)
    destroy_gals();
  destroy_bundlecells();
  logProfileTag(PROFILETAG_INITEND_LOADBAL);
}

static void set_plane_params(void)
{
  double bundleLength = sqrt(4.0*M_PI/order2npix(rayTraceData.bundleOrder));
  double binL = (rayTraceData.maxComvDistance)/((double) (rayTraceData.NumLensPlanes));
  
  //get comv distances
  if(rayTraceData.CurrentPlaneNum - 1 < 0)
    rayTraceData.planeRadMinus1 = 0.0;
  else
    rayTraceData.planeRadMinus1 = (rayTraceData.CurrentPlaneNum - 1.0)*binL + binL/2.0;
  
  rayTraceData.planeRad = rayTraceData.CurrentPlaneNum*binL + binL/2.0;
  
  if(rayTraceData.CurrentPlaneNum+1 == rayTraceData.NumLensPlanes)
    rayTraceData.planeRadPlus1 = rayTraceData.maxComvDistance;
  else
    rayTraceData.planeRadPlus1 = (rayTraceData.CurrentPlaneNum + 1.0)*binL + binL/2.0;
    
  /* set some units
     1) set densfact - need to divide by cell angular area in radians to get proper units - this is done where needed
     2) set backdens - in correct units already, but only use if NOBACKDENS is NOT set
     3) factors of binL are from integral over lens plane to define projected mass density
  */
  //NOTE: 2nd order vol estmate is exact for a point mass but screws up NFW test 
#if defined(NFWHALOTEST)
  double radialvolume = (pow(rayTraceData.planeRad + binL/2.0,3.0) - pow(rayTraceData.planeRad - binL/2.0,3.0))/3.0;
#elif defined(POINTMASSTEST)
  //2nd order estimate
  double radialvolume = rayTraceData.planeRad*rayTraceData.planeRad*binL;
#else
   //exact
  double radialvolume = (pow(rayTraceData.planeRad + binL/2.0,3.0) - pow(rayTraceData.planeRad - binL/2.0,3.0))/3.0;
#endif
  double zw = 1.0/acomvdist(rayTraceData.planeRad) - 1.0;
  rayTraceData.densfact = 3.0*100.0*100.0/CSOL/CSOL*rayTraceData.OmegaM*rayTraceData.planeRad*(1.0+zw)*binL/(radialvolume*RHO_CRIT*rayTraceData.OmegaM);
#ifdef NOBACKDENS
  rayTraceData.backdens = 0.0;
#else
  rayTraceData.backdens = 3.0*100.0*100.0/CSOL/CSOL*rayTraceData.OmegaM*rayTraceData.planeRad*(1.0+zw)*binL;
#endif
  
  //set absolute min and max smoothing lengths
  rayTraceData.maxSL = rayTraceData.maxComvSmoothingScale/rayTraceData.planeRad;
#ifndef POINTMASSTEST
    if(rayTraceData.maxSL < MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder)))
      rayTraceData.maxSL = MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder));
    if(rayTraceData.maxSL > M_PI)
      rayTraceData.maxSL = M_PI;
#endif
    
    rayTraceData.minSL = rayTraceData.minComvSmoothingScale/rayTraceData.planeRad;
#ifndef POINTMASSTEST
    if(rayTraceData.minSL < MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder)))
      rayTraceData.minSL = MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder));
    if(rayTraceData.minSL > M_PI)
      rayTraceData.minSL = M_PI;
#endif
  
#ifdef SHTONLY  
  rayTraceData.poissonOrder = rayTraceData.SHTOrder;
  rayTraceData.minSL = MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.poissonOrder));
  rayTraceData.maxSL = MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.poissonOrder));
  
  //big enough to make sure we get all parts needed 
  rayTraceData.partBuffRad = sqrt(4.0*M_PI/order2npix(rayTraceData.poissonOrder))*10.0 + 2.0*bundleLength + rayTraceData.maxSL*2.0;
  if(ThisTask == 0)
    {
      fprintf(stderr,"densfact = %le, backdens = %le, max smoothing scale = %le, cmv dist. = %lg\n",
	      rayTraceData.densfact,rayTraceData.backdens,rayTraceData.maxComvSmoothingScale/rayTraceData.planeRad,rayTraceData.planeRad);
      fprintf(stderr,"SHT order = %ld, partBuffRad = %lg\n",rayTraceData.poissonOrder
	      ,rayTraceData.partBuffRad);
    }
#elif defined(TREEPM)
  double thetaS;
  long testOrder;
  long minOrder;
  
  if(rayTraceData.bundleOrder < rayTraceData.minSHTOrder)
    minOrder = rayTraceData.bundleOrder;
  else if(rayTraceData.minSHTOrder > 0)
    minOrder = rayTraceData.minSHTOrder;
  else
    minOrder = rayTraceData.bundleOrder;
  
  if(rayTraceData.minSL >= MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.SHTOrder)))
    rayTraceData.TreePMOnlyDoSHT = 1;
  else
    rayTraceData.TreePMOnlyDoSHT = 0;
  
  if(!rayTraceData.TreePMOnlyDoSHT)
    {
      testOrder = rayTraceData.SHTOrder+1;
      do
	{
	  --testOrder;
	      
	  thetaS = sqrt(4.0*M_PI/order2npix(testOrder))*SHTSPLITFACTOR;
	      
	  if(thetaS/rayTraceData.maxSL >= MIN_SPLIT_TO_SMOOTH_RATIO)
	    break;
	}
      while(testOrder > minOrder);
      
      rayTraceData.poissonOrder = testOrder;
      
      if(thetaS/rayTraceData.maxSL < MIN_SPLIT_TO_SMOOTH_RATIO)
	thetaS = rayTraceData.maxSL*MIN_SPLIT_TO_SMOOTH_RATIO;
      rayTraceData.TreePMSplitScale = thetaS;
      
      rayTraceData.partBuffRad = MAX_RADTREEWALK_TO_SPLIT_RATIO*rayTraceData.TreePMSplitScale + 2.0*bundleLength + rayTraceData.maxSL*2.0;
    }
  else
    {
      if(ThisTask == 0)
	fprintf(stderr,"only doing SHT for this lens plane!\n");
      
      rayTraceData.poissonOrder = rayTraceData.SHTOrder;
      rayTraceData.partBuffRad = sqrt(4.0*M_PI/order2npix(rayTraceData.poissonOrder))*10.0 + 2.0*bundleLength + rayTraceData.maxSL*2.0;
    }

  if(ThisTask == 0)
    {
      fprintf(stderr,"densfact = %le, backdens = %le, max smoothing scale = %le, cmv dist. = %lg\n",
	      rayTraceData.densfact,rayTraceData.backdens,rayTraceData.maxComvSmoothingScale/rayTraceData.planeRad,rayTraceData.planeRad);
      fprintf(stderr,"SHT order = %ld, partBuffRad = %lg, splitScale = %le [%f SHT cells sizes]\n",rayTraceData.poissonOrder
	      ,rayTraceData.partBuffRad,rayTraceData.TreePMSplitScale,rayTraceData.TreePMSplitScale/sqrt(4.0*M_PI/order2npix(testOrder)));
    }
#else
  rayTraceData.poissonOrder = rayTraceData.SHTOrder;
  rayTraceData.NumMGPatch = MGPATCH_SIZE_FAC*bundleLength/(rayTraceData.minSL/SMOOTHKERN_MGRESOLVE_FAC);
  if(rayTraceData.NumMGPatch < NUM_MGPATCH_MIN)
    rayTraceData.NumMGPatch = NUM_MGPATCH_MIN;
    
  //big enough to make sure we get all parts needed
  rayTraceData.partBuffRad = MGPATCH_SIZE_FAC*bundleLength + 2.0*bundleLength + rayTraceData.maxSL*2.0;
  
  if(ThisTask == 0)
    {
      fprintf(stderr,"densfact = %le, backdens = %le, max smoothing scale = %le, cmv dist. = %lg\n",
	      rayTraceData.densfact,rayTraceData.backdens,rayTraceData.maxComvSmoothingScale/rayTraceData.planeRad,rayTraceData.planeRad);
      fprintf(stderr,"SHT order = %ld, partBuffRad = %lg, minSL = %le, apprx. # of cells in MG patch = %ld, # of MG cells per minSL = %lf, MG res fact = %lf\n",rayTraceData.poissonOrder
	      ,rayTraceData.partBuffRad,rayTraceData.minSL,rayTraceData.NumMGPatch,rayTraceData.minSL/((MGPATCH_SIZE_FAC*bundleLength)/(rayTraceData.NumMGPatch)),SMOOTHKERN_MGRESOLVE_FAC);
    }
#endif
}
