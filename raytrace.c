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
#include "checked_alloc.h"
#include "checked_io.h"

static void set_plane_params(void);

void raytrace(void)
{
  long i,j;
  double time,minTime,maxTime;//,totTime,avgTime;
  double stepTime,restTime,startTime;
  int writeRestartFile;
  char pname[MAX_FILENAME];
  FILE *fpStepTime = NULL;
#ifdef NOBACKDENS
  long totNlensPlaneParts;
#endif

  //Need to clean this up. Quick and dirty for map debugging.
  long NMaps;
  char* output_filename[MAX_FILENAME];
  char* temporary_output_filename[MAX_FILENAME];
  char sys                      [MAX_FILENAME];  
  int MapPlaneFlag = 0;
  int* lp_map;
  long* map_pixel_sum_1;
  double* map_pixel_sum_A00;
  double* map_pixel_sum_A01;
  double* map_pixel_sum_A10;
  double* map_pixel_sum_A11;
  double* map_pixel_sum_ra;
  double* map_pixel_sum_dec;
  const long map_order    = 11;
  const long map_n_side   = (1 << map_order);
  const long map_n_pixels = 12 * map_n_side * map_n_side;

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
	{
	  fprintf(stderr,"\nreading restart files in directory '%s'\n",rayTraceData.OutputPath);
	  fflush(stderr);
	}

      logProfileTag(PROFILETAG_RESTART);
      read_restart();
      logProfileTag(PROFILETAG_RESTART);
    }
  else
    {
      logProfileTag(PROFILETAG_INITEND_LOADBAL);
      if(ThisTask == 0)
	{
	  fprintf(stderr,"initializing domain decomposition.\n");
	  fflush(stderr);
	}
      init_bundlecells();

      if(ThisTask == 0)
	{
	  fprintf(stderr,"initializing rays.\n\n");
	  fflush(stderr);
	}
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

  if(strlen(rayTraceData.MapRedshiftList) > 0)
    {
      getNMaps(&NMaps);
      lp_map = (int*)malloc(sizeof(int)*NMaps);
      getMapLensPlaneNums(lp_map, NMaps);
      for (i=0;i<NMaps;i++)
	{
	  fprintf(stderr, "Map lens plane %d: %d\n", i, lp_map[i]);
	}
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
	      fflush(stderr);
	    }
	  else if(time >= rayTraceData.WallTimeBetweenRestart)
	    {
	      restTime = MPI_Wtime();
	      writeRestartFile = 1;
	      fprintf(stderr,"\nwriting restart files at plane %ld.\n",rayTraceData.CurrentPlaneNum);
	      fflush(stderr);
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

      //set units and poisson solve type
      set_plane_params();

      //load balance the nodes
      logProfileTag(PROFILETAG_INITEND_LOADBAL);
      load_balance_tasks();
      logProfileTag(PROFILETAG_INITEND_LOADBAL);

      if (strlen(rayTraceData.MapRedshiftList)>0)
      {
        if (((rayTraceData.CurrentPlaneNum+1)==lp_map[rayTraceData.CurrentMapNum]) && !rayTraceData.MaxResMap)
        {
            MapPlaneFlag = 1;
            map_pixel_sum_1              = (long*  ) checked_calloc(map_n_pixels, sizeof(long  ));
            map_pixel_sum_A00            = (double*) checked_calloc(map_n_pixels, sizeof(double));
            map_pixel_sum_A01            = (double*) checked_calloc(map_n_pixels, sizeof(double));
            map_pixel_sum_A10            = (double*) checked_calloc(map_n_pixels, sizeof(double));
            map_pixel_sum_A11            = (double*) checked_calloc(map_n_pixels, sizeof(double));
            map_pixel_sum_ra             = (double*) checked_calloc(map_n_pixels, sizeof(double));
            map_pixel_sum_dec            = (double*) checked_calloc(map_n_pixels, sizeof(double));
        } else if ((rayTraceData.CurrentPlaneNum+1)==lp_map[rayTraceData.CurrentMapNum])
        {
            write_rays(rayTraceData.CurrentMapNum);
        }
      }

      //do gals grid search
      if(strlen(rayTraceData.GalsFileList) > 0)
	{
	  logProfileTag(PROFILETAG_GRIDSEARCH);
	  gridsearch(rayTraceData.planeRad,rayTraceData.planeRadMinus1);
	  logProfileTag(PROFILETAG_GRIDSEARCH);
	}

      //zero everything before force computation
      for(i=0;i<NbundleCells;++i)
	{
	  if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	    {
	      for(j=0;j<bundleCells[i].Nrays;++j)
		{
		  bundleCells[i].rays[j].phi = 0.0;

		  bundleCells[i].rays[j].alpha[0] = 0.0;
		  bundleCells[i].rays[j].alpha[1] = 0.0;

		  bundleCells[i].rays[j].U[0] = 0.0;
		  bundleCells[i].rays[j].U[1] = 0.0;
		  bundleCells[i].rays[j].U[2] = 0.0;
		  bundleCells[i].rays[j].U[3] = 0.0;
		}
	    }
	}

      //run Poisson solver
#ifndef THREEDPOT
#ifdef USE_FULLSKY_PARTDIST
      fullsky_partdist_poissondriver();
#else
      cutsky_partdist_poissondriver();
#endif
#else
      threedpot_poissondriver();
#endif

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
          if (MapPlaneFlag)
          {
            updateLensMap(&bundleCells[i], map_order, map_pixel_sum_1, map_pixel_sum_A00, map_pixel_sum_A01, map_pixel_sum_A10, map_pixel_sum_A11, map_pixel_sum_ra, map_pixel_sum_dec);
          }
	    }
	}

    if(MapPlaneFlag)
    { /* reduce and write map */
      if (ThisTask==0)
      { fprintf(stderr, "task %d: reducing rays...\n", ThisTask); }

       MPI_ReduceLensMap(map_pixel_sum_1, map_pixel_sum_A00, map_pixel_sum_A01,
                          map_pixel_sum_A10, map_pixel_sum_A11,
                          map_pixel_sum_ra, map_pixel_sum_dec,
                          map_n_pixels, 0);

      if(ThisTask==0)
      { fprintf(stderr, "task %d: reduced rays.\n", ThisTask); }

      if (ThisTask==0)
      {
        sprintf(output_filename, "%s/Convergence_%ld_%ld.fits", rayTraceData.OutputPath, map_n_side, rayTraceData.CurrentMapNum);
        fprintf(stderr, "task %d: writing convergence map to file '%s'...\n", ThisTask, output_filename);

        if(!file_exists(output_filename))
        {
          sprintf(temporary_output_filename, "%s.tmp", output_filename);

          /* fits_create_file() (in writeHealpixLensMap) seems not to like to overwrite existing files */
          if(file_exists(temporary_output_filename))
          { remove(temporary_output_filename);}

          float* convergence = (float*) checked_malloc (sizeof(float) * map_n_pixels);
          for(i = 0; i < map_n_pixels; i++)
          { convergence[i] = (map_pixel_sum_1[i] <= 0) ? 0.0 : 1.0 - 0.5 * (map_pixel_sum_A00[i] + map_pixel_sum_A11[i]) / map_pixel_sum_1[i]; }
          writeSingleFITSHEALPixLensMap(convergence, map_n_side, temporary_output_filename);
          free(convergence);

          sprintf(sys,"mv %s %s",temporary_output_filename, output_filename);
          system(sys);
        }

        sprintf(output_filename, "%s/Rays_%ld_%ld.fits", rayTraceData.OutputPath, map_n_side, rayTraceData.CurrentMapNum);
        fprintf(stderr, "task %d: writing ray map to file '%s'...\n", ThisTask, output_filename);

        if(!file_exists(output_filename))
        {
          /* fits_create_file() (in writeLensMap) seems not to like to overwrite existing files */
          if(file_exists(temporary_output_filename))
          { remove(temporary_output_filename); }

          writeFITSHEALPixLensMap(map_pixel_sum_1, map_pixel_sum_A00, map_pixel_sum_A01, map_pixel_sum_A10,
                       map_pixel_sum_A11, map_pixel_sum_ra, map_pixel_sum_dec,
                       map_n_side, temporary_output_filename);

          sprintf(sys,"mv %s %s", temporary_output_filename, output_filename);
          system(sys);
        }
      }
      free(map_pixel_sum_1);
      free(map_pixel_sum_A00);
      free(map_pixel_sum_A01);
      free(map_pixel_sum_A10);
      free(map_pixel_sum_A11);
      free(map_pixel_sum_ra);
      free(map_pixel_sum_dec);
      rayTraceData.CurrentMapNum += 1;
      MapPlaneFlag = 0;
    } /* reduce and write map */

      logProfileTag(PROFILETAG_STEPTIME);
      stepTime += MPI_Wtime();

      if(ThisTask == 0)
	{
	  printStepTimesProfileTags(fpStepTime,rayTraceData.CurrentPlaneNum,NULL);
	  fprintf(stderr,"\n");
	  fflush(stderr);
	}

    } // end of main driver loop for simulation

  ///////////////////////////
  // clean up and finalize //
  ///////////////////////////
  if(rayTraceData.CurrentPlaneNum == rayTraceData.NumLensPlanes)
    {
      if(ThisTask == 0)
	{
	  fprintf(stderr,"finished ray tracing for all lens planes.\n");
	  fflush(stderr);
	}

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
#elif defined(NFWHALOTEST)
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
#elif defined(NFWHALOTEST)
  if(rayTraceData.minSL < MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder)))
    rayTraceData.minSL = MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder));
  if(rayTraceData.minSL > M_PI)
    rayTraceData.minSL = M_PI;
#endif

  if(!rayTraceData.UseHEALPixLensPlaneMaps)
    rayTraceData.poissonOrder = rayTraceData.SHTOrder;
  else
    rayTraceData.poissonOrder = rayTraceData.HEALPixLensPlaneMapOrder;

  //do some helpful I/O
  if(ThisTask == 0) {
    fprintf(stderr,"planeNum = %04ld (% 4ld of % 4ld) [dist = %.2lf Mpc/h, z = %.2lf]\n",rayTraceData.CurrentPlaneNum,rayTraceData.CurrentPlaneNum+1,rayTraceData.NumLensPlanes,
	    rayTraceData.planeRad,1.0/acomvdist(rayTraceData.planeRad)-1.0);
    /*#ifndef THREEDPOT
    fprintf(stderr,"lens plane: '%s/%s%04ld.h5'\n", rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,rayTraceData.CurrentPlaneNum);
    #endif*/
    fflush(stderr);
  }

#ifdef SHTONLY
  rayTraceData.minSL = MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.poissonOrder));
  rayTraceData.maxSL = MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.poissonOrder));

  //big enough to make sure we get all parts needed
  rayTraceData.partBuffRad = sqrt(4.0*M_PI/order2npix(rayTraceData.poissonOrder))*10.0 + 2.0*bundleLength + rayTraceData.maxSL*2.0;
  if(ThisTask == 0)
    {
      fprintf(stderr,"densfact = %le, backdens = %le, cmv dist. = %lg\n",
	      rayTraceData.densfact,rayTraceData.backdens,rayTraceData.planeRad);
      fprintf(stderr,"SHT order = %ld, partBuffRad = %lg\n",rayTraceData.poissonOrder
	      ,rayTraceData.partBuffRad);
      fflush(stderr);
    }
#elif defined(THREEDPOT)
  rayTraceData.minSL = MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder));
  rayTraceData.maxSL = MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder));
#else
  rayTraceData.NumMGPatch = MGPATCH_SIZE_FAC*bundleLength/(rayTraceData.minSL/SMOOTHKERN_MGRESOLVE_FAC);
  if(rayTraceData.NumMGPatch < NUM_MGPATCH_MIN)
    rayTraceData.NumMGPatch = NUM_MGPATCH_MIN;

  //big enough to make sure we get all parts needed
  rayTraceData.partBuffRad = MGPATCH_SIZE_FAC*bundleLength + 2.0*bundleLength + rayTraceData.maxSL*2.0;
  if(ThisTask == 0)
    {
      fprintf(stderr,"densfact = %le, backdens = %le, cmv dist. = %lg\n",
	      rayTraceData.densfact,rayTraceData.backdens,rayTraceData.planeRad);
      fprintf(stderr,"SHT order = %ld, partBuffRad = %lg, apprx. # of cells in MG patch = %ld, # of MG cells per minSL = %lf, MG res fact = %lf\n",rayTraceData.poissonOrder
	      ,rayTraceData.partBuffRad,rayTraceData.NumMGPatch,rayTraceData.minSL/((MGPATCH_SIZE_FAC*bundleLength)/(rayTraceData.NumMGPatch)),SMOOTHKERN_MGRESOLVE_FAC);
      fflush(stderr);
    }
#endif
}
