#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>

#include "raytrace.h"

static SourceGal *distribute_gals_to_nodes(long NumGalsForThisPlane, long start, long *NumGals);
static void gridsearch_gals(SourceGal *gals, long NumGals, double wpm1, double wpm2, HEALPixRay *bufferRays, long NumBufferRays);
static void gridsearch_gals_bornapprx(SourceGal *gals, long NumGals, double wpm1, double wpm2);
static void gridsearch_gals_nobornapprx(SourceGal *gals, long NumGals, double wpm1, double wpm2, HEALPixRay *bufferRays, long NumBufferRays);
static int tritest_getbarycoords(double a[2], double b[2], double c[2], double q[2], double barycoords[3]);
static double trisarea(double a[2], double b[2], double c[2]);
static HEALPixRay *get_buffer_rays(long *NumBufferRays);
static void destroy_buffer_rays(HEALPixRay *bufferRays);
static void rayprop_gridsearch(HEALPixRay *ray, double wp, double wpm1, double wpm2);
static int interp_invmagmat_to_point(double ivec[3], double galRad, double wpm1, double wpm2, double A[2][2]);

void gridsearch(double wpm1, double wpm2)
{
  SourceGal *galsForThisPlane;
  long i,NumGalsForThisPlane,start;
  double rad,binL;
  long bind;
  long TotNumImageGalsGlobal,TotNumGalsForThisPlane;
  double time,t;
  HEALPixRay *bufferRays;
  long NumBufferRays;
    
  time = -MPI_Wtime();
  if(ThisTask == 0)
    fprintf(stderr,"finding galaxy images with grid search.\n");
  
  //get gals for this plane
  binL = rayTraceData.maxComvDistance/rayTraceData.NumLensPlanes;
  NumGalsForThisPlane = 0;
  start = -1;
  for(i=NumSourceGalsGlobal-1;i>=0;--i)
    {
      rad = sqrt(SourceGalsGlobal[i].pos[0]*SourceGalsGlobal[i].pos[0] + 
		 SourceGalsGlobal[i].pos[1]*SourceGalsGlobal[i].pos[1] + 
		 SourceGalsGlobal[i].pos[2]*SourceGalsGlobal[i].pos[2]);
      bind = (long) (rad/binL);
      
      //catch gals past last plane within 1 kpc/h of edge for last plane
      if(bind == rayTraceData.NumLensPlanes && fabs(rad-rayTraceData.maxComvDistance) < 1e-3)
	bind = rayTraceData.NumLensPlanes-1;
            
      if(bind == rayTraceData.CurrentPlaneNum)
	++NumGalsForThisPlane;
	      
      if(bind > rayTraceData.CurrentPlaneNum)
	{
	  start = i+1; //go back one index - an increment because we are working from the back of the list
	  break;
	}
    }
  if(i == -1)
    start = 0;
  
  if(NumGalsForThisPlane == 0)
    start = 0;
  
  assert(start >= 0);
  
#ifdef DEBUG
#if DEBUG_LEVEL > 1
  fprintf(stderr,"%05d: # of gals for this plane = %ld (of %ld), start = %ld\n",ThisTask,NumGalsForThisPlane,NumSourceGalsGlobal,start);
#endif
#endif
  
  MPI_Allreduce(&NumGalsForThisPlane,&TotNumGalsForThisPlane,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  if(ThisTask == 0)
    fprintf(stderr,"found %ld source galaxies for this plane.\n",TotNumGalsForThisPlane);
  
  //do grid search if there are any gals on any task that need images
  if(TotNumGalsForThisPlane > 0)
    {
      /*
	Do this as follows
	1) get buffer rays
	2) loop through gal chunks and do grid search
	3) write images to disk
	4) destroy buffer rays
      */
            
      // 1)
      //get buffer rays
      logProfileTag(PROFILETAG_GRIDSEARCH);
      logProfileTag(PROFILETAG_RAYBUFF);
      t = -MPI_Wtime();
      if(ThisTask == 0)
	fprintf(stderr,"starting to get buffer rays.\n");
      bufferRays = get_buffer_rays(&NumBufferRays);
      t += MPI_Wtime();
      if(ThisTask == 0)
	fprintf(stderr,"got buffer rays in %lf seconds.\n",t);
      logProfileTag(PROFILETAG_RAYBUFF);
      logProfileTag(PROFILETAG_GRIDSEARCH);
      
      // 2)
      //get right galaxies for this plane
      logProfileTag(PROFILETAG_GRIDSEARCH_GALMOVE);
      t = -MPI_Wtime();
      galsForThisPlane = distribute_gals_to_nodes(NumGalsForThisPlane,start,&i);
      NumGalsForThisPlane = i;
      t += MPI_Wtime();
      //if(ThisTask == 0)
      //fprintf(stderr,"distributing gals to tasks took %lf seconds.\n",t);
      logProfileTag(PROFILETAG_GRIDSEARCH_GALMOVE);
      
      //reorder gals by their nest index - should help with cache misses during grid search
      if(NumGalsForThisPlane > 0)
	{
	  logProfileTag(PROFILETAG_GRIDSEARCH_GALGRIDSEARCH);
	  reorder_gals_nest(galsForThisPlane,NumGalsForThisPlane);
	}
      else
	{
	  galsForThisPlane = NULL;
	  NumGalsForThisPlane = 0;
	}
      
      if(ThisTask == 0)
	fprintf(stderr,"found %ld gals for this plane on this task.\n",NumGalsForThisPlane);
      
      //do grid search
      gridsearch_gals(galsForThisPlane,NumGalsForThisPlane,wpm1,wpm2,bufferRays,NumBufferRays);
#ifdef DEBUG
#if DEBUG_LEVEL > 1
      if(NumGalsForThisPlane > 0)
	fprintf(stderr,"%05d: finished grid search, found %ld images\n",ThisTask,NumImageGalsGlobal);
#endif
#endif
      
      if(NumGalsForThisPlane > 0)
	logProfileTag(PROFILETAG_GRIDSEARCH_GALGRIDSEARCH);
      
      MPI_Allreduce(&NumImageGalsGlobal,&TotNumImageGalsGlobal,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
      if(ThisTask == 0)
	fprintf(stderr,"found %ld image gals from %ld source gals.\n",TotNumImageGalsGlobal,TotNumGalsForThisPlane);
      
      // 3) - write images to disk if needed
      if(TotNumImageGalsGlobal > 0)
	{
	  logProfileTag(PROFILETAG_GRIDSEARCH);
	  logProfileTag(PROFILETAG_GALIO);
	  logProfileTag(PROFILETAG_GRIDSEARCH_IMAGEGALIO);
	  
	  write_gals2fits();
	  if(NumImageGalsGlobal > 0)
	    {
	      free(ImageGalsGlobal);
	      ImageGalsGlobal = NULL;
	      NumImageGalsGlobal = 0;
	    }
	  
	  logProfileTag(PROFILETAG_GRIDSEARCH_IMAGEGALIO);
	  logProfileTag(PROFILETAG_GALIO);
	  logProfileTag(PROFILETAG_GRIDSEARCH);
	}
#ifdef DEBUG
#if DEBUG_LEVEL > 1
      fprintf(stderr,"%05d: wrote gals to disk if needed\n",ThisTask);
#endif
#endif
      
      if(NumGalsForThisPlane > 0)
	free(galsForThisPlane);
      
      // 4) - free some mem
      destroy_buffer_rays(bufferRays);
    }
  
  time += MPI_Wtime();
  if(ThisTask == 0)
    fprintf(stderr,"found galaxy images in %lg seconds.\n",time);
  
  /*MPI_Reduce(&time,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);                                                                                                                                
    MPI_Reduce(&time,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);                                                                                                                                
    MPI_Reduce(&time,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);                                                                                                                                
    avgTime = totTime/((double) NTasks);                                                                                                                                                             
    if(ThisTask == 0)                                                                                                                                                                                
    fprintf(stderr,"grid search time max,min,avg = %lf|%lf|%lf - %.2f percent (total time %lf)\n"                                                                                                    
    ,maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,totTime);                                                                                                                               
  */
}

//util function for finding rays with nest value
static int compHEALPixRayNest(const void *a, const void *b)
{
  if(((const HEALPixRay*)a)->nest > ((const HEALPixRay*)b)->nest)
    return 1;
  else if(((const HEALPixRay*)a)->nest < ((const HEALPixRay*)b)->nest)
    return -1;
  else
    return 0;
}

//macro for error checking grid search prints out info for galaxy with ind CHECK_GS_IND
//#define CHECK_GS
#define CHECK_GS_IND 1370

static int interp_invmagmat_to_point(double ivec[3], double galRad, double wpm1, double wpm2, double A[2][2])
{
  long n;
  long fnd;
  double nvec[3],ttens[2][2],rttens[2][2];
  double theta,phi,wgt[4];
  long Nwgt,wgtpix[4];
  HEALPixRay wgtRays[4];
  long snest,bnest,bundleRayShift,roffset;
  HEALPixRay key,*fndRay;
  bundleRayShift = 2*(rayTraceData.rayOrder - rayTraceData.bundleOrder);
  
  //get shear matrix
  A[0][0] = 0.0;
  A[0][1] = 0.0;
  A[1][0] = 0.0;
  A[1][1] = 0.0;
  
  vec2ang(ivec,&theta,&phi);
  Nwgt = 4;
  get_interpol(theta,phi,wgtpix,wgt,rayTraceData.rayOrder);
  fnd = 1;
  for(n=0;n<Nwgt;++n)
    {
      //find the ray you need
      snest = ring2nest(wgtpix[n],rayTraceData.rayOrder);
      bnest = snest >> bundleRayShift;
      
      if(ISSETBITFLAG(bundleCells[bnest].active,PRIMARY_BUNDLECELL) && bundleCells[bnest].Nrays > 0)
	{
	  roffset = snest - (bnest << bundleRayShift);
	  
	  wgtRays[n] = bundleCells[bnest].rays[roffset]; //makes a copy of the ray via a structure assignemnt
	  assert(wgtRays[n].nest == snest);
	}
      else if(ISSETBITFLAG(bundleCells[bnest].active,RAYBUFF_BUNDLECELL) && bundleCells[bnest].Nrays > 0)
	{
	  key.nest = snest;
	  fndRay = (HEALPixRay*)bsearch(&key,bundleCells[bnest].rays,(size_t) (bundleCells[bnest].Nrays),sizeof(HEALPixRay),compHEALPixRayNest);
	  
	  if(fndRay != NULL)
	    {
	      wgtRays[n] = *fndRay;  //makes a copy of the ray via a structure assignemnt
	      assert(wgtRays[n].nest == snest);
	    }
	  else
	    {
	      fnd = 0;
	      break;
	    }
	}
      else if(bundleCellsNest2RestrictedPeanoInd[bnest] == -1) //never made ray in first place, so move on
	{
	  fnd = 0;
	  break;
	}
      else
	{
#ifndef CHECK_GS
	  fprintf(stderr,"%d: ray buffer regions for grid search are not large enough\n",ThisTask);
	  MPI_Abort(MPI_COMM_WORLD,999);
#else
	  fnd = -1;
	  break;
#endif
	}

      //prop to right spot in plane
      rayprop_gridsearch(&(wgtRays[n]),galRad,wpm1,wpm2);

      //para trans to image spot
      rttens[0][0] = wgtRays[n].A[2*0+0];
      rttens[0][1] = wgtRays[n].A[2*0+1];
      rttens[1][0] = wgtRays[n].A[2*1+0];
      rttens[1][1] = wgtRays[n].A[2*1+1];
      nest2vec(snest,nvec,rayTraceData.rayOrder);
      paratrans_tangtensor(rttens,wgtRays[n].n,nvec,ttens);

      //para trans to galaxy
      paratrans_tangtensor(ttens,nvec,ivec,rttens);

      //interp
      A[0][0] += rttens[0][0]*wgt[n];
      A[0][1] += rttens[0][1]*wgt[n];
      A[1][0] += rttens[1][0]*wgt[n];
      A[1][1] += rttens[1][1]*wgt[n];
    }

  return fnd;
}

static void gridsearch_gals(SourceGal *gals, long NumGals, double wpm1, double wpm2, HEALPixRay *bufferRays, long NumBufferRays)
{
  int useborn;

#ifdef BORNAPPRX
  useborn = 1;
#else
  useborn = 0;
#endif
  
  if(useborn)
    gridsearch_gals_bornapprx(gals,NumGals,wpm1,wpm2);
  else
    gridsearch_gals_nobornapprx(gals,NumGals,wpm1,wpm2,bufferRays,NumBufferRays);
}

static void gridsearch_gals_bornapprx(SourceGal *gals, long NumGals, double wpm1, double wpm2)
{
  long i;
  long NumImageGalsGlobalAlloc;
  ImageGal *tmpImageGal;
  double vec[3];
  double galRad;
  long fnd;
  double ivec[3],ra,dec;
  double ttens_interp[2][2],Aradec[2][2];
    
  if(ThisTask == 0)
    fprintf(stderr,"doing grid search w/ born apprx.\n");
  
  double timeBCSTestInterp;
  timeBCSTestInterp = 0.0;
  
  if(NumGals > 0)
    {
      //alloc image gals
      NumImageGalsGlobalAlloc = NumGals;
      ImageGalsGlobal = (ImageGal*)malloc(sizeof(ImageGal)*NumImageGalsGlobalAlloc);
      assert(ImageGalsGlobal != NULL);
      NumImageGalsGlobal = 0;
      
      //get the images
      timeBCSTestInterp -= MPI_Wtime();
      for(i=0;i<NumGals;++i)
        {
	  //get gal pos on sky
          vec[0] = gals[i].pos[0];
          vec[1] = gals[i].pos[1];
          vec[2] = gals[i].pos[2];
          galRad = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
          vec[0] = vec[0]/galRad;
          vec[1] = vec[1]/galRad;
          vec[2] = vec[2]/galRad;
	  ivec[0] = vec[0]/galRad;
          ivec[1] = vec[1]/galRad;
          ivec[2] = vec[2]/galRad;
	  
	  //do shear interp - if works, then record an image gal
	  fnd = interp_invmagmat_to_point(ivec,galRad,wpm1,wpm2,ttens_interp);
	  if(fnd == 1)
	    {
	      
	      //rotate gal to ra dec coords
	      vec2radec(ivec,&ra,&dec);
	      Aradec[0][0] = ttens_interp[1][1];
	      Aradec[0][1] = -ttens_interp[1][0];
	      Aradec[1][0] = -ttens_interp[0][1];
	      Aradec[1][1] = ttens_interp[0][0];
	      
	      //add image to list
	      ImageGalsGlobal[NumImageGalsGlobal].index = gals[i].index;
	      ImageGalsGlobal[NumImageGalsGlobal].ra = ra;
	      ImageGalsGlobal[NumImageGalsGlobal].dec = dec;
	      ImageGalsGlobal[NumImageGalsGlobal].A00 = Aradec[0][0];
	      ImageGalsGlobal[NumImageGalsGlobal].A01 = Aradec[0][1];
	      ImageGalsGlobal[NumImageGalsGlobal].A10 = Aradec[1][0];
	      ImageGalsGlobal[NumImageGalsGlobal].A11 = Aradec[1][1];
	      ++NumImageGalsGlobal;
	      
	      if(NumImageGalsGlobal >= NumImageGalsGlobalAlloc)
		{
		  NumImageGalsGlobalAlloc += 10;
		  tmpImageGal = (ImageGal*)realloc(ImageGalsGlobal,sizeof(ImageGal)*NumImageGalsGlobalAlloc);
		  assert(tmpImageGal != NULL);
		  ImageGalsGlobal = tmpImageGal;
		}
	    }
	  
	}//for(i=0;i<NumGals;++i)
      timeBCSTestInterp += MPI_Wtime();
      
      //realloc image gals if needed
      if(NumImageGalsGlobal < NumImageGalsGlobalAlloc && NumImageGalsGlobal > 0)
        {
          tmpImageGal = (ImageGal*)realloc(ImageGalsGlobal,sizeof(ImageGal)*NumImageGalsGlobal);
          assert(tmpImageGal != NULL);
          ImageGalsGlobal = tmpImageGal;
        }
      else if(NumImageGalsGlobal == 0)
        {
          free(ImageGalsGlobal);
          ImageGalsGlobal = NULL;
        }
    }
  else
    {
      NumImageGalsGlobal = 0;
      ImageGalsGlobal = NULL;
    }
  
  
  if(ThisTask == 0)
    fprintf(stderr,"galaxy image interp took %lg seconds.\n",timeBCSTestInterp);
}

static void gridsearch_gals_nobornapprx(SourceGal *gals, long NumGals, double wpm1, double wpm2, HEALPixRay *bufferRays, long NumBufferRays)
{
  long i,j,k,n;
  long NumImageGalsGlobalAlloc;
  ImageGal *tmpImageGal;
  long ring;
  double rvec[3],vec[3],tvec[3],pvec[3],npvec;
  HEALPixTreeData *tdvec[2];
  long tdInd;
  HEALPixRay *raysVec[2];
  NNbrData *NNbrs;
  long maxNumNNbrs,NumNNbrs;
  double galRad;
  long Ntri,tri[4][3];
  HEALPixRay triRays[3],tmpRay;
  long fnd;
  double area,bcs[3],galPos[2],triPos[3][2],triPosCurr[3][2],x,y;
  double ivec[3],ra,dec;
  double ttens_interp[2][2],Aradec[2][2];
  double cosangCurr[3];
  long snest,bnest,bundleRayShift,roffset;
  bundleRayShift = 2*(rayTraceData.rayOrder - rayTraceData.bundleOrder);
  HEALPixRay keyHEALPixRay,*fndHEALPixRay;
  
#ifdef CHECK_GS
  long inval;
  FILE *fp;
  char fname[MAX_FILENAME];
  sprintf(fname,"%s/gridsearchinfo_ind%d.txt",rayTraceData.OutputPath,CHECK_GS_IND);
#endif
  
  if(ThisTask == 0)
    fprintf(stderr,"doing grid search.\n");
  
  double timeTreeBuild,timeTreeSearch,timeBCSTestInterp;
  timeTreeBuild = 0.0;
  timeTreeSearch = 0.0;
  timeBCSTestInterp = 0.0;
  
  if(NumGals > 0)
    {
      //build a tree for each bundle cell
      timeTreeBuild -= MPI_Wtime();
      tdvec[0] = buildHEALPixTree(NumAllRaysGlobal,AllRaysGlobal);
      raysVec[0] = AllRaysGlobal;
      tdvec[1] = buildHEALPixTree(NumBufferRays,bufferRays);
      raysVec[1] = bufferRays;
      timeTreeBuild += MPI_Wtime();
      
      NumImageGalsGlobalAlloc = NumGals;
      ImageGalsGlobal = (ImageGal*)malloc(sizeof(ImageGal)*NumImageGalsGlobalAlloc);
      assert(ImageGalsGlobal != NULL);
      NumImageGalsGlobal = 0;
      
      NNbrs = NULL;
      maxNumNNbrs = 0;
      
      galPos[0] = 0.0;
      galPos[1] = 0.0;
      
      for(i=0;i<NumGals;++i)
	{      
	  
#ifdef CHECK_GS
	  if(CHECK_GS_IND == gals[i].index)
	    {
	      fp = fopen(fname,"w");
	      assert(fp != NULL);
	    }
#endif
	  
	  //get gal pos on sky and init basis vecs
	  timeBCSTestInterp -= MPI_Wtime();
	  
	  vec[0] = gals[i].pos[0];
	  vec[1] = gals[i].pos[1];
	  vec[2] = gals[i].pos[2];
	  galRad = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
	  vec[0] = vec[0]/galRad;
	  vec[1] = vec[1]/galRad;
	  vec[2] = vec[2]/galRad;
	  
	  npvec = sqrt(vec[0]*vec[0] + vec[1]*vec[1]);
	  pvec[0] = -vec[1]/npvec;
	  pvec[1] = vec[0]/npvec;
	  pvec[2] = 0.0;
	  
	  tvec[0] = vec[2]*vec[0]/npvec;
	  tvec[1] = vec[2]*vec[1]/npvec;
	  tvec[2] = -1.0*(vec[0]*vec[0] + vec[1]*vec[1])/npvec;
	  
	  timeBCSTestInterp += MPI_Wtime();
	  
	  for(tdInd=0;tdInd<2;++tdInd)
	    {
	      //find all rays near it
	      timeTreeSearch -= MPI_Wtime();
	      NumNNbrs = nnbrsHEALPixTree(vec,rayTraceData.galImageSearchRad,wpm1,raysVec[tdInd],tdvec[tdInd],&NNbrs,&maxNumNNbrs);
	      timeTreeSearch += MPI_Wtime();
	      
	      timeBCSTestInterp -= MPI_Wtime();
#ifdef DEBUG
#if DEBUG_LEVEL > 2
	      fprintf(stderr,"%05d: %ld of %ld, # of nbrs = %ld, pos = %f|%f|%f, theta,phi = %e|%e, x,y = %f|%f, phi,theta vec norm = %lg|%lg\n",
		      ThisTask,i,NumGals,NumNNbrs,vec[0],vec[1],vec[2],theta,phi,galPos[0],galPos[1],
		      sqrt(pvec[0]*pvec[0]+pvec[1]*pvec[1]+pvec[2]*pvec[2]),
		      sqrt(tvec[0]*tvec[0]+tvec[1]*tvec[1]+tvec[2]*tvec[2]));
#endif
#endif
	      
#ifdef CHECK_GS
	      if(CHECK_GS_IND == gals[i].index)
		fprintf(stderr,"%05d: gal %ld of %ld, # of nbrs = %ld, pos = %f|%f|%f, x,y = %f|%f, phi,theta vec norm = %lg|%lg, galRad = %lg\n",
			ThisTask,i,NumGals,NumNNbrs,vec[0],vec[1],vec[2],galPos[0],galPos[1],
			sqrt(pvec[0]*pvec[0]+pvec[1]*pvec[1]+pvec[2]*pvec[2]),
			sqrt(tvec[0]*tvec[0]+tvec[1]*tvec[1]+tvec[2]*tvec[2]),galRad);
#endif
	      
	      //search around each gal
	      for(j=0;j<NumNNbrs;++j)
		{
		  ring = nest2ring(raysVec[tdInd][NNbrs[j].ind].nest,rayTraceData.rayOrder);
		  Ntri = ring2triangle(ring,tri,rayTraceData.rayOrder);
		  
#ifdef CHECK_GS
		  if(CHECK_GS_IND == gals[i].index)
		    fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, %ld triangles\n",
			    ThisTask,i,NumGals,j,NumNNbrs,Ntri);
#endif
		  
		  if(Ntri > 0)
		    {
		      assert(tri[0][0] == ring);
		      triRays[0] = raysVec[tdInd][NNbrs[j].ind]; //makes a copy of the ray via a structure assignemnt
		      
		      //propagate the ray to the galaxy's comoving location
		      rayprop_gridsearch(&(triRays[0]),galRad,wpm1,wpm2);
		      
		      //get ray's projected loc near galaxy
		      cosangCurr[0] = (triRays[0].n[0]*vec[0] + triRays[0].n[1]*vec[1] + triRays[0].n[2]*vec[2])/galRad;
		      triPosCurr[0][0] = (triRays[0].n[0]*tvec[0] + triRays[0].n[1]*tvec[1] + triRays[0].n[2]*tvec[2])/galRad/cosangCurr[0];
		      triPosCurr[0][1] = (triRays[0].n[0]*pvec[0] + triRays[0].n[1]*pvec[1] + triRays[0].n[2]*pvec[2])/galRad/cosangCurr[0];
		      
		      //get ray's starting location in same coords
		      nest2vec(triRays[0].nest,rvec,rayTraceData.rayOrder);
		      triPos[0][0] = rvec[0]*tvec[0] + rvec[1]*tvec[1] + rvec[2]*tvec[2];
		      triPos[0][1] = rvec[0]*pvec[0] + rvec[1]*pvec[1] + rvec[2]*pvec[2];
#ifdef CHECK_GS
		      if(CHECK_GS_IND == gals[i].index)
			fprintf(stderr,"%05d: gal %ld of %ld (index = %ld), nbr %ld of %ld, base ray start flat pos = %lg|%lg, base ray img flat pos = %lg|%lg, ray norm = %lg\n",
				ThisTask,i,NumGals,gals[i].index,j,NumNNbrs,triPos[0][0],triPos[0][1],triPosCurr[0][0],triPosCurr[0][1],
				sqrt(triRays[0].n[0]*triRays[0].n[0] + triRays[0].n[1]*triRays[0].n[1] + triRays[0].n[2]*triRays[0].n[2]));
#endif
		    }
		  
		  for(k=0;k<Ntri;++k)
		    {
		      assert(tri[k][0] == ring);
		      
#ifdef CHECK_GS
		      if(CHECK_GS_IND == gals[i].index)
			{
			  vec2ang(vec,&theta,&phi);
			  fprintf(fp,"%.20le %.20le 0.0 0.0 ",theta,phi);
			  vec2ang(triRays[0].n,&theta,&phi);
			  fprintf(fp,"%.20le %.20le %.20le %.20le ",theta,phi,triPosCurr[0][0],triPosCurr[0][1]);
			}
#endif
		      
		      //find the rays for the triangle
		      for(n=1;n<3;++n)
			{
			  snest = ring2nest(tri[k][n],rayTraceData.rayOrder);
			  bnest = snest >> bundleRayShift;
			  
			  if(ISSETBITFLAG(bundleCells[bnest].active,PRIMARY_BUNDLECELL) && bundleCells[bnest].Nrays > 0) 
			    {
			      fnd = 1;
			      roffset = snest - (bnest << bundleRayShift);
			      triRays[n] = bundleCells[bnest].rays[roffset]; //makes a copy of the ray via a structure assignemnt
			      assert(triRays[n].nest == snest);
			    }
			  else if(ISSETBITFLAG(bundleCells[bnest].active,RAYBUFF_BUNDLECELL) && bundleCells[bnest].Nrays > 0)
			    {
			      keyHEALPixRay.nest = snest;
			      fndHEALPixRay = (HEALPixRay*)bsearch(&keyHEALPixRay,bundleCells[bnest].rays,(size_t) (bundleCells[bnest].Nrays),sizeof(HEALPixRay),compHEALPixRayNest);
			      
			      if(fndHEALPixRay != NULL)
				{
				  fnd = 1;
				  triRays[n] = *fndHEALPixRay;  //makes a copy of the ray via a structure assignemnt 
				  assert(triRays[n].nest == snest);
				}   
			      else
				{
#ifdef CHECK_GS
				  if(CHECK_GS_IND == gals[i].index)
				    fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, did not find vert %ld buffer ray search\n",
					    ThisTask,i,NumGals,j,NumNNbrs,k,Ntri,n);
#endif
				  fnd = 0;
				  break;
				}   
			    }
			  else
			    {
#ifdef CHECK_GS
			      if(CHECK_GS_IND == gals[i].index)
				fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, did not find vert %ld\n",
					ThisTask,i,NumGals,j,NumNNbrs,k,Ntri,n);
#endif
			      fnd = 0;
			      break;
			    }
			  
			  //propagate the ray to the galaxy's comoving location
#ifdef CHECK_GS
			  if(CHECK_GS_IND == gals[i].index)
			    {
			      vec2ang(triRays[n].n,&theta,&phi);
			      fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, sphere pos before prop %ld = %lg|%lg, norm = %lg\n",
				      ThisTask,i,NumGals,j,NumNNbrs,k,Ntri,n,theta,phi,
				      sqrt(triRays[n].n[0]*triRays[n].n[0] + triRays[n].n[1]*triRays[n].n[1] + triRays[n].n[2]*triRays[n].n[2]));
			    }
#endif
			  rayprop_gridsearch(&(triRays[n]),galRad,wpm1,wpm2);
#ifdef CHECK_GS
			  if(CHECK_GS_IND == gals[i].index)
			    {
			      vec2ang(triRays[n].n,&theta,&phi);
			      fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, sphere pos after prop %ld = %lg|%lg, norm = %lg\n",
				      ThisTask,i,NumGals,j,NumNNbrs,k,Ntri,n,theta,phi,
				      sqrt(triRays[n].n[0]*triRays[n].n[0] + triRays[n].n[1]*triRays[n].n[1] + triRays[n].n[2]*triRays[n].n[2]));
			    }
#endif
			  //get ray's projected loc near galaxy
			  cosangCurr[n] = (triRays[n].n[0]*vec[0] + triRays[n].n[1]*vec[1] + triRays[n].n[2]*vec[2])/galRad;
			  triPosCurr[n][0] = (triRays[n].n[0]*tvec[0] + triRays[n].n[1]*tvec[1] + triRays[n].n[2]*tvec[2])/galRad/cosangCurr[n];
			  triPosCurr[n][1] = (triRays[n].n[0]*pvec[0] + triRays[n].n[1]*pvec[1] + triRays[n].n[2]*pvec[2])/galRad/cosangCurr[n];
			  
			  //get ray's starting location in same coords                                                                                                     
			  nest2vec(triRays[n].nest,rvec,rayTraceData.rayOrder);
			  triPos[n][0] = rvec[0]*tvec[0] + rvec[1]*tvec[1] + rvec[2]*tvec[2];
			  triPos[n][1] = rvec[0]*pvec[0] + rvec[1]*pvec[1] + rvec[2]*pvec[2];
			  
#ifdef CHECK_GS
			  if(CHECK_GS_IND == gals[i].index)
			    fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, start flat pos %ld = %lg|%lg, img flat pos = %lg|%lg\n",
				    ThisTask,i,NumGals,j,NumNNbrs,k,Ntri,n,triPos[n][0],triPos[n][1],triPosCurr[n][0],triPosCurr[n][1]);
#endif
			  
#ifdef CHECK_GS
			  if(CHECK_GS_IND == gals[i].index)
			    {
			      vec2ang(triRays[n].n,&theta,&phi);
			      fprintf(fp,"%.20le %.20le %.20le %.20le ",theta,phi,triPosCurr[n][0],triPosCurr[n][1]);
			      if(n == 2)
				fprintf(fp,"\n");
			    }
#endif
			}
		      
		      //this catches the case if we do not have all the rays in the triangle
		      if(fnd == 0)
			continue;
		      
#ifdef CHECK_GS
		      if(CHECK_GS_IND == gals[i].index)
			fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, found all verts\n",
				ThisTask,i,NumGals,j,NumNNbrs,k,Ntri);
#endif      
		      
		      //test if the triangle is oriented properly - it needs to be in counter-clockwise order
		      // if area is zero, continue
		      area = trisarea(triPosCurr[0],triPosCurr[1],triPosCurr[2]);
		      
#ifdef CHECK_GS
		      if(CHECK_GS_IND == gals[i].index)
			fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, area = %e\n",
				ThisTask,i,NumGals,j,NumNNbrs,k,Ntri,area);
#endif      
		      
		      if(area == 0.0) //triangle has zero area - cannot get barycoords and do interpolation so skip
			{
#ifdef CHECK_GS
			  if(CHECK_GS_IND == gals[i].index)
			    fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, area = 0.0\n",
				    ThisTask,i,NumGals,j,NumNNbrs,k,Ntri);
#endif
			  continue;
			}
		      else if(area < 0)  //in clockwise order, swap last two rays to get to counter-clockwise order - needed for interpolation below
			{
			  tmpRay = triRays[2];
			  triRays[2] = triRays[1];
			  triRays[1] = tmpRay;
			  
			  x = triPos[2][0];
			  y = triPos[2][1];
			  triPos[2][0] = triPos[1][0];
			  triPos[2][1] = triPos[1][1];
			  triPos[1][0] = x;
			  triPos[1][1] = y;
			  
			  x = triPosCurr[2][0];
			  y = triPosCurr[2][1];
			  triPosCurr[2][0] = triPosCurr[1][0];
			  triPosCurr[2][1] = triPosCurr[1][1];
			  triPosCurr[1][0] = x;
			  triPosCurr[1][1] = y;
			  
			  x = cosangCurr[2];
			  cosangCurr[2] = cosangCurr[1];
			  cosangCurr[1] = x;
			}
		      
		      //make sure in counter-clockwise order
		      if(!(trisarea(triPosCurr[0],triPosCurr[1],triPosCurr[2]) > 0))
			{
			  fprintf(stderr,"%05d: gal %ld of %ld, index = %ld, area = %lg\n",
				  ThisTask,i,NumGals,gals[i].index,area);
			  assert(trisarea(triPosCurr[0],triPosCurr[1],triPosCurr[2]) > 0);
			}
		      
#ifdef CHECK_GS
		      if(CHECK_GS_IND == gals[i].index)
			{
			  inval = tritest_getbarycoords(triPosCurr[0],triPosCurr[1],triPosCurr[2],galPos,bcs);
			  fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, testing triangle, bcs = %lg|%lg|%lg, interp = %ld\n",
				  ThisTask,i,NumGals,j,NumNNbrs,k,Ntri,bcs[0],bcs[1],bcs[2],inval);
			  
			}
#endif        
		      
		      //now see if the galaxy is in the triangle - if it is, then record the image
		      if(tritest_getbarycoords(triPosCurr[0],triPosCurr[1],triPosCurr[2],galPos,bcs))
			{
			  //finally found an image! woot woot!
			  
			  //get final barycoords
			  bcs[0] *= cosangCurr[0];
			  bcs[1] *= cosangCurr[1];
			  bcs[2] *= cosangCurr[2];
			  
			  //get gal pos
			  x = triPos[0][0]*bcs[0];
			  y = triPos[0][1]*bcs[0];
			  for(n=1;n<3;++n)
			    {
			      x += triPos[n][0]*bcs[n];
			      y += triPos[n][1]*bcs[n];
			    }
			  ivec[0] = vec[0] + x*tvec[0] + y*pvec[0];
			  ivec[1] = vec[1] + x*tvec[1] + y*pvec[1];
			  ivec[2] = vec[2] + x*tvec[2] + y*pvec[2];
			  
			  //get gal shear matrix
			  /// NOT using this code
			  //ttens_interp[0][0] = 0.0;
			  //ttens_interp[0][1] = 0.0;
			  //ttens_interp[1][0] = 0.0;
			  //ttens_interp[1][1] = 0.0;
			  //for(n=0;n<3;++n)
			  //{
			      //para trans to image spot
			  //  rttens[0][0] = triRays[n].A[2*0+0];
			  //  rttens[0][1] = triRays[n].A[2*0+1];
			  //  rttens[1][0] = triRays[n].A[2*1+0];
			  //  rttens[1][1] = triRays[n].A[2*1+1];
			  //  nest2vec(triRays[n].nest,nvec,rayTraceData.rayOrder);
			  //  paratrans_tangtensor(rttens,triRays[n].n,nvec,ttens);

			      //para trans to galaxy
			  //  paratrans_tangtensor(ttens,nvec,ivec,rttens);

			      //interp
			  //  ttens_interp[0][0] += rttens[0][0]*bcs[n]/cosangCurr[n];
			  //  ttens_interp[0][1] += rttens[0][1]*bcs[n]/cosangCurr[n];
			  //  ttens_interp[1][0] += rttens[1][0]*bcs[n]/cosangCurr[n];
			  //  ttens_interp[1][1] += rttens[1][1]*bcs[n]/cosangCurr[n];
			  //}
			  fnd = interp_invmagmat_to_point(ivec,galRad,wpm1,wpm2,ttens_interp);
			  if(fnd == 1)
			    {
			      //rotate gal to ra dec coords
			      vec2radec(ivec,&ra,&dec);
			      Aradec[0][0] = ttens_interp[1][1];
			      Aradec[0][1] = -ttens_interp[1][0];
			      Aradec[1][0] = -ttens_interp[0][1];
			      Aradec[1][1] = ttens_interp[0][0];
			      
			      //add image to list
			      ImageGalsGlobal[NumImageGalsGlobal].index = gals[i].index;
			      ImageGalsGlobal[NumImageGalsGlobal].ra = ra;
			      ImageGalsGlobal[NumImageGalsGlobal].dec = dec;
			      ImageGalsGlobal[NumImageGalsGlobal].A00 = Aradec[0][0];
			      ImageGalsGlobal[NumImageGalsGlobal].A01 = Aradec[0][1];
			      ImageGalsGlobal[NumImageGalsGlobal].A10 = Aradec[1][0];
			      ImageGalsGlobal[NumImageGalsGlobal].A11 = Aradec[1][1];
			      ++NumImageGalsGlobal;
			      
#ifdef CHECK_GS
			      if(CHECK_GS_IND == gals[i].index)
				fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, found an image! vec = %lg|%lg|%lg, ivec = %lg|%lg|%lg\n",
					ThisTask,i,NumGals,j,NumNNbrs,k,Ntri,
					vec[0],vec[1],vec[2],ivec[0],ivec[1],ivec[2]);
#endif      
			      
			      if(NumImageGalsGlobal >= NumImageGalsGlobalAlloc)
				{
				  NumImageGalsGlobalAlloc += 10;
				  tmpImageGal = (ImageGal*)realloc(ImageGalsGlobal,sizeof(ImageGal)*NumImageGalsGlobalAlloc);
				  assert(tmpImageGal != NULL);
				  ImageGalsGlobal = tmpImageGal;
				}
			    }
#ifdef CHECK_GS
			  else if(fnd == -1)
			    {
			      if(CHECK_GS_IND == gals[i].index)
				fprintf(stderr,"%05d: gal %ld of %ld, nbr %ld of %ld, tri %ld of %ld, did not find wgt ray %ld\n",
					ThisTask,i,NumGals,j,NumNNbrs,k,Ntri,n);
			      
			      fprintf(stderr,"%d: ray buffer regions for grid search are not large enough\n",ThisTask);
			      MPI_Abort(MPI_COMM_WORLD,999);
			    }
#endif			  
			}//if(tritest_getbarycoords(triPosCurr[0],triPosCurr[1],triPosCurr[2],galPos,bcs))
		      
#ifdef CHECK_GS
		      if(CHECK_GS_IND == gals[i].index)
			fprintf(stderr,"\n");
#endif
		    }//for(k=0;k<Ntri;++k)
		}//for(j=0;j<NumNNbrs;++j)
	      
	      timeBCSTestInterp += MPI_Wtime();
	    }//for(tdInd=0;tdInd<2;++tdInd)
	  
#ifdef CHECK_GS
	  if(CHECK_GS_IND == gals[i].index)
	    fclose(fp);
#endif
	}//for(i=0;i<NumGals;++i)
      
      if(NumImageGalsGlobal < NumImageGalsGlobalAlloc && NumImageGalsGlobal > 0)
	{
	  tmpImageGal = (ImageGal*)realloc(ImageGalsGlobal,sizeof(ImageGal)*NumImageGalsGlobal);
	  assert(tmpImageGal != NULL);
	  ImageGalsGlobal = tmpImageGal;
	}
      else if(NumImageGalsGlobal == 0)
	{
	  free(ImageGalsGlobal);
	  ImageGalsGlobal = NULL;
	}
      
      if(maxNumNNbrs > 0)
	free(NNbrs);
      destroyHEALPixTree(tdvec[0]);
      destroyHEALPixTree(tdvec[1]);
    }
  else
    {
      NumImageGalsGlobal = 0;
      ImageGalsGlobal = NULL;
    }

  if(ThisTask == 0)
    fprintf(stderr,"tree build+search took %lg seconds.\ngalaxy image interp took %lg seconds.\n",
	    timeTreeBuild+timeTreeSearch,timeBCSTestInterp);
  
  /*  //print out some profiling info
      double minTime,maxTime,totTime,avgTime;
      MPI_Reduce(&timeTreeBuild,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
      MPI_Reduce(&timeTreeBuild,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Reduce(&timeTreeBuild,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      avgTime = totTime/((double) NTasks);
      #ifdef DEBUG
      if(ThisTask == 0)
      fprintf(stderr,"tree build time max,min,avg = %lf|%lf|%lf - %.2f (total time %lf)\n",
      maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,totTime);
      #endif
      
      MPI_Reduce(&timeTreeSearch,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
      MPI_Reduce(&timeTreeSearch,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Reduce(&timeTreeSearch,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      avgTime = totTime/((double) NTasks);
      #ifdef DEBUG
      if(ThisTask == 0)
      fprintf(stderr,"tree search time max,min,avg = %lf|%lf|%lf - %.2f (total time %lf)\n",
      maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,totTime);
      #endif
      
      MPI_Reduce(&timeBCSTestInterp,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
      MPI_Reduce(&timeBCSTestInterp,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Reduce(&timeBCSTestInterp,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      avgTime = totTime/((double) NTasks);
      #ifdef DEBUG
      if(ThisTask == 0)
      fprintf(stderr,"BCs test interp time max,min,avg = %lf|%lf|%lf - %.2f (total time %lf)\n",
      maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,totTime);
      #endif
  */
}

#undef CHECK_GS
#undef CHECK_GS_IND

//FIXME - need to test routine to distribute gals to tasks
static SourceGal *distribute_gals_to_nodes(long NumGalsForThisPlane, long start, long *NumGals)
{
  MPI_Status Stat;
  SourceGal *galsForThisPlane,*buffGals,*tmpSourceGal;
  long NumGalsAlloc,NumBuffGals;
  long i,Nsend,Nrecv,nest,NumGalsToSend,TotNumGalsToSend,round;
  double vec[3];
  
  long log2NTasks;
  long level,sendTask,recvTask;
    
  if(ThisTask == 0)
    fprintf(stderr,"sending gals to correct tasks.\n");
      
  log2NTasks = 0;
  while(NTasks > (1 << log2NTasks))
    ++log2NTasks;
  
  NumGalsAlloc = NumGalsForThisPlane;
  if(NumGalsAlloc == 0)
    NumGalsAlloc = 10000;
  galsForThisPlane = (SourceGal*)malloc(sizeof(SourceGal)*NumGalsAlloc);
  assert(galsForThisPlane != NULL);
  *NumGals = 0;
  
  NumBuffGals = 50.0*1024l*1024l/sizeof(SourceGal);
  buffGals = (SourceGal*)malloc(sizeof(SourceGal)*NumBuffGals);
  assert(buffGals != NULL);
  
  //get total # of gals to send - move gals to keep to other vector
  NumGalsToSend = 0;
  for(i=0;i<NumGalsForThisPlane;++i)
    {
      vec[0] = SourceGalsGlobal[start+i].pos[0];
      vec[1] = SourceGalsGlobal[start+i].pos[1];
      vec[2] = SourceGalsGlobal[start+i].pos[2];
      nest = vec2nest(vec,rayTraceData.bundleOrder);
      
      if(!(ISSETBITFLAG(bundleCells[nest].active,PRIMARY_BUNDLECELL)))
	++NumGalsToSend;
      else
	{
	  if((*NumGals) == NumGalsAlloc)
	    {
	      if(NumGalsForThisPlane*0.1 < 10000)
		NumGalsAlloc += 10000;
	      else
		NumGalsAlloc += NumGalsForThisPlane*0.1;
	      		
	      tmpSourceGal = (SourceGal*)realloc(galsForThisPlane,sizeof(SourceGal)*NumGalsAlloc);
	      assert(tmpSourceGal != NULL);
	      galsForThisPlane = tmpSourceGal;
	    }
	  
	  galsForThisPlane[*NumGals] = SourceGalsGlobal[start+i];
	  ++(*NumGals);
	  
	  SourceGalsGlobal[start+i] = SourceGalsGlobal[start+NumGalsForThisPlane-1];
	  --i;
	  --NumGalsForThisPlane;
	}
    }
  MPI_Allreduce(&NumGalsToSend,&TotNumGalsToSend,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    
  //realloc global gals vector down to size
  if((*NumGals) > 0)
    {
      NumSourceGalsGlobal -= (*NumGals);
      if(NumSourceGalsGlobal > 0)
        {
          tmpSourceGal = (SourceGal*)realloc(SourceGalsGlobal,sizeof(SourceGal)*NumSourceGalsGlobal);
          assert(tmpSourceGal != NULL);
          SourceGalsGlobal = tmpSourceGal;
        }
      else
        {
          free(SourceGalsGlobal);
          SourceGalsGlobal = NULL;
          NumSourceGalsGlobal = 0;
        }
    }
  
  //now exchange gals with other tasks tasks
  round = 0;
  while(TotNumGalsToSend > 0)
    {
      if(ThisTask == 0)
	fprintf(stderr,"round %ld has %ld gals left to send.\n",round,TotNumGalsToSend);
      ++round;
      
      /*algorithm to loop through pairs of tasks linearly
        -lifted from Gadget-2 under GPL (http://www.gnu.org/copyleft/gpl.html)
        -see pm_periodic.c from Gadget-2 at http://www.mpa-garching.mpg.de/gadget/
      */
      for(level = 0; level < (1 << log2NTasks); level++) /* note: for level=0, target is the same task */
	{
	  sendTask = ThisTask;
	  recvTask = ThisTask ^ level;
	  if(recvTask < NTasks && sendTask != recvTask)
	    {
	      Nsend = 0;
	      for(i=0;i<NumGalsForThisPlane;++i)
		{
		  vec[0] = SourceGalsGlobal[start+i].pos[0];
		  vec[1] = SourceGalsGlobal[start+i].pos[1];
		  vec[2] = SourceGalsGlobal[start+i].pos[2];
		  nest = vec2nest(vec,rayTraceData.bundleOrder);

		  if(firstRestrictedPeanoIndTasks[recvTask] <= bundleCellsNest2RestrictedPeanoInd[nest] &&
		     bundleCellsNest2RestrictedPeanoInd[nest] <= lastRestrictedPeanoIndTasks[recvTask])
		    {
		      buffGals[Nsend] = SourceGalsGlobal[start+i];
		      ++Nsend;
		      --NumGalsToSend;
		      
		      SourceGalsGlobal[start+i] = SourceGalsGlobal[start+NumGalsForThisPlane-1];
		      --i;
		      --NumGalsForThisPlane;
		    }
		  
		  if(Nsend == NumBuffGals)
		    break;
		}
	      
	      if(Nsend > 0)
		{
		  NumSourceGalsGlobal -= Nsend;
		  if(NumSourceGalsGlobal > 0)
		    {
		      tmpSourceGal = (SourceGal*)realloc(SourceGalsGlobal,sizeof(SourceGal)*NumSourceGalsGlobal);
		      assert(tmpSourceGal != NULL);
		      SourceGalsGlobal = tmpSourceGal;
		    }
		  else
		    {
		      free(SourceGalsGlobal);
		      SourceGalsGlobal = NULL;
		      NumSourceGalsGlobal = 0;
		    }
 		}
	      
	      MPI_Sendrecv(&Nsend,1,MPI_LONG,(int) recvTask,TAG_NUMBUFF_GALSDIST,
			   &Nrecv,1,MPI_LONG,(int) recvTask,TAG_NUMBUFF_GALSDIST,
			   MPI_COMM_WORLD,&Stat);
	      
	      if(Nsend > 0 || Nrecv > 0)
		{
		  if(Nrecv > 0)
		    {
		      if((*NumGals)+Nrecv >= NumGalsAlloc)
			{
			  NumGalsAlloc += Nrecv;
			  tmpSourceGal = (SourceGal*)realloc(galsForThisPlane,sizeof(SourceGal)*NumGalsAlloc);
			  assert(tmpSourceGal != NULL);
			  galsForThisPlane = tmpSourceGal;
			}
		    }
		  
		  MPI_Sendrecv(buffGals,(int) (sizeof(SourceGal)*Nsend),MPI_BYTE,(int) recvTask,TAG_BUFF_GALSDIST,
			       &(galsForThisPlane[*NumGals]),(int) (sizeof(SourceGal)*Nrecv),MPI_BYTE,(int) recvTask,TAG_BUFF_GALSDIST,
			       MPI_COMM_WORLD,&Stat);
		  
		  (*NumGals) = (*NumGals) + Nrecv;
		}
	    }
	}
      
      MPI_Allreduce(&NumGalsToSend,&TotNumGalsToSend,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    }
    
  //free some mem
  free(buffGals);
  
  if((*NumGals) > 0)
    {
      tmpSourceGal = (SourceGal*)realloc(galsForThisPlane,sizeof(SourceGal)*(*NumGals));
      assert(tmpSourceGal != NULL);
      galsForThisPlane = tmpSourceGal;
    }
  else
    {
      free(galsForThisPlane);
      galsForThisPlane = NULL;
    }
  
  //realloc gals vector down to size
  if(NumGalsForThisPlane > 0)
    {
      fprintf(stderr,"%d: %ld gals did not find their proper task!\n",ThisTask,NumGalsForThisPlane);
      MPI_Abort(MPI_COMM_WORLD,999);
    }
  
  return galsForThisPlane;
}

static void rayprop_gridsearch(HEALPixRay *ray, double wp, double wpm1, double wpm2)
{
#ifndef BORNAPPRX
  double np[3];
  double lambda;
  double q,qa,qc,qb;
  double ttensor[2][2];
  double rttensor[2][2];
#endif

  long n,m;
  double Ap[4];
  
#ifdef BORNAPPRX
  //change pos
  ray->n[0] = ray->n[0]/wpm1*wp;
  ray->n[1] = ray->n[1]/wpm1*wp;
  ray->n[2] = ray->n[2]/wpm1*wp;
  
  //shift inv mag mat
  for(n=0;n<2;++n)
    for(m=0;m<2;++m)
          Ap[m + 2*n] =
            (1.0 - wpm1*(wp - wpm2)/wp/(wpm1-wpm2))*ray->Aprev[m + 2*n]
            + (wpm1*(wp - wpm2)/wp/(wpm1-wpm2))*ray->A[m + 2*n];
  //- ((wp-wpm1)/wp)*(ray->U[m + 2*n]); U is zero for this routine
  
  ray->A[0] = Ap[0];
  ray->A[1] = Ap[1];
  ray->A[2] = Ap[2];
  ray->A[3] = Ap[3];
#else
  //get lambda
  //take solution which moves ray less distance 
  qa = 1.0;
  qb = 2.0*(ray->n[0]*ray->beta[0] + ray->n[1]*ray->beta[1] + ray->n[2]*ray->beta[2]);
  qc = wpm1*wpm1 - wp*wp;
  q = -0.5*(qb + qb/fabs(qb)*sqrt(qb*qb - 4.0*qa*qc));
  lambda = qc/q;
  if(fabs(lambda) > fabs(q/qa))
    lambda = q/qa;
  
  //get new ray loc
  np[0] = ray->n[0] + ray->beta[0]*lambda;
  np[1] = ray->n[1] + ray->beta[1]*lambda;
  np[2] = ray->n[2] + ray->beta[2]*lambda;
  
  for(n=0;n<2;++n)
    for(m=0;m<2;++m)
          Ap[m + 2*n] =
            (1.0 - wpm1*(wp - wpm2)/wp/(wpm1-wpm2))*ray->Aprev[m + 2*n]
            + (wpm1*(wp - wpm2)/wp/(wpm1-wpm2))*ray->A[m + 2*n];
  //- ((wp-wpm1)/wp)*(ray->U[0 + 2*n]*ray->A[m + 2*0] + ray->U[1 + 2*n]*ray->A[m + 2*1]); U is zero for this routine
      
  //shift ray info
  ray->A[0] = Ap[0];
  ray->A[1] = Ap[1];
  ray->A[2] = Ap[2];
  ray->A[3] = Ap[3];
  
  ttensor[0][0] = ray->A[2*0+0];
  ttensor[0][1] = ray->A[2*0+1];
  ttensor[1][0] = ray->A[2*1+0];
  ttensor[1][1] = ray->A[2*1+1];
  paratrans_tangtensor(ttensor,ray->n,np,rttensor);
  ray->A[2*0+0] = rttensor[0][0];
  ray->A[2*0+1] = rttensor[0][1];
  ray->A[2*1+0] = rttensor[1][0];
  ray->A[2*1+1] = rttensor[1][1];
      
  //shift pos and dir info
  ray->n[0] = np[0];
  ray->n[1] = np[1];
  ray->n[2] = np[2];
#endif
}

/* formula for barycoords from NR in C 3rd Edition pg 1116
   returns true if point is inside the triangle - also all barycoords are positive in this case
   a,b,c verticies need to be in counter-clockwise order
*/
static int tritest_getbarycoords(double a[2], double b[2], double c[2], double q[2], double barycoords[3])
{
  double ap[2],bp[2],qp[2],denom;
  
  ap[0] = a[0] - c[0];
  ap[1] = a[1] - c[1];
  
  bp[0] = b[0] - c[0];
  bp[1] = b[1] - c[1];
  
  qp[0] = q[0] - c[0];
  qp[1] = q[1] - c[1];
  
  denom = (ap[0]*ap[0] + ap[1]*ap[1])*(bp[0]*bp[0] + bp[1]*bp[1]) - (ap[0]*bp[0] + ap[1]*bp[1])*(ap[0]*bp[0] + ap[1]*bp[1]);
  
  barycoords[0] = ( (bp[0]*bp[0] + bp[1]*bp[1])*(ap[0]*qp[0] + ap[1]*qp[1]) - (ap[0]*bp[0] + ap[1]*bp[1])*(bp[0]*qp[0] + bp[1]*qp[1])
                    )/denom;
  barycoords[1] = ( (ap[0]*ap[0] + ap[1]*ap[1])*(bp[0]*qp[0] + bp[1]*qp[1]) - (ap[0]*bp[0] + ap[1]*bp[1])*(ap[0]*qp[0] + ap[1]*qp[1])
                    )/denom;
  barycoords[2] = 1.0 - barycoords[0] - barycoords[1];

  if(barycoords[0] > 0.0 && barycoords[1] > 0.0 && barycoords[2] > 0.0)
    return 1;
  else
    return 0;
}

/* formula for signed triangle area NR in C 3rd Edition pg 1111
   returns positive area for verticies in counter-clockwise order, negative area otherwise
*/
static double trisarea(double a[2], double b[2], double c[2])
{
  return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]))/2.0;
}

//util data structs and functions for getting buffer rays
static int compLong(const void *a, const void *b)
{
  if((*((const long*)a)) > (*((const long*)b)))
    return 1;
  else if((*((const long*)a)) < (*((const long*)b)))
    return -1;
  else
    return 0;
}

struct nctg {
  long nest;
  long task;
};

static int compNCTGTask(const void *a, const void *b)
{
  if(((const struct nctg*)a)->task > ((const struct nctg*)b)->task)
    return 1;
  else if(((const struct nctg*)a)->task < ((const struct nctg*)b)->task)
    return -1;
  else
    return 0;
}

static int compNCTGNest(const void *a, const void *b)
{
  if(((const struct nctg*)a)->nest > ((const struct nctg*)b)->nest)
    return 1;
  else if(((const struct nctg*)a)->nest < ((const struct nctg*)b)->nest)
    return -1;
  else
    return 0;
}

//define this macro to debug the ray buffer exchange code
//#define DEBUG_RAYBUFF

#ifdef DEBUG_RAYBUFF
static int printThisTask(int tnum)
{
  if(tnum == 76 || tnum == 48)
    return 1;
  else
    return 0;
}
#endif

static HEALPixRay *get_buffer_rays(long *NumBufferRays)
{
  long i,j,order;
  long bundleRayShift,NumRaysPerBundleCell;
  long log2NTasks;
  long level,sendTask,recvTask;
  long Nsend,Nrecv;
  long rpInd,bundleNestIndToRecv;
  long NumBufferCells;
  struct nctg *nestCellsToGet, *nestCellsToSend=NULL, *tmpNCTG;
  long NnestCellsToSend;
  long firstNestCellForRecvTask,NnestCellsForRecvTask;
  MPI_Status Stat;
  int count,NnestCellsToSendAlloc=0;
  MPI_Request requestSend,requestRecv;
  int didSend,didRecv;
  HEALPixRay *bufferRays,*bundleRays,*tmpRay;
  long NumBufferRaysUsed;
  long MaxNumBufferRays;
  double runTime;
  
  long rayBuffOrder;
  long bundleRayBuffShift,rayRayBuffShift;
  long *rayBuffCells = NULL,*tmpLong;
  long NumRayBuffCells,NumRayBuffCellsAlloc=0;
  long NumRayBuffCellsPerBundleCell;
  long nest,k,bnest;
  long *listpix = NULL;
  long NlistpixMax = 0;
  long Nlistpix;
  double theta,phi;
  double mapRad;
  long key,*fnd;
  
  runTime = 0.0;
  runTime -= MPI_Wtime();
  
  //prep
  order = rayTraceData.rayOrder;
  log2NTasks = 0;
  while(NTasks > (1 << log2NTasks))
    ++log2NTasks;
  bundleRayShift = 2*(order - rayTraceData.bundleOrder);
  NumRaysPerBundleCell = 1;
  NumRaysPerBundleCell = (NumRaysPerBundleCell << bundleRayShift);
  
  for(i=0;i<=HEALPIX_UTILS_MAXORDER;++i)
    if(sqrt(4.0*M_PI/order2npix(i)) < RAYBUFF_RADIUS_ARCMIN/60.0/180.0*M_PI/3.0)
      break;
  rayBuffOrder = i;
  if(rayBuffOrder > HEALPIX_UTILS_MAXORDER)
    rayBuffOrder = HEALPIX_UTILS_MAXORDER;
  if(rayBuffOrder > rayTraceData.rayOrder)
    rayBuffOrder = rayTraceData.rayOrder;
  if(rayBuffOrder < rayTraceData.bundleOrder)
    rayBuffOrder = rayTraceData.bundleOrder;
  
  bundleRayBuffShift = 2*(rayBuffOrder - rayTraceData.bundleOrder);
  rayRayBuffShift = 2*(rayTraceData.rayOrder - rayBuffOrder);
  NumRayBuffCellsPerBundleCell = 1 << bundleRayBuffShift;
  
  //make list of ray buff cells needed
  NumRayBuffCellsAlloc = 100;
  rayBuffCells = (long*)malloc(sizeof(long)*NumRayBuffCellsAlloc);
  assert(rayBuffCells != NULL);
  NumRayBuffCells = 0;
  mapRad = sqrt(4.0*M_PI/order2npix(rayBuffOrder)) + RAYBUFF_RADIUS_ARCMIN/60.0/180.0*M_PI;
  for(i=0;i<NbundleCells;++i)
    {
      //clear the ray buffer flags - will be set below to only cells that have buffer rays
      CLEARBITFLAG(bundleCells[i].active,RAYBUFF_BUNDLECELL);
      
      //search around each primary cell for potential buffer cells
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	{
	  nest = i << bundleRayBuffShift;
	  for(j=0;j<NumRayBuffCellsPerBundleCell;++j)
	    {
	      nest2ang(j+nest,&theta,&phi,rayBuffOrder);
	      Nlistpix = query_disc_inclusive_nest_fast(theta,phi,mapRad,&listpix,&NlistpixMax,rayBuffOrder);
	      
	      for(k=0;k<Nlistpix;++k)
		{
		  bnest = listpix[k] >> bundleRayBuffShift;
		  
		  if(!(ISSETBITFLAG(bundleCells[bnest].active,PRIMARY_BUNDLECELL)))
		    {
		      rayBuffCells[NumRayBuffCells] = listpix[k];
		      ++NumRayBuffCells;
		      
		      if(NumRayBuffCells >= NumRayBuffCellsAlloc)
			{
			  NumRayBuffCellsAlloc += 100;
			  tmpLong = (long*)realloc(rayBuffCells,sizeof(long)*NumRayBuffCellsAlloc);
			  assert(tmpLong != NULL);
			  rayBuffCells = tmpLong;
			}
		    }
		}//for(k=0;k<Nlistpix;++k)
	    }//for(j=0;j<NumRayBuffCellsPerBundleCell;++j)
	}
    }
  if(NlistpixMax > 0)
    free(listpix);
  NumRayBuffCellsAlloc = NumRayBuffCells;
  tmpLong = (long*)realloc(rayBuffCells,sizeof(long)*NumRayBuffCellsAlloc);
  assert(tmpLong != NULL);
  rayBuffCells = tmpLong;
  
  //get unique ray buff cells
  gsl_sort_long(rayBuffCells,(size_t) 1,(size_t) NumRayBuffCells);
  j = 1;
  for(i=1;i<NumRayBuffCells;++i)
    if(rayBuffCells[i] != rayBuffCells[j-1])
      {
	rayBuffCells[j] = rayBuffCells[i];
	++j;
      }
  NumRayBuffCells = j;
  NumRayBuffCellsAlloc = NumRayBuffCells;
  tmpLong = (long*)realloc(rayBuffCells,sizeof(long)*NumRayBuffCellsAlloc);
  assert(tmpLong != NULL);
  rayBuffCells = tmpLong;
  
#ifdef DEBUG_RAYBUFF
  if(printThisTask(ThisTask) || 1)
    fprintf(stderr,"%04d: found %ld unique ray buffer cells.\n",ThisTask,NumRayBuffCells);
#endif

  //now get set of bundle nest cells on other tasks which have these ray buff cells
  NumBufferCells = NumRayBuffCells;
  nestCellsToGet = (struct nctg*)malloc(sizeof(struct nctg)*NumBufferCells);
  assert(nestCellsToGet != NULL);
  NumBufferCells = 0;
  for(i=0;i<NumRayBuffCells;++i)
    {
      bnest = rayBuffCells[i] >> bundleRayBuffShift;
      
      if(!(ISSETBITFLAG(bundleCells[bnest].active,PRIMARY_BUNDLECELL)))
	{
	  nestCellsToGet[NumBufferCells].nest = bnest;
	  rpInd = bundleCellsNest2RestrictedPeanoInd[bnest];
	  nestCellsToGet[NumBufferCells].task = -1;
	  if(rpInd != -1)
	    {
	      for(j=0;j<NTasks;++j)
		{
		  if(firstRestrictedPeanoIndTasks[j] <= rpInd && rpInd <= lastRestrictedPeanoIndTasks[j])
		    {
		      nestCellsToGet[NumBufferCells].task = j;
		      break;
		    }
		}
	    
	      if(nestCellsToGet[NumBufferCells].task == -1)
		{
		  fprintf(stderr,"%d: could not find task for rpInd = %ld\n",ThisTask,rpInd);
		  MPI_Abort(MPI_COMM_WORLD,888);
		}
	      
	      ++NumBufferCells;
	    }
	}
    }
  
  //get unique set of bundle cells
  qsort(nestCellsToGet,(size_t) NumBufferCells,sizeof(struct nctg),compNCTGNest);
  j = 1;
  for(i=1;i<NumBufferCells;++i)
    if(nestCellsToGet[i].nest != nestCellsToGet[j-1].nest)
      {
	nestCellsToGet[j] = nestCellsToGet[i];
	++j;
      }
  NumBufferCells = j;
  tmpNCTG = (struct nctg*)realloc(nestCellsToGet,sizeof(struct nctg)*NumBufferCells);
  assert(tmpNCTG != NULL);
  nestCellsToGet = tmpNCTG;
  
#ifdef DEBUG_RAYBUFF
  if(printThisTask(ThisTask) || 1)
    fprintf(stderr,"%04d: found %ld unique bundle cells for buffer rays.\n",ThisTask,NumBufferCells);
#endif

  //sort nest inds to get by task
  qsort(nestCellsToGet,(size_t) NumBufferCells,sizeof(struct nctg),compNCTGTask);
  
  //alloc memory
  MaxNumBufferRays = ((double) (NumRayBuffCells))/NumRayBuffCellsPerBundleCell*NumRaysPerBundleCell;
  bufferRays = (HEALPixRay*)malloc(sizeof(HEALPixRay)*MaxNumBufferRays);
  assert(bufferRays != NULL);
  NumBufferRaysUsed = 0;
  bundleRays = (HEALPixRay*)malloc(sizeof(HEALPixRay)*NumRaysPerBundleCell);
  assert(bundleRays != NULL);
  
#ifdef DEBUG_RAYBUFF
  if(printThisTask(ThisTask) || 1)
    fprintf(stderr,"%04d: alloced %lf MB for buffer rays.\n",ThisTask,sizeof(HEALPixRay)*(MaxNumBufferRays+NumRaysPerBundleCell)/1024.0/1024.0);
#endif
  
  /*algorithm to loop through pairs of tasks linearly
    -lifted from Gadget-2 under GPL (http://www.gnu.org/copyleft/gpl.html)
    -see pm_periodic.c from Gadget-2 at http://www.mpa-garching.mpg.de/gadget/
  */
  for(level = 0; level < (1 << log2NTasks); level++) /* note: for level=0, target is the same task */
    {
#ifdef DEBUG_RAYBUFF
      if(ThisTask == 0 && 0)
	fprintf(stderr,"on level %ld of %ld.\n",level+1l,1l << log2NTasks);
#endif
      
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;
      if(recvTask < NTasks && sendTask != recvTask)
        {
	  //first see if we need to get any bundle cells parts from recvTask
          firstNestCellForRecvTask = -1;
          NnestCellsForRecvTask = 0;
          for(j=0;j<NumBufferCells;++j)
            {
              if(nestCellsToGet[j].task == recvTask)
                {
                  if(firstNestCellForRecvTask == -1)
                    firstNestCellForRecvTask = j;

                  ++NnestCellsForRecvTask;
                }
            }

          if(NnestCellsForRecvTask == 0)
            firstNestCellForRecvTask = 0;
	  
          Nsend = NnestCellsForRecvTask;
          MPI_Sendrecv(&Nsend,1,MPI_LONG,(int) recvTask,TAG_NUMNEST_GBR,
                       &Nrecv,1,MPI_LONG,(int) recvTask,TAG_NUMNEST_GBR,
                       MPI_COMM_WORLD,&Stat);
          NnestCellsToSend = Nrecv;
	  
	  if(Nsend > 0 || Nrecv > 0) //there is overlap between tasks so they may need to send stuff                                        
            {
              //get bundle cells for which parts should be sent back                                                                    
              if(NnestCellsToSendAlloc < NnestCellsToSend)
                {
                  tmpNCTG = (struct nctg*)realloc(nestCellsToSend,sizeof(struct nctg)*NnestCellsToSend);
                  if(tmpNCTG != NULL)
                    {
                      nestCellsToSend = tmpNCTG;
                      NnestCellsToSendAlloc = NnestCellsToSend;
                    }
                  else
                    {
                      fprintf(stderr,"%d: could not realloc nestCellsToSend!\n",ThisTask);
                      MPI_Abort(MPI_COMM_WORLD,888);
                    }
                }

              MPI_Sendrecv(nestCellsToGet+firstNestCellForRecvTask,(int) (Nsend*sizeof(struct nctg)),MPI_BYTE,(int) recvTask,TAG_NEST_GBR,
                           nestCellsToSend,(int) (NnestCellsToSend*sizeof(struct nctg)),MPI_BYTE,(int) recvTask,TAG_NEST_GBR,
                           MPI_COMM_WORLD,&Stat);
	      
	      //get total # of rays to send back to recvTask
              Nsend = 0;
              for(i=0;i<NnestCellsToSend;++i)
                if(ISSETBITFLAG(bundleCells[nestCellsToSend[i].nest].active,PRIMARY_BUNDLECELL))
                  Nsend += NumRaysPerBundleCell;
              
              //get total # of rays to recv from recvTask
              MPI_Sendrecv(&Nsend,1,MPI_LONG,(int) recvTask,TAG_NUMBUFF_GBR,
                           &Nrecv,1,MPI_LONG,(int) recvTask,TAG_NUMBUFF_GBR,
                           MPI_COMM_WORLD,&Stat);
	      
	      //if have rays to send or recv, do it
              if(Nsend > 0 || Nrecv > 0)
                {
                  i = 0;
                  while(Nsend > 0 || Nrecv > 0)
                    {
                      if(Nrecv > 0) //recv bundle index for rays
                        {
			  MPI_Irecv(&bundleNestIndToRecv,1,MPI_LONG,(int) recvTask,TAG_NESTIND_GBR,MPI_COMM_WORLD,&requestRecv);
			  didRecv = 1;
			}
		      else
			didRecv = 0;
		      
		      if(Nsend > 0) //send bundle index for rays
                        {
                          while(i < NnestCellsToSend && !(ISSETBITFLAG(bundleCells[nestCellsToSend[i].nest].active,PRIMARY_BUNDLECELL) && bundleCells[nestCellsToSend[i].nest].Nrays > 0))
                            ++i;
			  
			  if(i >= NnestCellsToSend)
                            {
                              fprintf(stderr,"%d: out of nest cells in while Nsend, Nrecv loop\n",ThisTask);
                              MPI_Abort(MPI_COMM_WORLD,888);
                            }

			  MPI_Isend(&(nestCellsToSend[i].nest),1,MPI_LONG,(int) recvTask,TAG_NESTIND_GBR,MPI_COMM_WORLD,&requestSend);
			  didSend = 1;
			}
		      else
			didSend = 0;
		      
		      //make sure both send/recv complete
		      if(didRecv)
			MPI_Wait(&requestRecv,&Stat);
		      
		      if(didSend)
                        MPI_Wait(&requestSend,&Stat);
		      
		      if(Nrecv > 0) //recv rays
			{
			  //error check to make sure cell does not already have rays
			  if(bundleCells[bundleNestIndToRecv].Nrays > 0 || bundleCells[bundleNestIndToRecv].rays != NULL || ISSETBITFLAG(bundleCells[bundleNestIndToRecv].active,PRIMARY_BUNDLECELL))
			    {
			      fprintf(stderr,"%d: bundleCell to recv rays for already has rays! Nrays = %ld\n",
				      ThisTask,bundleCells[bundleNestIndToRecv].Nrays);
			      MPI_Abort(MPI_COMM_WORLD,888);
			    }
			  			  
                          MPI_Irecv(bundleRays,
                                    (int) (sizeof(HEALPixRay)*NumRaysPerBundleCell),MPI_BYTE,
                                    (int) recvTask,TAG_BUFF_GBR,MPI_COMM_WORLD,&requestRecv);
                          didRecv = 1;
                        }
                      else
                        didRecv = 0;
                      
                      if(Nsend > 0) //send rays
                        {
			  MPI_Issend(bundleCells[nestCellsToSend[i].nest].rays,
                                     (int) (sizeof(HEALPixRay)*NumRaysPerBundleCell),MPI_BYTE,
                                     (int) recvTask,TAG_BUFF_GBR,MPI_COMM_WORLD,&requestSend);
                          
                          Nsend -= bundleCells[nestCellsToSend[i].nest].Nrays;
                          didSend = 1;
                          ++i;
                        }
                      else
                        didSend = 0;
                      
		      //make sure both send/recv complete
                      if(didRecv)
                        {
                          MPI_Wait(&requestRecv,&Stat);
                          
                          MPI_Get_count(&Stat,MPI_BYTE,&count);
                          count /= sizeof(HEALPixRay);
			  Nrecv -= count;
			  assert(NumRaysPerBundleCell == count);
			  
			  //keep only rays needed
			  j = 0;
			  for(k=0;k<NumRaysPerBundleCell;++k)
			    {
			      key = bundleRays[k].nest >> rayRayBuffShift;
			      fnd = (long*)bsearch(&key,rayBuffCells,(size_t) NumRayBuffCells,sizeof(long),compLong);
			      
			      if(fnd != NULL) //ray is in a ray Buff cell
				{
				  //add ray to bundle cells
				  bufferRays[NumBufferRaysUsed] = bundleRays[k];
				  ++j;
				  ++NumBufferRaysUsed;
				  
				  //realloc rays if needed
				  if(NumBufferRaysUsed >= MaxNumBufferRays)
				    {
				      MaxNumBufferRays += NumRaysPerBundleCell;
				      tmpRay = (HEALPixRay*)realloc(bufferRays,sizeof(HEALPixRay)*MaxNumBufferRays);
				      assert(tmpRay != NULL);
				      bufferRays = tmpRay;
				    }
				}
			    }
			  bundleCells[bundleNestIndToRecv].Nrays = j;
			  
			  if(bundleCells[bundleNestIndToRecv].Nrays > 0)
			    SETBITFLAG(bundleCells[bundleNestIndToRecv].active,RAYBUFF_BUNDLECELL);
			}
                      
                      if(didSend)
                        MPI_Wait(&requestSend,&Stat);
                    }
		  
		  //error check
		  assert(Nsend == 0);
		  assert(Nrecv == 0);
		  assert(i == NnestCellsToSend);
		
		}//if(Nsend > 0 || Nrecv > 0)
	    }//if(Nsend > 0 || Nrecv > 0)
	}//if(recvTask < NTasks && sendTask != recvTask)
    }//for(level = 0; level < (1 << log2NTasks); level++)
  
  //realloc rays
  MaxNumBufferRays = NumBufferRaysUsed;
  tmpRay = (HEALPixRay*)realloc(bufferRays,sizeof(HEALPixRay)*MaxNumBufferRays);
  assert(tmpRay != NULL);
  bufferRays = tmpRay;
  
  //sort buffer rays by bundle nest
  qsort(bufferRays,(size_t) NumBufferRaysUsed,sizeof(HEALPixRay),compHEALPixRayNest);
  
  //link buffer rays to bundle cells
  j = 0;
  for(i=0;i<NbundleCells;++i)
    if(ISSETBITFLAG(bundleCells[i].active,RAYBUFF_BUNDLECELL))
      {
	//find first ray
	nest = bufferRays[j].nest;
	bnest = nest >> bundleRayShift;
	
	k = 0;
	if(bnest == i)
	  {
	    bundleCells[i].rays = bufferRays + j;
	    ++k;
	    ++j;
	  }
	else
	  {
	    fprintf(stderr,"%d: buffer rays out of order in final linking! ray bnest = %ld, bundle bnest = %ld\n",ThisTask,bnest,i);
	    MPI_Abort(MPI_COMM_WORLD,888);
	  }
	
	//count rest of rays
	while(1)
	  {
	    nest = bufferRays[j].nest;
	    bnest = nest >> bundleRayShift;
	    
	    if(bnest == i)
	      {
		++k;
		++j;
	      }
	    else
	      break;
	  }
	
	bundleCells[i].Nrays = k;
      }
  assert(j == NumBufferRaysUsed);
  
  //clean it all up
  free(bundleRays);
  if(NnestCellsToSendAlloc > 0)
    free(nestCellsToSend);
  free(nestCellsToGet);
  free(rayBuffCells);
  
#ifdef DEBUG_RAYBUFF
  if(printThisTask(ThisTask) || 1)
    fprintf(stderr,"%04d: used %lf MB for buffer rays.\n",ThisTask,sizeof(HEALPixRay)*(MaxNumBufferRays)/1024.0/1024.0);
#endif
  
  runTime += MPI_Wtime();;
  /*
    double mintm,maxtm,avgtm;
    MPI_Reduce(&runTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&runTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&runTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    avgtm = avgtm/NTasks;
    #ifdef DEBUG
    if(ThisTask == 0)
    fprintf(stderr,"get_buffer_rays run time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
    #endif
    
    if(ThisTask == 0)
    fprintf(stderr,"got buffer rays in %lg seconds.\n",runTime);
  */
  
  *NumBufferRays = NumBufferRaysUsed;
  return bufferRays;
}

static void destroy_buffer_rays(HEALPixRay *bufferRays)
{
  long i;
  
  free(bufferRays);
  
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,RAYBUFF_BUNDLECELL))
	{
	  bundleCells[i].Nrays = 0;
	  bundleCells[i].rays = NULL;
	}
    }
}

