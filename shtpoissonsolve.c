#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>

#include "raytrace.h"
#include "healpix_shtrans.h"

#ifdef SHTONLY
static int shearinterp_comp(double rvec[3], double *pot, double alpha[2], double U[4]);
//static int shearinterp_poly(double rvec[3], double *pot, double alpha[2], double U[4]);
#endif

#ifdef DEBUG_IO
static void write_ringmap(char name[], float *mapvec, HEALPixSHTPlan plan);
static void write_localmap(char name[], HEALPixMapCell *localMapCells, long NumLocalMapCells);
#endif

void do_healpix_sht_poisson_solve(double densfact, double backdens)
{
  long i,j,k,n;
  long bundleMapShift,mapNest,bundleNest;
  float *mapvec;
#ifdef SHTONLY
  float *mapvec_gradtheta,*mapvec_gradphi;
  float *mapvec_gradthetatheta,*mapvec_gradthetaphi,*mapvec_gradphiphi;
#endif
  fftwf_complex *mapvec_complex;
  HEALPixSHTPlan plan;
  long Nside,nring,ringpix;
  double vec[3],theta,phi;
  long firstRing,lastRing;
  double *alm_real,*alm_imag;
  long l,m;
  double poissonHEALPixArea;
  double mapbuffrad;
#ifndef USE_FULLSKY_PARTDIST
  double ra,dec;
  long ring;
#endif
  double smoothingRad;
  long *listpix=NULL,Nlistpix=0,Ntotmass;
  double totmass,r,cosdis,nvec[3];
  double *listdens=NULL,*tmp;
  long Nlistdens = 0,NlistpixMax = 0;
  long shift,queryOrder,queryNest,numQueryPixPerGridPix;
  double gs[HEALPIX_UTILS_MAXORDER+1];
#ifdef DEBUG_IO
  char name[MAX_FILENAME];
#endif            
  double alm2mapTime,map2almTime;  
#ifdef CICSHTDENS
  long wgtpix[4];
  double wgt[4];
#endif
  
  FILE *fp;
  char fname[MAX_FILENAME];
  
  logProfileTag(PROFILETAG_SHT);
  
  /* init vars*/
  for(i=0;i<=HEALPIX_UTILS_MAXORDER;++i)
    gs[i] = sqrt(4.0*M_PI/order2npix(i));
  Nside = order2nside(rayTraceData.poissonOrder);
  bundleMapShift = 2*(rayTraceData.poissonOrder - rayTraceData.bundleOrder);
  poissonHEALPixArea = 4.0*M_PI/order2npix(rayTraceData.poissonOrder);
  
  /* basic steps for SHT poisson solve
     0) set the map buffer cells
     1) zero map cells and then record the mass of particles (or add the weights) for each cell in each local map - only sum over particles in bit 0 set cells
     2) then do the peano to ring map shuffle - this step adds the cells so does the global reduction to form correct density map as well
     3) then multiply by the density factor and subtract the background density if needed on the rings
     4) call map2alm_mpi, divide by -l(l+1), then call alm2map_mpi
     5) then do the ring to peano shuffle
  */
  
  //this computes = actual radius needed + max dist a ray moves out of bundleCell
  mapbuffrad = rayTraceData.partBuffRad + rayTraceData.maxSL*2.0;
  mark_bundlecells(mapbuffrad,PRIMARY_BUNDLECELL,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL);
#ifdef USE_FULLSKY_PARTDIST
  if(!rayTraceData.UseHEALPixLensPlaneMaps) 
    {
      mark_bundlecells(mapbuffrad,FULLSKY_PARTDIST_PRIMARY_BUNDLECELL,FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL);
      alloc_mapcells(FULLSKY_PARTDIST_PRIMARY_BUNDLECELL,FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL);
    }
#else
  alloc_mapcells(PRIMARY_BUNDLECELL,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL);
#endif

  /* step 1 - grid parts*/  
  if(!rayTraceData.UseHEALPixLensPlaneMaps) 
    {
      for(i=0;i<NmapCells;++i)
	mapCells[i].val = 0.0;
      
      for(i=0;i<NbundleCells;++i)
	{
	  if(
#ifdef USE_FULLSKY_PARTDIST
	     ISSETBITFLAG(bundleCells[i].active,FULLSKY_PARTDIST_PRIMARY_BUNDLECELL)
#else
	     ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL)
#endif
	     && bundleCells[i].Nparts > 0)
	    {
	      for(k=0;k<bundleCells[i].Nparts;++k)
		{
		  vec[0] = (double) (lensPlaneParts[k+bundleCells[i].firstPart].pos[0]);
		  vec[1] = (double) (lensPlaneParts[k+bundleCells[i].firstPart].pos[1]);
		  vec[2] = (double) (lensPlaneParts[k+bundleCells[i].firstPart].pos[2]);
		  
		  r = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
		  vec2ang(vec,&theta,&phi);
		  
#ifdef NGPSHTDENS
		  //NGP SHT STEP
		  mapNest = ang2nest(theta,phi,rayTraceData.poissonOrder);
		  bundleNest = (mapNest >> bundleMapShift);
		  j = (bundleNest << bundleMapShift);
		  
		  if(
#ifdef USE_FULLSKY_PARTDIST
		     (ISSETBITFLAG(bundleCells[bundleNest].active,FULLSKY_PARTDIST_PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL))
#else
		     (ISSETBITFLAG(bundleCells[bundleNest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL))
#endif
		     && bundleCells[bundleNest].firstMapCell >= 0)
		    {
		      mapCells[bundleCells[bundleNest].firstMapCell+mapNest-j].val +=
			(float) (lensPlaneParts[k+bundleCells[i].firstPart].mass);
		  
		      assert(mapNest == mapCells[bundleCells[bundleNest].firstMapCell+mapNest-j].index);
		    }
		  
		  continue;
#endif
 

#ifdef CICSHTDENS
		  //CIC SHT STEP
		  long wgtpix[4];
		  double wgt[4];
		  get_interpol(theta,phi,wgtpix,wgt,rayTraceData.poissonOrder);
		  for(m=0;m<4;++m)
		    {
		      mapNest = ring2nest(wgtpix[m],rayTraceData.poissonOrder);
		      bundleNest = (mapNest >> bundleMapShift);
		      j = (bundleNest << bundleMapShift);
		      if(
#ifdef USE_FULLSKY_PARTDIST
			 (ISSETBITFLAG(bundleCells[bundleNest].active,FULLSKY_PARTDIST_PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL))
#else
			 (ISSETBITFLAG(bundleCells[bundleNest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL))
#endif
			 && bundleCells[bundleNest].firstMapCell >= 0)
			{
			  mapCells[bundleCells[bundleNest].firstMapCell+mapNest-j].val +=
			    (float) (lensPlaneParts[k+bundleCells[i].firstPart].mass*wgt[m]);
			  
			  assert(mapNest == mapCells[bundleCells[bundleNest].firstMapCell+mapNest-j].index);
			}
		    }

		  continue;
#endif
		  
		  //use smoothing lengths otherwise
		  smoothingRad = lensPlaneParts[k+bundleCells[i].firstPart].smoothingLength;
		  
		  queryOrder = 0;
		  while(gs[queryOrder] > smoothingRad/SMOOTHKERN_SHTRESOLVE_FAC && queryOrder < rayTraceData.poissonOrder)
		    ++queryOrder;
		  
		  shift = 2*(rayTraceData.poissonOrder-queryOrder);
		  numQueryPixPerGridPix = (1ll) << shift;
		  assert(numQueryPixPerGridPix >= 1);
		  
		  Nlistpix = query_disc_inclusive_nest_fast(theta,phi,smoothingRad,&listpix,&NlistpixMax,queryOrder);
		  
		  if(Nlistdens < Nlistpix)
		    {
		      tmp = (double*)realloc(listdens,sizeof(double)*Nlistpix);
		      
		      if(tmp != NULL)
			{
			  listdens = tmp;
			  Nlistdens = Nlistpix;
			}
		      else
			{
			  fprintf(stderr,"%d: could not realloc memory for list dens!\n",ThisTask);
			  MPI_Abort(MPI_COMM_WORLD,123);
			}
		    }
		  
		  totmass = 0.0;
		  Ntotmass = 0;
		  for(n=0;n<Nlistpix;++n)
		    {
		      nest2vec(listpix[n],nvec,queryOrder);
		      cosdis = (vec[0]*nvec[0] + vec[1]*nvec[1] + vec[2]*nvec[2])/r;
		      listdens[n] = spline_part_dens(cosdis,smoothingRad);
		      if(listdens[n] > 0.0)
			{
			  listpix[Ntotmass] = listpix[n];
			  listdens[Ntotmass] = listdens[n];
			  totmass += listdens[Ntotmass];
			  ++Ntotmass;
			}
		    }
		  
		  for(n=0;n<Ntotmass;++n)
		    {
		      queryNest = listpix[n];
		      
		      for(m=0;m<numQueryPixPerGridPix;++m)
			{
			  mapNest = (queryNest << shift) + m;
			  bundleNest = (mapNest >> bundleMapShift);
			  j = (bundleNest << bundleMapShift);
			  
			  if(
#ifdef USE_FULLSKY_PARTDIST
			     (ISSETBITFLAG(bundleCells[bundleNest].active,FULLSKY_PARTDIST_PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL))
#else
			     (ISSETBITFLAG(bundleCells[bundleNest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL))
#endif
			     && bundleCells[bundleNest].firstMapCell >= 0)
			    {
			      mapCells[bundleCells[bundleNest].firstMapCell+mapNest-j].val +=
				(float) (listdens[n]/totmass/numQueryPixPerGridPix*lensPlaneParts[k+bundleCells[i].firstPart].mass);
			      
			      assert(mapNest == mapCells[bundleCells[bundleNest].firstMapCell+mapNest-j].index);
			    }
			  
			  // this error no longer applies since we are reading a different range of particle and map cells to save memory
			  // else
			  // {
			  //this error happens if bit 0 set and bit 1 set cells are no in the cells returned by get_interpol
			  // fprintf(stderr,"%d: map buffer zones for patches are not big enough for kappa dens!\n",ThisTask);
			  // MPI_Abort(MPI_COMM_WORLD,123);
			  // }
			  //
			}
		    }
		  
		  //could be that part is in map, but smoothing rad is too small to find any pixels above.  
		  //if so, this code catches it and puts its mass on grid with NGP
		  if(Ntotmass == 0)
		    {
		      mapNest = vec2nest(vec,rayTraceData.poissonOrder);
		      bundleNest = (mapNest >> bundleMapShift);
		      j = (bundleNest << bundleMapShift);
		      
		      if(
#ifdef USE_FULLSKY_PARTDIST
			 (ISSETBITFLAG(bundleCells[bundleNest].active,FULLSKY_PARTDIST_PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL))
#else
			 (ISSETBITFLAG(bundleCells[bundleNest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL))
#endif
			 && bundleCells[bundleNest].firstMapCell >= 0)
			{
			  mapCells[bundleCells[bundleNest].firstMapCell+mapNest-j].val += (float) (lensPlaneParts[k+bundleCells[i].firstPart].mass);
			  
			  assert(mapNest == mapCells[bundleCells[bundleNest].firstMapCell+mapNest-j].index);
			}
		    }
		  
		}//for(k=0;k<bundleCells[i].Nparts;++k)
	    }
	}//for(i=0;i<NbundleCells;++i)
      
      if(Nlistdens > 0)
	free(listdens);
      
      if(NlistpixMax > 0)
	free(listpix);
      
#ifdef USE_FULLSKY_PARTDIST
      /* free parts since we do not need them anymore */
      destroy_parts();
#endif
  
#ifdef DEBUG_IO
      //print out map cells
      sprintf(name,"%s/localmap%ld.%d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
      write_localmap(name,mapCells,NmapCells);
#endif
    } /* if(!rayTraceData.UseHEALPixLensPlaneMaps) */

#ifdef DEBUG_IO_DD
  write_bundlecells2ascii("step1SHT");
#endif
      
  /* step 2  - get map rings on correct nodes */
  plan = healpixsht_plan(rayTraceData.poissonOrder);
  if(strlen(rayTraceData.HEALPixRingWeightPath) > 0)
    {
      read_ring_weights(rayTraceData.HEALPixRingWeightPath,&plan);
      
      if(ThisTask == 0)
	fprintf(stderr,"using ring weights!\n");
    }
  /*
  if(strlen(rayTraceData.HEALPixWindowFunctionPath) > 0)
    {
      read_window_function(rayTraceData.HEALPixWindowFunctionPath,&plan);
      
      if(ThisTask == 0)
	fprintf(stderr,"using pixel window!\n");
    }
  */
  assert(plan.Nmapvec);
  mapvec = (float*)malloc(sizeof(fftwf_complex)*plan.Nmapvec);
  assert(mapvec != NULL);
  if(!rayTraceData.UseHEALPixLensPlaneMaps) 
    {
      logProfileTag(PROFILETAG_MAPSUFFLE);
      healpixmap_peano2ring_shuffle(mapvec,plan);
      logProfileTag(PROFILETAG_MAPSUFFLE);
    }
  else
    {
      logProfileTag(PROFILETAG_SHT);
      
      logProfileTag(PROFILETAG_PARTIO);
      
      //read maps now if needed
      j = NTasks/rayTraceData.NumFilesIOInParallel;
      if(j*rayTraceData.NumFilesIOInParallel < NTasks)
	++j;
      k = ThisTask/rayTraceData.NumFilesIOInParallel;
      for(i=0;i<j;++i)
	{
	  if(i == k)
	    {
	      firstRing = plan.firstRingTasks[ThisTask];
	      lastRing = plan.lastRingTasks[ThisTask];
	      mapvec_complex = (fftwf_complex*) mapvec;
	      
	      sprintf(fname,"%s/%s.%ld",rayTraceData.HEALPixLensPlaneMapPath,
		      rayTraceData.HEALPixLensPlaneMapName,rayTraceData.CurrentPlaneNum);
	      
	      fp = fopen(fname,"r");
	      assert(fp != NULL);
	      fseek(fp,plan.northStartIndGlobalMap[0]*sizeof(float),SEEK_SET);
	      
	      //read all of the northern rings
	      for(nring=firstRing;nring<=lastRing;++nring)
		{
		  if(nring < Nside)
		    ringpix = 4*nring;
		  else
		    ringpix = 4*Nside;
		  
		  mapvec = (float*) (mapvec_complex+plan.northStartIndMapvec[nring-firstRing]);
		  fread(mapvec,(size_t) ringpix,sizeof(float),fp);
		}
	      
	      //now read southern rings - they are opposite order since HEALPix reflects
	      // over the equator
	      fseek(fp,plan.southStartIndGlobalMap[lastRing-firstRing]*sizeof(float),SEEK_SET);
	      for(nring=lastRing;nring>=firstRing;nring--)
		{
		  if(nring < Nside)
		    ringpix = 4*nring;
		  else
		    ringpix = 4*Nside;
		  
		  //ring on equator doesn't have a reflection
		  if(nring != 2*Nside)
		    {
		      mapvec = (float*) (mapvec_complex+plan.southStartIndMapvec[nring-firstRing]);
		      fread(mapvec,(size_t) ringpix,sizeof(float),fp);
		    }
		}
	      
	      fclose(fp);
	      mapvec = (float*) mapvec_complex;
	    }
	  
	  ////////////////////////////////////
	  MPI_Barrier(MPI_COMM_WORLD);
	  ////////////////////////////////////
	}
      
      logProfileTag(PROFILETAG_PARTIO);
      
      if(ThisTask == 0)
	fprintf(stderr,"read HEALPix lens plane maps in %lf seconds.\n",getTimeProfileTag(PROFILETAG_PARTIO));
      
      logProfileTag(PROFILETAG_SHT);
      
      firstRing = plan.firstRingTasks[ThisTask];
      lastRing = plan.lastRingTasks[ThisTask];
      mapvec_complex = (fftwf_complex*) mapvec;
      for(nring=firstRing;nring<=lastRing;++nring)
	{
	  if(nring < Nside)
	    ringpix = 4*nring;
	  else
	    ringpix = 4*Nside;
	  
	  mapvec = (float*) (mapvec_complex+plan.northStartIndMapvec[nring-firstRing]);
	  for(i=0;i<ringpix;++i)
	    mapvec[i] *= (float) (rayTraceData.partMass);
      
	  if(nring != 2*Nside)
	    {
	      mapvec = (float*) (mapvec_complex+plan.southStartIndMapvec[nring-firstRing]);
	      for(i=0;i<ringpix;++i)
		mapvec[i] *= (float) (rayTraceData.partMass);
	    }
	}
      mapvec = (float*) mapvec_complex;
    }
  
#ifdef DEBUG_IO
  sprintf(name,"%s/ringmap%ld.%d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  write_ringmap(name,mapvec,plan);
#endif
  
#ifdef DEBUG_IO_DD
  write_bundlecells2ascii("step2SHT");
#endif

#ifdef USE_FULLSKY_PARTDIST  
  if(!rayTraceData.UseHEALPixLensPlaneMaps) 
    {
      free_mapcells();
    }
#endif
  
  /* step 3 - multiply by densfact and subtract backdens */
  firstRing = plan.firstRingTasks[ThisTask];
  lastRing = plan.lastRingTasks[ThisTask];
  mapvec_complex = (fftwf_complex*) mapvec;
  for(nring=firstRing;nring<=lastRing;++nring)
    {
      if(nring < Nside)
	ringpix = 4*nring;
      else
	ringpix = 4*Nside;
      
      mapvec = (float*) (mapvec_complex+plan.northStartIndMapvec[nring-firstRing]);
      for(i=0;i<ringpix;++i)
	{
	  mapvec[i] *= (float) (densfact/poissonHEALPixArea);
#ifndef USE_FULLSKY_PARTDIST
	  ring = i + plan.northStartIndGlobalMap[nring-firstRing];
	  ring2ang(ring,&theta,&phi,rayTraceData.poissonOrder);
	  ang2radec(theta,phi,&ra,&dec);
	  if(!(test_vaccell(ra,dec)))
	    mapvec[i] -= (float) backdens;
	  else
	    mapvec[i] = 0.0;
#else
	  mapvec[i] -= (float) backdens;
#endif
	}
      
      if(nring != 2*Nside)
	{
	  mapvec = (float*) (mapvec_complex+plan.southStartIndMapvec[nring-firstRing]);
	  for(i=0;i<ringpix;++i)
	    {
	      mapvec[i] *= (float) (densfact/poissonHEALPixArea);
#ifndef USE_FULLSKY_PARTDIST
	      ring = i + plan.southStartIndGlobalMap[nring-firstRing];
	      ring2ang(ring,&theta,&phi,rayTraceData.poissonOrder);
	      ang2radec(theta,phi,&ra,&dec);
	      if(!(test_vaccell(ra,dec)))
		mapvec[i] -= (float) backdens;
	      else
		mapvec[i] = 0.0;
#else
	      mapvec[i] -= (float) backdens;
#endif
	    }
	}
    }
  mapvec = (float*) mapvec_complex;
  
#ifdef DEBUG_IO_DD
  write_bundlecells2ascii("step3SHT");
#endif
  
  /* step 4 - do SHT poisson solve */
#ifdef DEBUG_IO
  sprintf(name,"%s/shtdens%ld.%d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  write_ringmap(name,mapvec,plan);
#endif
  
  logProfileTag(PROFILETAG_SHT);
  
  logProfileTag(PROFILETAG_SHTSOLVE);
  alm_real = (double*)malloc(sizeof(double)*plan.Nlm);
  assert(alm_real != NULL);
  alm_imag = (double*)malloc(sizeof(double)*plan.Nlm);
  assert(alm_imag != NULL);
  map2almTime = -MPI_Wtime();
  map2alm_mpi(alm_real,alm_imag,mapvec,plan); //map -> alm
  map2almTime += MPI_Wtime();
  if(ThisTask == 0)
    fprintf(stderr,"map -> alm took %lf seocnds.\n",map2almTime);
  i = 0;
  for(m=plan.firstMTasks[ThisTask];m<=plan.lastMTasks[ThisTask];++m)
    for(l=m;l<=plan.lmax;++l)
      {
	if(l == 0 && m == 0)
	  {
	    alm_real[i] = 0.0;
	    alm_imag[i] = 0.0;
	  }
	else
	  {
	    alm_real[i] *= (double) (-1.0/((double) l)/(((double) l)+1.0));
	    alm_imag[i] *= (double) (-1.0/((double) l)/(((double) l)+1.0));
	    
	    /*
	    if(strlen(rayTraceData.HEALPixWindowFunctionPath) > 0)
	      {
		alm_real[i] /= pow(plan.window_function[l],HEALPIX_WINDOWFUNC_POWER);
		alm_imag[i] /= pow(plan.window_function[l],HEALPIX_WINDOWFUNC_POWER);
	      }
	    */
	  }
	
	++i;
      }
  
#ifdef SHTONLY
  //compute phi and derivs
  mapvec_gradtheta = (float*)malloc(sizeof(fftwf_complex)*plan.Nmapvec);
  assert(mapvec_gradtheta != NULL);
  mapvec_gradphi = (float*)malloc(sizeof(fftwf_complex)*plan.Nmapvec);
  assert(mapvec_gradphi != NULL);
  mapvec_gradphiphi = (float*)malloc(sizeof(fftwf_complex)*plan.Nmapvec);
  assert(mapvec_gradphiphi != NULL);
  mapvec_gradthetaphi = (float*)malloc(sizeof(fftwf_complex)*plan.Nmapvec);
  assert(mapvec_gradthetaphi != NULL);
  mapvec_gradthetatheta = (float*)malloc(sizeof(fftwf_complex)*plan.Nmapvec);
  assert(mapvec_gradthetatheta != NULL);

  alm2mapTime = -MPI_Wtime();
  alm2allmaps_mpi(alm_real,alm_imag,mapvec,mapvec_gradtheta,mapvec_gradphi,
		  mapvec_gradthetatheta,mapvec_gradthetaphi,mapvec_gradphiphi,plan);
  alm2mapTime += MPI_Wtime();
  if(ThisTask == 0)
    fprintf(stderr,"alm -> all maps took %lf seocnds.\n",alm2mapTime);
#else
  alm2mapTime = -MPI_Wtime();
  alm2map_mpi(alm_real,alm_imag,mapvec,plan);
  alm2mapTime += MPI_Wtime();
  if(ThisTask == 0)
    fprintf(stderr,"alm -> map took %lf seocnds.\n",alm2mapTime);
#endif
  
  free(alm_real);
  free(alm_imag);

  logProfileTag(PROFILETAG_SHTSOLVE);
  
#ifdef DEBUG_IO
  sprintf(name,"%s/shtlenspot%ld.%d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  write_ringmap(name,mapvec,plan);
  
#ifdef SHTONLY
  sprintf(name,"%s/shtgradtheta%ld.%d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  write_ringmap(name,mapvec_gradtheta,plan);
  
  sprintf(name,"%s/shtgradphi%ld.%d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  write_ringmap(name,mapvec_gradphi,plan);
  
  sprintf(name,"%s/shtgradthetatheta%ld.%d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  write_ringmap(name,mapvec_gradthetatheta,plan);
  
  sprintf(name,"%s/shtgradthetaphi%ld.%d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  write_ringmap(name,mapvec_gradthetaphi,plan);
  
  sprintf(name,"%s/shtgradphiphi%ld.%d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  write_ringmap(name,mapvec_gradphiphi,plan);
#endif
#endif
  
#ifdef DEBUG_IO_DD
  write_bundlecells2ascii("step4SHT");
#endif
  
  /* step 5 - do ring to peano map shuffle */
  logProfileTag(PROFILETAG_MAPSUFFLE);
#ifdef USE_FULLSKY_PARTDIST
  alloc_mapcells(PRIMARY_BUNDLECELL,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL);
#endif
  
#ifdef DEBUG_IO_DD
  write_bundlecells2ascii("step5SHT");
#endif
  
#ifdef SHTONLY
  healpixmap_ring2peano_shuffle(&mapvec_gradtheta,plan);
  mapCellsGradTheta = mapCells;
  
  NmapCells = 0;
  mapCells = NULL;
  alloc_mapcells(PRIMARY_BUNDLECELL,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL);
  healpixmap_ring2peano_shuffle(&mapvec_gradphi,plan);
  mapCellsGradPhi = mapCells;
  
  NmapCells = 0;
  mapCells = NULL;
  alloc_mapcells(PRIMARY_BUNDLECELL,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL);
  healpixmap_ring2peano_shuffle(&mapvec_gradphiphi,plan);
  mapCellsGradPhiPhi = mapCells;
  
  NmapCells = 0;
  mapCells = NULL;
  alloc_mapcells(PRIMARY_BUNDLECELL,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL);
  healpixmap_ring2peano_shuffle(&mapvec_gradthetaphi,plan);
  mapCellsGradThetaPhi = mapCells;
  
  NmapCells = 0;
  mapCells = NULL;
  alloc_mapcells(PRIMARY_BUNDLECELL,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL);
  healpixmap_ring2peano_shuffle(&mapvec_gradthetatheta,plan);
  mapCellsGradThetaTheta = mapCells;
    
  NmapCells = 0;
  mapCells = NULL;
  alloc_mapcells(PRIMARY_BUNDLECELL,NON_FULLSKY_PARTDIST_MAPBUFF_BUNDLECELL);
#endif
  
  healpixmap_ring2peano_shuffle(&mapvec,plan);
  
  logProfileTag(PROFILETAG_MAPSUFFLE);

#ifdef DEBUG_IO
  sprintf(name,"%s/localpot%ld.%d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  write_localmap(name,mapCells,NmapCells);
#endif
  
  logProfileTag(PROFILETAG_SHT);
  
  healpixsht_destroy_plan(plan);
  
#ifdef SHTONLY
  //now set ray defl and shear comps with long range part
  long doNotHaveCell;
  double rvec[3],alpha[2] = {0.0,0.0},U[4] = {0.0,0.0,0.0,0.0},lenspot = 0.0;
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
        {
          for(j=0;j<bundleCells[i].Nrays;++j)
            {
              rvec[0] = bundleCells[i].rays[j].n[0];
              rvec[1] = bundleCells[i].rays[j].n[1];
              rvec[2] = bundleCells[i].rays[j].n[2];
              
	      doNotHaveCell = shearinterp_comp(rvec,&lenspot,alpha,U);
	      //DO NOT USE THIS doNotHaveCell = shearinterp_poly(rvec,&lenspot,alpha,U);
	      
	      if(doNotHaveCell)
		{
		  vec2ang(rvec,&theta,&phi);
		  fprintf(stderr,"%d: buffer region for HEALPix map is not big enough for long range force interp! - theta,phi = %le|%le, nest = %ld, doNotHaveCell = %ld\n",
			  ThisTask,theta,phi,bundleCells[i].rays[j].nest,doNotHaveCell);
		  MPI_Abort(MPI_COMM_WORLD,123);
		}
	      
	      bundleCells[i].rays[j].phi += lenspot;
	      
	      bundleCells[i].rays[j].alpha[0] += -1.0*alpha[0];
              bundleCells[i].rays[j].alpha[1] += -1.0*alpha[1];
	      
              bundleCells[i].rays[j].U[0] += U[0];
              bundleCells[i].rays[j].U[1] += U[1];
              bundleCells[i].rays[j].U[2] += U[2];
              bundleCells[i].rays[j].U[3] += U[3];
            }
        }
    }
  
  free_mapcells();
#endif
  
  logProfileTag(PROFILETAG_SHT);
}

#ifdef SHTONLY

/*
static int compLong(const void *a, const void *b)
{
  if((*((const long*)a)) > (*((const long*)b)))
    return 1;
  else if((*((const long*)a)) < (*((const long*)b)))
    return -1;
  else
    return 0;
}

static long *get_rings_shearinterp_poly(double rvec[3], long Nrings, long *Ninds)
{
  long i,j,start,stop,ring,NindsMax,NindsCurr;
  long *inds,*tmp;
  long pix;
  long nbrs[8];
  
  pix = vec2nest(rvec,rayTraceData.poissonOrder);
  NindsMax = 1;
  for(ring=0;ring<Nrings;++ring)
    NindsMax *= 8;
  NindsMax += 1;
  
  inds = (long*)malloc(sizeof(long)*NindsMax);
  assert(inds != NULL);
  
  //add pix as first index
  NindsCurr = 1;
  inds[0] = pix;
  
  //now build out more nbrs using by get nbrs of nbrs of nbrs of ...
  start = 0;
  stop = NindsCurr-1;
  for(ring=0;ring<Nrings;++ring)
    {
      for(j=start;j<=stop;++j)
	{
	  getneighbors_nest(inds[j],nbrs,rayTraceData.poissonOrder);
	  for(i=0;i<8;++i)
	    if(nbrs[i] != -1)
	      {
		inds[NindsCurr] = nbrs[i];
		++NindsCurr;
		
		if(NindsCurr >= NindsMax)
		  {
		    NindsMax += 64;
		    tmp = (long*)realloc(inds,sizeof(long)*NindsMax);
		    assert(tmp != NULL);
		    inds = tmp;
		  }
	      }
	}
      
      start = stop + 1;
      stop = NindsCurr-1;
    }
  
  //now sort the inds and remove duplicates
  qsort(inds,(size_t) NindsCurr,sizeof(long),compLong);
  i = 1;
  for(j=1;j<NindsCurr;++j)
    {
      if(inds[j] != inds[j-1])
	{
	  inds[i] = inds[j];
	  ++i;
	}
    }
  
  //realloc memory
  NindsMax = i;
  tmp = (long*)realloc(inds,sizeof(long)*NindsMax);
  assert(tmp != NULL);
  inds = tmp;
  
  *Ninds = NindsMax;
  return inds;
}

static void rot_ray_shearinterp_poly(double vec[3], double rvec[3],
				     double alpha[2], double ralpha[2], double U[2][2], double rU[2][2],
				     double axis[3], double cosangle, double sinangle)
{
  double p[3];
  double rephi_vec[3],ephi_rvec[3],etheta_rvec[3];
  double sinpsi,cospsi,norm;
  
  rot_vec_axis_trigangle_countercw(vec,rvec,axis,cosangle,sinangle);
  
  p[0] = -vec[1];
  p[1] = vec[0];
  p[2] = 0.0;
  rot_vec_axis_trigangle_countercw(p,rephi_vec,axis,cosangle,sinangle);
  
  ephi_rvec[0] = -rvec[1];
  ephi_rvec[1] = rvec[0];
  ephi_rvec[2] = 0.0;
  
  etheta_rvec[0] = rvec[2]*rvec[0];
  etheta_rvec[1] = rvec[2]*rvec[1];
  etheta_rvec[2] = -1.0*(rvec[0]*rvec[0] + rvec[1]*rvec[1]);
  
  norm = sqrt((1.0 - rvec[2])*(1.0 + rvec[2])*(1.0 - vec[2])*(1.0 + vec[2]));
  
  sinpsi = (rephi_vec[0]*etheta_rvec[0] + rephi_vec[1]*etheta_rvec[1] + rephi_vec[2]*etheta_rvec[2])/norm;
  cospsi = (rephi_vec[0]*ephi_rvec[0] + rephi_vec[1]*ephi_rvec[1] + rephi_vec[2]*ephi_rvec[2])/norm;
  
  /// psi is defined as
  //   R(e_theta) = cos(psi) e_theta' - sin(psi) e_phi'
  //   R(e_phi)   = sin(psi) e_theta' + cos(psi) e_phi'
     
  //   thus to rotate tangent vector
  //   t = t_theta R(e_theta) + t_phi R(e_phi)
     
  //   we plug and chug to get
  //   t = (t_theta*cos(psi) + t_phi*sin(psi)) e_theta' + (-t_theta*sin(psi) + t_phi*cos(psi)) e_phi' 
  
  ralpha[0] = alpha[0]*cospsi + alpha[1]*sinpsi;
  ralpha[1] = -1.0*alpha[0]*sinpsi + alpha[1]*cospsi;
  
  // psi is defined as
  //   R(e_theta) = cos(psi) e_theta' - sin(psi) e_phi'
  //   R(e_phi)   = sin(psi) e_theta' + cos(psi) e_phi'
     
  //   under this coordinate change we have that 
     
  //   T' = R x T x Transpose(R)
  //   where 
  //   T = | t_00 t_01 |
  //       | t_10 t_11 |
         
  //   R = | cos(psi) -sin(psi) |
  //       | sin(psi)  cos(psi) |
   
  double r[2][2],rt[2][2],t1[2][2];
  int i,j;
  
  r[0][0] = cospsi;
  r[0][1] = -1.0*sinpsi;
  r[1][0] = sinpsi;
  r[1][1] = cospsi;
  
  rt[0][0] = r[0][0];
  rt[0][1] = r[1][0];
  rt[1][0] = r[0][1];
  rt[1][1] = r[1][1];
  
  for(i=0;i<2;++i)
    for(j=0;j<2;++j)
      t1[i][j] = U[i][0]*r[0][j]  + U[i][1]*r[1][j];
  
  for(i=0;i<2;++i)
    for(j=0;j<2;++j)
      rU[i][j] = rt[i][0]*t1[0][j]  + rt[i][1]*t1[1][j];
}

#include <gsl/gsl_multifit.h>
static int shearinterp_poly(double _vec[3], double *pot, double alpha[2], double U[4])
{
  long doNotHaveCell = 0;
  long bundleMapShift = 2*(rayTraceData.poissonOrder - rayTraceData.bundleOrder);
  long Ninds,*nestinds,*mapinds;
  struct ray {
    double alpha[2];
    double U[2][2];
    double theta;
    double phi;
  };
  struct ray *rays;
  double rvec[3],axis[3],sinangle,cosangle;
  double n[3],rn[3],defl[2],rdefl[2],A[2][2],rA[2][2];
  double pot_interp,norm;
  long k,mapNest,baseInd,bundleNest;
  double vec[3],thetap,phip;
  
  norm = sqrt(_vec[0]*_vec[0] + _vec[1]*_vec[1] +_vec[2]*_vec[2]);
  vec[0] = _vec[0]/norm;
  vec[1] = _vec[1]/norm;
  vec[2] = _vec[2]/norm;
  
  //get nbrs for interp
  nestinds = get_rings_shearinterp_poly(vec,1l,&Ninds);
  
  //alloc info for rays
  rays = (struct ray*)malloc(sizeof(struct ray)*Ninds);
  assert(rays != NULL);
  mapinds = (long*)malloc(sizeof(long)*Ninds);
  assert(mapinds != NULL);
  
  //get rot mat and axis for rot of rays
  rvec[0] = 1.0;
  rvec[1] = 0.0;
  rvec[2] = 0.0;
  axis[0] = vec[1]*rvec[2] - vec[2]*rvec[1];
  axis[1] = vec[2]*rvec[0] - vec[0]*rvec[2];
  axis[2] = vec[0]*rvec[1] - vec[1]*rvec[0];
  cosangle = vec[0]*rvec[0] + vec[1]*rvec[1] + vec[2]*rvec[2];
  sinangle = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
  if(sinangle != 0.0)
    {
      axis[0] /= sinangle;
      axis[1] /= sinangle;
      axis[2] /= sinangle;
    }
  else
    {
      axis[0] = 1.0;
      axis[1] = 0.0;
      axis[2] = 0.0;
    }
  vec2ang(rvec,&thetap,&phip);
  
  for(k=0;k<Ninds;++k)
    {
      if(nestinds[k] >= 0)
	{
	  doNotHaveCell = 0;
	  
	  mapNest = nestinds[k];
	  bundleNest = (mapNest >> bundleMapShift);
	  baseInd = (bundleNest << bundleMapShift);
	  if(bundleCells[bundleNest].firstMapCell >= 0
	     &&
	     (ISSETBITFLAG(bundleCells[bundleNest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,MAPBUFF_BUNDLECELL))
	     )
	    {
	      mapinds[k] = bundleCells[bundleNest].firstMapCell + mapNest - baseInd;
			  
	      if(mapCellsGradThetaTheta[mapinds[k]].index == -1)
		doNotHaveCell = 1;
	    }
	  else
	    doNotHaveCell = 1;
	  
	  if(doNotHaveCell)
	    return doNotHaveCell;
	  
	  assert(mapCells[mapinds[k]].index == mapNest);
	  
	  //rotate ray to new coords
	  nest2vec(nestinds[k],n,rayTraceData.poissonOrder);
	  defl[0] = mapCellsGradTheta[mapinds[k]].val;
	  defl[1] = mapCellsGradPhi[mapinds[k]].val;
	  A[0][0] = mapCellsGradThetaTheta[mapinds[k]].val;
	  A[0][1] = mapCellsGradThetaPhi[mapinds[k]].val;
	  A[1][0] = mapCellsGradThetaPhi[mapinds[k]].val;
	  A[1][1] = mapCellsGradPhiPhi[mapinds[k]].val;
	  rot_ray_shearinterp_poly(n,rn,defl,rdefl,A,rA,axis,cosangle,sinangle);
	  
	  rays[k].alpha[0] = rdefl[0];
	  rays[k].alpha[1] = rdefl[1];
	  rays[k].U[0][0] = rA[0][0];
	  rays[k].U[0][1] = rA[0][1];
	  rays[k].U[1][0] = rA[1][0];
	  rays[k].U[1][1] = rA[1][1];
	  
	  vec2ang(rn,&(rays[k].theta),&(rays[k].phi));
	  
	  while(rays[k].phi > M_PI)
	    rays[k].phi = rays[k].phi - 2.0*M_PI;
	  while(rays[k].phi < -M_PI)
	    rays[k].phi = rays[k].phi + 2.0*M_PI;
	}
      else
	{
	  fprintf(stderr,"%d: nestind is -1 in shearinterp_poly! nest = %ld\n",ThisTask,nestinds[k]);
	  assert(0);
	}
    }
  
  //now do poly fits for parameters  
  size_t nobs,p,order;
  size_t i,j,a,b,m;
  gsl_multifit_linear_workspace *work;
  gsl_matrix *X,*cov;
  gsl_vector *y,*c,*x;
  double chisq,yval,yerr;
  
  nobs = (size_t) (Ninds);
  p = 1;
  order = 0;
  for(i=1;i<nobs;++i)
    {
      if((long) (p+i+1) <= Ninds)
	{
	  p = p + i+1;
	  order = i;
	}
      else
	break;
    }
  
  work = gsl_multifit_linear_alloc(nobs,p);
  assert(work != NULL);
  X = gsl_matrix_alloc(nobs,p);
  assert(X != NULL);
  y = gsl_vector_alloc(nobs);
  assert(y != NULL);
  x = gsl_vector_alloc(p);
  assert(x != NULL);
  work = gsl_multifit_linear_alloc(nobs,p);
  assert(work != NULL);
  cov = gsl_matrix_alloc(p,p);
  assert(cov != NULL);
  c = gsl_vector_alloc(p);
  assert(c != NULL);
  
  //make X matrix and vector
  for(i=0;i<nobs;++i)
    {
      m = 0;
      j = order;
      for(a=0;a<=order;++a)
	{
	  for(b=0;b<=j;++b)
	    {
	      assert(m < p);
	      gsl_matrix_set(X,i,m,pow(rays[i].theta,(double) a)*pow(rays[i].phi,(double) b));
	      ++m;
	    }
	  --j;
	}
    }
  
  m = 0;
  j = order;
  for(a=0;a<=order;++a)
    {
      for(b=0;b<=j;++b)
	{
	  assert(m < p);
	  gsl_vector_set(x,m,pow(thetap,(double) a)*pow(phip,(double) b));
	  ++m;
	}
      --j;
    }
  
  //pot fit and interp
  for(i=0;i<nobs;++i)
    gsl_vector_set(y,i,mapCells[mapinds[i]].val);
  gsl_multifit_linear(X,y,c,cov,&chisq,work);
  gsl_multifit_linear_est(x,c,cov,&yval,&yerr);
  pot_interp = yval;
  
  //defl fit and interp
  for(i=0;i<nobs;++i)
    gsl_vector_set(y,i,rays[i].alpha[0]);
  gsl_multifit_linear(X,y,c,cov,&chisq,work);
  gsl_multifit_linear_est(x,c,cov,&yval,&yerr);
  defl[0] = yval;
  
  for(i=0;i<nobs;++i)
    gsl_vector_set(y,i,rays[i].alpha[1]);
  gsl_multifit_linear(X,y,c,cov,&chisq,work);
  gsl_multifit_linear_est(x,c,cov,&yval,&yerr);
  defl[1] = yval;
  
  //A fit and interp
  for(i=0;i<nobs;++i)
    gsl_vector_set(y,i,rays[i].U[0][0]);
  gsl_multifit_linear(X,y,c,cov,&chisq,work);
  gsl_multifit_linear_est(x,c,cov,&yval,&yerr);
  A[0][0] = yval;
  
  for(i=0;i<nobs;++i)
    gsl_vector_set(y,i,rays[i].U[0][1]);
  gsl_multifit_linear(X,y,c,cov,&chisq,work);
  gsl_multifit_linear_est(x,c,cov,&yval,&yerr);
  A[0][1] = yval;
  
  for(i=0;i<nobs;++i)
    gsl_vector_set(y,i,rays[i].U[1][0]);
  gsl_multifit_linear(X,y,c,cov,&chisq,work);
  gsl_multifit_linear_est(x,c,cov,&yval,&yerr);
  A[1][0] = yval;
  
  for(i=0;i<nobs;++i)
    gsl_vector_set(y,i,rays[i].U[1][1]);
  gsl_multifit_linear(X,y,c,cov,&chisq,work);
  gsl_multifit_linear_est(x,c,cov,&yval,&yerr);
  A[1][1] = yval;
  
  gsl_matrix_free(X);
  gsl_vector_free(y);
  gsl_vector_free(x);
  gsl_vector_free(c);
  gsl_multifit_linear_free(work);
  gsl_matrix_free(cov);
    
  //rotate back to ray loc
  rot_ray_shearinterp_poly(rvec,n,defl,rdefl,A,rA,axis,cosangle,-sinangle);
  
  *pot = pot_interp;
  alpha[0] = rdefl[0];
  alpha[1] = rdefl[1];
  U[0] = rA[0][0];
  U[1] = rA[0][1];
  U[2] = rA[1][0];
  U[3] = rA[1][1];
  
  free(rays);
  free(mapinds);
  free(nestinds);
  
  return doNotHaveCell;
}
*/

static int shearinterp_comp(double rvec[3], double *pot, double alpha[2], double U[4])
{
  double theta,phi,wgt[4],pot_interp;
  long k,pix[4];
  double gtheta,gphi;
  double tvec[2],rtvec[2],vec[3];
  long mapinds[4];
  double ttens_interp[2][2],ttens[2][2],rttens[2][2];
  long doNotHaveCell = 0;
  long baseInd,mapNest,bundleNest,bundleMapShift;
  long Nwgt = 4;
  
  bundleMapShift = 2*(rayTraceData.poissonOrder - rayTraceData.bundleOrder);
  
  vec2ang(rvec,&theta,&phi);
  get_interpol(theta,phi,pix,wgt,rayTraceData.poissonOrder);
  
  pot_interp = 0.0;
  gtheta = 0.0;
  gphi = 0.0;
  ttens_interp[0][0] = 0.0;
  ttens_interp[0][1] = 0.0;
  ttens_interp[1][0] = 0.0;
  ttens_interp[1][1] = 0.0;
  
  for(k=0;k<Nwgt;++k)
    {
      if(pix[k] >= 0)
	{
	  doNotHaveCell = 0;
	  
	  mapNest = ring2nest(pix[k],rayTraceData.poissonOrder);
	  bundleNest = (mapNest >> bundleMapShift);
	  baseInd = (bundleNest << bundleMapShift);
	  if(bundleCells[bundleNest].firstMapCell >= 0
	     &&
	     (ISSETBITFLAG(bundleCells[bundleNest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,MAPBUFF_BUNDLECELL))
	     )
	    {
	      mapinds[k] = bundleCells[bundleNest].firstMapCell + mapNest - baseInd;
			  
	      if(mapCellsGradThetaTheta[mapinds[k]].index == -1)
		doNotHaveCell = 1;
	    }
	  else
	    doNotHaveCell = 1;
	  
	  if(doNotHaveCell)
	    return doNotHaveCell;
	  
	  assert(mapCells[mapinds[k]].index == mapNest);

	  pot_interp += mapCells[mapinds[k]].val*wgt[k];
		      
	  nest2vec(mapNest,vec,rayTraceData.poissonOrder);
	  tvec[0] = mapCellsGradTheta[mapinds[k]].val;
	  tvec[1] = mapCellsGradPhi[mapinds[k]].val;
	  paratrans_tangvec(tvec,vec,rvec,rtvec);
	  gtheta += rtvec[0]*wgt[k];
	  gphi += rtvec[1]*wgt[k];
		      
	  ttens[0][0] = mapCellsGradThetaTheta[mapinds[k]].val;
	  ttens[0][1] = mapCellsGradThetaPhi[mapinds[k]].val;
	  ttens[1][0] = mapCellsGradThetaPhi[mapinds[k]].val;
	  ttens[1][1] = mapCellsGradPhiPhi[mapinds[k]].val;
	  paratrans_tangtensor(ttens,vec,rvec,rttens);
	  ttens_interp[0][0] += rttens[0][0]*wgt[k];
	  ttens_interp[0][1] += rttens[0][1]*wgt[k];
	  ttens_interp[1][0] += rttens[1][0]*wgt[k];
	  ttens_interp[1][1] += rttens[1][1]*wgt[k];
	}
    }
  
  *pot = pot_interp;
  alpha[0] = gtheta;
  alpha[1] = gphi;
  U[0] = ttens_interp[0][0];
  U[1] = ttens_interp[0][1];
  U[2] = ttens_interp[1][0];
  U[3] = ttens_interp[1][1];
  
  return doNotHaveCell;
}
#endif

#ifdef DEBUG_IO
static void write_ringmap(char name[], float *mapvec, HEALPixSHTPlan plan)
{
  fftwf_complex *mapvec_complex;
  FILE *fp;
  long nring,ring,Nside,ringpix,i,firstRing,lastRing;
  
  Nside = order2nside(rayTraceData.poissonOrder);
  
  fp = fopen(name,"w");
  
  ring = NTasks;
  fwrite(&ring,(size_t) 1,sizeof(long),fp);
  
  ring = Nside;
  fwrite(&ring,(size_t) 1,sizeof(long),fp);
      
  ring = 0;
  firstRing = plan.firstRingTasks[ThisTask];
  lastRing = plan.lastRingTasks[ThisTask];
  mapvec_complex = (fftwf_complex*) mapvec;
  for(nring=firstRing;nring<=lastRing;++nring)
    {
      if(nring < Nside)
	ringpix = 4*nring;
      else
	ringpix = 4*Nside;

      mapvec = (float*) (mapvec_complex+plan.northStartIndMapvec[nring-firstRing]);
      for(i=0;i<ringpix;++i)
	{
	  ++ring;
	}

      if(nring != 2*Nside)
	{
	  mapvec = (float*) (mapvec_complex+plan.southStartIndMapvec[nring-firstRing]);
	  for(i=0;i<ringpix;++i)
	    {
	      ++ring;
	    }
	}
    }
  mapvec = (float*) mapvec_complex;
      
  fwrite(&ring,(size_t) 1,sizeof(long),fp);
      
  firstRing = plan.firstRingTasks[ThisTask];
  lastRing = plan.lastRingTasks[ThisTask];
  mapvec_complex = (fftwf_complex*) mapvec;
  for(nring=firstRing;nring<=lastRing;++nring)
    {
      if(nring < Nside)
	ringpix = 4*nring;
      else
	ringpix = 4*Nside;

      mapvec = (float*) (mapvec_complex+plan.northStartIndMapvec[nring-firstRing]);
      for(i=0;i<ringpix;++i)
	{
	  ring = i + plan.northStartIndGlobalMap[nring-firstRing];
	  
	  fwrite(&(mapvec[i]),(size_t) 1,sizeof(float),fp);
	  fwrite(&ring,(size_t) 1,sizeof(long),fp);
	}

      if(nring != 2*Nside)
	{
	  mapvec = (float*) (mapvec_complex+plan.southStartIndMapvec[nring-firstRing]);
	  for(i=0;i<ringpix;++i)
	    {
	      ring = i + plan.southStartIndGlobalMap[nring-firstRing];
	      
	      fwrite(&(mapvec[i]),(size_t) 1,sizeof(float),fp);
	      fwrite(&ring,(size_t) 1,sizeof(long),fp);
	    }
	}
    }
  mapvec = (float*) mapvec_complex;
  
  fclose(fp);
}

static void write_localmap(char name[], HEALPixMapCell *localMapCells, long NumLocalMapCells)
{
  long ring,i;
  FILE *fp;
  
  fp = fopen(name,"w");
  ring = order2nside(rayTraceData.poissonOrder);
  fwrite(&ring,(size_t) 1,sizeof(long),fp);
  fwrite(&NumLocalMapCells,(size_t) 1,sizeof(long),fp);
  for(i=0;i<NumLocalMapCells;++i)
    {
      fwrite(&(localMapCells[i].val),(size_t) 1,sizeof(float),fp);
      ring = nest2ring(localMapCells[i].index,rayTraceData.poissonOrder);
      fwrite(&ring,(size_t) 1,sizeof(long),fp);
    }
  fclose(fp);
}
#endif

