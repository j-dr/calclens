#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>

#include "raytrace.h"
#include "mgpoissonsolve.h"

//macros for debugging 
//#define PRINT_MGGRID //prints out all grids used for MG poisson solve
//#define NANCHECK_MGGRID  //checks grid values for NaNs 

//macros for code
//#define NGP_FILL_U_MGGRID       /* controls what kind of interpolation is done for guess at lens pot */
                                  /* the NGP interp is faster, but then MG poisson solver is slower by approx same factor */
#define MGDERIV_METRIC_FAC_AT_END /* define to compute metrix factors for derivs on MG grid after interp to ray position */
//#define SCALE_MGDENS              /* define to scale the density and potential by a constant factor before poisson solver runs */

//prototypes
static void mgpoissonsolve_bundlecell(long bundleCell, const double densfact, const double backdens);
static void fill_rho_mggrid(MGGrid rho, long bundleCell, const double densfact, const double backdens);
static void fill_bcs_mggrid(MGGrid u);
static void fill_u_mggrid(MGGrid u);
static void fill_uderivs_rays(MGGrid u, long bundleCell);
static void getderiv_mggrid_xtheta(MGGrid u, double *gx);
static void getderiv_mggrid_xtheta_xtheta(MGGrid u, double *gxx);
static void getderiv_mggrid_yphi(MGGrid u, double *gy);
static void getderiv_mggrid_yphi_yphi(MGGrid u, double *gyy);
static void getderiv_mggrid_xtheta_yphi(MGGrid u, double *gxy);
static double getinterpval_healpix_mggrid(double vec[3], double RmatPatchToSphere[3][3]);
static void get_rmats_bundlecell(long bundleCell, double RmatSphereToPatch[3][3], double RmatPatchToSphere[3][3]);
static void rot_tangvectens(double _vec[3], double tvec[2], double ttens[2][2], double Rmat[3][3], double rvec[3], double rtvec[2], double rttens[2][2]);
//static double sphdist_haversine(double t1, double p1, double t2, double p2);
static void getderiv_mggrid(MGGrid u, double **gx, double **gy, double **gxx, double **gxy, double **gyy);

//timing variables
#define NUMRUNTIMES 11
static double runTimes[NUMRUNTIMES];
/* Timing variables
  runTimes[0] - total time to compute density
  runTimes[1] - total time to do MG solve
  runTimes[2] - time to prep density kernels
  runTimes[3] - time to normalize density kernels
  runTimes[4] - time to assign density kernels
  runTimes[5] - total time to comute BCs for MG solve
  runTimes[6] - total time to compute initial guess for MG solve solution
  runTimes[7] - total extra time not covered above by 0,1,5,6
  runTimes[8] - time to compute derivs of potential
  runTimes[9] - time to assign derivs to rays
  runTimes[10] - time to rot derivs back to global coords
*/

void mgpoissonsolve(double densfact, double backdens)
{
  long i;
    
  for(i=0;i<NUMRUNTIMES;++i)
    runTimes[i] = 0.0;
  double runTime,mintm,maxtm,avgtm,ptime;
  int pstart = 1,numd = 0,numt = lastRestrictedPeanoIndTasks[ThisTask] - firstRestrictedPeanoIndTasks[ThisTask] + 1;
  double totTime;
  
  logProfileTag(PROFILETAG_MG);
  
  totTime = -MPI_Wtime();
  runTime = -MPI_Wtime();
  ptime = -MPI_Wtime();
  
  runTimes[7] -= MPI_Wtime();
  //get smoothing lengths for particles
  get_smoothing_lengths();
  
  //loop through active bundle cells and run MG solver
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
        {
	  runTimes[7] += MPI_Wtime();
	  bundleCells[i].cpuTime -= MPI_Wtime();
	  mgpoissonsolve_bundlecell(i,densfact,backdens);
	  bundleCells[i].cpuTime += MPI_Wtime();
	  runTimes[7] -= MPI_Wtime();
	  
	  if(ThisTask == 0 && ((ptime + MPI_Wtime()) > 60.0 || pstart))
	    {
	      fprintf(stderr,"%5d of %5d - multi-grid poisson solve times: dens,BCS+Ufill,solve = %lf|%lf|%lf sec, prep,norm,assign,BCS,Ufill = %lf|%lf|%lf|%lf|%lf\n",
		      numd+1,numt,runTimes[0],runTimes[5]+runTimes[6],runTimes[1],runTimes[2],runTimes[3],runTimes[4],runTimes[5],runTimes[6]);
	      ptime = -MPI_Wtime();
	      pstart = 0;
	    }
	  
	  ++numd;
	}
    }
  
  free_mapcells();
      
  runTimes[7] += MPI_Wtime();
  runTime += MPI_Wtime();
  
  logProfileTag(PROFILETAG_MG);
  
  MPI_Reduce(&runTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&runTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&runTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
    
  if(ThisTask == 0)
    {
      //timing info that can be printed if wanted
      //print_runtimes_mgsteps();
      reset_runtimes_mgsteps();
      //fprintf(stderr,"deriv times - comp,interp,rot = %lf|%lf|%lf seconds.\n",runTimes[8],runTimes[9],runTimes[10]);
      fprintf(stderr,"dens+BCS+Ufill,solve,deriv+extra = %lf|%lf|%lf seconds.\n",
	      runTimes[0]+runTimes[5]+runTimes[6],runTimes[1],runTimes[7]);
      fprintf(stderr,"multi-grid poisson solve took %lf seconds (max,min,avg = %lf|%lf|%lf seconds, %.2f percent).\n",
	      runTime,maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
    }
}

//# of base grid sizes to test
#define MGGRID_NTEST 4

void mgpoissonsolve_bundlecell(long bundleCell, const double densfact, const double backdens)
{
  long N,Nlev,lev,Nmin,j,i;
  double L;
  MGGrid u;
  MGGridSet *grids;
  long NumPreSmooth,NumPostSmooth,NumOuterCycles,NumInnerCycles;
  long Nminv[MGGRID_NTEST] = {4,5,7,9};
  long Nmaxv[MGGRID_NTEST],Nlevv[MGGRID_NTEST];
#ifdef PRINT_MGGRID
  char fname[MAX_FILENAME];
#endif
  
  runTimes[7] -= MPI_Wtime();
  
  //setup
  L = MGPATCH_SIZE_FAC*sqrt(4.0*M_PI/order2npix(rayTraceData.bundleOrder));
  N = rayTraceData.NumMGPatch;
  NumPreSmooth = 1;
  NumPostSmooth = NumPreSmooth;
  NumOuterCycles = 200;
  NumInnerCycles = 2;
  
  //find most efficient grid
  for(j=0;j<MGGRID_NTEST;++j)
    {
      Nmin = Nminv[j];
      Nlev = 1;
      while(Nmin < N)
	{
	  Nmin *= 2;
	  ++Nlev;
	}
      Nlevv[j] = Nlev;
      Nmaxv[j] = Nmin;
    }
  
  N = Nmaxv[0];
  Nlev = Nlevv[0];
  for(j=1;j<MGGRID_NTEST;++j)
    {
      if(Nmaxv[j] < N)
	{
	  N = Nmaxv[j];
	  Nlev = Nlevv[j];
	}
    }
  
  grids = (MGGridSet*)malloc(sizeof(MGGridSet)*Nlev);
  assert(grids != NULL);
  
  for(lev=Nlev-1;lev>=0;--lev)
    {
      grids[lev].u = alloc_mggrid(N,L);
      get_rmats_bundlecell(bundleCell,grids[lev].u->RmatSphereToPatch,grids[lev].u->RmatPatchToSphere);
      grids[lev].rho = copy_mggrid(grids[lev].u);
      
      N /= 2;
    }
  
  runTimes[7] += MPI_Wtime();
  
  //get dens and BCS
  runTimes[0] -= MPI_Wtime();
  
  fill_rho_mggrid(grids[Nlev-1].rho,bundleCell,densfact,backdens);
  
#ifdef PRINT_MGGRID  
  sprintf(fname,"%s/mgpatch_rho_bcell%ld.dat",rayTraceData.OutputPath,bundleCell);
  write_mggrid(fname,grids[Nlev-1].rho);
#endif
  
  runTimes[0] += MPI_Wtime();
  
  runTimes[5] -= MPI_Wtime();
  for(lev=Nlev-1;lev>=0;--lev)
    fill_bcs_mggrid(grids[lev].u);
  runTimes[5] += MPI_Wtime();
  
  runTimes[6] -= MPI_Wtime();
  fill_u_mggrid(grids[Nlev-1].u);
  runTimes[6] += MPI_Wtime();
  
#ifdef SCALE_MGDENS
  double nfact = backdens;
  for(lev=Nlev-1;lev>=0;--lev)
    {
      //scale dens
      for(i=0;i<grids[lev].rho->N;++i)
	for(j=0;j<grids[lev].rho->N;++j)
	  grids[lev].rho->grid[j + (grids[lev].rho->N)*i] = (grids[lev].rho->grid[j + (grids[lev].rho->N)*i])/nfact;
      
      //scale pot
      for(i=0;i<grids[lev].u->N;++i)
	for(j=0;j<grids[lev].u->N;++j)
	  grids[lev].u->grid[j + (grids[lev].u->N)*i] = (grids[lev].u->grid[j + (grids[lev].u->N)*i])/nfact;
    }
#endif
  
#ifdef PRINT_MGGRID
  sprintf(fname,"%s/mgpatch_ustart_bcell%ld.dat",rayTraceData.OutputPath,bundleCell);
  write_mggrid(fname,grids[Nlev-1].u);
#endif  
  
  //solve poisson equation
  double t,L1norm;
  t = runTimes[1];
  runTimes[1] -= MPI_Wtime();
  
  logProfileTag(PROFILETAG_MG);
  
  logProfileTag(PROFILETAG_MGSOLVE);
  
  /*debugging code for pure relaxation method
    lev = Nlev-1;
    j = 0;
    do 
    {
    smooth_mggrid(grids[Nlev-1].u,grids[Nlev-1].rho,10l);
    L1norm = L1norm_mggrid(grids[Nlev-1].u,grids[Nlev-1].rho);
    fprintf(stderr,"L1norm = %le at %ld\n",L1norm,j);
    ++j;
    }
    while(L1norm > 1e-8);
    
    exit(1);
  */
  L1norm = solve_fas_mggrid(grids,Nlev,NumPreSmooth,NumPostSmooth,NumOuterCycles,NumInnerCycles,rayTraceData.MGConvFact);
      
  logProfileTag(PROFILETAG_MGSOLVE);
  
  logProfileTag(PROFILETAG_MG);
  
  runTimes[1] += MPI_Wtime();
  
  //fprintf(stderr,"bundle cell %ld took %lf seconds for a %ld cell sized grid with %ld levels w/ L1norm = %le.\n",bundleCell,runTimes[1]-t,N,Nlev,L1norm);

#ifdef PRINT_MGGRID  
  sprintf(fname,"%s/mgpatch_u_bcell%ld.dat",rayTraceData.OutputPath,bundleCell);
  write_mggrid(fname,grids[Nlev-1].u);
#endif  
  
  runTimes[7] -= MPI_Wtime();
  
  //free some memory
  u = grids[Nlev-1].u;
  free_mggrid(grids[Nlev-1].rho);
  for(lev=Nlev-2;lev>=0;--lev)
    {
      free_mggrid(grids[lev].u);
      free_mggrid(grids[lev].rho);
    }
  free(grids);
  
#ifdef SCALE_MGDENS
  //scale pot back
  for(i=0;i<u->N;++i)
    for(j=0;j<u->N;++j)
      u->grid[j + (u->N)*i] = (u->grid[j + (u->N)*i])*nfact;
#endif
  
  //get derivs for rays
  fill_uderivs_rays(u,bundleCell);
  
  //clean up
  free_mggrid(u);
  
  runTimes[7] += MPI_Wtime();
}

static void fill_rho_mggrid(MGGrid rho, long bundleCell, const double densfact, const double backdens)
{
  double L,vecp[3],vec[3],rvec[3],disLim,cosDis,smoothingRad;
  long i,j,n,m,k,b;
  double thetap,phip,theta,phi,sfac,sfac1,sfac2;
  long pmin,pmax,tmin,tmax;
  double totmass,r,vol;
  struct binData {
    long xind;
    long yind;
    double dens;
    double vol;
  } *bd;
  long Nbd,bdind;
  long *listpix,Nlistpix,NlistpixMax;
  double z,z0,xa,x,ysq,cosang,dphi;
      
  runTimes[2] -= MPI_Wtime();
  
  listpix = NULL;
  NlistpixMax = 0;

  //mem for binning
  Nbd = 200;
  bd = (struct binData*)malloc(sizeof(struct binData)*Nbd);
  assert(bd != NULL);
  
  //setup
  L = MGPATCH_SIZE_FAC*sqrt(4.0*M_PI/order2npix(rayTraceData.bundleOrder));
  zero_mggrid(rho);
  disLim = L + rayTraceData.maxSL*2.0;
  nest2ang(bundleCell,&theta,&phi,rayTraceData.bundleOrder);
  Nlistpix = query_disc_inclusive_nest_fast(theta,phi,disLim,&listpix,&NlistpixMax,rayTraceData.bundleOrder);
    
  runTimes[2] += MPI_Wtime();
  
  //assign dens with parts
  for(b=0;b<Nlistpix;++b)
    {
      i = listpix[b];
      if(bundleCells[i].Nparts > 0)
	{
	  for(k=0;k<bundleCells[i].Nparts;++k)
            {
	      runTimes[2] -= MPI_Wtime();
	      	      
	      //get part pos in patch coords
              vecp[0] = (double) (lensPlaneParts[k+bundleCells[i].firstPart].pos[0]);
              vecp[1] = (double) (lensPlaneParts[k+bundleCells[i].firstPart].pos[1]);
              vecp[2] = (double) (lensPlaneParts[k+bundleCells[i].firstPart].pos[2]);
	      
	      rvec[0] = 0.0;
	      rvec[1] = 0.0;
	      rvec[2] = 0.0;
	      for(n=0;n<3;++n)
		for(m=0;m<3;++m)
		  rvec[n] += (rho->RmatSphereToPatch[n][m])*vecp[m];
	      r = sqrt(rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2]);
	      rvec[0] /= r;
	      rvec[1] /= r;
	      rvec[2] /= r;
	      
	      vec2ang(rvec,&thetap,&phip);
	      while(phip > M_PI)
		phip -= 2.0*M_PI;
	      while(phip < -M_PI)
		phip += 2.0*M_PI;
	      
	      //determine range of cells to add mass to
	      smoothingRad = lensPlaneParts[k+bundleCells[i].firstPart].smoothingLength;
	      
	      //code cribbed from healpix
              cosang = cos(smoothingRad);
              z0 = cos(thetap);
              xa = 1.0/sqrt((1-z0)*(1+z0));
              //end of code cribbed from healpix 
	      
	      tmin = (thetap - rho->thetaLoc - smoothingRad)/(rho->dL) - 2;
	      tmax = (thetap - rho->thetaLoc + smoothingRad)/(rho->dL) + 2;
	      
	      sfac1 = sin(thetap + smoothingRad);
	      sfac2 = sin(thetap - smoothingRad);
	      if(sfac1 < sfac2)
		sfac = sfac1;
	      else
		sfac = sfac2;
	      
	      pmin = (phip - rho->phiLoc - smoothingRad/sfac)/(rho->dL) - 2;
	      pmax = (phip - rho->phiLoc + smoothingRad/sfac)/(rho->dL) + 2;
	      
	      runTimes[2] += MPI_Wtime();
	      
	      if(((tmin >= 0 && tmin <= rho->N) || (tmax >= 0 && tmax <= rho->N) || 
		  (0 >= tmin && 0 <= tmax) || (rho->N >= tmin && rho->N <= tmax))
		 &&
		 ((pmin >= 0 && pmin <= rho->N) || (pmax >= 0 && pmax <= rho->N) ||
		  (0 >= pmin && 0 <= pmax) || (rho->N >= pmin && rho->N <= pmax)))
		{
		  runTimes[3] -= MPI_Wtime();
		  
		  bdind = (tmax-tmin+1)*(pmax-pmin+1);
		  if(bdind >= Nbd)
		    {
		      Nbd = bdind;
		      bd = (struct binData*)realloc(bd,sizeof(struct binData)*Nbd);
		      if(bd == NULL)
			{
			  fprintf(stderr,"%d: Nbd = %ld, t = %ld|%ld, p = %ld|%ld\n",ThisTask,Nbd,tmin,tmax,pmin,pmax);
			  assert(bd != NULL);
			}
		    }
		  
		  //get total mass by summing over all cells, even if not in grid
		  bdind = 0;
		  totmass = 0.0;
		  for(n=tmin;n<=tmax;++n)
		    {
		      theta = n*(rho->dL) + rho->thetaLoc;
		      if(n >= 0 && n < rho->N)
			{
			  vol = (rho->dL)*(rho->cosfacs[n]);
			  z = rho->costheta[n];
			}
		      else
			{
			  vol = (rho->dL)*(cos(n*(rho->dL) + rho->thetaLoc - (rho->dL)/2.0) - cos(n*(rho->dL) + rho->thetaLoc + (rho->dL)/2.0));
			  z = cos(theta);
			}
		      
		      //code cribbed from healpix
		      //z = cos(theta); - computed above
		      x = (cosang-z*z0)*xa;
		      ysq = 1-z*z-x*x;
		      if(ysq < 0.0)
			continue;
		      //not needed - just continue assert(ysq>=0);
		      dphi=atan2(sqrt(ysq),x);
		      //end of code cribbed from healpix 
		      
		      pmin = (phip - rho->phiLoc - dphi)/(rho->dL) - 2;
		      pmax = (phip - rho->phiLoc + dphi)/(rho->dL) + 2;
		      
		      for(m=pmin;m<=pmax;++m)
			{
			  phi = m*(rho->dL) + rho->phiLoc;
			  if(n >= 0 && n < rho->N && m >= 0 && m < rho->N)
			    {
			      vecp[0] = (rho->sintheta[n])*(rho->cosphi[m]);
			      vecp[1] = (rho->sintheta[n])*(rho->sinphi[m]);
			      vecp[2] = rho->costheta[n];
			    }
			  else
			    ang2vec(vecp,theta,phi);
			  
			  cosDis = rvec[0]*vecp[0] + rvec[1]*vecp[1] + rvec[2]*vecp[2];
			  r = spline_part_dens(cosDis,smoothingRad)*vol;

#ifdef NANCHECK_MGGRID  ///////////////////////////////////////////////////////////////////////////////
			  if(!(gsl_finite(r)))
			    {
			      fprintf(stderr,"bcell = %ld, tp,pp = %le|%le, t,p = %le|%le, rvec = %le|%le|%le, vecp = %le|%le|%le, cosDis = %le, vol = %le, sr = %le, dens = %le\n",
				      bundleCell,thetap,phip,theta,phi,rvec[0],rvec[1],rvec[2],vecp[0],vecp[1],vecp[2],cosDis,vol,smoothingRad,spline_part_dens(cosDis,smoothingRad));
			      MPI_Abort(MPI_COMM_WORLD,999);
			    }
#endif                  ///////////////////////////////////////////////////////////////////////////////
			  
			  if(n >= 0 && n < rho->N && m >= 0 && m < rho->N && r > 0.0)
			    {
			      bd[bdind].xind = n;
			      bd[bdind].yind = m;
			      bd[bdind].dens = r;
			      bd[bdind].vol = vol;
			      ++bdind;
			      
			      assert(bdind <= Nbd);
			    }
			  
			  totmass += r;
			}
		    }
		  
		  runTimes[3] += MPI_Wtime();
		  
		  runTimes[4] -= MPI_Wtime();
		  
		  //assign mass using only cells in grid
		  if(totmass == 0.0) //make sure smoothing rad is not smaller than cell size
		    {
		      n = (thetap - rho->thetaLoc)/(rho->dL);
		      m = (phip - rho->phiLoc)/(rho->dL);
		      
		      if(n >= 0 && n < rho->N && m >= 0 && m < rho->N)
			{
			  rho->grid[n*(rho->N)+m] += lensPlaneParts[k+bundleCells[i].firstPart].mass;
			  
#ifdef NANCHECK_MGGRID  ///////////////////////////////////////////////////////////////////////////////
			  if(!(gsl_finite(rho->grid[n*(rho->N)+m])))
			    {
			      fprintf(stderr,"bcell = %ld, n,m = %ld|%ld, mass = %le\n",
				      bundleCell,n,m,lensPlaneParts[k+bundleCells[i].firstPart].mass);
			      
			      MPI_Abort(MPI_COMM_WORLD,999);
			    }
#endif                  ///////////////////////////////////////////////////////////////////////////////

			}
		    }
		  else
		    {
		      r = lensPlaneParts[k+bundleCells[i].firstPart].mass/totmass;
		      
		      for(j=0;j<bdind;++j)
			{
			  rho->grid[(bd[j].xind)*(rho->N)+(bd[j].yind)] += bd[j].dens*r;
			  				
#ifdef NANCHECK_MGGRID  ///////////////////////////////////////////////////////////////////////////////
			  if(!(gsl_finite(rho->grid[(bd[j].xind)*(rho->N)+(bd[j].yind)])))
			    {
			      fprintf(stderr,"bcell = %ld, ifn,jfn,N = %ld|%ld|%ld, dens,r,totmass,vol,sinf = %le|%le|%le\n",
				      bundleCell,bd[j].xind,bd[j].yind,rho->N,bd[j].dens,r,totmass); 
			      fflush(stderr);
			      
			      MPI_Abort(MPI_COMM_WORLD,999);
			    }
#endif                   ///////////////////////////////////////////////////////////////////////////////
			
			}//for(j=0;j<bdind;++j)
		    }//else ...
		      
		  runTimes[4] += MPI_Wtime();
		  
		}//if((tmin >= 0 && tmin <= rho->N) || ...
	    }//for(k=0;k<bundleCells[i].Nparts;++k)
	}//if(bundleCells[i].Nparts > 0 && cosDis > cosDisLim)
    }//for(b=0;b<Nlistpix;++b)
  
  runTimes[2] -= MPI_Wtime();
  
  if(bd != NULL)
    free(bd);
  
  if(NlistpixMax > 0)
    free(listpix);
  
  //factors for kappa
  double nfact;
  for(i=0;i<rho->N;++i)
    {
      nfact = 1.0/(rho->dL)/(rho->cosfacs[i])*densfact;
      for(j=0;j<rho->N;++j)
	{
	  rho->grid[j + (rho->N)*i] = (rho->grid[j + (rho->N)*i])*nfact; // /(rho->dL)/(rho->cosfacs[i])*densfact;
	  rho->grid[j + (rho->N)*i] = rho->grid[j + (rho->N)*i] - backdens;
	}
    }
  
  //check if on boundary if needed
#ifndef USE_FULLSKY_PARTDIST  
  double ra,dec;
  long checkVacCells;
  nest2ang(bundleCell,&theta,&phi,rayTraceData.bundleOrder);
  ang2radec(theta,phi,&ra,&dec);
  checkVacCells = test_vaccell_boundary(ra,dec,(rho->L)*2.0);
  if(checkVacCells)
    {
      for(i=0;i<rho->N;++i)
	for(j=0;j<rho->N;++j)
	  {
	    //theta = i*(rho->dL) + rho->thetaLoc;
	    //phi = j*(rho->dL) + rho->phiLoc;
	    //ang2vec(vec,theta,phi);
	    
	    vec[0] = (rho->sintheta[i])*(rho->cosphi[j]);
	    vec[1] = (rho->sintheta[i])*(rho->sinphi[j]);
	    vec[2] = rho->costheta[i];
	    
	    /* unrolled this
	       rvec[0] = 0.0;
	       rvec[1] = 0.0;
	       rvec[2] = 0.0;
	       for(n=0;n<3;++n)
	       for(m=0;m<3;++m)
	       rvec[n] += (rho->RmatPatchToSphere[n][m])*vec[m];
	    */
	    rvec[0] = (rho->RmatPatchToSphere[0][0])*vec[0] + (rho->RmatPatchToSphere[0][1])*vec[1] + (rho->RmatPatchToSphere[0][2])*vec[2];
	    rvec[1] = (rho->RmatPatchToSphere[1][0])*vec[0] + (rho->RmatPatchToSphere[1][1])*vec[1] + (rho->RmatPatchToSphere[1][2])*vec[2];
	    rvec[2] = (rho->RmatPatchToSphere[2][0])*vec[0] + (rho->RmatPatchToSphere[2][1])*vec[1] + (rho->RmatPatchToSphere[2][2])*vec[2];
	    vec2ang(rvec,&theta,&phi);
	    ang2radec(theta,phi,&ra,&dec);
	    
	    if(test_vaccell(ra,dec))
	      rho->grid[j + (rho->N)*i] = 0.0;
	  }
    }
#endif

  runTimes[2] += MPI_Wtime();
}

static void fill_u_mggrid(MGGrid u)
{
  long i,j;
  double vec[3],rvec[3],phiv;
  long bnest,offset,nest;
  long bundleMapShift = 2*(rayTraceData.poissonOrder - rayTraceData.bundleOrder);
#ifndef NGP_FILL_U_MGGRID
  double wgt[4],phi,theta;
  long n,pix[4];
#endif
  
  //not using this option - allows one to do fewer interps, but slows down MG convergence by same factor
  //long numg = 1;
  
  //for(i=1;i<u->N-1;i+=numg)
  //for(j=1;j<u->N-1;j+=numg)
  for(i=1;i<u->N-1;++i)
    {
      for(j=1;j<u->N-1;++j)
	{
	  //thetag = i*(u->dL) + u->thetaLoc + (numg - 1.0)*(u->dL)/2.0;
	  //phig = j*(u->dL) + u->phiLoc + (numg - 1.0)*(u->dL)/2.0;
	  
	  //thetag = i*(u->dL) + u->thetaLoc;
	  //phig = j*(u->dL) + u->phiLoc;
	  //ang2vec(vec,thetag,phig);
	  
	  vec[0] = (u->sintheta[i])*(u->cosphi[j]);
	  vec[1] = (u->sintheta[i])*(u->sinphi[j]);
	  vec[2] = u->costheta[i];

	  /* unrolled this
	     rvec[0] = 0.0;
	     rvec[1] = 0.0;
	     rvec[2] = 0.0;
	     for(n=0;n<3;++n)
	     for(m=0;m<3;++m)
	     rvec[n] += (u->RmatPatchToSphere[n][m])*vec[m];
	  */
	  rvec[0] = (u->RmatPatchToSphere[0][0])*vec[0] + (u->RmatPatchToSphere[0][1])*vec[1] + (u->RmatPatchToSphere[0][2])*vec[2];
	  rvec[1] = (u->RmatPatchToSphere[1][0])*vec[0] + (u->RmatPatchToSphere[1][1])*vec[1] + (u->RmatPatchToSphere[1][2])*vec[2];
	  rvec[2] = (u->RmatPatchToSphere[2][0])*vec[0] + (u->RmatPatchToSphere[2][1])*vec[1] + (u->RmatPatchToSphere[2][2])*vec[2];
	  
#ifdef NGP_FILL_U_MGGRID
	  nest = vec2nest(rvec,rayTraceData.poissonOrder);
	  bnest = nest >> bundleMapShift;
	  offset = nest - (bnest << bundleMapShift);
	  
	  if((ISSETBITFLAG(bundleCells[bnest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bnest].active,MAPBUFF_BUNDLECELL))
	     && bundleCells[bnest].firstMapCell >= 0)
	    {
	      phiv = mapCells[bundleCells[bnest].firstMapCell + offset].val;
	      assert(nest == mapCells[bundleCells[bnest].firstMapCell + offset].index);
	    }
	  else
	    {
	      fprintf(stderr,"%d: could not get map cell value in MG poisson solve Ufill!\n",ThisTask);
	      MPI_Abort(MPI_COMM_WORLD,987);
	    }
#else
	  vec2ang(rvec,&theta,&phi);
	  get_interpol(theta,phi,pix,wgt,rayTraceData.poissonOrder);
		  
	  phiv = 0.0;
	  for(n=0;n<4;++n)
	    {
	      nest = ring2nest(pix[n],rayTraceData.poissonOrder);
	      bnest = nest >> bundleMapShift;
	      offset = nest - (bnest << bundleMapShift);
	      
	      if((ISSETBITFLAG(bundleCells[bnest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bnest].active,MAPBUFF_BUNDLECELL))
		 && bundleCells[bnest].firstMapCell >= 0)
		{
		  phiv += mapCells[bundleCells[bnest].firstMapCell + offset].val*wgt[n];
		  assert(nest == mapCells[bundleCells[bnest].firstMapCell + offset].index);
		}
	      else
		{
		  fprintf(stderr,"%d: could not get map cell value in MG poisson solve Ufill!\n",ThisTask);
		  MPI_Abort(MPI_COMM_WORLD,987);
		}
	    }
#endif	  
	  //for(n=0;n<numg;++n)
	  //for(m=0;m<numg;++m)
	  //u->grid[(i+n)*(u->N)+j+m] = phiv;
	  
	  u->grid[i*(u->N)+j] = phiv;
	  
	  //u->grid[i*(u->N)+j] = getinterpval_healpix_mggrid(vec,u->RmatPatchToSphere);
	}
    }
}

static void fill_bcs_mggrid(MGGrid u)
{
  long i,j;
  double vec[3];
  
  i = 0;
  for(j=0;j<u->N;++j)
    {
      vec[0] = (u->sintheta[i])*(u->cosphi[j]);
      vec[1] = (u->sintheta[i])*(u->sinphi[j]);
      vec[2] = u->costheta[i];
      
      u->grid[i*(u->N)+j] = getinterpval_healpix_mggrid(vec,u->RmatPatchToSphere);
    }
  
  i = u->N-1;
  for(j=0;j<u->N;++j)
    {
      vec[0] = (u->sintheta[i])*(u->cosphi[j]);
      vec[1] = (u->sintheta[i])*(u->sinphi[j]);
      vec[2] = u->costheta[i];

      u->grid[i*(u->N)+j] = getinterpval_healpix_mggrid(vec,u->RmatPatchToSphere);
    }
  
  j = 0;
  for(i=1;i<u->N-1;++i)
    {
      vec[0] = (u->sintheta[i])*(u->cosphi[j]);
      vec[1] = (u->sintheta[i])*(u->sinphi[j]);
      vec[2] = u->costheta[i];

      u->grid[i*(u->N)+j] = getinterpval_healpix_mggrid(vec,u->RmatPatchToSphere);
    }
  
  j = u->N-1;
  for(i=1;i<u->N-1;++i)
    {
      vec[0] = (u->sintheta[i])*(u->cosphi[j]);
      vec[1] = (u->sintheta[i])*(u->sinphi[j]);
      vec[2] = u->costheta[i];

      u->grid[i*(u->N)+j] = getinterpval_healpix_mggrid(vec,u->RmatPatchToSphere);
    }
}

static double getinterpval_healpix_mggrid(double vec[3], double RmatPatchToSphere[3][3])
{
  double theta,phi,rvec[3],phiv,wgt[4];
  long n,pix[4],bnest,offset,nest;
  long bundleMapShift = 2*(rayTraceData.poissonOrder - rayTraceData.bundleOrder);
    
  /* unrolled this
  rvec[0] = 0.0;
  rvec[1] = 0.0;
  rvec[2] = 0.0;
  for(n=0;n<3;++n)
    for(m=0;m<3;++m)
      rvec[n] += RmatPatchToSphere[n][m]*vec[m];
  */
  rvec[0] = RmatPatchToSphere[0][0]*vec[0] + RmatPatchToSphere[0][1]*vec[1] + RmatPatchToSphere[0][2]*vec[2];
  rvec[1] = RmatPatchToSphere[1][0]*vec[0] + RmatPatchToSphere[1][1]*vec[1] + RmatPatchToSphere[1][2]*vec[2];
  rvec[2] = RmatPatchToSphere[2][0]*vec[0] + RmatPatchToSphere[2][1]*vec[1] + RmatPatchToSphere[2][2]*vec[2];

  vec2ang(rvec,&theta,&phi);
  get_interpol(theta,phi,pix,wgt,rayTraceData.poissonOrder);
  
  phiv = 0.0;
  for(n=0;n<4;++n)
    {
      nest = ring2nest(pix[n],rayTraceData.poissonOrder);
      bnest = nest >> bundleMapShift;
      offset = nest - (bnest << bundleMapShift);
      
      if((ISSETBITFLAG(bundleCells[bnest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bnest].active,MAPBUFF_BUNDLECELL))
	 && bundleCells[bnest].firstMapCell >= 0)
	{
	  phiv += mapCells[bundleCells[bnest].firstMapCell + offset].val*wgt[n];
	  assert(nest == mapCells[bundleCells[bnest].firstMapCell + offset].index);
	}
      else
	{
	  fprintf(stderr,"%d: could not get map cell value in MG poisson solve BCs!\n",ThisTask);
	  MPI_Abort(MPI_COMM_WORLD,987);
	}
    }
  
  return phiv;
}

static void fill_uderivs_rays(MGGrid u, long bundleCell)
{
  long j,n,m,xind,yind,xindp,yindp;
  double *deriv,wgtx,wgty;
  double vecp[3],norm,rvec[3],thetap,phip,vec[3],phiv;
  double tvec[2],rtvec[2],ttens[2][2],rttens[2][2];
  long notFinite = 0;
  double *thetap_vec,*phip_vec;
#ifdef MGDERIV_METRIC_FAC_AT_END
  double sint,cost;
#endif
  
  /*
    Each deriv is computed one at a time
    then transform to global coords at the end
  */
  
  if(bundleCells[bundleCell].Nrays > 0)
    {
      runTimes[8] -= MPI_Wtime();
      thetap_vec = (double*)malloc(sizeof(double)*bundleCells[bundleCell].Nrays);
      assert(thetap_vec != NULL);
  
      phip_vec = (double*)malloc(sizeof(double)*bundleCells[bundleCell].Nrays);
      assert(phip_vec != NULL);
      
      deriv = (double*)malloc(sizeof(double)*(u->N)*(u->N));
      assert(deriv != NULL);
      
      //theta deriv and phi value
      getderiv_mggrid_xtheta(u,deriv);
      runTimes[8] += MPI_Wtime();
      runTimes[9] -= MPI_Wtime();
      for(j=0;j<bundleCells[bundleCell].Nrays;++j)
	{
	  vecp[0] = bundleCells[bundleCell].rays[j].n[0];
	  vecp[1] = bundleCells[bundleCell].rays[j].n[1];
	  vecp[2] = bundleCells[bundleCell].rays[j].n[2];
	  norm = sqrt(vecp[0]*vecp[0] + vecp[1]*vecp[1] + vecp[2]*vecp[2]);
	  vecp[0] /= norm;
	  vecp[1] /= norm;
	  vecp[2] /= norm;
	  
	  rvec[0] = 0.0;
	  rvec[1] = 0.0;
	  rvec[2] = 0.0;
	  for(n=0;n<3;++n)
	    for(m=0;m<3;++m)
	      rvec[n] += (u->RmatSphereToPatch[n][m])*vecp[m];
	  vec2ang(rvec,&thetap,&phip);
	  while(phip > M_PI)
	    phip -= 2.0*M_PI;
	  while(phip < -M_PI)
	    phip += 2.0*M_PI;
	  
	  thetap_vec[j] = thetap;
	  phip_vec[j] = phip;
	  
	  xind = (thetap - u->thetaLoc)/(u->dL);
	  yind = (phip - u->phiLoc)/(u->dL);
	  
	  if(xind >= 0 && xind < u->N-1 && yind >= 0 && yind < u->N-1)
	    {
	      wgtx = 1.0 - (thetap - (xind*(u->dL) + u->thetaLoc))/(u->dL);
	      wgty = 1.0 - (phip - (yind*(u->dL) + u->phiLoc))/(u->dL);
	      
	      xindp = xind + 1;
	      yindp = yind + 1;
	      
	      phiv = (u->grid[yind  + (u->N)*xind])*wgtx*wgty;
	      phiv += (u->grid[yind  + (u->N)*xindp])*(1.0-wgtx)*wgty;
	      phiv += (u->grid[yindp + (u->N)*xind ])*wgtx*(1.0-wgty);
	      phiv += (u->grid[yindp + (u->N)*xindp])*(1.0-wgtx)*(1.0-wgty);
	      
	      tvec[0] = deriv[yind  + (u->N)*xind ]*wgtx*wgty;
	      tvec[0] += deriv[yind  + (u->N)*xindp]*(1.0-wgtx)*wgty;
	      tvec[0] += deriv[yindp + (u->N)*xind ]*wgtx*(1.0-wgty);
	      tvec[0] += deriv[yindp + (u->N)*xindp]*(1.0-wgtx)*(1.0-wgty);
	      
	      //error check
	      if(!(gsl_finite(phiv)))
		{
		  fprintf(stderr,"phi is not finite! bcell = %ld, phi = %le, wx,wy = %lf|%lf, phi = %le|%le|%le|%le\n",
			  bundleCell,phiv,wgtx,wgty,
			  u->grid[yind  + (u->N)*xind],u->grid[yind  + (u->N)*xindp],u->grid[yindp + (u->N)*xind ],u->grid[yindp + (u->N)*xindp]);
		  fflush(stderr);
		  
		  notFinite = 1;
		}
	      
	      if(!(gsl_finite(tvec[0])))
		{
		  fprintf(stderr,"defl. ang. is not finite! bcell = %ld, tv[0] = %le, wx,wy = %lf|%lf, gx = %le|%le|%le|%le\n",
			  bundleCell,tvec[0],wgtx,wgty,
			  deriv[yind  + (u->N)*xind ],deriv[yind  + (u->N)*xindp],deriv[yindp + (u->N)*xind ],deriv[yindp + (u->N)*xindp]);
		  fflush(stderr);
		  
		  notFinite = 1;
		}                                
	      
	      if(notFinite)
		MPI_Abort(MPI_COMM_WORLD,999);
	      
	      bundleCells[bundleCell].rays[j].phi = phiv;
	      bundleCells[bundleCell].rays[j].alpha[0] = tvec[0];
	    }
	  else
	    {
	      fprintf(stderr,"%d: ray not in MG patch!\n",ThisTask);
	      MPI_Abort(MPI_COMM_WORLD,987);
	    }
	}
      runTimes[9] += MPI_Wtime();
      
      //phi deriv
      runTimes[8] -= MPI_Wtime();
      getderiv_mggrid_yphi(u,deriv);     
      runTimes[8] += MPI_Wtime();
      runTimes[9] -= MPI_Wtime();
      for(j=0;j<bundleCells[bundleCell].Nrays;++j)
	{
	  thetap = thetap_vec[j];
	  phip = phip_vec[j];
	  
	  xind = (thetap - u->thetaLoc)/(u->dL);
	  yind = (phip - u->phiLoc)/(u->dL);
	  
	  if(xind >= 0 && xind < u->N-1 && yind >= 0 && yind < u->N-1)
	    {
	      wgtx = 1.0 - (thetap - (xind*(u->dL) + u->thetaLoc))/(u->dL);
	      wgty = 1.0 - (phip - (yind*(u->dL) + u->phiLoc))/(u->dL);
	      
	      xindp = xind + 1;
	      yindp = yind + 1;
	      
	      tvec[1] = deriv[yind  + (u->N)*xind ]*wgtx*wgty;
	      tvec[1] += deriv[yind  + (u->N)*xindp]*(1.0-wgtx)*wgty;
	      tvec[1] += deriv[yindp + (u->N)*xind ]*wgtx*(1.0-wgty);
	      tvec[1] += deriv[yindp + (u->N)*xindp]*(1.0-wgtx)*(1.0-wgty);
	      
	      if(!(gsl_finite(tvec[1])))
		{
		  fprintf(stderr,"defl. ang. is not finite! bcell = %ld, tv[1] = %le, wx,wy = %lf|%lf, gy = %le|%le|%le|%le\n",
			  bundleCell,tvec[1],wgtx,wgty,
			  deriv[yind  + (u->N)*xind ],deriv[yind  + (u->N)*xindp],deriv[yindp + (u->N)*xind ],deriv[yindp + (u->N)*xindp]);
		  fflush(stderr);
		  
		  notFinite = 1;
		}                                
	      
	      if(notFinite)
		MPI_Abort(MPI_COMM_WORLD,999);
	      
	      bundleCells[bundleCell].rays[j].alpha[1] = tvec[1];
	    }
	  else
	    {
	      fprintf(stderr,"%d: ray not in MG patch!\n",ThisTask);
	      MPI_Abort(MPI_COMM_WORLD,987);
	    }
	}
      runTimes[9] += MPI_Wtime();
      
      //theta-theta deriv
      runTimes[8] -= MPI_Wtime();
      getderiv_mggrid_xtheta_xtheta(u,deriv);     
      runTimes[8] += MPI_Wtime();
      runTimes[9] -= MPI_Wtime();
      for(j=0;j<bundleCells[bundleCell].Nrays;++j)
	{
	  thetap = thetap_vec[j];
	  phip = phip_vec[j];
	  
	  xind = (thetap - u->thetaLoc)/(u->dL);
	  yind = (phip - u->phiLoc)/(u->dL);
	  
	  if(xind >= 0 && xind < u->N-1 && yind >= 0 && yind < u->N-1)
	    {
	      wgtx = 1.0 - (thetap - (xind*(u->dL) + u->thetaLoc))/(u->dL);
	      wgty = 1.0 - (phip - (yind*(u->dL) + u->phiLoc))/(u->dL);
	      
	      xindp = xind + 1;
	      yindp = yind + 1;
	      
	      ttens[0][0] = deriv[yind  + (u->N)*xind ]*wgtx*wgty;
	      ttens[0][0] += deriv[yind  + (u->N)*xindp]*(1.0-wgtx)*wgty;
	      ttens[0][0] += deriv[yindp + (u->N)*xind ]*wgtx*(1.0-wgty);
	      ttens[0][0] += deriv[yindp + (u->N)*xindp]*(1.0-wgtx)*(1.0-wgty);
	      
	      if(!(gsl_finite(ttens[0][0])))
		{
		  fprintf(stderr,"shear is not finite! bcell = %ld, tt[0][0] = %le, wx,wy = %lf|%lf\n\tgxx = %le|%le|%le|%le\n",
			  bundleCell,ttens[0][0],wgtx,wgty,
			  deriv[yind  + (u->N)*xind ],deriv[yind  + (u->N)*xindp],deriv[yindp + (u->N)*xind ],deriv[yindp + (u->N)*xindp]);
		  fflush(stderr);
		  
		  notFinite = 1;
		}
	      
	      if(notFinite)
		MPI_Abort(MPI_COMM_WORLD,999);
	      
	      bundleCells[bundleCell].rays[j].U[0] = ttens[0][0];
	    }
	  else
	    {
	      fprintf(stderr,"%d: ray not in MG patch!\n",ThisTask);
	      MPI_Abort(MPI_COMM_WORLD,987);
	    }
	}
      runTimes[9] += MPI_Wtime();
      
      //phi-phi deriv
      runTimes[8] -= MPI_Wtime();
      getderiv_mggrid_yphi_yphi(u,deriv);     
      runTimes[8] += MPI_Wtime();
      runTimes[9] -= MPI_Wtime();
      for(j=0;j<bundleCells[bundleCell].Nrays;++j)
	{
	  thetap = thetap_vec[j];
	  phip = phip_vec[j];
	  
	  xind = (thetap - u->thetaLoc)/(u->dL);
	  yind = (phip - u->phiLoc)/(u->dL);
	  
	  if(xind >= 0 && xind < u->N-1 && yind >= 0 && yind < u->N-1)
	    {
	      wgtx = 1.0 - (thetap - (xind*(u->dL) + u->thetaLoc))/(u->dL);
	      wgty = 1.0 - (phip - (yind*(u->dL) + u->phiLoc))/(u->dL);
	      
	      xindp = xind + 1;
	      yindp = yind + 1;
	      
	      ttens[1][1] = deriv[yind  + (u->N)*xind ]*wgtx*wgty;
	      ttens[1][1] += deriv[yind  + (u->N)*xindp]*(1.0-wgtx)*wgty;
	      ttens[1][1] += deriv[yindp + (u->N)*xind ]*wgtx*(1.0-wgty);
	      ttens[1][1] += deriv[yindp + (u->N)*xindp]*(1.0-wgtx)*(1.0-wgty);
	      
	      if(!(gsl_finite(ttens[1][1])))
		{
		  fprintf(stderr,"shear is not finite! bcell = %ld, tt[1][1] = %le, wx,wy = %lf|%lf\n\tgyy = %le|%le|%le|%le\n",
			  bundleCell,ttens[1][1],wgtx,wgty,
			  deriv[yind  + (u->N)*xind ],deriv[yind  + (u->N)*xindp],deriv[yindp + (u->N)*xind ],deriv[yindp + (u->N)*xindp]);
		  fflush(stderr);
		  
		  notFinite = 1;
		}
	      
	      if(notFinite)
		MPI_Abort(MPI_COMM_WORLD,999);
	      
	      bundleCells[bundleCell].rays[j].U[3] = ttens[1][1];
	    }
	  else
	    {
	      fprintf(stderr,"%d: ray not in MG patch!\n",ThisTask);
	      MPI_Abort(MPI_COMM_WORLD,987);
	    }
	}
      runTimes[9] += MPI_Wtime();
      
      //theta-phi deriv
      runTimes[8] -= MPI_Wtime();
      getderiv_mggrid_xtheta_yphi(u,deriv);     
      runTimes[8] += MPI_Wtime();
      runTimes[9] -= MPI_Wtime();
      for(j=0;j<bundleCells[bundleCell].Nrays;++j)
	{
	  thetap = thetap_vec[j];
	  phip = phip_vec[j];
	  
	  xind = (thetap - u->thetaLoc)/(u->dL);
	  yind = (phip - u->phiLoc)/(u->dL);
	  
	  if(xind >= 0 && xind < u->N-1 && yind >= 0 && yind < u->N-1)
	    {
	      wgtx = 1.0 - (thetap - (xind*(u->dL) + u->thetaLoc))/(u->dL);
	      wgty = 1.0 - (phip - (yind*(u->dL) + u->phiLoc))/(u->dL);
	      
	      xindp = xind + 1;
	      yindp = yind + 1;
	      
	      ttens[0][1] = deriv[yind  + (u->N)*xind ]*wgtx*wgty;
	      ttens[0][1] += deriv[yind  + (u->N)*xindp]*(1.0-wgtx)*wgty;
	      ttens[0][1] += deriv[yindp + (u->N)*xind ]*wgtx*(1.0-wgty);
	      ttens[0][1] += deriv[yindp + (u->N)*xindp]*(1.0-wgtx)*(1.0-wgty);
	      
	      if(!(gsl_finite(ttens[0][1])))
		{
		  fprintf(stderr,"shear is not finite! bcell = %ld, tt[0][1] = %le, wx,wy = %lf|%lf\n\tgxy = %le|%le|%le|%le\n",
			  bundleCell,ttens[0][1],wgtx,wgty,
			  deriv[yind  + (u->N)*xind ],deriv[yind  + (u->N)*xindp],deriv[yindp + (u->N)*xind ],deriv[yindp + (u->N)*xindp]);
		  fflush(stderr);
		  
		  notFinite = 1;
		}
	      
	      if(notFinite)
		MPI_Abort(MPI_COMM_WORLD,999);
	      
	      bundleCells[bundleCell].rays[j].U[1] = ttens[0][1];
	    }
	  else
	    {
	      fprintf(stderr,"%d: ray not in MG patch!\n",ThisTask);
	      MPI_Abort(MPI_COMM_WORLD,987);
	    }
	}
      runTimes[9] += MPI_Wtime();
      
      //free mem
      runTimes[8] -= MPI_Wtime();
      free(deriv);
      runTimes[8] += MPI_Wtime();
      
      //now rotate back to global coords
      runTimes[10] -= MPI_Wtime();
      for(j=0;j<bundleCells[bundleCell].Nrays;++j)
	{
	  vecp[0] = bundleCells[bundleCell].rays[j].n[0];
	  vecp[1] = bundleCells[bundleCell].rays[j].n[1];
	  vecp[2] = bundleCells[bundleCell].rays[j].n[2];
	  norm = sqrt(vecp[0]*vecp[0] + vecp[1]*vecp[1] + vecp[2]*vecp[2]);
	  vecp[0] /= norm;
	  vecp[1] /= norm;
	  vecp[2] /= norm;
	  
	  rvec[0] = 0.0;
	  rvec[1] = 0.0;
	  rvec[2] = 0.0;
	  for(n=0;n<3;++n)
	    for(m=0;m<3;++m)
	      rvec[n] += (u->RmatSphereToPatch[n][m])*vecp[m];
	  
	  //rot comps
	  tvec[0] = bundleCells[bundleCell].rays[j].alpha[0];
	  tvec[1] = bundleCells[bundleCell].rays[j].alpha[1];
	  
	  ttens[0][0] = bundleCells[bundleCell].rays[j].U[0];
	  ttens[0][1] = bundleCells[bundleCell].rays[j].U[1];
	  ttens[1][0] = ttens[0][1];
	  ttens[1][1] = bundleCells[bundleCell].rays[j].U[3];
	  
#ifdef MGDERIV_METRIC_FAC_AT_END
	  sint = sin(thetap_vec[j]);
	  cost = cos(thetap_vec[j]);
	  
	  tvec[1] /= sint;
	  
	  ttens[0][1] /= sint;
	  ttens[0][1] -= cost/sint*tvec[0];
	  
	  ttens[1][1] /= sint;
	  ttens[1][1] /= sint;
	  ttens[1][1] += cost/sint*tvec[0];
#endif

	  rot_tangvectens(rvec,tvec,ttens,u->RmatPatchToSphere,vec,rtvec,rttens);
	  
	  //fill in comps
	  bundleCells[bundleCell].rays[j].alpha[0] = -1.0*rtvec[0];
	  bundleCells[bundleCell].rays[j].alpha[1] = -1.0*rtvec[1];
	  
	  bundleCells[bundleCell].rays[j].U[0] = rttens[0][0];
	  bundleCells[bundleCell].rays[j].U[1] = rttens[0][1];
	  bundleCells[bundleCell].rays[j].U[2] = rttens[1][0];
	  bundleCells[bundleCell].rays[j].U[3] = rttens[1][1];
	}
      runTimes[10] += MPI_Wtime();
      
      runTimes[8] -= MPI_Wtime();
      free(thetap_vec);
      free(phip_vec);
      runTimes[8] += MPI_Wtime();
    }//if(bundleCells[bundleCell].Nrays > 0)

  /* OLD CODE - not using since it uses too much memory
  //take derivs
  getderiv_mggrid(u,&gx,&gy,&gxx,&gxy,&gyy);
  
  //interp to rays and rot back
  for(j=0;j<bundleCells[bundleCell].Nrays;++j)
    {
      vecp[0] = bundleCells[bundleCell].rays[j].n[0];
      vecp[1] = bundleCells[bundleCell].rays[j].n[1];
      vecp[2] = bundleCells[bundleCell].rays[j].n[2];
      norm = sqrt(vecp[0]*vecp[0] + vecp[1]*vecp[1] + vecp[2]*vecp[2]);
      vecp[0] /= norm;
      vecp[1] /= norm;
      vecp[2] /= norm;
      
      rvec[0] = 0.0;
      rvec[1] = 0.0;
      rvec[2] = 0.0;
      for(n=0;n<3;++n)
	for(m=0;m<3;++m)
	  rvec[n] += (u->RmatSphereToPatch[n][m])*vecp[m];
      vec2ang(rvec,&thetap,&phip);
      while(phip > M_PI)
	phip -= 2.0*M_PI;
      while(phip < -M_PI)
	phip += 2.0*M_PI;
      
      xind = (thetap - u->thetaLoc)/(u->dL);
      yind = (phip - u->phiLoc)/(u->dL);
      
      if(xind >= 0 && xind < u->N-1 && yind >= 0 && yind < u->N-1)
	{
	  wgtx = 1.0 - (thetap - (xind*(u->dL) + u->thetaLoc))/(u->dL);
	  wgty = 1.0 - (phip - (yind*(u->dL) + u->phiLoc))/(u->dL);
	  
	  xindp = xind + 1;
	  yindp = yind + 1;
	  
	  phiv = (u->grid[yind  + (u->N)*xind])*wgtx*wgty;
	  phiv += (u->grid[yind  + (u->N)*xindp])*(1.0-wgtx)*wgty;
	  phiv += (u->grid[yindp + (u->N)*xind ])*wgtx*(1.0-wgty);
	  phiv += (u->grid[yindp + (u->N)*xindp])*(1.0-wgtx)*(1.0-wgty);
	  
	  tvec[0] = gx[yind  + (u->N)*xind ]*wgtx*wgty;
	  tvec[0] += gx[yind  + (u->N)*xindp]*(1.0-wgtx)*wgty;
	  tvec[0] += gx[yindp + (u->N)*xind ]*wgtx*(1.0-wgty);
	  tvec[0] += gx[yindp + (u->N)*xindp]*(1.0-wgtx)*(1.0-wgty);

	  tvec[1] = gy[yind  + (u->N)*xind ]*wgtx*wgty;
	  tvec[1] += gy[yind  + (u->N)*xindp]*(1.0-wgtx)*wgty;
	  tvec[1] += gy[yindp + (u->N)*xind ]*wgtx*(1.0-wgty);
	  tvec[1] += gy[yindp + (u->N)*xindp]*(1.0-wgtx)*(1.0-wgty);
	  
	  ttens[0][0] = gxx[yind  + (u->N)*xind ]*wgtx*wgty;
	  ttens[0][0] += gxx[yind  + (u->N)*xindp]*(1.0-wgtx)*wgty;
	  ttens[0][0] += gxx[yindp + (u->N)*xind ]*wgtx*(1.0-wgty);
	  ttens[0][0] += gxx[yindp + (u->N)*xindp]*(1.0-wgtx)*(1.0-wgty);
	    
	  ttens[0][1] = gxy[yind  + (u->N)*xind ]*wgtx*wgty;
	  ttens[0][1] += gxy[yind  + (u->N)*xindp]*(1.0-wgtx)*wgty;
	  ttens[0][1] += gxy[yindp + (u->N)*xind ]*wgtx*(1.0-wgty);
	  ttens[0][1] += gxy[yindp + (u->N)*xindp]*(1.0-wgtx)*(1.0-wgty);
	    
	  ttens[1][0] = ttens[0][1];
	    
	  ttens[1][1] = gyy[yind  + (u->N)*xind ]*wgtx*wgty;
	  ttens[1][1] += gyy[yind  + (u->N)*xindp]*(1.0-wgtx)*wgty;
	  ttens[1][1] += gyy[yindp + (u->N)*xind ]*wgtx*(1.0-wgty);
	  ttens[1][1] += gyy[yindp + (u->N)*xindp]*(1.0-wgtx)*(1.0-wgty);
	  
	  rot_tangvectens(rvec,tvec,ttens,u->RmatPatchToSphere,vec,rtvec,rttens);
	  
	  //error check forces
	  notFinite = 0;
	  
	  if(!(gsl_finite(phiv)))
            {
              fprintf(stderr,"phi is not finite! bcell = %ld, phi = %le, wx,wy = %lf|%lf, phi = %le|%le|%le|%le\n",
                      bundleCell,phiv,wgtx,wgty,
		      u->grid[yind  + (u->N)*xind ],u->grid[yind  + (u->N)*xindp],u->grid[yindp + (u->N)*xind ],u->grid[yindp + (u->N)*xindp]);
	      fflush(stderr);
	      
	      notFinite = 1;
	    }
	  
	  if(!(gsl_finite(tvec[0]) && gsl_finite(tvec[1]) && gsl_finite(rtvec[0]) && gsl_finite(rtvec[1])))
	    {
	      fprintf(stderr,"defl. ang. is not finite! bcell = %ld, tv = %le|%le, rtv = %le|%le, wx,wy = %lf|%lf, gx = %le|%le|%le|%le, gy = %le|%le|%le|%le\n",
		      bundleCell,tvec[0],tvec[1],rtvec[0],rtvec[1],wgtx,wgty,
		      gx[yind  + (u->N)*xind ],gx[yind  + (u->N)*xindp],gx[yindp + (u->N)*xind ],gx[yindp + (u->N)*xindp],
		      gy[yind  + (u->N)*xind ],gy[yind  + (u->N)*xindp],gy[yindp + (u->N)*xind ],gy[yindp + (u->N)*xindp]);
	      fflush(stderr);
	      
	      notFinite = 1;
	    }
	  
	  if(!(gsl_finite(ttens[0][0]) && gsl_finite(ttens[0][1]) && gsl_finite(ttens[1][1]) && 
	       gsl_finite(rttens[0][0]) && gsl_finite(rttens[0][1]) && gsl_finite(rttens[1][1])))
            {
	      fprintf(stderr,"shear is not finite! bcell = %ld, tt = %le|%le|%le, rtt = %le|%le|%le, wx,wy = %lf|%lf\n\tgxx = %le|%le|%le|%le, gxy = %le|%le|%le|%le, gyy = %le|%le|%le|%le\n",
                      bundleCell,ttens[0][0],ttens[0][1],ttens[1][1],rttens[0][0],rttens[0][1],rttens[1][1],wgtx,wgty,
                      gxx[yind  + (u->N)*xind ],gxx[yind  + (u->N)*xindp],gxx[yindp + (u->N)*xind ],gxx[yindp + (u->N)*xindp],
		      gxy[yind  + (u->N)*xind ],gxy[yind  + (u->N)*xindp],gxy[yindp + (u->N)*xind ],gxy[yindp + (u->N)*xindp],
		      gyy[yind  + (u->N)*xind ],gyy[yind  + (u->N)*xindp],gyy[yindp + (u->N)*xind ],gyy[yindp + (u->N)*xindp]);
	      fflush(stderr);
	      
	      notFinite = 1;
	    }
	  
	  if(notFinite)
	    MPI_Abort(MPI_COMM_WORLD,999);
	  
	  bundleCells[bundleCell].rays[j].phi = phiv;
	  
	  bundleCells[bundleCell].rays[j].alpha[0] = -1.0*rtvec[0];
	  bundleCells[bundleCell].rays[j].alpha[1] = -1.0*rtvec[1];
	  
	  
	  bundleCells[bundleCell].rays[j].U[0] = rttens[0][0];
	  bundleCells[bundleCell].rays[j].U[1] = rttens[0][1];
	  bundleCells[bundleCell].rays[j].U[2] = rttens[1][0];
	  bundleCells[bundleCell].rays[j].U[3] = rttens[1][1];
	}
      else
	{
	  fprintf(stderr,"%d: ray not in MG patch!\n",ThisTask);
	  MPI_Abort(MPI_COMM_WORLD,987);
	}
    }
  
  free(gx);
  free(gy);
  free(gxx);
  free(gxy);
  free(gyy);
  */
}

static void get_rmats_bundlecell(long bundleCell, double RmatSphereToPatch[3][3], double RmatPatchToSphere[3][3])
{
  double bvec[3],rmat1[3][3],rmat2[3][3],angle,axis[3],rvec[3],norm,theta,phi;
  long i,j;
  
  //setup and get rot mats 1 & 2
  nest2vec(bundleCell,bvec,rayTraceData.bundleOrder);
  nest2ang(bundleCell,&theta,&phi,rayTraceData.bundleOrder);
  rvec[0] = bvec[0];
  rvec[1] = bvec[1];
  rvec[2] = 0.0;
  axis[0] = bvec[1]*rvec[2] - bvec[2]*rvec[1];
  axis[1] = bvec[2]*rvec[0] - bvec[0]*rvec[2];
  axis[2] = bvec[0]*rvec[1] - bvec[1]*rvec[0];
  norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
  if(norm != 0.0)
    {
      axis[0] /= norm;
      axis[1] /= norm;
      axis[2] /= norm;
      angle = fabs(M_PI/2.0 - theta);
      generate_rotmat_axis_angle_countercw(axis,angle,rmat1);
    }
  else
    {
      for(i=0;i<3;++i)
	for(j=0;j<3;++j)
	  rmat1[i][j] = 0.0;
      
      for(i=0;i<3;++i)
	rmat1[i][i] = 1.0;
    }
  
  axis[0] = 0.0;
  axis[1] = 0.0;
  axis[2] = 1.0;
  angle = -phi;
  generate_rotmat_axis_angle_countercw(axis,angle,rmat2);
  
  //now do matrix product to form RmatSphereToPatch
  for(i=0;i<3;++i)
    for(j=0;j<3;++j)
      RmatSphereToPatch[i][j] = rmat2[i][0]*rmat1[0][j] + rmat2[i][1]*rmat1[1][j] + rmat2[i][2]*rmat1[2][j];
  
  //and get transpose to go the other way
  for(i=0;i<3;++i)
    for(j=0;j<3;++j)
      RmatPatchToSphere[i][j] = RmatSphereToPatch[j][i];
}

static void rot_tangvectens(double _vec[3], double tvec[2], double ttens[2][2], double Rmat[3][3], double rvec[3], double rtvec[2], double rttens[2][2])
{
  long n,m;
  
  //get rotation info
  double norm_vec = sqrt(_vec[0]*_vec[0] + _vec[1]*_vec[1] + _vec[2]*_vec[2]);
  double vec[3];
  vec[0] = _vec[0]/norm_vec;
  vec[1] = _vec[1]/norm_vec;
  vec[2] = _vec[2]/norm_vec;
  
  rvec[0] = 0.0;
  rvec[1] = 0.0;
  rvec[2] = 0.0;
  for(n=0;n<3;++n)
    for(m=0;m<3;++m)
      rvec[n] += Rmat[n][m]*vec[m];
  
  double p[3];
  p[0] = -vec[1];
  p[1] = vec[0];
  p[2] = 0.0;
  double rephi_vec[3];
  rephi_vec[0] = 0.0;
  rephi_vec[1] = 0.0;
  rephi_vec[2] = 0.0;
  for(n=0;n<3;++n)
    for(m=0;m<3;++m)
      rephi_vec[n] += Rmat[n][m]*p[m];
    
  double ephi_rvec[3];
  ephi_rvec[0] = -rvec[1];
  ephi_rvec[1] = rvec[0];
  ephi_rvec[2] = 0.0;
  
  double etheta_rvec[3];
  etheta_rvec[0] = rvec[2]*rvec[0];
  etheta_rvec[1] = rvec[2]*rvec[1];
  etheta_rvec[2] = -1.0*(rvec[0]*rvec[0] + rvec[1]*rvec[1]);
  
  double norm;
  norm = sqrt((1.0 - rvec[2])*(1.0 + rvec[2])*(1.0 - vec[2])*(1.0 + vec[2]));
  
  double sinpsi,cospsi;
  sinpsi = (rephi_vec[0]*etheta_rvec[0] + rephi_vec[1]*etheta_rvec[1] + rephi_vec[2]*etheta_rvec[2])/norm;
  cospsi = (rephi_vec[0]*ephi_rvec[0] + rephi_vec[1]*ephi_rvec[1] + rephi_vec[2]*ephi_rvec[2])/norm;
  
  rtvec[0] = tvec[0]*cospsi + tvec[1]*sinpsi;
  rtvec[1] = -1.0*tvec[0]*sinpsi + tvec[1]*cospsi;
  
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
      t1[i][j] = ttens[i][0]*r[0][j]  + ttens[i][1]*r[1][j];
  
  for(i=0;i<2;++i)
    for(j=0;j<2;++j)
      rttens[i][j] = rt[i][0]*t1[0][j]  + rt[i][1]*t1[1][j];
}

/*
static double sphdist_haversine(double t1, double p1, double t2, double p2)
{
  double dra = p2-p1;
  double havsinedra = sin(dra/2.0);
  havsinedra = havsinedra*havsinedra;
  
  double ddec = t1-t2;
  double havsineddec = sin(ddec/2.0);
  havsineddec = havsineddec*havsineddec;
  
  double sqrthavdist = sqrt(havsineddec + cos(M_PI/2.0 - t1)*cos(M_PI/2.0 - t2)*havsinedra);
  
  return fabs(2.0*asin(sqrthavdist));
}
*/

/* The following functions compute derivs of the lensing potential.
   Some notes about them.  They  
     1) finite diff at 4th order  
     2) use special asymmetirc stencils at the edges of the patch
   
   Some of the partial derivatives are computed twice because they are needed for real gradients on the sphere.
   In other words, we have
   
   grad_{theta,phi} = partial_{theta,phi}/sin(theta) - cos(theta)/sin(theta)/sin(theta)*partial_{phi}
   grad_{phi,phi} = partial_{phi,phi}/sin(theta)/sin(theta) + cos(theta)/sin(theta)*partial_{theta}
   
   where grad_{} is a gradient defined via deifferential geometry and partial is a partial derivative.  
   
   Finally, there is one function that computes all derivatives at once, but it uses a ton of memory for large grids.
*/

//theta deriv
static void getderiv_mggrid_xtheta(MGGrid u, double *gx)
{
  long Nf = u->N;
  mgfloat *phi = u->grid;
  double df = u->dL;
  long i,j;
  double fac = 1.0/12.0/df;
  
  for(i=0;i<2;++i)
    for(j=0;j<Nf;++j)
      gx[j + Nf*i] = (-25.0*phi[j + Nf*i]
		      +48.0*phi[j + Nf*(i+1)]
		      -36.0*phi[j + Nf*(i+2)]
		      +16.0*phi[j + Nf*(i+3)]
		      - 3.0*phi[j + Nf*(i+4)]
		      )*fac;
  
  for(i=2;i<Nf-2;++i)
    for(j=0;j<Nf;++j)
      gx[j + Nf*i] = (phi[j + Nf*(i-2)]
		      -8.0*phi[j + Nf*(i-1)]
		      +8.0*phi[j + Nf*(i+1)]
		      -phi[j + Nf*(i+2)]
		      )*fac;
  
  for(i=Nf-2;i<Nf;++i)
    for(j=0;j<Nf;++j)
      gx[j + Nf*i] = (25.0*phi[j + Nf*i]
		      -48.0*phi[j + Nf*(i-1)]
		      +36.0*phi[j + Nf*(i-2)]
		      -16.0*phi[j + Nf*(i-3)]
		      + 3.0*phi[j + Nf*(i-4)]
		      )*fac;
  
  /* OLD CODE  
  for(i=0;i<Nf;++i)
    for(j=0;j<Nf;++j)
      {
	xind = i;
	yind = j;
        
	if(xind == 0 || xind == 1)
	  {
	    gx[yind + Nf*xind] =  (-25.0*phi[yind + Nf*xind]
				      +48.0*phi[yind + Nf*(xind+1)]
				      -36.0*phi[yind + Nf*(xind+2)]
				      +16.0*phi[yind + Nf*(xind+3)]
				      - 3.0*phi[yind + Nf*(xind+4)]
				      )/12.0/df;
	  }
	else if(xind == Nf-1 || xind == Nf-2)
	  {
	    gx[yind + Nf*xind] = (25.0*phi[yind + Nf*xind]
				  -48.0*phi[yind + Nf*(xind-1)]
				     +36.0*phi[yind + Nf*(xind-2)]
				     -16.0*phi[yind + Nf*(xind-3)]
				     + 3.0*phi[yind + Nf*(xind-4)]
				     )/12.0/df;
	  }
	else
	  {
	    gx[yind + Nf*xind] = (phi[yind + Nf*(xind-2)]
				     -8.0*phi[yind + Nf*(xind-1)]
				     +8.0*phi[yind + Nf*(xind+1)]
				     -phi[yind + Nf*(xind+2)]
				     )/12.0/df;
	  }
      }
  */
}

//theta-theta deriv
static void getderiv_mggrid_xtheta_xtheta(MGGrid u, double *gxx)
{
  long Nf = u->N;
  mgfloat *phi = u->grid;
  double df = u->dL;
  long i,j;
  double fac = 1.0/12.0/df/df;
  
  for(i=0;i<2;++i)
    for(j=0;j<Nf;++j)
      gxx[j + Nf*i] = (45.0*phi[j + Nf*(i)]
		       -154.0*phi[j + Nf*(i+1)]
		       +214.0*phi[j + Nf*(i+2)]
		       -156.0*phi[j + Nf*(i+3)]
		       + 61.0*phi[j + Nf*(i+4)]
		       - 10.0*phi[j + Nf*(i+5)]
		       )*fac;
  
  for(i=2;i<Nf-2;++i)
    for(j=0;j<Nf;++j)
      gxx[j + Nf*i] = (-1.0*phi[j + Nf*(i-2)]
		       +16.0*phi[j + Nf*(i-1)]
		       -30.0*phi[j + Nf*(i)]
		       +16.0*phi[j + Nf*(i+1)]
		       -1.0*phi[j + Nf*(i+2)]
		       )*fac;
  
  for(i=Nf-2;i<Nf;++i)
    for(j=0;j<Nf;++j)
      gxx[j + Nf*i] = (45.0*phi[j + Nf*(i)]
		       -154.0*phi[j + Nf*(i-1)]
		       +214.0*phi[j + Nf*(i-2)]
		       -156.0*phi[j + Nf*(i-3)]
		       + 61.0*phi[j + Nf*(i-4)]
		       - 10.0*phi[j + Nf*(i-5)]
		       )*fac;
  
  /* OLD CODE
  for(i=0;i<Nf;++i)
  for(j=0;j<Nf;++j)
      {
	xind = i;
	yind = j;
	
	if(xind == 0 || xind == 1)
	  {
            gxx[yind + Nf*xind] = (45.0*phi[yind + Nf*(xind)]
				      -154.0*phi[yind + Nf*(xind+1)]
				      +214.0*phi[yind + Nf*(xind+2)]
				      -156.0*phi[yind + Nf*(xind+3)]
				      + 61.0*phi[yind + Nf*(xind+4)]
				      - 10.0*phi[yind + Nf*(xind+5)]
				      )/12.0/df/df;
	  }
	else if(xind == Nf-1 || xind == Nf-2)
	  {
	    gxx[yind + Nf*xind] = (45.0*phi[yind + Nf*(xind)]
				      -154.0*phi[yind + Nf*(xind-1)]
				      +214.0*phi[yind + Nf*(xind-2)]
				      -156.0*phi[yind + Nf*(xind-3)]
				      + 61.0*phi[yind + Nf*(xind-4)]
				      - 10.0*phi[yind + Nf*(xind-5)]
				      )/12.0/df/df;
	  }
	else
	  {
	    gxx[yind + Nf*xind] = (-1.0*phi[yind + Nf*(xind-2)]
				      +16.0*phi[yind + Nf*(xind-1)]
				      -30.0*phi[yind + Nf*(xind)]
				      +16.0*phi[yind + Nf*(xind+1)]
				      -1.0*phi[yind + Nf*(xind+2)]
				      )/12.0/df/df;
	  }
      }
  */
}

//phi deriv
static void getderiv_mggrid_yphi(MGGrid u, double *gy)
{
  long Nf = u->N;
  mgfloat *phi = u->grid;
  double df = u->dL;
  long i,j;
  double fac = 1.0/12.0/df;
  
#ifdef MGDERIV_METRIC_FAC_AT_END
  for(i=0;i<Nf;++i)
    {
      for(j=0;j<2;++j)
	gy[j + Nf*i] = (-25.0*phi[j + Nf*i]
			+48.0*phi[j+1 + Nf*i]
			-36.0*phi[j+2 + Nf*i]
			+16.0*phi[j+3 + Nf*i]
			- 3.0*phi[j+4 + Nf*i]
			)*fac;
      
      for(j=2;j<Nf-2;++j)
	gy[j + Nf*i] = (phi[j-2 + Nf*i]
			-8.0*phi[j-1 + Nf*i]
			+8.0*phi[j+1 + Nf*i]
			-    phi[j+2 + Nf*i]
			)*fac;
      
      for(j=Nf-2;j<Nf;++j)
	gy[j + Nf*i] = (25.0*phi[j + Nf*i]
			-48.0*phi[j-1 + Nf*i]
			+36.0*phi[j-2 + Nf*i]
			-16.0*phi[j-3 + Nf*i]
			+ 3.0*phi[j-4 + Nf*i]
			)*fac;
    }
#else
  for(i=0;i<Nf;++i)
    {
      for(j=0;j<2;++j)
	gy[j + Nf*i] = (-25.0*phi[j + Nf*i]
			+48.0*phi[j+1 + Nf*i]
			-36.0*phi[j+2 + Nf*i]
			+16.0*phi[j+3 + Nf*i]
			- 3.0*phi[j+4 + Nf*i]
			)*fac/(u->sintheta[i]);
      
      for(j=2;j<Nf-2;++j)
	gy[j + Nf*i] = (phi[j-2 + Nf*i]
			-8.0*phi[j-1 + Nf*i]
			+8.0*phi[j+1 + Nf*i]
			-    phi[j+2 + Nf*i]
			)*fac/(u->sintheta[i]);
      
      for(j=Nf-2;j<Nf;++j)
	gy[j + Nf*i] = (25.0*phi[j + Nf*i]
			-48.0*phi[j-1 + Nf*i]
			+36.0*phi[j-2 + Nf*i]
			-16.0*phi[j-3 + Nf*i]
			+ 3.0*phi[j-4 + Nf*i]
			)*fac/(u->sintheta[i]);
    }
#endif
  
  /* OLD CODE
  for(i=0;i<Nf;++i)
    {
      sintheta = u->sintheta[i]; //sin(i*df + u->thetaLoc);

      for(j=0;j<Nf;++j)
	{
	  xind = i;
	  yind = j;
	  
	  if(yind == 0 || yind == 1)
	    {
	      gy[yind + Nf*xind] =   (-25.0*phi[yind   + Nf*xind]
					 +48.0*phi[yind+1 + Nf*xind]
					 -36.0*phi[yind+2 + Nf*xind]
					 +16.0*phi[yind+3 + Nf*xind]
					 - 3.0*phi[yind+4 + Nf*xind]
					 )/12.0/df;
	    }
	  else if(yind == Nf-1 || yind == Nf-2)
	    {
	      gy[yind + Nf*xind] =   (25.0*phi[yind   + Nf*xind]
					 -48.0*phi[yind-1 + Nf*xind]
					 +36.0*phi[yind-2 + Nf*xind]
					 -16.0*phi[yind-3 + Nf*xind]
					 + 3.0*phi[yind-4 + Nf*xind]
					 )/12.0/df;
	    }
	  else
	    {
	      gy[yind + Nf*xind] =   (phi[yind-2 + Nf*xind]
					 -8.0*phi[yind-1 + Nf*xind]
					 +8.0*phi[yind+1 + Nf*xind]
					 -    phi[yind+2 + Nf*xind]
					 )/12.0/df;
	    }
	  
	  //metric factor
	  gy[yind + Nf*xind] /= sintheta;
	}
    }
  */
}

//phi-phi deriv
static void getderiv_mggrid_yphi_yphi(MGGrid u, double *gyy)
{
  long Nf = u->N;
  mgfloat *phi = u->grid;
  double df = u->dL;
  long i,j;
  double fac = 1.0/12.0/df/df;
#ifndef MGDERIV_METRIC_FAC_AT_END
  double s2,cs,gx;
#endif
  
  for(i=0;i<Nf;++i)
    {
#ifdef MGDERIV_METRIC_FAC_AT_END
      for(j=0;j<2;++j)
	gyy[j + Nf*i] = (45.0*phi[j + Nf*i]
			 -154.0*phi[j+1 + Nf*i]
			 +214.0*phi[j+2 + Nf*i]
			 -156.0*phi[j+3 + Nf*i]
			 + 61.0*phi[j+4 + Nf*i]
			 - 10.0*phi[j+5 + Nf*i]
			 )*fac;
      
      for(j=2;j<Nf-2;++j)
	gyy[j + Nf*i] = (-1.00*phi[j-2 + Nf*i]
			 +16.0*phi[j-1 + Nf*i]
			 -30.0*phi[j   + Nf*i]
			 +16.0*phi[j+1 + Nf*i]
			 -1.00*phi[j+2 + Nf*i]
			 )*fac;
      
      for(j=Nf-2;j<Nf;++j)
	gyy[j + Nf*i] = (45.0*phi[j + Nf*i]
			 -154.0*phi[j-1 + Nf*i]
			 +214.0*phi[j-2 + Nf*i]
			 -156.0*phi[j-3 + Nf*i]
			 + 61.0*phi[j-4 + Nf*i]
			 - 10.0*phi[j-5 + Nf*i]
			 )*fac;
#else
      s2 = (u->sintheta[i])*(u->sintheta[i]);
      
      for(j=0;j<2;++j)
	gyy[j + Nf*i] = (45.0*phi[j + Nf*i]
			 -154.0*phi[j+1 + Nf*i]
			 +214.0*phi[j+2 + Nf*i]
			 -156.0*phi[j+3 + Nf*i]
			 + 61.0*phi[j+4 + Nf*i]
			 - 10.0*phi[j+5 + Nf*i]
			 )*fac/s2;
      
      for(j=2;j<Nf-2;++j)
	gyy[j + Nf*i] = (-1.00*phi[j-2 + Nf*i]
			 +16.0*phi[j-1 + Nf*i]
			 -30.0*phi[j   + Nf*i]
			 +16.0*phi[j+1 + Nf*i]
			 -1.00*phi[j+2 + Nf*i]
			 )*fac/s2;
      
      for(j=Nf-2;j<Nf;++j)
	gyy[j + Nf*i] = (45.0*phi[j + Nf*i]
			 -154.0*phi[j-1 + Nf*i]
			 +214.0*phi[j-2 + Nf*i]
			 -156.0*phi[j-3 + Nf*i]
			 + 61.0*phi[j-4 + Nf*i]
			 - 10.0*phi[j-5 + Nf*i]
			 )*fac/s2;
#endif
    }
  
#ifndef MGDERIV_METRIC_FAC_AT_END
  //comp gx factor
  fac = 1.0/12.0/df;
  for(i=0;i<2;++i)
    {
      cs = (u->costheta[i])/(u->sintheta[i]);
      
      for(j=0;j<Nf;++j)
	{
	  gx = (-25.0*phi[j + Nf*i]
		+48.0*phi[j + Nf*(i+1)]
		-36.0*phi[j + Nf*(i+2)]
		+16.0*phi[j + Nf*(i+3)]
		- 3.0*phi[j + Nf*(i+4)]
		)*fac;
	  gyy[j + Nf*i] += cs*gx;
	}
    }
  
  for(i=2;i<Nf-2;++i)
    {
      cs = (u->costheta[i])/(u->sintheta[i]);
      
      for(j=0;j<Nf;++j)
	{
	  gx = (phi[j + Nf*(i-2)]
		-8.0*phi[j + Nf*(i-1)]
		+8.0*phi[j + Nf*(i+1)]
		-phi[j + Nf*(i+2)]
		)*fac;
	  gyy[j + Nf*i] += cs*gx;
	}
    }
  
  for(i=Nf-2;i<Nf;++i)
    {
      cs = (u->costheta[i])/(u->sintheta[i]);
      
      for(j=0;j<Nf;++j)
	{
	  gx = (25.0*phi[j + Nf*i]
		-48.0*phi[j + Nf*(i-1)]
		+36.0*phi[j + Nf*(i-2)]
		-16.0*phi[j + Nf*(i-3)]
		+ 3.0*phi[j + Nf*(i-4)]
		)*fac;
	  gyy[j + Nf*i] += cs*gx;
	}
    }
#endif
  
  /* OLD CODE
  for(i=0;i<Nf;++i)
    {
      sintheta = u->sintheta[i]; //sin(i*df + u->thetaLoc);
      costheta = u->costheta[i]; //cos(i*df + u->thetaLoc);
      
      for(j=0;j<Nf;++j)
	{
	  xind = i;
	  yind = j;
	  
	  //partial phi-phi
	  if(yind == 0 || yind == 1)
	    {
	      gyy[yind + Nf*xind] = (45.0*phi[yind   + Nf*xind]
					-154.0*phi[yind+1 + Nf*xind]
					+214.0*phi[yind+2 + Nf*xind]
					-156.0*phi[yind+3 + Nf*xind]
					+ 61.0*phi[yind+4 + Nf*xind]
					- 10.0*phi[yind+5 + Nf*xind]
					)/12.0/df/df;
	    }
	  else if(yind == Nf-1 || yind == Nf-2)
	    {
	      gyy[yind + Nf*xind] = (45.0*phi[yind   + Nf*xind]
					-154.0*phi[yind-1 + Nf*xind]
					+214.0*phi[yind-2 + Nf*xind]
					-156.0*phi[yind-3 + Nf*xind]
					+ 61.0*phi[yind-4 + Nf*xind]
					- 10.0*phi[yind-5 + Nf*xind]
					)/12.0/df/df;
	    }
	  else
	    {
	      gyy[yind + Nf*xind] = (-1.00*phi[yind-2 + Nf*xind]
					+16.0*phi[yind-1 + Nf*xind]
					-30.0*phi[yind   + Nf*xind]
					+16.0*phi[yind+1 + Nf*xind]
					-1.00*phi[yind+2 + Nf*xind]
					)/12.0/df/df;
	    }
	  
	  //metric factors
	  gyy[yind + Nf*xind] /= sintheta;
	  gyy[yind + Nf*xind] /= sintheta;
	  
	  //comp gx factor
	  if(xind == 0 || xind == 1)
	    {
	      gx =  (-25.0*phi[yind + Nf*xind]
		     +48.0*phi[yind + Nf*(xind+1)]
		     -36.0*phi[yind + Nf*(xind+2)]
		     +16.0*phi[yind + Nf*(xind+3)]
		     - 3.0*phi[yind + Nf*(xind+4)]
		     )/12.0/df;
	    }
	  else if(xind == Nf-1 || xind == Nf-2)
	    {
	      gx = (25.0*phi[yind + Nf*xind]
		    -48.0*phi[yind + Nf*(xind-1)]
		    +36.0*phi[yind + Nf*(xind-2)]
		    -16.0*phi[yind + Nf*(xind-3)]
		    + 3.0*phi[yind + Nf*(xind-4)]
		    )/12.0/df;
	    }
	  else
	    {
	      gx = (phi[yind + Nf*(xind-2)]
		    -8.0*phi[yind + Nf*(xind-1)]
		    +8.0*phi[yind + Nf*(xind+1)]
		    -phi[yind + Nf*(xind+2)]
		    )/12.0/df;
	    }
	  
	  gyy[yind + Nf*xind] += costheta/sintheta*gx;
	}
    }
  */
}

//theta-phi deriv
static void getderiv_mggrid_xtheta_yphi(MGGrid u, double *gxy)
{
  long Nf = u->N;
  mgfloat *phi = u->grid;
  double df = u->dL;
  long i,j,n,nstart,nend;
  double yderivs[5];
  double cs;
  double fac = 1.0/12.0/df;
#ifndef MGDERIV_METRIC_FAC_AT_END
  double gy;
#endif
  
  //now save partial phi and comp partial x
  for(i=0;i<2;++i)
    {
      nstart = i;
      nend = i + 4;
      
      cs = (u->costheta[i])/(u->sintheta[i]);
      
      for(j=0;j<2;++j)
	{
	  //do y derivs
	  for(n=nstart;n<=nend;++n)
	    yderivs[n-nstart] = (-25.0*phi[j   + Nf*n]
				 +48.0*phi[j+1 + Nf*n]
				 -36.0*phi[j+2 + Nf*n]
				 +16.0*phi[j+3 + Nf*n]
				 - 3.0*phi[j+4 + Nf*n]
				 )*fac;
	  
	  gxy[j + Nf*i] = (-25.0*yderivs[0]
			   +48.0*yderivs[1]
			   -36.0*yderivs[2]
			   +16.0*yderivs[3]
			   - 3.0*yderivs[4]
			   )*fac;
	  
#ifndef MGDERIV_METRIC_FAC_AT_END
	  //metric factors
	  gy = yderivs[0]/(u->sintheta[i]);
	  gxy[j + Nf*i] /= u->sintheta[i];
          gxy[j + Nf*i] -= cs*gy;
#endif
	}
      
      for(j=2;j<Nf-2;++j)
	{
	  //do y derivs
	  for(n=nstart;n<=nend;++n)
	    yderivs[n-nstart] = (phi[j-2 + Nf*n]
				 -8.0*phi[j-1 + Nf*n]
				 +8.0*phi[j+1 + Nf*n]
				 -phi[j+2 + Nf*n]
				 )*fac;
	  	  
	  gxy[j + Nf*i] = (-25.0*yderivs[0]
			   +48.0*yderivs[1]
			   -36.0*yderivs[2]
			   +16.0*yderivs[3]
			   - 3.0*yderivs[4]
			   )*fac;

#ifndef MGDERIV_METRIC_FAC_AT_END	  
	  //metric factors
	  gy = yderivs[0]/(u->sintheta[i]);
	  gxy[j + Nf*i] /= u->sintheta[i];
          gxy[j + Nf*i] -= cs*gy;
#endif
	}
      
      for(j=Nf-2;j<Nf;++j)
	{
	  //do y derivs
	  for(n=nstart;n<=nend;++n)
	    yderivs[n-nstart] = (25.0*phi[j   + Nf*n]
				 -48.0*phi[j-1 + Nf*n]
				 +36.0*phi[j-2 + Nf*n]
				 -16.0*phi[j-3 + Nf*n]
				 + 3.0*phi[j-4 + Nf*n]
				 )*fac;
	  
	  gxy[j + Nf*i] = (-25.0*yderivs[0]
			   +48.0*yderivs[1]
			   -36.0*yderivs[2]
			   +16.0*yderivs[3]
			   - 3.0*yderivs[4]
			   )*fac;

#ifndef MGDERIV_METRIC_FAC_AT_END	  
	  //metric factors
	  gy = yderivs[0]/(u->sintheta[i]);
	  gxy[j + Nf*i] /= u->sintheta[i];
          gxy[j + Nf*i] -= cs*gy;
#endif
	}
    }
    
  for(i=2;i<Nf-2;++i)
    {
      nstart = i - 2;
      nend = i + 2;
      
      cs = (u->costheta[i])/(u->sintheta[i]);
      
      for(j=0;j<2;++j)
        {
          //do y derivs
          for(n=nstart;n<=nend;++n)
            yderivs[n-nstart] = (-25.0*phi[j   + Nf*n]
                                 +48.0*phi[j+1 + Nf*n]
                                 -36.0*phi[j+2 + Nf*n]
                                 +16.0*phi[j+3 + Nf*n]
                                 - 3.0*phi[j+4 + Nf*n]
                                 )*fac;
	  
	  gxy[j + Nf*i] =  (yderivs[0]
			    -8.0*yderivs[1]
			    +8.0*yderivs[3]
			    -yderivs[4]
			    )*fac;

#ifndef MGDERIV_METRIC_FAC_AT_END	  
	  //metric factors
	  gy = yderivs[2]/(u->sintheta[i]);
	  gxy[j + Nf*i] /= u->sintheta[i];
          gxy[j + Nf*i] -= cs*gy;
#endif
        }
      
      for(j=2;j<Nf-2;++j)
        {
          //do y derivs
          for(n=nstart;n<=nend;++n)
            yderivs[n-nstart] = (phi[j-2 + Nf*n]
                                 -8.0*phi[j-1 + Nf*n]
                                 +8.0*phi[j+1 + Nf*n]
                                 -phi[j+2 + Nf*n]
                                 )*fac;
                  
	  gxy[j + Nf*i] = (yderivs[0]
			   -8.0*yderivs[1]
			   +8.0*yderivs[3]
			   -yderivs[4]
			   )*fac;

#ifndef MGDERIV_METRIC_FAC_AT_END
	  //metric factors
	  gy = yderivs[2]/(u->sintheta[i]);
	  gxy[j + Nf*i] /= u->sintheta[i];
          gxy[j + Nf*i] -= cs*gy;
#endif
        }
      
      for(j=Nf-2;j<Nf;++j)
        {
          //do y derivs
          for(n=nstart;n<=nend;++n)
            yderivs[n-nstart] = (25.0*phi[j   + Nf*n]
                                 -48.0*phi[j-1 + Nf*n]
                                 +36.0*phi[j-2 + Nf*n]
                                 -16.0*phi[j-3 + Nf*n]
                                 + 3.0*phi[j-4 + Nf*n]
                                 )*fac;
          
	  gxy[j + Nf*i] = (yderivs[0]
			   -8.0*yderivs[1]
			   +8.0*yderivs[3]
			   -yderivs[4]
			   )*fac;

#ifndef MGDERIV_METRIC_FAC_AT_END	  
	  //metric factors
	  gy = yderivs[2]/(u->sintheta[i]);
	  gxy[j + Nf*i] /= u->sintheta[i];
          gxy[j + Nf*i] -= cs*gy;
#endif
        }
    }
  
  for(i=Nf-2;i<Nf;++i)
    {
      nstart = i-4;
      nend = i;
      
      cs = (u->costheta[i])/(u->sintheta[i]);
      
      for(j=0;j<2;++j)
        {
          //do y derivs
          for(n=nstart;n<=nend;++n)
            yderivs[n-nstart] = (-25.0*phi[j   + Nf*n]
                                 +48.0*phi[j+1 + Nf*n]
                                 -36.0*phi[j+2 + Nf*n]
                                 +16.0*phi[j+3 + Nf*n]
                                 - 3.0*phi[j+4 + Nf*n]
                                 )*fac;
          
	  gxy[j + Nf*i] = (25.0*yderivs[4]
			   -48.0*yderivs[3]
			   +36.0*yderivs[2]
			   -16.0*yderivs[1]
			   + 3.0*yderivs[0]
			   )*fac;

#ifndef MGDERIV_METRIC_FAC_AT_END	  
	  //metric factors
	  gy = yderivs[4]/(u->sintheta[i]);
	  gxy[j + Nf*i] /= u->sintheta[i];
          gxy[j + Nf*i] -= cs*gy;
#endif
        }
      
      for(j=2;j<Nf-2;++j)
        {
          //do y derivs
          for(n=nstart;n<=nend;++n)
            yderivs[n-nstart] = (phi[j-2 + Nf*n]
                                 -8.0*phi[j-1 + Nf*n]
                                 +8.0*phi[j+1 + Nf*n]
                                 -phi[j+2 + Nf*n]
                                 )*fac;
                  
	  gxy[j + Nf*i] = (25.0*yderivs[4]
			   -48.0*yderivs[3]
			   +36.0*yderivs[2]
			   -16.0*yderivs[1]
			   + 3.0*yderivs[0]
			   )*fac;

#ifndef MGDERIV_METRIC_FAC_AT_END	  
	  //metric factors
	  gy = yderivs[4]/(u->sintheta[i]);
	  gxy[j + Nf*i] /= u->sintheta[i];
          gxy[j + Nf*i] -= cs*gy;
#endif
        }
      
      for(j=Nf-2;j<Nf;++j)
        {
          //do y derivs
          for(n=nstart;n<=nend;++n)
            yderivs[n-nstart] = (25.0*phi[j   + Nf*n]
                                 -48.0*phi[j-1 + Nf*n]
                                 +36.0*phi[j-2 + Nf*n]
                                 -16.0*phi[j-3 + Nf*n]
                                 + 3.0*phi[j-4 + Nf*n]
                                 )*fac;
          
	  gxy[j + Nf*i] = (25.0*yderivs[4]
			   -48.0*yderivs[3]
			   +36.0*yderivs[2]
			   -16.0*yderivs[1]
			   + 3.0*yderivs[0]
			   )*fac;

#ifndef MGDERIV_METRIC_FAC_AT_END	  
	  //metric factors
	  gy = yderivs[4]/(u->sintheta[i]);
	  gxy[j + Nf*i] /= u->sintheta[i];
          gxy[j + Nf*i] -= cs*gy;
#endif
        }
    }
  
  /* OLD CODE
  //now save partial phi and comp partial x
  for(i=0;i<Nf;++i)
    {
      sintheta = u->sintheta[i]; //sin(i*df + u->thetaLoc);
      costheta = u->sintheta[i]; //cos(i*df + u->thetaLoc);
      
      for(j=0;j<Nf;++j)
	{
	  xind = i;
	  yind = j;
	  
	  //save partial y and add metric factor
	  if(yind == 0 || yind == 1)
	    {
	      gy =   (-25.0*phi[yind   + Nf*xind]
		      +48.0*phi[yind+1 + Nf*xind]
		      -36.0*phi[yind+2 + Nf*xind]
		      +16.0*phi[yind+3 + Nf*xind]
		      - 3.0*phi[yind+4 + Nf*xind]
		      )/12.0/df;
	    }
	  else if(yind == Nf-1 || yind == Nf-2)
	    {
	      gy =   (25.0*phi[yind   + Nf*xind]
		      -48.0*phi[yind-1 + Nf*xind]
		      +36.0*phi[yind-2 + Nf*xind]
		      -16.0*phi[yind-3 + Nf*xind]
		      + 3.0*phi[yind-4 + Nf*xind]
		      )/12.0/df;
	    }
	  else
	    {
	      gy =   (phi[yind-2 + Nf*xind]
		      -8.0*phi[yind-1 + Nf*xind]
		      +8.0*phi[yind+1 + Nf*xind]
		      -phi[yind+2 + Nf*xind]
		      )/12.0/df;
	    }
	  gy /= sintheta;
	  
	  //do parital in y directions
	  if(xind == 0 || xind == 1)
	    {
	      nstart = xind;
	      nend = xind + 4;
	    }
	  else if(xind == Nf-1 || xind == Nf-2)
	    {
	      nstart = xind-4;
	      nend = xind;
	    }
	  else
	    {
	      nstart = xind - 2;
	      nend = xind + 2;
	    }
	  
	  for(n=nstart;n<=nend;++n)
	    {
	      nxind = n;
	      
	      if(yind == 0 || yind == 1)
		{
		  yderivs[n-nstart] =   (-25.0*phi[yind   + Nf*nxind]
					 +48.0*phi[yind+1 + Nf*nxind]
					 -36.0*phi[yind+2 + Nf*nxind]
					 +16.0*phi[yind+3 + Nf*nxind]
					 - 3.0*phi[yind+4 + Nf*nxind]
					 )/12.0/df;
		}
	      else if(yind == Nf-1 || yind == Nf-2)
		{
		  yderivs[n-nstart] =   (25.0*phi[yind   + Nf*nxind]
					 -48.0*phi[yind-1 + Nf*nxind]
					 +36.0*phi[yind-2 + Nf*nxind]
					 -16.0*phi[yind-3 + Nf*nxind]
					 + 3.0*phi[yind-4 + Nf*nxind]
					 )/12.0/df;
		}
	      else
		{
		  yderivs[n-nstart] =   (phi[yind-2 + Nf*nxind]
					 -8.0*phi[yind-1 + Nf*nxind]
					 +8.0*phi[yind+1 + Nf*nxind]
					 -phi[yind+2 + Nf*nxind]
					 )/12.0/df;
		}
	    }
	  
	  //now do partials in x direction
	  if(xind == 0 || xind == 1)
	    {
	      gxy[yind + Nf*xind] =   (-25.0*yderivs[0]
					  +48.0*yderivs[1]
					  -36.0*yderivs[2]
					  +16.0*yderivs[3]
					  - 3.0*yderivs[4]
					  )/12.0/df;
	    }
	  else if(xind == Nf-1 || xind == Nf-2)
	    {
	      gxy[yind + Nf*xind] =    (25.0*yderivs[4]
					   -48.0*yderivs[3]
					   +36.0*yderivs[2]
					   -16.0*yderivs[1]
					   + 3.0*yderivs[0]
					   )/12.0/df;
	    }
	  else
	    {
	      gxy[yind + Nf*xind] =   (yderivs[0]
					  -8.0*yderivs[1]
					  +8.0*yderivs[3]
					  -yderivs[4]
					  )/12.0/df;
	    }
	  
	  //now do final metric factors
	  gxy[yind + Nf*xind] /= sintheta;
          gxy[yind + Nf*xind] -= costheta/sintheta*gy;
	}
    }
  */
}

/*
  1) computes all derivatives needed (i.e. x-, y-, xx-, yy-, and xy-derivative) by finite diff at 4th order  
  2) uses special asymmetirc stencils at the edges of the patch
*/
static void getderiv_mggrid(MGGrid u, double **gx, double **gy, double **gxx, double **gxy, double **gyy)
{
  long Nf = u->N;
  mgfloat *phi = u->grid;
  double df = u->dL;
  long i,j,xind,yind;
  
  (*gx) = (double*)malloc(sizeof(double)*Nf*Nf);
  assert((*gx) != NULL);
  (*gy) = (double*)malloc(sizeof(double)*Nf*Nf);
  assert((*gy) != NULL);
  (*gxx) = (double*)malloc(sizeof(double)*Nf*Nf);
  assert((*gxx) != NULL);
  (*gxy) = (double*)malloc(sizeof(double)*Nf*Nf);
  assert((*gxy) != NULL);
  (*gyy) = (double*)malloc(sizeof(double)*Nf*Nf);
  assert((*gyy) != NULL);
  
  /* compute derivs of lensing potential */
  for(i=0;i<Nf;++i)
    for(j=0;j<Nf;++j)
      {
	xind = i;
	yind = j;
        
	if(xind == 0 || xind == 1)
	  {
	    (*gx)[yind + Nf*xind] =  (-25.0*phi[yind + Nf*xind]
				      +48.0*phi[yind + Nf*(xind+1)]
				      -36.0*phi[yind + Nf*(xind+2)]
				      +16.0*phi[yind + Nf*(xind+3)]
				      - 3.0*phi[yind + Nf*(xind+4)]
				      )/12.0/df;
	    (*gxx)[yind + Nf*xind] = ( 45.0*phi[yind + Nf*(xind)]
				       -154.0*phi[yind + Nf*(xind+1)]
				       +214.0*phi[yind + Nf*(xind+2)]
				       -156.0*phi[yind + Nf*(xind+3)]
				       + 61.0*phi[yind + Nf*(xind+4)]
				       - 10.0*phi[yind + Nf*(xind+5)]
				       )/12.0/df/df;
	  }
	else if(xind == Nf-1 || xind == Nf-2)
	  {
	    (*gx)[yind + Nf*xind] = ( 25.0*phi[yind + Nf*xind]
				      -48.0*phi[yind + Nf*(xind-1)]
				      +36.0*phi[yind + Nf*(xind-2)]
				      -16.0*phi[yind + Nf*(xind-3)]
				      + 3.0*phi[yind + Nf*(xind-4)]
				      )/12.0/df;
	    (*gxx)[yind + Nf*xind] = ( 45.0*phi[yind + Nf*(xind)]
				       -154.0*phi[yind + Nf*(xind-1)]
				       +214.0*phi[yind + Nf*(xind-2)]
				       -156.0*phi[yind + Nf*(xind-3)]
				       + 61.0*phi[yind + Nf*(xind-4)]
				       - 10.0*phi[yind + Nf*(xind-5)]
				       )/12.0/df/df;
	  }
	else
	  {
	    (*gx)[yind + Nf*xind] = (      phi[yind + Nf*(xind-2)]
					   -8.0*phi[yind + Nf*(xind-1)]
					   +8.0*phi[yind + Nf*(xind+1)]
					   -phi[yind + Nf*(xind+2)]
					   )/12.0/df;
	    (*gxx)[yind + Nf*xind] = (-1.0*phi[yind + Nf*(xind-2)]
				      +16.0*phi[yind + Nf*(xind-1)]
				      -30.0*phi[yind + Nf*(xind)]
				      +16.0*phi[yind + Nf*(xind+1)]
				      -1.00*phi[yind + Nf*(xind+2)]
				      )/12.0/df/df;
	  }
        
	if(yind == 0 || yind == 1)
	  {
	    (*gy)[yind + Nf*xind] =   (-25.0*phi[yind   + Nf*xind]
                                       +48.0*phi[yind+1 + Nf*xind]
                                       -36.0*phi[yind+2 + Nf*xind]
                                       +16.0*phi[yind+3 + Nf*xind]
                                       - 3.0*phi[yind+4 + Nf*xind]
				       )/12.0/df;
	    (*gyy)[yind + Nf*xind] = (  45.0*phi[yind   + Nf*xind]
					-154.0*phi[yind+1 + Nf*xind]
					+214.0*phi[yind+2 + Nf*xind]
					-156.0*phi[yind+3 + Nf*xind]
					+ 61.0*phi[yind+4 + Nf*xind]
					- 10.0*phi[yind+5 + Nf*xind]
					)/12.0/df/df;
	  }
	else if(yind == Nf-1 || yind == Nf-2)
	  {
	    (*gy)[yind + Nf*xind] =   ( 25.0*phi[yind   + Nf*xind]
					-48.0*phi[yind-1 + Nf*xind]
					+36.0*phi[yind-2 + Nf*xind]
					-16.0*phi[yind-3 + Nf*xind]
					+ 3.0*phi[yind-4 + Nf*xind]
					)/12.0/df;
	    (*gyy)[yind + Nf*xind] = (  45.0*phi[yind   + Nf*xind]
					-154.0*phi[yind-1 + Nf*xind]
					+214.0*phi[yind-2 + Nf*xind]
					-156.0*phi[yind-3 + Nf*xind]
					+ 61.0*phi[yind-4 + Nf*xind]
					- 10.0*phi[yind-5 + Nf*xind]
					)/12.0/df/df;
	  }
	else
	  {
	    (*gy)[yind + Nf*xind] =   (     phi[yind-2 + Nf*xind]
					    -8.0*phi[yind-1 + Nf*xind]
					    +8.0*phi[yind+1 + Nf*xind]
					    -    phi[yind+2 + Nf*xind]
					    )/12.0/df;
	    (*gyy)[yind + Nf*xind] = (-1.00*phi[yind-2 + Nf*xind]
				      +16.0*phi[yind-1 + Nf*xind]
				      -30.0*phi[yind   + Nf*xind]
				      +16.0*phi[yind+1 + Nf*xind]
				      -1.00*phi[yind+2 + Nf*xind]
				      )/12.0/df/df;
	  }
      }
  
  for(i=0;i<Nf;++i)
    for(j=0;j<Nf;++j)
      {
	xind = i;
	yind = j;
	
	if(xind == 0 || xind == 1)
	  {
	    (*gxy)[yind + Nf*xind] =   (-25.0*(*gy)[yind + Nf*xind]
					+48.0*(*gy)[yind + Nf*(xind+1)]
					-36.0*(*gy)[yind + Nf*(xind+2)]
					+16.0*(*gy)[yind + Nf*(xind+3)]
					- 3.0*(*gy)[yind + Nf*(xind+4)]
					)/12.0/df;
	  }
	else if(xind == Nf-1 || xind == Nf-2)
	  {
	    (*gxy)[yind + Nf*xind] =    (  25.0*(*gy)[yind + Nf*xind]
					   -48.0*(*gy)[yind + Nf*(xind-1)]
					   +36.0*(*gy)[yind + Nf*(xind-2)]
					   -16.0*(*gy)[yind + Nf*(xind-3)]
					   + 3.0*(*gy)[yind + Nf*(xind-4)]
					   )/12.0/df;
	  }
	else
	  {
	    (*gxy)[yind + Nf*xind] =   ((*gy)[yind + Nf*(xind-2)]
					-8.0*(*gy)[yind + Nf*(xind-1)]
					+8.0*(*gy)[yind + Nf*(xind+1)]
					-    (*gy)[yind + Nf*(xind+2)]
					)/12.0/df;
	  }
      }
  
  //do final comp with metric factors
  double theta,sintheta,costheta;
  for(i=0;i<Nf;++i)
    {
      theta = i*df + u->thetaLoc;
      sintheta = sin(theta);
      costheta = cos(theta);
            
      for(j=0;j<Nf;++j)
	{
	  xind = i;
	  yind = j;
	  
	  (*gy)[yind + Nf*xind] /= sintheta;
	  
	  (*gxy)[yind + Nf*xind] /= sintheta;
	  (*gxy)[yind + Nf*xind] -= costheta/sintheta*(*gy)[yind + Nf*xind];
	  
	  (*gyy)[yind + Nf*xind] /= sintheta;
	  (*gyy)[yind + Nf*xind] /= sintheta;
	  (*gyy)[yind + Nf*xind] += costheta/sintheta*(*gx)[yind + Nf*xind];
	}
    }
}
