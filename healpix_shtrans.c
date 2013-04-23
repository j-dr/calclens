/*
  utils for MPI healpix spherical harmonic transforms
  - I have used parts of the C++ and fortran 90 routines and written a lot of my own code as well.
  - the HEALPix copyright has been retained for the parts of the code from HEALPix
  
  -Matthew R. Becker, UofC 2010
*/

/*
  !-----------------------------------------------------------------------------
  !
  !  Copyright (C) 1997-2008 Krzysztof M. Gorski, Eric Hivon, 
  !                          Benjamin D. Wandelt, Anthony J. Banday, 
  !                          Matthias Bartelmann, Hans K. Eriksen, 
  !                          Frode K. Hansen, Martin Reinecke
  !
  !
  !  This file is part of HEALPix.
  !
  !  HEALPix is free software; you can redistribute it and/or modify
  !  it under the terms of the GNU General Public License as published by
  !  the Free Software Foundation; either version 2 of the License, or
  !  (at your option) any later version.
  !
  !  HEALPix is distributed in the hope that it will be useful,
  !  but WITHOUT ANY WARRANTY; without even the implied warranty of
  !  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  !  GNU General Public License for more details.
  !
  !  You should have received a copy of the GNU General Public License
  !  along with HEALPix; if not, write to the Free Software
  !  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
  !
  !  For more information about HEALPix see http://healpix.jpl.nasa.gov
  !
  !-----------------------------------------------------------------------------
  ! Written by Hans Kristian Eriksen and Snorre Boasson, 
  ! but copying large parts from the serial HEALPix code.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <fftw3.h>
#include <gsl/gsl_math.h>
#include <mpi.h>
#include <fitsio.h>

#include "healpix_utils.h"
#include "healpix_shtrans.h"

HEALPixSHTPlan healpixsht_plan(long order)
{
  int NTasks,ThisTask;
  HEALPixSHTPlan plan;
  long Nmapvec,nring,firstRing,lastRing,lastRingLoop,sstartpix,nstartpix,shifted,ringpix,NringsThisTask,Npix;
  double ncostheta,nsintheta;
  long firstM,lastM,m;
  
  MPI_Comm_size(MPI_COMM_WORLD,&NTasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&ThisTask);
  
#ifndef STATIC_LOADBAL_SHT
  if(ThisTask == 0)
    fprintf(stderr,"using adaptive load balancing functions for SHT!\n");
#endif

  /* rings of a healpix map are split into matching north and south pairs and the set of pairs is distributed over the nodes
     - the rest of the work done here is just to get the proper load-balance and some indexing so we know what is where*/
  plan.order = order;
  plan.firstRingTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(plan.firstRingTasks != NULL);
  plan.lastRingTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(plan.lastRingTasks != NULL);
  get_ringrange_map2alm_healpix_mpi(NTasks,plan.firstRingTasks,plan.lastRingTasks,order);

  Npix = order2npix(order);
  firstRing = plan.firstRingTasks[ThisTask];
  lastRing = plan.lastRingTasks[ThisTask];
  lastRingLoop = lastRing;
  if(lastRing == 2*order2nside(order))
    --lastRingLoop;
  NringsThisTask = lastRing - firstRing + 1;
  plan.northStartIndMapvec = (long*)malloc(sizeof(long)*NringsThisTask);
  assert(plan.northStartIndMapvec != NULL);
  plan.southStartIndMapvec = (long*)malloc(sizeof(long)*NringsThisTask);
  assert(plan.southStartIndMapvec != NULL);
  plan.northStartIndGlobalMap = (long*)malloc(sizeof(long)*NringsThisTask);
  assert(plan.northStartIndGlobalMap != NULL);
  plan.southStartIndGlobalMap = (long*)malloc(sizeof(long)*NringsThisTask);
  assert(plan.southStartIndGlobalMap != NULL);
  Nmapvec = 0;
  for(nring=firstRing;nring<=lastRingLoop;++nring)
    {
      get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,order);
      sstartpix = Npix - nstartpix - ringpix;
      
      plan.northStartIndMapvec[nring-firstRing] = Nmapvec;
      plan.northStartIndGlobalMap[nring-firstRing] = nstartpix;
      Nmapvec += (ringpix/2+1);
      
      plan.southStartIndMapvec[nring-firstRing] = Nmapvec;
      plan.southStartIndGlobalMap[nring-firstRing] = sstartpix;
      Nmapvec += (ringpix/2+1);
    }
  if(lastRing == 2*order2nside(order))
    {
      get_ring_info2(lastRing,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,order);
      
      plan.northStartIndMapvec[lastRing-firstRing] = Nmapvec;
      plan.northStartIndGlobalMap[lastRing-firstRing] = nstartpix;
      Nmapvec += (ringpix/2+1);
      
      plan.southStartIndMapvec[lastRing-firstRing] = -1;
      plan.southStartIndGlobalMap[lastRing-firstRing] = -1;
    }
  plan.Nmapvec = Nmapvec;
  
  /* the alms are ordered so that all l values which have the same m are stored contiguously
   - the m values are then split over the nodes to get load balance correct
   - the storage order is related to how the plm's are generated (recursions at fixed m) and cache usage*/
  plan.lmax = order2lmax(order);
  plan.firstMTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(plan.firstMTasks != NULL);
  plan.lastMTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(plan.lastMTasks != NULL);
  get_mrange_alm2map_healpix_mpi(NTasks,plan.firstMTasks,plan.lastMTasks,plan.order);
  firstM = plan.firstMTasks[ThisTask];
  lastM = plan.lastMTasks[ThisTask];
  plan.Nlm = 0;
  for(m=firstM;m<=lastM;++m)
    plan.Nlm += (plan.lmax-m+1);
  
  plan.ring_weights = NULL;
  plan.window_function = NULL;
  
  //get rings for this task
  long i,j,good=1;
  for(j=0;j<NTasks;++j)
    if(plan.firstRingTasks[j] == -1 || plan.lastRingTasks[j] == -1 || plan.firstMTasks[j] == -1 || plan.lastMTasks[j] == -1)
      good = 0;
  
  if(ThisTask == 0 && good == 0)
    {
      fprintf(stderr,"problem with assigning rings/ms to tasks! (Did you ask for more tasks than rings?)\n");
      fflush(stderr);
      for(i=0;i<NTasks;++i)
        {
	  fprintf(stderr,"\t%ld: firstRing,lastRing,Nrings = %ld|%ld|%ld, firstM,lastM,NM = %ld|%ld|%ld\n",i,
		  plan.firstRingTasks[i],plan.lastRingTasks[i],2*order2nside(order),
		  plan.firstMTasks[i],plan.lastMTasks[i],plan.lmax+1);
	  fflush(stderr);
	}
      MPI_Abort(MPI_COMM_WORLD,123);
    }
  
  return plan;
}

void healpixsht_destroy_internaldata(void)
{
  destroy_mrange_alm2map_healpix_mpi();
  destroy_ringrange_map2alm_healpix_mpi();
}

void ring_synthesis(long Nphi, long shifted, float *ringvals)
{
  long mp;
  double cosmp,sinmp,tmp[2];
  
  float *ringvals_out;
  fftwf_complex *cring,*cring_in;
  fftwf_plan plan;
  
  cring = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*(Nphi/2+1));
  assert(cring != NULL);
  ringvals_out = (float*) cring;
  cring_in = (fftwf_complex*) ringvals;
  plan = fftwf_plan_dft_c2r_1d((int) Nphi,cring,ringvals_out,FFTW_ESTIMATE);
  
  memmove(cring,cring_in,sizeof(fftwf_complex)*(Nphi/2+1));
    
  /* phase factors from transformation on pixels shifted by half a pixel length*/
  if(shifted)
    {
      for(mp=0;mp<Nphi/2+1;++mp)
	{
	  cosmp = cos(mp*M_PI/Nphi);
	  sinmp = sin(mp*M_PI/Nphi);
	  tmp[0] = (double) (cring[mp][0]);
	  tmp[1] = (double) (cring[mp][1]);
	  cring[mp][0] = (float) (tmp[0]*cosmp - tmp[1]*sinmp);
	  cring[mp][1] = (float) (tmp[1]*cosmp + tmp[0]*sinmp);
	}
    }
  
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  
  memmove(ringvals,ringvals_out,sizeof(float)*Nphi);
  
  fftwf_free(cring);
}

/*
  load-balancing scheme for m values
  
  - init the scheme with a polynomial fit to some timing data
  - after each alm2map_mpi call, a global var with timing data is updated
  - if you replan the healpix sht by destroying the old pland and calling the planner again, the new timing data is used
  
*/

double *alm2mapMTimesGlobal = NULL;
long Nalm2mapMTimesGlobal = 0;

void init_mrange_alm2map_healpix_mpi(long order)
{
  long lmax;
  double a = 2.6636271;
  double b = 2.0*(-2.3851605);
  double c = 3.0*(0.71754502);
  long m;
  double x,dx,integral;
  
  lmax = order2lmax(order);
  Nalm2mapMTimesGlobal = lmax+1;
  alm2mapMTimesGlobal = (double*)malloc(sizeof(double)*Nalm2mapMTimesGlobal);
  assert(alm2mapMTimesGlobal != NULL);

  dx = 1.0/(lmax+1);
  x = 0.0;
  for(m=0;m<=lmax;++m)
    {
      x = x+dx;
      alm2mapMTimesGlobal[m] = (a+b*x+c*x*x)*dx;
    }
  
  integral = 0.0;
  for(m=0;m<=lmax;++m)
    integral += alm2mapMTimesGlobal[m];
  
  for(m=0;m<=lmax-1;++m)
    alm2mapMTimesGlobal[m+1] += alm2mapMTimesGlobal[m];

  for(m=0;m<=lmax;++m)
    alm2mapMTimesGlobal[m] /= integral;
}

void destroy_mrange_alm2map_healpix_mpi(void)
{
  if(alm2mapMTimesGlobal != NULL)
    free(alm2mapMTimesGlobal);
  alm2mapMTimesGlobal = NULL;
  Nalm2mapMTimesGlobal = 0;
}

void get_mrange_alm2map_healpix_mpi(int MyNTasks, long *firstMTasks, long *lastMTasks, long order)
{
  long i,m;
  double int_per_proc;
  long lmax = order2lmax(order);

#ifndef STATIC_LOADBAL_SHT
  static int initFlag = 1;
  double *localTimes,totMTimes;
  long NlocalTimes;
#endif

#ifdef STATIC_LOADBAL_SHT
  destroy_mrange_alm2map_healpix_mpi();
  init_mrange_alm2map_healpix_mpi(order);
#else
  if(initFlag)
    {
      init_mrange_alm2map_healpix_mpi(order);
      initFlag = 0;
    }
  else if(Nalm2mapMTimesGlobal != (lmax+1))
    {
      destroy_mrange_alm2map_healpix_mpi();
      init_mrange_alm2map_healpix_mpi(order);
    }
  else if(alm2mapMTimesGlobal == NULL || Nalm2mapMTimesGlobal == 0)
    {
      init_mrange_alm2map_healpix_mpi(order);
    }
  else
    {
      /* compute normalized cumulative sum */
      NlocalTimes = 1 + order2lmax(order);
      localTimes = (double*)malloc(sizeof(double)*NlocalTimes);
      assert(localTimes != NULL);
      for(i=0;i<Nalm2mapMTimesGlobal;++i)
        localTimes[i] = 0.0;
      for(i=0;i<Nalm2mapMTimesGlobal;++i)
        localTimes[i] = alm2mapMTimesGlobal[i];
      
      MPI_Allreduce(localTimes,alm2mapMTimesGlobal,(int) Nalm2mapMTimesGlobal,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); 
      free(localTimes);
      
      totMTimes = 0.0;
      for(i=0;i<Nalm2mapMTimesGlobal;++i)
        totMTimes += alm2mapMTimesGlobal[i];
      
      for(i=0;i<Nalm2mapMTimesGlobal-1;++i)
        alm2mapMTimesGlobal[i+1] += alm2mapMTimesGlobal[i];

      for(i=0;i<Nalm2mapMTimesGlobal;++i)
        alm2mapMTimesGlobal[i] /= totMTimes;
      
      if(totMTimes == 0.0)
	{
	  destroy_mrange_alm2map_healpix_mpi();
	  init_mrange_alm2map_healpix_mpi(order);
	}
    }
#endif
  
  if(MyNTasks == 1)
    {
      firstMTasks[0] = 0;
      lastMTasks[0]  = lmax;
    }
  else
    {
      m = 0;
      int_per_proc = 1.0/MyNTasks;
      
      for(i=0;i<MyNTasks;++i)
        {
          firstMTasks[i] = -1;
          lastMTasks[i] = -1;
        }
      
      for(i=0;i<MyNTasks;++i)
        {
	  if(m >= Nalm2mapMTimesGlobal)
	    break;
          
	  firstMTasks[i] = m;
          
          while(alm2mapMTimesGlobal[m] < (i+1)*int_per_proc && m < Nalm2mapMTimesGlobal-1)
	    ++m;
	  
	  if(m == firstMTasks[i])
	    ++m;
	  
	  lastMTasks[i] = m - 1;
        }
      
      lastMTasks[MyNTasks-1] = lmax;
    }
  
  for(i=0;i<Nalm2mapMTimesGlobal;++i)
    alm2mapMTimesGlobal[i] = 0.0;
}

void read_ring_weights(char *path, HEALPixSHTPlan *plan)
{
  int ThisTask;
  char fname[2048];
  fitsfile *fptr;
  int status=0;
  int ext=1,anynul,colnum=1;
  double nulval=0;
  long nring_weights,repeat,width;
  LONGLONG firstrow=1,firstelem=1;
  int typecode;
  
  MPI_Comm_rank(MPI_COMM_WORLD,&ThisTask);
  
  if(ThisTask == 0)
    {
      //get name of file
      sprintf(fname,"%s/weight_ring_n%05ld.fits[%d]",path,order2nside(plan->order),ext);
      
      //open file
      fits_open_file(&fptr,fname,READONLY,&status);
      if(status)
	fits_report_error(stderr,status);
      
      //read the ring weights
      fits_get_num_rows(fptr,&nring_weights,&status);
      if(status)
	fits_report_error(stderr,status);
      fits_get_coltype(fptr,colnum,&typecode,&repeat,&width,&status);
      nring_weights *= repeat;
#ifdef TEST_CODE
      //fprintf(stderr,"# of weights = %ld (?= %ld)\n",nring_weights,2*order2nside(plan->order));
#endif
      assert(nring_weights == 2*order2nside(plan->order));
            
      plan->ring_weights = (double*)malloc(sizeof(double)*nring_weights);
      assert(plan->ring_weights != NULL);
      
      fits_read_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nring_weights,&nulval,plan->ring_weights,&anynul,&status);
      if(status)
	fits_report_error(stderr,status);
      
      fits_close_file(fptr,&status);
      if(status)
	fits_report_error(stderr,status);
      
#ifdef TEST_CODE
      //for(ext=0;ext<nring_weights;++ext)
      //fprintf(stderr,"rw[%d] = %le\n",ext,plan->ring_weights[ext]);
#endif
    }
    
  MPI_Bcast(&nring_weights,1,MPI_LONG,0,MPI_COMM_WORLD); 
  
  if(ThisTask != 0)
    {
      plan->ring_weights = (double*)malloc(sizeof(double)*nring_weights);
      assert(plan->ring_weights != NULL);
    }
  
  MPI_Bcast(plan->ring_weights,(int) nring_weights,MPI_DOUBLE,0,MPI_COMM_WORLD);
    
}

void read_window_function(char *path, HEALPixSHTPlan *plan)
{
  int ThisTask;
  char fname[2048];
  fitsfile *fptr;
  int status=0;
  int ext=1,anynul,colnum=1;
  double nulval=0;
  long nwf,repeat,width;
  LONGLONG firstrow=1,firstelem=1;
  int typecode;
  
  MPI_Comm_rank(MPI_COMM_WORLD,&ThisTask);
  
  if(ThisTask == 0)
    {
      //get name of file
      sprintf(fname,"%s/pixel_window_n%04ld.fits[%d]",path,order2nside(plan->order),ext);
      
      //open file
      fits_open_file(&fptr,fname,READONLY,&status);
      if(status)
	fits_report_error(stderr,status);
      
      //read the ring weights
      fits_get_num_rows(fptr,&nwf,&status);
      if(status)
	fits_report_error(stderr,status);
      fits_get_coltype(fptr,colnum,&typecode,&repeat,&width,&status);
      nwf *= repeat;
#ifdef TEST_CODE
      //fprintf(stderr,"# of window func. vals = %ld (?= %ld)\n",nwf,4*order2nside(plan->order)+1);
#endif
      assert(nwf == 4*order2nside(plan->order)+1);
            
      plan->window_function = (double*)malloc(sizeof(double)*nwf);
      assert(plan->window_function != NULL);
      
      fits_read_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nwf,&nulval,plan->window_function,&anynul,&status);
      if(status)
	fits_report_error(stderr,status);
      
      fits_close_file(fptr,&status);
      if(status)
	fits_report_error(stderr,status);
      
#ifdef TEST_CODE
      //for(ext=0;ext<nwf;++ext)
      //fprintf(stderr,"wf[%d] = %le\n",ext,plan->window_function[ext]);
#endif
    }
    
  MPI_Bcast(&nwf,1,MPI_LONG,0,MPI_COMM_WORLD); 
  
  if(ThisTask != 0)
    {
      plan->window_function = (double*)malloc(sizeof(double)*nwf);
      assert(plan->window_function != NULL);
    }
  
  long limit = (1l << 30);
  if(nwf > limit)
    {
      
      MPI_Bcast(plan->window_function,(int) limit,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(plan->window_function+limit,(int) (nwf-limit),MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
  else
    MPI_Bcast(plan->window_function,(int) nwf,MPI_DOUBLE,0,MPI_COMM_WORLD);
}

void healpixsht_destroy_plan(HEALPixSHTPlan plan)
{
  free(plan.firstRingTasks);
  free(plan.lastRingTasks);
  free(plan.northStartIndMapvec);
  free(plan.southStartIndMapvec);
  free(plan.northStartIndGlobalMap);
  free(plan.southStartIndGlobalMap);
  free(plan.firstMTasks);
  free(plan.lastMTasks);
  if(plan.ring_weights != NULL)
    {
      free(plan.ring_weights);
      plan.ring_weights = NULL;
    }
  if(plan.window_function != NULL)
    {
      free(plan.window_function);
      plan.window_function = NULL;
    }
}

long order2lmax(long _order)
{
  return 3*order2nside(_order)-1;
}

long lm2index(long l, long m, long lmax)
{
  return (m+1)*(lmax+1) - m*(m+1)/2 - (lmax - m + 1) + l - m;
}

long num_lms(long lmax)
{
  return (lmax+1)*(lmax+1) - lmax*(lmax+1)/2;
}

#define HPX_MXL0 40
#define HPX_MXL1 1.35

long get_lmin_ylm(long m, double sintheta)
{
  long lmin = m;
  long lmincut = (long) ((m-HPX_MXL0)/HPX_MXL1/sintheta);
  
  if(lmincut > lmin)
    lmin = lmincut;
  return lmin;
}

#undef HPX_MXL0
#undef HPX_MXL1

void ring_analysis(long Nphi, float *ringvals)
{
  float *ringvals_in;
  fftwf_complex *cring,*cring_out;
  fftwf_plan plan;
  
  cring = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*(Nphi/2+1));
  assert(cring != NULL);
  ringvals_in = (float*) cring;
  cring_out = (fftwf_complex*) ringvals;
  
  plan = fftwf_plan_dft_r2c_1d((int) Nphi,ringvals_in,cring,FFTW_ESTIMATE); 
  
  memmove(ringvals_in,ringvals,sizeof(float)*Nphi);
  
  fftwf_execute(plan);
  
  fftwf_destroy_plan(plan);
  
  memmove(cring_out,cring,sizeof(fftwf_complex)*(Nphi/2+1));
  
  fftwf_free(cring);
}

/*
  load-balancing scheme stolen from HEALPix package, but now uses a 3rd order polynomial
  
  from ring = 1 to ring = nside use a cubic polynomial to give the cumulative total amount of work
  from ring = nside+1 to ring = 2*nside the cumulative amount of work is linear
  
  integral over ring numbers is converted to dimensionless units via x = ring/2/nside
  
  to fix the slope and intercept of the line, we integrate the differential amount of work (so quadratic polynomial) from 0 to 0.5
  then using this constant, we enforce that the integral over the rest of the differential work totals unity in order to fix the slope of the linear part
  
  this gives the formulas below
  
  can only seem to get the load balance to a factor of two or so
  
  thus use the function described above to intialize the scheme, then the code
  records the amount of time each ring takes and then dynamically adjusts the load-balancing scheme 
  
  you need to replan the healpix sht to use the new timing data
*/

double *map2almRingTimesGlobal = NULL;
long Nmap2almRingTimesGlobal = 0;

void init_ringrange_map2alm_healpix_mpi(long order)
{
  double dx,x,integral;
  long i;
  const double beta = 0.26612010;
  const double alpha = 2.0*(2.2382936);
  const double c = 3.0*(-1.1734647);
  double gam = 2.0*(1.0 - (alpha/8 + beta/2 + c/24.0));
  long nside = order2nside(order);
  
  assert(fabs(gam/2.0 + beta/2.0 + alpha/8.0 + c/24.0 - 1.0) < 1e-10); /* make sure integral of work is unity */
  
  Nmap2almRingTimesGlobal = 2*nside;
  map2almRingTimesGlobal = (double*)malloc(sizeof(double)*Nmap2almRingTimesGlobal);
  assert(map2almRingTimesGlobal != NULL);
  
  integral = 0.0;
  dx = 1.0/2.0/nside;
  x = 0.0;
  for(i=0;i<2*nside;++i)
    {
      x = x + dx;
      if (i+1 <= nside)
	integral = integral + (alpha*x+beta+c*x*x)*dx;
      else
	integral = integral + gam*dx;
      
      map2almRingTimesGlobal[i] = integral;
    }
}

void destroy_ringrange_map2alm_healpix_mpi(void)
{
  if(map2almRingTimesGlobal != NULL)
    free(map2almRingTimesGlobal);
  map2almRingTimesGlobal = NULL;
  Nmap2almRingTimesGlobal = 0;
}

void get_ringrange_map2alm_healpix_mpi(int MyNTasks, long *firstRing, long *lastRing, long order)
{
  long i,ring;
  double int_per_proc;
  long nside = order2nside(order);
  
#ifndef STATIC_LOADBAL_SHT
  static int initFlag = 1;
  double totRingTimes;
  double *localTimes;
  long NlocalTimes;
#endif

#ifdef STATIC_LOADBAL_SHT
   destroy_ringrange_map2alm_healpix_mpi();
   init_ringrange_map2alm_healpix_mpi(order);
#else
  if(initFlag)
    {
      init_ringrange_map2alm_healpix_mpi(order);
      initFlag = 0;
    }
  else if(Nmap2almRingTimesGlobal != 2*nside)
    {
      destroy_ringrange_map2alm_healpix_mpi();
      init_ringrange_map2alm_healpix_mpi(order);
    }
  else if(map2almRingTimesGlobal == NULL || Nmap2almRingTimesGlobal == 0)
    {
      init_ringrange_map2alm_healpix_mpi(order);
    }
  else
    {
      NlocalTimes = 2*nside;
      localTimes = (double*)malloc(sizeof(double)*NlocalTimes);
      assert(localTimes != NULL);
      for(i=0;i<Nmap2almRingTimesGlobal;++i)
        localTimes[i] = 0.0;
      for(i=0;i<Nmap2almRingTimesGlobal;++i)
        localTimes[i] = map2almRingTimesGlobal[i];

      MPI_Allreduce(localTimes,map2almRingTimesGlobal,(int) Nmap2almRingTimesGlobal,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      free(localTimes);
      
      //compute normalized cumulative sum
      totRingTimes = 0.0;
      for(i=0;i<Nmap2almRingTimesGlobal;++i)
	totRingTimes += map2almRingTimesGlobal[i];

      for(i=0;i<Nmap2almRingTimesGlobal-1;++i)
	map2almRingTimesGlobal[i+1] += map2almRingTimesGlobal[i];

      for(i=0;i<Nmap2almRingTimesGlobal;++i)
	map2almRingTimesGlobal[i] /= totRingTimes;
      
      if(totRingTimes == 0.0)
	{
	  destroy_ringrange_map2alm_healpix_mpi();
	  init_ringrange_map2alm_healpix_mpi(order);
	}
    }
#endif
  
  if(MyNTasks == 1)
    {
      firstRing[0] = 1;
      lastRing[0]  = 2*nside;
    }
  else if(MyNTasks == 2*nside)
    {
      ring = 1;
      for(i=0;i<MyNTasks;++i)
        {
          firstRing[i] = ring;
          lastRing[i] = ring;
	  ++ring;
        }
    }
  else
    {
      ring = 1;
      int_per_proc = 1.0/MyNTasks;
      
      for(i=0;i<MyNTasks;++i)
        {
          firstRing[i] = -1;
          lastRing[i] = -1;
        }
      
      for(i=0;i<MyNTasks;++i)
        {
	  if(ring > Nmap2almRingTimesGlobal)
	    break;

          firstRing[i] = ring;
          
          while(map2almRingTimesGlobal[ring-1] < (i+1)*int_per_proc && ring <= Nmap2almRingTimesGlobal-1)
	    ++ring;
	  
	  if(ring == firstRing[i])
	    ++ring;
	  
          lastRing[i] = ring - 1;
        }
      
      lastRing[MyNTasks-1] = 2*nside;
    }
  
  for(i=0;i<Nmap2almRingTimesGlobal;++i)
    map2almRingTimesGlobal[i] = 0.0;
}

