/*
  map analysis HEALPix function w/ MPI - does MPI parallel map -> alm
  
  This is a complete rewrite in C of the HEALPix analysis function. Parts of these routines were 
  taken from the public HEALPix package. See copyright and GPL info below.
  
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
#include <fftw3.h>
#include <gsl/gsl_math.h>
#include <mpi.h>
#include <assert.h>

#include "healpix_utils.h"
#include "healpix_shtrans.h"

void map2alm_mpi(double *alm_real, double *alm_imag, float *mapvec, HEALPixSHTPlan plan)
{
  int NTasks,ThisTask;
  
  long i,m,l,nring,j,lmin,mind,lmind;
  long nstartpix,ringpix,shifted;
  double nsintheta,ncostheta;
  double sfact,fac1;
  long Nside,Npix;
  long lmax;
  
  long firstRing,lastRing,lastRingLoop;
      
  fftwf_complex *ngm,*sgm;
  double ngmval[2],sgmval[2];
  double *plm;
  plmgen_data *plmdata;
  double plmeps;
  long firstl;
  double quadweight;
  double *ring_weights;
  double phase[2],tmp[2];
  
  long Nrings;
  long NringsThisTask;
  
  double runTime;
#ifdef OUTPUT_SHT_LOADBALANCE
  double mintm,maxtm,avgtm;
#endif
  
  fftwf_complex *mapvec_complex;
  long memTot = 0;

  double transTime,fftTime,sumTime,plmTime;
  double *ringTime;
    
  runTime = -MPI_Wtime();
  
  MPI_Comm_size(MPI_COMM_WORLD,&NTasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&ThisTask);
  
  //timing 
  plmTime = 0.0;
  fftTime = 0.0;
  sumTime = 0.0;
  transTime = 0.0;

  //init healpix vars
  Npix = order2npix(plan.order);
  Nside = order2nside(plan.order);
  lmax = plan.lmax;
  Nrings = 2*Nside;
  plmeps = 1e-30;
  quadweight = 4.0*M_PI/Npix;
  
  //get ring_weights
  ring_weights = (double*)malloc(sizeof(double)*Nrings);
  assert(ring_weights != NULL);
  memTot += sizeof(double)*Nrings;
  for(i=0;i<Nrings;++i)
    {
      if(plan.ring_weights != NULL)
	ring_weights[i] = plan.ring_weights[i];
      else
	ring_weights[i] = 0.0;
      
      ring_weights[i] += 1.0;
      
      ring_weights[i] *= quadweight;
    }
  
  //get rings for this task
  firstRing = plan.firstRingTasks[ThisTask];
  lastRing = plan.lastRingTasks[ThisTask];
  if(firstRing == -1 || lastRing == -1 || plan.firstMTasks[ThisTask] == -1 || plan.lastMTasks[ThisTask] == -1)
    {
      fprintf(stderr,"problem with assigning rings/ms to tasks! (Did you ask for more tasks than rings?)\n");
      for(i=0;i<NTasks;++i)
        fprintf(stderr,"\t%ld: firstRing,lastRing,Nrings = %ld|%ld,%ld, firstM,lastM,NM = %ld|%ld|%ld\n",i,
                plan.firstRingTasks[i],plan.lastRingTasks[i],2*Nside,
                plan.firstMTasks[i],plan.lastMTasks[i],plan.lmax+1);
      
      MPI_Abort(MPI_COMM_WORLD,123);
    }
  lastRingLoop = lastRing;
  NringsThisTask = lastRing - firstRing + 1;
  if(lastRing == Nrings)
    --lastRingLoop;
  
  /*
  if(ThisTask == 0)
    {
      for(i=0;i<NTasks;++i)
        {
	  fprintf(stderr,"\t%ld: firstRing,lastRing,Nrings = %ld|%ld,%ld, firstM,lastM,NM = %ld|%ld|%ld\n",i,
		  plan.firstRingTasks[i],plan.lastRingTasks[i],2*Nside,
		  plan.firstMTasks[i],plan.lastMTasks[i],plan.lmax+1);
	  fflush(stderr);
	}
    }
  */
  
  //time the rings to get load balance correct
  ringTime = (double*)malloc(sizeof(double)*NringsThisTask);
  memTot += sizeof(double)*NringsThisTask;
  assert(ringTime != NULL);
  for(i=0;i<NringsThisTask;++i)
    ringTime[i] = 0.0;
  
  //do FFT of rings in place first
  fftTime -= MPI_Wtime();
  mapvec_complex = (fftwf_complex*) mapvec;
  for(nring=firstRing;nring<=lastRingLoop;++nring)
    {
      ringTime[nring-firstRing] -= MPI_Wtime();
      
      get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
      
      mapvec = (float*) (mapvec_complex + plan.northStartIndMapvec[nring-firstRing]);
      for(i=0;i<ringpix;++i)
	mapvec[i] = (float) (mapvec[i]*ring_weights[nring-1]);
      ring_analysis(ringpix,mapvec);
      
      mapvec = (float*) (mapvec_complex + plan.southStartIndMapvec[nring-firstRing]);
      for(i=0;i<ringpix;++i)
	mapvec[i] = (float) (mapvec[i]*ring_weights[nring-1]);
      ring_analysis(ringpix,mapvec);
      
      ringTime[nring-firstRing] += MPI_Wtime();
    }
  
  if(lastRing == Nrings)
    {
      ringTime[lastRing-firstRing] -= MPI_Wtime();

      get_ring_info2(lastRing,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
            
      mapvec = (float*) (mapvec_complex + plan.northStartIndMapvec[lastRing-firstRing]);
      for(i=0;i<ringpix;++i)
	mapvec[i] = (float) (mapvec[i]*ring_weights[lastRing-1]);
      ring_analysis(ringpix,mapvec);
            
      ringTime[lastRing-firstRing] += MPI_Wtime();
    }
  
  mapvec = (float*) mapvec_complex;
  free(ring_weights);
  memTot -= sizeof(double)*Nrings;
  
  fftTime += MPI_Wtime();
  
  //pack the data and do transpose
  transTime -= MPI_Wtime();
  double *transDataReal,*transDataImag;
  double *transDataRealRecv,*transDataImagRecv;
  long NmTD,NrTD,NmTDR,NrTDR;
  
  NmTD = lmax+1;
  NrTD = 2*(lastRing - firstRing + 1);
  if(lastRing == Nrings)
    --NrTD;
  transDataReal = (double*)malloc(sizeof(double)*NrTD*NmTD);
  assert(transDataReal != NULL);
  memTot += sizeof(double)*NrTD*NmTD;
  transDataImag = (double*)malloc(sizeof(double)*NrTD*NmTD);
  assert(transDataImag != NULL);
  memTot += sizeof(double)*NrTD*NmTD;
  
  NmTDR = plan.lastMTasks[ThisTask] - plan.firstMTasks[ThisTask] + 1;
  NrTDR = 4*Nside-1;
  transDataRealRecv = (double*)malloc(sizeof(double)*NrTDR*NmTDR);
  assert(transDataRealRecv != NULL);
  memTot += sizeof(double)*NrTDR*NmTDR;
  transDataImagRecv = (double*)malloc(sizeof(double)*NrTDR*NmTDR);
  assert(transDataImagRecv != NULL);
  memTot += sizeof(double)*NrTDR*NmTDR;
  
  for(i=0;i<NmTD;++i)
    for(j=0;j<NrTD;++j)
      {
	transDataReal[i*NrTD + j] = 0.0;
	transDataImag[i*NrTD + j] = 0.0;
      }
  
  long nind;
  for(nring=firstRing;nring<=lastRingLoop;++nring)
    {
      get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
      ngm = mapvec_complex + plan.northStartIndMapvec[nring-firstRing];
      sgm = mapvec_complex + plan.southStartIndMapvec[nring-firstRing];
      nind = nring - firstRing;
      
      for(m=0;m<=lmax;++m)
	{
	  //get FFTed vals - only store half since map is real so need to get correct val for given m
	  mind = m%ringpix;
	  if(mind > ringpix/2)
	    {
	      mind = ringpix - mind;
	      ngmval[0] = ngm[mind][0];
	      ngmval[1] = -ngm[mind][1];
	      sgmval[0] = sgm[mind][0];
	      sgmval[1] = -sgm[mind][1];
	    }
	  else
	    {
	      ngmval[0] = ngm[mind][0];
	      ngmval[1] = ngm[mind][1];
	      sgmval[0] = sgm[mind][0];
	      sgmval[1] = sgm[mind][1];
	    }
	  
	  //phase factors of shifted rings
	  if(shifted)
	    {
	      phase[0] = cos(m*M_PI/ringpix);                                                                                                              
	      phase[1] = -sin(m*M_PI/ringpix);    
	      
	      tmp[0] = ngmval[0]*phase[0] - ngmval[1]*phase[1];
	      tmp[1] = ngmval[0]*phase[1] + ngmval[1]*phase[0];
	      ngmval[0] = tmp[0];
	      ngmval[1] = tmp[1];
	      
	      tmp[0] = sgmval[0]*phase[0] - sgmval[1]*phase[1];
	      tmp[1] = sgmval[0]*phase[1] + sgmval[1]*phase[0];
	      sgmval[0] = tmp[0];
	      sgmval[1] = tmp[1];
	    }
	  
	  transDataReal[m*NrTD + 2*nind] = ngmval[0];
	  transDataImag[m*NrTD + 2*nind] = ngmval[1];
	  transDataReal[m*NrTD + 2*nind + 1] = sgmval[0];
	  transDataImag[m*NrTD + 2*nind + 1] = sgmval[1];
	}//for(m=0;m<=lmax;++m) 
    }//for(nring=firstRing;nring<=lastRingLoop;++nring)
  
  if(lastRing == Nrings)
    {
      nring = lastRing;
      get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
      ngm = mapvec_complex + plan.northStartIndMapvec[nring-firstRing];
      nind = nring - firstRing;
      
      for(m=0;m<=lmax;++m)
	{
	  //get FFTed vals - only store half since map is real so need to get correct val for given m
	  mind = m%ringpix;
	  if(mind > ringpix/2)
	    {
	      mind = ringpix - mind;
	      ngmval[0] = ngm[mind][0];
	      ngmval[1] = -ngm[mind][1];
	    }
	  else
	    {
	      ngmval[0] = ngm[mind][0];
	      ngmval[1] = ngm[mind][1];
	    }
	
	  //phase factors of shifted rings
	  if(shifted)
	    {
	      phase[0] = cos(m*M_PI/ringpix);                                                                                                              
	      phase[1] = -sin(m*M_PI/ringpix);    
	      tmp[0] = ngmval[0]*phase[0] - ngmval[1]*phase[1];
	      tmp[1] = ngmval[0]*phase[1] + ngmval[1]*phase[0];
	      ngmval[0] = tmp[0];
	      ngmval[1] = tmp[1];
	    }
	
	  transDataReal[m*NrTD + 2*nind] = ngmval[0];
	  transDataImag[m*NrTD + 2*nind] = ngmval[1];
	}//for(m=0;m<=lmax;++m)
    }//if(lastRing == Nrings)
  
  //now do transpose
  int *sendcnts,*sdispls,*rdispls,*recvcnts;
  sendcnts = (int*)malloc(sizeof(int)*NTasks);
  assert(sendcnts != NULL);
  sdispls = (int*)malloc(sizeof(int)*NTasks);
  assert(sdispls != NULL);
  recvcnts = (int*)malloc(sizeof(int)*NTasks);
  assert(recvcnts != NULL);
  rdispls = (int*)malloc(sizeof(int)*NTasks);
  assert(rdispls != NULL);
  memTot += 4*sizeof(int)*NTasks;
  
  for(i=0;i<NTasks;++i)
    {
      //compute number of m vals on task i
      j = plan.lastMTasks[i] - plan.firstMTasks[i] + 1;
      sendcnts[i] = NrTD*j;
    }
  sdispls[0] = 0;
  for(i=1;i<NTasks;++i)
    sdispls[i] = sdispls[i-1] + sendcnts[i-1];
  
  MPI_Alltoall(sendcnts,1,MPI_INT,recvcnts,1,MPI_INT,MPI_COMM_WORLD);
  
  rdispls[0] = 0;
  for(i=1;i<NTasks;++i)
    rdispls[i] = rdispls[i-1] + recvcnts[i-1];
    
  int log2NTasks = 0;
  while(NTasks > (1 << log2NTasks))
    ++log2NTasks;
  int level;
  int sendTask,recvTask;
  MPI_Status Stat;
  
  //algorithm to loop through pairs of tasks linearly
  // -lifted from Gadget-2 under GPL (http://www.gnu.org/copyleft/gpl.html)
  // -see pm_periodic.c from Gadget-2 at http://www.mpa-garching.mpg.de/gadget/
  //
  for(level = 0; level < (1 << log2NTasks); level++) // note: for level=0, target is the same task
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;
      
      if(recvTask < NTasks)
        {
          if(sendTask != recvTask)
            {
              MPI_Sendrecv(transDataReal+sdispls[recvTask],sendcnts[recvTask],MPI_DOUBLE,recvTask,13,
                           transDataRealRecv+rdispls[recvTask],recvcnts[recvTask],MPI_DOUBLE,recvTask,13,
                           MPI_COMM_WORLD,&Stat);
              
              MPI_Sendrecv(transDataImag+sdispls[recvTask],sendcnts[recvTask],MPI_DOUBLE,recvTask,14,
                           transDataImagRecv+rdispls[recvTask],recvcnts[recvTask],MPI_DOUBLE,recvTask,14,
                           MPI_COMM_WORLD,&Stat);
            }
          else // just move cells into workspace since sendTask == recvTask and Nsend == Nrecv
            {
              for(i=0;i<recvcnts[recvTask];++i)
                transDataRealRecv[rdispls[recvTask]+i] = transDataReal[sdispls[recvTask]+i];
              for(i=0;i<recvcnts[recvTask];++i)
                transDataImagRecv[rdispls[recvTask]+i] = transDataImag[sdispls[recvTask]+i];
            }
        }
    }
  
  free(transDataReal);
  memTot -= sizeof(double)*NrTD*NmTD;
  free(transDataImag);
  memTot -= sizeof(double)*NrTD*NmTD;
  free(sendcnts);
  free(sdispls);
  free(recvcnts);
  memTot -= 3*sizeof(int)*NTasks;
  
  transTime += MPI_Wtime();
  
  //now do sum over rings for all l,m w/ MPI_Reduce 
  sumTime -= MPI_Wtime();
  
  //m vars
  long firstM,lastM,NMThisTask;
  double *mTimes;
  firstM = plan.firstMTasks[ThisTask];
  lastM = plan.lastMTasks[ThisTask];
  NMThisTask = lastM - firstM + 1;
  mTimes = (double*)malloc(sizeof(double)*NMThisTask);
  assert(mTimes != NULL);
  memTot += sizeof(double)*NMThisTask;
  for(i=0;i<NMThisTask;++i)
    mTimes[i] = 0.0;
  
  //init plm generator
  plm = (double*)malloc(sizeof(double)*(lmax+1));
  memTot += sizeof(double)*(lmax+1);
  assert(plm != NULL);
  plmTime -= MPI_Wtime();
  plmdata = plmgen_init(lmax,plmeps);
  plmTime += MPI_Wtime();

  //zero alms
  i = 0;
  for(m=firstM;m<=lastM;++m)
    for(l=m;l<=lmax;++l)
      {
	alm_real[i] = 0.0;
	alm_imag[i] = 0.0;
	++i;
      }
  
  int task;
  long start;
  long ind;
  for(task=0;task<NTasks;++task)
    {
      start = rdispls[task];
      
      firstRing = plan.firstRingTasks[task];
      lastRing = plan.lastRingTasks[task];
      lastRingLoop = lastRing;
      NrTD = 2*(lastRing - firstRing + 1);
      if(lastRing == Nrings)
	{
	  --lastRingLoop;
	  --NrTD;
	}
      
      lmind = 0;
      for(m=firstM;m<=lastM;++m)
	{
	  mind = m-firstM;
	  mTimes[mind] -= MPI_Wtime();
	  ind = start + mind*NrTD;
	    
	  for(nring=firstRing;nring<=lastRingLoop;++nring)
	    {
	      get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
	      nind = nring - firstRing;
	      
	      //do not sum over Ylm which are too small
	      lmin = get_lmin_ylm(m,nsintheta);
	      if(lmin <= lmax)
		{
		  //get plms for ring
		  plmTime -= MPI_Wtime();
		  plmgen(ncostheta,nsintheta,m,plm,&firstl,plmdata);
		  plmTime += MPI_Wtime();
		  
		  //loop over l values and update all at fixed m
		  if(firstl <= lmax)
		    {
		      sfact = 1.0 - 2.0*((firstl+m)%2);
		      for(l=firstl;l<=lmax;++l)
			{
			  //this is always true (before plm[l] factor) assert((((l+m)%2 == 0 && sfact == 1.0)) || (((l+m)%2 == 1 && sfact == -1.0)));
			  alm_real[lmind + l-m] += transDataRealRecv[ind + 2*nind]*plm[l];
			  alm_imag[lmind + l-m] += transDataImagRecv[ind + 2*nind]*plm[l];
			  
			  fac1 = sfact*plm[l];
			  alm_real[lmind + l-m] += transDataRealRecv[ind + 2*nind + 1]*fac1;
			  alm_imag[lmind + l-m] += transDataImagRecv[ind + 2*nind + 1]*fac1;
			  sfact = -sfact;
			}
		      
		      /* old, slower version 
			 sfact = 1.0 + -2.0*((firstl+m)%2);
			 for(l=firstl;l<=lmax;++l)
			 {
			 almloop_real[lmind + l-m] += sfact*sgmval[0]*plm[l];
			 almloop_imag[lmind + l-m] += sfact*sgmval[1]*plm[l];
			 almloop_real[lmind + l-m] += ngmval[0]*plm[l];
			 almloop_imag[lmind + l-m] += ngmval[1]*plm[l];
			 sfact = -1.0*sfact;
			 }
		      */
		    }//if(firstl <= lmax)
		}//if(lmin <= lmax)
	    }//for(nring=firstRing;nring<=lastRingLoop;++nring)
	  
	  lmind += lmax-m+1;
	  mTimes[mind] += MPI_Wtime();
	}//for(m=firstM;m<=lastM;++m)
      
      if(lastRing == Nrings)
	{
	  nring = lastRing;
	  get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
	  nind = nring - firstRing;
	  
	  lmind = 0;
	  for(m=firstM;m<=lastM;++m)
	    {
	      mind = m-firstM;
	      mTimes[mind] -= MPI_Wtime();
	      ind = start + mind*NrTD;
	      	      	  
	      //do not sum over Ylm which are too small
	      lmin = get_lmin_ylm(m,nsintheta);
	      if(lmin <= lmax)
		{
		  //get plms for ring
		  plmTime -= MPI_Wtime();
		  plmgen(ncostheta,nsintheta,m,plm,&firstl,plmdata);
		  plmTime += MPI_Wtime();
		  
		  //loop over l values and update all at fixed m
		  for(l=firstl;l<=lmax;++l)
		    {
		      //this is always true (before plm[l] factor) assert((((l+m)%2 == 0 && sfact == 1.0)) || (((l+m)%2 == 1 && sfact == -1.0)));
		      alm_real[lmind + l-m] += transDataRealRecv[ind + 2*nind]*plm[l];
		      alm_imag[lmind + l-m] += transDataImagRecv[ind + 2*nind]*plm[l];
		    }
		}//if(lmin <= lmax)
	      
	      lmind += lmax-m+1;
	      mTimes[mind] += MPI_Wtime();
	    }//for(m=firstM;m<=lastM;++m)
	}//if(lastRing == Nrings)
    
    }//for(task=0;task<NTasks;++task)
  
  free(transDataRealRecv);
  memTot -= sizeof(double)*NrTDR*NmTDR;
  free(transDataImagRecv);
  memTot -= sizeof(double)*NrTDR*NmTDR;
  free(rdispls);
  memTot -= sizeof(int)*NTasks;
  free(plm);
  memTot -= sizeof(double)*(lmax+1);
  plmgen_destroy(plmdata);
  
  sumTime += MPI_Wtime();
  runTime += MPI_Wtime();
  
#ifdef OUTPUT_SHT_LOADBALANCE
  MPI_Reduce(&runTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&runTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&runTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"map -> alm                          run time max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
  
  MPI_Reduce(&fftTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&fftTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&fftTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"map -> alm fft time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);

  MPI_Reduce(&transTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&transTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&transTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"map -> alm trans time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
  
  MPI_Reduce(&sumTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&sumTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&sumTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"map -> alm sum time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
  
  MPI_Reduce(&plmTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&plmTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&plmTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"map -> alm plm time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
#endif

  //do load balance stuff
  firstRing = plan.firstRingTasks[ThisTask];
  lastRing = plan.lastRingTasks[ThisTask];
  lastRingLoop = lastRing;
  NringsThisTask = lastRing - firstRing + 1;
  if(lastRing == Nrings)
    --lastRingLoop;
  
  //add in times
  for(j=0;j<NringsThisTask;++j)
    map2almRingTimesGlobal[j+firstRing-1] += ringTime[j];
  free(ringTime);
  memTot -= sizeof(double)*NringsThisTask;
  for(j=0;j<NMThisTask;++j)
    alm2mapMTimesGlobal[j+firstM] += mTimes[j];
  free(mTimes);
  memTot -= sizeof(double)*NMThisTask;
  
  assert(memTot == 0);
  
#ifdef OUTPUT_SHT_LOADBALANCE  
  FILE *fp;
  char name[1024];
  double *localTimes,*globalTimes;
  long NlocalTimes;
  
  NlocalTimes = 2*Nside;
  localTimes = (double*)malloc(sizeof(double)*NlocalTimes);
  assert(localTimes != NULL);
  globalTimes = (double*)malloc(sizeof(double)*NlocalTimes);
  assert(globalTimes != NULL);
  for(i=0;i<Nmap2almRingTimesGlobal;++i)
    localTimes[i] = 0.0;
  for(i=0;i<Nmap2almRingTimesGlobal;++i)
    localTimes[i] = map2almRingTimesGlobal[i];

  MPI_Allreduce(localTimes,globalTimes,(int) Nmap2almRingTimesGlobal,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    
  sprintf(name,"./map2alm_ringtimes.%ld",plan.order);
  
  if(ThisTask == 0)
    {
      remove(name);
      fp = fopen(name,"w");
      for(i=0;i<Nmap2almRingTimesGlobal;++i)
	fprintf(fp,"%ld %.20e \n",i,globalTimes[i]);
      fclose(fp);
    }
  
  free(localTimes);
  free(globalTimes);
#endif  
}
