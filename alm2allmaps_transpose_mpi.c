/*
  map synthesis HEALPix function w/ MPI - does MPI parallel alm -> map and all gradient maps up to second order
  
  This is a complete rewrite in C of the HEALPix synthesis function. Parts of these routines were 
  taken from the public HEALPix package. See copyright and GPL info below.
  
  -Matthew R. Becker, UofC 2011
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
#include <mpi.h>
#include <assert.h>

#include "healpix_utils.h"
#include "healpix_shtrans.h"

void alm2allmaps_mpi(double *alm_real, double *alm_imag, float *mapvec, float *mapvec_gt, float *mapvec_gp,
		     float *mapvec_gtt, float *mapvec_gtp, float *mapvec_gpp,
		     HEALPixSHTPlan plan)
{
  int NTasks,ThisTask;
  
  int *sendcnts,*sdispls,*rdispls,*recvcnts;
  double *transDataRealRecv,*transDataImagRecv;
  long NrTDR,NmTDR;
  
  long mapNum,Nmaps;
  long nring,j,i,lmind,l,m,mp,k,lmin,ringpix_complex;
  double sfact,skfact;
  double rval,ival;
  
  long nstartpix,ringpix,shifted;
  double nsintheta,ncostheta;
    
  long Nside,lmax;
  long Nrings;
  long firstRing,lastRing,lastLoopRing;
  long NringsThisTask,NringsLoop;
  
  const long NringChunkBase = 20;
  long ringChunkStart,ringChunkStop,*ringpixRingChunk,*shiftedRingChunk,NringChunk;
  double *nsinthetaRingChunk,*ncosthetaRingChunk;
  double **qmn_real[6],**qmn_imag[6],**qms_real[6],**qms_imag[6];
  long Nqm,firstChunkRing,lastChunkRing;
  
  double *plm;
  plmgen_data *plmdata;
  double plmeps;
  long firstl;
  long firstM,lastM,NMThisTask;
  
  double runTime;
#ifdef OUTPUT_SHT_LOADBALANCE
  double mintm,maxtm,avgtm;
#endif
  long memTot = 0;
  double transTime,fftTime,sumTime,pixTime,plmTime;
  
  fftwf_complex *mapvec_complex[6];
  double *mTime;
  long Nchunks,chunkInd;
  
  long lmaxp1;
  double gradquantn,gradquants,gradfact,*svec;
  double gradplmn,gradplms,gradplm_lm1_n,gradplm_lm1_s;
  double ncos_nsin,ncos_nsin2,ncos2_nsin2,fac1,fac2,fac3;
  long task,start,NmT,nind,sind;
  double *transDataReal;
  double *transDataImag;
  long NmTD,NrTD;
  long mind;
  
  //init timing and basic data for plan
  runTime = -MPI_Wtime();
  
  MPI_Comm_size(MPI_COMM_WORLD,&NTasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&ThisTask);
  
  transTime = 0.0;
  fftTime = 0.0;
  sumTime = 0.0;
  pixTime = 0.0;
  plmTime = 0.0;
  
  Nmaps = 6;
  Nside = order2nside(plan.order);
  Nrings = 2*Nside;
  plmeps = 1e-30;
  lmax = plan.lmax;
  mapvec_complex[0] = (fftwf_complex*) mapvec;
  mapvec_complex[1] = (fftwf_complex*) mapvec_gt;
  mapvec_complex[2] = (fftwf_complex*) mapvec_gp;
  mapvec_complex[3] = (fftwf_complex*) mapvec_gtt;
  mapvec_complex[4] = (fftwf_complex*) mapvec_gtp;
  mapvec_complex[5] = (fftwf_complex*) mapvec_gpp;
  
  //get rings for this task
  firstRing = plan.firstRingTasks[ThisTask];
  lastRing = plan.lastRingTasks[ThisTask];
  firstM = plan.firstMTasks[ThisTask];
  lastM = plan.lastMTasks[ThisTask];
  if(firstRing == -1 || lastRing == -1 || firstM == -1 || lastM == -1)
    {
      fprintf(stderr,"problem with assigning rings/ms to tasks! (Did you ask for more tasks than rings?)\n");
      for(i=0;i<NTasks;++i)
	fprintf(stderr,"\t%ld: firstRing,lastRing,Nrings = %ld|%ld,%ld, firstM,lastM,NM = %ld|%ld|%ld\n",i,
		plan.firstRingTasks[i],plan.lastRingTasks[i],2*Nside,
		plan.firstMTasks[i],plan.lastMTasks[i],plan.lmax+1);
      MPI_Abort(MPI_COMM_WORLD,123);
    }
#ifdef DEBUG_IO
  if(ThisTask == 0)
    for(i=0;i<NTasks;++i)
	fprintf(stderr,"\t%ld: firstRing,lastRing,Nrings = %ld|%ld,%ld, firstM,lastM,NM = %ld|%ld|%ld\n",i,
		plan.firstRingTasks[i],plan.lastRingTasks[i],2*Nside,
		plan.firstMTasks[i],plan.lastMTasks[i],plan.lmax+1);
#endif

  NMThisTask = lastM-firstM+1;
  NringsThisTask = lastRing - firstRing + 1;
  NringsLoop = NringsThisTask;
  lastLoopRing = lastRing;
  if(lastRing == Nrings)
    {
      --NringsLoop;
      --lastLoopRing;
    }
  Nqm = NMThisTask;

  //init plm generator
  plm = (double*)malloc(sizeof(double)*(lmax + 1));
  assert(plm != NULL);
  memTot += sizeof(double)*(lmax + 1);
  plmTime -= MPI_Wtime();
  plmdata = plmgen_init(lmax,plmeps);
  plmTime += MPI_Wtime();
  
  //compute layout for transposed data
  NmTD = NMThisTask;
  NrTD = 4*Nside-1;
  
  //memory buffers 
  transDataReal = (double*)malloc(sizeof(double)*NrTD*Nmaps*NmTD);
  assert(transDataReal != NULL);
  memTot += sizeof(double)*NrTD*Nmaps*NmTD;
  transDataImag = (double*)malloc(sizeof(double)*NrTD*Nmaps*NmTD);
  assert(transDataImag != NULL);
  memTot += sizeof(double)*NrTD*Nmaps*NmTD;
  for(i=0;i<NrTD;++i)
    for(j=0;j<Nmaps;++j)
      for(k=0;k<NmTD;++k)
	{
	  transDataReal[(i*Nmaps + j)*NmTD + k] = 0.0;
	  transDataImag[(i*Nmaps + j)*NmTD + k] = 0.0;
	}

  ringpixRingChunk = (long*)malloc(sizeof(long)*NringChunkBase);
  assert(ringpixRingChunk != NULL);
  memTot += sizeof(long)*NringChunkBase;
  shiftedRingChunk = (long*)malloc(sizeof(long)*NringChunkBase);
  assert(shiftedRingChunk != NULL);
  memTot += sizeof(long)*NringChunkBase;
  nsinthetaRingChunk = (double*)malloc(sizeof(double)*NringChunkBase);
  assert(nsinthetaRingChunk !=NULL);
  memTot += sizeof(double)*NringChunkBase;
  ncosthetaRingChunk = (double*)malloc(sizeof(double)*NringChunkBase);
  assert(ncosthetaRingChunk !=NULL);
  memTot += sizeof(double)*NringChunkBase;
  
  for(j=0;j<6;++j)
    {
      qmn_real[j] = (double**)malloc(sizeof(double*)*NringChunkBase);
      assert(qmn_real[j] != NULL);
      memTot += sizeof(double*)*NringChunkBase;
      qmn_imag[j] = (double**)malloc(sizeof(double*)*NringChunkBase);
      assert(qmn_imag[j] != NULL);
      memTot += sizeof(double*)*NringChunkBase;
      qms_real[j] = (double**)malloc(sizeof(double*)*NringChunkBase);
      assert(qms_real[j] != NULL);
      memTot += sizeof(double*)*NringChunkBase;
      qms_imag[j] = (double**)malloc(sizeof(double*)*NringChunkBase);
      assert(qms_imag[j] != NULL);
      memTot += sizeof(double*)*NringChunkBase;
      
      for(i=0;i<NringChunkBase;++i)
	{
	  qmn_real[j][i] = (double*)malloc(sizeof(double)*Nqm);
	  assert(qmn_real[j][i] != NULL);
	  memTot += sizeof(double)*Nqm;
	  
	  qmn_imag[j][i] = (double*)malloc(sizeof(double)*Nqm);
	  assert(qmn_imag[j][i] != NULL);
	  memTot += sizeof(double)*Nqm;
	  
	  qms_real[j][i] = (double*)malloc(sizeof(double)*Nqm);
	  assert(qms_real[j][i] != NULL);
	  memTot += sizeof(double)*Nqm;
	  
	  qms_imag[j][i] = (double*)malloc(sizeof(double)*Nqm);
	  assert(qms_imag[j][i] != NULL);
	  memTot += sizeof(double)*Nqm;
	}
    }
  
  lmaxp1 = lmax+1;
  svec = (double*)malloc(sizeof(double)*(lmax + 1)*NMThisTask);
  assert(svec != NULL);
  memTot += sizeof(double)*(lmax + 1)*NMThisTask;
  for(m=firstM;m<=lastM;++m)
    for(l=m;l<=lmax;++l)
      svec[l+lmaxp1*(m-firstM)] = sqrt((2.0*l+1.0)/(2.0*l-1.0)*((double) (l*l-m*m)));
  
  //time each m value to get load balance correct
  mTime = (double*)malloc(sizeof(double)*NMThisTask);
  assert(mTime != NULL);
  memTot += sizeof(double)*NMThisTask;
  for(i=0;i<NMThisTask;++i)
    mTime[i] = 0.0;

  //setup chunks
  firstChunkRing = 1;
  lastChunkRing = 2*Nside-1;
  Nchunks = (lastChunkRing-firstChunkRing+1)/NringChunkBase;
  if(Nchunks*NringChunkBase < lastChunkRing-firstChunkRing+1)
    ++Nchunks;
  
  sumTime -= MPI_Wtime();

#ifdef DEBUG
#if DEBUG_LEVEL > 0  
  fprintf(stderr,"%f MB allocated in alm2allmaps_mpi.\n",((double) (sizeof(double)*(memTot)))/1024.0/1024.0);
#endif
#endif  
  
  //do sums over plms - store rings in transData for MPI transpose
  for(chunkInd=0;chunkInd<Nchunks;++chunkInd)
    {
      //get index range of subchunks
      ringChunkStart = chunkInd*NringChunkBase + firstChunkRing;
      ringChunkStop = ringChunkStart + NringChunkBase - 1;
      if(ringChunkStop > lastChunkRing)
	ringChunkStop = lastChunkRing;
      NringChunk = ringChunkStop - ringChunkStart + 1;
	  
      //init parms for subchunks
      for(i=0;i<NringChunk;++i)
	{
	  nring = i + ringChunkStart;
	  get_ring_info2(nring,&nstartpix,ringpixRingChunk+i,ncosthetaRingChunk+i,nsinthetaRingChunk+i,shiftedRingChunk+i,plan.order);
	  
	  for(mapNum=0;mapNum<6;++mapNum)
	    for(j=0;j<Nqm;++j)
	      {
		qmn_real[mapNum][i][j] = 0.0;
		qmn_imag[mapNum][i][j] = 0.0;
		qms_real[mapNum][i][j] = 0.0;
		qms_imag[mapNum][i][j] = 0.0;
	      }
	}
      
      //sum over l at fixed m for each ring using only alms on ThisTask
      for(m=firstM;m<=lastM;++m)
	{
	  mind = m-firstM;
	  mTime[mind] -= MPI_Wtime();
	  
	  //sum over rings - lmind tracks alm coefficients for this m
	  lmind = (lmax+1)*(m-firstM) + ((firstM-2)*(firstM+1))/2 - ((m-2)*(m+1))/2;
	  for(i=0;i<NringChunk;++i)
	    {
	      //do not sum over Ylm which are too small
	      lmin = get_lmin_ylm(m,(float) (nsinthetaRingChunk[i]));
	      if(lmin <= lmax)
		{
		  plmTime -= MPI_Wtime();
		  plmgen(ncosthetaRingChunk[i],nsinthetaRingChunk[i],m,plm,&firstl,plmdata);
		  plmTime += MPI_Wtime();
		  
		  //now do sum over l
		  if(firstl <= lmax)
		    {
		      //alm -> map - mapNum = 0 
		      sfact = 1.0 - (((firstl+m)%2) << 1);
		      for(l=firstl;l<=lmax;++l)
			{
			  //this is always true assert((((l+m)%2 == 0 && sfact == 1.0)) || (((l+m)%2 == 1 && sfact == -1.0)));
			  
			  //pure synthesis of map
			  rval = alm_real[lmind + l-m]*plm[l];
			  ival = alm_imag[lmind + l-m]*plm[l];
			  qmn_real[0][i][mind] += rval;
			  qmn_imag[0][i][mind] += ival;
			  qms_real[0][i][mind] += sfact*rval;
			  qms_imag[0][i][mind] += sfact*ival;			  
			  
			  //gradient w.r.t. phi is i*m/sin(\theta)P_{lm}\exp(i*m*\phi)*a_{lm} \hat{\phi}
			  rval *= m;
			  ival *= m;
			  qmn_real[2][i][mind] -= ival;//+= -1.0*alm_imag[lmind + l-m]*plm[l]*((double) m);
			  qmn_imag[2][i][mind] += rval;//alm_real[lmind + l-m]*plm[l]*((double) m);
			  qms_real[2][i][mind] -= sfact*ival;//+= sfact*(-1.0)*alm_imag[lmind + l-m]*plm[l]*((double) m);
			  qms_imag[2][i][mind] += sfact*rval;//sfact*alm_real[lmind + l-m]*plm[l]*((double) m);
			  
			  //del2phi_div_sin2theta is -m^2/sin^{2}(\theta)P_{lm}\exp(i*m*\phi)*a_{lm}
			  rval *= m;
			  ival *= m;
			  qmn_real[5][i][mind] -= rval;//+= -1.0*alm_real[lmind + l-m]*plm[l]*((double) (m*m));
			  qmn_imag[5][i][mind] -= ival;//+= -1.0*alm_imag[lmind + l-m]*plm[l]*((double) (m*m));
			  qms_real[5][i][mind] -= sfact*rval;//+= sfact*(-1.0)*alm_real[lmind + l-m]*plm[l]*((double) (m*m));
			  qms_imag[5][i][mind] -= sfact*ival;//+= sfact*(-1.0)*alm_imag[lmind + l-m]*plm[l]*((double) (m*m));
			  
			  sfact = -sfact;
			}
		      
		      //grad theta, grad2 theta, and gradthetgradphi transforms
		      sfact = 1.0 - 2.0*((firstl+m)%2);
		      ncos_nsin = ncosthetaRingChunk[i]/nsinthetaRingChunk[i];
		      ncos_nsin2 = ncosthetaRingChunk[i]/nsinthetaRingChunk[i]/nsinthetaRingChunk[i];
		      ncos2_nsin2 = ncos_nsin*ncos_nsin;
		      
		      //this is the first iteration of the loop
		      //it is broken out here to avoid branching in the loop
		      l = firstl;
		      if(l > 0)
			{
			  //grad theta
			  gradplmn = ((double) l)*ncos_nsin*plm[l];
			  gradplms = -sfact*gradplmn;  //-1 for cos(pi-nsintheta)
			  qmn_real[1][i][mind] += alm_real[lmind + l-m]*gradplmn;
			  qmn_imag[1][i][mind] += alm_imag[lmind + l-m]*gradplmn;
			  qms_real[1][i][mind] += alm_real[lmind + l-m]*gradplms;
			  qms_imag[1][i][mind] += alm_imag[lmind + l-m]*gradplms;
			  
			  
			  //we need grad^2\theta which is l*cos\theta/sin\theta\gradplm[l] - l*plm[l] 
			  //                               - l*cos^{2}\theta/sin^{2}\theta*plm[l]
			  //                               + (l+m)*sqrt((2*l+1)/(2*l-1)*(l-m)/(l+m))*plm[l-1]*cos\theta/sin^{2}\theta
			  //                               - (l+m)*sqrt((2*l+1)/(2*l-1)*(l-m)/(l+m))*\gradplm[l-1]/sin\theta
			  fac1 = ((double) l)*ncos_nsin;
			  fac2 = ((double) l)*plm[l]*(1.0 + ncos2_nsin2);
			  gradquantn = fac1*gradplmn - fac2;
			  gradquants = -fac1*gradplms - sfact*fac2;
			  
			  qmn_real[3][i][mind] += alm_real[lmind + l-m]*gradquantn;
			  qmn_imag[3][i][mind] += alm_imag[lmind + l-m]*gradquantn;
			  qms_real[3][i][mind] += alm_real[lmind + l-m]*gradquants;
			  qms_imag[3][i][mind] += alm_imag[lmind + l-m]*gradquants;
			  
			  gradplm_lm1_n = gradplmn;
			  gradplm_lm1_s = gradplms;
			  
			  //grad theta grad phi gives factor of i*m/sintheta
			  gradplmn *= m;
			  gradplms *= m;
			  qmn_real[4][i][mind] -= alm_imag[lmind + l-m]*gradplmn;
			  qmn_imag[4][i][mind] += alm_real[lmind + l-m]*gradplmn;
			  qms_real[4][i][mind] -= alm_imag[lmind + l-m]*gradplms;  //sfact is over all sign of plm(pi-ncostheta) at l
			  qms_imag[4][i][mind] += alm_real[lmind + l-m]*gradplms;
			}
		      else
			{
			  gradplm_lm1_n = 0.0;
			  gradplm_lm1_s = 0.0;
			}
		      
		      sfact = -sfact;
		      ++firstl;
                      
		      for(l=firstl;l<=lmax;++l)
			{
			  //gradient w.r.t. theta is l*cos\theta/sin\theta*plm[l] - (l+m)*sqrt((2*l+1)/(2*l-1)*(l-m)/(l+m))*plm[l-1] \hat{\theta}
			  gradplmn = ((double) l)*ncos_nsin*plm[l] 
			    - svec[l+lmaxp1*(m-firstM)]*plm[l-1]/nsinthetaRingChunk[i];
			  gradplms = -sfact*gradplmn; //-1 for cos(pi-nsintheta) and also -1 because plm of l-1 differs by -1 from plm of l for pi-theta
                          
			  qmn_real[1][i][mind] += alm_real[lmind + l-m]*gradplmn;
			  qmn_imag[1][i][mind] += alm_imag[lmind + l-m]*gradplmn;
			  qms_real[1][i][mind] += alm_real[lmind + l-m]*gradplms;
			  qms_imag[1][i][mind] += alm_imag[lmind + l-m]*gradplms;
			  
			  //we need grad^2\theta which is l*cos\theta/sin\theta\gradplm[l] - l*plm[l] 
			  //                               - l*cos^{2}\theta/sin^{2}\theta*plm[l]
			  //                               + (l+m)*sqrt((2*l+1)/(2*l-1)*(l-m)/(l+m))*plm[l-1]*cos\theta/sin^{2}\theta
			  //                               - (l+m)*sqrt((2*l+1)/(2*l-1)*(l-m)/(l+m))*\gradplm[l-1]/sin\theta
			  fac1 = ((double) l)*ncos_nsin;
			  fac2 = ((double) l)*plm[l]*(1.0 + ncos2_nsin2);
			  fac3 = svec[l+lmaxp1*(m-firstM)]/nsinthetaRingChunk[i];
			  gradquantn = fac1*gradplmn - fac2 - fac3*gradplm_lm1_n;
			  gradquants = -fac1*gradplms - sfact*fac2 - fac3*gradplm_lm1_s;
			  gradfact = svec[l+lmaxp1*(m-firstM)]*plm[l-1]*ncos_nsin2;
			  gradquantn += gradfact;
			  gradquants += gradfact*sfact;
			  
			  qmn_real[3][i][mind] += alm_real[lmind + l-m]*gradquantn;
			  qmn_imag[3][i][mind] += alm_imag[lmind + l-m]*gradquantn;
			  qms_real[3][i][mind] += alm_real[lmind + l-m]*gradquants;
			  qms_imag[3][i][mind] += alm_imag[lmind + l-m]*gradquants;
			  
			  gradplm_lm1_n = gradplmn;
			  gradplm_lm1_s = gradplms;
			  
			  //grad theta grad phi gives factor of i*m/sintheta
			  gradplmn *= m;
			  gradplms *= m;
			  qmn_real[4][i][mind] -= alm_imag[lmind + l-m]*gradplmn;
			  qmn_imag[4][i][mind] += alm_real[lmind + l-m]*gradplmn;
			  qms_real[4][i][mind] -= alm_imag[lmind + l-m]*gradplms;  //sfact is over all sign of plm(pi-ncostheta) at l
			  qms_imag[4][i][mind] += alm_real[lmind + l-m]*gradplms;
			  
			  sfact = -sfact;
			} //for(l=firstl;l<=lmax;++l)
		    } //if(firstl <= lmax)
		} //if(lmax >= lmin)
	    } //for(i=0;i<NringChunk;++i)
	  
	  mTime[mind] += MPI_Wtime();
	
	} //for(m=firstM;m<=lastM;++m)
      
      //put data in array for transpose
      for(i=0;i<NringChunk;++i)
	{
	  nring = i + ringChunkStart - 1;
	  for(j=0;j<Nmaps;++j)
	    for(k=0;k<NmTD;++k)
	      {
		//north ring
		transDataReal[((2*nring)*Nmaps + j)*NmTD + k] = qmn_real[j][i][k];
		transDataImag[((2*nring)*Nmaps + j)*NmTD + k] = qmn_imag[j][i][k];
		
		//south ring
		transDataReal[((2*nring+1)*Nmaps + j)*NmTD + k] = qms_real[j][i][k];
		transDataImag[((2*nring+1)*Nmaps + j)*NmTD + k] = qms_imag[j][i][k];
	      }
	}
      
    }//for(chunkInd=0;chunkInd<Nchunks;++chunkInd)
  
  ///do last ring on equator as a special case to avoid branching in loops
  get_ring_info2(2*Nside,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
  
  //sum over l at fixed m for each ring using only alms on ThisTask
  for(m=firstM;m<=lastM;++m)
    {
      mind = m-firstM;
      mTime[mind] -= MPI_Wtime();
      
      for(mapNum=0;mapNum<6;++mapNum)
	{
	  qmn_real[mapNum][0][mind] = 0.0;
	  qmn_imag[mapNum][0][mind] = 0.0;
	}

      //lmind tracks alm coefficients for this m
      lmind = (lmax+1)*(m-firstM) + ((firstM-2)*(firstM+1))/2 - ((m-2)*(m+1))/2;
      
      //do not sum over Ylm which are too small
      lmin = get_lmin_ylm(m,(float) (nsintheta));
      if(lmin <= lmax)
	{
	  plmTime -= MPI_Wtime();
	  plmgen(ncostheta,nsintheta,m,plm,&firstl,plmdata);
	  plmTime += MPI_Wtime();
	  
	  //now do sum over l
	  if(firstl <= lmax)
	    {
	      for(l=firstl;l<=lmax;++l)
		{
		  //alm -> map
		  rval = alm_real[lmind + l-m]*plm[l];
		  ival = alm_imag[lmind + l-m]*plm[l];
		  qmn_real[0][0][mind] += rval;//alm_real[lmind + l-m]*plm[l];
		  qmn_imag[0][0][mind] += ival;//alm_imag[lmind + l-m]*plm[l];
		  
		  //gradient w.r.t. phi is i*m/sin(\theta)P_{lm}\exp(i*m*\phi)*a_{lm} \hat{\phi}
		  rval *= m;
		  ival *= m;
		  qmn_real[2][0][mind] -= ival;//+= -1.0*alm_imag[lmind + l-m]*plm[l]*((double) m);
		  qmn_imag[2][0][mind] += rval;//alm_real[lmind + l-m]*plm[l]*((double) m);
		  
		  //del2phi_div_sin2theta is -m^2/sin^{2}(\theta)P_{lm}\exp(i*m*\phi)*a_{lm}
		  rval *= m;
		  ival *= m;
		  qmn_real[5][0][mind] -= rval;//+= -1.0*alm_real[lmind + l-m]*plm[l]*((double) (m*m));
		  qmn_imag[5][0][mind] -= ival;//+= -1.0*alm_imag[lmind + l-m]*plm[l]*((double) (m*m));
		}
	      
	      l = firstl;
	      gradfact = ncostheta/nsintheta;
	      if(l > 0)
		{
		  //grad theta
		  gradplmn = ((double) l)*gradfact*plm[l];
		  qmn_real[1][0][mind] += alm_real[lmind + l-m]*gradplmn;
		  qmn_imag[1][0][mind] += alm_imag[lmind + l-m]*gradplmn;
		  
		  //we need grad^2\theta which is l*cos\theta/sin\theta\gradplm[l] - l*plm[l] 
		  //                               - l*cos^{2}\theta/sin^{2}\theta*plm[l]
		  //                               + (l+m)*sqrt((2*l+1)/(2*l-1)*(l-m)/(l+m))*plm[l-1]*cos\theta/sin^{2}\theta
		  //                               - (l+m)*sqrt((2*l+1)/(2*l-1)*(l-m)/(l+m))*\gradplm[l-1]/sin\theta
		  gradquantn = ((double) l)*gradfact*gradplmn - ((double) l)*plm[l]*(1.0 + gradfact*gradfact);
                  
		  qmn_real[3][0][mind] += alm_real[lmind + l-m]*gradquantn;
		  qmn_imag[3][0][mind] += alm_imag[lmind + l-m]*gradquantn;
                  
		  gradplm_lm1_n = gradplmn;
		  
		  //grad theta grad phi
		  gradplmn *= m;
		  qmn_real[4][0][mind] -= alm_imag[lmind + l-m]*gradplmn;
		  qmn_imag[4][0][mind] += alm_real[lmind + l-m]*gradplmn;
		}
	      else
		gradplm_lm1_n = 0.0;
	      
	      ++firstl;
              
	      for(l=firstl;l<=lmax;++l)
		{
		  //grad theta
		  gradplmn = ((double) l)*gradfact*plm[l] - svec[l+lmaxp1*(m-firstM)]*plm[l-1]/nsintheta;
		  qmn_real[1][0][mind] += alm_real[lmind + l-m]*gradplmn;
		  qmn_imag[1][0][mind] += alm_imag[lmind + l-m]*gradplmn;
		  
		  //we need grad^2\theta which is l*cos\theta/sin\theta\gradplm[l] - l*plm[l] 
		  //                               - l*cos^{2}\theta/sin^{2}\theta*plm[l]
		  //                               + (l+m)*sqrt((2*l+1)/(2*l-1)*(l-m)/(l+m))*plm[l-1]*cos\theta/sin^{2}\theta
		  //                               - (l+m)*sqrt((2*l+1)/(2*l-1)*(l-m)/(l+m))*\gradplm[l-1]/sin\theta
		  gradquantn = ((double) l)*gradfact*gradplmn - ((double) l)*plm[l]*(1.0 + gradfact*gradfact)
		    + svec[l+lmaxp1*(m-firstM)]/nsintheta*(plm[l-1]*gradfact - gradplm_lm1_n);
		  
		  qmn_real[3][0][mind] += alm_real[lmind + l-m]*gradquantn;
		  qmn_imag[3][0][mind] += alm_imag[lmind + l-m]*gradquantn;
                  
		  gradplm_lm1_n = gradplmn;
		  
		  //grad theta grad phi
		  gradplmn *= m;
		  qmn_real[4][0][mind] -= alm_imag[lmind + l-m]*gradplmn;
		  qmn_imag[4][0][mind] += alm_real[lmind + l-m]*gradplmn;
		}//for(l=firstl;l<=lmax;++l)
	    }//if(firstl <= lmax)
	}//if(lmax >= lmin)
      
      mTime[mind] += MPI_Wtime();
      
    }//for(m=firstM;m<=lastM;++m)
  
  nring = 2*Nside - 1;
  for(j=0;j<Nmaps;++j)
    {
      for(k=0;k<NmTD;++k)
	{
	  //north ring                                                                                                                                                                               
	  transDataReal[((2*nring)*Nmaps + j)*NmTD + k] = qmn_real[j][0][k];
	  transDataImag[((2*nring)*Nmaps + j)*NmTD + k] = qmn_imag[j][0][k];
	}//for(k=0;k<NmTD;++k)                                                                                                                                                                       
    }//for(j=0;j<Nmaps;++j)                                                                                                                                                                          
  
  //free some mem
  free(plm);
  memTot -= sizeof(double)*(lmax+1);
  plmgen_destroy(plmdata);
  
  free(svec);
  memTot -= sizeof(double)*(lmax+1)*NMThisTask;
  
  for(mapNum=0;mapNum<6;++mapNum)
    {
      for(i=0;i<NringChunkBase;++i)
	{
	  free(qmn_real[mapNum][i]);
	  memTot -= sizeof(double)*Nqm;
	  free(qmn_imag[mapNum][i]);
	  memTot -= sizeof(double)*Nqm;
	  free(qms_real[mapNum][i]);
	  memTot -= sizeof(double)*Nqm;
	  free(qms_imag[mapNum][i]);
	  memTot -= sizeof(double)*Nqm;
	}
      
      free(qmn_real[mapNum]);
      memTot -= sizeof(double*)*NringChunkBase;
      free(qmn_imag[mapNum]);
      memTot -= sizeof(double*)*NringChunkBase;
      free(qms_real[mapNum]);
      memTot -= sizeof(double*)*NringChunkBase;
      free(qms_imag[mapNum]);
      memTot -= sizeof(double*)*NringChunkBase;
    }
  free(ringpixRingChunk);
  memTot -= sizeof(long)*NringChunkBase;
  free(shiftedRingChunk);
  memTot -= sizeof(long)*NringChunkBase;
  free(nsinthetaRingChunk);
  memTot -= sizeof(double)*NringChunkBase;
  free(ncosthetaRingChunk);
  memTot -= sizeof(double)*NringChunkBase;
  
  sumTime += MPI_Wtime();
  
  /* now do transpose step
     done with MPI_Alltoallv
     1) compute displacements and counts
     2) transpose real map
     3) transpose imag map
  */
  transTime -= MPI_Wtime();
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
      //compute number of rings on task i
      j = 2*(plan.lastRingTasks[i] - plan.firstRingTasks[i] + 1);
      if(plan.lastRingTasks[i] == 2*Nside)
	--j;
      
      sendcnts[i] = NMThisTask*Nmaps*j;
    }
  sdispls[0] = 0;
  for(i=1;i<NTasks;++i)
    sdispls[i] = sdispls[i-1] + sendcnts[i-1];
  
  MPI_Alltoall(sendcnts,1,MPI_INT,recvcnts,1,MPI_INT,MPI_COMM_WORLD);
  
  rdispls[0] = 0;
  for(i=1;i<NTasks;++i)
    rdispls[i] = rdispls[i-1] + recvcnts[i-1];
  
  NrTDR = 2*(plan.lastRingTasks[ThisTask] - plan.firstRingTasks[ThisTask] + 1);
  if(plan.lastRingTasks[ThisTask] == 2*Nside)
    --NrTDR;
  NmTDR = plan.lmax+1;
  
  transDataRealRecv = (double*)malloc(sizeof(double)*NrTDR*Nmaps*NmTDR);
  assert(transDataRealRecv != NULL);
  memTot += sizeof(double)*NrTDR*Nmaps*NmTDR;
  
  transDataImagRecv = (double*)malloc(sizeof(double)*NrTDR*Nmaps*NmTDR);
  assert(transDataImagRecv != NULL);
  memTot += sizeof(double)*NrTDR*Nmaps*NmTDR;
  
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
  memTot -= sizeof(double)*NrTD*Nmaps*NmTD;
  free(transDataImag);
  memTot -= sizeof(double)*NrTD*Nmaps*NmTD;
  
  /* this version uses MPI_Alltoallv - the version above seems to be faster
     transDataRealRecv = (double*)malloc(sizeof(double)*NrTDR*Nmaps*NmTDR);
     assert(transDataRealRecv != NULL);
     memTot += sizeof(double)*NrTDR*Nmaps*NmTDR;
     
     #ifdef DEBUG //mem high water for this part happens here
     #if DEBUG_LEVEL > 0  
     fprintf(stderr,"%f MB allocated in alm2allmaps_mpi.\n",((double) (sizeof(double)*(memTot)))/1024.0/1024.0);
     #endif
     #endif  
     
     MPI_Alltoallv(transDataReal,sendcnts,sdispls,MPI_DOUBLE,transDataRealRecv,recvcnts,rdispls,MPI_DOUBLE,MPI_COMM_WORLD); 
     free(transDataReal);
     memTot -= sizeof(double)*NrTD*Nmaps*NmTD;
     
     transDataImagRecv = (double*)malloc(sizeof(double)*NrTDR*Nmaps*NmTDR);
     assert(transDataImagRecv != NULL);
     memTot += sizeof(double)*NrTDR*Nmaps*NmTDR;
     MPI_Alltoallv(transDataImag,sendcnts,sdispls,MPI_DOUBLE,transDataImagRecv,recvcnts,rdispls,MPI_DOUBLE,MPI_COMM_WORLD); 
     free(transDataImag);
     memTot -= sizeof(double)*NrTD*Nmaps*NmTD;
  */
  
  free(sendcnts);
  free(sdispls);
  free(recvcnts);
  memTot -= 3*sizeof(int)*NTasks;
    
  transTime += MPI_Wtime();
  
  //now sort rings into their proper pixels
  pixTime -= MPI_Wtime();
  
  double *ringTime;
  ringTime = (double*)malloc(sizeof(double)*NringsThisTask);
  memTot += sizeof(double)*NringsThisTask;
  assert(ringTime != NULL);
  for(i=0;i<NringsThisTask;++i)
    ringTime[i] = 0.0;
  
#ifdef DEBUG
#if DEBUG_LEVEL > 0  
  fprintf(stderr,"%f MB allocated in alm2allmaps_mpi.\n",((double) (sizeof(double)*(memTot)))/1024.0/1024.0);
#endif
#endif

  //zero mapvecs
  for(nring=firstRing;nring<=lastLoopRing;++nring)
    {
      ringTime[nring-firstRing] -= MPI_Wtime();
      
      get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
      ringpix_complex = ringpix/2+1;
      
      for(mapNum=0;mapNum<6;++mapNum)
	for(i=0;i<ringpix_complex;++i)
	  {
	    //north rings
	    mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+i][0] = 0.0;
	    mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+i][1] = 0.0;
	    
	    //south rings
	    mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]+i][0] = 0.0;
	    mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]+i][1] = 0.0;
	  }
      
      ringTime[nring-firstRing] += MPI_Wtime();
    }  
  
  if(lastRing == Nrings)
    {
      ringTime[lastRing-firstRing] -= MPI_Wtime();
      
      get_ring_info2(lastRing,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
      ringpix_complex = ringpix/2+1;
      
      for(mapNum=0;mapNum<6;++mapNum)
	for(i=0;i<ringpix_complex;++i)
	  {
	    mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+i][0] = 0.0;
	    mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+i][1] = 0.0;
	  }
      
      ringTime[lastRing-firstRing] += MPI_Wtime();
    }
  
  //unpack map rings
  for(task=0;task<NTasks;++task)
    {
      start = rdispls[task];
      NmT = plan.lastMTasks[task] - plan.firstMTasks[task] + 1;
      
      for(i=0;i<NringsLoop;++i)
	{
	  nring = i + firstRing;
	  
	  ringTime[nring-firstRing] -= MPI_Wtime();
	  
	  get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
	  ringpix_complex = ringpix/2+1;
	  
	  for(k=0;k<NmT;++k)
	    {
	      m = k + plan.firstMTasks[task];
	      
	      //positive m
	      mp = m%ringpix;
	      if(mp < ringpix_complex)
		{
		  l = (m-mp)/ringpix;
		  if(shifted && (l%2))
		    skfact = -1.0;
		  else
		    skfact = 1.0;
		  
		  for(mapNum=0;mapNum<6;++mapNum)
		    {
		      nind = start + ((2*(nring-firstRing))*Nmaps + mapNum)*NmT;
		      sind = start + ((2*(nring-firstRing)+1)*Nmaps + mapNum)*NmT;
		      mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+mp][0] += transDataRealRecv[nind+k]*skfact;
		      mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+mp][1] += transDataImagRecv[nind+k]*skfact;
		      mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]+mp][0] += transDataRealRecv[sind+k]*skfact;
		      mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]+mp][1] += transDataImagRecv[sind+k]*skfact;
		    }
		}
	      
	      //negative m
	      if(m > 0)
		{
		  mp = ringpix - 1 - ((m-1)%(ringpix));
		  //assert(mp >= 0 && mp < ringpixRingChunk[i]); //this is true 
		  
		  if(mp < ringpix_complex)
		    {
		      //assert(mp < ringpix_complex) //this needs to be true and should be
		      l = (-m - mp)/ringpix;
		      if(shifted && (l%2))
			skfact = -1.0;
		      else
			skfact = 1.0;
		      
		      for(mapNum=0;mapNum<6;++mapNum)
			{
			  nind = start + ((2*(nring-firstRing))*Nmaps + mapNum)*NmT;
			  sind = start + ((2*(nring-firstRing)+1)*Nmaps + mapNum)*NmT;
			  mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+mp][0] += transDataRealRecv[nind+k]*skfact;
			  mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+mp][1] -= transDataImagRecv[nind+k]*skfact;
			  mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]+mp][0] += transDataRealRecv[sind+k]*skfact;
			  mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]+mp][1] -= transDataImagRecv[sind+k]*skfact;
			}
		    }
		}
	    }//for(k=0;k<NmT;++k)
	
	  ringTime[nring-firstRing] += MPI_Wtime();
	}//for(i=0;i<NringsLoop;++i)

      //do last ring if needed
      if(lastRing == 2*Nside)
	{
	  nring = lastRing;
	  
	  ringTime[nring-firstRing] -= MPI_Wtime();
	  
	  get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
	  ringpix_complex = ringpix/2+1;
	  
	  for(k=0;k<NmT;++k)
	    {
	      m = k + plan.firstMTasks[task];
	      
	      //positive m
	      mp = m%ringpix;
	      if(mp < ringpix_complex)
		{
		  l = (m-mp)/ringpix;
		  if(shifted && (l%2))
		    skfact = -1.0;
		  else
		    skfact = 1.0;
		  
		  for(mapNum=0;mapNum<6;++mapNum)
		    {
		      nind = start + ((2*(nring-firstRing))*Nmaps + mapNum)*NmT;
		      mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+mp][0] += transDataRealRecv[nind+k]*skfact;
		      mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+mp][1] += transDataImagRecv[nind+k]*skfact;
		    }
		}
	      
	      //negative m
	      if(m > 0)
		{
		  mp = ringpix - 1 - ((m-1)%(ringpix));
		  //assert(mp >= 0 && mp < ringpixRingChunk[i]); //this is true 
		  
		  if(mp < ringpix_complex)
		    {
		      //assert(mp < ringpix_complex) //this needs to be true and should be
		      l = (-m - mp)/ringpix;
		      if(shifted && (l%2))
			skfact = -1.0;
		      else
			skfact = 1.0;
		      
		      for(mapNum=0;mapNum<6;++mapNum)
			{
			  nind = start + ((2*(nring-firstRing))*Nmaps + mapNum)*NmT;
			  mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+mp][0] += transDataRealRecv[nind+k]*skfact;
			  mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]+mp][1] -= transDataImagRecv[nind+k]*skfact;
			}
		    }
		}
	    }//for(k=0;k<NmT;++k)
	  
	  ringTime[nring-firstRing] += MPI_Wtime();
	}//if(lastRing == 2*Nside)
    }//for(task=0;task<NTasks;++task)
  
  free(transDataRealRecv);
  memTot -= sizeof(double)*NrTDR*Nmaps*NmTDR;
  free(transDataImagRecv);
  memTot -= sizeof(double)*NrTDR*Nmaps*NmTDR;
  free(rdispls);
  memTot -= sizeof(int)*NTasks;
  
  pixTime += MPI_Wtime();
  
  /* debugging code
     i = 0;
     for(m=plan.firstMTasks[ThisTask];m<=plan.lastMTasks[ThisTask];++m)
     for(l=m;l<=plan.lmax;++l)
     {
     if(isinf(alm_real[i]) || isnan(alm_real[i]) ||
     isinf(alm_imag[i]) || isnan(alm_imag[i]))
     {
     fprintf(stderr,"alm is nan or inf: l,m = %ld|%ld, value = %lg|%lg\n",l,m,alm_real[i],alm_imag[i]);
     assert(0);
     }
     ++i;
     }
     
     for(nring=firstRing;nring<=lastLoopRing;++nring)
     {
     get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
     ringpix = ringpix/2+1;
     
     for(mapNum=0;mapNum<6;++mapNum)
     for(i=0;i<ringpix;++i)
     if(isnan(mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]][0]) || 
     isinf(mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]][0]) || 
     isnan(mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]][1]) || 
     isinf(mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]][1]))
     {
     fprintf(stderr,"complex mapvec is nan or inf: mapNum = %ld, north ring,ind = %ld|%ld, value = %lg|%lg\n",mapNum,nring,i,
     mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]][0],
     mapvec_complex[mapNum][plan.northStartIndMapvec[nring-firstRing]][1]);
     assert(0);
     }
     
     for(mapNum=0;mapNum<6;++mapNum)
     for(i=0;i<ringpix;++i)
     if(isnan(mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]][0]) || 
     isinf(mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]][0]) || 
     isnan(mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]][1]) || 
     isinf(mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]][1]))
     {
     fprintf(stderr,"complex mapvec is nan or inf: mapNum = %ld, south ring,ind = %ld|%ld, value = %lg|%lg\n",mapNum,nring,i,
     mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]][0],
     mapvec_complex[mapNum][plan.southStartIndMapvec[nring-firstRing]][1]);
     assert(0);
     }
     }
     
     if(lastRing == Nrings)
     {
     get_ring_info2(lastRing,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
     ringpix = ringpix/2+1;
     
     for(mapNum=0;mapNum<6;++mapNum)
     for(i=0;i<ringpix;++i)
     if(isnan(mapvec_complex[mapNum][plan.northStartIndMapvec[lastRing-firstRing]][0]) || 
     isinf(mapvec_complex[mapNum][plan.northStartIndMapvec[lastRing-firstRing]][0]) || 
     isnan(mapvec_complex[mapNum][plan.northStartIndMapvec[lastRing-firstRing]][1]) || 
     isinf(mapvec_complex[mapNum][plan.northStartIndMapvec[lastRing-firstRing]][1]))
     {
     fprintf(stderr,"complex mapvec is nan or inf: mapNum = %ld, north ring,ind = %ld|%ld, value = %lg|%lg\n",mapNum,lastRing,i,
     mapvec_complex[mapNum][plan.northStartIndMapvec[lastRing-firstRing]][0],
     mapvec_complex[mapNum][plan.northStartIndMapvec[lastRing-firstRing]][1]);
     assert(0);
     }
     }
     end of debugging code 
  */
  
  //finally do ring FFTs
  fftTime -= MPI_Wtime();

#ifdef DEBUG
#if DEBUG_LEVEL > 0  
  fprintf(stderr,"%f MB allocated in alm2allmaps_mpi.\n",((double) (sizeof(double)*(memTot)))/1024.0/1024.0);
#endif
#endif
  
  for(nring=firstRing;nring<=lastLoopRing;++nring)
    {
      ringTime[nring-firstRing] -= MPI_Wtime();
      
      get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
      
      for(mapNum=0;mapNum<6;++mapNum)
	{
	  mapvec = (float*) (mapvec_complex[mapNum] + plan.northStartIndMapvec[nring-firstRing]);
	  ring_synthesis(ringpix,shifted,mapvec);
	  
	  if(mapNum == 2 || mapNum == 5 || mapNum == 4)
	    for(i=0;i<ringpix;++i)
	      mapvec[i] /= nsintheta;
	  
	  if(mapNum == 5)
	    for(i=0;i<ringpix;++i)
	      mapvec[i] /= nsintheta;
	  
	  mapvec = (float*) (mapvec_complex[mapNum] + plan.southStartIndMapvec[nring-firstRing]);
	  ring_synthesis(ringpix,shifted,mapvec);
	  
	  if(mapNum == 2 || mapNum == 5 || mapNum == 4)
	    for(i=0;i<ringpix;++i)
	      mapvec[i] /= nsintheta;

	  if(mapNum == 5)
	    for(i=0;i<ringpix;++i)
	      mapvec[i] /= nsintheta;	  
	}
      
      ringTime[nring-firstRing] += MPI_Wtime();
    }
  
  if(lastRing == Nrings)
    {
      ringTime[lastRing-firstRing] -= MPI_Wtime();
      
      get_ring_info2(lastRing,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
      
      for(mapNum=0;mapNum<6;++mapNum)
	{
	  mapvec = (float*) (mapvec_complex[mapNum] + plan.northStartIndMapvec[lastRing-firstRing]);
	  ring_synthesis(ringpix,shifted,mapvec);
	  
	  if(mapNum == 2 || mapNum == 5 || mapNum == 4)
	    for(i=0;i<ringpix;++i)
	      mapvec[i] /= nsintheta;
	  
	  if(mapNum == 5)
	    for(i=0;i<ringpix;++i)
	      mapvec[i] /= nsintheta;
	}
      
      ringTime[lastRing-firstRing] += MPI_Wtime();
    }
  fftTime += MPI_Wtime();
  
  mapvec = (float*) (mapvec_complex[0]);
  
  //compute final derivs
  float *mvtp,*mvpp,*mvt,*mvp;
  
  for(nring=firstRing;nring<=lastLoopRing;++nring)
    {
      ringTime[nring-firstRing] -= MPI_Wtime();
      
      get_ring_info2(nring,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
      
      mvt = (float*) (mapvec_complex[1] + plan.northStartIndMapvec[nring-firstRing]);
      mvp = (float*) (mapvec_complex[2] + plan.northStartIndMapvec[nring-firstRing]);
      mvtp = (float*) (mapvec_complex[4] + plan.northStartIndMapvec[nring-firstRing]);
      mvpp = (float*) (mapvec_complex[5] + plan.northStartIndMapvec[nring-firstRing]);
      
      for(i=0;i<ringpix;++i)
	{
	  mvtp[i] = (float) (mvtp[i] - ncostheta/nsintheta*mvp[i]);
	  mvpp[i] = (float) (mvpp[i] + ncostheta/nsintheta*mvt[i]);
	}
      
      mvt = (float*) (mapvec_complex[1] + plan.southStartIndMapvec[nring-firstRing]);
      mvp = (float*) (mapvec_complex[2] + plan.southStartIndMapvec[nring-firstRing]);
      mvtp = (float*) (mapvec_complex[4] + plan.southStartIndMapvec[nring-firstRing]);
      mvpp = (float*) (mapvec_complex[5] + plan.southStartIndMapvec[nring-firstRing]);
      
      for(i=0;i<ringpix;++i)
	{
	  //the sign of costheta swaps for the southern rings
	  mvtp[i] = (float) (mvtp[i] + ncostheta/nsintheta*mvp[i]);
	  mvpp[i] = (float) (mvpp[i] - ncostheta/nsintheta*mvt[i]);
	}
      
      ringTime[nring-firstRing] += MPI_Wtime();
    }
  
  if(lastRing == Nrings)
    {
      ringTime[lastRing-firstRing] -= MPI_Wtime();
      
      get_ring_info2(lastRing,&nstartpix,&ringpix,&ncostheta,&nsintheta,&shifted,plan.order);
      
      mvt = (float*) (mapvec_complex[1] + plan.northStartIndMapvec[lastRing-firstRing]);
      mvp = (float*) (mapvec_complex[2] + plan.northStartIndMapvec[lastRing-firstRing]);
      mvtp = (float*) (mapvec_complex[4] + plan.northStartIndMapvec[lastRing-firstRing]);
      mvpp = (float*) (mapvec_complex[5] + plan.northStartIndMapvec[lastRing-firstRing]);
      
      for(i=0;i<ringpix;++i)
	{
	  mvtp[i] = (float) (mvtp[i] - ncostheta/nsintheta*mvp[i]);
	  mvpp[i] = (float) (mvpp[i] + ncostheta/nsintheta*mvt[i]);
	}
      
      ringTime[lastRing-firstRing] += MPI_Wtime();
    }
    
  runTime += MPI_Wtime();
  
#ifdef OUTPUT_SHT_LOADBALANCE
  MPI_Reduce(&runTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&runTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&runTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"alm -> all maps                     run time max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);  

  MPI_Reduce(&plmTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&plmTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&plmTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"alm -> map plm time and load balance: max,min,avg = %f|%f|%f (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
  
  MPI_Reduce(&sumTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&sumTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&sumTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"alm -> map sum time and load balance: max,min,avg = %f|%f|%f (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
  
  MPI_Reduce(&transTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&transTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&transTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"alm -> map trans time and load balance: max,min,avg = %f|%f|%f (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
  
  MPI_Reduce(&pixTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&pixTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&pixTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"alm -> map unpack time and load balance: max,min,avg = %f|%f|%f (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
  
  MPI_Reduce(&fftTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&fftTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&fftTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"alm -> map fft time and load balance: max,min,avg = %f|%f|%f (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
#endif
    
  //get load balance info
  for(j=0;j<NMThisTask;++j)
    alm2mapMTimesGlobal[j+firstM] += mTime[j];
  free(mTime);
  memTot -= sizeof(double)*NMThisTask;
  
  for(j=0;j<NringsThisTask;++j)
    map2almRingTimesGlobal[j+firstRing-1] += ringTime[j];
  free(ringTime);
  memTot -= sizeof(double)*NringsThisTask;
  
  assert(memTot == 0);
  
#ifdef OUTPUT_SHT_LOADBALANCE 
  FILE *fp;
  char fname[1024];
  double *localTimes,*globalTimes;
  long NlocalTimes;
  
  NlocalTimes = 1 + order2lmax(plan.order);
  localTimes = (double*)malloc(sizeof(double)*NlocalTimes);
  assert(localTimes != NULL);
  globalTimes = (double*)malloc(sizeof(double)*NlocalTimes);
  assert(globalTimes != NULL);
  for(i=0;i<Nalm2mapMTimesGlobal;++i)
    localTimes[i] = 0.0;
  for(i=0;i<Nalm2mapMTimesGlobal;++i)
    localTimes[i] = alm2mapMTimesGlobal[i];

  MPI_Allreduce(localTimes,globalTimes,(int) Nalm2mapMTimesGlobal,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  
  if(ThisTask == 0)
    {
      sprintf(fname,"./alm2allmaps_mtimes.%ld",plan.order);
      fp = fopen(fname,"w");
      assert(fp != NULL);
      for(i=0;i<Nalm2mapMTimesGlobal;++i)
	fprintf(fp,"%ld %le\n",i,globalTimes[i]);
      fclose(fp);
    }
  
  free(localTimes);
  free(globalTimes);
#endif
}

