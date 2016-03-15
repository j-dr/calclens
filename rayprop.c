#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <float.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_sort_long.h>

#include "raytrace.h"

/*
  propagates rays using lensing potential phi 
*/
void rayprop_sphere(double wp, double wpm1, double wpm2, long bundleCellInd)
{
  long i,n,m;
  HEALPixRay *ray;
  double Ap[4],r;
  
#ifndef BORNAPPRX
  double np[3];
  double betap[3];
  double thetahat[3],phihat[3],ncrossa[3];
  double a[3];
  double R[3][3];
  double lambda,norm,alpha;
  double q,qa,qc,qb;
  double ttensor[2][2];
  double rttensor[2][2];
#endif

  for(i=0;i<bundleCells[bundleCellInd].Nrays;++i)
    {
      ray = &(bundleCells[bundleCellInd].rays[i]);
      
#ifdef BORNAPPRX
      //change pos
      ray->n[0] = ray->n[0]/wpm1*wp;
      ray->n[1] = ray->n[1]/wpm1*wp;
      ray->n[2] = ray->n[2]/wpm1*wp;
      
      //shift ray info
      for(n=0;n<2;++n)
	for(m=0;m<2;++m)
	  Ap[m + 2*n] =
	    (1.0 - wpm1*(wp - wpm2)/wp/(wpm1-wpm2))*ray->Aprev[m + 2*n]
	    + (wpm1*(wp - wpm2)/wp/(wpm1-wpm2))*ray->A[m + 2*n]
	    - ((wp-wpm1)/wp)*(ray->U[m + 2*n]);
      
      ray->Aprev[0] = ray->A[0];
      ray->Aprev[1] = ray->A[1];
      ray->Aprev[2] = ray->A[2];
      ray->Aprev[3] = ray->A[3];
      ray->A[0] = Ap[0];
      ray->A[1] = Ap[1];
      ray->A[2] = Ap[2];
      ray->A[3] = Ap[3];
      
#else
      alpha = sqrt(ray->alpha[0]*ray->alpha[0] + ray->alpha[1]*ray->alpha[1]);
      
      if(alpha > 0.0)
	{
	  //get theta-phi unit vectors at ray location
	  phihat[0] = -1.0*ray->n[1];
	  phihat[1] = ray->n[0];
	  phihat[2] = 0.0;
	  norm = sqrt(phihat[0]*phihat[0] + phihat[1]*phihat[1]);
	  phihat[0] /= norm;
	  phihat[1] /= norm;
	  
	  thetahat[0] = ray->n[2]*ray->n[0];
	  thetahat[1] = ray->n[2]*ray->n[1];
	  thetahat[2] = -1.0*(ray->n[0]*ray->n[0] + ray->n[1]*ray->n[1]);
	  norm = sqrt(thetahat[0]*thetahat[0] + thetahat[1]*thetahat[1] + thetahat[2]*thetahat[2]);
	  thetahat[0] /= norm;
	  thetahat[1] /= norm;
	  thetahat[2] /= norm;
	  
	  //get alpha in vector form
	  a[0] = ray->alpha[0]*thetahat[0] + ray->alpha[1]*phihat[0];
	  a[1] = ray->alpha[0]*thetahat[1] + ray->alpha[1]*phihat[1];
	  a[2] = ray->alpha[0]*thetahat[2] + ray->alpha[1]*phihat[2];
	  
	  //compute nxa and norm of a
	  ncrossa[0] = ray->n[1]*a[2] - ray->n[2]*a[1];
	  ncrossa[1] = ray->n[2]*a[0] - ray->n[0]*a[2];
	  ncrossa[2] = ray->n[0]*a[1] - ray->n[1]*a[0];
	  norm = sqrt(ncrossa[0]*ncrossa[0] + ncrossa[1]*ncrossa[1] + ncrossa[2]*ncrossa[2]);
	  ncrossa[0] /= norm;
	  ncrossa[1] /= norm;
	  ncrossa[2] /= norm;
	  
	  //now get rot mat for beta
	  generate_rotmat_axis_angle_countercw(ncrossa,alpha,R);
	  
	  //rot beta
	  for(n=0;n<3;++n)
	    {
	      betap[n] = R[n][0]*ray->beta[0];
	      betap[n] += R[n][1]*ray->beta[1];
	      betap[n] += R[n][2]*ray->beta[2];
	    }
	  
	  //get lambda
	  qa = 1.0;
	  qb = 2.0*(ray->n[0]*betap[0] + ray->n[1]*betap[1] + ray->n[2]*betap[2]);
	  qc = wpm1*wpm1 - wp*wp;
	  q = -0.5*(qb + qb/fabs(qb)*sqrt(qb*qb - 4.0*qa*qc));
	  lambda = qc/q;
	  if(lambda < 0.0)
	    lambda = q/qa;
	  
	  //get new ray loc
	  np[0] = ray->n[0] + betap[0]*lambda;
	  np[1] = ray->n[1] + betap[1]*lambda;
	  np[2] = ray->n[2] + betap[2]*lambda;
	}
      else
	{
	  betap[0] = ray->beta[0];
	  betap[1] = ray->beta[1];
	  betap[2] = ray->beta[2];
	  
	  np[0] = ray->n[0]/wpm1*wp;
	  np[1] = ray->n[1]/wpm1*wp;
	  np[2] = ray->n[2]/wpm1*wp;
	}
      
      for(n=0;n<2;++n)
	for(m=0;m<2;++m)
	  Ap[m + 2*n] =
	    (1.0 - wpm1*(wp - wpm2)/wp/(wpm1-wpm2))*ray->Aprev[m + 2*n]
	    + (wpm1*(wp - wpm2)/wp/(wpm1-wpm2))*ray->A[m + 2*n]
	    - ((wp-wpm1)/wp)*(ray->U[0 + 2*n]*ray->A[m + 2*0] + ray->U[1 + 2*n]*ray->A[m + 2*1]);
      
      //shift ray info
      ray->Aprev[0] = ray->A[0];
      ray->Aprev[1] = ray->A[1];
      ray->Aprev[2] = ray->A[2];
      ray->Aprev[3] = ray->A[3];
      ray->A[0] = Ap[0];
      ray->A[1] = Ap[1];
      ray->A[2] = Ap[2];
      ray->A[3] = Ap[3];
      
      //need to parallel transport tensors
      ttensor[0][0] = ray->Aprev[2*0+0];
      ttensor[0][1] = ray->Aprev[2*0+1];
      ttensor[1][0] = ray->Aprev[2*1+0];
      ttensor[1][1] = ray->Aprev[2*1+1];
      paratrans_tangtensor(ttensor,ray->n,np,rttensor);
      ray->Aprev[2*0+0] = rttensor[0][0];
      ray->Aprev[2*0+1] = rttensor[0][1];
      ray->Aprev[2*1+0] = rttensor[1][0];
      ray->Aprev[2*1+1] = rttensor[1][1];
      
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
      
      ray->beta[0] = betap[0];
      ray->beta[1] = betap[1];
      ray->beta[2] = betap[2];
#endif
      
      //make sure ray loc is properly normalized 
      r = sqrt(ray->n[0]*ray->n[0] + ray->n[1]*ray->n[1] + ray->n[2]*ray->n[2]);
      r = wp/r;
      ray->n[0] *= r;
      ray->n[1] *= r;
      ray->n[2] *= r;
    }
}
