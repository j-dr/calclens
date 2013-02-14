#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>

#include "raytrace.h"

void generate_rotmat_axis_angle_countercw(double axis[3], double angle, double rotmat[3][3])
{
  double cosangle,sinangle;
  sinangle = sin(angle);
  cosangle = cos(angle);
  int i,j;
  
  assert(fabs(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2] - 1.0) < 1e-15);
  
  for(i=0;i<3;++i)
    for(j=0;j<3;++j)
      rotmat[i][j] = 0.0;
  
  rotmat[0][0] = cosangle;
  rotmat[1][1] = cosangle;
  rotmat[2][2] = cosangle;
  
  for(i=0;i<3;++i)
    for(j=0;j<3;++j)
      rotmat[i][j] += axis[i]*axis[j]*(1.0 - cosangle);
  
  rotmat[0][1] -= axis[2]*sinangle;
  rotmat[0][2] += axis[1]*sinangle;
  rotmat[1][2] -= axis[0]*sinangle;
  
  rotmat[1][0] += axis[2]*sinangle;
  rotmat[2][0] -= axis[1]*sinangle;
  rotmat[2][1] += axis[0]*sinangle;
}

void generate_rotmat_axis_angle_cw(double axis[3], double angle, double rotmat[3][3])
{
  //just call function above with angle -> -1*angle
  generate_rotmat_axis_angle_countercw(axis,-1.0*angle,rotmat);
}

void rot_vec_axis_angle_countercw(double vec[3], double rvec[3], double axis[3], double angle)
{
  double cosangle,sinangle;
  double axisdotvec,axiscrossvec[3];
  
  assert(fabs(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2] - 1.0) < 1e-15);
  
  sinangle = sin(angle);
  cosangle = cos(angle);
  axisdotvec = axis[0]*vec[0] + axis[1]*vec[1] + axis[2]*vec[2];
  axiscrossvec[0] = axis[1]*vec[2] - axis[2]*vec[1];
  axiscrossvec[1] = axis[2]*vec[0] - axis[0]*vec[2];
  axiscrossvec[2] = axis[0]*vec[1] - axis[1]*vec[0];
  
  rvec[0] = vec[0]*cosangle + axis[0]*axisdotvec*(1.0 - cosangle) + axiscrossvec[0]*sinangle;
  rvec[1] = vec[1]*cosangle + axis[1]*axisdotvec*(1.0 - cosangle) + axiscrossvec[1]*sinangle;
  rvec[2] = vec[2]*cosangle + axis[2]*axisdotvec*(1.0 - cosangle) + axiscrossvec[2]*sinangle;
}

void rot_vec_axis_angle_cw(double vec[3], double rvec[3], double axis[3], double angle)
{
  //just call function above with angle -> -1*angle
  rot_vec_axis_angle_countercw(vec,rvec,axis,-1.0*angle);
}

void rot_vec_axis_trigangle_countercw(double vec[3], double rvec[3], double axis[3], double cosangle, double sinangle)
{
  double axisdotvec,axiscrossvec[3];
  
  assert(fabs(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2] - 1.0) < 1e-15);
  
  axisdotvec = axis[0]*vec[0] + axis[1]*vec[1] + axis[2]*vec[2];
  axiscrossvec[0] = axis[1]*vec[2] - axis[2]*vec[1];
  axiscrossvec[1] = axis[2]*vec[0] - axis[0]*vec[2];
  axiscrossvec[2] = axis[0]*vec[1] - axis[1]*vec[0];
  
  rvec[0] = vec[0]*cosangle + axis[0]*axisdotvec*(1.0 - cosangle) + axiscrossvec[0]*sinangle;
  rvec[1] = vec[1]*cosangle + axis[1]*axisdotvec*(1.0 - cosangle) + axiscrossvec[1]*sinangle;
  rvec[2] = vec[2]*cosangle + axis[2]*axisdotvec*(1.0 - cosangle) + axiscrossvec[2]*sinangle;
}

/* parallel transport a vector on the sphere along the great circle connecting vec to rvec
   this function is based on the healpix package rotate_coord.pro IDL routine
   tvec is the tangent vector on the sphere where tvec[0] points along theta unit vector and tvec[1] points along phi unit vector
   vec is the location on the sphere of this tangent vector
   rvec is position to which tvec is to be parallel transported from vec
   rtvec is the parallel transported vector
*/
void paratrans_tangvec(double tvec[2], double _vec[3], double _rvec[3], double rtvec[2])
{
  double cospsi,sinpsi,norm,axis[3],cosangle,sinangle,p[3];
  double rephi_vec[3],etheta_rvec[3],ephi_rvec[3];
  double norm_rvec,norm_vec;
  double vec[3],rvec[3];
  
  //get rotation info
  norm_vec = sqrt(_vec[0]*_vec[0] + _vec[1]*_vec[1] + _vec[2]*_vec[2]);
  vec[0] = _vec[0]/norm_vec;
  vec[1] = _vec[1]/norm_vec;
  vec[2] = _vec[2]/norm_vec;
  
  norm_rvec = sqrt(_rvec[0]*_rvec[0] + _rvec[1]*_rvec[1] + _rvec[2]*_rvec[2]);
  rvec[0] = _rvec[0]/norm_rvec;
  rvec[1] = _rvec[1]/norm_rvec;
  rvec[2] = _rvec[2]/norm_rvec;
  
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
    
  //debugging fprintf statements
  //fprintf(stderr,"cospsi = %f, sinpsi = %f\n",cospsi,sinpsi);
  
  /* psi is defined as
     R(e_theta) = cos(psi) e_theta' - sin(psi) e_phi'
     R(e_phi)   = sin(psi) e_theta' + cos(psi) e_phi'
     
     thus to rotate tangent vector
     t = t_theta R(e_theta) + t_phi R(e_phi)
     
     we plug and chug to get
     t = (t_theta*cos(psi) + t_phi*sin(psi)) e_theta' + (-t_theta*sin(psi) + t_phi*cos(psi)) e_phi' 
  */
  rtvec[0] = tvec[0]*cospsi + tvec[1]*sinpsi;
  rtvec[1] = -1.0*tvec[0]*sinpsi + tvec[1]*cospsi;
}

/* parallel transport a tensor on the sphere along the great circle connecting vec to rvec
   this function is based on the healpix package rotate_coord.pro IDL routine
   ttensor is a tensor on the sphere where the i,j component has basis vectors with 0 = e_theta, and 1 = e_phi for i,j in {0,1} 
   vec is the location on the sphere of this tensor
   rvec is position to which ttensor is to be parallel transported from vec
   rttensor is the parallel transported vector
*/
void paratrans_tangtensor(double ttensor[2][2], double _vec[3], double _rvec[3], double rttensor[2][2])
{
  double cospsi,sinpsi,norm,axis[3],cosangle,sinangle,p[3];
  double rephi_vec[3],etheta_rvec[3],ephi_rvec[3];
  double norm_rvec,norm_vec;
  double vec[3],rvec[3];
  
  //get rotation info
  norm_vec = sqrt(_vec[0]*_vec[0] + _vec[1]*_vec[1] + _vec[2]*_vec[2]);
  vec[0] = _vec[0]/norm_vec;
  vec[1] = _vec[1]/norm_vec;
  vec[2] = _vec[2]/norm_vec;
  
  norm_rvec = sqrt(_rvec[0]*_rvec[0] + _rvec[1]*_rvec[1] + _rvec[2]*_rvec[2]);
  rvec[0] = _rvec[0]/norm_rvec;
  rvec[1] = _rvec[1]/norm_rvec;
  rvec[2] = _rvec[2]/norm_rvec;
  
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

  //debugging fprintf statements
  //fprintf(stderr,"cospsi = %f, sinpsi = %f\n",cospsi,sinpsi);
  
 /* psi is defined as
     R(e_theta) = cos(psi) e_theta' - sin(psi) e_phi'
     R(e_phi)   = sin(psi) e_theta' + cos(psi) e_phi'
     
     under this coordinate change we have that 
     
     T' = R x T x Transpose(R)
     where 
     T = | t_00 t_01 |
         | t_10 t_11 |
	 
     R = |  cos(psi) sin(psi) |
         | -sin(psi) cos(psi) |
  */
 
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
      t1[i][j] = ttensor[i][0]*r[0][j]  + ttensor[i][1]*r[1][j];
  
  for(i=0;i<2;++i)
    for(j=0;j<2;++j)
      rttensor[i][j] = rt[i][0]*t1[0][j]  + rt[i][1]*t1[1][j];
}

// parallel transports a ray to the observer position from its current position                                                                                                                                                                                            
void paratrans_ray_curr2obs(HEALPixRay *ray)
{
  double ttensor[2][2],rttensor[2][2];
  double obs[3];

  //get obseerver position                                                                                                                                                                                                                                            
  nest2vec(ray->nest,obs,rayTraceData.rayOrder);

  //do the tensors                                                                                                                                                                                                                                         
  ttensor[0][0] = ray->Aprev[2*0+0];
  ttensor[0][1] = ray->Aprev[2*0+1];
  ttensor[1][0] = ray->Aprev[2*1+0];
  ttensor[1][1] = ray->Aprev[2*1+1];
  paratrans_tangtensor(ttensor,ray->n,obs,rttensor);
  ray->Aprev[2*0+0] = rttensor[0][0];
  ray->Aprev[2*0+1] = rttensor[0][1];
  ray->Aprev[2*1+0] = rttensor[1][0];
  ray->Aprev[2*1+1] = rttensor[1][1];

  ttensor[0][0] = ray->A[2*0+0];
  ttensor[0][1] = ray->A[2*0+1];
  ttensor[1][0] = ray->A[2*1+0];
  ttensor[1][1] = ray->A[2*1+1];
  paratrans_tangtensor(ttensor,ray->n,obs,rttensor);
  ray->A[2*0+0] = rttensor[0][0];
  ray->A[2*0+1] = rttensor[0][1];
  ray->A[2*1+0] = rttensor[1][0];
  ray->A[2*1+1] = rttensor[1][1];
}

// parallel transports a ray to the current position from its observer position
void paratrans_ray_obs2curr(HEALPixRay *ray)
{
  double ttensor[2][2],rttensor[2][2];
  double obs[3];
  
  //get obseerver position 
  nest2vec(ray->nest,obs,rayTraceData.rayOrder);
  
  //do the tensors
  ttensor[0][0] = ray->Aprev[2*0+0];
  ttensor[0][1] = ray->Aprev[2*0+1];
  ttensor[1][0] = ray->Aprev[2*1+0];
  ttensor[1][1] = ray->Aprev[2*1+1];
  paratrans_tangtensor(ttensor,obs,ray->n,rttensor);
  ray->Aprev[2*0+0] = rttensor[0][0];
  ray->Aprev[2*0+1] = rttensor[0][1];
  ray->Aprev[2*1+0] = rttensor[1][0];
  ray->Aprev[2*1+1] = rttensor[1][1];
      
  ttensor[0][0] = ray->A[2*0+0];
  ttensor[0][1] = ray->A[2*0+1];
  ttensor[1][0] = ray->A[2*1+0];
  ttensor[1][1] = ray->A[2*1+1];
  paratrans_tangtensor(ttensor,obs,ray->n,rttensor);
  ray->A[2*0+0] = rttensor[0][0];
  ray->A[2*0+1] = rttensor[0][1];
  ray->A[2*1+0] = rttensor[1][0];
  ray->A[2*1+1] = rttensor[1][1];
}

// converts a ray from ra-dec basis to theta-phi basis
void rot_ray_radec2ang(HEALPixRay *ray)
{
  double Ard[2][2];
  double alphard[2];
  
  alphard[0] = ray->alpha[0];
  alphard[1] = ray->alpha[1];
  ray->alpha[0] = -1.0*alphard[1];
  ray->alpha[1] = alphard[0];
  
  Ard[0][0] = ray->A[2*0+0];
  Ard[1][0] = ray->A[2*1+0];
  Ard[0][1] = ray->A[2*0+1];
  Ard[1][1] = ray->A[2*1+1];
  ray->A[2*0+0] = Ard[1][1];
  ray->A[2*1+0] = -1.0*Ard[0][1];
  ray->A[2*0+1] = -1.0*Ard[1][0];
  ray->A[2*1+1] = Ard[0][0];
  
  Ard[0][0] = ray->Aprev[2*0+0];
  Ard[1][0] = ray->Aprev[2*1+0];
  Ard[0][1] = ray->Aprev[2*0+1];
  Ard[1][1] = ray->Aprev[2*1+1];
  ray->Aprev[2*0+0] = Ard[1][1];
  ray->Aprev[2*1+0] = -1.0*Ard[0][1];
  ray->Aprev[2*0+1] = -1.0*Ard[1][0];
  ray->Aprev[2*1+1] = Ard[0][0];
  
  Ard[0][0] = ray->U[2*0+0];
  Ard[1][0] = ray->U[2*1+0];
  Ard[0][1] = ray->U[2*0+1];
  Ard[1][1] = ray->U[2*1+1];
  ray->U[2*0+0] = Ard[1][1];
  ray->U[2*1+0] = -1.0*Ard[0][1];
  ray->U[2*0+1] = -1.0*Ard[1][0];
  ray->U[2*1+1] = Ard[0][0];
}

// converts a ray from ra-dec basis to theta-phi basis
void rot_ray_ang2radec(HEALPixRay *ray)
{
  double Atp[2][2];
  double alphatp[2];
  
  alphatp[0] = ray->alpha[0];
  alphatp[1] = ray->alpha[1];
  ray->alpha[0] = alphatp[1];
  ray->alpha[1] = -1.0*alphatp[0];
  
  Atp[0][0] = ray->A[2*0+0];
  Atp[1][0] = ray->A[2*1+0];
  Atp[0][1] = ray->A[2*0+1];
  Atp[1][1] = ray->A[2*1+1];
  ray->A[2*0+0] = Atp[1][1];
  ray->A[2*1+0] = -1.0*Atp[0][1];
  ray->A[2*0+1] = -1.0*Atp[1][0];
  ray->A[2*1+1] = Atp[0][0];

  Atp[0][0] = ray->Aprev[2*0+0];
  Atp[1][0] = ray->Aprev[2*1+0];
  Atp[0][1] = ray->Aprev[2*0+1];
  Atp[1][1] = ray->Aprev[2*1+1];
  ray->Aprev[2*0+0] = Atp[1][1];
  ray->Aprev[2*1+0] = -1.0*Atp[0][1];
  ray->Aprev[2*0+1] = -1.0*Atp[1][0];
  ray->Aprev[2*1+1] = Atp[0][0];
  
  Atp[0][0] = ray->U[2*0+0];
  Atp[1][0] = ray->U[2*1+0];
  Atp[0][1] = ray->U[2*0+1];
  Atp[1][1] = ray->U[2*1+1];
  ray->U[2*0+0] = Atp[1][1];
  ray->U[2*1+0] = -1.0*Atp[0][1];
  ray->U[2*0+1] = -1.0*Atp[1][0];
  ray->U[2*1+1] = Atp[0][0];
}
