/*
 hacked apart healpix base2 and healpix_base functions 
 converted int64 to long to work in C on 64 bit machines 

 -Matthew R Becker, Univ. of Chicago 2009
*/

/*
 *  This file is part of Healpix_cxx.
 *
 *  Healpix_cxx is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  Healpix_cxx is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Healpix_cxx; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix, see http://healpix.jpl.nasa.gov
 */

/*
 *  Healpix_cxx is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file healpix_base2.h
 *  Copyright (C) 2003, 2004, 2005, 2006 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <gsl/gsl_math.h>
#include "healpix_utils.h"

static long ctab[0x100];
static long utab[0x100];
static const long jrll[] = { 2,2,2,2,3,3,3,3,4,4,4,4 };
static const long jpll[] = { 1,3,5,7,0,2,4,6,1,3,5,7 };
static int HEALPIX_TOOLS_INIT = 1;

long isqrt(long i)
{
  return sqrt(((double) (i)) + 0.5);
}

long ilog2(long arg)
{
  unsigned long res=0;
  while(arg > 0x0000FFFF)
    { 
      res += 16;
      arg >>= 16;
    }
  
  if(arg > 0x000000FF)
    { 
      res |= 8; 
      arg >>= 8;
    }
  if(arg > 0x0000000F)
    { 
      res |= 4;
      arg >>= 4;
    }
  if(arg > 0x00000003)
    { 
      res |= 2; 
      arg >>= 2;
    }
  if(arg > 0x00000001)
    {
      res |= 1;
    }
  return res;
}

long imodulo(long v1, long v2)
{ 
  return (v1>=0) ? ((v1<v2) ? v1 : (v1%v2)) : ((v1%v2)+v2);
}

void ang2radec(double theta, double phi, double *ra, double *dec)
{
  *ra = phi/M_PI*180.0;
  *dec = (M_PI/2.0 - theta)/M_PI*180.0;
}

void radec2ang(double *theta, double *phi, double ra, double dec)
{
  *phi = ra/180.0*M_PI;
  *theta = M_PI/2.0 - dec/180.0*M_PI;
}

void vec2radec(double vec[3], double *ra, double *dec)
{
  /*double rad;
  rad = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
  if(vec[0] == 0.0 && vec[1] == 0.0)
    *ra = 0.0;
  else
    *ra = (float) ((vec[1] < 0.0) ? 360.0 + atan2(vec[1],vec[0])*180.0/M_PI : atan2(vec[1],vec[0])*180.0/M_PI);
  *dec = (float) (90.0-acos(vec[2]/rad)*180.0/M_PI);
  */
  double theta,phi;
  vec2ang(vec,&theta,&phi);
  ang2radec(theta,phi,ra,dec);
}

void vec2ang(double vec[3], double *theta, double *phi)
{
  double norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
  
  if(vec[0] == 0.0 && vec[1] == 0.0)
    *phi = 0.0;
  else
    *phi = atan2(vec[1],vec[0]);
  if(*phi < 0.0) 
    *phi  = *phi + 2.0*M_PI;
  *theta = acos(vec[2]/norm);
}

void ang2vec(double vec[3], double theta, double phi)
{
  double costheta = cos(theta);
  double sintheta = sqrt((1.0 + costheta)*(1.0 - costheta));
  
  vec[0] = sintheta*cos(phi);
  vec[1] = sintheta*sin(phi);
  vec[2] = costheta;
}


void tablefiller(void)
{
  assert(CHAR_BIT == 8);

  int m;
  for (m=0; m<0x100; ++m)
    {
      ctab[m] =
	(m&0x1 )       | ((m&0x2 ) << 7) | ((m&0x4 ) >> 1) | ((m&0x8 ) << 6)
	| ((m&0x10) >> 2) | ((m&0x20) << 5) | ((m&0x40) >> 3) | ((m&0x80) << 4);
      utab[m] =
	(m&0x1 )       | ((m&0x2 ) << 1) | ((m&0x4 ) << 2) | ((m&0x8 ) << 3)
	| ((m&0x10) << 4) | ((m&0x20) << 5) | ((m&0x40) << 6) | ((m&0x80) << 7);
    }
}

long order2npix(long order_)
{
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  return npix_;
}

long order2nside(long order_)
{  
  long nside_ = 1;
  nside_ = nside_ << order_;
  return nside_;
}
  
long npix2nside(long npix)
{
  long res=isqrt(npix/12);
  assert(npix==res*res*12);
  return res;
}

long npix2order(long npix)
{
  return nside2order(npix2nside(npix));
}

long nside2npix(long nside)
{
  assert(nside>0 && !((nside)&(nside-1)));
  return 12*nside*nside;
}

long nside2order(long nside)
{
  assert(nside>0 && !((nside)&(nside-1)));
  return ilog2(nside);
}

void nest2xyf(long pix, long *ix, long *iy, long *face_num, long order_)
{
  if(HEALPIX_TOOLS_INIT)
    {
      tablefiller();
      HEALPIX_TOOLS_INIT = 0;
    }
  
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  assert(pix >= 0 && pix < npix_);
  
  long npface_ = 1;
  npface_ = npface_ << (2*order_);
  
  *face_num = pix>>(2*order_);
  pix &= (npface_-1);
  long raw = ((pix&0x555500000000ull)>>16) 
    | ((pix&0x5555000000000000ull)>>31)
    | (pix&0x5555)
    | ((pix&0x55550000)>>15);
  *ix =  ctab[raw&0xff]
    | (ctab[(raw>>8)&0xff]<<4)
    | (ctab[(raw>>16)&0xff]<<16)
    | (ctab[(raw>>24)&0xff]<<20);
  pix >>= 1;
  raw = ((pix&0x555500000000ull)>>16) 
    | ((pix&0x5555000000000000ull)>>31)
    | (pix&0x5555)
    | ((pix&0x55550000)>>15);
  *iy =  ctab[raw&0xff]
    | (ctab[(raw>>8)&0xff]<<4)
    | (ctab[(raw>>16)&0xff]<<16)
    | (ctab[(raw>>24)&0xff]<<20);
}

long xyf2nest(long ix, long iy, long face_num, long order_)
{
  if(HEALPIX_TOOLS_INIT)
    {
      tablefiller();
      HEALPIX_TOOLS_INIT = 0;
    }
  
  long pix = ((face_num)<<(2*order_)) +
    (   ((utab[ ix     &0xff]))
	| ((utab[(ix>> 8)&0xff])<<16)
	| ((utab[(ix>>16)&0xff])<<32)
	| ((utab[(ix>>24)&0xff])<<48)
	| ((utab[ iy     &0xff])<<1)
	| ((utab[(iy>> 8)&0xff])<<17)
	| ((utab[(iy>>16)&0xff])<<33)
	| ((utab[(iy>>24)&0xff])<<49) ); 
  
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  assert(pix >= 0 && pix < npix_);
  
  return pix;
}

void ring2xyf(long pix, long *ix, long *iy, long *face_num, long order_)
{
  long iring, iphi, kshift, nr;
  
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  assert(pix >= 0 && pix < npix_);
  
  long nside_ = 1;
  nside_ = nside_ << order_;
    
  long npface_ = 1;
  npface_ = npface_ << (2*order_);
  
  long ncap_ = (npface_-nside_)<<1;
  
  long nl2 = 2*nside_;
  
  if (pix<ncap_) // North Polar cap
    {
      iring = (long) (0.5*(1+isqrt(1+2*pix))); //counted from North pole
      iphi  = (pix+1) - 2*iring*(iring-1);
      kshift = 0;
      nr = iring;
      *face_num=0;
      long tmp = iphi-1;
      if (tmp>=(2*iring))
	{
	  *face_num=2;
	  tmp-=2*iring;
	}
      if (tmp>=iring) (*face_num) = (*face_num) + 1;
    }
  else if (pix<(npix_-ncap_)) // Equatorial region
    {
      long ip = pix - ncap_;
      if (order_>=0)
	{
	  iring = (ip>>(order_+2)) + nside_; // counted from North pole
	  iphi  = (ip&(4*nside_-1)) + 1;
	}
      else
	{
	  iring = (ip/(4*nside_)) + nside_; // counted from North pole
	  iphi  = (ip%(4*nside_)) + 1;
	}
      kshift = (iring+nside_)&1;
      nr = nside_;
      long ire = iring-nside_+1;
      long irm = nl2+2-ire;
      long ifm, ifp;
      if (order_>=0)
	{
	  ifm = (iphi - ire/2 + nside_ -1) >> order_;
	  ifp = (iphi - irm/2 + nside_ -1) >> order_;
	}
      else
	{
	  ifm = (iphi - ire/2 + nside_ -1) / nside_;
	  ifp = (iphi - irm/2 + nside_ -1) / nside_;
	}
      if (ifp == ifm) // faces 4 to 7
	*face_num = (ifp==4) ? 4 : ifp+4;
      else if (ifp<ifm) // (half-)faces 0 to 3
	*face_num = ifp;
      else // (half-)faces 8 to 11
	*face_num = ifm + 8;
    }
  else // South Polar cap
    {
      long ip = npix_ - pix;
      iring = (long) (0.5*(1+isqrt(2*ip-1))); //counted from South pole
      iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1));
      kshift = 0;
      nr = iring;
      iring = 2*nl2-iring;
      *face_num=8;
      long tmp = iphi-1;
      if (tmp>=(2*nr))
	{
	  *face_num=10;
	  tmp-=2*nr;
	}
      if (tmp>=nr) (*face_num) = (*face_num) + 1;
    }

  long irt = iring - (jrll[*face_num]*nside_) + 1;
  long ipt = 2*iphi- jpll[*face_num]*nr - kshift -1;
  if (ipt>=nl2) ipt-=8*nside_;

  *ix =  (ipt-irt) >>1;
  *iy =(-(ipt+irt))>>1;
}

long xyf2ring(long ix, long iy, long face_num, long order_)
{
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
    
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  long npface_ = 1;
  npface_ = npface_ << (2*order_);
  
  long ncap_ = (npface_-nside_)<<1;
  
  long nl4 = 4*nside_;
  long jr = (jrll[face_num]*nside_) - ix - iy  - 1;

  long nr, kshift, n_before;
  if (jr<nside_)
    {
      nr = jr;
      n_before = 2*nr*(nr-1);
      kshift = 0;
    }
  else if (jr > 3*nside_)
    {
      nr = nl4-jr;
      n_before = npix_ - 2*(nr+1)*nr;
      kshift = 0;
    }
  else
    {
      nr = nside_;
      n_before = ncap_ + (jr-nside_)*nl4;
      kshift = (jr-nside_)&1;
    }

  long jp = (jpll[face_num]*nr + ix - iy + 1 + kshift) / 2;
  if (jp>nl4)
    jp-=nl4;
  else
    if (jp<1) jp+=nl4;

  long pix = n_before + jp - 1;
  assert(pix >= 0 && pix < npix_);
  
  return pix;
}

long nest2ring(long pix, long order_)
{
  long ix, iy, face_num;
  nest2xyf(pix,&ix,&iy,&face_num,order_);
  return xyf2ring(ix,iy,face_num,order_);
}

long ring2nest(long pix, long order_)
{
  long ix, iy, face_num;
  ring2xyf(pix,&ix,&iy,&face_num,order_);
  return xyf2nest(ix,iy,face_num,order_);
}

long nest2peano(long pix, long order_)
{
  static const unsigned long subpix[8][4] = {
    { 0, 1, 3, 2 }, { 3, 0, 2, 1 }, { 2, 3, 1, 0 }, { 1, 2, 0, 3 },
    { 0, 3, 1, 2 }, { 1, 0, 2, 3 }, { 2, 1, 3, 0 }, { 3, 2, 0, 1 } };
  static const unsigned long subpath[8][4] = {
    { 4, 0, 6, 0 }, { 7, 5, 1, 1 }, { 2, 4, 2, 6 }, { 3, 3, 7, 5 },
    { 0, 2, 4, 4 }, { 5, 1, 5, 3 }, { 6, 6, 0, 2 }, { 1, 7, 3, 7 } };
  static const unsigned long face2path[12] = {
    2, 5, 2, 5, 3, 6, 3, 6, 2, 3, 2, 3 };
  static const unsigned long face2peanoface[12] = {
    0, 5, 6, 11, 10, 1, 4, 7, 2, 3, 8, 9 };
  
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  assert(pix >= 0 && pix < npix_);
  
  long face = pix>>(2*order_);
  unsigned long path = face2path[face];
  long result = 0;
  
  long shift;
  for(shift=2*order_-2; shift>=0; shift-=2)
    {
      unsigned char spix = (pix>>shift) & 0x3;
      result <<= 2;
      result |= subpix[path][spix];
      path=subpath[path][spix];
    }

  return result + ((face2peanoface[face])<<(2*order_));
}

long peano2nest(long pix, long order_)
{
  static const unsigned long subpix[8][4] = {
    { 0, 1, 3, 2 }, { 1, 3, 2, 0 }, { 3, 2, 0, 1 }, { 2, 0, 1, 3 },
    { 0, 2, 3, 1 }, { 1, 0, 2, 3 }, { 3, 1, 0, 2 }, { 2, 3, 1, 0 } };
  static const unsigned long subpath[8][4] = {
    { 4, 0, 0, 6 }, { 5, 1, 1, 7 }, { 6, 2, 2, 4 }, { 7, 3, 3, 5 },
    { 0, 4, 4, 2 }, { 1, 5, 5, 3 }, { 2, 6, 6, 0 }, { 3, 7, 7, 1 } };
  static const unsigned long face2path[12] = {
    2, 6, 2, 3, 3, 5, 2, 6, 2, 3, 3, 5 };
  static const unsigned long peanoface2face[12] = {
    0, 5, 8, 9, 6, 1, 2, 7, 10, 11, 4, 3 };
  
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  assert(pix >= 0 && pix < npix_);
  
  long face = pix>>(2*order_);
  unsigned long path = face2path[face];
  long result = 0;
  
  long shift;
  for(shift=2*order_-2; shift>=0; shift-=2)
    {
      unsigned long spix = (pix>>shift) & 0x3;
      result <<= 2;
      result |= subpix[path][spix];
      path=subpath[path][spix];
    }

  return result + ((peanoface2face[face])<<(2*order_));
}

long ang2ring(double theta, double phi, long order_)
{
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  long npface_ = 1;
  npface_ = npface_ << (2*order_);
  
  long ncap_ = (npface_-nside_)<<1;
  
  double z = cos(theta);
  double za = fabs(z);
  double tt = phi;
  long tt_long = (floor(tt/2/M_PI));
  tt = tt - ((double) (tt_long))*2*M_PI;
  tt *= M_2_PI; // in [0,4)

  if(za <= 2.0/3.0) // Equatorial region
    {
      double temp1 = nside_*(0.5+tt);
      double temp2 = nside_*z*0.75;
      long jp = (long) (temp1-temp2); // index of  ascending edge line
      long jm = (long) (temp1+temp2); // index of descending edge line
      
      // ring number counted from z=2/3
      long ir = nside_ + 1 + jp - jm; // in {1,2n+1}
      long kshift = 1-(ir&1); // kshift=1 if ir even, 0 otherwise
      
      long ip = (jp+jm-nside_+kshift+1)/2; // in {0,4n-1}
      ip = imodulo(ip,4*nside_);
      
      return ncap_ + (ir-1)*4*nside_ + ip;
    }
  else  // North & South polar caps
    {
      double tp = tt- ((long) (tt));
      double tmp = nside_*sqrt(3*(1-za));
      
      long jp = (long) (tp*tmp); // increasing edge line index
      long jm = (long) ((1.0-tp)*tmp); // decreasing edge line index
      
      long ir = jp+jm+1; // ring number counted from the closest pole
      long ip = (long) (tt*ir); // in {0,4*ir-1}
      ip = imodulo(ip,4*ir);
      
      if(z>0)
	return 2*ir*(ir-1) + ip;
      else
	return npix_ - 2*ir*(ir+1) + ip;
    }
}

long ang2nest(double theta, double phi, long inorder_)
{
  //done at highest resolution and then degraded to specified resolution
  long order_ = 29;
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  long innside_ = 1;
  innside_ = innside_ << inorder_;
  
  double z = cos(theta);
  double za = fabs(z);
  double tt = phi;
  long tt_long = (floor(tt/2/M_PI));
  tt = tt - ((double) (tt_long))*2*M_PI;
  tt *= M_2_PI; // in [0,4)
  
  long face_num, ix, iy;

  if(za<=2.0/3.0) // Equatorial region
    {
      double temp1 = nside_*(0.5+tt);
      double temp2 = nside_*(z*0.75);
      long jp = (long) (temp1-temp2); // index of  ascending edge line
      long jm = (long) (temp1+temp2); // index of descending edge line
      long ifp = jp >> order_;  // in {0,4}
      long ifm = jm >> order_;
      if(ifp == ifm)           // faces 4 to 7
	face_num = (ifp==4) ? 4: ifp+4;
      else if(ifp < ifm)       // (half-)faces 0 to 3
	face_num = ifp;
      else                      // (half-)faces 8 to 11
	face_num = ifm + 8;

      ix = jm & (nside_-1);
      iy = nside_ - (jp & (nside_-1)) - 1;
    }
  else // polar region, za > 2/3
    {
      long ntt = (long) (tt);
      if (ntt>=4) ntt=3;
      double tp = tt-ntt;
      double tmp = nside_*sqrt(3*(1-za));
      
      long jp = (long) (tp*tmp); // increasing edge line index
      long jm = (long) ((1.0-tp)*tmp); // decreasing edge line index
      if(jp>=nside_) jp = nside_-1; // for points too close to the boundary
      if(jm>=nside_) jm = nside_-1;
      if(z >= 0)
	{
	  face_num = ntt;  // in {0,3}
	  ix = nside_ - jm - 1;
	  iy = nside_ - jp - 1;
	}
      else
	{
	  face_num = ntt + 8; // in {8,11}
	  ix =  jp;
	  iy =  jm;
	}
    }
  
  long opix = xyf2nest(ix,iy,face_num,order_);
  //fprintf(stderr,"opix = %ld\n",opix);
  
  //degrade to inorder_ map resolution
  long ip = opix - nside_*nside_*face_num;
  //fprintf(stderr,"ip = %ld, nside_ = %ld, innside_ = %ld\n",ip,nside_,innside_);
  long difffac = 1;
  difffac = difffac << 2*(order_ - inorder_);
  ip = ip/difffac;
  //fprintf(stderr,"ip = %ld, nside_ = %ld, innside_ = %ld\n",ip,nside_,innside_);
    
  return (ip + face_num*innside_*innside_);
}

long vec2nest(double *vec, long order_)
{
  double theta,phi;
  vec2ang(vec,&theta,&phi);
  
  return ang2nest(theta,phi,order_);
}

long vec2ring(double *vec, long order_)
{
  double theta,phi;
  vec2ang(vec,&theta,&phi);
  
  return ang2ring(theta,phi,order_);
}

void ring2ang(long pix, double *theta, double *phi, long order_)
{
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  assert(pix >= 0 && pix < npix_);
  
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  long npface_ = 1;
  npface_ = npface_ << (2*order_);
  
  long ncap_ = (npface_-nside_)<<1;
  
  double z;
  
  double fact2_  = 4./npix_;
  double fact1_  = (nside_<<1)*fact2_;

  if(pix<ncap_) // North Polar cap
    {
      long iring = (long) (0.5*(1+isqrt(1+2*pix))); //counted from North pole
      long iphi  = (pix+1) - 2*iring*(iring-1);

      z = 1.0 - (iring*iring)*fact2_;
      *theta = acos(z);
      *phi = (iphi-0.5) * M_PI_2/iring;
    }
  else if(pix<(npix_-ncap_)) // Equatorial region
    {
      long ip  = pix - ncap_;
      long iring = ip/(4*nside_) + nside_; // counted from North pole
      long iphi  = ip%(4*nside_) + 1;
      // 1 if iring+nside is odd, 1/2 otherwise
      double fodd = ((iring+nside_)&1) ? 1 : 0.5;

      long nl2 = 2*nside_;
      z = (nl2-iring)*fact1_;
      *theta = acos(z);
      *phi = (iphi-fodd) * M_PI/nl2;
    }
  else // South Polar cap
    {
      long ip = npix_ - pix;
      long iring = (long) (0.5*(1+isqrt(2*ip-1))); //counted from South pole
      long iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1));
      
      z = -1.0 + (iring*iring)*fact2_;
      *theta = acos(z);
      *phi = (iphi-0.5) * M_PI_2/iring;
    }
}

void ring2vec(long pix, double *vec, long order_)
{
  double theta,phi;
  ring2ang(pix,&theta,&phi,order_);
  ang2vec(vec,theta,phi);
}

void nest2ang(long pix, double *theta, double *phi, long order_)
{
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  assert(pix >= 0 && pix < npix_);
  
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  double z;
  
  double fact2_  = 4./npix_;
  double fact1_  = (nside_<<1)*fact2_;
  
  long nl4 = nside_*4;

  long face_num, ix, iy;
  nest2xyf(pix,&ix,&iy,&face_num,order_);

  long jr = ((jrll[face_num])<<order_) - ix - iy - 1;

  long nr;
  long kshift;
  if (jr<nside_)
    {
      nr = jr;
      z = 1 - nr*nr*fact2_;
      kshift = 0;
    }
  else if (jr > 3*nside_)
    {
      nr = nl4-jr;
      z = nr*nr*fact2_ - 1;
      kshift = 0;
    }
  else
    {
      nr = nside_;
      z = (2*nside_-jr)*fact1_;
      kshift = (jr-nside_)&1;
    }
  
  long jp = (jpll[face_num]*nr + ix -iy + 1 + kshift) / 2;
  if (jp>nl4) jp-=nl4;
  if (jp<1) jp+=nl4;
  
  *phi = (jp-(kshift+1)*0.5)*(M_PI_2/nr);
  *theta = acos(z);
}

void nest2vec(long pix, double *vec, long order_)
{
  double theta,phi;
  nest2ang(pix,&theta,&phi,order_);
  ang2vec(vec,theta,phi);
}

void getneighbors_nest(long pix, long *result, long order_)
{
  static const long xoffset[] = { -1,-1, 0, 1, 1, 1, 0,-1 };
  static const long yoffset[] = {  0, 1, 1, 1, 0,-1,-1,-1 };
  static const long facearray[][12] =
    { {  8, 9,10,11,-1,-1,-1,-1,10,11, 8, 9 },   // S
      {  5, 6, 7, 4, 8, 9,10,11, 9,10,11, 8 },   // SE
      { -1,-1,-1,-1, 5, 6, 7, 4,-1,-1,-1,-1 },   // E
      {  4, 5, 6, 7,11, 8, 9,10,11, 8, 9,10 },   // SW
      {  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11 },   // center
      {  1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4 },   // NE
      { -1,-1,-1,-1, 7, 4, 5, 6,-1,-1,-1,-1 },   // W
      {  3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7 },   // NW
      {  2, 3, 0, 1,-1,-1,-1,-1, 0, 1, 2, 3 } }; // N
  static const long swaparray[][12] =
    { {  0,0,0,0,0,0,0,0,3,3,3,3 },   // S
      {  0,0,0,0,0,0,0,0,6,6,6,6 },   // SE
      {  0,0,0,0,0,0,0,0,0,0,0,0 },   // E
      {  0,0,0,0,0,0,0,0,5,5,5,5 },   // SW
      {  0,0,0,0,0,0,0,0,0,0,0,0 },   // center
      {  5,5,5,5,0,0,0,0,0,0,0,0 },   // NE
      {  0,0,0,0,0,0,0,0,0,0,0,0 },   // W
      {  6,6,6,6,0,0,0,0,0,0,0,0 },   // NW
      {  3,3,3,3,0,0,0,0,0,0,0,0 } }; // N

  long ix, iy, face_num;
  nest2xyf(pix,&ix,&iy,&face_num,order_);
  
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  long nsm1 = nside_-1;
  if((ix>0)&&(ix<nsm1)&&(iy>0)&&(iy<nsm1))
    {
      int m;
      for(m=0; m<8; ++m)
	result[m] = xyf2nest(ix+xoffset[m],iy+yoffset[m],face_num,order_);
    }
  else
    {
      long i;
      for(i=0; i<8; ++i)
	{
	  long x=ix+xoffset[i];
	  long y=iy+yoffset[i];
	  long nbnum=4;
	  if (x<0)
	    { x+=nside_; nbnum-=1; }
	  else if (x>=nside_)
	    { x-=nside_; nbnum+=1; }
	  if (y<0)
	    { y+=nside_; nbnum-=3; }
	  else if (y>=nside_)
	    { y-=nside_; nbnum+=3; }
	  
	  long f = facearray[nbnum][face_num];
	  if(f>=0)
	    {
	      if (swaparray[nbnum][face_num]&1) x=nside_-x-1;
	      if (swaparray[nbnum][face_num]&2) y=nside_-y-1;
	      if (swaparray[nbnum][face_num]&4)
		{
		  long temp = x;
		  x = y;
		  y = temp;
		  //swap(x,y);
		}
	      result[i] = xyf2nest(x,y,f,order_);
	    }
	  else
	    result[i] = -1;
	}
    }
}

void getneighbors_ring(long pix, long *result, long order_)
{
  static const long xoffset[] = { -1,-1, 0, 1, 1, 1, 0,-1 };
  static const long yoffset[] = {  0, 1, 1, 1, 0,-1,-1,-1 };
  static const long facearray[][12] =
    { {  8, 9,10,11,-1,-1,-1,-1,10,11, 8, 9 },   // S
      {  5, 6, 7, 4, 8, 9,10,11, 9,10,11, 8 },   // SE
      { -1,-1,-1,-1, 5, 6, 7, 4,-1,-1,-1,-1 },   // E
      {  4, 5, 6, 7,11, 8, 9,10,11, 8, 9,10 },   // SW
      {  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11 },   // center
      {  1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4 },   // NE
      { -1,-1,-1,-1, 7, 4, 5, 6,-1,-1,-1,-1 },   // W
      {  3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7 },   // NW
      {  2, 3, 0, 1,-1,-1,-1,-1, 0, 1, 2, 3 } }; // N
  static const long swaparray[][12] =
    { {  0,0,0,0,0,0,0,0,3,3,3,3 },   // S
      {  0,0,0,0,0,0,0,0,6,6,6,6 },   // SE
      {  0,0,0,0,0,0,0,0,0,0,0,0 },   // E
      {  0,0,0,0,0,0,0,0,5,5,5,5 },   // SW
      {  0,0,0,0,0,0,0,0,0,0,0,0 },   // center
      {  5,5,5,5,0,0,0,0,0,0,0,0 },   // NE
      {  0,0,0,0,0,0,0,0,0,0,0,0 },   // W
      {  6,6,6,6,0,0,0,0,0,0,0,0 },   // NW
      {  3,3,3,3,0,0,0,0,0,0,0,0 } }; // N

  long ix, iy, face_num;
  ring2xyf(pix,&ix,&iy,&face_num,order_);
  
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  long nsm1 = nside_-1;
  if((ix>0)&&(ix<nsm1)&&(iy>0)&&(iy<nsm1))
    {
      long m;
      for(m=0; m<8; ++m)
	result[m] = xyf2ring(ix+xoffset[m],iy+yoffset[m],face_num,order_);
    }
  else
    {
      long i;
      for(i=0; i<8; ++i)
	{
	  long x=ix+xoffset[i];
	  long y=iy+yoffset[i];
	  long nbnum=4;
	  if (x<0)
	    { x+=nside_; nbnum-=1; }
	  else if (x>=nside_)
	    { x-=nside_; nbnum+=1; }
	  if (y<0)
	    { y+=nside_; nbnum-=3; }
	  else if (y>=nside_)
	    { y-=nside_; nbnum+=3; }
	  
	  long f = facearray[nbnum][face_num];
	  if (f>=0)
	    {
	      if (swaparray[nbnum][face_num]&1) x=nside_-x-1;
	      if (swaparray[nbnum][face_num]&2) y=nside_-y-1;
	      if (swaparray[nbnum][face_num]&4)
		{
		  long temp = x;
		  x = y;
		  y = temp;
		  //swap(x,y);
		}
	      result[i] = xyf2ring(x,y,f,order_);
	    }
	  else
	    result[i] = -1;
	}
    }
}

void get_ring_info2(long ring, long *startpix, long *ringpix, double *costheta, double *sintheta, long *shifted, long order_)
{
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  long npface_ = 1;
  npface_ = npface_ << (2*order_);
  
  long ncap_ = (npface_-nside_)<<1;
  
  double fact2_  = 4./npix_;
  double fact1_  = (nside_<<1)*fact2_;

  long northring = (ring>2*nside_) ? 4*nside_-ring : ring;
  if(northring < nside_)
    {
      double tmp = northring*northring*fact2_;
      *costheta = 1 - tmp;
      *sintheta = sqrt(tmp*(2-tmp));
      //double theta = atan2(*sintheta,*costheta);
      *ringpix = 4*northring;
      *shifted = 1;
      *startpix = 2*northring*(northring-1);
    }
  else
    {
      *costheta = (2*nside_-northring)*fact1_;
      *sintheta = sqrt((1.0 - *costheta)*(1.0 + *costheta));
      //*theta = acos(*costheta);
      *ringpix = 4*nside_;
      if(((northring-nside_) & 1) == 0)
	*shifted = 1;
      else
	*shifted = 0;
      *startpix = ncap_ + (northring-nside_)*(*ringpix);
    }
  
  if(northring != ring) // southern hemisphere
    {
      //*theta = M_PI-(*theta);
      *costheta = -1.0*(*costheta);
      *startpix = npix_ - (*startpix) - (*ringpix);
    }
}

long ring_above(double z, long order_)
{
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  double az=fabs(z);
  if (az > 2.0/3.0) // polar caps
    {
      long iring = (long) (nside_*sqrt(3*(1-az)));
      return (z>0) ? iring : 4*nside_-iring-1;
    }
  else // ----- equatorial region ---------
    return (long) (nside_*(2-1.5*z));
}

/* returns ring indexed pixels and weights for interpolation */
void get_interpol(double theta, double phi, long pix[4], double wgt[4], long order_)
{
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  double z = cos(theta);
  long ir1 = ring_above(z,order_);
  long ir2 = ir1+1;
  double theta1=0.0, theta2=0.0, w1, tmp, dphi;
  double cth1,cth2,sth1,sth2;
  long sp,nr;
  long shift;
  long i1,i2;
  
  if (ir1>0)
    {
      get_ring_info2(ir1,&sp,&nr,&cth1,&sth1,&shift,order_);
      theta1 = atan2(sth1,cth1);
      //get_ring_info2 (ir1, sp, nr, theta1, shift);
      dphi = 2.0*M_PI/nr;
      tmp = (phi/dphi - .5*shift);
      i1 = (tmp<0) ? ((long) (tmp))-1 : (long) (tmp);
      w1 = (phi-(i1+.5*shift)*dphi)/dphi;
      i2 = i1+1;
      if (i1<0) i1 +=nr;
      if (i2>=nr) i2 -=nr;
      pix[0] = sp+i1; pix[1] = sp+i2;
      wgt[0] = 1-w1; wgt[1] = w1;
    }
  if (ir2<(4*nside_))
    {
      get_ring_info2(ir2,&sp,&nr,&cth2,&sth2,&shift,order_);
      theta2 = atan2(sth2,cth2);
      //get_ring_info2 (ir2, sp, nr, theta2, shift);
      dphi = 2.0*M_PI/nr;
      tmp = (phi/dphi - .5*shift);
      i1 = (tmp<0) ? ((long) (tmp))-1 : (long) (tmp);
      w1 = (phi-(i1+.5*shift)*dphi)/dphi;
      i2 = i1+1;
      if (i1<0) i1 +=nr;
      if (i2>=nr) i2 -=nr;
      pix[2] = sp+i1; pix[3] = sp+i2;
      wgt[2] = 1-w1; wgt[3] = w1;
    }

  if (ir1==0)
    {
      double wtheta = theta/theta2;
      wgt[2] *= wtheta; wgt[3] *= wtheta;
      double fac = (1-wtheta)*0.25;
      wgt[0] = fac; wgt[1] = fac; wgt[2] += fac; wgt[3] +=fac;
      pix[0] = (pix[2]+2)%4;
      pix[1] = (pix[3]+2)%4;
    }
  else if (ir2==4*nside_)
    {
      double wtheta = (theta-theta1)/(M_PI-theta1);
      wgt[0] *= (1-wtheta); wgt[1] *= (1-wtheta);
      double fac = wtheta*0.25;
      wgt[0] += fac; wgt[1] += fac; wgt[2] = fac; wgt[3] =fac;
      pix[2] = ((pix[0]+2)&3)+npix_-4;
      pix[3] = ((pix[1]+2)&3)+npix_-4;
    }
  else
    {
      double wtheta = (theta-theta1)/(theta2-theta1);
      wgt[0] *= (1-wtheta); wgt[1] *= (1-wtheta);
      wgt[2] *= wtheta; wgt[3] *= wtheta;
    }
}


long alm2index(long l, long m)
{
  return (long) (l*l + l + m);
}

void index2alm(long almindex, long *l, long *m)
{
  *l = isqrt(almindex);
  *m = almindex - (*l)*(*l) - (*l);
}

long num_alms(long l)
{
  return (long) (l*l + l + l + 1L);
}

//! Returns the largest integer which is smaller than (or equal to) \a arg.
long ifloor(double arg)
{
  return (arg>=0) ? ((long) (arg)) : ((long) (arg))-1;
}

void query_disc_inclusive_nest(double theta, double phi, double radius, long **listpix, long *Nlistpix, long order_)
{
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));

  long nside_ = 1;
  nside_ = nside_ << order_;
  
  double fact2_  = 4./npix_;
  double fact1_  = (nside_<<1)*fact2_;
  
  //add fudge factor to make sure all pixels needed are returned - will cause some extra pixels to be returned as well
  radius = radius + 1.362*M_PI/(4*nside_); 
  long loopstart;
  if((*listpix) == NULL)
    loopstart = 0;
  else
    loopstart = *Nlistpix;
  
  double dth1 = fact2_;
  double dth2 = fact1_;
  double cosang = cos(radius);

  double z0 = cos(theta);
  double xa = 1./sqrt((1-z0)*(1+z0));

  double rlat1  = theta - radius;
  double zmax = cos(rlat1);
  long irmin = ring_above(zmax,order_)+1;
  
  long m;
  if (rlat1<=0) // north pole in the disc
    for (m=1; m<irmin; ++m) // rings completely in the disc
      in_ring(m, 0.0, M_PI, listpix, Nlistpix, order_);

  double rlat2  = theta + radius;
  double zmin = cos(rlat2);
  long irmax = ring_above(zmin,order_);

  // ------------- loop on ring number ---------------------
  long iz;
  for (iz=irmin; iz<=irmax; ++iz) // rings partially in the disc
    {
      double z;
      if (iz<nside_) // north polar cap
	z = 1.0 - iz*iz*dth1;
      else if (iz <= (3*nside_)) // tropical band + equat.
	z = (2*nside_-iz) * dth2;
      else
	z = -1.0 + (4*nside_-iz)*(4*nside_-iz)*dth1;

      // --------- phi range in the disc for each z ---------
      double x = (cosang-z*z0)*xa;
      double ysq = 1-z*z-x*x;
      assert(ysq>=0);
      double dphi=atan2(sqrt(ysq),x);
      in_ring(iz, phi, dphi, listpix, Nlistpix, order_);
    }

  if (rlat2>=M_PI) // south pole in the disc
    for (m=irmax+1; m<(4*nside_); ++m)  // rings completely in the disc
      in_ring (m, 0.0, M_PI, listpix, Nlistpix, order_);

  //if (scheme_==NEST)
  if(loopstart != (*Nlistpix))
    {
      for (m=loopstart; m<(*Nlistpix); ++m)
	(*listpix)[m] = ring2nest((*listpix)[m],order_);
    }
}

void in_ring(long iz, double phi0, double dphi, long **listir, long *Nlistir, long order_)
{
  long nr, ir, ipix1;
  double shift=0.5;

  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));

  long nside_ = 1;
  nside_ = nside_ << order_;
  
  long npface_ = 1;
  npface_ = npface_ << (2*order_);
  
  long ncap_ = (npface_-nside_)<<1;
  
  if (iz<nside_) // north pole
    {
      ir = iz;
      nr = ir*4;
      ipix1 = 2*ir*(ir-1);        //    lowest pixel number in the ring
    }
  else if (iz>(3*nside_)) // south pole
    {
      ir = 4*nside_ - iz;
      nr = ir*4;
      ipix1 = npix_ - 2*ir*(ir+1); // lowest pixel number in the ring
    }
  else // equatorial region
    {
      ir = iz - nside_ + 1;           //    within {1, 2*nside + 1}
      nr = nside_*4;
      if ((ir&1)==0) shift = 0;
      ipix1 = ncap_ + (ir-1)*nr; // lowest pixel number in the ring
    }

  long ipix2 = ipix1 + nr - 1;       //    highest pixel number in the ring

  // ----------- constructs the pixel list --------------
  long i,loopstart;
  if (dphi > (M_PI-1e-7))
    {
      loopstart = *Nlistir;
      in_ring_realloc(listir,Nlistir,ipix2-ipix1+1);
      for (i=ipix1; i<=ipix2; ++i) //listir.push_back(i);
	(*listir)[i-ipix1+loopstart] = i;
    }
  else
    {
      long ip_lo = ifloor(nr*(phi0-dphi)/2.0/M_PI - shift)+1;
      long ip_hi = ifloor(nr*(phi0+dphi)/2.0/M_PI - shift);
      long pixnum = ip_lo+ipix1;
      if (pixnum<ipix1) pixnum += nr;
      loopstart = *Nlistir;
      in_ring_realloc(listir,Nlistir,ip_hi-ip_lo+1);
      for (i=ip_lo; i<=ip_hi; ++i, ++pixnum)
	{
	  if (pixnum>ipix2) 
	    pixnum -= nr;
	  (*listir)[i-ip_lo+loopstart] = pixnum;
	  //listir.push_back(pixnum);
	}
    }
}

void in_ring_realloc(long **listir, long *Nlistir, long Nextra)
{
  long *listir_new;
  
  if(Nextra > 0)
    {
      if((*listir) == NULL || (*Nlistir) == 0)
	{
	  *listir = (long*)malloc(sizeof(long)*Nextra);
	  assert((*listir) != NULL);
	  *Nlistir = Nextra;
	}
      else
	{
	  listir_new = (long*)realloc(*listir,sizeof(long)*(*Nlistir + Nextra));
	  assert(listir_new != NULL);
	  
	  *listir = listir_new;
	  *Nlistir = (*Nlistir) + Nextra;
	}
    }
}

long ring2ringnum(long ringind, long order_)
{
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  assert(ringind >=0 && ringind < npix_);

  long nside_ = 1;
  nside_ = nside_ << order_;
  long npface_ = 1;
  npface_ = npface_ << (2*order_);
  long ncap_ = (npface_-nside_)<<1;
  long nmiddle_ = 4*(2*npface_ + nside_) + ncap_; /* cumulative number of pixels from north pole to bottom of southern equatorial zone */
  long ring;
  
  if(ringind < ncap_)
    ring = (2+sqrt((double) (8*ringind+4)))/4;
  else if(ringind < nmiddle_)
    ring = (ringind-ncap_)/4/nside_ + nside_;
  else
    {
      ringind = ncap_ - 1 - (ringind - nmiddle_);
      ring = (2+sqrt((double) (8*ringind+4)))/4;
      ring = 4*nside_-ring;
    }

  return ring;
}

/* gets a set of triangles for each pixel on the sphere
   ring - ring index of the pixel
   tri[4][3] - array where the jth vertex of the ith triangle is in tri[i][j]
   order - the order of the healpix grid
   
   returns the number of triangles per pixel
   
   The number of triangles returned per pixel varies because of the geometry of healpix.
   Thus to use this function properly, do this                                                                                                                                                                              
   Ntri = ring2triangle(ring,tri,order);
   for(i=0;i<Ntri;++i)
     for(j=0;j<3;++j)
       trivertex_ij = tri[i][j];
       
   If you call this function for every ring index on the sphere at a given order,
   the tirangles will completely cover the sphere and every triangle will be retruned at most once.                                                                                                                                                                              
   Some pixles near the north pole return no triangles (Ntri = 0) and others return up to four (Ntri = 4).
*/
int ring2triangle(long ring, long tri[4][3], long order)
{
  long ringnum = ring2ringnum(ring,order);
  long ringnumA = ringnum - 1;
  long sp,Np,sh;
  long spA,NpA,shA;
  double ct,st;
  long Nside = 1;
  Nside = Nside << order;
  long Npix = 12l*Nside*Nside;
  long nringnum,nringnumA;
  long i,j;
  for(i=0;i<4;++i)
    for(j=0;j<3;++j)
      tri[i][j] = -1;
  long Ntri = 0;
  long ip,ib;
  long bnum;

  /* get ring info for rings above and below the pixel */
  get_ring_info2(ringnum,&sp,&Np,&ct,&st,&sh,order);
  get_ring_info2(ringnumA,&spA,&NpA,&ct,&st,&shA,order);

  if(ringnum == 1)
    {
      switch(ring)
        {
        case 0:
          tri[0][0] = 0;
          tri[0][1] = 2;
          tri[0][2] = 3;
          Ntri = 1;
          break;
        case 2:
          tri[0][0] = 2;
          tri[0][1] = 0;
          tri[0][2] = 1;
          Ntri = 1;
          break;
        }
    }
  else if(ringnum <= Nside)
    {
      tri[0][0] = ring;
      ip = ring - sp;
      ib = ip%ringnum;
      bnum = ip/ringnum;

      Ntri = 1;
      tri[0][1] = ip+1;
      tri[0][1] = ((tri[0][1])%Np) + sp;
      if(ib < ringnum-1)
        {
          tri[0][2] = ib+bnum*ringnumA;
          tri[0][2] = (tri[0][2])%NpA+spA;

          tri[1][0] = ring;
          tri[1][1] = tri[0][2];
          tri[1][2] = (tri[0][2]-spA-1);
          while(tri[1][2] < 0)
            tri[1][2] += NpA;
          tri[1][2] = (tri[1][2])%NpA + spA;
          Ntri = 2;
        }
      else
        {
          tri[0][2] = ib+bnum*ringnumA-1;
          tri[0][2] = (tri[0][2])%NpA+spA;
        }
    }
  else if(ringnum <= 3*Nside)
    {
      tri[0][0] = ring;
      ip = ring - sp;
      ib = ip%Nside;
      bnum = ip/Nside;

      tri[0][1] = ip+1;
      tri[0][1] = ((tri[0][1])%Np) + sp;
      if(ringnumA > Nside)
        tri[0][2] = sh+ib+bnum*Nside;
      else
        tri[0][2] = sh+ib+bnum*ringnumA;
      tri[0][2] = (tri[0][2])%NpA+spA;

      tri[1][0] = ring;
      tri[1][1] = tri[0][2];
      tri[1][2] = (tri[0][2]-spA-1);
      while(tri[1][2] < 0)
        tri[1][2] += NpA;
      tri[1][2] = (tri[1][2])%NpA + spA;

      Ntri = 2;
    }
  else if(ringnum <= 4*Nside-1)
    {
      tri[0][0] = ring;
      ip = ring - sp;
      nringnum = 4*Nside - ringnum;
      ib = ip%nringnum;
      bnum = ip/nringnum;

      nringnumA = 4*Nside - ringnumA;
      tri[0][1] = ip+1;
      tri[0][1] = ((tri[0][1])%Np) + sp;
      tri[0][2] = sh+ib+bnum*nringnumA;
      tri[0][2] = (tri[0][2])%NpA+spA;

      tri[1][0] = ring;
      tri[1][1] = tri[0][2];
      tri[1][2] = (tri[0][2]-spA-1);
      while(tri[1][2] < 0)
        tri[1][2] += NpA;
      tri[1][2] = (tri[1][2])%NpA + spA;

      Ntri = 2;

      if(ib == 0)
        {
          tri[2][0] = ring;
          tri[2][1] = tri[1][2];
          tri[2][2] = (tri[1][2]-spA-1);
          while(tri[2][2] < 0)
            tri[2][2] += NpA;
          tri[2][2] = (tri[2][2])%NpA + spA;

          Ntri = 3;
        }
    }
  
  if(ringnum == 4*Nside-1)
    {
      switch(Npix-1-ring)
        {
        case 0:
          tri[Ntri][0] = Npix-1 - 0;
          tri[Ntri][1] = Npix-1 - 2;
          tri[Ntri][2] = Npix-1 - 3;
          ++Ntri;
          break;
        case 2:
          tri[Ntri][0] = Npix-1 - 2;
          tri[Ntri][1] = Npix-1 - 1;
          tri[Ntri][2] = Npix-1 - 0;
          ++Ntri;
          break;
        }
    }

  return Ntri;
}

//implementation of method for finite diff. in Smith (2006)                                                                                                  
long get_healpix_finitediff_info(double theta, double phi, long pixnest, double mabinv[2][2], double thetadiff[8], double phidiff[8], long nest[8], long order)
{
  double costheta,sintheta,cosphi,sinphi;
  double x[3],xp[3];
  double thetahat[3];
  double phihat[3];
  double mab[2][2];
  double diff[2],detmab;
  long i,j,k;

  if(pixnest < 0)
    pixnest = ang2nest(theta,phi,order);
  else
    nest2ang(pixnest,&theta,&phi,order);
  
  costheta = cos(theta);
  sintheta = sin(theta);
  cosphi = cos(phi);
  sinphi = sin(phi);

  x[0] = cosphi*sintheta;
  x[1] = sinphi*sintheta;
  x[2] = costheta;

  thetahat[0] = cosphi*costheta;
  thetahat[1] = sinphi*costheta;
  thetahat[2] = -1.0*sintheta;

  phihat[0] = -1.0*sinphi;
  phihat[1] = cosphi;
  phihat[2] = 0.0;

  getneighbors_nest(pixnest,nest,order);

  for(i=0;i<2;++i)
    for(j=0;j<2;++j)
      mab[i][j] = 0.0;

  for(k=0;k<8;++k)
    if(nest[k] >= 0)
      {
	nest2vec(nest[k],xp,order);

        diff[0] = thetahat[0]*(xp[0]-x[0]) + thetahat[1]*(xp[1]-x[1]) + thetahat[2]*(xp[2]-x[2]);
        diff[1] = phihat[0]*(xp[0]-x[0]) + phihat[1]*(xp[1]-x[1]); //always zero! + phihat[2]*(xp[2]-x[2]);

        for(i=0;i<2;++i)
          for(j=0;j<2;++j)
            mab[i][j] += diff[i]*diff[j];

        thetadiff[k] = diff[0];
        phidiff[k] = diff[1];
      }

  //invert mab
  detmab = mab[0][0]*mab[1][1] - mab[0][1]*mab[1][0];
  mabinv[0][0] = mab[1][1]/detmab;
  mabinv[1][1] = mab[0][0]/detmab;
  mabinv[0][1] = -1.0/detmab*mab[0][1];
  mabinv[1][0] = -1.0/detmab*mab[1][0];
  
  return pixnest;
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
  
  //FIXME - code here to correct for round off errors
  if(fabs(barycoords[0]) < 1e-15)
    barycoords[0] = 0.0;
  if(fabs(barycoords[1]) < 1e-15)
    barycoords[1] = 0.0;
  if(fabs(barycoords[2]) < 1e-15)
    barycoords[2] = 0.0;
  
  if(barycoords[0] >= 0.0 && barycoords[1] >= 0.0 && barycoords[2] >= 0.0)
    return 1;
  else
    return 0;
  
  /* old ocde assumes exact arithemetic
  if(barycoords[0] > 0.0 && barycoords[1] > 0.0 && barycoords[2] > 0.0)
    return 1;
  else
    return 0;
  */
}

/* formula for signed triangle area NR in C 3rd Edition pg 1111
   returns positive area for verticies in counter-clockwise order, negative area otherwise
*/
static double trisarea(double a[2], double b[2], double c[2])
{
  return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]))/2.0;
}

void get_interp_triangle(double theta, double phi, long ring[3], double wgt[3], long order)
{
  long ringpix;
  double costheta,sintheta,cosphi,sinphi;
  double xp[3],thetahat[3],phihat[3];
  double tripos[3][2],a[2],b[2],c[2],q[2],area,bcs[3],d[3];
  long ringnbrs[9],tri[4][3],Ntri,found,i,j,k;
  long *tmp,*ringstack,Nringstack,NringstackTotal;
  long Nround = 2,ind,round,jp;
  
  //get projection vectors
  costheta = cos(theta);
  sintheta = sin(theta);
  cosphi = cos(phi);
  sinphi = sin(phi);

  thetahat[0] = cosphi*costheta;
  thetahat[1] = sinphi*costheta;
  thetahat[2] = -1.0*sintheta;

  phihat[0] = -1.0*sinphi;
  phihat[1] = cosphi;
  phihat[2] = 0.0;
  
  double x[3];
  x[0] = cosphi*sintheta;
  x[1] = sinphi*sintheta;
  x[2] = costheta;
  
  q[0] = 0.0;
  q[1] = 0.0;
  
  ringpix = ang2ring(theta,phi,order);
  NringstackTotal = 100;
  ringstack = (long*)malloc(sizeof(long)*NringstackTotal);
  assert(ringstack != NULL);
  ringstack[0] = ringpix;
  Nringstack = 1;
  
  found = 0;
  
  ind = 0;
  j = 1;
  for(round=0;round<Nround;++round)
    {
      jp = 0;
      for(k=0;k<j;++k)
	{
	  ringpix = ringstack[ind];
	  ++ind;
	  getneighbors_ring(ringpix,ringnbrs,order);
	  
	  if(Nringstack + 8 > NringstackTotal)
	    {
	      tmp = (long*)realloc(ringstack,sizeof(long)*(NringstackTotal+100));
	      assert(tmp != NULL);
	      NringstackTotal += 100;
	      ringstack = tmp;
	    }
	  
	  for(i=0;i<8;++i)
	    {
	      if(ringnbrs[i] >= 0)
		{
		  ringstack[Nringstack] = ringnbrs[i];
		  ++jp;
		  ++Nringstack;
		}
	    }
	}
      
      j = jp;
    }
  
  while(Nringstack > 0 && !found)
    {
      ringpix = ringstack[Nringstack-1];
      --Nringstack;
      
      Ntri = ring2triangle(ringpix,tri,order);
      
      for(i=0;i<Ntri;++i)
	{
	  for(j=0;j<3;++j)
	    {
	      ring2vec(tri[i][j],xp,order);
	      
	      d[j] = 1.0/(xp[0]*x[0] + xp[1]*x[1] + xp[2]*x[2]);
	      
	      tripos[j][0] = d[j]*((xp[0])*thetahat[0] + (xp[1])*thetahat[1] + (xp[2])*thetahat[2]);
	      tripos[j][1] = d[j]*((xp[0])*phihat[0] + (xp[1])*phihat[1]); //always zero! + (xp[2] - x[2])*phihat[2];
	    }
	  
	  //make the triangle
	  a[0] = tripos[0][0];
	  a[1] = tripos[0][1];
	  b[0] = tripos[1][0];
	  b[1] = tripos[1][1];
	  c[0] = tripos[2][0];
	  c[1] = tripos[2][1];
	  area =  trisarea(a,b,c);
	  if(area < 0.0) //swap last two points to get verts into counter-clockwise order
	    {
	      b[0] = tripos[2][0];
	      b[1] = tripos[2][1];
	      
	      c[0] = tripos[1][0];
	      c[1] = tripos[1][1];
	    }
	  
	  //do barycoord test
	  if(tritest_getbarycoords(a,b,c,q,bcs))
	    {
	      wgt[0] = bcs[0]/d[0];
	      ring[0] = tri[i][0];
	      
	      if(area < 0.0)
		{
		  ring[1] = tri[i][1];
		  ring[2] = tri[i][2];
		  
		  wgt[1] = bcs[1]/d[1];
		  wgt[2] = bcs[2]/d[2];
		}
	      else
		{
		  ring[1] = tri[i][2];
		  ring[2] = tri[i][1];
		  
		  wgt[1] = bcs[1]/d[2];
		  wgt[2] = bcs[2]/d[1];
		}
	      
	      found = 1;
	      break;
	    }
	}
    }
  
  free(ringstack);
  assert(found);
}

static void getnormalvec(double invec[2], double outvec[2])
{
  double norm;
  
  if(invec[0] != 0.0)
    {
      outvec[1] = 1.0;
      outvec[0] = -1.0*invec[1]/invec[0];
      norm = sqrt(outvec[0]*outvec[0] + outvec[1]*outvec[1]);
      outvec[0] /= norm;
      outvec[1] /= norm;
    }
  else
    {
      outvec[0] = 1.0;
      outvec[1] = 0.0;
    }
}

//does an interpolation using all neighbors of the pixel containing the point
//returns the number of points in wgt and ringpix - some pixels only have 7 neighbors to form the polygon
int get_interp_polygon(double theta, double phi, long ringpix[8], double wgt[8], long order)
{
  long ring,nbrs[8];
  double costheta,sintheta,cosphi,sinphi;
  double xp[3],thetahat[3],phihat[3];
  
  double verts[8][2],normals[8][2],fvec[2],nvec[2],dists[8],d[8],denom,numer,totwgt;
  long Nverts,i,j,ip1,im1;
  
  //get projection vectors
  costheta = cos(theta);
  sintheta = sin(theta);
  cosphi = cos(phi);
  sinphi = sin(phi);

  thetahat[0] = cosphi*costheta;
  thetahat[1] = sinphi*costheta;
  thetahat[2] = -1.0*sintheta;

  phihat[0] = -1.0*sinphi;
  phihat[1] = cosphi;
  phihat[2] = 0.0;
  
  //double x[3];
  //x[0] = cosphi*sintheta;
  //x[1] = sinphi*sintheta;
  //x[2] = costheta;
  
  ring = ang2ring(theta,phi,order);
  getneighbors_ring(ring,nbrs,order);
  
  Nverts = 0;
  for(j=0;j<8;++j)
    ringpix[j] = -1;
  for(j=0;j<8;++j)
    if(nbrs[j] >= 0 && j%2 == 0)
      {
	ring2vec(nbrs[j],xp,order);
	
	d[Nverts] = 1.0;///(xp[0]*x[0] + xp[1]*x[1] + xp[2]*x[2]);
	
	verts[Nverts][0] = d[Nverts]*(xp[0]*thetahat[0] + xp[1]*thetahat[1] + xp[2]*thetahat[2]);
	verts[Nverts][1] = d[Nverts]*(xp[0]*phihat[0] + xp[1]*phihat[1]); //always zero! + xp[2]*phihat[2];
	ringpix[Nverts] = nbrs[j];
	++Nverts;
      }
  
  //now get normals to facets - each normal is stored for facet with vert increased by one index
  for(i=0;i<Nverts;++i)
    {
      ip1 = i+1;
      if(ip1 >= Nverts)
	ip1 = 0;
      
      fvec[0] = verts[ip1][0] - verts[i][0];
      fvec[1] = verts[ip1][1] - verts[i][1];
      
      getnormalvec(fvec,nvec);
      
      assert(fabs(fvec[0]*nvec[0] + fvec[1]*nvec[1]) < 1e-10);
      
      normals[i][0] = nvec[0];
      normals[i][1] = nvec[1];
      
      dists[i] = fabs(verts[i][0]*nvec[0] + verts[i][1]*nvec[1]);
    }
  
  //now finally get barycoords
  for(i=0;i<Nverts;++i)
    {
      im1 = i-1;
      if(im1 < 0)
	im1 = Nverts-1;
      numer = fabs(normals[i][0]*normals[im1][1] - normals[i][1]*normals[im1][0]);
      denom = dists[i]*dists[im1];
      wgt[i] = numer/denom;
    }
  
  totwgt = 0.0;
  for(i=0;i<Nverts;++i)
    totwgt += wgt[i];
  for(i=0;i<Nverts;++i)
    wgt[i] /= totwgt;
  for(i=0;i<Nverts;++i)
    wgt[i] /= d[i];
  
  return Nverts;
}
