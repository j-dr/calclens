/*
  functions to generate legendre polynomials quickly and index them
  
  This C++ object was made into a set of C programs. I also added utils to index plms.
  
  -Matthew R Becker, University of Chicago, 2010
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

/*
 *  Code for efficient calculation of Y_lm(theta,phi=0)
 *
 *  Copyright (C) 2005, 2006 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <gsl/gsl_math.h>
#include "healpix_utils.h"
#include "healpix_shtrans.h"

long plm2index(long l, long m)
{
  return (1+l)*(l)/2 + m;
}

void index2plm(long plmindex, long *l, long *m)
{
  *l = isqrt(plmindex*2);

  if((1+(*l))*(*l)/2 > plmindex)
    *l = *l - 1;
  *m = plmindex - (1+(*l))*(*l)/2;
}

long num_plms(long lmax)
{
  return (2+lmax)*(1+lmax)/2;
}

#define LARGE_EXPONENT2 90
#define MINSCALE -4

void plmgen(double cth, double sth, long m, double *vec, long *firstl, plmgen_data *plmdata)
{
  /*! For a colatitude given by \a cth and \a sth (representing cos(theta)
    and sin(theta)) and a multipole moment \a m, calculate the
    Y_lm(theta,phi=0) for \a m<=l<=lmax and return in it \a result[l].
    On exit, \a firstl is the \a l index of the first Y_lm with an
    absolute magnitude larger than \a epsilon. If \a firstl>lmax, all
    absolute values are smaller than \a epsilon.
    \a result[l] is undefined for all \a l<firstl. */
  
  //some internal constants
  const long large_exponent2 = LARGE_EXPONENT2;
  const long minscale = MINSCALE;
  double inv_ln2 = 1.0/log(2.0);
  double ln2 = log(2.0);
  
  assert(m <= plmdata->mmax);

  if(((m >= plmdata->m_crit) && (fabs(cth) >= plmdata->cth_crit)) || ((m > 0) && (sth == 0)))
    { 
      *firstl = plmdata->lmax+1; 
      return; 
    }

  plmgen_recalc_recfac(m,plmdata);
  
  double logval = plmdata->mfac[m];
  if(m > 0) 
    logval += m*inv_ln2*log(sth);
  
  long scale = (long) ((logval/large_exponent2)-minscale);
  double corfac = (scale < 0) ? 0.0 : plmdata->cf[scale];
  double lam_1 = 0;
  double lam_2 = exp(ln2*(logval-(scale+minscale)*large_exponent2));
  if(m & 1) 
    lam_2 = -lam_2;
  double lam_0;
  
  long l=m;
  while(1)
    {
      if(fabs(lam_2*corfac) > plmdata->eps) 
	break;
      if(++l > plmdata->lmax) 
	break;
      lam_0 = cth*lam_2*plmdata->recfac[0 + 2*(l-1)] - lam_1*plmdata->recfac[1 + 2*(l-1)];
      if(fabs(lam_0*corfac) > plmdata->eps) 
	{ 
	  lam_1 = lam_2; 
	  lam_2 = lam_0; 
	  break; 
	}
      if(++l > plmdata->lmax) 
	break;
      lam_1 = cth*lam_0*plmdata->recfac[0 + 2*(l-1)] - lam_2*plmdata->recfac[1 + 2*(l-1)];
      if(fabs(lam_1*corfac) > plmdata->eps) 
	{ 
	  lam_2 = lam_1; 
	  lam_1 = lam_0; 
	  break; 
	}
      if(++l > plmdata->lmax) 
	break;
      lam_2 = cth*lam_1*plmdata->recfac[0 + 2*(l-1)] - lam_0*plmdata->recfac[1 + 2*(l-1)];
      
      while(fabs(lam_2) > plmdata->fbig)
	{
          lam_1 *= plmdata->fsmall;
          lam_2 *= plmdata->fsmall;
          ++scale;
          corfac = (scale < 0) ? 0. : plmdata->cf[scale];
	}
    }
  
  *firstl = l;
  if(l > plmdata->lmax)
    { 
      plmdata->m_crit = m; 
      plmdata->cth_crit = fabs(cth); 
      return; 
    }

  lam_1*=corfac;
  lam_2*=corfac;
  
  for(;l<plmdata->lmax-2;l+=3)
    {
      vec[l] = lam_2;
      lam_0 = cth*lam_2*plmdata->recfac[0 + 2*l] - lam_1*plmdata->recfac[1 + 2*l];
      vec[l+1] = lam_0;
      lam_1 = cth*lam_0*plmdata->recfac[0 + 2*(l+1)] - lam_2*plmdata->recfac[1 + 2*(l+1)];
      vec[l+2] = lam_1;
      lam_2 = cth*lam_1*plmdata->recfac[0 + 2*(l+2)] - lam_0*plmdata->recfac[1 + 2*(l+2)];
    }
  
  while(1)
    {
      vec[l] = lam_2;
      if(++l > plmdata->lmax) 
	break;
      lam_0 = cth*lam_2*plmdata->recfac[0 + 2*(l-1)] - lam_1*plmdata->recfac[1 + 2*(l-1)];
      vec[l] = lam_0;
      if(++l > plmdata->lmax) 
	break;
      lam_1 = cth*lam_0*plmdata->recfac[0 + 2*(l-1)] - lam_2*plmdata->recfac[1 + 2*(l-1)];
      vec[l] = lam_1;
      if(++l > plmdata->lmax) 
	break;
      lam_2 = cth*lam_1*plmdata->recfac[0 + 2*(l-1)] - lam_0*plmdata->recfac[1 + 2*(l-1)];
    }
}

plmgen_data *plmgen_init(long lmax, double eps)
{
  //some internal constants
  const long large_exponent2 = LARGE_EXPONENT2;
  const long minscale = MINSCALE;
  
  double inv_sqrt4pi = 1.0/sqrt(4.0*M_PI);
  double inv_ln2 = 1.0/log(2.0);
  long m;
  plmgen_data *plmdata;
  
  m = (11 - minscale);
  m += (lmax + 1)*2;
  m += (lmax + 1);
  m += (lmax + 1);
  m += (2*lmax + 1);
    
  plmdata = (plmgen_data*)malloc(sizeof(plmgen_data) + sizeof(double)*m);
  assert(plmdata != NULL);
  plmdata->cf = (double*)(plmdata + 1);
  m = (11 - minscale);
  plmdata->recfac = plmdata->cf + m;
  m += (lmax + 1)*2;
  plmdata->mfac = plmdata->cf + m;
  m += (lmax + 1);
  plmdata->t1fac = plmdata->cf + m;
  m += (lmax + 1);
  plmdata->t2fac = plmdata->cf + m;
  
  //init values and arrays
  plmdata->fsmall = ldexp(1.0,(int) (-large_exponent2));
  plmdata->fbig = ldexp(1.0,(int) large_exponent2);
  plmdata->eps = eps;
  plmdata->cth_crit = 2.0;
  plmdata->lmax = lmax;
  plmdata->mmax = lmax;
  plmdata->m_last = -1;
  plmdata->m_crit = lmax + 1;
  
  for(m=0;m<(11 - minscale);++m)
    plmdata->cf[m] = ldexp(1.0,(int) ((m+minscale)*large_exponent2));
  
  plmdata->mfac[0] = 1;
  for(m=1;m<lmax+1;++m)
    plmdata->mfac[m] = plmdata->mfac[m-1]*sqrt((2*m+1.0)/(2*m));
  for(m=0;m<lmax+1;++m)
    plmdata->mfac[m] = inv_ln2*log(inv_sqrt4pi*plmdata->mfac[m]);
  for(m=0;m<lmax+1;++m)
    plmdata->t1fac[m] = sqrt(4.0*(m+1)*(m+1)-1.0);
  for(m=0;m<2*lmax+1;++m)
    plmdata->t2fac[m] = 1./sqrt(m+1.0);
  
  return plmdata;
}

void plmgen_destroy(plmgen_data *plmdata)
{
  free(plmdata);
}

void plmgen_recalc_recfac(long m, plmgen_data *plmdata)
{
  if(plmdata->m_last == m)
    return;
  
  double f_old=1.0;
  long l;
  for(l=m;l<plmdata->lmax+1;++l)
    {
      plmdata->recfac[0 + 2*l] = plmdata->t1fac[l]*plmdata->t2fac[l+m]*plmdata->t2fac[l-m];
      plmdata->recfac[1 + 2*l] = plmdata->recfac[0 + 2*l]/f_old;
      f_old = plmdata->recfac[0 + 2*l];
    }
  
  plmdata->m_last = m;
}

#undef LARGE_EXPONENT2
#undef MINSCALE
