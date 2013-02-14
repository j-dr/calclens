#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>
#include <gsl/gsl_heapsort.h>

#include "raytrace.h"

static long query_disc_inclusive_nest_realloc(double theta, double phi, double radius, long **listpix, long *NlistpixMax, long order_);
static void in_ring_realloc_fun(long iz, double phi0, double dphi, long **listir, long *NlistirMax, long *Nlistir, long order_);
static void in_ring_realloc_realloc(long **listir, long *Nlistir, long Nextra);
static long query_disc_inclusive_nest_tree(double theta, double phi, double radius, long **listpix, long *NlistpixMax, long queryOrder);

/* code to quickly query disc in healpix
   uses rings of pixels for small angles (<0.5 radians) or a tree for large angles(> 0.5)
   if the rings of pixels function fails, it just defualts to the tree version
   
   you feed it a pointer to either previousl allocated memory and NlistpixMax = size of memory, or a null pointer if there is no memory
   on return NlistpixMax holds the amount of memory allocated to listpix
   the function itself returns the number of pixels found in the disc
   
   Matthew R. Becker, UofC 2012
*/
long query_disc_inclusive_nest_fast(double theta, double phi, double radius, long **listpix, long *NlistpixMax, long queryOrder)
{
  long Nlistpix;
  
  if(radius < 0.5)
    Nlistpix = query_disc_inclusive_nest_realloc(theta,phi,radius,listpix,NlistpixMax,queryOrder);
  else
    Nlistpix = query_disc_inclusive_nest_tree(theta,phi,radius,listpix,NlistpixMax,queryOrder);
  
  if(Nlistpix == -1)
    Nlistpix = query_disc_inclusive_nest_tree(theta,phi,radius,listpix,NlistpixMax,queryOrder);
  
  return Nlistpix;
}

//basic query function from healpix - uses rings of pixels to find pixels in the disc - this version has slighty different memory allocation
static long query_disc_inclusive_nest_realloc(double theta, double phi, double radius, long **listpix, long *NlistpixMax, long order_)
{
  long Nlistpix = 0;
  
  long npix_ = 1;
  npix_ = 12*(npix_ << (2*order_));
  
  long nside_ = 1;
  nside_ = nside_ << order_;
  
  double fact2_  = 4./npix_;
  double fact1_  = (nside_<<1)*fact2_;
  
  //add fudge factor to make sure all pixels needed are returned - will cause some extra pixels to be returned as well
  radius = radius + 1.362*M_PI/(4*nside_); 
  long loopstart = (2.0*M_PI*(1.0 - cos(radius)))/(4.0*M_PI/npix_);
  if(*NlistpixMax < loopstart)
    in_ring_realloc_realloc(listpix,NlistpixMax,loopstart-*NlistpixMax);
      
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
      in_ring_realloc_fun(m, 0.0, M_PI, listpix, NlistpixMax, &Nlistpix, order_);
  
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
      
      //FIXME - this is a hack that uses the tree version if this function fails
      if(ysq < 0)
        {
	  //fprintf(stderr,"theta,phi = %.20le|%.20le, radius = %.20le, order = %ld, z,x,z0,xa,cosang = %lg|%lg|%lg|%lg|%lg\n",theta,phi,radius,order_,z,x,z0,xa,cosang);
	  return -1;
	}
      
      assert(ysq>=0);
      double dphi=atan2(sqrt(ysq),x);
      in_ring_realloc_fun(iz, phi, dphi, listpix, NlistpixMax, &Nlistpix, order_);
    }
  
  if (rlat2>=M_PI) // south pole in the disc
    for (m=irmax+1; m<(4*nside_); ++m)  // rings completely in the disc
      in_ring_realloc_fun(m, 0.0, M_PI, listpix, NlistpixMax, &Nlistpix, order_);
  
  //if (scheme_==NEST)
  for (m=0; m<(Nlistpix); ++m)
    (*listpix)[m] = ring2nest((*listpix)[m],order_);
  
  return Nlistpix;
}

//helper function for query_disc_inclusive_nest_realloc - looks in each ring and adds pixels 
static void in_ring_realloc_fun(long iz, double phi0, double dphi, long **listir, long *NlistirMax, long *Nlistir, long order_)
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
      
      if(*Nlistir + ipix2-ipix1+1 >= *NlistirMax)
	in_ring_realloc_realloc(listir,NlistirMax,ipix2-ipix1+1);
      
      for (i=ipix1; i<=ipix2; ++i) //listir.push_back(i);
        (*listir)[i-ipix1+loopstart] = i;
      
      (*Nlistir) += ipix2-ipix1+1;
    }
  else
    {
      long ip_lo = ifloor(nr*(phi0-dphi)/2.0/M_PI - shift)+1;
      long ip_hi = ifloor(nr*(phi0+dphi)/2.0/M_PI - shift);
      long pixnum = ip_lo+ipix1;
      
      if (pixnum<ipix1) pixnum += nr;
      loopstart = *Nlistir;
      
      if(*Nlistir + ip_hi-ip_lo+1 >= *NlistirMax)
	in_ring_realloc_realloc(listir,NlistirMax,ip_hi-ip_lo+1);
      
      for (i=ip_lo; i<=ip_hi; ++i, ++pixnum)
        {
          if (pixnum>ipix2) 
            pixnum -= nr;
          (*listir)[i-ip_lo+loopstart] = pixnum;
          //listir.push_back(pixnum);
        }
      
      (*Nlistir) += ip_hi-ip_lo+1;
    }
}

static void in_ring_realloc_realloc(long **listir, long *Nlistir, long Nextra)
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

//tree walk based query of pixels in a disc
static long query_disc_inclusive_nest_tree(double theta, double phi, double radius, long **listpix, long *NlistpixMax, long queryOrder)
{
  double vec[3],nvec[3];
  double cosd,ps;
  long i,shift,np,nest;
  long Nlistpix,*tmpLong;
  long NumStack,NumStackAlloc;
  double cosrList[HEALPIX_UTILS_MAXORDER+1];
  double cosnsList[HEALPIX_UTILS_MAXORDER+1];
  
  struct stk {
    long order;
    long nest;
  } *stack,currPix,*tmpStack;
  
  ang2vec(vec,theta,phi);
  
  i = (2.0*M_PI*(1.0 - cos(radius+1.362*M_PI/(4*order2nside(queryOrder)))))/(4.0*M_PI/order2npix(queryOrder));
  if(*NlistpixMax < i)
    {
      *NlistpixMax = i;
      tmpLong = (long*)realloc(*listpix,sizeof(long)*(*NlistpixMax));
      
      if(tmpLong != NULL)
	*listpix = tmpLong;
      else
	{
	  fprintf(stderr,"out of mem in query_disc_inclusive_nest_tree realloc! (requested %ld longs)\n",i);
	  assert(tmpLong != NULL);
	}
    }
  
  NumStack = 12;
  NumStackAlloc = 100;
  stack = (struct stk*)malloc(sizeof(struct stk)*NumStackAlloc);
  assert(stack != NULL);
  for(i=0;i<NumStack;++i)
    {
      stack[i].order = 0;
      stack[i].nest = i;
    }
  
  for(i=0;i<=queryOrder;++i)
    {
      ps = sqrt(4.0*M_PI/order2npix(i));
      
      cosd = radius + 1.362*M_PI/(4*order2nside(i));
      if(cosd > M_PI)
	cosrList[i] = -2.0;
      else
	cosrList[i] = cos(cosd);
      
      cosd = radius - ps;
      if(cosd > 0.0)
	cosnsList[i] = cos(cosd);
      else
	cosnsList[i] = 2.0;  //test for radius containing cell will always fail when cell is too big
    }
  
  Nlistpix = 0;
  while(NumStack > 0)
    {
      currPix = stack[NumStack-1];
      --NumStack;
      
      nest2vec(currPix.nest,nvec,currPix.order);
      cosd = vec[0]*nvec[0] + vec[1]*nvec[1] + vec[2]*nvec[2];
      
      if(cosd >= cosnsList[currPix.order]) //pixel is completely contained in the circle so just add cells at queryOrder
	{
	  shift = 2*(queryOrder - currPix.order);
	  np = (1LL) << shift;
	  nest = currPix.nest << shift;
	    
	  if(Nlistpix + np >= *NlistpixMax)
	    {
	      *NlistpixMax = *NlistpixMax + np*4;
	      tmpLong = (long*)realloc(*listpix,sizeof(long)*(*NlistpixMax));
	            
	      if(tmpLong != NULL)
		*listpix = tmpLong;
	      else
		{
		  fprintf(stderr,"out of mem in query_disc_inclusive_nest_tree realloc! (requested %ld longs)\n",(*NlistpixMax));
		  assert(tmpLong != NULL);
		}
	    }
	    
	  for(i=0;i<np;++i)
	    (*listpix)[Nlistpix+i] = nest + i;
	  Nlistpix += np;
	}
      else if(cosd >= cosrList[currPix.order])
	{
	  nest = currPix.nest << 2;  
	    
	  if(currPix.order + 1 < queryOrder)
	    {
	      //add to stack 
	      if(NumStack + 4 >= NumStackAlloc)
		{
		  NumStackAlloc += 100;
		  tmpStack = (struct stk*)realloc(stack,sizeof(struct stk)*NumStackAlloc);
		    
		  if(tmpStack != NULL)
		    stack = tmpStack;
		  else
		    {
		      fprintf(stderr,"out of mem in query_disc_inclusive_nest_tree stack realloc! (requested %ld stack structs)\n",NumStackAlloc);
		      assert(tmpStack != NULL);
		    }
		}
	            
	      for(i=0;i<4;++i)
		{
		  stack[NumStack+i].order = currPix.order + 1;
		  stack[NumStack+i].nest = nest + i;
		}
	      NumStack += 4;
	    }
	  else
	    {
	      //add to list
	      if(Nlistpix + 4 >= *NlistpixMax)
		{
		  *NlistpixMax = *NlistpixMax + 16;
		  tmpLong = (long*)realloc(*listpix,sizeof(long)*(*NlistpixMax));
		    
		  if(tmpLong != NULL)
		    *listpix = tmpLong;
		  else
		    {
		      fprintf(stderr,"out of mem in query_disc_inclusive_nest_tree realloc! (requested %ld longs)\n",(*NlistpixMax));
		      assert(tmpLong != NULL);
		    }
		}
	            
	      for(i=0;i<4;++i)
		(*listpix)[Nlistpix+i] = nest + i;
	      Nlistpix += 4;
	    }
	}
    }
  
  free(stack);
  
  return Nlistpix;
}
