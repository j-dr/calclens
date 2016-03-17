#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_spline.h>

#include "raytrace.h"

void get_smoothing_lengths(void)
{
  long i;
  double obsSLVal[3];
  double r;
  double vec[3];
  
  //get part radii set up
  for(i=0;i<NlensPlaneParts;++i)
    {
      r = sqrt(lensPlaneParts[i].pos[0]*lensPlaneParts[i].pos[0] + 
               lensPlaneParts[i].pos[1]*lensPlaneParts[i].pos[1] + 
	       lensPlaneParts[i].pos[2]*lensPlaneParts[i].pos[2]);
      
      lensPlaneParts[i].pos[0] /= r;
      lensPlaneParts[i].pos[1] /= r;
      lensPlaneParts[i].pos[2] /= r;
      
      vec[0] = lensPlaneParts[i].pos[0];
      vec[1] = lensPlaneParts[i].pos[1];
      vec[2] = lensPlaneParts[i].pos[2];
      
      lensPlaneParts[i].r = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    }
  
  //enforce these mins and maxes
  if(NlensPlaneParts > 0)
    {
      obsSLVal[0] = 0.0;
      obsSLVal[1] = lensPlaneParts[0].smoothingLength;
      obsSLVal[2] = lensPlaneParts[0].smoothingLength;
      for(i=0;i<NlensPlaneParts;++i)
	{
	  obsSLVal[0] += log(lensPlaneParts[i].smoothingLength);
	  if(lensPlaneParts[i].smoothingLength < obsSLVal[1])
	    obsSLVal[1] = lensPlaneParts[i].smoothingLength;
	  if(lensPlaneParts[i].smoothingLength > obsSLVal[2])
	    obsSLVal[2] = lensPlaneParts[i].smoothingLength;
	  
	  if(lensPlaneParts[i].smoothingLength > rayTraceData.maxSL)
	    lensPlaneParts[i].smoothingLength = rayTraceData.maxSL;
	  if(lensPlaneParts[i].smoothingLength < rayTraceData.minSL)
	    lensPlaneParts[i].smoothingLength = rayTraceData.minSL;
	  
	  lensPlaneParts[i].cosSmoothingLength = cos(lensPlaneParts[i].smoothingLength);
	}
      
      if(ThisTask == 0)
	{
	  fprintf(stderr,"mean,min,max obs. smoothing len. = %lg|%lg|%lg [radians] (min,max actual smoothing len. %lg|%lg [radians])\n",
		  exp(obsSLVal[0]/NlensPlaneParts),obsSLVal[1],obsSLVal[2],
		  rayTraceData.minSL,rayTraceData.maxSL);
	  fprintf(stderr,"min,max comoving smoothing len. %lg|%lg [Mpc/h]\n",
		  rayTraceData.minSL*rayTraceData.planeRad,rayTraceData.maxSL*rayTraceData.planeRad);
	  fflush(stderr);
	}
      
#ifdef DEBUG_IO
      FILE *fp;
      char name[MAX_FILENAME];
      sprintf(name,"%s/smoothlengths%04ld.%04d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
      fp = fopen(name,"w");
      for(i=0;i<NlensPlaneParts;++i)
	fprintf(fp,"%.20e\n",lensPlaneParts[i].smoothingLength*rayTraceData.planeRad);
      fclose(fp);
#endif
    }
}

#define EPKERN
double spline_part_dens(double cosr, double sigma)
{
#ifdef EPKERN
#define NSVEC 10000
  static int init = 1;
  static double svec[NSVEC],nvec[NSVEC];
  static gsl_spline *spline;
  static gsl_interp_accel *accel;
  double dens,norm,rs;
  int i;
  
  if(init) 
    {
      init = 0;
      for(i=0;i<NSVEC;++i)
	{
	  svec[i] = i*M_PI/(NSVEC-1.0);
	  
	  sigma = svec[i];
	  norm = gsl_sf_sinc(sigma/M_PI/2.0);
	  norm = 4.0*M_PI*(0.5*norm*norm - gsl_sf_sinc(sigma/M_PI) + 0.5);
	  
	  nvec[i] = norm;
	}
      
      spline = gsl_spline_alloc(gsl_interp_cspline,(size_t) (NSVEC));
      gsl_spline_init(spline,svec,nvec,(size_t) (NSVEC));
      accel = gsl_interp_accel_alloc();
    }
#undef NSVEC
  
  //error check cosr
  if(cosr >= 1.0)
    {
      cosr = 1.0;
      rs = 0.0;
    }
  else if(cosr <= -1.0)
    {
      cosr = 1.0;
      rs = M_PI/sigma;
    }
  else
    rs = acos(cosr)/sigma;
  
  if(sigma > 0.0 && rs < 1.0)
    {
      //FIXME - using spline to compute the smoothing kernel normalization
      //norm = gsl_sf_sinc(sigma/M_PI/2.0);
      //norm = 4.0*M_PI*(0.5*norm*norm - gsl_sf_sinc(sigma/M_PI) + 0.5);
      norm = gsl_spline_eval(spline,sigma,accel);
      
      dens = (1.0 - rs*rs)/norm;
    }
  else
    dens = 0.0;
  
  return dens;
#else
  //factors needed for comp of f, gammaE, and pot 
  double b,d,f,h,k,p,r,v;
  double coss_2,sins_2,sins_2_sqr;
  double sins_2_4th,coss,sins;
  double sin2s,s2,s3,s4;
  double one_m_coss;
  double rs;
  double dens;
  double rs2,rs3,rs4;
  double sinr;
  
  //error check cosr
  if(cosr > 1.0)
    {
      cosr = 1.0;
      rs = 0.0;
    }
  else if(cosr < -1.0)
    {
      cosr = 1.0;
      rs = M_PI/sigma;
    }
  else
    rs = acos(cosr)/sigma;
  
  sinr = (1.0 - cosr)*(1.0 + cosr);
  if(sinr < 0.0)
    sinr = 0.0;
  else
    sinr = sqrt(sinr);
  
  if(sigma > 0.0 && rs < 1.0)
    {
      //factors needed for comp of f, gammaE, and pot 
      coss_2 = cos(sigma/2.0);
      sins_2 = sin(sigma/2.0);
      sins_2_sqr = sins_2*sins_2;
      sins_2_4th = sins_2_sqr*sins_2_sqr;
      coss = 1.0 - 2.0*sins_2*sins_2;
      sins = 2.0*coss_2*sins_2;
      sin2s = 2.0*coss*sins;
      s2 = sigma*sigma;
      s3 = s2*sigma;
      s4 = s3*sigma;
      one_m_coss = 1.0 - coss;
      
      v = -(sigma/sins_2_4th*(48*sigma + 14.0*s3 + 
                              sigma*(-48.0 + 7.0*s2)*coss + 
                              (48.0 + 23.0*s2)*sins - 24.0*sin2s) 
            )/9047.7868423386045267724129438449683064878478702003;  
      //9047.7868423386045267724129438449683064878478702003 = 2880.0*M_PI
      
      r = s4*(coss - 1.0 - sins*sins)/(301.59289474462015089241376479483227688292826234001)/one_m_coss/one_m_coss/one_m_coss - 5.0*v;
      //301.59289474462015089241376479483227688292826234001 = 96*M_PI
      p = s3*sins/(75.398223686155037723103441198708069220732065585003)/one_m_coss/one_m_coss - 4.0*r - 10.0*v;
      //75.398223686155037723103441198708069220732065585003 = 24*M_PI
      k = -s2/(25.132741228718345907701147066236023073577355195001)/one_m_coss - 3.0*p - 6.0*r - 10.0*v;
      //25.132741228718345907701147066236023073577355195001 = 8*M_PI
      h = sigma*sins/(12.5663706143591729538505735331180115367886775975)/one_m_coss - 2.0*k - 3.0*p - 4.0*r - 5.0*v;
      //12.5663706143591729538505735331180115367886775975 = 4*M_PI
      
      rs2 = rs*rs;
      rs3 = rs2*rs;
      rs4 = rs3*rs;
      
      if(rs < 0.5)
        {
          f = v - 3.2*h;  //16/5 = 3.2
          d = 8.0*h+r;
          //c = -8.0*h+p; this is always zero
          b = 4.0*h+k;
          
	  //dens = cosr/sinr*(2.0*b*rs + 4.0*d*rs3 + 5.0*f*rs4)/sigma
	  //  + (2.0*b + 12.0*d*rs2 + 20.0*f*rs3)/sigma/sigma + 0.079577471545947667884441881686257181017229822870228;
	  
	  dens = (2.0*b + 12.0*d*rs2 + 20.0*f*rs3)/sigma/sigma + 0.079577471545947667884441881686257181017229822870228;
	  if(sinr > 0.0)
	    dens += cosr/sinr*(2.0*b*rs + 4.0*d*rs3 + 5.0*f*rs4)/sigma;
	  
	  //0.079577471545947667884441881686257181017229822870228 = 1.0/4/M_PI
        }
      else 
        {
	  dens = cosr/sinr*(h + 2.0*k*rs + 3.0*p*rs2 + 4.0*r*rs3 + 5.0*v*rs4)/sigma
	    + (2.0*k + 6.0*p*rs + 12.0*r*rs2 + 20.0*v*rs3)/sigma/sigma + 0.079577471545947667884441881686257181017229822870228;
	  //0.079577471545947667884441881686257181017229822870228 = 1.0/4/M_PI
        }
    }
  else
    {
      dens = 0.0;
    }

  return dens;   
#endif
}

