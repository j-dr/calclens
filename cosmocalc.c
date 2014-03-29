#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include "raytrace.h"

#define COSMOCALC_COMVDIST_TABLE_LENGTH 20000
#define AEXPN_MIN 0.01
#define AEXPN_MAX 1.0

int init_cosmocalc_flag = 1;
double comvdist_table[COSMOCALC_COMVDIST_TABLE_LENGTH];
double aexpn_table[COSMOCALC_COMVDIST_TABLE_LENGTH];

/* function for integration using gsl integration */
double comvdist_integ_funct(double a, void *p)
{
  return 1.0/sqrt(a*(*((double*)p)) + a*a*a*a*(1.0-(*((double*)p))));
}

/* init function  - some help from Gadget-2 applied here */
void init_cosmocalc(void)
{
#define WORKSPACE_NUM 100000
#define ABSERR 0.0
#define RELERR 1e-8
  gsl_integration_workspace *workspace;
  gsl_function F;
  long i;
  double result,abserr,afact;
  
  workspace = gsl_integration_workspace_alloc((size_t) WORKSPACE_NUM);
  
  for(i=0;i<COSMOCALC_COMVDIST_TABLE_LENGTH-1;++i)
    {
      afact = (AEXPN_MAX - AEXPN_MIN)/(COSMOCALC_COMVDIST_TABLE_LENGTH-1.0)*((double) i) + AEXPN_MIN;
      F.function = &comvdist_integ_funct;
      F.params = &(rayTraceData.OmegaM);
      gsl_integration_qag(&F,afact,1.0,ABSERR,RELERR,(size_t) WORKSPACE_NUM,GSL_INTEG_GAUSS51,workspace,&result,&abserr);
      aexpn_table[i] = afact;
      comvdist_table[i] = result*2997.92458; //always set h = 1 /rayTraceData.h;
    }
  aexpn_table[i] = 1.0;
  comvdist_table[i] = 0.0;
  
  gsl_integration_workspace_free(workspace);
#undef ABSERR
#undef RELERR
#undef WORKSPACE_NUM  
  
  init_cosmocalc_flag = 0;
}

double acomvdist(double dist)
{
  long i;
  double w,a;
  
  if(init_cosmocalc_flag == 1)
    init_cosmocalc();
  
  if(dist < comvdist_table[COSMOCALC_COMVDIST_TABLE_LENGTH-1])
    {
      a = AEXPN_MAX;
      fprintf(stderr,"\n\n%d: WARNING: comving distance given to acomvdist is out of range! min,max,val = %lf|%lf|%lf\n\n\n",
	      ThisTask,comvdist_table[COSMOCALC_COMVDIST_TABLE_LENGTH-1],comvdist_table[0],dist);
    }
  else if(dist > comvdist_table[0])
    {
      a = AEXPN_MIN;
      fprintf(stderr,"\n\n%d: WARNING: comving distance given to acomvdist is out of range! min,max,val = %lf|%lf|%lf\n\n\n",
	      ThisTask,comvdist_table[COSMOCALC_COMVDIST_TABLE_LENGTH-1],comvdist_table[0],dist);
    }
  else
    {
      for(i=COSMOCALC_COMVDIST_TABLE_LENGTH-1;i>=1;--i)
	{
	  if(comvdist_table[i] > dist)
	    break;
	}
      w = (dist - comvdist_table[i-1])/(comvdist_table[i] - comvdist_table[i-1]);
      a = (1.0-w)*aexpn_table[i-1] + w*aexpn_table[i];
    }
  
  return a;
}

double comvdist(double a)
{
  long i;
  double w,cd;
    
  if(init_cosmocalc_flag == 1)
    init_cosmocalc();
  
  i = (long) ((a - AEXPN_MIN)/(AEXPN_MAX - AEXPN_MIN)*(COSMOCALC_COMVDIST_TABLE_LENGTH-1));
  if(i >= COSMOCALC_COMVDIST_TABLE_LENGTH-1)
    {
      cd = comvdist_table[COSMOCALC_COMVDIST_TABLE_LENGTH-1];
      if(a > AEXPN_MAX)
	fprintf(stderr,"\n\n%d: WARNING: expansion factor given to comvdist is out of range! min,max,val = %lf|%lf|%lf\n\n\n",
		ThisTask,AEXPN_MIN,AEXPN_MAX,a);
    }
  else if(i < 0)
    {
      cd = comvdist_table[0];
      if(a < AEXPN_MIN)
	fprintf(stderr,"\n\n%d: WARNING: expansion factor given to comvdist is out of range! min,max,val = %lf|%lf|%lf\n\n\n",
		ThisTask,AEXPN_MIN,AEXPN_MAX,a);
    }
  else
    {
      w = (a - aexpn_table[i])/(AEXPN_MAX - AEXPN_MIN)*(COSMOCALC_COMVDIST_TABLE_LENGTH-1);
      cd = (1.0 - w)*comvdist_table[i] + w*comvdist_table[i+1];
    }
  
  return cd;
}

double angdist(double a)
{
  if(init_cosmocalc_flag == 1)
    init_cosmocalc();
  
  return comvdist(a)*a;
}

double angdistdiff(double amin, double amax)
{
  if(init_cosmocalc_flag == 1)
    init_cosmocalc();
  
  return (comvdist(amin)-comvdist(amax))*amin;
}
