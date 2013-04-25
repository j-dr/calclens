#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>

#include "raytrace.h"

static double NFW_boxmass(float r200, float conc, float overdens, float boxW, float boxL, gsl_rng *rng);
static void NFW_ptgen(float *x, float *y, float *z, int np, float r200, float conc, float overdens, float boxW, float boxL, gsl_rng *rng);

void make_lensplanes_pointmass_test(void)
{
#ifdef NFWHALOTEST
  fprintf(stderr,"making lensing planes for a NFW test...\n");
  
  //vars
  FILE *fp;
  int np,partid;
  float xx[6],rad,ra,dec,zred,*x,*y,*z;
  double vec[3],vech[3];
  char name[MAX_FILENAME];
  float r200,conc;
  float overdens;
  float boxmass,partmass,boxW,boxL;
  double radbl,decdbl;
  double galRad = rayTraceData.galRadPointNFWTest;
  
  gsl_rng *rng;  
  rng = gsl_rng_alloc(gsl_rng_ranlxd2);
  gsl_rng_set(rng,(unsigned long) (ThisTask+1));
  
  //set parms of halo
  r200 = 3.0;
  conc = 4.0;
  overdens = RHO_CRIT*rayTraceData.OmegaM*200.0;
  
  //get radius and bin
  rad = (float) (rayTraceData.radPointMass);
  ra = (float) (rayTraceData.raPointMass);
  dec = (float) (rayTraceData.decPointMass);
  ang2vec(vech,(double) (M_PI/2.0 - dec/180.0*M_PI),(double) (ra/180.0*M_PI));
  vech[0] = vech[0]*rad;
  vech[1] = vech[1]*rad;
  vech[2] = vech[2]*rad;
  xx[3] = 0.0;
  xx[4] = 0.0;
  xx[5] = 0.0;
  zred = -1.0;

  if(ThisTask == 0)
    fprintf(stderr,"%d: halo loc = %f|%f|%f\n",ThisTask,vech[0],vech[1],vech[2]);
  
  //get box mass and number of particles
  partmass = (float) (rayTraceData.partMass);
  boxL = (float) (rayTraceData.maxComvDistance*3.0);
  boxW = (float) (rayTraceData.maxComvDistance*3.0);
  boxmass = NFW_boxmass(r200,conc,overdens,boxW,boxL,rng);
  np = (int) (boxmass/partmass);
  
  if(ThisTask == 0)
    fprintf(stderr,"%d: r200,c200,ovderdens = %f|%f|%e, boxL,boxW = %f|%f, NFW box mass = %e, part mass = %e, # of parts = %d\n",
	    ThisTask,r200,conc,overdens,boxL,boxW,boxmass,partmass,np);
  
  //generate points
  x = (float*)malloc(sizeof(float)*np);
  assert(x != NULL);
  y = (float*)malloc(sizeof(float)*np);
  assert(y != NULL);
  z = (float*)malloc(sizeof(float)*np);
  assert(z != NULL);
  NFW_ptgen(x,y,z,np,r200,conc,overdens,boxW,boxL,rng);
  
  //write part info
  sprintf(name,"%s/NFWHaloLCParticles.dat",rayTraceData.LensPlanePath);
  fp = fopen(name,"wb");
  assert(fp != NULL);
  assert(fwrite(&np,sizeof(int),(size_t) 1,fp) == 1);
  for(partid=0;partid<np;++partid)
    {
      xx[0] = z[partid] + vech[0];
      xx[1] = y[partid] + vech[1];
      xx[2] = x[partid] + vech[2];
      vec[0] = xx[0];
      vec[1] = xx[1];
      vec[2] = xx[2];
      vec2radec(vec,&radbl,&decdbl);
      ra = radbl;
      dec = decdbl;
      
      assert(fwrite(&partid,sizeof(int),(size_t) 1,fp) == 1);
      assert(fwrite(xx,6*sizeof(float),(size_t) 1,fp) == 1);
      assert(fwrite(&ra,sizeof(float),(size_t) 1,fp)== 1);
      assert(fwrite(&dec,sizeof(float),(size_t) 1,fp) == 1);
      assert(fwrite(&zred,sizeof(float),(size_t) 1,fp) == 1);
    }
  fclose(fp);
  
  free(x);
  free(y);
  free(z);
  gsl_rng_free(rng);
  
  //write filelist
  sprintf(rayTraceData.LightConeFileList,"%s/filelist.txt",rayTraceData.LensPlanePath);
  sprintf(name,"%s/filelist.txt",rayTraceData.LensPlanePath);
  fp = fopen(name,"w");
  assert(fp != NULL);
  fprintf(fp,"%s/NFWHaloLCParticles.dat\n",rayTraceData.LensPlanePath);
  fclose(fp);
  
  sprintf(name,"%s/nfwhalotest_locdata.txt",rayTraceData.OutputPath);
  fp = fopen(name,"w");
  assert(fp != NULL);
  fprintf(fp,"# ra dec r200 c200 overdens bundleorder rayorder omegam maxcmvd numplanes \n");
  fprintf(fp,"%.20e %.20e %.20e %.20e %.20le %ld %ld %.20e %.20e %ld\n",rayTraceData.raPointMass,rayTraceData.decPointMass,
	  r200,conc,overdens,rayTraceData.bundleOrder,rayTraceData.rayOrder,rayTraceData.OmegaM,rayTraceData.maxComvDistance,rayTraceData.NumLensPlanes);
  fclose(fp);
  
  //write extra info for testing code analysis
  sprintf(name,"%s/nfwhalotest_data.txt",rayTraceData.OutputPath);
  fp = fopen(name,"w");
  assert(fp != NULL);
  fprintf(fp,"# aexpn_lens angd_lens aexp_source angd_source angd_source_minus_lens\n");
  fprintf(fp,"%.20e %.20e %.20e %.20e %.20e\n",acomvdist(rayTraceData.radPointMass),acomvdist(rayTraceData.radPointMass)*rayTraceData.radPointMass,
	  acomvdist(rayTraceData.maxComvDistance),acomvdist(rayTraceData.maxComvDistance)*rayTraceData.maxComvDistance,
	  angdistdiff(acomvdist(rayTraceData.maxComvDistance),acomvdist(rayTraceData.radPointMass)));
  fclose(fp);
    
  sprintf(name,"%s/nfwhalotest_data_npminus1.txt",rayTraceData.OutputPath);
  fp = fopen(name,"w");
  assert(fp != NULL);
  fprintf(fp,"# aexpn_lens angd_lens aexp_source angd_source angd_source_minus_lens\n");
  fprintf(fp,"%.20e %.20e %.20e %.20e %.20e\n",acomvdist(rayTraceData.radPointMass),acomvdist(rayTraceData.radPointMass)*rayTraceData.radPointMass,
	  acomvdist(rayTraceData.maxComvDistance*(1.0 - 1.0/rayTraceData.NumLensPlanes/2.0)),
	  acomvdist(rayTraceData.maxComvDistance*(1.0 - 1.0/rayTraceData.NumLensPlanes/2.0))*rayTraceData.maxComvDistance*(1.0 - 1.0/rayTraceData.NumLensPlanes/2.0),
	  angdistdiff(acomvdist(rayTraceData.maxComvDistance*(1.0 - 1.0/rayTraceData.NumLensPlanes/2.0)),acomvdist(rayTraceData.radPointMass)));
  fclose(fp);

  if(strlen(rayTraceData.GalOutputName) > 0)
    {
      sprintf(name,"%s/nfwhalotest_data_gridsearch.txt",rayTraceData.OutputPath);
      fp = fopen(name,"w");
      assert(fp != NULL);
      fprintf(fp,"# aexpn_lens angd_lens aexp_source angd_source angd_source_minus_lens\n");
      fprintf(fp,"%.20e %.20e %.20e %.20e %.20e\n",acomvdist(rayTraceData.radPointMass),acomvdist(rayTraceData.radPointMass)*rayTraceData.radPointMass,
	      acomvdist(galRad),acomvdist(galRad)*galRad,angdistdiff(acomvdist(galRad),acomvdist(rayTraceData.radPointMass)));
      fclose(fp);
      
      if(ThisTask == 0)
	fprintf(stderr,"\n\nassuming galaxies at %lf radius for NFW test!\n\n\n",galRad);
    }
#else /* NFWHALOTEST code is above, pointmasstest below */
  fprintf(stderr,"making lensing planes for a point mass test...\n");
  
  //vars
  FILE *fp;
  int np,partid;
  float xx[6],rad,ra,dec,z;
  double vec[3];
  char name[MAX_FILENAME];
  double galRad = rayTraceData.galRadPointNFWTest;
  
  //get radius and bin
  np = 1;
  rad = (float) (rayTraceData.radPointMass);
  ra = (float) (rayTraceData.raPointMass);
  dec = (float) (rayTraceData.decPointMass);
  ang2vec(vec,(double) (M_PI/2.0 - dec/180.0*M_PI),(double) (ra/180.0*M_PI));
  z = (float) (1.0/acomvdist(rad) - 1.0);
  xx[0] = (float) (vec[0]*rad);
  xx[1] = (float) (vec[1]*rad);
  xx[2] = (float) (vec[2]*rad);
  xx[3] = 0.0;
  xx[4] = 0.0;
  xx[5] = 0.0;
  partid = 1;
  
  //write part info
  sprintf(name,"%s/pointMassLCParticle.dat",rayTraceData.LensPlanePath);
  fp = fopen(name,"wb");
  assert(fp != NULL);
  assert(fwrite(&np,sizeof(int),(size_t) 1,fp) == 1);
  assert(fwrite(&partid,sizeof(int),(size_t) 1,fp) == 1);
  assert(fwrite(xx,6*sizeof(float),(size_t) 1,fp) == 1);
  assert(fwrite(&ra,sizeof(float),(size_t) 1,fp)== 1);
  assert(fwrite(&dec,sizeof(float),(size_t) 1,fp) == 1);
  assert(fwrite(&z,sizeof(float),(size_t) 1,fp) == 1);
  fclose(fp);
  
  //write filelist
  sprintf(rayTraceData.LightConeFileList,"%s/filelist.txt",rayTraceData.LensPlanePath);
  sprintf(name,"%s/filelist.txt",rayTraceData.LensPlanePath);
  fp = fopen(name,"w");
  assert(fp != NULL);
  fprintf(fp,"%s/pointMassLCParticle.dat\n",rayTraceData.LensPlanePath);
  fclose(fp);
  
  //write extra info for testing code analysis
  sprintf(name,"%s/pointmasstest_data.txt",rayTraceData.OutputPath);
  fp = fopen(name,"w");
  assert(fp != NULL);
  fprintf(fp,"# aexpn_lens angd_lens aexp_source angd_source angd_source_minus_lens\n");
  fprintf(fp,"%.20e %.20e %.20e %.20e %.20e\n",acomvdist(rayTraceData.radPointMass),acomvdist(rayTraceData.radPointMass)*rayTraceData.radPointMass,
	  acomvdist(rayTraceData.maxComvDistance),acomvdist(rayTraceData.maxComvDistance)*rayTraceData.maxComvDistance,
	  angdistdiff(acomvdist(rayTraceData.maxComvDistance),acomvdist(rayTraceData.radPointMass)));
  fclose(fp);

  sprintf(name,"%s/pointmasstest_locdata.txt",rayTraceData.OutputPath);
  fp = fopen(name,"w");
  assert(fp != NULL);
  fprintf(fp,"# ra dec mass bundleorder rayorder omegam maxcmvd numplanes \n");
  fprintf(fp,"%.20e %.20e %.20e %ld %ld %.20e %.20e %ld\n",rayTraceData.raPointMass,rayTraceData.decPointMass,rayTraceData.partMass,
	  rayTraceData.bundleOrder,rayTraceData.rayOrder,rayTraceData.OmegaM,rayTraceData.maxComvDistance,rayTraceData.NumLensPlanes);
  fclose(fp);
  
  sprintf(name,"%s/pointmasstest_data_npminus1.txt",rayTraceData.OutputPath);
  fp = fopen(name,"w");
  assert(fp != NULL);
  fprintf(fp,"# aexpn_lens angd_lens aexp_source angd_source angd_source_minus_lens\n");
  fprintf(fp,"%.20e %.20e %.20e %.20e %.20e\n",acomvdist(rayTraceData.radPointMass),acomvdist(rayTraceData.radPointMass)*rayTraceData.radPointMass,
	  acomvdist(rayTraceData.maxComvDistance*(1.0 - 1.0/rayTraceData.NumLensPlanes/2.0)),
	  acomvdist(rayTraceData.maxComvDistance*(1.0 - 1.0/rayTraceData.NumLensPlanes/2.0))*rayTraceData.maxComvDistance*(1.0 - 1.0/rayTraceData.NumLensPlanes/2.0),
	  angdistdiff(acomvdist(rayTraceData.maxComvDistance*(1.0 - 1.0/rayTraceData.NumLensPlanes/2.0)),acomvdist(rayTraceData.radPointMass)));
  fclose(fp);

  if(strlen(rayTraceData.GalOutputName) > 0)
    {
      sprintf(name,"%s/pointmasstest_data_gridsearch.txt",rayTraceData.OutputPath);
      fp = fopen(name,"w");
      assert(fp != NULL);
      fprintf(fp,"# aexpn_lens angd_lens aexp_source angd_source angd_source_minus_lens\n");
      fprintf(fp,"%.20e %.20e %.20e %.20e %.20e\n",acomvdist(rayTraceData.radPointMass),acomvdist(rayTraceData.radPointMass)*rayTraceData.radPointMass,
	      acomvdist(galRad),acomvdist(galRad)*galRad,angdistdiff(acomvdist(galRad),acomvdist(rayTraceData.radPointMass)));
      fclose(fp);
      
      if(ThisTask == 0)
	fprintf(stderr,"\n\nassuming galaxies at %lf radius for NFW test!\n\n\n",galRad);
    }
#endif
  
  //make them into HDF5 format
  makeRayTracingPlanesHDF5();
}

/*3D NFW profile 
  -- for an overdensity of overdens in units of whatever, radii in untis of whatever, and concentration c
  -- returns in same units as overdens
*/
static double threedNFWprof(double r, double rover, double c, double overdens)
{
  double x=r/rover*c,D=overdens*c*c*c/3.0/(log(1.0+c)-c/(1.0+c));
  return D/x/(1.0+x)/(1.0+x);
}

/*
generates random draws from an NFW given r200 conc and overdens
points are in coords with respect to center of halo in box that is boxW wide on two sides and boxL long on one side
set idnum to a negative value to seed random number generator
*/
static void NFW_ptgen(float *x, float *y, float *z, int np, float r200, float conc, float overdens, float boxW, float boxL, gsl_rng *rng)
{
  int i;
  float r,prad,maxNFW,maxR;
  float costheta,sintheta,phi1,signs;
  
  maxR = sqrt(2.0*pow(boxW/2.0,2.0) + pow(boxL/2.0,2.0));
  maxNFW = threedNFWprof(r200/conc,r200,conc,overdens)*r200/conc*r200/conc;

  i = 0;
  while(i<np)
    {

      r = maxR*gsl_rng_uniform(rng);
      prad = threedNFWprof(r,r200,conc,overdens)*r*r/maxNFW;

      if(gsl_rng_uniform(rng) < prad)
        {
          
          costheta=2.*(gsl_rng_uniform(rng)-.5);
          sintheta=sqrt(1.-costheta*costheta);
          signs=2.*(gsl_rng_uniform(rng)-.5);
          costheta=signs*costheta/fabs(signs);
          phi1=2.0*M_PI*gsl_rng_uniform(rng);
          
          x[i] = r*sintheta*cos(phi1);
          y[i] = r*sintheta*sin(phi1);
          z[i] = r*costheta;

          if(fabs(x[i]) <= boxW/2.0 && fabs(y[i]) <= boxW/2.0 && fabs(z[i]) <= boxL/2.0)
            ++i;
        }
    }
}

static double func_NFW_boxmass(double x[], size_t dim, void *p)
{
  double r = sqrt(pow(x[0],2.0) + pow(x[1],2.0) + pow(x[2],2.0));
  if(r < 1e-5)
    r = 1e-5;
  
  double *ps;
  ps = (double*)p;
  
  return threedNFWprof(r,ps[0],ps[1],ps[2]);
}

//gives total mass of particles for an NFW halo centered in a box of boxW width on two sides and boxL length on the other
static double NFW_boxmass(float r200, float conc, float overdens, float boxW, float boxL, gsl_rng *rng)
{
  gsl_monte_function f;
  double ps[3];
  ps[0] = r200;
  ps[1] = conc;
  ps[2] = overdens;
  
  fprintf(stderr,"getting total box mass: r200,c200,overdens = %f|%f|%e, boxW,boxL = %f|%f\n",r200,conc,overdens,boxW,boxL);
  
  size_t dim = 3,calls;
  double xl[3],xu[3],tgral,err;
  gsl_monte_vegas_state *vst;
  //gsl_monte_vegas_params vparms;
  
  f.f = &func_NFW_boxmass;
  f.dim = dim;
  f.params = &ps;
  
  vst = gsl_monte_vegas_alloc(dim);
  gsl_monte_vegas_init(vst);
  
  xl[0] = 0.0;
  xl[1] = 0.0;
  xl[2] = 0.0;
  
  xu[0] = boxW/2.0;
  xu[1] = boxW/2.0;
  xu[2] = boxL/2.0;
  
  vst->mode = GSL_VEGAS_MODE_IMPORTANCE_ONLY;
  
  vst->stage = 0;
  vst->iterations = 10;
  calls = 100000;
  gsl_monte_vegas_integrate(&f,xl,xu,dim,calls,rng,vst,&tgral,&err);
  
  tgral = tgral*8.0;
  err = err*8.0;
  fprintf(stderr,"box mass = %le, err = %le, chi2 = %lf\n",tgral,err,vst->chisq);
  tgral = tgral/8.0;
  err = err/8.0;
  
  vst->stage = 1;
  vst->iterations = 5;
  calls = 1000000;
  gsl_monte_vegas_integrate(&f,xl,xu,dim,calls,rng,vst,&tgral,&err);
  
  //need a factor of 8 to account for only doing one octant of the box
  tgral = tgral*8.0;
  err = err*8.0;
  
  fprintf(stderr,"box mass = %le, err = %le, chi2 = %lf\n",tgral,err,vst->chisq);
  
  gsl_monte_vegas_free(vst);
  
  return tgral;
}
