#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <float.h>
#include <assert.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include "raytrace.h"
#include "treecode.h"

static void epkern_alpha_shear(double r, double cosr, double sinr, double sigma, double *fr, double *two_gammaE)
{
#define NSVEC 10000
  static int init = 1;
  static double svec[NSVEC],nvec1[NSVEC],nvec2[NSVEC];
  static gsl_spline *spline_s1;
  static gsl_interp_accel *accel_s1;
  static gsl_spline *spline_s2;
  static gsl_interp_accel *accel_s2;
  long i;
  double isigma;
  
  if(init) 
    {
      init = 0;
      for(i=0;i<NSVEC;++i)
	{
	  svec[i] = i*M_PI/(NSVEC-1.0);
          
	  isigma = svec[i];
	  nvec1[i] = gsl_sf_sinc(isigma/M_PI);
	  nvec2[i] = gsl_sf_sinc(isigma/M_PI/2.0);
	  nvec2[i] = nvec2[i]*nvec2[i];
	}
      
      spline_s1 = gsl_spline_alloc(gsl_interp_cspline,(size_t) (NSVEC));
      gsl_spline_init(spline_s1,svec,nvec1,(size_t) (NSVEC));
      accel_s1 = gsl_interp_accel_alloc();
      
      spline_s2 = gsl_spline_alloc(gsl_interp_cspline,(size_t) (NSVEC));
      gsl_spline_init(spline_s2,svec,nvec2,(size_t) (NSVEC));
      accel_s2 = gsl_interp_accel_alloc();
    }
#undef NSVEC
  
  double s1,s2,s1r,s2r;
  
  s1 = gsl_spline_eval(spline_s1,sigma,accel_s1);
  s2 = gsl_spline_eval(spline_s2,sigma,accel_s2);
  s1r = gsl_spline_eval(spline_s1,r,accel_s1);
  s2r = gsl_spline_eval(spline_s2,r,accel_s2);
  
  /*
  s1 = gsl_sf_sinc(sigma/M_PI);
  s2 = gsl_sf_sinc(sigma/M_PI/2.0);
  s2 = s2*s2;
  s1r = gsl_sf_sinc(r/M_PI);
  s2r = gsl_sf_sinc(r/M_PI/2.0);
  s2r = s2r*s2r;
  */
  
  double ns,rs,ht;
  ns = 4.0*M_PI*(0.5*s2 - s1 + 0.5);    
  rs = r/sigma;
  ht = 1.0/ns/(1.0 - cosr)*(rs*rs*(-2.0*s1r + cosr + s2r) + 1.0 - cosr) - 1.0/4.0/M_PI;
  
  *fr = -1.0*ht*(1.0 - cosr)/sinr;
  *two_gammaE = -1.0*(2.0*cosr/(1.0 + cosr)*ht - (1.0 - rs*rs)/ns + 1.0/4.0/M_PI);
}

static const double hpix_cell_areas[] = {
  1.04719755119659763132e+00,
  2.61799387799149407829e-01,
  6.54498469497873519574e-02,
  1.63624617374468379893e-02,
  4.09061543436170949734e-03,
  1.02265385859042737433e-03,
  2.55663464647606843583e-04,
  6.39158661619017108959e-05,
  1.59789665404754277240e-05,
  3.99474163511885693099e-06,
  9.98685408779714232748e-07,
  2.49671352194928558187e-07,
  6.24178380487321395467e-08,
  1.56044595121830348867e-08,
  3.90111487804575872167e-09,
  9.75278719511439680418e-10,
  2.43819679877859920104e-10,
  6.09549199694649800261e-11,
  1.52387299923662450065e-11,
  3.80968249809156125163e-12,
  9.52420624522890312908e-13,
  2.38105156130722578227e-13,
  5.95262890326806445568e-14,
  1.48815722581701611392e-14,
  3.72039306454254028480e-15,
  9.30098266135635071199e-16,
  2.32524566533908767800e-16,
  5.81311416334771919500e-17,
  1.45327854083692979875e-17,
  3.63319635209232449687e-18};

static void expandTreeNodes(TreeData td)
{
  TreeNode *tempnodes;
  long nextra = td->Nparts*0.2;
  
  if(nextra < 1000)
    nextra = 1000;
  
  rayTraceData.treeAllocFactor += 0.1;
  
  //if(ThisTask == 0)
  //fprintf(stderr,"increased treeAllocFactor by 0.1 to %lg.\n",rayTraceData.treeAllocFactor);
  
  tempnodes = (TreeNode*)realloc(td->nodes,sizeof(TreeNode)*(td->NnodesAlloc+nextra));
  if(tempnodes != NULL)
    {
      td->nodes = tempnodes;
      td->NnodesAlloc += nextra;
    }
  else
    {
      fprintf(stderr,"could not expand tree nodes: old size = %ld, new size = %ld\n",td->NnodesAlloc,td->NnodesAlloc+nextra);
      fflush(stderr);
      assert(tempnodes != NULL);
    }
}

static void refineTreeNode(TreeData td, long nodeInd)
{
  //if too few parts, return
  if(td->nodes[nodeInd].np <= MIN_NUM_PARTS_PER_TREENODE)
    return;
  
  long childOrder = td->nodes[nodeInd].order + 1;
  long shift = 2*(HEALPIX_UTILS_MAXORDER - childOrder);
  long partNest,currNest;
  long np_child[4] = {0,0,0,0};
  long maxInd,ind,i,j;
  
  //split up parts
  ind = td->nodes[nodeInd].pstart;
  maxInd = td->nodes[nodeInd].pstart + td->nodes[nodeInd].np;
  for(i=0;i<4;++i)
    {
      currNest = (td->nodes[nodeInd].nest << 2) + i;
      while(ind < maxInd)
	{
	  partNest = td->parts[ind].nest >> shift;
	  
	  if(partNest != currNest)
	    break;
	  
	  ++(np_child[i]);
	  ++ind;
	}
    }
  
  //now init new nodes
  for(i=0;i<4;++i)
    {
      if(np_child[i] == 0)
	continue;
      
      if(td->Nnodes + 1 > td->NnodesAlloc)
	expandTreeNodes(td);
      
      ind = td->Nnodes;
      td->nodes[nodeInd].child[i] = ind;
      ++(td->Nnodes);
      
      td->nodes[ind].order = childOrder;
      td->nodes[ind].nest = (td->nodes[nodeInd].nest << 2) + i;
      td->nodes[ind].child[0] = -1;
      td->nodes[ind].child[1] = -1;
      td->nodes[ind].child[2] = -1;
      td->nodes[ind].child[3] = -1;
      td->nodes[ind].pstart = td->nodes[nodeInd].pstart;
      for(j=0;j<i;++j)
	td->nodes[ind].pstart += np_child[j];
      td->nodes[ind].np = np_child[i];
      
      td->nodes[ind].mass = 0.0;
      td->nodes[ind].vec[0] = 0.0;
      td->nodes[ind].vec[1] = 0.0;
      td->nodes[ind].vec[2] = 0.0;
      for(j=td->nodes[ind].pstart;j<td->nodes[ind].np+td->nodes[ind].pstart;++j)
	{
	  if(td->nodes[ind].cosMaxSL < td->parts[j].smoothingLength)
	    td->nodes[ind].cosMaxSL = td->parts[j].smoothingLength;
	  td->nodes[ind].mass += td->parts[j].mass;
	  td->nodes[ind].vec[0] += td->parts[j].mass*td->parts[j].pos[0];
	  td->nodes[ind].vec[1] += td->parts[j].mass*td->parts[j].pos[1];
	  td->nodes[ind].vec[2] += td->parts[j].mass*td->parts[j].pos[2];
	}
      
      refineTreeNode(td,ind);
    }
}

TreeData buildTree(Part *parts, long Nparts, double thetaSplit)
{
  double timeB = 0.0;
  TreeData td;
  long baseOrder = 0;
  long NpixBaseOrder = order2npix(baseOrder);
  long i,j;
  long partNest,shift,currNest,ind;
  double s,r;
  
  timeB -= MPI_Wtime();
  
  //init tree data
  td = (TreeData)malloc(sizeof(_TreeData));
  assert(td != NULL);
  
  for(i=0;i<=HEALPIX_UTILS_MAXORDER;++i)
    {
      s = thetaSplit*MAX_RADTREEWALK_TO_SPLIT_RATIO + 2.0*sqrt(hpix_cell_areas[i]);
      if(s < M_PI)
	td->nodeCosRCut[i] = cos(s);
      else
	td->nodeCosRCut[i] = -1.0;
      
      //td->cosNodeArcSizesGeomLim[i] = cos(td->nodeArcSizes[i]*2.0);
    }
  td->thetaS = thetaSplit;
  td->thetaS2 = thetaSplit*thetaSplit;
  td->parts = parts;
  td->Nparts = Nparts;
  
  //init node array
  td->NnodesAlloc = Nparts*rayTraceData.treeAllocFactor;
  if(td->NnodesAlloc < NpixBaseOrder)
    td->NnodesAlloc = 4*NpixBaseOrder;
  td->nodes = (TreeNode*)malloc(sizeof(TreeNode)*td->NnodesAlloc);
  assert(td->nodes != NULL);
  td->Nnodes = NpixBaseOrder;
  
  for(i=0;i<NpixBaseOrder;++i)
    {
      td->nodes[i].order = baseOrder;
      td->nodes[i].nest = i;
      td->nodes[i].child[0] = -1;
      td->nodes[i].child[1] = -1;
      td->nodes[i].child[2] = -1;
      td->nodes[i].child[3] = -1;
      td->nodes[i].pstart = -1;
      td->nodes[i].np = 0;
      td->nodes[i].mass = 0.0;
      td->nodes[i].vec[0] = 0.0;
      td->nodes[i].vec[1] = 0.0;
      td->nodes[i].vec[2] = 0.0;
      td->nodes[i].cosMaxSL = 0.0;
    }
    
  //add inital set of parts to tree
  ind = 0;
  shift = 2*(HEALPIX_UTILS_MAXORDER - 0);
  for(i=0;i<NpixBaseOrder;++i)
    {
      currNest = td->nodes[i].nest;
      while(ind < Nparts)
	{
	  partNest = parts[ind].nest >> shift;
	  
	  if(partNest != currNest)
	    break;
	  
	  if(td->nodes[i].np == 0)
	    td->nodes[i].pstart = ind;
	  
	  ++(td->nodes[i].np);
	  
	  ++ind;
	}
    }
  for(i=0;i<NpixBaseOrder;++i)
    {
      for(j=td->nodes[i].pstart;j<td->nodes[i].np+td->nodes[i].pstart;++j)
	{
	  if(td->nodes[i].cosMaxSL < td->parts[j].smoothingLength)
	    td->nodes[i].cosMaxSL = td->parts[j].smoothingLength;
	  td->nodes[i].mass += td->parts[j].mass;
	  td->nodes[i].vec[0] += td->parts[j].mass*td->parts[j].pos[0];
	  td->nodes[i].vec[1] += td->parts[j].mass*td->parts[j].pos[1];
	  td->nodes[i].vec[2] += td->parts[j].mass*td->parts[j].pos[2];
	}
    }
  
  //now refine each tree node
  for(i=0;i<NpixBaseOrder;++i)
    refineTreeNode(td,i);
  
  //cut down nodes to size
  TreeNode *tempnodes;
  if(td->Nnodes == 0)
    {
      free(td->nodes);
      td->nodes = NULL;
    }
  else
    {
      tempnodes = (TreeNode*)realloc(td->nodes,sizeof(TreeNode)*(td->Nnodes));
      if(tempnodes != NULL)
	{
	  td->nodes = tempnodes;
	  td->NnodesAlloc = td->Nnodes;
	}
      else
	{
	  fprintf(stderr,"could not realloc tree nodes after building!\n");
	  fflush(stderr);
	  assert(tempnodes != NULL);
	}
    }
  
  //finish up nodes
  for(i=0;i<td->Nnodes;++i)
    {
      nest2vec(td->nodes[i].nest,td->nodes[i].vecG,td->nodes[i].order);
      
      if(td->nodes[i].mass > 0)
	{
	  td->nodes[i].vec[0] /= td->nodes[i].mass;
	  td->nodes[i].vec[1] /= td->nodes[i].mass;
	  td->nodes[i].vec[2] /= td->nodes[i].mass;
	  
	  r = sqrt(td->nodes[i].vec[0]*td->nodes[i].vec[0] + 
		   td->nodes[i].vec[1]*td->nodes[i].vec[1] + 
		   td->nodes[i].vec[2]*td->nodes[i].vec[2]);
	  
	  td->nodes[i].vec[0] /= r;
	  td->nodes[i].vec[1] /= r;
	  td->nodes[i].vec[2] /= r;
	}
      
      td->nodes[i].cosMaxSL += 2.0*sqrt(hpix_cell_areas[td->nodes[i].order]);
      if(td->nodes[i].cosMaxSL >= M_PI)
	td->nodes[i].cosMaxSL = -1.0;
      else
	td->nodes[i].cosMaxSL = cos(td->nodes[i].cosMaxSL);
    }
  
  if(ThisTask == 0)
    {
      fprintf(stderr,"tree has %ld nodes using %lf MB.\n",td->Nnodes,1.0*(td->Nnodes)*sizeof(TreeNode)/1024.0/1024.0);
      fflush(stderr);
    }
  
  timeB += MPI_Wtime();
  
  return td;
}

void destroyTree(TreeData td)
{
  if(td->nodes != NULL)
    free(td->nodes);
  free(td);
}

static inline void force_shear_parang_ppshort_sigma(double vec[3], double vnorm, double vecp[3], double cosr, 
						    double sigma, double cosSigma, TreeData td,
						    double mass, TreeWalkData *twd)
{
  // sin and cos of arcdist between points 
  // note that cosr = vec[0]*vecp[0] + vec[1]*vecp[1] + vec[2]*vecp[2];
  double sinr = sqrt((1.0 - cosr)*(1.0 + cosr));
  double sinp,cosp,norm;
  norm = -vnorm*sinr;
  sinp = (vecp[1]*vec[0] - vecp[0]*vec[1])/norm;
  cosp = (vec[2]*(vecp[0]*vec[0] + vecp[1]*vec[1]) - vecp[2]*vnorm*vnorm)/norm;
  
  //factors needed for comp of f, gammaE, and pot 
  double mass_4pi;
  double mass_4pi_cosrm1;
  double rs2;
  double fr,two_gammaE,pot,potshift;
  double rs;
  
  mass_4pi = mass/(12.5663706143591729538505735331180115367886775975);
  mass_4pi_cosrm1 = mass_4pi/(cosr-1.0);
  
  if(cosr >= cosSigma)
    {
      epkern_alpha_shear(acos(cosr),cosr,sinr,sigma,&fr,&two_gammaE);
      fr *= mass;
      two_gammaE *= mass;
    }
  else
    {
      //FIXME twd->pot += mass_4pi*(td->potTable[i]*onemw + td->potTable[ip1]*w);
      fr = -mass_4pi_cosrm1*sinr;
      two_gammaE = mass_4pi_cosrm1*(cosr + 1.0);
    }
  
  double expf;
  expf = exp((cosr-1.0)/td->thetaS2);
  fr *= expf;
  two_gammaE *= expf*(1.0 + sinr*sinr/td->thetaS2/(1.0 + cosr));
  
  twd->alpha[0] += fr*cosp;
  twd->alpha[1] += fr*sinp;
  
  twd->U[0] += two_gammaE*(cosp*cosp - 0.5);     //2*0+0 = 0
  twd->U[1] += two_gammaE*sinp*cosp;             //2*0+1 = 1
  //twd->U[2] += shtens[1];                      //2*1+0 = 2 //2*0+1 = 1
  //twd->U[3] += -shtens[3];                     //2*1+1 = 3
}

static void computePotentialForceShearTree_recursive(double vec[3], double vnorm, double BHCrit2, TreeData td, long nodeInd, TreeWalkData *twd)
{
  double cosarcDistG,cosarcDistCOM;
  double BHRat2;
  double vecp[3];
  long i,maxp;
  
  //too far away
  cosarcDistG = vec[0]*td->nodes[nodeInd].vecG[0] + vec[1]*td->nodes[nodeInd].vecG[1] + vec[2]*td->nodes[nodeInd].vecG[2];
  if(cosarcDistG < td->nodeCosRCut[td->nodes[nodeInd].order])
    return;
  
  //if small enough, treat as single point mass
  cosarcDistCOM = vec[0]*td->nodes[nodeInd].vec[0] + vec[1]*td->nodes[nodeInd].vec[1] + vec[2]*td->nodes[nodeInd].vec[2];
  BHRat2 = (hpix_cell_areas[td->nodes[nodeInd].order])/(2.0*M_PI*(1.0 - cosarcDistCOM));
  if(BHRat2 < BHCrit2 && cosarcDistCOM < td->nodes[nodeInd].cosMaxSL)
    {
      if(td->nodes[nodeInd].mass > 0.0)
	{
	  force_shear_parang_ppshort_sigma(vec,vnorm,td->nodes[nodeInd].vec,cosarcDistCOM,
					   0.0,1.0,td,td->nodes[nodeInd].mass,twd);
#ifdef GET_TREE_STATS
	      ++(twd->NumInteractTreeWalkNode);
#endif
	}
      
      return;
    }
  else
    {
      //check if leaf node
      if(td->nodes[nodeInd].child[0] == -1 && td->nodes[nodeInd].child[1] == -1 && td->nodes[nodeInd].child[2] == -1 && td->nodes[nodeInd].child[3] == -1)
	{
	  maxp = td->nodes[nodeInd].np + td->nodes[nodeInd].pstart;
	  for(i=td->nodes[nodeInd].pstart;i<maxp;++i)
	    {
	      vecp[0] = td->parts[i].pos[0];
	      vecp[1] = td->parts[i].pos[1];
	      vecp[2] = td->parts[i].pos[2];
	      vecp[0] /= td->parts[i].r;
	      vecp[1] /= td->parts[i].r;
	      vecp[2] /= td->parts[i].r;
	      cosarcDistCOM = vec[0]*vecp[0] + vec[1]*vecp[1] + vec[2]*vecp[2];
	      
	      force_shear_parang_ppshort_sigma(vec,vnorm,vecp,cosarcDistCOM,
					       td->parts[i].smoothingLength,td->parts[i].cosSmoothingLength,td,
					       td->parts[i].mass,twd);
#ifdef GET_TREE_STATS
	      ++(twd->NumInteractTreeWalk);
#endif
	    }
	  
	  return;
	}
      else //open node
	{
	  if(td->nodes[nodeInd].child[0] != -1)
	    computePotentialForceShearTree_recursive(vec,vnorm,BHCrit2,td,td->nodes[nodeInd].child[0],twd);
	  if(td->nodes[nodeInd].child[1] != -1)
	    computePotentialForceShearTree_recursive(vec,vnorm,BHCrit2,td,td->nodes[nodeInd].child[1],twd);
	  if(td->nodes[nodeInd].child[2] != -1)
	    computePotentialForceShearTree_recursive(vec,vnorm,BHCrit2,td,td->nodes[nodeInd].child[2],twd);
	  if(td->nodes[nodeInd].child[3] != -1)
	    computePotentialForceShearTree_recursive(vec,vnorm,BHCrit2,td,td->nodes[nodeInd].child[3],twd);
	  
	  return;
	}
    }
}

TreeWalkData computePotentialForceShearTree(double vec[3], double BHCrit2, TreeData td)
{
  long i;
  TreeWalkData twd;
  double alpha[2],U[4],vecp[3];
  double vnorm = sqrt((1.0 - vec[2])*(1.0 + vec[2]));
  twd.pot = 0.0;
  twd.alpha[0] = 0.0;
  twd.alpha[1] = 0.0;
  twd.U[0] = 0.0;
  twd.U[1] = 0.0;
  twd.NumInteractTreeWalk = 0;
  twd.NumInteractTreeWalkNode = 0;
  twd.Nempty = 0;
  
  for(i=0;i<12;++i)
    if(td->nodes[i].mass > 0.0)
      computePotentialForceShearTree_recursive(vec,vnorm,BHCrit2,td,i,&twd);
  
  twd.U[2] = twd.U[1];
  twd.U[3] = -1.0*twd.U[0];
  
  return twd;
}

TreeWalkData computePotentialForceShearDirectSummation(double vec[3], TreeData td)
{
  long i;
  TreeWalkData twd;
  double vecp[3];
  double cosarcDistCOM;
  double vnorm = sqrt((1.0 - vec[2])*(1.0 + vec[2]));
  twd.pot = 0.0;
  twd.alpha[0] = 0.0;
  twd.alpha[1] = 0.0;
  twd.U[0] = 0.0;
  twd.U[1] = 0.0;
  twd.NumInteractTreeWalk = 0;
  twd.NumInteractTreeWalkNode = 0;
  twd.Nempty = 0;
  
  for(i=0;i<td->Nparts;++i)
    {
      vecp[0] = td->parts[i].pos[0];
      vecp[1] = td->parts[i].pos[1];
      vecp[2] = td->parts[i].pos[2];
      vecp[0] /= td->parts[i].r;
      vecp[1] /= td->parts[i].r;
      vecp[2] /= td->parts[i].r;
      cosarcDistCOM = vec[0]*vecp[0] + vec[1]*vecp[1] + vec[2]*vecp[2];
      
      force_shear_parang_ppshort_sigma(vec,vnorm,vecp,cosarcDistCOM,
				       td->parts[i].smoothingLength,td->parts[i].cosSmoothingLength,td,
				       td->parts[i].mass,&twd);
#ifdef GET_TREE_STATS
      ++(twd.NumInteractTreeWalk);
#endif
    }
  
  twd.U[2] = twd.U[1];
  twd.U[3] = -1.0*twd.U[0];
  
  return twd;
}
