#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <float.h>
#include <assert.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include "raytrace.h"
#include "treecode.h"

static int addPartToTree(TreeData td, long nodeInd, long partInd);
static void refineTreeNode(TreeData td, long nodeInd);
static void expandTreeNodes(TreeData td);
static void pruneTree(TreeData td);
static int addExtraPartStuffToTree(TreeData td, long nodeInd, long partInd);
static double force_shear_parang_ppshort_sigma(double vec[3], double vnorm, double vecp[3], double cosr, 
					       double sigma, TreeData td,
					       double mass, double fvec[2], double shtens[4]);
static double force_shear_parang_ppshort(double vec[3], double vnorm, double vecp[3], double cosr, 
					 TreeData td, double mass, double fvec[2], double shtens[4]);
static void initKernelTables(void);

#define NumKernelTable 5000
static int initTables = 1;
static double dSigmaTable;
static double minSigmaTable;
static struct _kernelTableStr {
  double potshift;
  double a;
  double v;
  double r;
  double p;
  double k;
  double h;
  double cosSigma;
} kernelTable[NumKernelTable];

static gsl_spline *spline_alpha = NULL;
static gsl_interp_accel *acc_alpha = NULL; 
static gsl_spline *spline_gamma = NULL;
static gsl_interp_accel *acc_gamma = NULL; 

TreeWalkData computePotentialForceShearTree(double vec[3], double BHCrit, TreeData td)
{
  TreeWalkData twd;
  double cosarcDistCOM,cosarcDistG;
  double BHRat2,vecp[3];
  long nextPart,nodeInd;
  double BHCrit2 = BHCrit*BHCrit;
  double vnorm = sqrt((1.0 - vec[2])*(1.0 + vec[2]));
  double vnorm2 = vnorm*vnorm;
  //double U[4],alpha[2];
  double expf;
  
#ifdef CHECKTREEWALK
  int checkMe = 0;
  if(BHCrit < 0.0)
    {
      BHCrit = -1.0*BHCrit;
      checkMe = 1;
    }
#endif
  
  twd.pot = 0.0;
  twd.alpha[0] = 0.0;
  twd.alpha[1] = 0.0;
  twd.U[0] = 0.0;
  twd.U[1] = 0.0;
  //twd.U[2] = 0.0;
  //twd.U[3] = 0.0;
  twd.NumInteractTreeWalk = 0;
  twd.NumInteractTreeWalkNode = 0;
  twd.Nempty = 0;
  
  double sinr,cosr;
  double sinp,cosp,norm;
  double mass_4pi;
  double mass_4pi_cosrm1;
  double rs2;
  double fr,two_gammaE,pot,potshift;
  double rs;
  double onemw,w;
  long i,ip1;
  double a,b,d,f,g,h,k,p,r,v;
  double ws,onemws;
  long is,isp1;
  double sigma,mass;
  double cosrm1,cosrp1;
  double tmp;
  
  nodeInd = td->firstNode;
  while(nodeInd != -1)
    {
      
#ifdef CHECKTREEWALK
      if(checkMe)
        fprintf(stderr,"checkTreeWalk: nest,order = %ld|%ld, %ld nodeInd, over|down = %ld|%ld\n",
		td->nodes[nodeInd].nest,td->nodes[nodeInd].order,nodeInd,td->nodes[nodeInd].over,td->nodes[nodeInd].down);
#endif
      
      //do not need this anymore since we are pruning the tree
      //skip node if it has zero mass
      //if(td->nodes[nodeInd].mass <= 0.0)
      // {
      //nodeInd = td->nodes[nodeInd].over;
      //++(twd.Nempty);
      //continue;
      //}
      
      //skip node if it is too far away
      cosarcDistG = vec[0]*td->nodes[nodeInd].vecG[0] + vec[1]*td->nodes[nodeInd].vecG[1] + vec[2]*td->nodes[nodeInd].vecG[2];
      if(cosarcDistG < td->nodeCosRCut[td->nodes[nodeInd].order])
	{
#ifdef CHECKTREEWALK
	  if(checkMe)
	    fprintf(stderr,"checkTreeWalk: nest,order = %ld|%ld, skipping node that is too far away, over = %ld\n",
		    td->nodes[nodeInd].nest,td->nodes[nodeInd].order,td->nodes[nodeInd].over);
#endif
	  nodeInd = td->nodes[nodeInd].over;
	  continue;
	}
      
      //check if node is a leaf
      if(td->nodes[nodeInd].down == -1)
	{
	  if(td->links[td->nodes[nodeInd].partInd] == -1)
	    {
	      //node is leaf with one particle
	      sigma = td->parts[td->nodes[nodeInd].partInd].smoothingLength;
	      mass = td->nodes[nodeInd].mass;
	      
	      //get arcdist and paralactic angles	      
	      cosr = vec[0]*td->nodes[nodeInd].vec[0] + vec[1]*td->nodes[nodeInd].vec[1];
	      cosp = vec[2]*cosr - td->nodes[nodeInd].vec[2]*vnorm2;
	      cosr += vec[2]*td->nodes[nodeInd].vec[2];
	      cosrp1 = cosr + 1.0;
	      cosrm1 = cosr - 1.0;
	      
	      sinr = sqrt(-cosrm1*cosrp1);
	      norm = -vnorm*sinr;
	      sinp = (td->nodes[nodeInd].vec[1]*vec[0] - td->nodes[nodeInd].vec[0]*vec[1])/norm;
	      cosp /= norm;
	      
	      //factors needed for comp of f, gammaE, and pot 
	      /*w = (cosr - td->minCosr)/td->dcosr;
	      i = (long) w;
	      ip1 = i + 1;
	      w -= i;
	      onemw = 1.0 - w;
	      */
	      
	      if(cosr >= td->parts[td->nodes[nodeInd].partInd].cosSmoothingLength)
		{ 
		  rs = acos(cosr)/sigma;
		  		  
		  //factors needed for comp of f, gammaE, and pot
		  ws = (sigma - minSigmaTable)/dSigmaTable;
		  is = (long) ws;
		  isp1 = is + 1;
		  ws -= is;
		  onemws = 1.0 - ws;
		  
		  a = kernelTable[is].a*onemws + kernelTable[isp1].a*ws;
		  v = kernelTable[is].v*onemws + kernelTable[isp1].v*ws;
		  r = kernelTable[is].r*onemws + kernelTable[isp1].r*ws;
		  k = kernelTable[is].k*onemws + kernelTable[isp1].k*ws;
		  h = kernelTable[is].h*onemws + kernelTable[isp1].h*ws;
		  		  
		  potshift = kernelTable[is].potshift*onemws + kernelTable[isp1].potshift*ws;
		  
		  if(rs < 0.5)
		    {
		      f = v - 3.2*h;  //16/5 = 3.2
		      d = 8.0*h+r;
		      //c = -8.0*h+p; this is always zero
		      b = 4.0*h+k;
		      
		      rs2 = rs*rs;
		      pot = mass*(a + rs2*(b + rs2*(d + rs*f)) + potshift);
		      fr = mass*rs*(2.0*b + rs2*(4.0*d + rs*5.0*f))/sigma;
		      two_gammaE = mass*(2.0*b + rs2*(12.0*d + rs*20.0*f))/sigma/sigma - cosr/sinr*fr; 
		    }
		  else 
		    {
		      p = kernelTable[is].p*onemws + kernelTable[isp1].p*ws;
		      g = a - h*0.1;

		      pot = mass*(g + rs*(h + rs*(k + rs*(p + rs*(r + v*rs)))) + potshift);
		      fr = mass*(h + rs*(2.0*k + rs*(3.0*p + rs*(4.0*r + 5.0*v*rs))))/sigma;
		      two_gammaE = mass*(2.0*k + rs*(6.0*p + rs*(12.0*r + 20.0*v*rs)))/sigma/sigma - cosr/sinr*fr;
		    }
		}
	      else
		{
		  mass_4pi = mass/(12.5663706143591729538505735331180115367886775975);
		  mass_4pi_cosrm1 = mass_4pi/cosrm1;
		  
		  pot = mass_4pi*(td->potTable[i]*onemw + td->potTable[ip1]*w);
		  fr = -mass_4pi_cosrm1*sinr;
		  two_gammaE = mass_4pi_cosrm1*cosrp1;
		}
	      
	      //multiply by the treePM force splitting function to remove long-range part
	      //fr *= (td->expAlphaTable[i]*onemw + td->expAlphaTable[ip1]*w);
	      //two_gammaE *= (td->expGammaTable[i]*onemw + td->expGammaTable[ip1]*w);
	      
	      //fr *= gsl_spline_eval(spline_alpha,cosr,acc_alpha);
	      //two_gammaE *= gsl_spline_eval(spline_gamma,cosr,acc_gamma);
	      
	      expf = exp((cosr-1.0)/td->thetaSplit2);
	      fr *= expf;
	      two_gammaE *= expf*(1.0 + sinr*sinr/td->thetaSplit2/(1.0 + cosr));
	      
	      twd.pot += pot;
	      twd.alpha[0] += fr*cosp;
	      twd.alpha[1] += fr*sinp;
	      twd.U[0] += two_gammaE*(cosp*cosp - 0.5);
	      twd.U[1] += two_gammaE*sinp*cosp;
	      //shtens[2] = shtens[1];  //2*1+0 = 2 //2*0+1 = 1
	      //shtens[3] = -shtens[3]; //2*1+1 = 3
	      
	      /* old code uses function call but is slower by ~25%
		 cosarcDistCOM = vec[0]*td->nodes[nodeInd].vec[0] + vec[1]*td->nodes[nodeInd].vec[1] + vec[2]*td->nodes[nodeInd].vec[2];
		 twd.pot += force_shear_parang_ppshort_sigma(vec,vnorm,td->nodes[nodeInd].vec,cosarcDistCOM,
		 td->parts[td->nodes[nodeInd].partInd].smoothingLength,td,
		 td->nodes[nodeInd].mass,alpha,U);
		 
		 twd.alpha[0] += alpha[0];
		 twd.alpha[1] += alpha[1];
		 twd.U[0] += U[0];
		 twd.U[1] += U[1];
		 //twd.U[2] += U[2];                                                                                                                                                                        
		 //twd.U[3] += U[3];                                                                                                                                                                        
		 */
	      
	      ++(twd.NumInteractTreeWalk);

#ifdef CHECKTREEWALK
              if(checkMe)
                fprintf(stderr,"checkTreeWalk: nest,order = %ld|%ld, single part added, vec = %lf|%lf|%lf, partInd = %ld\n",
                        td->nodes[nodeInd].nest,td->nodes[nodeInd].order,td->nodes[nodeInd].vec[0],td->nodes[nodeInd].vec[1],td->nodes[nodeInd].vec[2],td->nodes[nodeInd].partInd);
#endif

	    }
	  else //multiple particles in the node
	    {
	      nextPart = td->nodes[nodeInd].partInd;
	      while(nextPart != -1)
		{
		  vecp[0] = td->parts[nextPart].pos[0];
		  vecp[1] = td->parts[nextPart].pos[1];
		  vecp[2] = td->parts[nextPart].pos[2];
		  vecp[0] /= td->parts[nextPart].r;
		  vecp[1] /= td->parts[nextPart].r;
		  vecp[2] /= td->parts[nextPart].r;
		  
		  sigma = td->parts[nextPart].smoothingLength;
		  mass = td->parts[nextPart].mass;
		  	      
		  //arcdist and paralactic angles
		  cosr = vec[0]*vecp[0] + vec[1]*vecp[1];
		  cosp = vec[2]*cosr - vecp[2]*vnorm2;
		  cosr += vec[2]*vecp[2];
		  cosrp1 = cosr + 1.0;
		  cosrm1 = cosr - 1.0;

		  sinr = sqrt(-cosrm1*cosrp1);
		  norm = -vnorm*sinr;
		  sinp = (vecp[1]*vec[0] - vecp[0]*vec[1])/norm;
		  cosp /= norm;
		  
		  //factors needed for comp of f, gammaE, and pot 
		  /*w = (cosr - td->minCosr)/td->dcosr;
		  i = (long) w;
		  ip1 = i + 1;
		  w -= i;
		  onemw = 1.0 - w;
		  */
		  
		  if(cosr >= td->parts[nextPart].cosSmoothingLength)
		    { 
		      rs = acos(cosr)/sigma;
		      
		      //factors needed for comp of f, gammaE, and pot
		      ws = (sigma - minSigmaTable)/dSigmaTable;
		      is = (long) ws;
		      isp1 = is + 1;
		      ws -= is;
		      onemws = 1.0 - ws;
		      
		      a = kernelTable[is].a*onemws + kernelTable[isp1].a*ws;
		      v = kernelTable[is].v*onemws + kernelTable[isp1].v*ws;
		      r = kernelTable[is].r*onemws + kernelTable[isp1].r*ws;
		      k = kernelTable[is].k*onemws + kernelTable[isp1].k*ws;
		      h = kernelTable[is].h*onemws + kernelTable[isp1].h*ws;
		      		      
		      potshift = kernelTable[is].potshift*onemws + kernelTable[isp1].potshift*ws;
		      
		      if(rs < 0.5)
			{
			  f = v - 3.2*h;  //16/5 = 3.2
			  d = 8.0*h+r;
			  //c = -8.0*h+p; this is always zero
			  b = 4.0*h+k;
			  
			  rs2 = rs*rs;
			  pot = mass*(a + rs2*(b + rs2*(d + rs*f)) + potshift);
			  fr = mass*rs*(2.0*b + rs2*(4.0*d + rs*5.0*f))/sigma;
			  two_gammaE = mass*(2.0*b + rs2*(12.0*d + rs*20.0*f))/sigma/sigma - cosr/sinr*fr; 
			}
		      else 
			{
			  p = kernelTable[is].p*onemws + kernelTable[isp1].p*ws;
			  g = a - h*0.1;
			  
			  pot = mass*(g + rs*(h + rs*(k + rs*(p + rs*(r + v*rs)))) + potshift);
			  fr = mass*(h + rs*(2.0*k + rs*(3.0*p + rs*(4.0*r + 5.0*v*rs))))/sigma;
			  two_gammaE = mass*(2.0*k + rs*(6.0*p + rs*(12.0*r + 20.0*v*rs)))/sigma/sigma - cosr/sinr*fr;
			}
		    }
		  else
		    {
		      mass_4pi = mass/(12.5663706143591729538505735331180115367886775975);
		      mass_4pi_cosrm1 = mass_4pi/cosrm1;
		      
		      pot = mass_4pi*(td->potTable[i]*onemw + td->potTable[ip1]*w);
		      fr = -mass_4pi_cosrm1*sinr;
		      two_gammaE = mass_4pi_cosrm1*cosrp1;
		    }
		  
		  //multiply by the treePM force splitting function to remove long-range part
		  //fr *= (td->expAlphaTable[i]*onemw + td->expAlphaTable[ip1]*w);
		  //two_gammaE *= (td->expGammaTable[i]*onemw + td->expGammaTable[ip1]*w);
		  
		  //fr *= gsl_spline_eval(spline_alpha,cosr,acc_alpha);
		  //two_gammaE *= gsl_spline_eval(spline_gamma,cosr,acc_gamma);

		  expf = exp((cosr-1.0)/td->thetaSplit2);
		  fr *= expf;
		  two_gammaE *= expf*(1.0 + sinr*sinr/td->thetaSplit2/(1.0 + cosr));
	      
		  twd.pot += pot;
		  twd.alpha[0] += fr*cosp;
		  twd.alpha[1] += fr*sinp;
		  twd.U[0] += two_gammaE*(cosp*cosp - 0.5);
		  twd.U[1] += two_gammaE*sinp*cosp;
		  //shtens[2] = shtens[1];   //2*1+0 = 2 //2*0+1 = 1
		  //shtens[3] = -shtens[3];  //2*1+1 = 3
		  
		  /* old code uses function call but is slower by ~25%
		     cosarcDistCOM = vec[0]*vecp[0] + vec[1]*vecp[1] + vec[2]*vecp[2];
		     twd.pot += force_shear_parang_ppshort_sigma(vec,vnorm,vecp,cosarcDistCOM,
		     td->parts[nextPart].smoothingLength,td,
		     td->parts[nextPart].mass,alpha,U);
		     
		     twd.alpha[0] += alpha[0];
		     twd.alpha[1] += alpha[1];
		     twd.U[0] += U[0];
		     twd.U[1] += U[1];
		     //twd.U[2] += U[2];
		     //twd.U[3] += U[3];
		     */
		  
		  ++(twd.NumInteractTreeWalk);

#ifdef CHECKTREEWALK
		  if(checkMe)
		    fprintf(stderr,"checkTreeWalk: nest,order = %ld|%ld, multi parts added, vec = %lf|%lf|%lf, partInd = %ld\n",
			    td->nodes[nodeInd].nest,td->nodes[nodeInd].order,vecp[0],vecp[1],vecp[2],nextPart);
#endif
  
		  nextPart = td->links[nextPart];
		}
	    }
	  
	  //go over to continue tree walk
#ifdef CHECKTREEWALK
	  if(checkMe)
	    fprintf(stderr,"checkTreeWalk: nest,order = %ld|%ld, going over after single/multipart, over = %ld\n",
		    td->nodes[nodeInd].nest,td->nodes[nodeInd].order,td->nodes[nodeInd].over);
#endif
          nodeInd = td->nodes[nodeInd].over;
          continue;
	}
      
      //always open nodes that are not too far away, are marked to be opened, and are not leaf nodes
      if(td->nodes[nodeInd].alwaysOpen)
        {

#ifdef CHECKTREEWALK
          if(checkMe)
            fprintf(stderr,"checkTreeWalk: nest,order = %ld|%ld, node forced open, down = %ld\n",
                    td->nodes[nodeInd].nest,td->nodes[nodeInd].order,td->nodes[nodeInd].down);
#endif

          nodeInd = td->nodes[nodeInd].down;
          continue;
        }
      
      //check if node satisfies BH crit and satisfies geom. criterion - node is already not a leaf, is not forced to be opened, and is not too far away
      cosarcDistCOM = vec[0]*td->nodes[nodeInd].vec[0] + vec[1]*td->nodes[nodeInd].vec[1] + vec[2]*td->nodes[nodeInd].vec[2];
      BHRat2 = (td->nodeArcSizes2[td->nodes[nodeInd].order])/(2.0*M_PI*(1.0 - cosarcDistCOM));
      if(BHRat2 < BHCrit2 && cosarcDistG < td->cosNodeArcSizesGeomLim[td->nodes[nodeInd].order])
	{
	  mass = td->nodes[nodeInd].mass;

	  //arcdist and paralactic angles
	  cosr = cosarcDistCOM;
	  tmp = cosr - vec[2]*td->nodes[nodeInd].vec[2];
	  cosrp1 = cosr + 1.0;
	  cosrm1 = cosr - 1.0;

	  sinr = sqrt(-cosrm1*cosrp1);
	  norm = -vnorm*sinr;
	  sinp = (td->nodes[nodeInd].vec[1]*vec[0] - td->nodes[nodeInd].vec[0]*vec[1])/norm;
	  cosp = (vec[2]*tmp - td->nodes[nodeInd].vec[2]*vnorm2)/norm;
	  
	  //factors needed for comp of f, gammaE, and pot 
	  /*w = (cosr - td->minCosr)/td->dcosr;
	  i = (long) w;
	  ip1 = i + 1;
	  w -= i;
	  onemw = 1.0 - w;
	  */
	  
	  mass_4pi = mass/(12.5663706143591729538505735331180115367886775975);
	  mass_4pi_cosrm1 = mass_4pi/cosrm1;
	  
	  pot = mass_4pi*(td->potTable[i]*onemw + td->potTable[ip1]*w);
	  fr = -mass_4pi_cosrm1*sinr;
	  two_gammaE = mass_4pi_cosrm1*cosrp1;
	  
	  //multiply by the treePM force splitting function to remove long-range part
	  //fr *= (td->expAlphaTable[i]*onemw + td->expAlphaTable[ip1]*w);
	  //two_gammaE *= (td->expGammaTable[i]*onemw + td->expGammaTable[ip1]*w);
	  
	  //fr *= gsl_spline_eval(spline_alpha,cosr,acc_alpha);
	  //two_gammaE *= gsl_spline_eval(spline_gamma,cosr,acc_gamma);
	      
	  expf = exp((cosr-1.0)/td->thetaSplit2);
	  fr *= expf;
	  two_gammaE *= expf*(1.0 + sinr*sinr/td->thetaSplit2/(1.0 + cosr));
	  
	  twd.pot += pot;
	  twd.alpha[0] += fr*cosp;
	  twd.alpha[1] += fr*sinp;
	  twd.U[0] += two_gammaE*(cosp*cosp - 0.5);
	  twd.U[1] += two_gammaE*sinp*cosp;
	  //shtens[2] = shtens[1];  //2*1+0 = 2 //2*0+1 = 1                                                                                                                                  
	  //shtens[3] = -shtens[3]; //2*1+1 = 3                                                                                                                                              

	  /* old code uses function call but is slower by ~25%
	     twd.pot += force_shear_parang_ppshort(vec,vnorm,td->nodes[nodeInd].vec,cosarcDistCOM,td,
	     td->nodes[nodeInd].mass,alpha,U);
	     
	     twd.alpha[0] += alpha[0];
	     twd.alpha[1] += alpha[1];
	     twd.U[0] += U[0];
	     twd.U[1] += U[1];
	     //twd.U[2] += U[2];
	     //twd.U[3] += U[3];
	     */
	  
	  ++(twd.NumInteractTreeWalkNode);
	  
#ifdef CHECKTREEWALK
          if(checkMe)
            fprintf(stderr,"checkTreeWalk: nest,order = %ld|%ld, node added, vec = %lf|%lf|%lf, mass = %lf, over = %ld\n",
                    td->nodes[nodeInd].nest,td->nodes[nodeInd].order,
		    td->nodes[nodeInd].vec[0],td->nodes[nodeInd].vec[1],td->nodes[nodeInd].vec[2],td->nodes[nodeInd].mass,td->nodes[nodeInd].over);
#endif
	  
	  //go over to continue tree walk
	  nodeInd = td->nodes[nodeInd].over;
	  continue;
	}
      else
	{
	  //node does not satisify BH criterion but has mass and children - so open tree node
#ifdef CHECKTREEWALK
          if(checkMe)
            fprintf(stderr,"checkTreeWalk: nest,order = %ld|%ld, node opened since failed BH crit, down = %ld\n",
		    td->nodes[nodeInd].nest,td->nodes[nodeInd].order,td->nodes[nodeInd].down);
#endif
	  nodeInd = td->nodes[nodeInd].down;
	  continue;
	}
      
      //never get here
      fprintf(stderr,"error in tree walk! nodeInd = %ld, down = %ld, mass = %g\n",nodeInd,td->nodes[nodeInd].down,td->nodes[nodeInd].mass);
      assert(0);
    }
  
  twd.U[2] = twd.U[1];
  twd.U[3] = -1.0*twd.U[0];
  
  return twd;
}

static double times[10];

TreeData buildTree(Part *parts, long Nparts, double thetaSplit, long baseOrder)
{
  long i,partInTree,baseInd;
  long shift;
  long NpixBaseOrder = order2npix(baseOrder);
  TreeData td;
  double r,s;
  double timeB = 0.0;
  
  timeB -= MPI_Wtime();
  for(i=0;i<10;++i)
    times[i] = 0.0;
  
  times[0] -= MPI_Wtime();
  td = (TreeData)malloc(sizeof(_TreeData));
  assert(td != NULL);
  
  for(i=0;i<=HEALPIX_UTILS_MAXORDER;++i)
    {
      td->nodeArcSizes2[i] = 4.0*M_PI/order2npix(i);
      td->nodeArcSizes[i] = sqrt(td->nodeArcSizes2[i]);
      s = thetaSplit*MAX_RADTREEWALK_TO_SPLIT_RATIO + td->nodeArcSizes[i];
      if(s < M_PI)
	td->nodeCosRCut[i] = cos(s);
      else
	td->nodeCosRCut[i] = -1.0;
      
      td->cosNodeArcSizesGeomLim[i] = cos(td->nodeArcSizes[i]*2.0);
    }
  
  //pre-compute factors to save time later
  td->thetaSplit = thetaSplit;
  td->thetaSplit2 = thetaSplit*thetaSplit;
    
  td->baseOrder = baseOrder;
  td->NpixBaseOrder = NpixBaseOrder;
  
  if((MAX_RADTREEWALK_TO_SPLIT_RATIO*td->thetaSplit+td->nodeArcSizes[baseOrder])*2.0 > M_PI)
    s = -1.0;
  else
    s = cos((MAX_RADTREEWALK_TO_SPLIT_RATIO*td->thetaSplit+td->nodeArcSizes[baseOrder])*2.0);
  td->dcosr = acos(s)/((double) (NumTreeExpFactTable-1)); //(1.0 - s)/((double) (NumTreeExpFactTable-1));
  td->minCosr = 0.0; //s;
  
  for(i=0;i<NumTreeExpFactTable-1;++i)
    {
      //td->cosrTable[i] = i*td->dcosr + s;
      td->cosrTable[i] = cos((NumTreeExpFactTable-1-i)*td->dcosr + td->minCosr);
      
      if(td->cosrTable[i] > 1.0)
	td->cosrTable[i] = 1.0;
      
      if(1.0 - td->cosrTable[i]*td->cosrTable[i] < 0.0)
	r = 0.0;
      else
	r = sqrt(1.0 - td->cosrTable[i]*td->cosrTable[i]);
      
      td->expAlphaTable[i] = exp((td->cosrTable[i]-1.0)/td->thetaSplit/td->thetaSplit);
      td->expGammaTable[i] = td->expAlphaTable[i]*(1.0 + r*r/td->thetaSplit/td->thetaSplit/(td->cosrTable[i]+1.0));
      
      r = (td->cosrTable[i] - 1.0)/td->thetaSplit/td->thetaSplit;
      if(r < -20.0)
        td->potTable[i]  = 0.0;
      else
        {
	  if(r >= 0.0)
	    r = -1e-120;
	  td->potTable[i] = gsl_sf_expint_Ei(r);
	}
    }
  
  td->cosrTable[NumTreeExpFactTable-1] = 1.0;
  td->expAlphaTable[NumTreeExpFactTable-1] = 1.0;
  td->expGammaTable[NumTreeExpFactTable-1] = 1.0;
  r = -1e-120;
  td->potTable[i] = gsl_sf_expint_Ei(r);
  
  //if(ThisTask == 0)
  //for(i=0;i<NumTreeExpFactTable;++i)
  //  fprintf(stderr,"cosr = %g, a,g = %g|%g\n",td->cosrTable[i],td->expAlphaTable[i],td->expGammaTable[i]);
  
  if(spline_alpha != NULL)
    gsl_spline_free(spline_alpha);
  spline_alpha = gsl_spline_alloc(gsl_interp_cspline,(size_t) (NumTreeExpFactTable));
  gsl_spline_init(spline_alpha,td->cosrTable,td->expAlphaTable,(size_t) (NumTreeExpFactTable));
  if(acc_alpha != NULL)
    gsl_interp_accel_reset(acc_alpha);
  else
    acc_alpha = gsl_interp_accel_alloc();
  
  if(spline_gamma != NULL)
    gsl_spline_free(spline_gamma);
  spline_gamma = gsl_spline_alloc(gsl_interp_cspline,(size_t) (NumTreeExpFactTable));
  gsl_spline_init(spline_gamma,td->cosrTable,td->expGammaTable,(size_t) (NumTreeExpFactTable));
  if(acc_gamma != NULL)
    gsl_interp_accel_reset(acc_gamma);
  else
    acc_gamma = gsl_interp_accel_alloc();
  
  if(initTables)
    initKernelTables();
  
  td->NnodesAlloc = Nparts*rayTraceData.treeAllocFactor;
  if(td->NnodesAlloc < NpixBaseOrder)
    td->NnodesAlloc = 4*NpixBaseOrder;
  td->nodes = (TreeNode*)malloc(sizeof(TreeNode)*td->NnodesAlloc);
  assert(td->nodes != NULL);
  td->Nnodes = NpixBaseOrder;
    
  for(i=0;i<NpixBaseOrder;++i)
    {
      td->nodes[i].order =  baseOrder;
      td->nodes[i].nest = i;
      td->nodes[i].down = -1;
      td->nodes[i].over = i+1;
      td->nodes[i].partInd = -1;
      td->nodes[i].mass = 0.0;
      td->nodes[i].vec[0] = 0.0;
      td->nodes[i].vec[1] = 0.0;
      td->nodes[i].vec[2] = 0.0;
      td->nodes[i].cosMaxSL = 0.0;
      td->nodes[i].alwaysOpen = 0;
    }
  td->nodes[NpixBaseOrder-1].over = -1;
  
  td->parts = parts;
  td->Nparts = Nparts;
  td->links = (long*)malloc(sizeof(long)*Nparts);
  assert(td->links != NULL);
  
  for(i=0;i<Nparts;++i)
    td->links[i] = -1;
  times[0] += MPI_Wtime();
  
  times[1] -= MPI_Wtime();
  shift = 2*(HEALPIX_UTILS_MAXORDER-baseOrder);
  for(i=0;i<Nparts;++i)
    {
      baseInd = td->parts[i].nest >> shift;
      partInTree = addPartToTree(td,baseInd,i);
      assert(partInTree);
    }
  times[1] += MPI_Wtime();
  
  times[0] -= MPI_Wtime();
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
	  assert(tempnodes != NULL);
	}
    }
  times[0] += MPI_Wtime();
  
  times[2] -= MPI_Wtime();
  //get extra stuff for each part
  shift = 2*(HEALPIX_UTILS_MAXORDER-baseOrder);
  for(i=0;i<Nparts;++i)
    {
      baseInd = td->parts[i].nest >> shift;
      partInTree = addExtraPartStuffToTree(td,baseInd,i);
      assert(partInTree);
    }
  
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
    }
  
  pruneTree(td);
  
  //get first non-zero node
  long nodeInd = 0;
  if(td->Nnodes > 0)
    {
      while(td->nodes[nodeInd].mass <= 0.0)
	{
	  assert(td->nodes[nodeInd].down == -1);
	  nodeInd = td->nodes[nodeInd].over;
	  
	  if(nodeInd == -1)
	    break;
	}
      td->firstNode = nodeInd;
    }
  else
    td->firstNode = -1;
  
  times[2] += MPI_Wtime();
  
  timeB += MPI_Wtime();

#ifdef DEBUG  
  if(ThisTask == 0)
    fprintf(stderr,"tree built in %lg seconds. (init,addparts,finalize,refine,expand = %lg|%lg|%lg|%lg|%lg seconds, baseOrder = %ld)\n",
	    timeB,times[0],times[1],times[2],times[3],times[4],baseOrder);
#endif
  //#else
  //if(ThisTask == 0)
  //fprintf(stderr,"tree built in %lg seconds.\n",timeB);
  //#endif
    
  return td;
}

void destroyTree(TreeData td)
{
  if(td->nodes != NULL)
    free(td->nodes);
  free(td->links);
  free(td);
}

static int addExtraPartStuffToTree(TreeData td, long nodeInd, long partInd)
{
  long i,shift;
  long baseNest = td->parts[partInd].nest;
  
  while(1)
    {
      if(td->parts[partInd].smoothingLength > td->nodes[nodeInd].cosMaxSL)
	td->nodes[nodeInd].cosMaxSL = td->parts[partInd].smoothingLength;
      
      if(td->nodes[nodeInd].down == -1) //has no children - so done
	{
	  return 1;
	}
      else //node has children so go to them
	{
	  shift = 2*(HEALPIX_UTILS_MAXORDER - td->nodes[nodeInd].order);
	  i = (baseNest >> (shift-2)) - ((baseNest >> (shift)) << 2);
	  nodeInd = td->nodes[nodeInd].down+i;
	}
    }
  
  //never get here
  return 0;
}

static int addPartToTree(TreeData td, long nodeInd, long partInd)
{
  long i,nextPart,NumPartsInNode;
  long shift;
  long baseNest = td->parts[partInd].nest;
  
  while(1)
    {
      //accumulate mass in node
      td->nodes[nodeInd].mass += td->parts[partInd].mass;
      td->nodes[nodeInd].vec[0] += td->parts[partInd].pos[0]*td->parts[partInd].mass;
      td->nodes[nodeInd].vec[1] += td->parts[partInd].pos[1]*td->parts[partInd].mass;
      td->nodes[nodeInd].vec[2] += td->parts[partInd].pos[2]*td->parts[partInd].mass;
      
      if(td->nodes[nodeInd].down == -1) //has no children - so try and add part to node
	{
	  nextPart = td->nodes[nodeInd].partInd;
	  if(nextPart == -1)
	    NumPartsInNode = 0;
	  else
	    {
	      NumPartsInNode = 1;
	      while(td->links[nextPart] != -1)
		{
		  ++NumPartsInNode;
		  nextPart = td->links[nextPart];
		}
	    }
	  
	  if(NumPartsInNode == 0) //no other parts in this node - add part to node and return since done
	    {
	      td->nodes[nodeInd].partInd = partInd;
	      td->links[partInd] = -1;
	      //FIXME td->parts[partInd].smoothingLength = td->nodeArcSizes[td->nodes[nodeInd].order];
	      return 1;
	    }
	  else if(td->nodes[nodeInd].order < HEALPIX_UTILS_MAXORDER)
	    {
	      //not at max depth so need to refine the node
	      refineTreeNode(td,nodeInd);
	      
	      //set nodeInd to correct child and conitnue
	      shift = 2*(HEALPIX_UTILS_MAXORDER - td->nodes[nodeInd].order);
	      i = (baseNest >> (shift-2)) - ((baseNest >> (shift)) << 2);
	      nodeInd = td->nodes[nodeInd].down+i;
	    }
	  else  //at max depth so add part to list
	    {
	      //add part to top of list
	      td->links[partInd] = td->nodes[nodeInd].partInd;
	      td->nodes[nodeInd].partInd = partInd;
	      //FIXME td->parts[partInd].smoothingLength = td->nodeArcSizes[td->nodes[nodeInd].order];
	      return 1;
	    }
	}
      else //node has children so go to them
	{
	  shift = 2*(HEALPIX_UTILS_MAXORDER - td->nodes[nodeInd].order);
	  i = (baseNest >> (shift-2)) - ((baseNest >> (shift)) << 2);
	  nodeInd = td->nodes[nodeInd].down+i;
	}
    }
  
  //never get here
  return 0;
}

static void refineTreeNode(TreeData td, long nodeInd)
{
  long refOrder,refNest,refNodeLoc;
  long i,nest,shift,tailPart[4],nextPart;
  double sL;
  
  times[3] -= MPI_Wtime();
  
  if(td->Nnodes+4 > td->NnodesAlloc)
    expandTreeNodes(td); //make sure we have memory
  
  td->nodes[nodeInd].down = td->Nnodes;
  refOrder = td->nodes[nodeInd].order + 1;
  refNest = td->nodes[nodeInd].nest*4; //bit shift 2 bits to right, then we will fill from 0 to 3
  refNodeLoc = td->Nnodes;
  for(i=0;i<4;++i)
    {
      td->nodes[td->Nnodes].order = refOrder;
      td->nodes[td->Nnodes].nest = refNest + i;
      td->nodes[td->Nnodes].down = -1;
      td->nodes[td->Nnodes].over = refNodeLoc + i + 1;
      td->nodes[td->Nnodes].partInd = -1;
      td->nodes[td->Nnodes].vec[0] = 0.0;
      td->nodes[td->Nnodes].vec[1] = 0.0;
      td->nodes[td->Nnodes].vec[2] = 0.0;
      td->nodes[td->Nnodes].mass = 0.0;
      td->nodes[td->Nnodes].cosMaxSL = 0.0;
      td->nodes[td->Nnodes].alwaysOpen = 0;
      
      ++(td->Nnodes);
    }
  td->nodes[td->Nnodes-1].over = td->nodes[nodeInd].over;
  
  //give parts to nodes
  sL = td->nodeArcSizes[refOrder];
  if(td->nodes[nodeInd].partInd != -1)
    {
      shift = 2*(HEALPIX_UTILS_MAXORDER - refOrder);
      for(i=0;i<4;++i)
	tailPart[i] = -1;
      
      nextPart = td->nodes[nodeInd].partInd;
      while(nextPart != -1)
	{
	  nest = td->parts[nextPart].nest;
	  i = (nest >> shift) - ((nest >> (shift+2)) << 2);
	  
	  if(tailPart[i] == -1)
	    td->nodes[refNodeLoc+i].partInd = nextPart;
	  else
	    td->links[tailPart[i]] = nextPart;
	  
	  //compute mass and add to COM coords
	  td->nodes[refNodeLoc+i].mass += td->parts[nextPart].mass;
	  td->nodes[refNodeLoc+i].vec[0] += td->parts[nextPart].pos[0]*td->parts[nextPart].mass;
	  td->nodes[refNodeLoc+i].vec[1] += td->parts[nextPart].pos[1]*td->parts[nextPart].mass;
	  td->nodes[refNodeLoc+i].vec[2] += td->parts[nextPart].pos[2]*td->parts[nextPart].mass;
	  
	  //FIXME td->parts[nextPart].smoothingLength = sL;
	  
	  tailPart[i] = nextPart;
	  nextPart = td->links[nextPart];
	}
      
      for(i=0;i<4;++i)
	if(tailPart[i] != -1)
	  td->links[tailPart[i]] = -1;
      td->nodes[nodeInd].partInd = -1;
    }
  
  times[3] += MPI_Wtime();
}

static void expandTreeNodes(TreeData td)
{
  TreeNode *tempnodes;
  long nextra = td->Nparts*0.2;
  
  if(nextra < 1000)
    nextra = 1000;
  
  rayTraceData.treeAllocFactor += 0.1;
  
  if(ThisTask == 0)
    fprintf(stderr,"increased treeAllocFactor by 0.1 to %lg.\n",rayTraceData.treeAllocFactor);
  
  times[4] -= MPI_Wtime();
  
  tempnodes = (TreeNode*)realloc(td->nodes,sizeof(TreeNode)*(td->NnodesAlloc+nextra));
  if(tempnodes != NULL)
    {
      td->nodes = tempnodes;
      td->NnodesAlloc += nextra;
    }
  else
    {
      fprintf(stderr,"could not expand tree nodes: old size = %ld, new size = %ld\n",td->NnodesAlloc,td->NnodesAlloc+nextra);
      assert(tempnodes != NULL);
    }
  
  times[4] += MPI_Wtime();
}

//#define OLDPRUNE
#ifdef OLDPRUNE
static void pruneTree(TreeData td)
{
  long nodeInd,overNode,downNode;
  //prune over links
  for(nodeInd=0;nodeInd<td->Nnodes;++nodeInd)
    {
      if(td->nodes[nodeInd].over != -1)
	{
	  overNode = td->nodes[nodeInd].over;
	  while(td->nodes[overNode].mass <= 0.0)
	    {
	      assert(td->nodes[overNode].down == -1);
	      overNode = td->nodes[overNode].over;
	      
	      if(overNode == -1)
		break;
	    }
	  td->nodes[nodeInd].over = overNode;
	}
    }
  
  //prune down links
  for(nodeInd=0;nodeInd<td->Nnodes;++nodeInd)
    {
      if(td->nodes[nodeInd].down != -1)
	{
	  downNode = td->nodes[nodeInd].down;
	  while(td->nodes[downNode].mass <= 0.0)
	    {
	      assert(td->nodes[downNode].down == -1);
	      downNode = td->nodes[downNode].over;
	      
	      if(downNode == -1)
		break;
	    }
	  td->nodes[nodeInd].down = downNode;
 	}
     }
}
#else /* the code below is faster and uses less memory */
static void pruneTree(TreeData td)
{
  long *skipIndex,*tmp;
  long NumSkipIndex,NumSkipIndexAlloc;
  long i,j;
  
  long kids[4],mkids[4],nodeInd,currInd,firstKid,firstKidIndex,order;
  
  //prune over links of base nodes first
  for(i=0;i<11;++i)
    {
      for(j=i+1;j<12;++j)
	{
	  if(td->nodes[j].mass > 0.0)
	    {
	      td->nodes[i].over = j;
	      break;
	    }
	}      
    }
  
  //set over link of last base node with mass to -1
  for(i=11;i>=0;--i)
    {
      if(td->nodes[i].mass > 0.0)
	{
	  td->nodes[i].over = -1;
	  break;
	}
    }
  
  //now prune the rest of the nodes - do one level at a time, starting from the top of the tree
  for(order=0;order<=HEALPIX_UTILS_MAXORDER;++order)
    {
      for(nodeInd=0;nodeInd<td->Nnodes;++nodeInd)
	{
	  if(td->nodes[nodeInd].down != -1 && td->nodes[nodeInd].order == order)
	    {
	      //find the kids, make sure they are in the correct order, and check for mass
	      currInd = td->nodes[nodeInd].down;
	      for(i=0;i<4;++i)
		{
		  kids[i] = currInd;
		  
		  if(td->nodes[currInd].nest != td->nodes[nodeInd].nest*4 + i)
		    {
		      fprintf(stderr,"%d: parent nest|order = %ld|%ld down|over = %ld|%ld, kid %ld nest,order = %ld|%ld down|over = %ld|%ld, kid index = %ld\n",
			      ThisTask,td->nodes[nodeInd].nest,td->nodes[nodeInd].order,td->nodes[nodeInd].down,td->nodes[nodeInd].over,
			      i,td->nodes[currInd].nest,td->nodes[currInd].order,td->nodes[currInd].down,td->nodes[currInd].over,currInd);
		    }
		  assert(td->nodes[currInd].nest == td->nodes[nodeInd].nest*4 + i);
		  
		  if(td->nodes[currInd].mass > 0.0)
		    mkids[i] = 1;
		  else
		    mkids[i] = 0;
		  
		  currInd = td->nodes[currInd].over;
		}
	      	      
	      //get first kid with mass
	      firstKid = -1;
	      for(i=0;i<4;++i)
		if(mkids[i])
		  {
		    firstKidIndex = i;
		    firstKid = kids[i];
		    break;
		  }
	      assert(firstKid != -1);
	      
	      //set over links of kids
	      for(i=firstKidIndex;i<3;++i)
		{
		  if(mkids[i])
		    {
		      for(j=i+1;j<4;++j)
			{
			  if(mkids[j])
			    {
			      td->nodes[kids[i]].over = kids[j];
			      break;
			    }
			}
		    }
		}
	      
	      //make sure last kid points to the over node of the parent
	      for(i=3;i>=0;--i)
		{
		  if(mkids[i])
		    {
		      td->nodes[kids[i]].over = td->nodes[nodeInd].over;
		      break;
		    }
		}
	      
	      //set down links of the parent
	      td->nodes[nodeInd].down = firstKid;
	    }
	}
    }
  
  //now remove mass-less nodes
  NumSkipIndexAlloc = 10000;
  skipIndex = (long*)malloc(sizeof(long)*NumSkipIndexAlloc);
  assert(skipIndex != NULL);
  NumSkipIndex = 0;
  
  for(i=0;i<td->Nnodes;++i)
    {
      if(td->nodes[i].mass <= 0.0)
	{
	  if(NumSkipIndex >= NumSkipIndexAlloc)
	    {
	      NumSkipIndexAlloc += 10000;
	      tmp = (long*)realloc(skipIndex,sizeof(long)*NumSkipIndexAlloc);
	      assert(tmp != NULL);
	      skipIndex = tmp;
	    }
	  
	  skipIndex[NumSkipIndex] = i;
	  ++NumSkipIndex;
	}
      else
	{
	  //check that nodes only point to used nodes
	  if(td->nodes[i].down != -1)
	    assert(td->nodes[td->nodes[i].down].mass > 0.0);
	      
	  if(td->nodes[i].over != -1)
	    assert(td->nodes[td->nodes[i].over].mass > 0.0);
	}
    }
  
#ifdef DEBUG
  if(ThisTask == 0)
    fprintf(stderr,"found %ld empty nodes to skip of %ld total nodes.\n",NumSkipIndex,td->Nnodes);
#endif
  
  j = 0;
  for(i=0;i<td->Nnodes;++i)
    {
      if(td->nodes[i].mass > 0.0)
	{
	  td->nodes[j] = td->nodes[i];
	  ++j;
	}
    }
  assert(td->Nnodes - NumSkipIndex == j);
  td->Nnodes = j;
  
  long lind,hind,ind;
  for(i=0;i<td->Nnodes;++i)
    {
      currInd = td->nodes[i].over;
      lind = 0;
      hind = NumSkipIndex - 1;
      ind = (hind-lind)/2 + lind;
      while(hind-lind > 1)
	{
	  if(currInd > skipIndex[ind])
	    lind = ind;
	  else
	    hind = ind;
	  
	  ind = (hind-lind)/2 + lind;
	}
      
      lind = ind - 2;
      if(lind < 0)
	lind = 0;
      
      hind = ind + 2;
      if(hind > NumSkipIndex-1)
	hind = NumSkipIndex-1;
      
      for(j=hind;j>=lind;--j)
        {
          if(currInd > skipIndex[j])
            {
	      td->nodes[i].over = td->nodes[i].over - (j+1);
	      break;
	    }
	}
      
      currInd = td->nodes[i].down;
      lind = 0;
      hind = NumSkipIndex - 1;
      ind = (hind-lind)/2 + lind;
      while(hind-lind >1)
	{
          if(currInd > skipIndex[ind])
            lind = ind;
          else
            hind = ind;

          ind = (hind-lind)/2 + lind;
        }

      lind = ind - 2;
      if(lind <0)
	lind = 0;

      hind = ind + 2;
      if(hind > NumSkipIndex-1)
	hind = NumSkipIndex-1;

      for(j=hind;j>=lind;--j)
	{
          if(currInd > skipIndex[j])
            {
	      td->nodes[i].down = td->nodes[i].down - (j+1);
	      break;
	    }   
        }
    }

  free(skipIndex);
  
  //error checking
  for(i=0;i<td->Nnodes;++i)
    {
      assert(td->nodes[i].over >= -1);
      assert(td->nodes[i].over < td->Nnodes);
      assert(td->nodes[i].down >= -1);
      assert(td->nodes[i].down < td->Nnodes);
    }
  
  //cut down nodes to size
  TreeNode *tempnodes;
  if(td->Nnodes > 0)
    {
      tempnodes = (TreeNode*)realloc(td->nodes,sizeof(TreeNode)*(td->Nnodes));
      if(tempnodes != NULL)
	{
	  td->nodes = tempnodes;
	  td->NnodesAlloc = td->Nnodes;
	}
      else
	{
	  fprintf(stderr,"could not realloc tree nodes after pruning!\n");
	  assert(tempnodes != NULL);
	}
    }
  else
    {
      free(td->nodes);
      td->nodes = NULL;
    }
}
#endif

static void initKernelTables(void)
{
  double sigma;
  long i;
  double a,h,k,p,r,v;
  double coss_2,sins_2,sins_2_sqr;
  double sins_2_4th,coss,sins;
  double sin2s,s2,s3,s4;
  double one_m_coss;
  
  initTables = 0;
  
  if(rayTraceData.minComvSmoothingScale/rayTraceData.maxComvDistance < MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder)))
    minSigmaTable = rayTraceData.minComvSmoothingScale/rayTraceData.maxComvDistance;
  else
    minSigmaTable = MIN_SMOOTH_TO_RAY_RATIO*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder));
  
  dSigmaTable = (M_PI - minSigmaTable)/((double) (NumKernelTable-1));
  
  for(i=0;i<NumKernelTable;++i)
    {
      sigma = i*dSigmaTable + minSigmaTable;
  
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
      
      a = (-1104.0*sigma*coss_2/sins_2 + 1440.0*log(one_m_coss)
	   - (4.0*s2*(96.0 + 4.0*s2 + 2.0*(-48.0 + s2)*coss 
		      + 19.0*sigma*sins))/one_m_coss/one_m_coss
	   )/18095.573684677209053544825887689936612975695740401;   
      //18095.573684677209053544825887689936612975695740401 = 5760.0*M_PI
      
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
      
      kernelTable[i].a = a;
      kernelTable[i].v = v;
      kernelTable[i].r = r;
      kernelTable[i].p = p;
      kernelTable[i].k = k;
      kernelTable[i].h = h;
      
      kernelTable[i].potshift = 0.045933363149576995189821117035939045706814640280321 - (0.15915494309189533576888376337251436203445964574046)*log(sigma);
      //0.15915494309189533576888376337251436203445964574046 = 1/2.0/M_PI
      //0.045933363149576995189821117035939045706814640280321 = Euler's constant/4/pi
      
      kernelTable[i].cosSigma = cos(sigma);
    }
}

/*
  static double sphdist_haversine(double t1, double p1, double t2, double p2)
  {
  double sinDt = sin((t2-t1)/2.0);
  double sinDp = sin((p2-p1)/2.0);
  double h = sinDt*sinDt + sin(t1)*sin(t2)*sinDp*sinDp;
  h = sqrt(h);
  //double d = 2.0*asin(h);
  //assert(d >= 0.0 && d <= M_PI);
  
  return 2.0*asin(h);
  }
*/

static double force_shear_parang_ppshort(double vec[3], double vnorm, double vecp[3], double cosr, 
					 TreeData td, double mass, double fvec[2], double shtens[4])
{
  /* sin and cos of arcdist between points 
     note that cosr = vec[0]*vecp[0] + vec[1]*vecp[1] + vec[2]*vecp[2];
  */
  double sinr = sqrt((1.0 - cosr)*(1.0 + cosr));
  
  double sinp,cosp,norm;

  norm = -vnorm*sinr;
  sinp = (vecp[1]*vec[0] - vecp[0]*vec[1])/norm;
  cosp = (vec[2]*(vecp[0]*vec[0] + vecp[1]*vec[1]) - vecp[2]*vnorm*vnorm)/norm;

  /* another older version       
     double sinp,cosp;
     double df[3],norm;
     
     norm = -vnorm*sinr;
     df[0] = vecp[0] - vec[0]*cosr;
     df[1] = vecp[1] - vec[1]*cosr;
     df[2] = vecp[2] - vec[2]*cosr;
     sinp = (df[1]*vec[0] - df[0]*vec[1])/norm;
     cosp = (vec[2]*(df[0]*vec[0] + df[1]*vec[1]) - df[2]*(vec[0]*vec[0] + vec[1]*vec[1]))/norm;
  */
  
  /* older vector version 
     double ephi[2],etheta[3];
     
     ephi[0] = -vec[1];
     ephi[1] = vec[0];
     etheta[0] = vec[2]*vec[0];
     etheta[1] = vec[2]*vec[1];
     etheta[2] = -1.0*(vec[0]*vec[0] + vec[1]*vec[1]);
     norm = -1.0*sqrt((1.0 - vec[2])*(1.0 + vec[2]))*sinr;
     df[0] = vecp[0] - vec[0]*cosr;
     df[1] = vecp[1] - vec[1]*cosr;
     df[2] = vecp[2] - vec[2]*cosr;
     sinp = (df[0]*ephi[0] + df[1]*ephi[1])/norm;
     cosp = (df[0]*etheta[0] + df[1]*etheta[1] + df[2]*etheta[2])/norm;
  */
  
  /* old paralactic angle comp - replaced with vector version above
     double ph,ppp,t,tpp;
     vec2ang(vec,&t,&ph);
     vec2ang(vecp,&tpp,&ppp);
     double sinpp = sin(ph-ppp);
     double cospp = sin(t)*cos(tpp)/sin(tpp)-cos(t)*cos(ph-ppp);
     double normp = sqrt(cospp*cospp + sinpp*sinpp);
     sinpp /= normp;
     cospp /= normp;
     
     fprintf(stderr,"cosp,sinp = %g|%g (%g|%g)\n",cosp,sinp,cospp,sinpp);
     
     sinp = sinpp;
     cosp = cospp;
  */
    
  //factors needed for comp of f, gammaE, and pot 
  double mass_4pi;
  double mass_4pi_cosrm1;
  double fr,two_gammaE,pot;
    
  double onemw,w;
  long i,ip1;
  
  w = (cosr - td->minCosr)/td->dcosr;
  i = (long) w;
  ip1 = i + 1;
  w -= i;
  onemw = 1.0 - w;
  
  mass_4pi = mass/(12.5663706143591729538505735331180115367886775975);
  mass_4pi_cosrm1 = mass_4pi/(cosr-1.0);
  
  pot = mass_4pi*(td->potTable[i]*onemw + td->potTable[ip1]*w);
  fr = -mass_4pi_cosrm1*sinr;
  two_gammaE = mass_4pi_cosrm1*(cosr + 1.0);
  
  //multiply by the treePM force splitting function to remove long-range part
#ifndef DIRECTSUMMATION
  fr *= (td->expAlphaTable[i]*onemw + td->expAlphaTable[ip1]*w);
  two_gammaE *= (td->expGammaTable[i]*onemw + td->expGammaTable[ip1]*w);
#endif
  
  fvec[0] = fr*cosp;
  fvec[1] = fr*sinp;
  
  shtens[0] = two_gammaE*(cosp*cosp - 0.5);     //2*0+0 = 0
  shtens[1] = two_gammaE*sinp*cosp;             //2*0+1 = 1
  //shtens[2] = shtens[1];                      //2*1+0 = 2 //2*0+1 = 1
  //shtens[3] = -shtens[3];                     //2*1+1 = 3
  
  return pot;
}

static double force_shear_parang_ppshort_sigma(double vec[3], double vnorm, double vecp[3], double cosr, 
					       double sigma, TreeData td,
					       double mass, double fvec[2], double shtens[4])
{
  /* sin and cos of arcdist between points 
     note that cosr = vec[0]*vecp[0] + vec[1]*vecp[1] + vec[2]*vecp[2];
  */
  double sinr = sqrt((1.0 - cosr)*(1.0 + cosr));
  
  double sinp,cosp,norm;
    
  norm = -vnorm*sinr;
  sinp = (vecp[1]*vec[0] - vecp[0]*vec[1])/norm;
  cosp = (vec[2]*(vecp[0]*vec[0] + vecp[1]*vec[1]) - vecp[2]*vnorm*vnorm)/norm;
      
  /* another older version
     double sinp,cosp;
     double df[3],norm;
     
     norm = -vnorm*sinr;
     df[0] = vecp[0] - vec[0]*cosr;
     df[1] = vecp[1] - vec[1]*cosr;
     df[2] = vecp[2] - vec[2]*cosr;
     sinp = (df[1]*vec[0] - df[0]*vec[1])/norm;
     cosp = (vec[2]*(df[0]*vec[0] + df[1]*vec[1]) - df[2]*(vec[0]*vec[0] + vec[1]*vec[1]))/norm;
  */
  
  /* older vector version 
     double ephi[2],etheta[3];
     
     ephi[0] = -vec[1];
     ephi[1] = vec[0];
     etheta[0] = vec[2]*vec[0];
     etheta[1] = vec[2]*vec[1];
     etheta[2] = -1.0*(vec[0]*vec[0] + vec[1]*vec[1]);
     norm = -1.0*sqrt((1.0 - vec[2])*(1.0 + vec[2]))*sinr;
     df[0] = vecp[0] - vec[0]*cosr;
     df[1] = vecp[1] - vec[1]*cosr;
     df[2] = vecp[2] - vec[2]*cosr;
     sinp = (df[0]*ephi[0] + df[1]*ephi[1])/norm;
     cosp = (df[0]*etheta[0] + df[1]*etheta[1] + df[2]*etheta[2])/norm;
  */
  
  /* old paralactic angle comp - replaced with vector version above
     double ph,ppp,t,tpp;
     vec2ang(vec,&t,&ph);
     vec2ang(vecp,&tpp,&ppp);
     double sinpp = sin(ph-ppp);
     double cospp = sin(t)*cos(tpp)/sin(tpp)-cos(t)*cos(ph-ppp);
     double normp = sqrt(cospp*cospp + sinpp*sinpp);
     sinpp /= normp;
     cospp /= normp;
     
     fprintf(stderr,"cosp,sinp = %g|%g (%g|%g)\n",cosp,sinp,cospp,sinpp);
     
     sinp = sinpp;
     cosp = cospp;
  */
    
  //factors needed for comp of f, gammaE, and pot 
  double mass_4pi;
  double mass_4pi_cosrm1;
  double rs2;
  double fr,two_gammaE,pot,potshift;
  double rs;
  
  double onemw,w;
  long i,ip1;
  
  double a,b,d,f,g,h,k,p,r,v;
  double ws,onemws,cosSigma;
  long is,isp1;
  
  w = (cosr - td->minCosr)/td->dcosr;
  i = (long) w;
  ip1 = i + 1;
  w -= i;
  onemw = 1.0 - w;
  
  ws = (sigma - minSigmaTable)/dSigmaTable;
  is = (long) ws;
  isp1 = is + 1;
  ws -= is;
  onemws = 1.0 - ws;
  cosSigma = kernelTable[is].cosSigma*onemws + kernelTable[isp1].cosSigma*ws;
  
  if(cosr >= cosSigma)
    { 
      rs = acos(cosr)/sigma;
      
      //factors needed for comp of f, gammaE, and pot
      a = kernelTable[is].a*onemws + kernelTable[isp1].a*ws;
      v = kernelTable[is].v*onemws + kernelTable[isp1].v*ws;
      r = kernelTable[is].r*onemws + kernelTable[isp1].r*ws;
      p = kernelTable[is].p*onemws + kernelTable[isp1].p*ws;
      k = kernelTable[is].k*onemws + kernelTable[isp1].k*ws;
      h = kernelTable[is].h*onemws + kernelTable[isp1].h*ws;
      g = a - h*0.1;
      
      potshift = kernelTable[is].potshift*onemws + kernelTable[isp1].potshift*ws;
      
      /* old code
	 rs2 = rs*rs;
	 rs3 = rs2*rs;
	 rs4 = rs3*rs;
	 rs5 = rs4*rs;
      */
      
      if(rs < 0.5)
	{
	  f = v - 3.2*h;  //16/5 = 3.2
	  d = 8.0*h+r;
	  //c = -8.0*h+p; this is always zero
	  b = 4.0*h+k;
	  
	  /* old code - using better version below
	     pot = mass*(a + b*rs2 + d*rs4 + f*rs5 + potshift);
	     fr = mass*(2.0*b*rs + 4.0*d*rs3 + 5.0*f*rs4)/sigma;
	     two_gammaE = mass*(2.0*b + 12.0*d*rs2 + 20.0*f*rs3)/sigma/sigma - cosr/sinr*fr; 
	  */
	  
	  rs2 = rs*rs;
	  pot = mass*(a + rs2*(b + rs2*(d + rs*f)) + potshift);
	  fr = mass*rs*(2.0*b + rs2*(4.0*d + rs*5.0*f))/sigma;
	  two_gammaE = mass*(2.0*b + rs2*(12.0*d + rs*20.0*f))/sigma/sigma - cosr/sinr*fr; 
	}
      else 
	{
	  /* old code - using better version below
	     pot = mass*(g + h*rs + k*rs2 + p*rs3 + r*rs4 + v*rs5 + potshift);
	     fr = mass*(h + 2.0*k*rs + 3.0*p*rs2 + 4.0*r*rs3 + 5.0*v*rs4)/sigma;
	     two_gammaE = mass*(2.0*k + 6.0*p*rs + 12.0*r*rs2 + 20.0*v*rs3)/sigma/sigma - cosr/sinr*fr;
	  */
	  
	  pot = mass*(g + rs*(h + rs*(k + rs*(p + rs*(r + v*rs)))) + potshift);
	  fr = mass*(h + rs*(2.0*k + rs*(3.0*p + rs*(4.0*r + 5.0*v*rs))))/sigma;
	  two_gammaE = mass*(2.0*k + rs*(6.0*p + rs*(12.0*r + 20.0*v*rs)))/sigma/sigma - cosr/sinr*fr;
	}
    }
  else
    {
      mass_4pi = mass/(12.5663706143591729538505735331180115367886775975);
      mass_4pi_cosrm1 = mass_4pi/(cosr-1.0);
      
      pot = mass_4pi*(td->potTable[i]*onemw + td->potTable[ip1]*w);
      fr = -mass_4pi_cosrm1*sinr;
      two_gammaE = mass_4pi_cosrm1*(cosr + 1.0);
    }
  
  //multiply by the treePM force splitting function to remove long-range part
  //FIXME #ifndef DIRECTSUMMATION
  //fr *= (td->expAlphaTable[i]*onemw + td->expAlphaTable[ip1]*w);
  //two_gammaE *= (td->expGammaTable[i]*onemw + td->expGammaTable[ip1]*w);
  //#endif
  
  /*double expf;
  expf = exp((cosr-1.0)/td->thetaSplit2);
  fr *= expf;
  two_gammaE *= expf*(1.0 + sinr*sinr/td->thetaSplit2/(1.0 + cosr));
  */
  
  fr *= gsl_spline_eval(spline_alpha,cosr,acc_alpha);
  two_gammaE *= gsl_spline_eval(spline_gamma,cosr,acc_gamma);
  
  fvec[0] = fr*cosp;
  fvec[1] = fr*sinp;
  
  shtens[0] = two_gammaE*(cosp*cosp - 0.5);     //2*0+0 = 0
  shtens[1] = two_gammaE*sinp*cosp;             //2*0+1 = 1
  //shtens[2] = shtens[1];                      //2*1+0 = 2 //2*0+1 = 1
  //shtens[3] = -shtens[3];                     //2*1+1 = 3
  
  /*
  static long iiii = 1;
  if(iiii == 1)
    {
      /*
      fprintf(stderr,"cosr = %f, i,ip1 = %d|%d, w,1-w = %f|%f, a,g = %g|%g (true %g|%g)\n",
	      cosr,i,ip1,w,onemw,
	      td->expAlphaTable[i]*onemw + td->expAlphaTable[ip1]*w,
	      td->expGammaTable[i]*onemw + td->expGammaTable[ip1]*w,
	      exp((cosr-1.0)/td->thetaSplit2),
	      exp((cosr-1.0)/td->thetaSplit2)*(1.0 + sinr*sinr/td->thetaSplit2/(1.0 + cosr)));
      */
  /*
      cosr = td->dcosr*i + td->minCosr + td->dcosr/2.0;
      if(1.0 - cosr*cosr >= 0.0)
	sinr = sqrt(1.0 - cosr*cosr);
      else
	sinr = 0.0;
      w = (cosr - td->minCosr)/td->dcosr;
      i = (long) w;
      ip1 = i + 1;
      w -= i;
      onemw = 1.0 - w;
    */  
      /*fprintf(stderr,"tt  cosr = %f, i,ip1 = %d|%d, w,1-w = %f|%f, a,g = %g|%g (true %g|%g), i a,g = %g|%g, ip1 a,g = %g|%g\n",
	      cosr,i,ip1,w,onemw,
	      td->expAlphaTable[i]*onemw + td->expAlphaTable[ip1]*w,
	      td->expGammaTable[i]*onemw + td->expGammaTable[ip1]*w,
	      exp((cosr-1.0)/td->thetaSplit2),
	      exp((cosr-1.0)/td->thetaSplit2)*(1.0 + sinr*sinr/td->thetaSplit2/(1.0 + cosr)),
	      td->expAlphaTable[i],td->expGammaTable[i],
	      td->expAlphaTable[ip1],td->expGammaTable[ip1]
	      );
      */
      /*
      fprintf(stderr,"tts cosr = %f, i,ip1 = %d|%d, w,1-w = %f|%f, a,g = %g|%g (true %g|%g), i a,g = %e|%e, ip1 a,g = %e|%e\n",
	      cosr,i,ip1,w,onemw,
	      (gsl_spline_eval(spline_alpha,cosr,acc_alpha)),
	      (gsl_spline_eval(spline_gamma,cosr,acc_gamma)),
	      exp((cosr-1.0)/td->thetaSplit2),
	      exp((cosr-1.0)/td->thetaSplit2)*(1.0 + sinr*sinr/td->thetaSplit2/(1.0 + cosr)),
	      (td->expAlphaTable[i]),(td->expGammaTable[i]),
	      (td->expAlphaTable[ip1]),(td->expGammaTable[ip1])
	      );
      
      */
  //  iiii = 0;
  //}
  
  return pot;
}

TreeWalkData computePotentialForceShearDirectSummation(double vec[3], TreeData td)
{
  //fprintf(stderr,"force direct summation!\n");
  
  long i;
  TreeWalkData twd;
  double alpha[2],U[4],vecp[3];
  double cosarcDistCOM;
  double vnorm = sqrt((1.0 - vec[2])*(1.0 + vec[2]));
  twd.pot = 0.0;
  twd.alpha[0] = 0.0;
  twd.alpha[1] = 0.0;
  twd.U[0] = 0.0;
  twd.U[1] = 0.0;
  //twd.U[2] = 0.0;
  //twd.U[3] = 0.0;
  twd.NumInteractTreeWalk = 0;
  twd.NumInteractTreeWalkNode = 0;
  twd.Nempty = 0;
  double maxR = cos(MAX_RADTREEWALK_TO_SPLIT_RATIO*td->thetaSplit);
  
  for(i=0;i<td->Nparts;++i)
    {
      vecp[0] = td->parts[i].pos[0];
      vecp[1] = td->parts[i].pos[1];
      vecp[2] = td->parts[i].pos[2];
      vecp[0] /= td->parts[i].r;
      vecp[1] /= td->parts[i].r;
      vecp[2] /= td->parts[i].r;
      cosarcDistCOM = vec[0]*vecp[0] + vec[1]*vecp[1] + vec[2]*vecp[2];
      
      if(cosarcDistCOM >= maxR)
	twd.pot += force_shear_parang_ppshort_sigma(vec,vnorm,vecp,cosarcDistCOM,
						    td->parts[i].smoothingLength,td,
						    td->parts[i].mass,alpha,U);
      twd.alpha[0] += alpha[0];
      twd.alpha[1] += alpha[1];
      twd.U[0] += U[0];
      twd.U[1] += U[1];
      //twd.U[2] += U[2];
      //twd.U[3] += U[3];
      
      ++(twd.NumInteractTreeWalk);
    }
  
  twd.U[2] = twd.U[1];
  twd.U[3] = -1.0*twd.U[0];
  
  return twd;
}

