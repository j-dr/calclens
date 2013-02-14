#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>

#include "raytrace.h"
#include "treecode.h"

/*
static int compPartNest(const void *a, const void *b)
{
  if(((const Part*)a)->nest > ((const Part*)b)->nest)
    return 1;
  else if(((const Part*)a)->nest < ((const Part*)b)->nest)
    return -1;
  else
    return 0;
}
*/

void do_tree_poisson_solve(double densfact)
{
  long i,j;
  TreeData td;
  TreeWalkData twd;
  double r,vec[3];
  double obsSLVal[3],thetaS;
#ifdef HEALPIX_NGP_PARTINTERP
  double gridFact2 = 0.0;
#else
  double gridFact2 = 4.0*M_PI/order2npix(rayTraceData.poissonOrder)*HEALPIX_GRID_SMOOTH_FACT*HEALPIX_GRID_SMOOTH_FACT;
#endif
  long Nwalk = 0,Nf = 0,Nn = 0,Ne = 0;
  double timeT,timeB;
    
  logProfileTag(PROFILETAG_TREEBUILD);
  timeB = -MPI_Wtime();
  /*for(i=0;i<NlensPlaneParts;++i)
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
      
      lensPlaneParts[i].nest = nest2peano(lensPlaneParts[i].nest,HEALPIX_UTILS_MAXORDER);
    }
  
  for(i=0;i<NbundleCells;++i)
    {
      if(bundleCells[i].Nparts > 0)
	qsort(&(lensPlaneParts[bundleCells[i].firstPart]),(size_t) (bundleCells[i].Nparts),sizeof(Part),compPartNest);
    }
  
  for(i=0;i<NlensPlaneParts;++i)
    lensPlaneParts[i].nest = peano2nest(lensPlaneParts[i].nest,HEALPIX_UTILS_MAXORDER);
  */
  
  thetaS = sqrt(rayTraceData.SHTSplitScale*rayTraceData.SHTSplitScale + gridFact2);
  td = buildTree(lensPlaneParts,NlensPlaneParts,thetaS,0l);
  
  /*for(i=0;i<NlensPlaneParts;++i)
    lensPlaneParts[i].smoothingLength = lensPlaneParts[i].smoothingLength*rayTraceData.NumNbrsSmooth;
  */
  
#ifdef DEBUG_IO
  FILE *fp;
  char name[MAX_FILENAME];
  sprintf(name,"%s/smoothlengths%04ld.%04d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  fp = fopen(name,"w");
  for(i=0;i<NlensPlaneParts;++i)
    fprintf(fp,"%.20e\n",lensPlaneParts[i].smoothingLength*rayTraceData.planeRad);
  fclose(fp);
#endif
  
  //enforce these mins and maxes
  /*obsSLVal[0] = 0.0;
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
      
      if(lensPlaneParts[i].smoothingLength >= M_PI)
	lensPlaneParts[i].cosSmoothingLength = -1.0;
      else
	lensPlaneParts[i].cosSmoothingLength = cos(lensPlaneParts[i].smoothingLength);
	}
  */
  for(i=0;i<td->Nnodes;++i)
    {
      if(td->nodes[i].mass > 0)
        {
	  //td->nodes[i].cosMaxSL = td->nodes[i].cosMaxSL*rayTraceData.NumNbrsSmooth;
	  if(td->nodes[i].cosMaxSL > rayTraceData.maxSL)
	    td->nodes[i].cosMaxSL = rayTraceData.maxSL;
	  if(td->nodes[i].cosMaxSL < rayTraceData.minSL)
	    td->nodes[i].cosMaxSL = rayTraceData.minSL;
	  
	  td->nodes[i].alwaysOpen = 0;
	  
	  if(td->nodes[i].cosMaxSL > td->nodeArcSizes[td->nodes[i].order]/MAX_SMOOTH_TO_TREENODE_FAC && td->nodes[i].down != -1)
	    td->nodes[i].alwaysOpen = 1;
	  
	  td->nodes[i].cosMaxSL = cos(td->nodes[i].cosMaxSL);
        }
    }
  timeB += MPI_Wtime();
  logProfileTag(PROFILETAG_TREEBUILD);
  
  if(ThisTask == 0)
    fprintf(stderr,"tree built in %lg seconds.\n",timeB);
  
  /*if(ThisTask == 0)
    fprintf(stderr,"geom. mean,min,max obs. smoothing len. = %lg|%lg|%lg [Mpc/h] (min,max smoothing len. %lg|%lg [Mpc/h], NNbrs = %lg)\n",
	    exp(obsSLVal[0]/NlensPlaneParts)*rayTraceData.planeRad,obsSLVal[1]*rayTraceData.planeRad,obsSLVal[2]*rayTraceData.planeRad,
	    rayTraceData.minSL*rayTraceData.planeRad,rayTraceData.maxSL*rayTraceData.planeRad,rayTraceData.NumNbrsSmooth);
  */
  
#ifdef CHECKTREEWALK
  int firstRay = 1;
#endif
  
  logProfileTag(PROFILETAG_TREEWALK);
  timeT = -MPI_Wtime();
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	{
	  bundleCells[i].cpuTime -= MPI_Wtime();
	  
	  Nwalk += bundleCells[i].Nrays;
	  
	  for(j=0;j<bundleCells[i].Nrays;++j)
	    {
	      vec[0] = bundleCells[i].rays[j].n[0]/rayTraceData.planeRad;
	      vec[1] = bundleCells[i].rays[j].n[1]/rayTraceData.planeRad;
	      vec[2] = bundleCells[i].rays[j].n[2]/rayTraceData.planeRad;
	      	      
#ifdef DIRECTSUMMATION
	      twd = computePotentialForceShearDirectSummation(vec,td);
#else
#ifdef CHECKTREEWALK
              if(firstRay && ThisTask == 0)
                {
                  twd = computePotentialForceShearTree(vec,-1.0*rayTraceData.BHCrit,td);
                  firstRay = 0;
                }
              else
                twd = computePotentialForceShearTree(vec,rayTraceData.BHCrit,td);
#else
              twd = computePotentialForceShearTree(vec,rayTraceData.BHCrit,td);
#endif
#endif
	      bundleCells[i].rays[j].phi += twd.pot*densfact;
	      
	      bundleCells[i].rays[j].alpha[0] -= twd.alpha[0]*densfact;
	      bundleCells[i].rays[j].alpha[1] -= twd.alpha[1]*densfact;
	      
	      bundleCells[i].rays[j].U[0] += twd.U[0]*densfact;
	      bundleCells[i].rays[j].U[1] += twd.U[1]*densfact;
	      bundleCells[i].rays[j].U[2] += twd.U[2]*densfact;
	      bundleCells[i].rays[j].U[3] += twd.U[3]*densfact;
	      
	      Nf += twd.NumInteractTreeWalk;
	      Nn += twd.NumInteractTreeWalkNode;
	      Ne += twd.Nempty;
	    }
	  
	  bundleCells[i].cpuTime += MPI_Wtime();
	}
    }
  
  timeT += MPI_Wtime();
  logProfileTag(PROFILETAG_TREEWALK);
  
#ifdef DEBUG
#if DEBUG_LEVEL > 0
  fprintf(stderr,"%05d: %ld parts, %lg seconds, %ld rays, part,node interactions = %ld|%ld, empty nodes = %ld\n",
	  ThisTask,td->Nparts,timeT,Nwalk,Nf,Nn,Ne);
#endif
#endif
  
  if(ThisTask == 0)
    fprintf(stderr,"tree walk done in %lg seconds (%lg rays per second, part,node interactions = %ld|%ld).\n",
	    timeT,((double) Nwalk)/timeT,Nf,Nn);
  
  /*
  double minTime,maxTime,totTime,avgTime;
  
  MPI_Reduce(&timeT,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&timeT,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&timeT,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgTime = totTime/((double) NTasks);
  
  if(ThisTask == 0)
    fprintf(stderr,"tree walk done in %lg seconds. (max,min,avg = %lf|%lf|%lf - %.2f percent, %lg rays per second, part,node interactions = %ld|%ld)\n",
	    timeT,maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,((double) Nwalk)/timeT,Nf,Nn);
    
    MPI_Reduce(&timeB,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&timeB,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&timeB,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    avgTime = totTime/((double) NTasks);
    
    if(ThisTask == 0)
    fprintf(stderr,"tree build time max,min,avg = %lg|%lg|%lg - %.2f percent (total time %lg)\n"
    ,maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,totTime);
    
    MPI_Reduce(&timeT,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&timeT,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&timeT,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    avgTime = totTime/((double) NTasks);
    
    if(ThisTask == 0)
    fprintf(stderr,"tree walk time max,min,avg = %lf|%lf|%lf - %.2f percent (total time %lf)\n"
    ,maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,totTime);
    
    timeT = timeT/((double) Nwalk);
    MPI_Reduce(&timeT,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&timeT,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&timeT,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    avgTime = totTime/((double) NTasks);
    
    if(ThisTask == 0)
    fprintf(stderr,"tree walk time per ray max,min,avg = %lg|%lg|%lg - %.2f percent (total time %lg)\n"
    ,maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,totTime);
    
    long tot[3];
    MPI_Reduce(&Nf,&(tot[0]),1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&Nn,&(tot[1]),1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&Ne,&(tot[2]),1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    
    if(ThisTask == 0)
    fprintf(stderr,"tree walk Nforce = %ld, Nnode = %ld, Nempty = %ld\n"
    ,tot[0],tot[1],tot[2]);
  */
  
  destroyTree(td);
}
  

