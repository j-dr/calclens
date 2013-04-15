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

void do_tree_poisson_solve(double densfact)
{
  long i,j;
  TreeData td;
  TreeWalkData twd;
  double r,vec[3];
  double obsSLVal[3],thetaS;
  double gridFact2 = 4.0*M_PI/order2npix(rayTraceData.poissonOrder)*HEALPIX_GRID_SMOOTH_FACT*HEALPIX_GRID_SMOOTH_FACT;
  long Nwalk = 0,Nf = 0,Nn = 0,Ne = 0;
  double timeT,timeB;
    
  logProfileTag(PROFILETAG_TREEBUILD);
  timeB = -MPI_Wtime();
    
  thetaS = sqrt(rayTraceData.TreePMSplitScale*rayTraceData.TreePMSplitScale + gridFact2);
  td = buildTree(lensPlaneParts,NlensPlaneParts,thetaS);
  
  /*
  for(i=0;i<td->Nnodes;++i)
    {
      if(td->nodes[i].mass > 0)
        {
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
  */
  
  timeB += MPI_Wtime();
  logProfileTag(PROFILETAG_TREEBUILD);
  
  if(ThisTask == 0)
    {
      fprintf(stderr,"tree built in %lg seconds.\n",timeB);
      fflush(stderr);
    }
  
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
                  twd = computePotentialForceShearTree(vec,-1.0*rayTraceData.BHCrit*rayTraceData.BHCrit,td);
                  firstRay = 0;
                }
              else
                twd = computePotentialForceShearTree(vec,rayTraceData.BHCrit*rayTraceData.BHCrit,td);
#else
              twd = computePotentialForceShearTree(vec,rayTraceData.BHCrit*rayTraceData.BHCrit,td);
#endif
#endif
	      bundleCells[i].rays[j].phi += twd.pot*densfact;
	      
	      bundleCells[i].rays[j].alpha[0] -= twd.alpha[0]*densfact;
	      bundleCells[i].rays[j].alpha[1] -= twd.alpha[1]*densfact;
	      
	      bundleCells[i].rays[j].U[0] += twd.U[0]*densfact;
	      bundleCells[i].rays[j].U[1] += twd.U[1]*densfact;
	      bundleCells[i].rays[j].U[2] += twd.U[2]*densfact;
	      bundleCells[i].rays[j].U[3] += twd.U[3]*densfact;
	      
#ifdef GET_TREE_STATS	      
	      Nf += twd.NumInteractTreeWalk;
	      Nn += twd.NumInteractTreeWalkNode;
	      Ne += twd.Nempty;
#endif
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
  fflush(stderr);
#endif
#endif
  
#ifdef GET_TREE_STATS
  if(ThisTask == 0)
    {
      fprintf(stderr,"tree walk done in %lg seconds (%lg rays per second, part,node interactions = %ld|%ld).\n",
	      timeT,((double) Nwalk)/timeT,Nf,Nn);
      fflush(stderr);
    }
#else
  if(ThisTask == 0)
    {
      fprintf(stderr,"tree walk done in %lg seconds (%lg rays per second).\n",timeT,((double) Nwalk)/timeT);
      fflush(stderr);
    }
#endif
  
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
  

