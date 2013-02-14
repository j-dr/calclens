#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <string.h>

#include "raytrace.h"

//define to never change domain decomp
#define STATIC_DOMAINDECOMP

//define this to always use equal area domain decomp
#define EQUALAREA_DOMAINDECOMP

//take care of cases where equal area always wins
#ifdef SHTONLY
#define EQUALAREA_DOMAINDECOMP
#endif

#if (defined POINTMASSTEST || defined NFWHALOTEST)
#define EQUALAREA_DOMAINDECOMP
#endif

static void loadBalanceBundleCellsPerCPU(void);
static void divide_tasks_domaindecomp(int firstTask, int lastTask, long firstPCell, long lastPCell, double *totCPUPerBundleCell);
static int mightNeedToSendBuffCellsRPI(long sendTask, long recvTask, long *minRPITasks, long *maxRPITasks, long *firstRPITasks, long *lastRPITasks);

void load_balance_tasks(void)
{
  double time;
  long i;
  
  //////////////////////////////
  //do the load balancing     //
  //////////////////////////////
#ifndef STATIC_DOMAINDECOMP
  loadBalanceBundleCellsPerCPU();
#endif
  
  //////////////////////////////                                                                                                                                                     
  //now set part buffer cells //                                                                                                                                                              
  //////////////////////////////                                                                                                                                                               
  time = -MPI_Wtime();
  //NOT NEEDED ANY MORE (ray buffer cell flag is set by buffer ray routine) - mark_bundlecells(rayTraceData.galImageSearchRayBufferRad,PRIMARY_BUNDLECELL,RAYBUFF_BUNDLECELL);
  mark_bundlecells(rayTraceData.partBuffRad,PRIMARY_BUNDLECELL,PARTBUFF_BUNDLECELL);
  time += MPI_Wtime();

  //if(ThisTask == 0)
  //fprintf(stderr,"marking part buffer regions took %lg seconds.\n",time);

  //////////////////////////////                                                                                                                                                               
  //reset the cpu times       //  
  //////////////////////////////                                                                                                                                                               
  for(i=0;i<NbundleCells;++i)
    bundleCells[i].cpuTime = 0.0;
}

void getDomainDecompPerCPU(int report)
{
  long i,j,k;    
  double *totCPUPerBundleCell,tot;
  double *cpuPerBundleCell;
  long setRestByHand;
  long maxNumCellsPerTask = ((long) ((1.0 + rayTraceData.maxRayMemImbalance)*((double) NrestrictedPeanoInd)/((double) NTasks)));
  double memPerBundleCell = 1.0/((double) NrestrictedPeanoInd);
  long NumBundleCellsPerTask = NrestrictedPeanoInd/NTasks;
  
  /*
    get the actual domain decomp
     - first divide up tasks by cost 
     - then check domain decomp for correctness
      - if not correct, revert to equal area domain decomp
  */
  
  //if(ThisTask == 0)
  //fprintf(stderr,"%d: maxNumCellsPerTask = %ld, memfac = %lf, num per task = %lf\n",ThisTask,maxNumCellsPerTask,
  //    1.0 + rayTraceData.maxRayMemImbalance,((double) NrestrictedPeanoInd)/((double) NTasks));
  
  // do all reduce for CPU load balance 
  totCPUPerBundleCell = (double*)malloc(sizeof(double)*NbundleCells);
  assert(totCPUPerBundleCell != NULL);
  cpuPerBundleCell = (double*)malloc(sizeof(double)*NbundleCells);
  assert(cpuPerBundleCell != NULL);
  for(i=0;i<NbundleCells;++i)
    cpuPerBundleCell[i] = 0.0;
  for(i=0;i<NbundleCells;++i)
    cpuPerBundleCell[i] = bundleCells[i].cpuTime;
  MPI_Allreduce(cpuPerBundleCell,totCPUPerBundleCell,(int) NbundleCells,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    
  //get first and last peano ind for each task
  for(i=0;i<NbundleCells;++i)
    if(bundleCellsNest2RestrictedPeanoInd[i] == -1)
      totCPUPerBundleCell[i] = 0.0;
  tot = 0.0;
  for(i=0;i<NbundleCells;++i)
    tot += totCPUPerBundleCell[i];
  for(i=0;i<NbundleCells;++i)
    totCPUPerBundleCell[i] /= tot;
  
  //does a recursive binary split of tasks along cost curve
  divide_tasks_domaindecomp(0,NTasks-1,0l,NrestrictedPeanoInd-1,totCPUPerBundleCell);
  
  //error check
  setRestByHand = 0;
  for(i=0;i<NTasks;++i)
    {
      if(!(firstRestrictedPeanoIndTasks[i] >= 0 && firstRestrictedPeanoIndTasks[i] < NrestrictedPeanoInd))
	setRestByHand = 1;
        
      if(!(lastRestrictedPeanoIndTasks[i] >= 0 && lastRestrictedPeanoIndTasks[i] < NrestrictedPeanoInd))
	setRestByHand = 1;
      
      if(!(lastRestrictedPeanoIndTasks[i] >= firstRestrictedPeanoIndTasks[i]))
	setRestByHand = 1;
      
      if(i < NTasks-1)
	{
	  if(!(lastRestrictedPeanoIndTasks[i] + 1 == firstRestrictedPeanoIndTasks[i+1]))
	    setRestByHand = 1;
	}
      else
	{
	  if(!(lastRestrictedPeanoIndTasks[i] == NrestrictedPeanoInd-1))
	    setRestByHand = 1;
	}
      
      if(lastRestrictedPeanoIndTasks[i] - firstRestrictedPeanoIndTasks[i] + 1 > maxNumCellsPerTask)
	setRestByHand = 1;
      
      if(setRestByHand)
	break;
    }
  
#ifdef EQUALAREA_DOMAINDECOMP
  setRestByHand = 1;
#endif
  if(setRestByHand)
    {
      if(ThisTask == 0)
	{
	  /*fprintf(stderr,"doman decomp for ray load balance failed! reverting to equal area domain decomp!\n");
	    
	    fprintf(stderr,"NrestrictedPeanoInd (total # of active cells) = %ld\n",NrestrictedPeanoInd);
	    for(i=0;i<NTasks;++i)
	    fprintf(stderr,"%ld: firstRestrictedPeanoInd|lastRestrictedPeanoInd = %ld|%ld|%ld|%lf|%ld\n",i,
	    firstRestrictedPeanoIndTasks[i],lastRestrictedPeanoIndTasks[i],
	    lastRestrictedPeanoIndTasks[i] - firstRestrictedPeanoIndTasks[i] + 1,
	    ((double) (lastRestrictedPeanoIndTasks[i] - firstRestrictedPeanoIndTasks[i] + 1))/((double) NrestrictedPeanoInd),
	    ((long) ((1.0 + rayTraceData.maxRayMemImbalance)*NrestrictedPeanoInd/NTasks)));
	  */
	}
      
      j = NrestrictedPeanoInd - NTasks*NumBundleCellsPerTask;
      k = 0;
      for(i=0;i<NTasks;++i)
	{
	  firstRestrictedPeanoIndTasks[i] = k;
	  if(i < j)
	    lastRestrictedPeanoIndTasks[i] = k + NumBundleCellsPerTask;
	  else
	    lastRestrictedPeanoIndTasks[i] = k + NumBundleCellsPerTask - 1;
	  k = lastRestrictedPeanoIndTasks[i] + 1;
	}
      lastRestrictedPeanoIndTasks[NTasks-1] = NrestrictedPeanoInd - 1;
    }
  
  /* mark domain of each node */
  for(i=0;i<NbundleCells;++i)
    CLEARBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL);
  for(i=firstRestrictedPeanoIndTasks[ThisTask];i<=lastRestrictedPeanoIndTasks[ThisTask];++i)
    if(bundleCellsRestrictedPeanoInd2Nest[i] != -1)
      SETBITFLAG(bundleCells[bundleCellsRestrictedPeanoInd2Nest[i]].active,PRIMARY_BUNDLECELL);

  //print some stats
  double tmem,maxMem = -1.0;
  double tcpu,maxCPU = -1.0;
  for(i=0;i<NTasks;++i)
    {
      tmem = 0.0;
      tcpu = 0.0;
      for(j=firstRestrictedPeanoIndTasks[i];j<=lastRestrictedPeanoIndTasks[i];++j)
	{
	  tmem += memPerBundleCell;
	  tcpu += totCPUPerBundleCell[bundleCellsRestrictedPeanoInd2Nest[j]];
	}
      
      if(tmem > maxMem)
	maxMem = tmem;
      
      if(tcpu > maxCPU)
	maxCPU = tcpu;
    }
  
  if(ThisTask == 0 && report)
    fprintf(stderr,"max mem,cpu for domain decomp = %lf|%lf of 1.0 (%lf per task).\n",maxMem,maxCPU,1.0/NTasks);
  
  free(totCPUPerBundleCell);
  free(cpuPerBundleCell);
    
#ifdef DEBUG
#if DEBUG_LEVEL > 0
  if(ThisTask == 0)
    {
      fprintf(stderr,"NrestrictedPeanoInd (total # of active cells) = %ld\n",NrestrictedPeanoInd);
      for(i=0;i<NTasks;++i)
        fprintf(stderr,"%ld: firstRestrictedPeanoInd|lastRestrictedPeanoInd = %ld|%ld|%ld|%lf|%ld\n",i,
                firstRestrictedPeanoIndTasks[i],lastRestrictedPeanoIndTasks[i],
                lastRestrictedPeanoIndTasks[i] - firstRestrictedPeanoIndTasks[i] + 1,
                ((double) (lastRestrictedPeanoIndTasks[i] - firstRestrictedPeanoIndTasks[i] + 1))/((double) NrestrictedPeanoInd),
		((long) ((1.0 + rayTraceData.maxRayMemImbalance)*NrestrictedPeanoInd/NTasks)));
    }
#endif
#endif
}

void loadBalanceBundleCellsPerCPU(void)
{
  long log2NTasks;
  long level,sendTask,recvTask;
  
  long i;
  long *firstRPITasks,*lastRPITasks;
  long *bundleCellHasRays;
  
  long NumSendThisTask,GlobalNumSend;
  long NumRecvThisTask,GlobalNumRecv;
  long maxNumCellsToSendRecv;
  long Nsend,Nrecv;
  long didSend,didRecv;
  MPI_Request requestSend,requestRecv;
  MPI_Status Stat;
  long bundleNestToRecv;
  long NumRaysPerBundleCell,shift,round;
  long bnestOfRaysToMove,rayStartOfRaysToMove;
  HEALPixRay *raysToMove;
  long *maxNumCellsToSend;
  
  if(ThisTask == 0)
    fprintf(stderr,"load balancing nodes.\n");
  
  //set up for loop
  log2NTasks = 0;
  while(NTasks > (1 << log2NTasks))
    ++log2NTasks;
  
  shift = rayTraceData.rayOrder - rayTraceData.bundleOrder;
  shift = 2*shift;
  NumRaysPerBundleCell = 1;
  NumRaysPerBundleCell = (NumRaysPerBundleCell << shift);
  
  maxNumCellsToSend = (long*)malloc(sizeof(long)*NTasks);
  assert(maxNumCellsToSend != NULL);
  
  //record information about old bundle cells and rays
  firstRPITasks = (long*)malloc(sizeof(long)*NTasks);
  assert(firstRPITasks != NULL);
  lastRPITasks = (long*)malloc(sizeof(long)*NTasks);
  assert(lastRPITasks != NULL);
  for(i=0;i<NTasks;++i)
    {
      firstRPITasks[i] = firstRestrictedPeanoIndTasks[i];
      lastRPITasks[i] = lastRestrictedPeanoIndTasks[i];
    }
  
  bundleCellHasRays = (long*)malloc(sizeof(long)*NbundleCells);
  assert(bundleCellHasRays != NULL);
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	bundleCellHasRays[i] = 1;
      else
        bundleCellHasRays[i] = 0;
    }
  
  //get actual domain decomp
  getDomainDecompPerCPU(1);
  
  //get number of cells each task needs to send and recv total
  NumSendThisTask = 0;
  NumRecvThisTask = 0;
  for(i=0;i<NbundleCells;++i)
    {
      if(bundleCellHasRays[i] && !(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL)))
        ++NumSendThisTask;
      
      if(!(bundleCellHasRays[i]) && ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
        ++NumRecvThisTask;
    }
  MPI_Allreduce(&NumSendThisTask,&GlobalNumSend,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD); 
  MPI_Allreduce(&NumRecvThisTask,&GlobalNumRecv,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD); 
    
  round = 0;
  while(GlobalNumSend > 0 || GlobalNumRecv > 0)
    {
      if(ThisTask == 0)
        fprintf(stderr,"round %ld: # of bundle cells left to send,recv = %ld|%ld\n",round,GlobalNumSend,GlobalNumRecv);
      ++round;
      
      /*algorithm to loop through pairs of tasks linearly
        -lifted from Gadget-2 under GPL (http://www.gnu.org/copyleft/gpl.html)
        -see pm_periodic.c from Gadget-2 at http://www.mpa-garching.mpg.de/gadget/
      */
      for(level = 0; level < (1 << log2NTasks); level++) /* note: for level=0, target is the same task */
        {
	  maxNumCellsToSendRecv = (MaxNumAllRaysGlobal - NumAllRaysGlobal)/NumRaysPerBundleCell;
	  MPI_Allgather(&maxNumCellsToSendRecv,1,MPI_LONG,maxNumCellsToSend,1,MPI_LONG,MPI_COMM_WORLD);
	  
	  sendTask = ThisTask;
          recvTask = ThisTask ^ level;
          if(recvTask < NTasks && sendTask != recvTask)
            {
              if(mightNeedToSendBuffCellsRPI(sendTask,recvTask,firstRestrictedPeanoIndTasks,lastRestrictedPeanoIndTasks,firstRPITasks,lastRPITasks) ||
                 mightNeedToSendBuffCellsRPI(recvTask,sendTask,firstRestrictedPeanoIndTasks,lastRestrictedPeanoIndTasks,firstRPITasks,lastRPITasks))
                {
                  Nsend = 0;
                  for(i=firstRPITasks[sendTask];i<=lastRPITasks[sendTask];++i)
                    {
                      if(bundleCellHasRays[bundleCellsRestrictedPeanoInd2Nest[i]] && 
                         (firstRestrictedPeanoIndTasks[recvTask] <= i && i <= lastRestrictedPeanoIndTasks[recvTask]))
                        ++Nsend;
                    }
                  
                  if(Nsend > maxNumCellsToSend[recvTask])
                    Nsend = maxNumCellsToSend[recvTask];
                  
                  MPI_Sendrecv(&Nsend,1,MPI_LONG,(int) recvTask,TAG_NUMBUFF_LOADBAL,
                               &Nrecv,1,MPI_LONG,(int) recvTask,TAG_NUMBUFF_LOADBAL,
                               MPI_COMM_WORLD,&Stat);
                  
                  i = firstRPITasks[sendTask];
                  while(Nsend > 0 || Nrecv > 0)
                    {
                      //first get bundle index to recv
                      //then send cells
                      
                      //get bundle ind to recv
                      if(Nrecv > 0)
                        {
                          MPI_Irecv(&bundleNestToRecv,1,MPI_LONG,(int) recvTask,TAG_BUFFIND_LOADBAL,MPI_COMM_WORLD,&requestRecv);
                          didRecv = 1;
                        }
                      else
                        didRecv = 0;
                      
                      if(Nsend > 0)
                        {
                          while(i <= lastRPITasks[sendTask])
                            {
                              if(firstRestrictedPeanoIndTasks[recvTask] <= i && i <= lastRestrictedPeanoIndTasks[recvTask] &&
                                 bundleCellHasRays[bundleCellsRestrictedPeanoInd2Nest[i]])
                                break;
                              
                              ++i;
                            }
                          
                          if(i > lastRPITasks[sendTask])
                            {
                              fprintf(stderr,"%d: out of restricted peano ind bundle cells in while Nsend, Nrecv loop\n",ThisTask);
                              MPI_Abort(MPI_COMM_WORLD,456);
                            }
                          
                          MPI_Issend(&(bundleCellsRestrictedPeanoInd2Nest[i]),1,MPI_LONG,(int) recvTask,TAG_BUFFIND_LOADBAL,MPI_COMM_WORLD,&requestSend);
                          didSend = 1;
                        }
                      else
                        didSend = 0;
                      
                      if(didRecv)
                        MPI_Wait(&requestRecv,&Stat);
                      
                      if(didSend)
                        MPI_Wait(&requestSend,&Stat);
                      
		      //now get rays for the bundle cell
                      if(Nrecv > 0)
                        {
			  if(NumAllRaysGlobal >= MaxNumAllRaysGlobal)
			    {
			      fprintf(stderr,"%d: out of memory for rays during load balance! MaxNumAllRaysGlobal = %ld, NumAllRaysGlobal = %ld\n",
				      ThisTask,MaxNumAllRaysGlobal,NumAllRaysGlobal);
			      MPI_Abort(MPI_COMM_WORLD,112);
			    }
			  
                          bundleCells[bundleNestToRecv].rays = AllRaysGlobal + NumAllRaysGlobal;
			  bundleCells[bundleNestToRecv].Nrays = NumRaysPerBundleCell;
			  NumAllRaysGlobal += NumRaysPerBundleCell;
                          
                          MPI_Irecv(bundleCells[bundleNestToRecv].rays,
                                    (int) (sizeof(HEALPixRay)*NumRaysPerBundleCell),MPI_BYTE,
                                    (int) recvTask,TAG_BUFF_LOADBAL,MPI_COMM_WORLD,&requestRecv);
                          didRecv = 1;
                        }
                      else
                        didRecv = 0;
                      
                      if(Nsend > 0)
                        {
                          MPI_Issend(bundleCells[bundleCellsRestrictedPeanoInd2Nest[i]].rays,
                                     (int) (sizeof(HEALPixRay)*NumRaysPerBundleCell),MPI_BYTE,
                                     (int) recvTask,TAG_BUFF_LOADBAL,MPI_COMM_WORLD,&requestSend);
                          
                          didSend = 1;
                        }
                      else
                        didSend = 0;
                      
                      if(didRecv)
                        {
                          MPI_Wait(&requestRecv,&Stat);
                          --Nrecv;
                          
                          bundleCellHasRays[bundleNestToRecv] = 1;
                        }
                      
                      if(didSend)
                        {
                          MPI_Wait(&requestSend,&Stat);
                          --Nsend;
                          
			  //move rays at end of vector to the old spot freed by bundleCellsRestrictedPeanoInd2Nest[i]
			  rayStartOfRaysToMove = NumAllRaysGlobal - NumRaysPerBundleCell;
			  bnestOfRaysToMove = (AllRaysGlobal[rayStartOfRaysToMove].nest >> shift);
			  if(bnestOfRaysToMove != bundleCellsRestrictedPeanoInd2Nest[i])
			    {
			      raysToMove = bundleCells[bnestOfRaysToMove].rays;
			      bundleCells[bnestOfRaysToMove].rays = bundleCells[bundleCellsRestrictedPeanoInd2Nest[i]].rays;
			      memcpy(bundleCells[bnestOfRaysToMove].rays,raysToMove,sizeof(HEALPixRay)*NumRaysPerBundleCell);
			    }
			  
			  bundleCellHasRays[bundleCellsRestrictedPeanoInd2Nest[i]] = 0;
			  bundleCells[bundleCellsRestrictedPeanoInd2Nest[i]].rays = NULL;
                          bundleCells[bundleCellsRestrictedPeanoInd2Nest[i]].Nrays = 0;
			  NumAllRaysGlobal -= NumRaysPerBundleCell;
			  
                          ++i;
                        }
                    }
                }
            }
        }
      
      //recompute # of cells to send and recv
      NumSendThisTask = 0;
      NumRecvThisTask = 0;
      for(i=0;i<NbundleCells;++i)
        {
          if(bundleCellHasRays[i] && !(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL)))
            ++NumSendThisTask;

          if(!(bundleCellHasRays[i]) && ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
            ++NumRecvThisTask;
        }
      MPI_Allreduce(&NumSendThisTask,&GlobalNumSend,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(&NumRecvThisTask,&GlobalNumRecv,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    }
  
  //clean it all up
  free(firstRPITasks);
  free(lastRPITasks);
  free(bundleCellHasRays);
  free(maxNumCellsToSend);
}

static void divide_tasks_domaindecomp(int firstTask, int lastTask, long firstPCell, long lastPCell, double *totCPUPerBundleCell)
{
  long i,Nt,ind;
  int splitTask;
  long splitPCell;
  long memUse,maxMemUse,maxMemUsePerTask;
  double maxCpuUse,cpuUse;
  
  /* - split array down the middle by cost
     - set split task so that [firstTask,splitTask] and (splitTask,lastTask]
       form a group of tasks
     - call recursively unit all tasks are split
  */
  
  Nt = lastTask - firstTask + 1;
  if(Nt == 1)
    {
      firstRestrictedPeanoIndTasks[firstTask] = firstPCell;
      lastRestrictedPeanoIndTasks[lastTask] = lastPCell;
      
      return;
    }
  else if((Nt & (Nt - 1)) != 0) //not a power of 2 so divide by hand
    {
      maxMemUsePerTask = ((long) ((1.0 + rayTraceData.maxRayMemImbalance)*((double) NrestrictedPeanoInd)/((double) NTasks)));
      
      maxCpuUse = 0.0;
      for(i=firstPCell;i<=lastPCell;++i)
	maxCpuUse += totCPUPerBundleCell[bundleCellsRestrictedPeanoInd2Nest[i]];
      maxCpuUse /= Nt;
      
      ind = firstPCell;
      firstRestrictedPeanoIndTasks[firstTask] = firstPCell;
      memUse = 1.0;
      cpuUse = totCPUPerBundleCell[bundleCellsRestrictedPeanoInd2Nest[ind]];
      ++ind;
      for(i=firstTask;i<lastTask;++i)
	{
	  while(cpuUse < maxCpuUse && memUse < maxMemUsePerTask)
	    {
	      ++ind;
	      memUse += 1.0;
	      cpuUse += totCPUPerBundleCell[bundleCellsRestrictedPeanoInd2Nest[ind]];
	    }
	  
	  lastRestrictedPeanoIndTasks[i] = ind;
	  firstRestrictedPeanoIndTasks[i+1] = ind + 1;
	  memUse = 1.0;
	  cpuUse = totCPUPerBundleCell[bundleCellsRestrictedPeanoInd2Nest[ind+1]];
	  ++ind;
	}
      
      lastRestrictedPeanoIndTasks[lastTask] = lastPCell;
      
      return;
    }
  else //recursive subdiv
    {
      maxMemUsePerTask = ((long) ((1.0 + rayTraceData.maxRayMemImbalance)*((double) NrestrictedPeanoInd)/((double) NTasks)));
      splitTask = (lastTask - firstTask)/2 + firstTask;
      maxMemUse = (splitTask - firstTask + 1)*maxMemUsePerTask;
      
      maxCpuUse = 0.0;
      for(i=firstPCell;i<=lastPCell;++i)
	maxCpuUse += totCPUPerBundleCell[bundleCellsRestrictedPeanoInd2Nest[i]];
      maxCpuUse /= 2.0;
      
      splitPCell = firstPCell + 1;
      memUse = 0.0;
      cpuUse = 0.0;
      for(i=firstPCell;i<=lastPCell;++i)
	{
	  if(memUse + 1 == maxMemUse || 
	     (cpuUse + totCPUPerBundleCell[bundleCellsRestrictedPeanoInd2Nest[i]] >= maxCpuUse && lastPCell - i <= (lastTask - splitTask)*maxMemUsePerTask))
	    {
	      splitPCell = i;
	      break;
	    }
	  
	  memUse += 1;
	  cpuUse += totCPUPerBundleCell[bundleCellsRestrictedPeanoInd2Nest[i]];
	}
      
      //if(ThisTask == 0)
      //fprintf(stderr,"%d: f,s,l task = %d|%d|%d, f,s,l pcell = %ld|%ld|%ld, memUse,cpuUse = %ld,%lf (%ld,%lf)\n",
      //    ThisTask,firstTask,splitTask,lastTask,firstPCell,splitPCell,lastPCell,memUse,cpuUse,maxMemUse,maxCpuUse);
      
      firstRestrictedPeanoIndTasks[firstTask] = firstPCell;
      lastRestrictedPeanoIndTasks[splitTask] = splitPCell;
      firstRestrictedPeanoIndTasks[splitTask+1] = splitPCell+1;
      lastRestrictedPeanoIndTasks[lastTask] = lastPCell;
      
      //recursive calls here
      divide_tasks_domaindecomp(firstTask,splitTask,firstPCell,splitPCell,totCPUPerBundleCell);
      divide_tasks_domaindecomp(splitTask+1,lastTask,splitPCell+1,lastPCell,totCPUPerBundleCell);
      
      return;
    }
}

static int mightNeedToSendBuffCellsRPI(long sendTask, long recvTask, long *minRPITasks, long *maxRPITasks, long *firstRPITasks, long *lastRPITasks)
{
  long needToSend = 0;
  
  if((firstRPITasks[recvTask] <= minRPITasks[sendTask] && minRPITasks[sendTask] <= lastRPITasks[recvTask])  ||
     (firstRPITasks[recvTask] <= maxRPITasks[sendTask] && maxRPITasks[sendTask] <= lastRPITasks[recvTask])  ||
     (minRPITasks[sendTask] <= firstRPITasks[recvTask] && firstRPITasks[recvTask] <= maxRPITasks[sendTask]) ||
     (minRPITasks[sendTask] <= lastRPITasks[recvTask]  && lastRPITasks[recvTask] <= maxRPITasks[sendTask])     )
    needToSend = 1;
  
  return needToSend;
}

