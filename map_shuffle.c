#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>

#include "raytrace.h"

//helper functions
static int needToSendBuffCellsNest(long sendTask, long recvTask, long *minNestTasks, long *maxNestTasks, long *firstNestTasks, long *lastNestTasks);
static void getRestrictedPeanoIndSendRange(long sendTask, long recvTask, long *minRestrictedPeanoIndTasks, long *maxRestrictedPeanoIndTasks,
					   long *Nsend, long *sendStart, HEALPixMapCell *mapvecCells, long NmapvecCells, long order);
static void getNorthSouthRingSendRange(long sendTask, long recvTask, long *minRingTasks, long *maxRingTasks, HEALPixSHTPlan plan,
				       long Nsend[2], long sendStart[2]);
static int compIndexHEALPixMapCell(const void *p1, const void *p2);
static int compLong(const void *p1, const void *p2);

void healpixmap_ring2peano_shuffle(float **mapvec_in, HEALPixSHTPlan plan)
{
  MPI_Status Stat;
  
  HEALPixMapCell *mapvecCells,mapvecCellSave,mapvecCellSource;
  long NmapvecCells;
  long i,j,firstRing,lastRing,nring,Nside,ringpix,bundleNest,order;
  long bundleMapShift,mapvecCellOffset;
  fftwf_complex *mapvec_complex;
  float *mapvec;
  long *mapvecCellRestrictedPeanoInd;
  size_t *index;
  long *rank,rankSave,dest,rankSource;
  
  long minRestrictedPeanoInd,maxRestrictedPeanoInd;
  long *minRestrictedPeanoIndTasks,*maxRestrictedPeanoIndTasks;
  
  long log2NTasks;
  long level,sendTask,recvTask;
  long sendStart,Nsend,Nrecv,Nworkspace;
  HEALPixMapCell *workspace,*workspacetmp,*workspaceCellsToSend,*workspaceCellsToRecv;
  long NworkspaceCellsToSend,NworkspaceCellsToRecv;
  long *nestIndsBuffCellsThisTask,*nestIndsBuffCellsToSend;
  long NnestIndsBuffCellsThisTask,NnestIndsBuffCellsToSend;
  long NnestRecv,*nestIndsBuffCellsToSendTmp;
  long firstNest,lastNest,minNest,maxNest;
  long *firstNestTasks,*lastNestTasks,*minNestTasks,*maxNestTasks,*match;
  
  double sortTime,runTime,activeTime,buffTime;
  double maxtm;
#ifdef DEBUG
#if DEBUG_LEVEL > 0
  double mintm,avgtm;
#endif
#endif

  runTime = 0.0;
  activeTime = 0.0;
  buffTime = 0.0;
  runTime -= MPI_Wtime();
  activeTime = runTime;
  
  order = plan.order;
  log2NTasks = 0;
  while(NTasks > (1 << log2NTasks))
    ++log2NTasks;
  Nside = order2nside(order);
  firstRing = plan.firstRingTasks[ThisTask];
  lastRing = plan.lastRingTasks[ThisTask];
  bundleMapShift = 2*(order - rayTraceData.bundleOrder);
  
  /* move cells to a HEALPixMapCell array*/
  NmapvecCells = 0;
  for(nring=firstRing;nring<=lastRing;++nring)
    {
      if(nring < Nside)
        ringpix = 4*nring;
      else
        ringpix = 4*Nside;
      
      if(nring != 2*Nside)
	NmapvecCells += 2*ringpix;
      else
	NmapvecCells += ringpix;
    }
  mapvecCells = (HEALPixMapCell*)malloc(sizeof(HEALPixMapCell)*NmapvecCells);
  assert(mapvecCells != NULL);
  j = 0;
  mapvec_complex = (fftwf_complex*) (*mapvec_in);
  for(nring=firstRing;nring<=lastRing;++nring)
    {
      if(nring < Nside)
        ringpix = 4*nring;
      else
        ringpix = 4*Nside;

      mapvec = (float*) (mapvec_complex+plan.northStartIndMapvec[nring-firstRing]);
      for(i=0;i<ringpix;++i)
        {
          mapvecCells[j].val = mapvec[i];
	  mapvecCells[j].index = i + plan.northStartIndGlobalMap[nring-firstRing];
	  ++j;
	}
      
      if(nring != 2*Nside)
        {
          mapvec = (float*) (mapvec_complex+plan.southStartIndMapvec[nring-firstRing]);
          for(i=0;i<ringpix;++i)
            {
              mapvecCells[j].val = mapvec[i];
	      mapvecCells[j].index = i + plan.southStartIndGlobalMap[nring-firstRing];
	      ++j;
	    }
        }
    }
  free(*mapvec_in);
  *mapvec_in = NULL;
  
  /* move cells to restricted peano index order */
  sortTime = 0.0;
  sortTime -= MPI_Wtime();
  mapvecCellRestrictedPeanoInd = (long*)malloc(sizeof(long)*NmapvecCells);
  assert(mapvecCellRestrictedPeanoInd != NULL);
  for(i=0;i<NmapvecCells;++i)
    {
      mapvecCells[i].index = ring2nest(mapvecCells[i].index,order);
      bundleNest = (mapvecCells[i].index >> bundleMapShift);
      mapvecCellRestrictedPeanoInd[i] = bundleCellsNest2RestrictedPeanoInd[bundleNest];
    }
  index = (size_t*)malloc(sizeof(size_t)*NmapvecCells);
  assert(index != NULL);
  gsl_sort_long_index(index,mapvecCellRestrictedPeanoInd,(size_t) 1,(size_t) NmapvecCells);
  free(mapvecCellRestrictedPeanoInd);
  rank = (long*)malloc(sizeof(long)*NmapvecCells);
  assert(rank != NULL);
  for(i=0;i<NmapvecCells;++i)
    rank[index[i]] = i;
  free(index);
  for(i=0;i<NmapvecCells;++i) /* reoder with an in-place algorithm - see Gadget-2 for details - destroys rank */
    {
      if(i != rank[i])
	{
	  mapvecCellSource = mapvecCells[i];
	  rankSource = rank[i];
	  dest = rank[i];
	  
	  do
	    {
	      mapvecCellSave = mapvecCells[dest];
	      rankSave = rank[dest];
	      
	      mapvecCells[dest] = mapvecCellSource;
	      rank[dest] = rankSource;
	      
	      if(dest == i)
		break;
	      
	      mapvecCellSource = mapvecCellSave;
	      rankSource = rankSave;
	      
	      dest = rankSource;

	    }
	  while(1);
	}
    }
  free(rank);
  sortTime += MPI_Wtime();
    
  /* get min and max to test for overlap between nodes */
  minRestrictedPeanoInd = NbundleCells;
  maxRestrictedPeanoInd = -1;
  for(i=0;i<NmapvecCells;++i)
    {
      bundleNest = (mapvecCells[i].index >> bundleMapShift);
      if(bundleCellsNest2RestrictedPeanoInd[bundleNest] > -1)
	{
	  if(bundleCellsNest2RestrictedPeanoInd[bundleNest] < minRestrictedPeanoInd)
	    minRestrictedPeanoInd = bundleCellsNest2RestrictedPeanoInd[bundleNest];
	  
	  if(bundleCellsNest2RestrictedPeanoInd[bundleNest] > maxRestrictedPeanoInd)
	    maxRestrictedPeanoInd = bundleCellsNest2RestrictedPeanoInd[bundleNest];
	}
    }
  minRestrictedPeanoIndTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(minRestrictedPeanoIndTasks != NULL);
  maxRestrictedPeanoIndTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(maxRestrictedPeanoIndTasks != NULL);
  MPI_Allgather(&minRestrictedPeanoInd,1,MPI_LONG,minRestrictedPeanoIndTasks,1,MPI_LONG,MPI_COMM_WORLD);
  MPI_Allgather(&maxRestrictedPeanoInd,1,MPI_LONG,maxRestrictedPeanoIndTasks,1,MPI_LONG,MPI_COMM_WORLD);
#ifdef DEBUG
#if DEBUG_LEVEL > 0
  if(ThisTask == 0)
    {
      for(i=0;i<NTasks;++i)
	fprintf(stderr,"%ld: minRestrictedPeanoInd,maxRestrictedPeanoInd = %ld|%ld, firstRestrictedPeanoInd,lastRestrictedPeanoInd = %ld|%ld\n",i,
		minRestrictedPeanoIndTasks[i],maxRestrictedPeanoIndTasks[i],
		firstRestrictedPeanoIndTasks[i],lastRestrictedPeanoIndTasks[i]);
    }
#endif
#endif
  
  /* mem for workspace */
  Nworkspace = NmapCells/NTasks;
  workspace = (HEALPixMapCell*)malloc(sizeof(HEALPixMapCell)*Nworkspace);
  assert(workspace != NULL);
  
  /*algorithm to loop through pairs of tasks linearly
    -lifted from Gadget-2 under GPL (http://www.gnu.org/copyleft/gpl.html)
    -see pm_periodic.c from Gadget-2 at http://www.mpa-garching.mpg.de/gadget/
  */
  for(level = 0; level < (1 << log2NTasks); level++) /* note: for level=0, target is the same task */
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;
      if(recvTask < NTasks)
        {
	  /* compute overlap of data to *send* from sendTask to recvTask */
          getRestrictedPeanoIndSendRange(sendTask,recvTask,minRestrictedPeanoIndTasks,maxRestrictedPeanoIndTasks,&Nsend,&sendStart,mapvecCells,NmapvecCells,plan.order);
	  
          if(sendTask != recvTask)
            {
              MPI_Sendrecv(&Nsend,1,MPI_LONG,(int) recvTask,TAG_NUMDATA_R2P,
                           &Nrecv,1,MPI_LONG,(int) recvTask,TAG_NUMDATA_R2P,
                           MPI_COMM_WORLD,&Stat);
            }
          else
            {
              Nrecv = Nsend;
	    }
          
          if(Nsend > 0 || Nrecv > 0) /* there exists data that either has to be sent or received */
            {
              /* make sure workspace is large enough */
              if(Nrecv > Nworkspace)
                {
                  workspacetmp = (HEALPixMapCell*)realloc(workspace,sizeof(HEALPixMapCell)*Nrecv);
                  if(workspacetmp != NULL)
                    {
                      workspace = workspacetmp;
                      Nworkspace = Nrecv;
                    }
                  else
                    {
                      fprintf(stderr,"%d: out of memory for workspace in healpixmap_ring2peano_shuffle!\n",ThisTask);
                      MPI_Abort(MPI_COMM_WORLD,123);
                    }
		}

              if(sendTask != recvTask)
                {
                  MPI_Sendrecv(mapvecCells+sendStart,(int) (Nsend*sizeof(HEALPixMapCell)),MPI_BYTE,(int) recvTask,TAG_DATA_R2P,
                               workspace,(int) (Nrecv*sizeof(HEALPixMapCell)),MPI_BYTE,(int) recvTask,TAG_DATA_R2P,
                               MPI_COMM_WORLD,&Stat);
		}
              else /* just move cells into workspace since sendTask == recvTask and Nsend == Nrecv */
                {
                  for(i=0;i<Nrecv;++i)
                    workspace[i] = mapvecCells[sendStart+i];
		}
            }

          /* put data into final buffer if we need to */
          if(Nrecv > 0) /* there exists data that has been received */
	    {
	      for(i=0;i<Nrecv;++i)
		{
		  bundleNest = (workspace[i].index >> bundleMapShift);
		  mapvecCellOffset = workspace[i].index - (bundleNest << bundleMapShift);
		  mapCells[bundleCells[bundleNest].firstMapCell+mapvecCellOffset].val = workspace[i].val;
		  assert(mapCells[bundleCells[bundleNest].firstMapCell+mapvecCellOffset].index ==  workspace[i].index);
		}
	    }
	}
    }
  free(minRestrictedPeanoIndTasks);
  free(maxRestrictedPeanoIndTasks);
  free(workspace);
  
  maxtm = MPI_Wtime();
  activeTime += maxtm;
  buffTime -= maxtm;
#ifdef DEBUG
#if DEBUG_LEVEL > 0
  MPI_Reduce(&sortTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&sortTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&sortTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"ring to peano map shuffle sort time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
  
  MPI_Reduce(&activeTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&activeTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&activeTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"ring to peano map shuffle active time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
#endif
#endif
  
  /* mem for workspace and nest inds of bundleCells */
  NworkspaceCellsToSend = NmapCells/NTasks;
  workspaceCellsToSend = (HEALPixMapCell*)malloc(sizeof(HEALPixMapCell)*NworkspaceCellsToSend);
  assert(workspaceCellsToSend != NULL);
  NworkspaceCellsToRecv = NmapCells/NTasks;
  workspaceCellsToRecv = (HEALPixMapCell*)malloc(sizeof(HEALPixMapCell)*NworkspaceCellsToRecv);
  assert(workspaceCellsToRecv != NULL);
  
  NnestIndsBuffCellsThisTask = 0;
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,MAPBUFF_BUNDLECELL))
	NnestIndsBuffCellsThisTask += 1;
    }
  nestIndsBuffCellsThisTask = (long*)malloc(sizeof(long)*NnestIndsBuffCellsThisTask);
  assert(nestIndsBuffCellsThisTask != NULL);
  j = 0;
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,MAPBUFF_BUNDLECELL))
	{
	  nestIndsBuffCellsThisTask[j] = bundleCells[i].nest;
	  ++j;
	}
    }
  NnestIndsBuffCellsToSend = NnestIndsBuffCellsThisTask;
  nestIndsBuffCellsToSend = (long*)malloc(sizeof(long)*NnestIndsBuffCellsToSend);
  assert(nestIndsBuffCellsToSend != NULL);
  
  /* move cells to nest order */
  sortTime = 0.0;
  sortTime -= MPI_Wtime();
  qsort(mapvecCells,(size_t) NmapvecCells,sizeof(HEALPixMapCell),compIndexHEALPixMapCell);
  sortTime += MPI_Wtime();
  
  /* get min,max and first,last nest inds to test for overlaps */
  firstNest = nestIndsBuffCellsThisTask[0];
  lastNest = nestIndsBuffCellsThisTask[NnestIndsBuffCellsThisTask-1];
  firstNestTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(firstNestTasks != NULL);
  lastNestTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(lastNestTasks != NULL);
  MPI_Allgather(&firstNest,1,MPI_LONG,firstNestTasks,1,MPI_LONG,MPI_COMM_WORLD);
  MPI_Allgather(&lastNest,1,MPI_LONG,lastNestTasks,1,MPI_LONG,MPI_COMM_WORLD);
  minNest = order2npix(rayTraceData.bundleOrder);
  maxNest = -1;
  for(i=0;i<NmapvecCells;++i)
    {
      bundleNest = (mapvecCells[i].index >> bundleMapShift);
      if(bundleNest < minNest)
	minNest = bundleNest;
      if(bundleNest > maxNest)
	maxNest = bundleNest;
    }
  minNestTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(minNestTasks != NULL);
  maxNestTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(maxNestTasks != NULL);
  MPI_Allgather(&minNest,1,MPI_LONG,minNestTasks,1,MPI_LONG,MPI_COMM_WORLD);
  MPI_Allgather(&maxNest,1,MPI_LONG,maxNestTasks,1,MPI_LONG,MPI_COMM_WORLD);
#ifdef DEBUG
#if DEBUG_LEVEL > 0
  if(ThisTask == 0)
    {
      for(i=0;i<NTasks;++i)
        fprintf(stderr,"%ld: minNest,maxNest = %ld|%ld, firstNest,lastNest = %ld|%ld\n",i,
                minNestTasks[i],maxNestTasks[i],
                firstNestTasks[i],lastNestTasks[i]);
    }
#endif
#endif

  /*algorithm to loop through pairs of tasks linearly
    -lifted from Gadget-2 under GPL (http://www.gnu.org/copyleft/gpl.html)
    -see pm_periodic.c from Gadget-2 at http://www.mpa-garching.mpg.de/gadget/
  */
  for(level = 0; level < (1 << log2NTasks); level++) /* note: for level=0, target is the same task */
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;
      if(recvTask < NTasks)
        {
	  /* check that range of nest inds overlaps, if yes then need to do sendrecv, else do not do it and save some work */
	  if(needToSendBuffCellsNest(sendTask,recvTask,minNestTasks,maxNestTasks,firstNestTasks,lastNestTasks) ||
	     needToSendBuffCellsNest(recvTask,sendTask,minNestTasks,maxNestTasks,firstNestTasks,lastNestTasks))
	    {
	      /* send nest inds of cells needed by sendTask from sendTask to recvTask */
	      if(sendTask != recvTask)
		{
		  MPI_Sendrecv(&NnestIndsBuffCellsThisTask,1,MPI_LONG,(int) recvTask,TAG_NUMNEST_R2P,
			       &NnestRecv,1,MPI_LONG,(int) recvTask,TAG_NUMNEST_R2P,
			       MPI_COMM_WORLD,&Stat);
		}
	      else
		{
		  NnestRecv = NnestIndsBuffCellsThisTask;
		}
	      
	      /* make sure we have enough memory */
	      if(NnestRecv > NnestIndsBuffCellsToSend)
		{
		  nestIndsBuffCellsToSendTmp = (long*)realloc(nestIndsBuffCellsToSend,sizeof(long)*NnestRecv);
		  if(nestIndsBuffCellsToSendTmp != NULL)
		    {
		      nestIndsBuffCellsToSend = nestIndsBuffCellsToSendTmp;
		      NnestIndsBuffCellsToSend = NnestRecv;
		    }
		  else
		    {
		      fprintf(stderr,"%d: out of memory for nestIndsBuffCellsToSend in healpixmap_ring2peano_shuffle!\n",ThisTask);
		      MPI_Abort(MPI_COMM_WORLD,123);
		    }
		}
	      
	      /* do actual send of nest inds */
	      if(sendTask != recvTask)
		{
		  MPI_Sendrecv(nestIndsBuffCellsThisTask,(int) NnestIndsBuffCellsThisTask,MPI_LONG,(int) recvTask,TAG_NEST_R2P,
			       nestIndsBuffCellsToSend,(int) NnestRecv,MPI_LONG,(int) recvTask,TAG_NEST_R2P,
			       MPI_COMM_WORLD,&Stat);
		}
	      else
		{
		  for(i=0;i<NnestRecv;++i)
		    nestIndsBuffCellsToSend[i] = nestIndsBuffCellsThisTask[i];
		}
	      
	      /* compute number of cells to send from sendTask to recvTask by looking for nestIndsBuffCellsToSend in mapvecCells 
		 also put cells to send in the workspace send buffer workspaceCellsToSend */
	      Nsend = 0;
	      bundleNest = -1;
	      match = NULL;
	      for(j=0;j<NmapvecCells;++j)
		{
		  if((mapvecCells[j].index >> bundleMapShift) != bundleNest)
		    {
		      bundleNest = (mapvecCells[j].index >> bundleMapShift);
		      match = (long*)bsearch(&bundleNest,nestIndsBuffCellsToSend,(size_t) NnestRecv,sizeof(long),compLong);
		    }
		  
		  if(match != NULL && (mapvecCells[j].index >> bundleMapShift) == bundleNest)
		    {
		      /* get extra mem if needed */
		      if(Nsend >= NworkspaceCellsToSend)
			{
			  workspacetmp = (HEALPixMapCell*)realloc(workspaceCellsToSend,sizeof(HEALPixMapCell)*(NworkspaceCellsToSend + 10000));
			  
			  if(workspacetmp != NULL)
			    {
			      workspaceCellsToSend = workspacetmp;
			      NworkspaceCellsToSend += 10000;
			    }
			  else
			    {
			      fprintf(stderr,"%d: out of memory for workspaceCellsToSend in healpixmap_ring2peano_shuffle!\n",ThisTask);
			      MPI_Abort(MPI_COMM_WORLD,123);
			    }
			}
		      
		      workspaceCellsToSend[Nsend] = mapvecCells[j];
		      ++Nsend;
		    }
		}
	      
	      /* get number of cells to recv*/
	      if(sendTask != recvTask)
		{
		  MPI_Sendrecv(&Nsend,1,MPI_LONG,(int) recvTask,TAG_NUMBUFF_R2P,
			       &Nrecv,1,MPI_LONG,(int) recvTask,TAG_NUMBUFF_R2P,
			       MPI_COMM_WORLD,&Stat);
		}
	      else
		{
		  Nrecv = Nsend;
		}
	      
	      if(Nrecv > 0 || Nsend > 0)
		{
		  /* make sure workspace recv buffer is large enough */
		  if(Nrecv > NworkspaceCellsToRecv)
		    {
		      workspacetmp = (HEALPixMapCell*)realloc(workspaceCellsToRecv,sizeof(HEALPixMapCell)*Nrecv);
		      if(workspacetmp != NULL)
			{
			  workspaceCellsToRecv = workspacetmp;
			  NworkspaceCellsToRecv = Nrecv;
			}
		      else
			{
			  fprintf(stderr,"%d: out of memory for workspaceCellsToRecv in healpixmap_ring2peano_shuffle!\n",ThisTask);
			  MPI_Abort(MPI_COMM_WORLD,123);
			}
		    }
		  
		  if(sendTask != recvTask)
		    {
		      MPI_Sendrecv(workspaceCellsToSend,(int) (Nsend*sizeof(HEALPixMapCell)),MPI_BYTE,(int) recvTask,TAG_BUFF_R2P,
				   workspaceCellsToRecv,(int) (Nrecv*sizeof(HEALPixMapCell)),MPI_BYTE,(int) recvTask,TAG_BUFF_R2P,
				   MPI_COMM_WORLD,&Stat);
		    }
		  else /* just move cells into workspace since sendTask == recvTask and Nsend == Nrecv */
		    {
		      for(i=0;i<Nrecv;++i)
			workspaceCellsToRecv[i] = workspaceCellsToSend[i];
		    }
		}
	      
	      /* put data into final buffer if we need to */
	      if(Nrecv > 0)
		{
		  for(i=0;i<Nrecv;++i)
		    {
		      bundleNest = (workspaceCellsToRecv[i].index >> bundleMapShift);
		      mapvecCellOffset = workspaceCellsToRecv[i].index - (bundleNest << bundleMapShift);
		      mapCells[bundleCells[bundleNest].firstMapCell+mapvecCellOffset].val = workspaceCellsToRecv[i].val;
		      assert(mapCells[bundleCells[bundleNest].firstMapCell+mapvecCellOffset].index == workspaceCellsToRecv[i].index);
		    }
		}
	    }
	}
    }
  
  free(nestIndsBuffCellsToSend);
  free(nestIndsBuffCellsThisTask);
  free(workspaceCellsToSend);
  free(workspaceCellsToRecv);
  free(minNestTasks);
  free(maxNestTasks);
  free(firstNestTasks);
  free(lastNestTasks);
  free(mapvecCells);
  
  maxtm = MPI_Wtime();
  runTime += maxtm;
  buffTime += maxtm;
#ifdef DEBUG
#if DEBUG_LEVEL > 0
  MPI_Reduce(&sortTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&sortTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&sortTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"ring to peano map shuffle sort time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);

  MPI_Reduce(&buffTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&buffTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&buffTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"ring to peano map shuffle buff time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);

  MPI_Reduce(&runTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&runTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&runTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  if(ThisTask == 0)
    fprintf(stderr,"ring to peano map shuffle run time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
#endif
#endif
  
  if(ThisTask == 0)
    fprintf(stderr,"ring to peano map shuffle took %lg seconds.\n",runTime);
}

static int needToSendBuffCellsNest(long sendTask, long recvTask, long *minNestTasks, long *maxNestTasks, long *firstNestTasks, long *lastNestTasks)
{
  long needToSend = 0;
  
  if((firstNestTasks[recvTask] <= minNestTasks[sendTask] && minNestTasks[sendTask] <= lastNestTasks[recvTask])  ||
     (firstNestTasks[recvTask] <= maxNestTasks[sendTask] && maxNestTasks[sendTask] <= lastNestTasks[recvTask])  ||
     (minNestTasks[sendTask] <= firstNestTasks[recvTask] && firstNestTasks[recvTask] <= maxNestTasks[sendTask]) ||
     (minNestTasks[sendTask] <= lastNestTasks[recvTask]  && lastNestTasks[recvTask] <= maxNestTasks[sendTask])     )
    needToSend = 1;
  
  return needToSend;
}

static void getRestrictedPeanoIndSendRange(long sendTask, long recvTask, long *minRestrictedPeanoIndTasks, long *maxRestrictedPeanoIndTasks,
					   long *Nsend, long *sendStart, HEALPixMapCell *mapvecCells, long NmapvecCells, long order)
{
  long i;
  long bundleNest,bundleMapShift,restrictedPeanoInd;
  long minRestrictedPeanoIndSend,maxRestrictedPeanoIndSend;
  
  bundleMapShift = 2*(order - rayTraceData.bundleOrder);

  /* do north+equator rings first */
  minRestrictedPeanoIndSend = -1;
  maxRestrictedPeanoIndSend = -1;
  *Nsend = 0;
  *sendStart = 0;
  if((firstRestrictedPeanoIndTasks[recvTask] <= minRestrictedPeanoIndTasks[sendTask] 
      && minRestrictedPeanoIndTasks[sendTask] <= lastRestrictedPeanoIndTasks[recvTask])  ||
     (firstRestrictedPeanoIndTasks[recvTask] <= maxRestrictedPeanoIndTasks[sendTask] 
      && maxRestrictedPeanoIndTasks[sendTask] <= lastRestrictedPeanoIndTasks[recvTask])  ||
     (minRestrictedPeanoIndTasks[sendTask] <= firstRestrictedPeanoIndTasks[recvTask] 
      && firstRestrictedPeanoIndTasks[recvTask] <= maxRestrictedPeanoIndTasks[sendTask]) ||
     (minRestrictedPeanoIndTasks[sendTask] <= lastRestrictedPeanoIndTasks[recvTask] 
      && lastRestrictedPeanoIndTasks[recvTask] <= maxRestrictedPeanoIndTasks[sendTask])     )
    {
      if(firstRestrictedPeanoIndTasks[recvTask] < minRestrictedPeanoIndTasks[sendTask])
	minRestrictedPeanoIndSend = minRestrictedPeanoIndTasks[sendTask];
      else
	minRestrictedPeanoIndSend = firstRestrictedPeanoIndTasks[recvTask];
      
      if(lastRestrictedPeanoIndTasks[recvTask] < maxRestrictedPeanoIndTasks[sendTask])
	maxRestrictedPeanoIndSend = lastRestrictedPeanoIndTasks[recvTask];
      else
	maxRestrictedPeanoIndSend = maxRestrictedPeanoIndTasks[sendTask];
      
      for(i=0;i<NmapvecCells;++i)
	{
	  bundleNest = mapvecCells[i].index >> bundleMapShift;
	  restrictedPeanoInd = bundleCellsNest2RestrictedPeanoInd[bundleNest];
	  if(restrictedPeanoInd < minRestrictedPeanoIndSend)
	    (*sendStart) += 1;
	  if(restrictedPeanoInd >= minRestrictedPeanoIndSend && restrictedPeanoInd <= maxRestrictedPeanoIndSend)
	    (*Nsend) += 1;
	}
    }
  
#ifdef DEBUG
#if DEBUG_LEVEL > 1
  fprintf(stderr,"%ld -> %ld: min,max = %ld|%ld, first,last = %ld|%ld, minSend,maxSend = %ld|%ld, Nsend = %ld, sendStart = %ld\n",sendTask,recvTask,
	  minRestrictedPeanoIndTasks[sendTask],maxRestrictedPeanoIndTasks[sendTask],
	  firstRestrictedPeanoIndTasks[recvTask],lastRestrictedPeanoIndTasks[recvTask],
	  minRestrictedPeanoIndSend,maxRestrictedPeanoIndSend,*Nsend,*sendStart);
#endif
#endif
}

void healpixmap_peano2ring_shuffle(float *mapvec, HEALPixSHTPlan plan)
{
  MPI_Status Stat;
  
  long i,minRing,maxRing,ringnum;
  long *minRingTasks,*maxRingTasks;
  long order;
  
  long log2NTasks;
  long level,sendTask,recvTask;
  long sendStart[2],Nsend[2],Nrecv[2],Nworkspace,totNrecv;
  HEALPixMapCell *workspace,*workspacetmp;
  fftwf_complex *mapvec_complex;
  
  long firstRing,lastRing,nringnum,startpix,Nside;
  float *mapvectmp;
  
  long nring,ringpix;
  double runTime;
  
#ifdef DEBUG
#if DEBUG_LEVEL > 2
  double barTime;
#endif
#endif
    
  runTime = 0.0;
  runTime -= MPI_Wtime();
  
  order = plan.order;
  log2NTasks = 0;
  while(NTasks > (1 << log2NTasks))
    ++log2NTasks;
  firstRing = plan.firstRingTasks[ThisTask];
  lastRing = plan.lastRingTasks[ThisTask];
  Nside = order2nside(order);
  
  /* mem for workspace */
  Nworkspace = NmapCells/NTasks;
  workspace = (HEALPixMapCell*)malloc(sizeof(HEALPixMapCell)*Nworkspace);
  assert(workspace != NULL);
      
  /* get ring ranges for all tasks */
  minRing = 4*order2nside(order);
  maxRing = -1;
  for(i=0;i<NmapCells;++i)
    {
      mapCells[i].index = nest2ring(mapCells[i].index,order);
      ringnum = ring2ringnum(mapCells[i].index,order);
      if(ringnum > maxRing)
	maxRing = ringnum;
      if(ringnum < minRing)
	minRing = ringnum;
    }
  minRingTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(minRingTasks != NULL);
  maxRingTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(maxRingTasks != NULL);
  MPI_Allgather(&minRing,1,MPI_LONG,minRingTasks,1,MPI_LONG,MPI_COMM_WORLD);
  MPI_Allgather(&maxRing,1,MPI_LONG,maxRingTasks,1,MPI_LONG,MPI_COMM_WORLD);
#ifdef DEBUG
  if(ThisTask == 0 && DEBUG_LEVEL > 0)
    {
      for(i=0;i<NTasks;++i)
	fprintf(stderr,"%ld: minRing,maxring = %ld|%ld, firstRing,lastRing = %ld|%ld\n",i,minRingTasks[i],maxRingTasks[i],
		plan.firstRingTasks[i],plan.lastRingTasks[i]);
    }
#endif

#ifdef DEBUG
#if DEBUG_LEVEL > 2
  barTime = -MPI_Wtime();
#endif
#endif
  
  /* sort map cells into ring order */
  qsort(mapCells,(size_t) NmapCells,sizeof(HEALPixMapCell),compIndexHEALPixMapCell);
  
  /* zero mapvec in order to recv cell vals  - needed if NGP is not used for density assignment*/
  mapvec_complex = (fftwf_complex*) mapvec;
  for(nring=firstRing;nring<=lastRing;++nring)
    {
      if(nring < Nside)
        ringpix = 4*nring;
      else
        ringpix = 4*Nside;
      
      mapvectmp = (float*) (mapvec_complex+plan.northStartIndMapvec[nring-firstRing]);
      for(i=0;i<ringpix;++i)
	mapvectmp[i] = 0.0;
      
      if(nring != 2*Nside)
        {
          mapvectmp = (float*) (mapvec_complex+plan.southStartIndMapvec[nring-firstRing]);
	  for(i=0;i<ringpix;++i)
	    mapvectmp[i] = 0.0;
	}
    }
  
  /*algorithm to loop through pairs of tasks linearly 
    -lifted from Gadget-2 under GPL (http://www.gnu.org/copyleft/gpl.html) 
    -see pm_periodic.c from Gadget-2 at http://www.mpa-garching.mpg.de/gadget/ 
  */
  for(level = 0; level < (1 << log2NTasks); level++) /* note: for level=0, target is the same task */
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;
      if(recvTask < NTasks)
	{

#ifdef DEBUG
#if DEBUG_LEVEL > 2
	  fprintf(stderr,"%d (%lg since start, %lg since allgather): send,recv task = %ld|%ld \n",ThisTask,MPI_Wtime()+runTime,MPI_Wtime()+barTime,sendTask,recvTask);
#endif
#endif
	  
	  /* compute overlap of data to *send* from sendTask to recvTask */
	  getNorthSouthRingSendRange(sendTask,recvTask,minRingTasks,maxRingTasks,plan,Nsend,sendStart);
	  
	  if(sendTask != recvTask)
	    {
	      MPI_Sendrecv(Nsend,2,MPI_LONG,(int) recvTask,TAG_NUMDATA_P2R,
			   Nrecv,2,MPI_LONG,(int) recvTask,TAG_NUMDATA_P2R,
			   MPI_COMM_WORLD,&Stat);
	    }
	  else
	    {
	      Nrecv[0] = Nsend[0];
	      Nrecv[1] = Nsend[1];
	    }
	  totNrecv = Nrecv[0] + Nrecv[1];
	  
	  if(Nsend[0] > 0 || Nsend[1] > 0 || Nrecv[0] > 0 || Nrecv[1] > 0) /* there exists data that either has to be sent or received */
	    {
	      /* make sure workspace is large enough */
	      if(totNrecv > Nworkspace)
		{
		  workspacetmp = (HEALPixMapCell*)realloc(workspace,sizeof(HEALPixMapCell)*totNrecv);
		  if(workspacetmp != NULL)
		    {
		      workspace = workspacetmp;
		      Nworkspace = totNrecv;
		    }
		  else
		    {
		      fprintf(stderr,"%d: out of memory for workspace in healpixmap_peano2ring_shuffle!\n",ThisTask);
		      MPI_Abort(MPI_COMM_WORLD,123);
		    }
		}
	      
	      if(sendTask != recvTask)
		{
		  MPI_Sendrecv(mapCells+sendStart[0],(int) (Nsend[0]*sizeof(HEALPixMapCell)),MPI_BYTE,(int) recvTask,TAG_DATA_P2R_NORTH,
			       workspace,(int) (Nrecv[0]*sizeof(HEALPixMapCell)),MPI_BYTE,(int) recvTask,TAG_DATA_P2R_NORTH,
			       MPI_COMM_WORLD,&Stat);
		  MPI_Sendrecv(mapCells+sendStart[1],(int) (Nsend[1]*sizeof(HEALPixMapCell)),MPI_BYTE,(int) recvTask,TAG_DATA_P2R_SOUTH,
			       workspace+Nrecv[0],(int) (Nrecv[1]*sizeof(HEALPixMapCell)),MPI_BYTE,(int) recvTask,TAG_DATA_P2R_SOUTH,
			       MPI_COMM_WORLD,&Stat);
		}
	      else /* just move cells into workspace since sendTask == recvTask and Nsend == Nrecv */
		{
		  for(i=0;i<Nrecv[0];++i)
		    workspace[i] = mapCells[sendStart[0]+i];
		  for(i=0;i<Nrecv[1];++i)
		    workspace[Nrecv[0]+i] = mapCells[sendStart[1]+i];
		}
	    }
	  
	  /* put data into final buffer if we need to */
	  if(totNrecv > 0) /* there exists data that has been received */
	    {
	      ringnum = ring2ringnum(workspace[0].index,order);
	      if(ringnum > 2*Nside)
		{
		  nringnum = 4*Nside - ringnum;
		  mapvectmp = (float*) (mapvec_complex + plan.southStartIndMapvec[nringnum-firstRing]);
		  startpix = plan.southStartIndGlobalMap[nringnum-firstRing];
		}
	      else
		{
		  nringnum = ringnum;
		  mapvectmp = (float*) (mapvec_complex + plan.northStartIndMapvec[nringnum-firstRing]);
		  startpix = plan.northStartIndGlobalMap[nringnum-firstRing];
		}
	      	      
	      for(i=0;i<totNrecv;++i)
		{
		  /* reset pointers to mem */
		  if(ringnum != ring2ringnum(workspace[i].index,order))
		    {
		      ringnum = ring2ringnum(workspace[i].index,order);
		      if(ringnum > 2*Nside)
			{
			  nringnum = 4*Nside - ringnum;
			  assert(firstRing <= nringnum && nringnum <= lastRing);
			  mapvectmp = (float*) (mapvec_complex + plan.southStartIndMapvec[nringnum-firstRing]);
			  startpix = plan.southStartIndGlobalMap[nringnum-firstRing];
			}
		      else
			{
			  nringnum = ringnum;
			  assert(firstRing <= nringnum && nringnum <= lastRing);
			  mapvectmp = (float*) (mapvec_complex + plan.northStartIndMapvec[nringnum-firstRing]);
			  startpix = plan.northStartIndGlobalMap[nringnum-firstRing];
			}
		    }
		  
		  /* fill in vals */
		  mapvectmp[workspace[i].index - startpix] += workspace[i].val;
		}
	    }
	}
    }
  
  /* put mapCells back into nest order */
  for(i=0;i<NmapCells;++i)
    mapCells[i].index = ring2nest(mapCells[i].index,order);
  qsort(mapCells,(size_t) NmapCells,sizeof(HEALPixMapCell),compIndexHEALPixMapCell);
  
  free(minRingTasks);
  free(maxRingTasks);
  free(workspace);
  
  runTime += MPI_Wtime();
  
#ifdef DEBUG
#if DEBUG_LEVEL > 0
  double mintm,maxtm,avgtm;
  
  MPI_Reduce(&runTime,&mintm,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&runTime,&maxtm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&runTime,&avgtm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgtm = avgtm/NTasks;
  
  if(ThisTask == 0)
    fprintf(stderr,"peano to ring map shuffle run time and load balance: max,min,avg = %f|%f|%f sec (%.2f percent)\n",maxtm,mintm,avgtm,(maxtm-avgtm)/avgtm*100.0);
#endif
#endif
  
  if(ThisTask == 0)
    fprintf(stderr,"peano to ring map shuffle took %lg seconds.\n",runTime);
}

static void getNorthSouthRingSendRange(long sendTask, long recvTask, long *minRingTasks, long *maxRingTasks, HEALPixSHTPlan plan,
				       long Nsend[2], long sendStart[2])
{
  long minRing,maxRing,minRingSend[2],maxRingSend[2],minRingNorth,maxRingNorth,i,ringnum;
  long Nside = order2nside(plan.order);
  long order = plan.order;
  
  /* do north+equator rings first */
  minRingSend[0] = -1;
  maxRingSend[0] = -1;
  Nsend[0] = 0;
  sendStart[0] = 0;
  minRing = minRingTasks[sendTask];
  maxRing = maxRingTasks[sendTask];
  if(maxRing > 2*Nside)
    maxRing = 2*Nside;
  if(minRing <= 2*Nside)
    {
      if((plan.firstRingTasks[recvTask] <= minRing && minRing <= plan.lastRingTasks[recvTask])  ||
	 (plan.firstRingTasks[recvTask] <= maxRing && maxRing <= plan.lastRingTasks[recvTask])  ||
	 (minRing <= plan.firstRingTasks[recvTask] && plan.firstRingTasks[recvTask] <= maxRing) ||
	 (minRing <= plan.lastRingTasks[recvTask] && plan.lastRingTasks[recvTask] <= maxRing)     )
	{
	  if(plan.firstRingTasks[recvTask] < minRing)
	    minRingSend[0] = minRing;
	  else
	    minRingSend[0] = plan.firstRingTasks[recvTask];
	  
	  if(plan.lastRingTasks[recvTask] < maxRing)
	    maxRingSend[0] = plan.lastRingTasks[recvTask];
	  else
	    maxRingSend[0] = maxRing;
	  
	  for(i=0;i<NmapCells;++i)
	    {
	      ringnum = ring2ringnum(mapCells[i].index,order);
	      if(ringnum < minRingSend[0])
		++sendStart[0];
	      if(ringnum >= minRingSend[0] && ringnum <= maxRingSend[0])
		++Nsend[0];
	    }
	}
    }
  
  /* do south rings */
  minRingSend[1] = -1;
  maxRingSend[1] = -1;
  Nsend[1] = 0;
  sendStart[1] = 0;
  minRing = minRingTasks[sendTask];
  maxRing = maxRingTasks[sendTask];
  if(minRing <= 2*Nside)
    minRing = 2*Nside+1;
  maxRingNorth = 4*Nside - minRing;
  minRingNorth = 4*Nside - maxRing;
  if(maxRing > 2*Nside)
    {
      if((plan.firstRingTasks[recvTask] <= minRingNorth && minRingNorth <= plan.lastRingTasks[recvTask])  ||
	 (plan.firstRingTasks[recvTask] <= maxRingNorth && maxRingNorth <= plan.lastRingTasks[recvTask])  ||
	 (minRingNorth <= plan.firstRingTasks[recvTask] && plan.firstRingTasks[recvTask] <= maxRingNorth) ||
	 (minRingNorth <= plan.lastRingTasks[recvTask] && plan.lastRingTasks[recvTask] <= maxRingNorth)     )
	{
	  if(plan.firstRingTasks[recvTask] < minRingNorth)
	    maxRingSend[1] = maxRing;
	  else
	    maxRingSend[1] = 4*Nside - plan.firstRingTasks[recvTask];
	  
	  if(plan.lastRingTasks[recvTask] < maxRingNorth)
	    minRingSend[1] = 4*Nside - plan.lastRingTasks[recvTask];
	  else
	    minRingSend[1] = minRing;
	  
	  for(i=0;i<NmapCells;++i)
	    {
	      ringnum = ring2ringnum(mapCells[i].index,order);
	      if(ringnum < minRingSend[1])
		++sendStart[1];
	      if(ringnum >= minRingSend[1] && ringnum <= maxRingSend[1])
		++Nsend[1];
	    }
	}
    }
#ifdef DEBUG
#if DEBUG_LEVEL > 1
  fprintf(stderr,"%ld -> %ld: min,max = %ld|%ld, first,last = %ld|%ld, minSend,maxSend north/south= %ld|%ld/%ld|%ld, Nsend = %ld|%ld, sendStart = %ld|%ld\n",
	  sendTask,recvTask,
	  minRingTasks[sendTask],maxRingTasks[sendTask],
	  plan.firstRingTasks[recvTask],plan.lastRingTasks[recvTask],
	  minRingSend[0],maxRingSend[0],minRingSend[1],maxRingSend[1],
	  Nsend[0],Nsend[1],sendStart[0],sendStart[1]);
#endif
#endif
}

static int compIndexHEALPixMapCell(const void *p1, const void *p2)
{
  if(((const HEALPixMapCell*)p1)->index > ((const HEALPixMapCell*)p2)->index)
    return 1;
  else if(((const HEALPixMapCell*)p1)->index < ((const HEALPixMapCell*)p2)->index)
    return -1;
  else
    return 0;
}

static int compLong(const void *p1, const void *p2)
{
  if((*((const long*)p1)) > (*((const long*)p2)))
    return 1;
  else if((*((const long*)p1)) < (*((const long*)p2)))
    return -1;
  else
    return 0;
}

