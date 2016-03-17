#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>
#include <fitsio.h>
#include <unistd.h>

#include "raytrace.h"
#include "read_lensplanes_hdf5.h"
#include "read_lensplanes_pixLC.h"

static int compPartNest(const void *a, const void *b)
{
  if(((const Part*)a)->nest > ((const Part*)b)->nest)
    return 1;
  else if(((const Part*)a)->nest < ((const Part*)b)->nest)
    return -1;
  else
    return 0;
}

struct nctg {
  long nest;
  long task;
};

static int compNCTGTask(const void *a, const void *b)
{
  if(((const struct nctg*)a)->task > ((const struct nctg*)b)->task)
    return 1;
  else if(((const struct nctg*)a)->task < ((const struct nctg*)b)->task)
    return -1;
  else
    return 0;
}

/* generic io interface */
void readRayTracingPlaneAtPeanoInds(long planeNum, long HEALPixOrder, long *PeanoIndsToRead, long NumPeanoIndsToRead, Part **LCParts, long *NumLCParts)
{
  void (*read_lens_plane)(long, long, long *, long, Part **, long *) = NULL;
  
  if(strcmp_caseinsens(rayTraceData.LensPlaneType,"HDF5") == 0)
    {
      read_lens_plane = &readRayTracingPlaneAtPeanoInds_HDF5;
    } 
  else if(strcmp_caseinsens(rayTraceData.LensPlaneType,"pixLC") == 0)
    {
      read_lens_plane = &readRayTracingPlaneAtPeanoInds_pixLC;      
    }
  else 
    {
      fprintf(stderr,"%d: readRayTracingPlaneAtPeanoInds - could not find I/O code for lens plane type '%s'!\n",ThisTask,rayTraceData.LensPlaneType);
      MPI_Abort(MPI_COMM_WORLD,666);      
    }
  
  read_lens_plane(planeNum,HEALPixOrder,PeanoIndsToRead,NumPeanoIndsToRead,LCParts,NumLCParts);
}

/* reads light cone particles into bundleCells for the given planeNum */
void read_lcparts_at_planenum(long planeNum)
{
  long i,n;
  long *PeanoIndsToRead;
  long NumPeanoIndsToRead;
    
  long bundleNest;
  double vec[3];
  
  long shift;
  long NumGroups,myGroup,currGroup,readFromPlane;

  double t0;
  
  shift = 2*(HEALPIX_UTILS_MAXORDER-rayTraceData.bundleOrder);
  
  /* set up reading vars
     1) get all cells which are either assigned to this task (bit 0 set) or are buffer cells from which we need particles (bit 1 set)
     2) find their Peano inds for reading from lens planes
     3) make vector which stores how many particles are currently allocated for a given bundle cell - used later for moving parts into bundleCells
  */
  NumPeanoIndsToRead = 0;
  for(i=0;i<NbundleCells;++i)
    if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
      ++NumPeanoIndsToRead;
  PeanoIndsToRead = (long*)malloc(sizeof(long)*NumPeanoIndsToRead);
  assert(PeanoIndsToRead != NULL);
  n = 0;
  for(i=0;i<NbundleCells;++i)
    if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
      {
        PeanoIndsToRead[n] = nest2peano(i,rayTraceData.bundleOrder);
        ++n;
      }
  assert(n == NumPeanoIndsToRead);
  
  /* init and free old parts if needed */
  destroy_parts();
  
  /* read particles from lens plane
     1) uses peano inds computed above
     2) init bundle cells and numPartsAlloc for allocating particle mem
     3) read in groups to limit I/O usage
  */
  NumGroups = NTasks/rayTraceData.NumFilesIOInParallel;
  if(NTasks - NumGroups*rayTraceData.NumFilesIOInParallel > 0)
    ++NumGroups;
  myGroup = ThisTask/rayTraceData.NumFilesIOInParallel;
  readFromPlane = 0;
  
  t0 = -MPI_Wtime();
  for(currGroup=0;currGroup<NumGroups;++currGroup)
    {
      if(currGroup == myGroup)
	{
	  readFromPlane = 1;
	  
	  readRayTracingPlaneAtPeanoInds(planeNum,rayTraceData.bundleOrder,PeanoIndsToRead,NumPeanoIndsToRead,&lensPlaneParts,&NlensPlaneParts);
	  free(PeanoIndsToRead);
	  
	  if(NlensPlaneParts > 0)
	    {
	      /* reorder parts and build indexing for bundleCells */
	      for(i=0;i<NlensPlaneParts;++i)
		{
		  vec[0] = (double) (lensPlaneParts[i].pos[0]);
		  vec[1] = (double) (lensPlaneParts[i].pos[1]);
		  vec[2] = (double) (lensPlaneParts[i].pos[2]);
		  
		  lensPlaneParts[i].nest = vec2nest(vec,HEALPIX_UTILS_MAXORDER);
		}
	      
	      qsort(lensPlaneParts,(size_t) NlensPlaneParts,sizeof(Part),compPartNest);
	      	      
	      /* now fill in index vals in bundleCells */
	      for(i=0;i<NlensPlaneParts;++i)
		{
		  bundleNest = lensPlaneParts[i].nest >> shift;
		  
		  if(bundleCells[bundleNest].Nparts == 0)
		    bundleCells[bundleNest].firstPart = i;
		  bundleCells[bundleNest].Nparts += 1;
		}
	    }
	}
      
      //////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      //////////////////////////////
    }
  
  assert(readFromPlane == 1);
  
  t0 += MPI_Wtime();
  if(ThisTask == 0) 
    {
      fprintf(stderr,"read parts in %f seconds\n",t0);
      fflush(stderr);
    }
  
  //now do an exchange to get buffer parts
  long j,log2NTasks;
  long level,sendTask,recvTask;
  long Nsend,Nrecv;
  long NumBufferCells = 0,NumPartsAlloc,NumBufferParts;
  Part *tmpPart;
  long rpInd;
  struct nctg *nestCellsToGet, *nestCellsToSend=NULL, *tmpNCTG;
  long NnestCellsToSend = 0;
  long firstNestCellForRecvTask,NnestCellsForRecvTask;
  MPI_Status Stat;
  int count,NnestCellsToSendAlloc=0;
  MPI_Request requestSend,requestRecv;
  int didSend,didRecv;

  t0 = -MPI_Wtime();
  
  log2NTasks = 0;
  while(NTasks > (1 << log2NTasks))
    ++log2NTasks;
  
  //get the bundle cells for which parts are needed and which task they are on
  for(i=0;i<NbundleCells;++i)
    if(ISSETBITFLAG(bundleCells[i].active,PARTBUFF_BUNDLECELL))
      ++NumBufferCells;
  
  nestCellsToGet = (struct nctg*)malloc(sizeof(struct nctg)*NumBufferCells);
  assert(nestCellsToGet != NULL);
  NumBufferCells = 0;
  for(i=0;i<NbundleCells;++i)
    if(ISSETBITFLAG(bundleCells[i].active,PARTBUFF_BUNDLECELL))
      {
	nestCellsToGet[NumBufferCells].nest = i;
	rpInd = bundleCellsNest2RestrictedPeanoInd[i];
	nestCellsToGet[NumBufferCells].task = -1;
	if(rpInd != -1)
	  {
	    for(j=0;j<NTasks;++j)
	      {
		if(firstRestrictedPeanoIndTasks[j] <= rpInd && rpInd <= lastRestrictedPeanoIndTasks[j])
		  {
		    nestCellsToGet[NumBufferCells].task = j;
		    break;
		  }
	      }
	    
	    if(nestCellsToGet[NumBufferCells].task == -1)
	      {
		fprintf(stderr,"%d: could not find task for rpInd = %ld\n",ThisTask,rpInd);
		MPI_Abort(MPI_COMM_WORLD,123);
	      }
	  }
	
	++NumBufferCells;
      }
  
  //sort by task
  qsort(nestCellsToGet,(size_t) NumBufferCells,sizeof(struct nctg),compNCTGTask);
  
  //make extra room for incoming parts
  NumPartsAlloc = NlensPlaneParts; 
  NumBufferParts = NlensPlaneParts*((double) NumBufferCells)/((double) NumPeanoIndsToRead)*1.1;
  if(NumBufferParts*sizeof(Part)/1024.0/1024.0 > 100.0)
    NumBufferParts = (100.0*1024.0*1024.0)/sizeof(Part);
  tmpPart = (Part*)realloc(lensPlaneParts,sizeof(Part)*(NlensPlaneParts + NumBufferParts));
  if(tmpPart != NULL)
    {
      lensPlaneParts = tmpPart;
      NumPartsAlloc = NlensPlaneParts + NumBufferParts;
    }
  else
    {
      fprintf(stderr,"%d: could not realloc lensPlaneParts for getting buffer regions! - wanted %lg MB extra for %lf MB total.\n",
	      ThisTask,NumBufferParts*sizeof(Part)/1024.0/1024.0,(NlensPlaneParts + NumBufferParts)*sizeof(Part)/1024.0/1024.0);
      MPI_Abort(MPI_COMM_WORLD,123);
    }
  NumBufferParts = 0;
  
  /*algorithm to loop through pairs of tasks linearly
    -lifted from Gadget-2 under GPL (http://www.gnu.org/copyleft/gpl.html)
    -see pm_periodic.c from Gadget-2 at http://www.mpa-garching.mpg.de/gadget/
  */
  for(level = 0; level < (1 << log2NTasks); level++) /* note: for level=0, target is the same task */
    {
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;
      if(recvTask < NTasks && sendTask != recvTask)  //never need to give parts to ourselves since part buffer cells are not primary cells on same task
        {
	  //first see if we need to get any bundle cells parts from recvTask
	  firstNestCellForRecvTask = -1;
	  NnestCellsForRecvTask = 0;
	  for(j=0;j<NumBufferCells;++j)
	    {
	      if(nestCellsToGet[j].task == recvTask)
		{
		  if(firstNestCellForRecvTask == -1)
		    firstNestCellForRecvTask = j;
		  
		  ++NnestCellsForRecvTask;
		}
	    }
	  
	  if(NnestCellsForRecvTask == 0)
	    firstNestCellForRecvTask = 0;
	  
	  Nsend = NnestCellsForRecvTask;
	  MPI_Sendrecv(&Nsend,1,MPI_LONG,(int) recvTask,TAG_NUMBUFF_PIO,
		       &Nrecv,1,MPI_LONG,(int) recvTask,TAG_NUMBUFF_PIO,
		       MPI_COMM_WORLD,&Stat);
	  NnestCellsToSend = Nrecv;
	  
	  if(Nsend > 0 || Nrecv > 0) //there is overlap between tasks so they may need to send stuff
	    {
	      //get bundle cells for which parts should be sent back
	      if(NnestCellsToSendAlloc < NnestCellsToSend)
		{
		  tmpNCTG = (struct nctg*)realloc(nestCellsToSend,sizeof(struct nctg)*NnestCellsToSend);
		  if(tmpNCTG != NULL)
		    {
		      nestCellsToSend = tmpNCTG;
		      NnestCellsToSendAlloc = NnestCellsToSend;
		    }
		  else
		    {
		      fprintf(stderr,"%d: could not realloc nestCellsToSend!\n",ThisTask);
		      MPI_Abort(MPI_COMM_WORLD,123);
		    }
		}
	      
	      MPI_Sendrecv(nestCellsToGet+firstNestCellForRecvTask,(int) (Nsend*sizeof(struct nctg)),MPI_BYTE,(int) recvTask,TAG_BUFF_PIO,
			   nestCellsToSend,(int) (NnestCellsToSend*sizeof(struct nctg)),MPI_BYTE,(int) recvTask,TAG_BUFF_PIO,
			   MPI_COMM_WORLD,&Stat);
	      
	      //get # of parts to send back
	      Nsend = 0;
	      for(i=0;i<NnestCellsToSend;++i)
		if(ISSETBITFLAG(bundleCells[nestCellsToSend[i].nest].active,PRIMARY_BUNDLECELL))
		  Nsend += bundleCells[nestCellsToSend[i].nest].Nparts;
	      
	      //get # of parts to recv and make sure have room
	      MPI_Sendrecv(&Nsend,1,MPI_LONG,(int) recvTask,TAG_NUMPBUFF_PIO,
			   &Nrecv,1,MPI_LONG,(int) recvTask,TAG_NUMPBUFF_PIO,
			   MPI_COMM_WORLD,&Stat);
	      
	      if(NlensPlaneParts+NumBufferParts+Nrecv > NumPartsAlloc)
		{
		  tmpPart = (Part*)realloc(lensPlaneParts,sizeof(Part)*(NumPartsAlloc+Nrecv*4));
		  if(tmpPart != NULL)
		    {
		      lensPlaneParts = tmpPart;
		      NumPartsAlloc += Nrecv*4;
		    }
		  else
		    {
		      fprintf(stderr,"%d: could not realloc lensPlaneParts for getting buffer regions in loop! - wanted %lg MB extra for %lf MB total.\n",
			      ThisTask,Nrecv*4*sizeof(Part)/1024.0/1024.0,(NumPartsAlloc+Nrecv*4)*sizeof(Part)/1024.0/1024.0);
		      MPI_Abort(MPI_COMM_WORLD,123);
		    }
		}
	      
	      //if have parts to send or recv, do it
	      if(Nsend > 0 || Nrecv > 0)
		{
		  i = 0;
		  while(Nsend > 0 || Nrecv > 0)
		    {
		      if(Nrecv > 0) //recv parts
			{
			  MPI_Irecv(lensPlaneParts+NlensPlaneParts+NumBufferParts,
				    (int) (sizeof(Part)*(NumPartsAlloc-(NlensPlaneParts+NumBufferParts))),MPI_BYTE,
				    (int) recvTask,TAG_PBUFF_PIO,MPI_COMM_WORLD,&requestRecv);
			  didRecv = 1;
			}
		      else
			didRecv = 0;
		      
		      if(Nsend > 0)
			{
			  while(i < NnestCellsToSend && !(ISSETBITFLAG(bundleCells[nestCellsToSend[i].nest].active,PRIMARY_BUNDLECELL) && bundleCells[nestCellsToSend[i].nest].Nparts > 0))
			    ++i;
			  
			  if(i >= NnestCellsToSend)
			    {
			      fprintf(stderr,"%d: out of nest cells in while Nsend, Nrecv loop\n",ThisTask);
			      MPI_Abort(MPI_COMM_WORLD,123);
			    }
			  
			  MPI_Issend(lensPlaneParts+bundleCells[nestCellsToSend[i].nest].firstPart,
				     (int) (sizeof(Part)*bundleCells[nestCellsToSend[i].nest].Nparts),MPI_BYTE,
				     (int) recvTask,TAG_PBUFF_PIO,MPI_COMM_WORLD,&requestSend);
			  
			  Nsend -= bundleCells[nestCellsToSend[i].nest].Nparts;
			  didSend = 1;
			  ++i;
			}
		      else
			didSend = 0;
		      
		      if(didRecv)
			{
			  MPI_Wait(&requestRecv,&Stat);
			  
			  MPI_Get_count(&Stat,MPI_BYTE,&count);
			  count /= sizeof(Part);
			  NumBufferParts += count;			  
			  Nrecv -= count;
			}
		      
		      if(didSend)
			MPI_Wait(&requestSend,&Stat);
		    }
		}
	    }
	}
    }
  
  //update total number of parts
  NlensPlaneParts += NumBufferParts; 

  //clean up
  if(NnestCellsToSendAlloc > 0)
    free(nestCellsToSend);
  free(nestCellsToGet);
  
  t0 += MPI_Wtime();
  if(ThisTask == 0) 
    {
      fprintf(stderr,"got buffer parts in %f seconds\n",t0);
      fflush(stderr);
    }

  //if using the full light cone part distribution,
  //need to read in buffer parts which are not primary cells in the domain
#ifdef USE_FULLSKY_PARTDIST
  Part *buffParts = NULL;
  long NumBuffParts = 0;
  long peanoInd;
  
  t0 = -MPI_Wtime();

  NumPeanoIndsToRead = 1;
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,PARTBUFF_BUNDLECELL) && bundleCellsNest2RestrictedPeanoInd[i] < 0)
	{
	  //read parts from file
	  peanoInd = nest2peano(i,rayTraceData.bundleOrder);	  
          readRayTracingPlaneAtPeanoInds(planeNum,rayTraceData.bundleOrder,&peanoInd,NumPeanoIndsToRead,&buffParts,&NumBuffParts);
	  
	  //now add to current parts vector if needed
	  if(NumBuffParts > 0)
	    {
	      //make sure have mem
	      if(NlensPlaneParts+NumBuffParts > NumPartsAlloc)
		{
		  tmpPart = (Part*)realloc(lensPlaneParts,sizeof(Part)*(NumPartsAlloc+NumBuffParts*4));
		  if(tmpPart != NULL)
		    {
		      lensPlaneParts = tmpPart;
		      NumPartsAlloc += NumBuffParts*4;
		    }
		  else
		    {
		      fprintf(stderr,"%d: could not realloc lensPlaneParts for getting USE_FULLSKY_PARTDIST buffer regions in loop! - wanted %lg MB extra for %lf MB total.\n",
			      ThisTask,NumBuffParts*4*sizeof(Part)/1024.0/1024.0,(NumPartsAlloc+NumBuffParts*4)*sizeof(Part)/1024.0/1024.0);
		      MPI_Abort(MPI_COMM_WORLD,123);
		    }
		}
	      
	      //add to code
	      for(j=0;j<NumBuffParts;++j)
                {
                  vec[0] = (double) (buffParts[j].pos[0]);
                  vec[1] = (double) (buffParts[j].pos[1]);
                  vec[2] = (double) (buffParts[j].pos[2]);
		  buffParts[j].nest = vec2nest(vec,HEALPIX_UTILS_MAXORDER);
		  
		  lensPlaneParts[NlensPlaneParts+j] = buffParts[j];
                }
              NlensPlaneParts += NumBuffParts;
	      
	      //clean up
	      free(buffParts);
	      buffParts = NULL;
	      NumBuffParts = 0;
	    }
	}
    }
  
  t0 += MPI_Wtime();
  if(ThisTask == 0) 
    {
      fprintf(stderr,"got fullsky buffer parts in %f seconds\n",t0);
      fflush(stderr);
    }
#endif
  
  //free extra mem
  if(NlensPlaneParts < NumPartsAlloc)
    {
      NumPartsAlloc = NlensPlaneParts;
      tmpPart = (Part*)realloc(lensPlaneParts,sizeof(Part)*NlensPlaneParts);
      if(tmpPart != NULL)
	{
	  lensPlaneParts = tmpPart;
	}
      else
	{
	  fprintf(stderr,"%d: could not do final realloc lensPlaneParts for getting buffer regions!\n",ThisTask);
	  MPI_Abort(MPI_COMM_WORLD,123);
	}
    }
  
  //redo index vals in bundleCells
  for(i=0;i<NbundleCells;++i)
    {
      bundleCells[i].Nparts = 0;
      bundleCells[i].firstPart = -1;
    }
  qsort(lensPlaneParts,(size_t) NlensPlaneParts,sizeof(Part),compPartNest);
  shift = 2*(HEALPIX_UTILS_MAXORDER-rayTraceData.bundleOrder);
  for(i=0;i<NlensPlaneParts;++i)
    {
      bundleNest = lensPlaneParts[i].nest >> shift;

      if(bundleCells[bundleNest].Nparts == 0)
	bundleCells[bundleNest].firstPart = i;
      bundleCells[bundleNest].Nparts += 1;
    }
}

/* reads light cone particles into bundleCells for the given planeNum */
void read_lcparts_at_planenum_fullsky_partdist(long planeNum)
{
  long i,n;
  long *PeanoIndsToRead;
  long NumPeanoIndsToRead;
    
  long bundleNest;
  double vec[3];
  
  long shift;
  long NumGroups,myGroup,currGroup,readFromPlane;
  
  double t0;
  
  shift = 2*(HEALPIX_UTILS_MAXORDER - rayTraceData.bundleOrder);
  
  /* set up reading vars
     1) get all cells which are either assigned to this task (bit 0 set) or are buffer cells from which we need particles (bit 1 set)
     2) find their Peano inds for reading from lens planes
     3) make vector which stores how many particles are currently allocated for a given bundle cell - used later for moving parts into bundleCells
  */
  NumPeanoIndsToRead = 0;
  for(i=0;i<NbundleCells;++i)
    if(ISSETBITFLAG(bundleCells[i].active,FULLSKY_PARTDIST_PRIMARY_BUNDLECELL))
      ++NumPeanoIndsToRead;
  PeanoIndsToRead = (long*)malloc(sizeof(long)*NumPeanoIndsToRead);
  assert(PeanoIndsToRead != NULL);
  n = 0;
  for(i=0;i<NbundleCells;++i)
    if(ISSETBITFLAG(bundleCells[i].active,FULLSKY_PARTDIST_PRIMARY_BUNDLECELL))
      {
        PeanoIndsToRead[n] = nest2peano(i,rayTraceData.bundleOrder);
        ++n;
      }
  assert(n == NumPeanoIndsToRead);
    
  /* init and free old parts if needed */
  destroy_parts();
  
  /* read particles from lens plane
     1) uses peano inds computed above
     2) init bundle cells and numPartsAlloc for allocating particle mem
     3) read in groups to limit I/O usage
  */
  NumGroups = NTasks/rayTraceData.NumFilesIOInParallel;
  if(NTasks - NumGroups*rayTraceData.NumFilesIOInParallel > 0)
    ++NumGroups;
  myGroup = ThisTask/rayTraceData.NumFilesIOInParallel;
  readFromPlane = 0;
  
  t0 = -MPI_Wtime();
  for(currGroup=0;currGroup<NumGroups;++currGroup)
    {
      if(currGroup == myGroup)
	{
	  readFromPlane = 1;
	  
	  readRayTracingPlaneAtPeanoInds(planeNum,rayTraceData.bundleOrder,PeanoIndsToRead,NumPeanoIndsToRead,&lensPlaneParts,&NlensPlaneParts);
	  free(PeanoIndsToRead);
	  
	  if(NlensPlaneParts > 0)
	    {
	      /* reorder parts and build indexing for bundleCells */
	      for(i=0;i<NlensPlaneParts;++i)
		{
		  vec[0] = (double) (lensPlaneParts[i].pos[0]);
		  vec[1] = (double) (lensPlaneParts[i].pos[1]);
		  vec[2] = (double) (lensPlaneParts[i].pos[2]);
		
		  lensPlaneParts[i].nest = vec2nest(vec,HEALPIX_UTILS_MAXORDER);
                }

              qsort(lensPlaneParts,(size_t) NlensPlaneParts,sizeof(Part),compPartNest);

              /* now fill in index vals in bundleCells */
              for(i=0;i<NlensPlaneParts;++i)
                {
                  bundleNest = lensPlaneParts[i].nest >> shift;

                  if(bundleCells[bundleNest].Nparts == 0)
                    bundleCells[bundleNest].firstPart = i;
                  bundleCells[bundleNest].Nparts += 1;
                }
	    }
	}
      
      //////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      //////////////////////////////
    }
  
  assert(readFromPlane == 1);
  
  t0 += MPI_Wtime();
  if(ThisTask == 0) 
    {
      fprintf(stderr,"read parts in %f seconds\n",t0);
      fflush(stderr);
    }
}

/* reads light cone particles into bundleCells for the given planeNum */
void read_lcparts_at_planenum_all(long planeNum)
{
  long i,n;
  long *PeanoIndsToRead;
  long NumPeanoIndsToRead;
    
  long bundleNest;
  double vec[3];
  
  long shift;
  long NumGroups,myGroup,currGroup,readFromPlane;
  
  double t0;
  
  shift = 2*(HEALPIX_UTILS_MAXORDER - rayTraceData.bundleOrder);
  
  /* set up reading vars
     1) get all cells which are either assigned to this task (bit 0 set) or are buffer cells from which we need particles (bit 1 set)
     2) find their Peano inds for reading from lens planes
     3) make vector which stores how many particles are currently allocated for a given bundle cell - used later for moving parts into bundleCells
  */
  NumPeanoIndsToRead = NbundleCells;
  PeanoIndsToRead = (long*)malloc(sizeof(long)*NumPeanoIndsToRead);
  assert(PeanoIndsToRead != NULL);
  n = 0;
  for(i=0;i<NbundleCells;++i)
    {
        PeanoIndsToRead[n] = nest2peano(i,rayTraceData.bundleOrder);
        ++n;
    }
  assert(n == NumPeanoIndsToRead);
  
  /* init and free old parts if needed */
  destroy_parts();
    
  /* read particles from lens plane
     1) uses peano inds computed above
     2) init bundle cells and numPartsAlloc for allocating particle mem
     3) read in groups to limit I/O usage
  */
  NumGroups = NTasks/rayTraceData.NumFilesIOInParallel;
  if(NTasks - NumGroups*rayTraceData.NumFilesIOInParallel > 0)
    ++NumGroups;
  myGroup = ThisTask/rayTraceData.NumFilesIOInParallel;
  readFromPlane = 0;
  
  t0 = -MPI_Wtime();
  for(currGroup=0;currGroup<NumGroups;++currGroup)
    {
      if(currGroup == myGroup)
	{
	  readFromPlane = 1;
	  
	  readRayTracingPlaneAtPeanoInds(planeNum,rayTraceData.bundleOrder,PeanoIndsToRead,NumPeanoIndsToRead,&lensPlaneParts,&NlensPlaneParts);
	  free(PeanoIndsToRead);
	  
	  if(NlensPlaneParts > 0)
	    {
	      /* reorder parts and build indexing for bundleCells */
	      for(i=0;i<NlensPlaneParts;++i)
		{
		  vec[0] = (double) (lensPlaneParts[i].pos[0]);
		  vec[1] = (double) (lensPlaneParts[i].pos[1]);
		  vec[2] = (double) (lensPlaneParts[i].pos[2]);
		
		  lensPlaneParts[i].nest = vec2nest(vec,HEALPIX_UTILS_MAXORDER);
                }

              qsort(lensPlaneParts,(size_t) NlensPlaneParts,sizeof(Part),compPartNest);

              /* now fill in index vals in bundleCells */
              for(i=0;i<NlensPlaneParts;++i)
                {
                  bundleNest = lensPlaneParts[i].nest >> shift;

                  if(bundleCells[bundleNest].Nparts == 0)
                    bundleCells[bundleNest].firstPart = i;
                  bundleCells[bundleNest].Nparts += 1;
                }
	    }
	}
      
      //////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      //////////////////////////////
    }
  
  assert(readFromPlane == 1);
  
  t0 += MPI_Wtime();
  if(ThisTask == 0) 
    {
      fprintf(stderr,"read parts in %f seconds\n",t0);
      fflush(stderr);
    }
}
