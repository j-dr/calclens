#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <fitsio.h>

#include "raytrace.h"

static const char *ttype[] = { "index", "ra", "dec", "A00", "A01", "A10", "A11"};
static const char *tform[] = { "K", "D", "D", "D", "D", "D", "D"};

static void file_write_gals2fits(long fileNum, long firstTask, long lastTask, MPI_Comm fileComm);
static void get_gal_iodecomp(long *firstTaskFiles, long *lastTaskFiles, long *fileNum);
static int read_posindex_galcat(char finname[MAX_FILENAME], long *NumBuffGals, SourceGal **buffGals);
static void distribute_gals_to_tasks(SourceGal *buffGals, int *sendCounts, int *displs);

void write_gals2fits(void)
{
  long group,NumGroups,i;
  long *firstTaskFiles,*lastTaskFiles,fileNum=-1;
  MPI_Comm fileComm;
  MPI_Group worldGroup,fileGroup;
  int *ranks,Nranks;
  double t;
  
  t = -MPI_Wtime();
  
  firstTaskFiles = (long*)malloc(sizeof(long)*rayTraceData.NumGalOutputFiles);
  assert(firstTaskFiles != NULL);
  lastTaskFiles = (long*)malloc(sizeof(long)*rayTraceData.NumGalOutputFiles);
  assert(lastTaskFiles != NULL);
  get_gal_iodecomp(firstTaskFiles,lastTaskFiles,&fileNum);

  /*make communicators for each file*/
  MPI_Comm_group(MPI_COMM_WORLD,&worldGroup);
  Nranks = lastTaskFiles[fileNum]-firstTaskFiles[fileNum]+1;
  ranks = (int*)malloc(sizeof(int)*Nranks);
  assert(ranks != NULL);
  for(i=0;i<Nranks;++i)
    ranks[i] = firstTaskFiles[fileNum] + i;
  MPI_Group_incl(worldGroup,Nranks,ranks,&fileGroup);
  MPI_Comm_create(MPI_COMM_WORLD,fileGroup,&fileComm);

  NumGroups = rayTraceData.NumGalOutputFiles/rayTraceData.NumFilesIOInParallel;
  if(rayTraceData.NumGalOutputFiles - NumGroups*rayTraceData.NumFilesIOInParallel > 0)
    ++NumGroups;
  for(group=0;group<NumGroups;++group)
    {
      if(fileNum/rayTraceData.NumFilesIOInParallel == group)
	file_write_gals2fits(fileNum,firstTaskFiles[fileNum],lastTaskFiles[fileNum],fileComm);

      /*\\\\\\\\\\\\\\\\\\\\\\\\\\*/
      MPI_Barrier(MPI_COMM_WORLD);
      /*\\\\\\\\\\\\\\\\\\\\\\\\\\*/
    }

  free(firstTaskFiles);
  free(lastTaskFiles);
  free(ranks);
  MPI_Comm_free(&fileComm);
  MPI_Group_free(&fileGroup);
  MPI_Group_free(&worldGroup);
  
  t += MPI_Wtime();
  
  if(ThisTask == 0)
    fprintf(stderr,"writing image gals to disk took %lf seconds.\n",t);
}

static void file_write_gals2fits(long fileNum, long firstTask, long lastTask, MPI_Comm fileComm)
{
  char name[MAX_FILENAME];
  char bangname[MAX_FILENAME];
  fitsfile *fptr;
  int tfields,status;
  long i,j;
  LONGLONG nrows,firstrow,firstelem,nelements;
  long chunkInd,firstInd,lastInd,NumGalsInChunkBase,NumGalsInChunk,NumChunks;
  char *buff;
  double *darr;
  long *larr;
  int colnum;
  double t0 = 0.0;
  
  long NumTotToRecv,NumWritten=0,NumToWrite;
  MPI_Status mpistatus;
  
  sprintf(name,"%s/%s%04ld.%04ld.fit",rayTraceData.OutputPath,rayTraceData.GalOutputName,rayTraceData.CurrentPlaneNum,fileNum);
  sprintf(bangname,"!%s",name);
  
  status = 0;
  firstrow = 1;
  firstelem = 1;
  
  if(ThisTask == firstTask) //make the file on disk
    {
      t0 = -MPI_Wtime();
      
      remove(name);
      
      sprintf(bangname,"!%s",name);
      
      fits_create_file(&fptr,bangname,&status);
      if(status)
        fits_report_error(stderr,status);
      
      nrows = 0;
      tfields = 7;
      fits_create_tbl(fptr,BINARY_TBL,nrows,tfields,ttype,tform,NULL,NULL,&status);
      if(status)
        fits_report_error(stderr,status);
      
      fits_get_rowsize(fptr,&NumGalsInChunkBase,&status);
      if(status)
	fits_report_error(stderr,status);
    }
  
  MPI_Bcast(&NumGalsInChunkBase,1,MPI_LONG,0,fileComm);
  if(sizeof(long) > sizeof(double))
    buff = (char*)malloc(sizeof(long)*NumGalsInChunkBase);
  else
    buff = (char*)malloc(sizeof(double)*NumGalsInChunkBase);
  assert(buff != NULL);
  darr = (double*) buff;
  larr = (long*) buff;
  
  for(i=firstTask;i<=lastTask;++i)
    {
      if(ThisTask == i)
        {

#ifdef DEBUG
#if DEBUG_LEVEL > 0
          fprintf(stderr,"%d: fileNum = %ld, first,last = %ld|%ld\n",ThisTask,fileNum,firstTask,lastTask);
#endif
#endif
	  if(ThisTask != firstTask)
            MPI_Send(&NumImageGalsGlobal,1,MPI_LONG,(int) firstTask,TAG_GALSIO_TOTNUM,MPI_COMM_WORLD);
	  
	  //now write gals to file in chunks
	  if(NumImageGalsGlobal > 0)
	    {
	      NumChunks = NumImageGalsGlobal/NumGalsInChunkBase;
	      if(NumChunks*NumGalsInChunkBase < NumImageGalsGlobal)
		NumChunks += 1;
	      
#ifdef DEBUG
#if DEBUG_LEVEL > 0
	      fprintf(stderr,"%d: NumImageGalsGlobal = %ld, NumGalsInChunkBase = %ld, NumChunks = %ld\n",ThisTask,
		      NumImageGalsGlobal,NumGalsInChunkBase,NumChunks);
#endif
#endif
	      
	      for(chunkInd=0;chunkInd<NumChunks;++chunkInd)
		{
		  firstInd = chunkInd*NumGalsInChunkBase;
		  lastInd = (chunkInd+1)*NumGalsInChunkBase-1;
		  if(lastInd >= NumImageGalsGlobal-1)
		    lastInd = NumImageGalsGlobal-1;
		  NumGalsInChunk = lastInd - firstInd + 1;
		  		  
		  if(ThisTask != firstTask)
		    {
		      MPI_Send(&NumGalsInChunk,1,MPI_LONG,(int) firstTask,TAG_GALSIO_NUMCHUNK,MPI_COMM_WORLD);
		      colnum = TAG_GALSIO_CHUNKDATA;
		      
		      for(j=firstInd;j<=lastInd;++j)
			larr[j-firstInd] = ImageGalsGlobal[j].index;
		      MPI_Ssend(larr,(int) NumGalsInChunk,MPI_LONG,(int) firstTask,colnum,MPI_COMM_WORLD);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].ra;
		      MPI_Ssend(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].dec;
		      MPI_Ssend(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].A00;
		      MPI_Ssend(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].A01;
		      MPI_Ssend(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].A10;
		      MPI_Ssend(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].A11;
		      MPI_Ssend(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
		    }
		  else
		    {
		      nelements = (LONGLONG) NumGalsInChunk;
		      firstelem = 1;
		      
		      colnum = 1;
		      for(j=firstInd;j<=lastInd;++j)
			larr[j-firstInd] = ImageGalsGlobal[j].index;
		      fits_write_col(fptr,TLONG,colnum,firstrow,firstelem,nelements,larr,&status);
		      if(status)
			fits_report_error(stderr,status);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].ra;
		      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
		      if(status)
			fits_report_error(stderr,status);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].dec;
		      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
		      if(status)
			fits_report_error(stderr,status);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].A00;
		      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
		      if(status)
			fits_report_error(stderr,status);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].A01;
		      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
		      if(status)
			fits_report_error(stderr,status);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].A10;
		      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
		      if(status)
			fits_report_error(stderr,status);
		      ++colnum;
		      
		      for(j=firstInd;j<=lastInd;++j)
			darr[j-firstInd] = ImageGalsGlobal[j].A11;
		      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
		      if(status)
			fits_report_error(stderr,status);
		      ++colnum;
		      
		      firstrow += nelements;
		      NumWritten += NumGalsInChunk;
		    }// else of if(ThisTask != firstTask)
		}// for(chunkInd=0;chunkInd<NumChunks;++chunkInd)
	    } // if NumImageGalsGlobal > 0
	} //if ThisTask == i
      
      if(i != firstTask && ThisTask == firstTask)
        {
	  MPI_Recv(&NumTotToRecv,1,MPI_LONG,(int) i,TAG_GALSIO_TOTNUM,MPI_COMM_WORLD,&mpistatus);
	  
	  while(NumTotToRecv > 0)
            {
	      MPI_Recv(&NumGalsInChunk,1,MPI_LONG,(int) i,TAG_GALSIO_NUMCHUNK,MPI_COMM_WORLD,&mpistatus);
	      nelements = (LONGLONG) NumGalsInChunk;
	      firstelem = 1;
	      colnum = 1;
	      
	      MPI_Recv(larr,(int) NumGalsInChunk,MPI_LONG,(int) i,colnum+TAG_GALSIO_CHUNKDATA-1,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TLONG,colnum,firstrow,firstelem,nelements,larr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) i,colnum+TAG_GALSIO_CHUNKDATA-1,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) i,colnum+TAG_GALSIO_CHUNKDATA-1,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) i,colnum+TAG_GALSIO_CHUNKDATA-1,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
		      
	      MPI_Recv(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) i,colnum+TAG_GALSIO_CHUNKDATA-1,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) i,colnum+TAG_GALSIO_CHUNKDATA-1,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumGalsInChunk,MPI_DOUBLE,(int) i,colnum+TAG_GALSIO_CHUNKDATA-1,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      firstrow += nelements;
	      NumWritten += NumGalsInChunk;
	      NumTotToRecv -= NumGalsInChunk;
	    }
	}
      
      //////////////////////////////
      MPI_Barrier(fileComm);
      //////////////////////////////
      
    } // loop through tasks

  if(ThisTask == firstTask) //make the file on disk
    {
      fits_close_file(fptr,&status);
      if(status)
        fits_report_error(stderr,status);
      
      t0 += MPI_Wtime();
#ifdef DEBUG
      fprintf(stderr,"writing %ld gals to file '%s' took %g seconds.\n",NumWritten,name,t0);
#endif
    }
  
  //error check
  MPI_Allreduce(&NumImageGalsGlobal,&NumToWrite,1,MPI_LONG,MPI_SUM,fileComm);
  if(ThisTask == firstTask)
    assert(NumWritten == NumToWrite);
  
  //clean it all up
  free(buff);
}

/* gets I/O decomp given number of Tasks, and the number of output files wanted
   inspired by Gadget-2
*/
static void get_gal_iodecomp(long *firstTaskFiles, long *lastTaskFiles, long *fileNum)
{
  long i,j;
  long numTasksPerFile,numExtraTasks;

  numTasksPerFile = NTasks/rayTraceData.NumGalOutputFiles;
  numExtraTasks = NTasks - numTasksPerFile*rayTraceData.NumGalOutputFiles;

  j = 0;
  for(i=0;i<rayTraceData.NumGalOutputFiles;++i)
    {
      firstTaskFiles[i] = j;
      if(i < numExtraTasks)
        lastTaskFiles[i] = numTasksPerFile + j;
      else
        lastTaskFiles[i] = numTasksPerFile + j - 1;

      j = lastTaskFiles[i] + 1;
    }

  for(i=0;i<rayTraceData.NumGalOutputFiles;++i)
    if(firstTaskFiles[i] <= ThisTask && ThisTask <= lastTaskFiles[i])
      *fileNum = i;

#ifdef DEBUG
#if DEBUG_LEVEL > 0
  if(ThisTask == 0)
    {
      fprintf(stderr,"gal I/O: file decomp info - # of files = %ld, NTasks = %d, numTasksPerFile = %ld, numExtraTasks = %ld\n"
              ,rayTraceData.NumGalOutputFiles,NTasks,numTasksPerFile,numExtraTasks);
      for(i=0;i<rayTraceData.NumGalOutputFiles;++i)
        {
          fprintf(stderr,"%ld: firstTaskFiles,lastTaskFiles = %ld|%ld|%ld\n",
                  i,firstTaskFiles[i],lastTaskFiles[i],lastTaskFiles[i]-firstTaskFiles[i]+1);
        }
      fprintf(stderr,"\n");
    }
#endif
#endif
}

void read_fits2gals(void)
{
  FILE *fp;
  long NumGalFiles;
  long NumIORounds,round,NumTasksPerIORound;
  long fileNumToRead;
  int GalFileDone,AllGalFilesDone;
  char fname[MAX_FILENAME];
  SourceGal *buffGals;
  long NumBuffGals,i;
  int *sendCounts,*displs;
  int numNotInDomain;
  long totNumGalaxiesOutsideDomain;
  long GlobalNumGalaxiesOutsideDomain;
  long totNumGalaxiesInsideDomain;
  long GlobalNumGalaxiesInsideDomain;
  
  /*
    reads in all gals and sorts them according to task
    
    only read NumTasksPerIORound files at a time out of NumGalFiles
    do this in NumIORounds rounds
    
    during each round, a file is read in chunks and GalFileDone = 1 when file on a given task is done
    when AllGalFilesDone is 1 for all nodes, then round is finished
    
    the function distribute_gals_to_tasks sends correct gals to each task
  */
  
  totNumGalaxiesOutsideDomain = 0;
  totNumGalaxiesInsideDomain = 0;
  
  sendCounts = (int*)malloc(sizeof(int)*NTasks);
  assert(sendCounts != NULL);
  displs = (int*)malloc(sizeof(int)*NTasks);
  assert(displs != NULL);
  
  if(ThisTask == 0)
    {
      fp = fopen(rayTraceData.GalsFileList,"r");
      assert(fp != NULL);
      NumGalFiles = fnumlines(fp);
      
      fprintf(stderr,"found %ld galaxy files to read.\n",NumGalFiles);
      
      fclose(fp);
    }
  MPI_Bcast(&NumGalFiles,1,MPI_LONG,0,MPI_COMM_WORLD);
  
  NumTasksPerIORound = rayTraceData.NumFilesIOInParallel;
  NumIORounds = NumGalFiles/NumTasksPerIORound;
  if(NumTasksPerIORound*NumIORounds != NumGalFiles)
    ++NumIORounds;
  
  for(round=0;round<NumIORounds;++round)
    {
      if(ThisTask == 0)
	fprintf(stderr,"doing gals I/O round %ld of %ld.\n",round+1,NumIORounds);
      
      GalFileDone = 0;
      do {
	
	//get file num to read
	fileNumToRead = round*NumTasksPerIORound+ThisTask;
	if(ThisTask < NumTasksPerIORound && fileNumToRead < NumGalFiles)
	  {
	    //get file name to read from
	    fp = fopen(rayTraceData.GalsFileList,"r");
	    assert(fp != NULL);
	    for(i=0;i<fileNumToRead+1;++i)
	      {
		fgets(fname,MAX_FILENAME,fp);
	      }
	    fclose(fp);
	    if(fname[strlen(fname)-1] == '\n')
	      fname[strlen(fname)-1] = '\0';
	    
	    //fprintf(stderr,"%d: before fileNumToRed = %ld, name = '%s'\n",ThisTask,fileNumToRead,fname);
	    
	    //read from file
	    if(!GalFileDone)
	      GalFileDone = read_posindex_galcat(fname,&NumBuffGals,&buffGals);
	    
	    //fprintf(stderr,"%d: after # of gals = %ld, done = %d\n",ThisTask,NumBuffGals,GalFileDone);
	    
	    //this index is nice because given any gal index and the number of files
	    //you can get back the position of the galaxy in the file and the file that has it
	    for(i=0;i<NumBuffGals;++i)  
	      buffGals[i].index = fileNumToRead + NumGalFiles*buffGals[i].index;
	      
	    //sort gals accroding to task
	    numNotInDomain = reorder_gals_for_tasks(NumBuffGals,buffGals,sendCounts);
	    totNumGalaxiesOutsideDomain += numNotInDomain;
	    totNumGalaxiesInsideDomain += (NumBuffGals - numNotInDomain);
	    
	    //fill in displs and sendCounts
	    displs[0] = numNotInDomain;
	    for(i=1;i<NTasks;++i)
	      displs[i] = displs[i-1] + sendCounts[i-1];
	  }
	else
	  {
	    fileNumToRead = -1;
	    GalFileDone = 1;
	    NumBuffGals = 0;
	    buffGals = NULL;
	    for(i=0;i<NTasks;++i)
	      {
		displs[i] = 0;
		sendCounts[i] = 0;
	      }
	  }
	
	distribute_gals_to_tasks(buffGals,sendCounts,displs);
	
	if(NumBuffGals > 0)
	  {
	    free(buffGals);
	    NumBuffGals = 0;
            buffGals = NULL;
	  }
	
	MPI_Allreduce(&GalFileDone,&AllGalFilesDone,1,MPI_INT,MPI_LAND,MPI_COMM_WORLD); 
	
      } while(!AllGalFilesDone);
    }
  
  //fprintf(stderr,"%d: befor final realloc! NumSourceGalsGlobal = %ld\n",ThisTask);
  
  //realloc mem
  if(NumSourceGalsGlobal > 0)
    {
      SourceGal *tmpSourceGals = (SourceGal*)realloc(SourceGalsGlobal,sizeof(SourceGal)*NumSourceGalsGlobal);
      assert(tmpSourceGals != NULL);
      SourceGalsGlobal = tmpSourceGals;
    }
  else if(SourceGalsGlobal != NULL)
    {
      free(SourceGalsGlobal);
      SourceGalsGlobal = NULL;
      NumSourceGalsGlobal = 0;
    }
  //fprintf(stderr,"%d: after final realloc!\n",ThisTask);
  
  if(NumSourceGalsGlobal > 0)
    reorder_gals_for_planes();
  
  MPI_Allreduce(&totNumGalaxiesOutsideDomain,&GlobalNumGalaxiesOutsideDomain,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD); 
  MPI_Allreduce(&totNumGalaxiesInsideDomain,&GlobalNumGalaxiesInsideDomain,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD); 
  MPI_Allreduce(&NumSourceGalsGlobal,&i,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD); 
  
  assert(i == GlobalNumGalaxiesInsideDomain);
  
  if(ThisTask == 0)
    fprintf(stderr,"found %ld galaxies inside domain and %ld galaxies outside of domain.\n\n",GlobalNumGalaxiesInsideDomain,GlobalNumGalaxiesOutsideDomain);
  
  //clean up
  free(sendCounts);
  free(displs);
}

static void distribute_gals_to_tasks(SourceGal *buffGals, int *sendCounts, int *displs)
{
  int log2NTasks;
  int level,sendTask,recvTask;
  long Nsend,Nrecv,i;
  MPI_Status Stat;
  SourceGal *tmpSourceGals;
  static long initFlag = 1,MaxNumSourceGalsGlobal;
  
  if(initFlag)
    {
      initFlag = 0;
      MaxNumSourceGalsGlobal = (long) (100.0*1024.0*1024.0/sizeof(SourceGal));
      SourceGalsGlobal = (SourceGal*)malloc(sizeof(SourceGal)*MaxNumSourceGalsGlobal);
      assert(SourceGalsGlobal != NULL);
    }
  
  log2NTasks = 0;
  while(NTasks > (1 << log2NTasks))
    ++log2NTasks;
  
  /*algorithm to loop through pairs of tasks linearly
    -lifted from Gadget-2 under GPL (http://www.gnu.org/copyleft/gpl.html)
    -see pm_periodic.c from Gadget-2 at http://www.mpa-garching.mpg.de/gadget/
  */
  for(level = 0; level < (1 << log2NTasks); level++) /* note: for level=0, target is the same task */
    {
#ifdef DEBUG
#if DEBUG_LEVEL > 1
      if(ThisTask == 0)
        fprintf(stderr,"level = %d of %d\n",level,(1 << log2NTasks));
#endif
#endif
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;
      if(recvTask < NTasks)
        {
          //get send and recv counts
          Nsend = (long) (sendCounts[recvTask]);
          MPI_Sendrecv(&Nsend,1,MPI_LONG,recvTask,TAG_NUMBUFF_SOURCEGAL,
                       &Nrecv,1,MPI_LONG,recvTask,TAG_NUMBUFF_SOURCEGAL,
                       MPI_COMM_WORLD,&Stat);
          
          //now do send or recv if needed
          if(Nsend > 0 || Nrecv > 0)
            {
              //make sure have enough mem
              if(Nrecv + NumSourceGalsGlobal >= MaxNumSourceGalsGlobal)
                {
		  //fprintf(stderr,"%d: realloc - curr = %ld, max = %ld, recv = %ld\n",ThisTask,
		  //NumSourceGalsGlobal,MaxNumSourceGalsGlobal,Nrecv);
		  
		  tmpSourceGals = (SourceGal*)realloc(SourceGalsGlobal,sizeof(SourceGal)*(MaxNumSourceGalsGlobal + 5*Nrecv));
		  assert(tmpSourceGals != NULL);
		  SourceGalsGlobal = tmpSourceGals;
		  MaxNumSourceGalsGlobal += 5*Nrecv;
		}                                                                          
              
              if(sendTask != recvTask)
                {
                  MPI_Sendrecv(buffGals+displs[recvTask],(int) (Nsend*sizeof(SourceGal)),MPI_BYTE,recvTask,TAG_BUFF_SOURCEGAL,
                               SourceGalsGlobal+NumSourceGalsGlobal,(int) (Nrecv*sizeof(SourceGal)),MPI_BYTE,recvTask,TAG_BUFF_SOURCEGAL,
                               MPI_COMM_WORLD,&Stat);
		}
              else
                {
                  //if sendTask == recvTask then just copy over parts
                  for(i=0;i<Nsend;++i)
		    SourceGalsGlobal[i+NumSourceGalsGlobal] = buffGals[i+displs[recvTask]];
		}
	      
	      NumSourceGalsGlobal += Nrecv;
            }
        }
    }
}

static int read_posindex_galcat(char finname[MAX_FILENAME], long *NumBuffGals, SourceGal **buffGals)
{
  static long currGalChunk = -1;
  static long NumGalChunks,BaseNumGalsInChunk,currNumGals,TotNumGals;
  static float *fbuff;
  static fitsfile *fptr;
  static int status;
  
  long NumGalsRead;
  long MaxNumGalsToRead = (long) (100.0*1024.0*1024.0/sizeof(SourceGal));
  int ext=1,colnum,anynul;
  long i;
  char fname[MAX_FILENAME];
  char pxstr[] = {"px"};
  char pystr[] = {"py"};
  char pzstr[] = {"pz"};
  float nulval=0;
  LONGLONG firstrow,firstelem,nelements;
  firstelem = 1;
  
  if(currGalChunk == -1)
    {
      sprintf(fname,"%s[%d]",finname,ext);
      
      fits_open_file(&fptr,fname,READONLY,&status);
      if(status)
	fits_report_error(stderr,status);
      
      fits_get_num_rows(fptr,&TotNumGals,&status);
      if(status)
	fits_report_error(stderr,status);
      
      if(TotNumGals == 0)
	{
	  fits_close_file(fptr,&status);
	  if(status)
	    fits_report_error(stderr,status);
	  
	  currGalChunk = -1;
	  return 1;
	}
      
      fits_get_rowsize(fptr,&BaseNumGalsInChunk,&status);
      if(status)
	fits_report_error(stderr,status);
      
      fbuff = (float*)malloc(sizeof(float)*BaseNumGalsInChunk);
      assert(fbuff != NULL);
      
      NumGalChunks = TotNumGals/BaseNumGalsInChunk;
      if(NumGalChunks*BaseNumGalsInChunk != TotNumGals)
	NumGalChunks += 1;
      
      //fprintf(stderr,"%d: init total # of gals = %ld, # of gals in chunk = %ld, # of chunks = %ld\n",
      //ThisTask,TotNumGals,BaseNumGalsInChunk,NumGalChunks);
      
      status = 0;
      currGalChunk = 0;
      currNumGals = 0;
    }
  
  //alloc buffer mem
  if(MaxNumGalsToRead < BaseNumGalsInChunk)
    MaxNumGalsToRead = BaseNumGalsInChunk;
  *buffGals = (SourceGal*)malloc(sizeof(SourceGal)*MaxNumGalsToRead);
  assert((*buffGals) != NULL);
  
  //read the chunk
  NumGalsRead = 0;
  while(NumGalsRead + BaseNumGalsInChunk <= MaxNumGalsToRead && currGalChunk < NumGalChunks)
    {
      if(NumGalsRead + currNumGals + BaseNumGalsInChunk > TotNumGals)
	nelements = TotNumGals - currNumGals - NumGalsRead;
      else
	nelements = BaseNumGalsInChunk;
      firstrow = currNumGals + NumGalsRead + 1;

      //fprintf(stderr,"%d: chunk total # of gals = %ld, # of gals read so far = %ld, # of gals in this chunk = %lld, chunk %ld of %ld\n",
      //ThisTask,TotNumGals,currNumGals,nelements,currGalChunk,NumGalChunks);
      
      // read px
      fits_get_colnum(fptr,CASEINSEN,pxstr,&colnum,&status);
      if(status)
	fits_report_error(stderr,status);
      fits_read_col(fptr,TFLOAT,colnum,firstrow,firstelem,nelements,&nulval,fbuff,&anynul,&status);
      if(status)
	fits_report_error(stderr,status);
#ifdef DEBUG
#if DEBUG_LEVEL > 1
      fprintf(stderr,"%d: px column # = %d, anynul = %d\n",ThisTask,colnum,anynul);
#endif
#endif
      for(i=0;i<nelements;++i)
	(*buffGals)[NumGalsRead + i].pos[0] = fbuff[i];
      
      // read py
      fits_get_colnum(fptr,CASEINSEN,pystr,&colnum,&status);
      if(status)
	fits_report_error(stderr,status);
      fits_read_col(fptr,TFLOAT,colnum,firstrow,firstelem,nelements,&nulval,fbuff,&anynul,&status);
      if(status)
	fits_report_error(stderr,status);
#ifdef DEBUG
#if DEBUG_LEVEL > 1
      fprintf(stderr,"%d: py column # = %d, anynul = %d\n",ThisTask,colnum,anynul);
#endif
#endif
      for(i=0;i<nelements;++i)
	(*buffGals)[NumGalsRead + i].pos[1] = fbuff[i];
      
      // read pz
      fits_get_colnum(fptr,CASEINSEN,pzstr,&colnum,&status);
      if(status)
	fits_report_error(stderr,status);
      fits_read_col(fptr,TFLOAT,colnum,firstrow,firstelem,nelements,&nulval,fbuff,&anynul,&status);
      if(status)
	fits_report_error(stderr,status);
#ifdef DEBUG
#if DEBUG_LEVEL > 1
      fprintf(stderr,"%d: pz column # = %d, anynul = %d\n",ThisTask,colnum,anynul);
#endif
#endif
      for(i=0;i<nelements;++i)
	(*buffGals)[NumGalsRead + i].pos[2] = fbuff[i];
      
      NumGalsRead += nelements;
      ++currGalChunk;
    }
  
  //set inds
  *NumBuffGals = NumGalsRead;
  for(i=0;i<NumGalsRead;++i)
    (*buffGals)[i].index = currNumGals + i;
  currNumGals += NumGalsRead;
  
  if(currGalChunk == NumGalChunks)
    {
      assert(currNumGals == TotNumGals);
      
      fits_close_file(fptr,&status);
      if(status)
	fits_report_error(stderr,status);
      
      free(fbuff);
      currGalChunk = -1;
      return 1;
    }
  else
    return 0;
}

typedef struct {
  int task;
  int index;
} SortGalTask;

static int compSortGalTask(const void *a, const void *b)
{
  if(((const SortGalTask*)a)->task > ((const SortGalTask*)b)->task)
    return 1;
  else if(((const SortGalTask*)a)->task < ((const SortGalTask*)b)->task)
    return -1;
  else
    return 0;
}

//reorder code a la Gadget-2
int reorder_gals_for_tasks(long NumBuffGals, SourceGal *buffGals, int *sendCounts)
{
  int i,j;
  SourceGal sourceGal,saveGal;
  int rankSource,rankSave,dest;
  SortGalTask *sg;
  long bundleNest,restrictedPeanoInd;
  double vec[3];
  int numNotInDomain;
  
  sg = (SortGalTask*)malloc(sizeof(SortGalTask)*NumBuffGals);
  assert(sg != NULL);
  numNotInDomain = 0;
  for(i=0;i<NTasks;++i)
    sendCounts[i] = 0;
  for(i=0;i<NumBuffGals;++i)
    {
      sg[i].index = i;
      
      vec[0] = buffGals[i].pos[0];
      vec[1] = buffGals[i].pos[1];
      vec[2] = buffGals[i].pos[2];
      
      bundleNest = vec2nest(vec,rayTraceData.bundleOrder);
      restrictedPeanoInd = bundleCellsNest2RestrictedPeanoInd[bundleNest];
      
      if(restrictedPeanoInd == -1)
	{
#ifdef DEBUG
#if DEBUG_LEVEL > 2
	  fprintf(stderr,"%d: found galaxie(s) outside of area!, index = %ld, bundleNest = %ld, restrictedPeanoInd = %ld, pos = %lf|%lf|%lf\n",
		  ThisTask,buffGals[i].index,bundleNest,restrictedPeanoInd,vec[0],vec[1],vec[2]);
#endif
#endif
	  sg[i].task = -1;
	  ++numNotInDomain;
	}
      else
	{
	  for(j=0;j<NTasks;++j)
	    if(firstRestrictedPeanoIndTasks[j] <= restrictedPeanoInd && restrictedPeanoInd <= lastRestrictedPeanoIndTasks[j])
	      {
		sg[i].task = j;
		break;
	      }
	  
	  sendCounts[sg[i].task] += 1;
	}
      
    }
  
  //sort them and make rank in sg.task field
  qsort(sg,(size_t) NumBuffGals,sizeof(SortGalTask),compSortGalTask);
  for(i=0;i<NumBuffGals;++i)
    sg[sg[i].index].task = i;
        
  for(i=0;i<NumBuffGals;++i) /* reoder with an in-place algorithm - see Gadget-2 for details - destroys rank */
    {
      if(i != sg[i].task)
	{
	  sourceGal = buffGals[i];
	  rankSource = sg[i].task;
	  dest = sg[i].task;
                      
	  do
	    {
	      saveGal = buffGals[dest];
	      rankSave = sg[dest].task;
                          
	      buffGals[dest] = sourceGal;
	      sg[dest].task = rankSource;
                  
	      if(dest == i)
		break;
	      
	      sourceGal = saveGal;
	      rankSource = rankSave;
                  
	      dest = rankSource;
	    }
	  while(1);
	}
    }
  
  free(sg);
  
  return numNotInDomain;
}

typedef struct {
  union usgr {
    float rad;
    long rank;
  } u;
  long index;
} SortGalRad;

//this function sorts gals in REVERSE order - notice sign flips in first two if statements
static int compSortGalRad(const void *a, const void *b)
{
  if(((const SortGalRad*)a)->u.rad > ((const SortGalRad*)b)->u.rad)
    return -1;
  else if(((const SortGalRad*)a)->u.rad < ((const SortGalRad*)b)->u.rad)
    return 1;
  else
    return 0;
}

void reorder_gals_for_planes(void)
{
  long i;
  SourceGal sourceGal,saveGal;
  long rankSource,rankSave,dest;
  SortGalRad *sg;
    
  sg = (SortGalRad*)malloc(sizeof(SortGalRad)*NumSourceGalsGlobal);
  assert(sg != NULL);
  
  for(i=0;i<NumSourceGalsGlobal;++i)
    {
      sg[i].index = i;
      sg[i].u.rad = 
	SourceGalsGlobal[i].pos[0]*SourceGalsGlobal[i].pos[0] + 
	SourceGalsGlobal[i].pos[1]*SourceGalsGlobal[i].pos[1] + 
	SourceGalsGlobal[i].pos[2]*SourceGalsGlobal[i].pos[2];
    }
  
  //sort them and make rank in sg.task field
  qsort(sg,(size_t) NumSourceGalsGlobal,sizeof(SortGalRad),compSortGalRad);
  for(i=0;i<NumSourceGalsGlobal;++i)
    sg[sg[i].index].u.rank = i;
        
  for(i=0;i<NumSourceGalsGlobal;++i) /* reoder with an in-place algorithm - see Gadget-2 for details - destroys rank */
    {
      if(i != sg[i].u.rank)
	{
	  sourceGal = SourceGalsGlobal[i];
	  rankSource = sg[i].u.rank;
	  dest = sg[i].u.rank;
                      
	  do
	    {
	      saveGal = SourceGalsGlobal[dest];
	      rankSave = sg[dest].u.rank;
                          
	      SourceGalsGlobal[dest] = sourceGal;
	      sg[dest].u.rank = rankSource;
                  
	      if(dest == i)
		break;
	      
	      sourceGal = saveGal;
	      rankSource = rankSave;
                  
	      dest = rankSource;
	    }
	  while(1);
	}
    }
  
  free(sg);
}

typedef struct {
  long nest;
  long index;
} SortGalNest;

static int compSortGalNest(const void *a, const void *b)
{
  if(((const SortGalNest*)a)->nest > ((const SortGalNest*)b)->nest)
    return 1;
  else if(((const SortGalNest*)a)->nest < ((const SortGalNest*)b)->nest)
    return -1;
  else
    return 0;
}

void reorder_gals_nest(SourceGal *buffSgs, long NumBuffSgs)
{
  long i;
  SourceGal sourceGal,saveGal;
  long rankSource,rankSave,dest;
  SortGalNest *sg;
  double vec[3];
  long buffGalSortOrder = HEALPIX_UTILS_MAXORDER;
  
  sg = (SortGalNest*)malloc(sizeof(SortGalNest)*NumBuffSgs);
  assert(sg != NULL);
  
  for(i=0;i<NumBuffSgs;++i)
    {
      sg[i].index = i;
      vec[0] = buffSgs[i].pos[0];
      vec[1] = buffSgs[i].pos[1];
      vec[2] = buffSgs[i].pos[2];
      sg[i].nest = vec2nest(vec,buffGalSortOrder);
    }
  
  //sort them and make rank in sg.task field
  qsort(sg,(size_t) NumBuffSgs,sizeof(SortGalNest),compSortGalNest);
  for(i=0;i<NumBuffSgs;++i)
    sg[sg[i].index].nest = i;
        
  for(i=0;i<NumBuffSgs;++i) /* reoder with an in-place algorithm - see Gadget-2 for details - destroys rank */
    {
      if(i != sg[i].nest)
	{
	  sourceGal = buffSgs[i];
	  rankSource = sg[i].nest;
	  dest = sg[i].nest;
                      
	  do
	    {
	      saveGal = buffSgs[dest];
	      rankSave = sg[dest].nest;
                          
	      buffSgs[dest] = sourceGal;
	      sg[dest].nest = rankSource;
                  
	      if(dest == i)
		break;
	      
	      sourceGal = saveGal;
	      rankSource = rankSave;
                  
	      dest = rankSource;
	    }
	  while(1);
	}
    }
  
  free(sg);
}
