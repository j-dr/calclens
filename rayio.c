#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>
#include <fitsio.h>
#include <unistd.h>

#include "raytrace.h"

#ifdef USE_FITS_RAYOUT
static void file_write_rays2fits(long fileNum, long firstTask, long lastTask, MPI_Comm fileComm);
#else
static void file_write_rays2bin(long fileNum, long firstTask, long lastTask, MPI_Comm fileComm);
#endif
static void get_ray_iodecomp(long *firstTaskFiles, long *lastTaskFiles, long *fileNum);

/* read rays from arbitrary # of files in arbitrary order 
   very much inspired by Gadget-2
*/
/*
  void read_rays(void)
  {
  long i,ind,offset;
  long group,NumGroups;
  long firstFile,lastFile,fileNum,NumFiles;
  long NumMod,NumFilesinGroup;
  long *readFile;
  
  char name[MAX_FILENAME];
  herr_t status;
  hid_t file_id;
  
  //read # of files
  if(ThisTask == 0)
  {
  sprintf(name,"%s/%s%04ld.%04ld",rayTraceData.OutputPath,rayTraceData.RayOutputName,rayTraceData.CurrentPlaneNum,0l);
  file_id = H5Fopen(name,H5F_ACC_RDWR,H5P_DEFAULT);
  assert(file_id >= 0);
  
  status = H5LTread_dataset(file_id,"/NumFiles",H5T_NATIVE_LONG,&NumFiles);
  assert(status >= 0);
  
  status = H5Fclose(file_id);
  assert(status >= 0);
  }
  
  MPI_Bcast(&NumFiles,1,MPI_LONG,0,MPI_COMM_WORLD); 
  
  NumGroups = NumFiles/rayTraceData.NumFilesIOInParallel;
  if(NumFiles - NumGroups*rayTraceData.NumFilesIOInParallel > 0)
  ++NumGroups;
  
  readFile = (long*)malloc(sizeof(long)*NumFiles);
  assert(readFile != NULL);
  for(i=0;i<NumFiles;++i)
  readFile[i] = 0;
  
  for(group=0;group<NumGroups;++group)
  {
  firstFile = group*rayTraceData.NumFilesIOInParallel;
  lastFile = firstFile + rayTraceData.NumFilesIOInParallel - 1;
  if(lastFile > NumFiles-1)
  lastFile = NumFiles-1;
  NumFilesinGroup = lastFile - firstFile + 1;
  NumMod = NTasks/NumFiles;
  if(NumMod > NTasks/NumFilesinGroup || NumMod < 1)
  NumMod = NTasks/NumFilesinGroup;
  
  for(i=0;i<NTasks;++i)
  {
  fileNum = ThisTask - i;
  while(fileNum < 0)
  fileNum += NTasks;
  ind = fileNum/NumMod;
  offset = fileNum%NumMod;
  fileNum = ind + firstFile;
  
  if(firstFile <= fileNum && fileNum <= lastFile && offset == 0)
  {
  #ifdef DEBUG
  if(DEBUG_LEVEL > 1)
  fprintf(stderr,"%d: fileNum = %ld\n",ThisTask,fileNum);
  #endif
  file_read_hdf52rays(fileNum);
  readFile[fileNum] += 1;
  }
  
  //\\\\\\\\\\\\\\\\\\\\\\\\\\
  MPI_Barrier(MPI_COMM_WORLD);
  //\\\\\\\\\\\\\\\\\\\\\\\\\\
  #ifdef DEBUG
  if(ThisTask == 0 && DEBUG_LEVEL > 1)
  fprintf(stderr,"\n");
  #endif
  }
  #ifdef DEBUG
  if(ThisTask == 0 && DEBUG_LEVEL > 1)
  fprintf(stderr,"\n");
  #endif
  }
  
  // error check to make sure we have read all files
  for(i=0;i<NumFiles;++i)
  assert(readFile[i] == 1);
  
  free(readFile);
  }
*/

/* does actual read of file
   inspired by Gadget-2
*/
/*
 void file_read_hdf52rays(long fileNum)
 {
 char name[MAX_FILENAME];
 hid_t file_id,dataset_id,dataspace_id,memspace_id;
 herr_t status;
 hsize_t dims[1],count[1],offset[1];
 hid_t memRayTypeId;
 long i,j;
 long *NumRaysInPeanoCell,*StartRaysInPeanoCell,peano,nest,readOrder;
 
 sprintf(name,"%s/%s%04ld.%04ld",rayTraceData.OutputPath,rayTraceData.RayOutputName,rayTraceData.CurrentPlaneNum,fileNum);
 
 // file layout
 NumRaysInPeanoCell = (long*)malloc(sizeof(long)*NbundleCells);
 assert(NumRaysInPeanoCell != NULL);
 StartRaysInPeanoCell = (long*)malloc(sizeof(long)*NbundleCells);
 assert(StartRaysInPeanoCell != NULL);
 
 // build mem data type for the rays
 // structure def - only reading and writing parts of it
 //typedef struct {
 //    long nest;
 //    float Bk[2];
 //    float Bkm1[2];
 //    double Ak[4];
 //    double Akm1[4];
 //  #ifdef OUTPUTRAYDEFLECTIONS
 //    float alphakm1[2];
 //#endif
 //  #ifdef OUTPUTPHI
 //  float phi;
 //#endif
 //  } HEALPixRay;
 //
 memRayTypeId = H5Tcreate(H5T_COMPOUND,sizeof(HEALPixRay));
 assert(memRayTypeId >= 0);
 status = H5Tinsert(memRayTypeId,"nest",HOFFSET(HEALPixRay,nest),H5T_NATIVE_LONG);
 assert(status >= 0);
 status = H5Tinsert(memRayTypeId,"ra",HOFFSET(HEALPixRay,Bk[0]),H5T_NATIVE_DOUBLE);
 assert(status >= 0);
 status = H5Tinsert(memRayTypeId,"dec",HOFFSET(HEALPixRay,Bk[1]),H5T_NATIVE_DOUBLE);
 assert(status >= 0);
 status = H5Tinsert(memRayTypeId,"A00",HOFFSET(HEALPixRay,Ak[2*0+0]),H5T_NATIVE_DOUBLE);
 assert(status >= 0);
 status = H5Tinsert(memRayTypeId,"A01",HOFFSET(HEALPixRay,Ak[2*0+1]),H5T_NATIVE_DOUBLE);
 assert(status >= 0);
 status = H5Tinsert(memRayTypeId,"A10",HOFFSET(HEALPixRay,Ak[2*1+0]),H5T_NATIVE_DOUBLE);
 assert(status >= 0);
 status = H5Tinsert(memRayTypeId,"A11",HOFFSET(HEALPixRay,Ak[2*1+1]),H5T_NATIVE_DOUBLE);
 assert(status >= 0);
 #ifdef OUTPUTRAYDEFLECTIONS
 status = H5Tinsert(memRayTypeId,"alpha0",HOFFSET(HEALPixRay,alphakm1[0]),H5T_NATIVE_DOUBLE);
 assert(status >= 0);
 status = H5Tinsert(memRayTypeId,"alpha1",HOFFSET(HEALPixRay,alphakm1[1]),H5T_NATIVE_DOUBLE);
 assert(status >= 0);
 #endif
 #ifdef OUTPUTPHI
 status = H5Tinsert(memRayTypeId,"phi",HOFFSET(HEALPixRay,phi),H5T_NATIVE_DOUBLE);
 assert(status >= 0);
 #endif
 
 // read header info
 file_id = H5Fopen(name,H5F_ACC_RDWR,H5P_DEFAULT);
 assert(file_id >= 0);
 
 status = H5LTread_dataset(file_id,"/RayHEALPixOrder",H5T_NATIVE_LONG,&readOrder);
 assert(status >= 0);
 assert(readOrder == rayTraceData.rayOrder);
 
 status = H5LTread_dataset(file_id,"/PeanoCellHEALPixOrder",H5T_NATIVE_LONG,&readOrder);
 assert(status >= 0);
 assert(readOrder == rayTraceData.bundleOrder);
 
 status = H5LTread_dataset(file_id,"/NumRaysInPeanoCell",H5T_NATIVE_LONG,NumRaysInPeanoCell);
 assert(status >= 0);
 
 status = H5LTread_dataset(file_id,"/StartRaysInPeanoCell",H5T_NATIVE_LONG,StartRaysInPeanoCell);
 assert(status >= 0);
 
 dataset_id = H5Dopen(file_id,"/Rays",H5P_DEFAULT);
 assert(dataset_id >= 0);
 dataspace_id = H5Dget_space(dataset_id);
 assert(dataspace_id >= 0);
 
 for(j=firstRestrictedPeanoIndTasks[ThisTask];j<=lastRestrictedPeanoIndTasks[ThisTask];++j)
 {
 nest = bundleCellsRestrictedPeanoInd2Nest[j];
 assert(nest >= 0 && nest < NbundleCells);
 assert(ISSETBITFLAG(bundleCells[nest].active,PRIMARY_BUNDLECELL));
 peano = nest2peano(nest,rayTraceData.bundleOrder);
 
 if(NumRaysInPeanoCell[peano] > 0)
 {
 dims[0] = (hsize_t) (bundleCells[nest].Nrays);
 memspace_id = H5Screate_simple(1,dims,NULL);
 assert(memspace_id >= 0);
 
 offset[0] = (hsize_t) (StartRaysInPeanoCell[peano]);
 count[0] = (hsize_t) (NumRaysInPeanoCell[peano]);
 status = H5Sselect_hyperslab(dataspace_id,H5S_SELECT_SET,offset,NULL,count,NULL);
 assert(status >= 0);
 
 assert(count[0] == dims[0]);
 
 status = H5Dread(dataset_id,memRayTypeId,memspace_id,dataspace_id,H5P_DEFAULT,bundleCells[nest].rays);
 assert(status >= 0);
 
 status = H5Sclose(memspace_id);
 assert(status >= 0);
 
 // convert all rays back to x-y basis
 for(i=0;i<bundleCells[nest].Nrays;++i)
 {
 rot_ray_radec2xy(&(bundleCells[nest].rays[i]),&(bundleCells[nest].rotData));
 }
 }
 }
 
 status = H5Sclose(dataspace_id);
 assert(status >= 0);
 status = H5Dclose(dataset_id);
 assert(status >= 0);
 status = H5Fclose(file_id);
 assert(status >= 0);
 
 status = H5Tclose(memRayTypeId);
 assert(status >= 0);
 free(StartRaysInPeanoCell);
 free(NumRaysInPeanoCell);
 }
*/

/* write ratys to disk in HDF5 format 
   inspired by Gadget-2
*/
void write_rays(void)
{
  long group,NumGroups,i,j;
  long *firstTaskFiles,*lastTaskFiles,fileNum=-1;
  MPI_Comm fileComm;
  MPI_Group worldGroup,fileGroup;
  int *ranks,Nranks;
  double t;
  
  t = -MPI_Wtime();
  
  firstTaskFiles = (long*)malloc(sizeof(long)*rayTraceData.NumRayOutputFiles);
  assert(firstTaskFiles != NULL);
  lastTaskFiles = (long*)malloc(sizeof(long)*rayTraceData.NumRayOutputFiles);
  assert(lastTaskFiles != NULL);
  get_ray_iodecomp(firstTaskFiles,lastTaskFiles,&fileNum);
  
  /*#ifdef DEBUG
    #if DEBUG_LEVEL > 1
    fprintf(stderr,"%d: before new comm - fileNum = %ld, my group = %ld, first,last = %ld|%ld\n",ThisTask,fileNum,fileNum/rayTraceData.NumFilesIOInParallel,
    firstTaskFiles[fileNum],lastTaskFiles[fileNum]);
    #endif
    #endif
  */
  
  /*make communicators for each file*/
  MPI_Comm_group(MPI_COMM_WORLD,&worldGroup);
  Nranks = lastTaskFiles[fileNum]-firstTaskFiles[fileNum]+1;
  ranks = (int*)malloc(sizeof(int)*Nranks);
  assert(ranks != NULL);
  for(i=0;i<Nranks;++i)
    ranks[i] = firstTaskFiles[fileNum] + i;

  MPI_Group_incl(worldGroup,Nranks,ranks,&fileGroup);
  MPI_Comm_create(MPI_COMM_WORLD,fileGroup,&fileComm);
  
  /*#ifdef DEBUG
    #if DEBUG_LEVEL > 1
    fprintf(stderr,"%d: after new comm - fileNum = %ld, my group = %ld, first,last = %ld|%ld\n",ThisTask,fileNum,fileNum/rayTraceData.NumFilesIOInParallel,
    firstTaskFiles[fileNum],lastTaskFiles[fileNum]);
    #endif
    #endif
  */
  
  /* convert all rays to ra-dec basis*/
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	{
	  for(j=0;j<bundleCells[i].Nrays;++j)
	    {
	      paratrans_ray_curr2obs(&(bundleCells[i].rays[j]));
	      rot_ray_ang2radec(&(bundleCells[i].rays[j]));
	    }
	}
    }
  
  NumGroups = rayTraceData.NumRayOutputFiles/rayTraceData.NumFilesIOInParallel;
  if(rayTraceData.NumRayOutputFiles - NumGroups*rayTraceData.NumFilesIOInParallel > 0)
    ++NumGroups;
  for(group=0;group<NumGroups;++group)
    {
      if(fileNum/rayTraceData.NumFilesIOInParallel == group)
	{
#ifdef DEBUG
#if DEBUG_LEVEL > 1
	  fprintf(stderr,"%d: doing write - group = %ld, fileNum = %ld, my group = %ld\n",ThisTask,group,fileNum,fileNum/rayTraceData.NumFilesIOInParallel);
#endif
#endif
#ifdef USE_FITS_RAYOUT
	  file_write_rays2fits(fileNum,firstTaskFiles[fileNum],lastTaskFiles[fileNum],fileComm);
#else
	  file_write_rays2bin(fileNum,firstTaskFiles[fileNum],lastTaskFiles[fileNum],fileComm);
#endif  
	}
      
      /*\\\\\\\\\\\\\\\\\\\\\\\\\\*/
      MPI_Barrier(MPI_COMM_WORLD);
      /*\\\\\\\\\\\\\\\\\\\\\\\\\\*/
    }
  
  /* convert all rays back to theta-phi basis*/
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	{
	  for(j=0;j<bundleCells[i].Nrays;++j)
	    {
	      rot_ray_radec2ang(&(bundleCells[i].rays[j]));
	      paratrans_ray_obs2curr(&(bundleCells[i].rays[j]));
	    }
	}
    }
  
  free(firstTaskFiles);
  free(lastTaskFiles);
  free(ranks);
  MPI_Comm_free(&fileComm);
  MPI_Group_free(&fileGroup);
  MPI_Group_free(&worldGroup);
  
  t += MPI_Wtime();
  
  if(ThisTask == 0)
    fprintf(stderr,"writing rays to disk took %lf seconds.\n",t);
}

#ifdef USE_FITS_RAYOUT
/* does the actual write of the file */
static void file_write_rays2fits(long fileNum, long firstTask, long lastTask, MPI_Comm fileComm)
{
  const char *ttype[] = 
    { "nest", "ra", "dec", "A00", "A01", "A10", "A11"
#ifdef OUTPUTRAYDEFLECTIONS
      , "alpha0", "alpha1"
#endif
#ifdef OUTPUTPHI
      , "phi"
#endif
    };
  
  const char *tform[] = 
    { "K", "D", "D", "D", "D", "D", "D"
#ifdef OUTPUTRAYDEFLECTIONS
      , "D", "D"
#endif
#ifdef OUTPUTPHI
      , "D"
#endif
    };
  
  char name[MAX_FILENAME];
  char bangname[MAX_FILENAME];
  long NumRaysInFile,i,j;
  long *NumRaysInPeanoCell,*StartRaysInPeanoCell,peano;
  
  fitsfile *fptr;
  int status = 0;
  int naxis = 1;
  long naxes[1],fpixel[1];
  LONGLONG nrows;
  int tfields,colnum;
  long k,chunkInd,firstInd,lastInd,NumRaysInChunkBase,NumRaysInChunk,NumChunks;
  LONGLONG firstrow,firstelem,nelements;
  double *darr;
  long *larr;
  char *buff;
  double ra,dec;
  
  long nwc=0,NtotToRecv,nw=0,nwg=0,rpeano,rowloc;
  MPI_Status mpistatus;
  double t0 = 0.0;
  
  sprintf(name,"%s/%s%04ld.%04ld",rayTraceData.OutputPath,rayTraceData.RayOutputName,rayTraceData.CurrentPlaneNum,fileNum);
  sprintf(bangname,"!%s",name);
  
  /* build fits table layout*/
  tfields = 7;
#ifdef OUTPUTRAYDEFLECTIONS
  tfields += 2;
#endif
#ifdef OUTPUTPHI
  tfields += 1;
#endif
  
  /* build file layout*/
  NumRaysInPeanoCell = (long*)malloc(sizeof(long)*NbundleCells);
  assert(NumRaysInPeanoCell != NULL);
  StartRaysInPeanoCell = (long*)malloc(sizeof(long)*NbundleCells);
  assert(StartRaysInPeanoCell != NULL);
  for(i=0;i<NbundleCells;++i)
    StartRaysInPeanoCell[i] = 0;
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	{
	  peano = nest2peano(bundleCells[i].nest,rayTraceData.bundleOrder);
	  StartRaysInPeanoCell[peano] = bundleCells[i].Nrays;
	  nwc += bundleCells[i].Nrays;
	}
    }
  MPI_Allreduce(StartRaysInPeanoCell,NumRaysInPeanoCell,(int) NbundleCells,MPI_LONG,MPI_SUM,fileComm);
  j = 0;
  for(i=0;i<NbundleCells;++i)
    {
      StartRaysInPeanoCell[i] = j;
      j += NumRaysInPeanoCell[i];
    }
  NumRaysInFile = j;
  
  /* make the file and write header info */
  if(ThisTask == firstTask)
    {
      t0 = -MPI_Wtime();
      
      remove(name);
      
      fits_create_file(&fptr,bangname,&status);
      if(status)
        fits_report_error(stderr,status);
      
      naxes[0] = 2l*NbundleCells;
      fits_create_img(fptr,LONGLONG_IMG,naxis,naxes,&status);
      if(status)
        fits_report_error(stderr,status);
      
      fpixel[0] = 0+1;
      fits_write_pix(fptr,TLONG,fpixel,(LONGLONG) (NbundleCells),NumRaysInPeanoCell,&status);
      if(status)
        fits_report_error(stderr,status);
      
      fpixel[0] = NbundleCells+1;
      fits_write_pix(fptr,TLONG,fpixel,(LONGLONG) (NbundleCells),StartRaysInPeanoCell,&status);
      if(status)
        fits_report_error(stderr,status);
      
      fits_write_key(fptr,TLONG,"NumFiles",&(rayTraceData.NumRayOutputFiles),"number of files that rays are split into",&status);
      if(status)
        fits_report_error(stderr,status);
      
      fits_write_key(fptr,TLONG,"PeanoCellHEALPixOrder",&(rayTraceData.bundleOrder),"HEALPix order of peano indexed cells rays are organized into",&status);
      if(status)
        fits_report_error(stderr,status);
      
      fits_write_key(fptr,TLONG,"RayHEALPixOrder",&(rayTraceData.rayOrder),"HEALPix order of ray grid",&status);
      if(status)
        fits_report_error(stderr,status);
      
      nrows = (LONGLONG) (NumRaysInFile);
      fits_create_tbl(fptr,BINARY_TBL,nrows,tfields,ttype,tform,NULL,"Rays",&status);
      if(status)
        fits_report_error(stderr,status);
      
      fits_get_rowsize(fptr,&NumRaysInChunkBase,&status);
      if(status)
	fits_report_error(stderr,status);
    }
  
  MPI_Bcast(&NumRaysInChunkBase,1,MPI_LONG,0,fileComm);
  if(sizeof(long) > sizeof(double))
    buff = (char*)malloc(sizeof(long)*NumRaysInChunkBase);
  else
    buff = (char*)malloc(sizeof(double)*NumRaysInChunkBase);
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
	    MPI_Send(&nwc,1,MPI_LONG,(int) firstTask,TAG_RAYIO_TOTNUM,MPI_COMM_WORLD);
	  
	  for(rpeano=0;rpeano<NrestrictedPeanoInd;++rpeano)
            {
              j = bundleCellsRestrictedPeanoInd2Nest[rpeano];
              
	      if(ISSETBITFLAG(bundleCells[j].active,PRIMARY_BUNDLECELL))
		{
		  peano = nest2peano(bundleCells[j].nest,rayTraceData.bundleOrder);
		  
		  assert(NumRaysInPeanoCell[peano] == ((1l) << (2*(rayTraceData.rayOrder-rayTraceData.bundleOrder))));
		  assert((StartRaysInPeanoCell[peano] - 
			  ((StartRaysInPeanoCell[peano])/(((1l) << (2*(rayTraceData.rayOrder-rayTraceData.bundleOrder)))))
			  *(((1l) << (2*(rayTraceData.rayOrder-rayTraceData.bundleOrder))))) == 0);
		  
		  NumChunks = NumRaysInPeanoCell[peano]/NumRaysInChunkBase;
		  if(NumChunks*NumRaysInChunkBase < NumRaysInPeanoCell[peano])
		    NumChunks += 1;
		  
		  firstrow = (LONGLONG) (StartRaysInPeanoCell[peano]) + (LONGLONG) 1;
		  firstelem = 1;
		  for(chunkInd=0;chunkInd<NumChunks;++chunkInd)
		    {
		      firstInd = chunkInd*NumRaysInChunkBase;
		      lastInd = (chunkInd+1)*NumRaysInChunkBase-1;
		      if(lastInd >= NumRaysInPeanoCell[peano]-1)
			lastInd = NumRaysInPeanoCell[peano]-1;
		      NumRaysInChunk = lastInd - firstInd + 1;
		      
		      nelements = (LONGLONG) NumRaysInChunk;
		      nw += NumRaysInChunk;
		      
		      if(ThisTask != firstTask)
			{
			  rowloc = firstrow;
			  MPI_Send(&rowloc,1,MPI_LONG,(int) firstTask,TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD);
			  MPI_Send(&NumRaysInChunk,1,MPI_LONG,(int) firstTask,TAG_RAYIO_NUMCHUNK,MPI_COMM_WORLD);
			  colnum = TAG_RAYIO_CHUNKDATA+1;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    larr[k-firstInd] = bundleCells[j].rays[k].nest;
			  MPI_Ssend(larr,(int) NumRaysInChunk,MPI_LONG,(int) firstTask,colnum,MPI_COMM_WORLD);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    {
			      vec2radec(bundleCells[j].rays[k].n,&ra,&dec);
			      darr[k-firstInd] = ra;
			    }
			  MPI_Ssend(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    {
			      vec2radec(bundleCells[j].rays[k].n,&ra,&dec);
			      darr[k-firstInd] = dec;
			    }
			  MPI_Ssend(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].A[2*0+0];
			  MPI_Ssend(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
                          ++colnum;
			  			  
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].A[2*0+1];
			  MPI_Ssend(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].A[2*1+0];
			  MPI_Ssend(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].A[2*1+1];
			  MPI_Ssend(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
			  ++colnum;
			  
#ifdef OUTPUTRAYDEFLECTIONS
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].alpha[0];
			  MPI_Ssend(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].alpha[1];
			  MPI_Ssend(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
			  ++colnum;
#endif
#ifdef OUTPUTPHI
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].phi;
			  MPI_Ssend(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) firstTask,colnum,MPI_COMM_WORLD);
			  ++colnum;
#endif
			  firstrow += nelements;
			}
		      else
			{
			  colnum = 1;
			  for(k=firstInd;k<=lastInd;++k)
			    larr[k-firstInd] = bundleCells[j].rays[k].nest;
			  fits_write_col(fptr,TLONG,colnum,firstrow,firstelem,nelements,larr,&status);
			  if(status)
			    fits_report_error(stderr,status);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    {
			      vec2radec(bundleCells[j].rays[k].n,&ra,&dec);
			      darr[k-firstInd] = ra;
			    }
			  fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
			  if(status)
			    fits_report_error(stderr,status);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    {
			      vec2radec(bundleCells[j].rays[k].n,&ra,&dec);
			      darr[k-firstInd] = dec;
			    }
			  fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
			  if(status)
			    fits_report_error(stderr,status);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].A[2*0+0];
			  fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
			  if(status)
			    fits_report_error(stderr,status);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].A[2*0+1];
			  fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
			  if(status)
			    fits_report_error(stderr,status);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].A[2*1+0];
			  fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
			  if(status)
			    fits_report_error(stderr,status);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].A[2*1+1];
			  fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
			  if(status)
			    fits_report_error(stderr,status);
			  ++colnum;
			  
#ifdef OUTPUTRAYDEFLECTIONS
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].alpha[0];
			  fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
			  if(status)
			    fits_report_error(stderr,status);
			  ++colnum;
			  
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].alpha[1];
			  fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
			  if(status)
			    fits_report_error(stderr,status);
			  ++colnum;
#endif
#ifdef OUTPUTPHI
			  for(k=firstInd;k<=lastInd;++k)
			    darr[k-firstInd] = bundleCells[j].rays[k].phi;
			  fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
			  if(status)
			    fits_report_error(stderr,status);
			  ++colnum;
#endif
			  firstrow += nelements;
			}
		    }// for(chunkInd=0;chunkInd<NumChunks;++chunkInd)
		} //if(ISSETBITFLAG(bundleCells[j].active,PRIMARY_BUNDLECELL)).
	    } //for(j=0;j<NbundleCells;++j)
	} //if(ThisTask == i)
      
      if(i != firstTask && ThisTask == firstTask)
	{
	  MPI_Recv(&NtotToRecv,1,MPI_LONG,(int) i,TAG_RAYIO_TOTNUM,MPI_COMM_WORLD,&mpistatus);
	  
	  firstelem = 1;
	  while(NtotToRecv > 0)
	    {
	      MPI_Recv(&rowloc,1,MPI_LONG,(int) i,TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      MPI_Recv(&NumRaysInChunk,1,MPI_LONG,(int) i,TAG_RAYIO_NUMCHUNK,MPI_COMM_WORLD,&mpistatus);
	      firstrow = (LONGLONG) (rowloc);
	      nelements = (LONGLONG) NumRaysInChunk;
	      colnum = 1;
	      
	      MPI_Recv(larr,(int) NumRaysInChunk,MPI_LONG,(int) i,colnum+TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TLONG,colnum,firstrow,firstelem,nelements,larr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) i,colnum+TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) i,colnum+TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) i,colnum+TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) i,colnum+TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) i,colnum+TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) i,colnum+TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
#ifdef OUTPUTRAYDEFLECTIONS
	      MPI_Recv(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) i,colnum+TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
	      
	      MPI_Recv(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) i,colnum+TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
#endif
#ifdef OUTPUTPHI
	      MPI_Recv(darr,(int) NumRaysInChunk,MPI_DOUBLE,(int) i,colnum+TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&mpistatus);
	      fits_write_col(fptr,TDOUBLE,colnum,firstrow,firstelem,nelements,darr,&status);
	      if(status)
		fits_report_error(stderr,status);
	      ++colnum;
#endif
	      	      
	      nwg += NumRaysInChunk;
              NtotToRecv -= NumRaysInChunk;
	    }
	}
      
      //////////////////////////////
      MPI_Barrier(fileComm);
      //////////////////////////////
    }
  
  if(ThisTask == firstTask)
    {
      fits_close_file(fptr,&status);
      if(status)
	fits_report_error(stderr,status);
      
      t0 += MPI_Wtime();

#ifdef DEBUG      
      fprintf(stderr,"writing %ld rays to file '%s' took %g seconds.\n",NumRaysInFile,name,t0);
#endif
      
      assert(nwg == NumRaysInFile-nw); //error check # of rays recvd
    }
  
  //error check # of rays written
  MPI_Allreduce(&nw,&nwg,1,MPI_LONG,MPI_SUM,fileComm);
  assert(nw == nwc);
  assert(nwg == NumRaysInFile);
  
  //clean up and close files for this task
  free(buff);
  free(StartRaysInPeanoCell);
  free(NumRaysInPeanoCell);
}
#else
static size_t fwrite_errcheck(const void *ptr, size_t size, size_t nobj, FILE *stream)
{
  size_t nret;
  
  nret = fwrite(ptr,size,nobj,stream);
  
  if(nret != nobj)
    {
      fprintf(stderr,"%d: problem writing to file! size = %zu, nobj = %zu, # written = %zu\n",ThisTask,size,nobj,nret);
      MPI_Abort(MPI_COMM_WORLD,666);
    }
  
  return nret;
}


/* does the actual write of the file */
static void file_write_rays2bin(long fileNum, long firstTask, long lastTask, MPI_Comm fileComm)
{
  char name[MAX_FILENAME];
  long NumRaysInFile,i,j;
  long *NumRaysInPeanoCell,*StartRaysInPeanoCell,peano,rpeano;
  size_t buffSizeMB = 10;
  MPI_Status status;
  
  char *chunkRays;
  long k,chunkInd,firstInd,lastInd,NumRaysInChunkBase,NumRaysInChunk,NumChunks;
  double ra,dec;
  long nw=0,nwg=0,nwc=0,NtotToRecv;
  
  struct IOheader {
    long NumFiles;
    long PeanoCellHEALPixOrder;
    long RayHEALPixOrder;
    long flag_defl;
    long flag_phi;
    char pad[216]; //pad to 256 bytes
  } header;
  
  int dummy;
  FILE *fp = NULL;
  double t0 = 0.0;
  
  size_t rays = 0;
  rays += sizeof(long);
  rays += 2*sizeof(double);
  rays += 4*sizeof(double);
#ifdef OUTPUTRAYDEFLECTIONS
  rays += 2*sizeof(double);
#endif
#ifdef OUTPUTPHI
  rays += sizeof(double);
#endif
  
  sprintf(name,"%s/%s%04ld.%04ld",rayTraceData.OutputPath,rayTraceData.RayOutputName,rayTraceData.CurrentPlaneNum,fileNum);
  
  NumRaysInChunkBase = buffSizeMB*1024l*1024l/rays;
  chunkRays = (char*)malloc(rays*NumRaysInChunkBase);
  assert(chunkRays != NULL);
  
  /* build file layout*/
  NumRaysInPeanoCell = (long*)malloc(sizeof(long)*NbundleCells);
  assert(NumRaysInPeanoCell != NULL);
  StartRaysInPeanoCell = (long*)malloc(sizeof(long)*NbundleCells);
  assert(StartRaysInPeanoCell != NULL);
  for(i=0;i<NbundleCells;++i)
    StartRaysInPeanoCell[i] = 0;
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	{
	  peano = nest2peano(bundleCells[i].nest,rayTraceData.bundleOrder);
	  StartRaysInPeanoCell[peano] = bundleCells[i].Nrays;
	  nwc += bundleCells[i].Nrays;
	}
    }
  MPI_Allreduce(StartRaysInPeanoCell,NumRaysInPeanoCell,(int) NbundleCells,MPI_LONG,MPI_SUM,fileComm);
  j = 0;
  for(i=0;i<NbundleCells;++i)
    {
      StartRaysInPeanoCell[i] = j;
      j += NumRaysInPeanoCell[i];
    }
  NumRaysInFile = j;
  
  //set header
  header.NumFiles = rayTraceData.NumRayOutputFiles;
  header.PeanoCellHEALPixOrder = rayTraceData.bundleOrder;
  header.RayHEALPixOrder = rayTraceData.rayOrder;
  
#ifdef OUTPUTRAYDEFLECTIONS
  header.flag_defl = 1;
#else
  header.flag_defl = 0;
#endif
#ifdef OUTPUTPHI
  header.flag_phi = 1;
#else
  header.flag_phi = 0;
#endif
  
  /* make the file and write header info */
  if(ThisTask == firstTask)
    {
      t0 = -MPI_Wtime();
      
      fp = fopen(name,"w");      
      if(fp == NULL)
	{
	  fprintf(stderr,"%d: could not open file '%s' for header!\n",ThisTask,name);
	  MPI_Abort(MPI_COMM_WORLD,666);
	}
      
      dummy = sizeof(struct IOheader);
      fwrite_errcheck(&dummy,(size_t) 1,sizeof(int),fp);
      fwrite_errcheck(&header,(size_t) 1,sizeof(struct IOheader),fp);
      fwrite_errcheck(&dummy,(size_t) 1,sizeof(int),fp);
      
      dummy = NbundleCells*sizeof(long);
      fwrite_errcheck(&dummy,(size_t) 1,sizeof(int),fp);
      fwrite_errcheck(NumRaysInPeanoCell,(size_t) NbundleCells,sizeof(long),fp);
      fwrite_errcheck(&dummy,(size_t) 1,sizeof(int),fp);
      
      dummy = NbundleCells*sizeof(long);
      fwrite_errcheck(&dummy,(size_t) 1,sizeof(int),fp);
      fwrite_errcheck(StartRaysInPeanoCell,(size_t) NbundleCells,sizeof(long),fp);
      fwrite_errcheck(&dummy,(size_t) 1,sizeof(int),fp);
      
      dummy = NumRaysInFile*rays;
      fwrite_errcheck(&dummy,(size_t) 1,sizeof(int),fp);
    }
  
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
	    {
	      MPI_Send(&nwc,1,MPI_LONG,(int) firstTask,TAG_RAYIO_TOTNUM,MPI_COMM_WORLD);
	    }
	  
	  for(rpeano=0;rpeano<NrestrictedPeanoInd;++rpeano)
	    {
	      j = bundleCellsRestrictedPeanoInd2Nest[rpeano];
	      
	      if(ISSETBITFLAG(bundleCells[j].active,PRIMARY_BUNDLECELL))
		{
		  peano = nest2peano(bundleCells[j].nest,rayTraceData.bundleOrder);
		  
		  assert(NumRaysInPeanoCell[peano] == ((1l) << (2*(rayTraceData.rayOrder-rayTraceData.bundleOrder))));
		  assert((StartRaysInPeanoCell[peano] - 
			  ((StartRaysInPeanoCell[peano])/(((1l) << (2*(rayTraceData.rayOrder-rayTraceData.bundleOrder)))))
			  *(((1l) << (2*(rayTraceData.rayOrder-rayTraceData.bundleOrder))))) == 0);
		  
		  NumChunks = NumRaysInPeanoCell[peano]/NumRaysInChunkBase;
		  if(NumChunks*NumRaysInChunkBase < NumRaysInPeanoCell[peano])
		    NumChunks += 1;
		  
		  for(chunkInd=0;chunkInd<NumChunks;++chunkInd)
		    {
		      firstInd = chunkInd*NumRaysInChunkBase;
		      lastInd = (chunkInd+1)*NumRaysInChunkBase-1;
		      if(lastInd >= NumRaysInPeanoCell[peano]-1)
			lastInd = NumRaysInPeanoCell[peano]-1;
		      NumRaysInChunk = lastInd - firstInd + 1;
		      
		      for(k=firstInd;k<=lastInd;++k)
			{
			  ++nw;
			  vec2radec(bundleCells[j].rays[k].n,&ra,&dec);
			  
			  *((long*) (&(chunkRays[(k-firstInd)*rays]))) = bundleCells[j].rays[k].nest;
			  *((double*) (&(chunkRays[(k-firstInd)*rays + sizeof(long)]))) = ra;
			  *((double*) (&(chunkRays[(k-firstInd)*rays + sizeof(long) + sizeof(double)]))) = dec;
			  *((double*) (&(chunkRays[(k-firstInd)*rays + sizeof(long) + 2*sizeof(double)]))) = bundleCells[j].rays[k].A[0];
			  *((double*) (&(chunkRays[(k-firstInd)*rays + sizeof(long) + 2*sizeof(double) + sizeof(double)]))) = bundleCells[j].rays[k].A[1];
			  *((double*) (&(chunkRays[(k-firstInd)*rays + sizeof(long) + 2*sizeof(double) + 2*sizeof(double)]))) = bundleCells[j].rays[k].A[2];
			  *((double*) (&(chunkRays[(k-firstInd)*rays + sizeof(long) + 2*sizeof(double) + 3*sizeof(double)]))) = bundleCells[j].rays[k].A[3];
			  
#ifdef OUTPUTRAYDEFLECTIONS
			  *((double*) (&(chunkRays[(k-firstInd)*rays + sizeof(long) + 2*sizeof(double) + 4*sizeof(double)]))) = bundleCells[j].rays[k].alpha[0];
			  *((double*) (&(chunkRays[(k-firstInd)*rays + sizeof(long) + 2*sizeof(double) + 4*sizeof(double) + sizeof(double)]))) = bundleCells[j].rays[k].alpha[1];
#endif
#ifdef OUTPUTPHI
			  *((double*) (&(chunkRays[(k-firstInd)*rays + sizeof(long) + 2*sizeof(double) + 4*sizeof(double) + 2*sizeof(double)]))) = bundleCells[j].rays[k].phi;
#endif
			}
		      
		      if(ThisTask != firstTask)
			{
			  MPI_Send(&NumRaysInChunk,1,MPI_LONG,(int) firstTask,TAG_RAYIO_NUMCHUNK,MPI_COMM_WORLD);
			  MPI_Ssend(chunkRays,(int) (rays*NumRaysInChunk),MPI_BYTE,(int) firstTask,TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD);
			}
		      else
			fwrite_errcheck(chunkRays,(size_t) NumRaysInChunk,rays,fp);
		      
		    }// for(chunkInd=0;chunkInd<NumChunks;++chunkInd)
		} //if(ISSETBITFLAG(bundleCells[j].active,PRIMARY_BUNDLECELL)).
	    } //for(j=0;j<NbundleCells;++j)
	} //if(ThisTask == i)
      
      if(i != firstTask && ThisTask == firstTask)
	{
	  MPI_Recv(&NtotToRecv,1,MPI_LONG,(int) i,TAG_RAYIO_TOTNUM,MPI_COMM_WORLD,&status);
	  
	  while(NtotToRecv > 0)
	    {
	      MPI_Recv(&NumRaysInChunk,1,MPI_LONG,(int) i,TAG_RAYIO_NUMCHUNK,MPI_COMM_WORLD,&status);
	      MPI_Recv(chunkRays,(int) (rays*NumRaysInChunk),MPI_BYTE,(int) i,TAG_RAYIO_CHUNKDATA,MPI_COMM_WORLD,&status);
	      fwrite_errcheck(chunkRays,(size_t) NumRaysInChunk,rays,fp);
	      nwg += NumRaysInChunk;
	      NtotToRecv -= NumRaysInChunk;
	    }
	}
      
      //////////////////////////////
      MPI_Barrier(fileComm);
      //////////////////////////////
    }
  
  if(ThisTask == firstTask)
    {
      dummy = NumRaysInFile*rays;
      fwrite_errcheck(&dummy,(size_t) 1,sizeof(int),fp);
      fclose(fp);
      t0 += MPI_Wtime();
      
      fprintf(stderr,"writing %ld rays to file '%s' took %g seconds.\n",NumRaysInFile,name,t0);
      
      assert(nwg == NumRaysInFile-nw); //error check # of rays recvd
    }

  //error check # of rays written
  MPI_Allreduce(&nw,&nwg,1,MPI_LONG,MPI_SUM,fileComm);
  assert(nw == nwc);
  assert(nwg == NumRaysInFile);
  
  free(StartRaysInPeanoCell);
  free(NumRaysInPeanoCell);
  free(chunkRays);
}
#endif /* USE_FITS_RAYOUT */

/* gets I/O decomp given number of Tasks, and the number of output files wanted 
   inspired by Gadget-2
*/
static void get_ray_iodecomp(long *firstTaskFiles, long *lastTaskFiles, long *fileNum)
{
  long i,j,n,setRestByHand;
  long *nrays,*totnrays;
  long numRaysPerFile,currNumRays;
  
  //get num rays per task
  nrays = (long*)malloc(sizeof(long)*NTasks);
  assert(nrays != NULL);
  totnrays = (long*)malloc(sizeof(long)*NTasks);
  assert(totnrays != NULL);
  
  for(i=0;i<NTasks;++i)
    nrays[i] = 0;
  for(i=0;i<NbundleCells;++i)
    if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
      nrays[ThisTask] += bundleCells[i].Nrays;
  
  MPI_Allreduce(nrays,totnrays,NTasks,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  
  j = 0;
  for(i=0;i<NTasks;++i)
    j += totnrays[i];
  numRaysPerFile = j/rayTraceData.NumRayOutputFiles;
  
  setRestByHand = 0;
  j = 0;
  currNumRays = totnrays[j];
  for(i=0;i<rayTraceData.NumRayOutputFiles;++i)
    {
      firstTaskFiles[i] = j;
      while(currNumRays < (i+1)*numRaysPerFile && j < NTasks-1)
	{
	  if((NTasks-1) - (j) + 1 <= (rayTraceData.NumRayOutputFiles-1) - (i+1) + 1)
	    {
	      lastTaskFiles[i] = j-1;
	      ++i;
	      setRestByHand = 1;
	      break;
	    }
	  
	  ++j;
	  currNumRays += totnrays[j];
	}
      
      if(setRestByHand)
	break;
      
      lastTaskFiles[i] = j-1;
    }
  
  if(setRestByHand)
    {
      for(n=i;n<rayTraceData.NumRayOutputFiles;++n)
	{
	  firstTaskFiles[n] = j;
	  lastTaskFiles[n] = j;
	  ++j;
	}
    }
  else
    lastTaskFiles[rayTraceData.NumRayOutputFiles-1] = NTasks-1;
  
  //error check - if assignment above fails - then revert to old way of doing it
  setRestByHand = 0;
  for(i=0;i<rayTraceData.NumRayOutputFiles;++i)
    {
      if(!(firstTaskFiles[i] >= 0 && firstTaskFiles[i] < NTasks))
	setRestByHand = 1;
	  
      if(!(lastTaskFiles[i] >= 0 && lastTaskFiles[i] < NTasks))
	setRestByHand = 1;
      
      if(!(lastTaskFiles[i] >= firstTaskFiles[i]))
	setRestByHand = 1;
      
      if(i < rayTraceData.NumRayOutputFiles-1)
	{
	  if(!(lastTaskFiles[i] + 1 == firstTaskFiles[i+1]))
	    setRestByHand = 1;
	}
      else
	{
	  if(!(lastTaskFiles[i] == NTasks-1))
	    setRestByHand = 1;
	}
      
      if(setRestByHand)
	break;
    }
  
  long numTasksPerFile,numExtraTasks;  
  if(setRestByHand)
    {
      numTasksPerFile = NTasks/rayTraceData.NumRayOutputFiles;
      numExtraTasks = NTasks - numTasksPerFile*rayTraceData.NumRayOutputFiles;
      
      j = 0;
      for(i=0;i<rayTraceData.NumRayOutputFiles;++i)
	{
	  firstTaskFiles[i] = j;
	  if(i < numExtraTasks)
	    lastTaskFiles[i] = numTasksPerFile + j;
	  else
	    lastTaskFiles[i] = numTasksPerFile + j - 1;
	  
	  j = lastTaskFiles[i] + 1;
	}
    }
  
  for(i=0;i<rayTraceData.NumRayOutputFiles;++i)
    if(firstTaskFiles[i] <= ThisTask && ThisTask <= lastTaskFiles[i])
      *fileNum = i;
  
#ifdef DEBUG
#if DEBUG_LEVEL > 0
  if(ThisTask == 0)
    {
      fprintf(stderr,"\nray I/O: file decomp info - # of files = %ld, NTasks = %d, numRaysPerFile = %ld\n"
	      ,rayTraceData.NumRayOutputFiles,NTasks,numRaysPerFile);
      for(i=0;i<rayTraceData.NumRayOutputFiles;++i)
	{
	  currNumRays = 0;
	  for(j=firstTaskFiles[i];j<=lastTaskFiles[i];++j)
	    currNumRays += totnrays[j];
	  fprintf(stderr,"%ld: firstTaskFiles,lastTaskFiles,numTasks,numRays = %ld|%ld|%ld|%ld\n",
		  i,firstTaskFiles[i],lastTaskFiles[i],lastTaskFiles[i]-firstTaskFiles[i]+1,currNumRays);
	}
      fprintf(stderr,"\n");
    }
#endif
#endif
  
  free(nrays);
  free(totnrays);  
}

