#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <gsl/gsl_sort_long.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#include "raytrace.h"

static void fillWriteBuffData(WriteBuffData *wb, long NumRayTracingPlanes, long HEALPixOrder, long MAX_NPART);
static void freeWriteBuffData(WriteBuffData *wb);
static long needToWriteRayTracingPlanes(WriteBuffData *wb, long *RayTracingPlaneIdMaxNumLCParts, long *TotNumLCPartsNonZero);
static void writeRayTracingPlane(long j, WriteBuffData *wb);

static void fillWriteBuffData(WriteBuffData *wb, long NumRayTracingPlanes, long HEALPixOrder, long MAX_NPART)
{
  long i;
  long NPix = order2npix(HEALPixOrder);
  
  //fill in basic info
  wb->NumRayTracingPlanes = NumRayTracingPlanes;
  wb->HEALPixOrder = HEALPixOrder;
  wb->NPix = NPix;
  wb->MaxTotNumLCParts = MAX_NPART;
  wb->ChunkSizeLCParts = (long) (((double) MAX_NPART)/rayTraceData.LightConePartChunkFactor/NumRayTracingPlanes);
  wb->NumLCPartWriteBuff = MAX_NPART + wb->ChunkSizeLCParts;
  
  //alloc mem for I/O buffering
  wb->NumLCParts = (long*)malloc(sizeof(long)*NumRayTracingPlanes);
  assert(wb->NumLCParts != NULL);
  wb->NumLCPartsUsed = (long*)malloc(sizeof(long)*NumRayTracingPlanes);
  assert(wb->NumLCParts != NULL);
  wb->LCParts = (LCParticle**)malloc(sizeof(LCParticle*)*NumRayTracingPlanes);
  assert(wb->LCParts != NULL);
  wb->TotNumLCPartsInPlane = (long*)malloc(sizeof(long)*NumRayTracingPlanes);
  assert(wb->TotNumLCPartsInPlane != NULL);
  for(i=0;i<NumRayTracingPlanes;++i)
    {
      wb->LCParts[i] = NULL;
      wb->NumLCParts[i] = 0;
      wb->NumLCPartsUsed[i] = 0;
      wb->TotNumLCPartsInPlane[i] = 0;
    }
  wb->NumLCPartsInPix = (long*)malloc(sizeof(long)*NPix);
  assert(wb->NumLCPartsInPix != NULL);
  wb->NumLCPartsInPixUpdate = (long*)malloc(sizeof(long)*NPix);
  assert(wb->NumLCPartsInPixUpdate != NULL);
  for(i=0;i<NPix;++i)
    {
      wb->NumLCPartsInPix[i] = 0;
      wb->NumLCPartsInPixUpdate[i] = 0;
    }
  wb->LCPartWriteBuff = (LCParticle*)malloc(sizeof(LCParticle)*wb->NumLCPartWriteBuff);
  assert(wb->LCPartWriteBuff != NULL);
  wb->PeanoInds = (long*)malloc(sizeof(long)*wb->NumLCPartWriteBuff);
  assert(wb->PeanoInds != NULL);
  wb->PeanoSortInds = (size_t*)malloc(sizeof(size_t)*wb->NumLCPartWriteBuff);
  assert(wb->PeanoSortInds != NULL);
}

static void freeWriteBuffData(WriteBuffData *wb)
{
  long i;
  
  //alloc mem for I/O buffering
  free(wb->NumLCParts);
  free(wb->NumLCPartsUsed);
  for(i=0;i<wb->NumRayTracingPlanes;++i)
    {
      if(wb->LCParts[i] != NULL)
	free(wb->LCParts[i]);
    }
  free(wb->LCParts);
  free(wb->TotNumLCPartsInPlane);
  free(wb->NumLCPartsInPix);
  free(wb->NumLCPartsInPixUpdate);
  free(wb->LCPartWriteBuff);
  free(wb->PeanoInds);
  free(wb->PeanoSortInds);
}

static long needToWriteRayTracingPlanes(WriteBuffData *wb, long *RayTracingPlaneIdMaxNumLCParts, long *TotNumLCPartsNonZero)
{
  long writeRayTracingPlanes;
  long TotNumLCParts = 0;
  long j;
  long MaxNumLCParts;
  
  //get total number of allocated particles
  //also test if we have planes with used particles
  *TotNumLCPartsNonZero = 0;
  for(j=0;j<wb->NumRayTracingPlanes;++j)
    {
      TotNumLCParts += wb->NumLCParts[j];
      if(wb->NumLCPartsUsed[j] > 0)
	*TotNumLCPartsNonZero = *TotNumLCPartsNonZero + 1;
    }
  
  //test if need to write ray tracing planes
  //passes if either:
  // 1) more than ten planes have LCParticles in mem
  // 2) the total number of allocated LCParticles exceeds the max set by wb->MaxTotNumLCParts
  if((TotNumLCParts >= wb->MaxTotNumLCParts && *TotNumLCPartsNonZero > 0) || *TotNumLCPartsNonZero > rayTraceData.MaxNumLensPlaneInMem)
    {
      writeRayTracingPlanes = 1;
      
      if(TotNumLCParts >= wb->MaxTotNumLCParts && *TotNumLCPartsNonZero > 0)
	{
	  //find plane that uses the most mem
	  *RayTracingPlaneIdMaxNumLCParts = 0;
	  MaxNumLCParts = wb->NumLCParts[0];
	  for(j=0;j<wb->NumRayTracingPlanes;++j)
	    {
	      if(wb->NumLCParts[j] > MaxNumLCParts && wb->NumLCPartsUsed[j] > 0)
		{
		  MaxNumLCParts = wb->NumLCParts[j];
		  *RayTracingPlaneIdMaxNumLCParts = j;
		}
	    }
	}
      else
	{
	  *RayTracingPlaneIdMaxNumLCParts = -1;
	}
    }
  else
    writeRayTracingPlanes = 0;    
  
  return writeRayTracingPlanes;
}

static void writeRayTracingPlane(long j, WriteBuffData *wb)
{
  long k,LCPartWriteBuffInd;
  double vec[3];
  hid_t file_id,dataset_id;
  herr_t status;
  char tablename[MAX_FILENAME];
  hsize_t chunk_size;
  int *fill_data = NULL;
  int compress = 0;
  LCParticle LCPartRead;
  char file_name[MAX_FILENAME];
  
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  assert(fapl >= 0);
  status = H5Pset_fclose_degree(fapl,H5F_CLOSE_STRONG);
  assert(status >= 0);
  
  //define LCParticle type in HDF5 for table I/O
  size_t dst_size = sizeof(LCParticle);
  size_t dst_offset[NFIELDS_LCPARTICLE] = { HOFFSET(LCParticle,partid),
					    HOFFSET(LCParticle,px),
					    HOFFSET(LCParticle,py),
					    HOFFSET(LCParticle,pz),
					    HOFFSET(LCParticle,vx),
					    HOFFSET(LCParticle,vy),
					    HOFFSET(LCParticle,vz),
					    HOFFSET(LCParticle,mass) };
  size_t dst_sizes[NFIELDS_LCPARTICLE] = { sizeof(LCPartRead.partid),
					   sizeof(LCPartRead.px),
					   sizeof(LCPartRead.py),
					   sizeof(LCPartRead.pz),
					   sizeof(LCPartRead.vx),
					   sizeof(LCPartRead.vy),
					   sizeof(LCPartRead.vz),
					   sizeof(LCPartRead.mass) };
  const char *field_names[NFIELDS_LCPARTICLE] = { "partid",
						  "px",
						  "py",
						  "pz",
						  "vx",
						  "vy",
						  "vz",
						  "mass" };
  hid_t field_type[NFIELDS_LCPARTICLE] = { H5T_NATIVE_LONG, 
					   H5T_NATIVE_FLOAT,
					   H5T_NATIVE_FLOAT,
					   H5T_NATIVE_FLOAT,
					   H5T_NATIVE_FLOAT,
					   H5T_NATIVE_FLOAT,
					   H5T_NATIVE_FLOAT,
					   H5T_NATIVE_FLOAT };
  
  //compute chunk size 
  chunk_size = (size_t) (RHO_CRIT*rayTraceData.OmegaM*
                         4.0/3.0*M_PI*(pow((j+1)*rayTraceData.maxComvDistance/rayTraceData.NumLensPlanes,3.0)-pow(j*rayTraceData.maxComvDistance/rayTraceData.NumLensPlanes,3.0))
                         /wb->LCParts[j][0].mass/10.0/order2npix(rayTraceData.LensPlaneOrder));
  if(chunk_size < 10)
    chunk_size = 10;
  if(chunk_size > 5000)
    chunk_size = 5000;
  
  //write this plane to disk
  fprintf(stderr,"\twriting to plane %ld: # of parts allocated = %ld, # of parts used = %ld (%.2f percent)\n",
	  j,wb->NumLCParts[j],wb->NumLCPartsUsed[j],((double) (wb->NumLCPartsUsed[j]))/((double) (wb->MaxTotNumLCParts))*100.0);
  
  //get indexing vector to sort particles by peano index
  for(k=0;k<wb->NumLCPartsUsed[j];++k)
    {
      vec[0] = wb->LCParts[j][k].px;
      vec[1] = wb->LCParts[j][k].py;
      vec[2] = wb->LCParts[j][k].pz;
      wb->PeanoInds[k] = nest2peano(vec2nest(vec,wb->HEALPixOrder),wb->HEALPixOrder);
    }
  gsl_sort_long_index(wb->PeanoSortInds,wb->PeanoInds,(size_t) 1,(size_t) (wb->NumLCPartsUsed[j]));
  
  //open the file and read its index
  sprintf(file_name,"%s/%s%04ld.h5",rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,j);
  file_id = H5Fopen(file_name,H5F_ACC_RDWR,fapl);
  assert(file_id >= 0);
  status = H5LTread_dataset(file_id,"/NumLCPartsInPix",H5T_NATIVE_LONG,wb->NumLCPartsInPix);
  assert(status >= 0);
  for(k=0;k<wb->NPix;++k)
    wb->NumLCPartsInPixUpdate[k] = wb->NumLCPartsInPix[k];
		      
  k = 0;
  LCPartWriteBuffInd = 0;
  wb->LCPartWriteBuff[LCPartWriteBuffInd] = wb->LCParts[j][wb->PeanoSortInds[k]];
  wb->NumLCPartsInPixUpdate[wb->PeanoInds[wb->PeanoSortInds[k]]] += 1;
  LCPartWriteBuffInd = 1;
  k = 1;
  while(k < wb->NumLCPartsUsed[j])
    {
      //if(k%100000 == 0)
      //fprintf(stderr,"k = %ld of %ld\n",k,wb->NumLCPartsUsed[j]);
      
      //append previous LCPartWriteBuff if peano ind has changed
      if(wb->PeanoInds[wb->PeanoSortInds[k]] != wb->PeanoInds[wb->PeanoSortInds[k-1]])
	{
	  //make table if needed
	  if(wb->NumLCPartsInPix[wb->PeanoInds[wb->PeanoSortInds[k-1]]] == 0)
	    {
	      sprintf(tablename,"PeanoInd%ld",wb->PeanoInds[wb->PeanoSortInds[k-1]]);
	      //fprintf(stderr,"\tmaking table '%s' with %ld particles\n",tablename,LCPartWriteBuffInd);
	      status = H5TBmake_table(tablename,file_id,tablename,NFIELDS_LCPARTICLE,(hsize_t) LCPartWriteBuffInd,
				      dst_size,field_names,dst_offset,field_type,
				      chunk_size,fill_data,compress,wb->LCPartWriteBuff);
	      assert(status >= 0);
	    }
	  else
	    {
	      //append LCPartWriteBuff
	      sprintf(tablename,"PeanoInd%ld",wb->PeanoInds[wb->PeanoSortInds[k-1]]);
	      //fprintf(stderr,"\tappending table '%s' with %ld particles\n",tablename,LCPartWriteBuffInd);
	      status = H5TBappend_records(file_id,tablename,(hsize_t) LCPartWriteBuffInd,dst_size,dst_offset,dst_sizes,wb->LCPartWriteBuff);
	      assert(status >= 0);
	    }
	  
	  //reset ind
	  LCPartWriteBuffInd = 0;
	}
			  
      //add particle to LCPartWriteBuff 
      assert(LCPartWriteBuffInd < wb->NumLCPartWriteBuff && LCPartWriteBuffInd >= 0);
      wb->LCPartWriteBuff[LCPartWriteBuffInd] = wb->LCParts[j][wb->PeanoSortInds[k]];
      ++LCPartWriteBuffInd;
      
      //add one to index vector
      wb->NumLCPartsInPixUpdate[wb->PeanoInds[wb->PeanoSortInds[k]]] += 1;
      
      ++k;
    };
  
  //append last set of particles
  if(LCPartWriteBuffInd != 0)
    {
      //make table if needed                                                                                                                      
      if(wb->NumLCPartsInPix[wb->PeanoInds[wb->PeanoSortInds[k-1]]] == 0)
	{
	  sprintf(tablename,"PeanoInd%ld",wb->PeanoInds[wb->PeanoSortInds[k-1]]);
	  //fprintf(stderr,"\tmaking table '%s' with %ld particles\n",tablename,LCPartWriteBuffInd);
	  status = H5TBmake_table(tablename,file_id,tablename,NFIELDS_LCPARTICLE,(hsize_t) LCPartWriteBuffInd,
				  dst_size,field_names,dst_offset,field_type,
				  chunk_size,fill_data,compress,wb->LCPartWriteBuff);
	  assert(status >= 0);
	}
      else
	{
	  //append LCPartWriteBuff
	  sprintf(tablename,"PeanoInd%ld",wb->PeanoInds[wb->PeanoSortInds[k-1]]);
	  //fprintf(stderr,"\tappending table '%s' with %ld particles\n",tablename,LCPartWriteBuffInd);
	  status = H5TBappend_records(file_id,tablename,(hsize_t) LCPartWriteBuffInd,dst_size,dst_offset,dst_sizes,wb->LCPartWriteBuff);
	  assert(status >= 0);
	}
    }
  
  //update index
  dataset_id = H5Dopen(file_id,"/NumLCPartsInPix",H5P_DEFAULT);
  assert(dataset_id >= 0);
  status = H5Dwrite(dataset_id,H5T_NATIVE_LONG,H5S_ALL,H5S_ALL,H5P_DEFAULT,wb->NumLCPartsInPixUpdate);
  assert(status >= 0);
  status = H5Dclose(dataset_id);
  assert(status >= 0);
  
  //flush it
  status = H5Fflush(file_id,H5F_SCOPE_GLOBAL);
  assert(status >= 0);
  
  //close the file
  status = H5Fclose(file_id);
  assert(status >= 0);
  
  status = H5Pclose(fapl);
  assert(status >= 0);
  
  LCPartWriteBuffInd = 0;
  for(k=0;k<wb->NPix;++k)
    LCPartWriteBuffInd += wb->NumLCPartsInPixUpdate[k];
  assert(LCPartWriteBuffInd == wb->TotNumLCPartsInPlane[j]);
  
  //reset count
  wb->NumLCPartsUsed[j] = 0;
  wb->NumLCParts[j] = 0;
  free(wb->LCParts[j]);
  wb->LCParts[j] = NULL;
}

void makeRayTracingPlanesHDF5(void)
{
  //vars
  long HEALPixOrder = rayTraceData.LensPlaneOrder;
  char *filelist = rayTraceData.LightConeFileList;
  hid_t file_id;
  WriteBuffData wb;
  hsize_t dims[1] = {1};
  herr_t status;
  FILE *infp=NULL,*listfp;
  long i,j,MaxTotNumLCParts,RayTracingPlaneId,Np=0;
  long Nlist,fileNum;
  long RayTracingPlaneIdMaxNumLCParts,TotNumLCPartsNonZero;
  char filename[MAX_FILENAME];
  double dcomvd,rad;
  long npplanetot=0,npplanetotcheck=0;
  LCParticle LCPartRead;
  
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  assert(fapl >= 0);
  status = H5Pset_fclose_degree(fapl,H5F_CLOSE_STRONG);
  assert(status >= 0);
  
  //set parameters of read/write and ray planes
  MaxTotNumLCParts = (long) (((double) (rayTraceData.memBuffSizeInMB))*1024.0*1024.0/((double) (sizeof(LCParticle))));
  assert(MaxTotNumLCParts > 0);
  dcomvd = (double) (rayTraceData.maxComvDistance/((float) (rayTraceData.NumLensPlanes)));
  fillWriteBuffData(&wb,(long) (rayTraceData.NumLensPlanes),HEALPixOrder,MaxTotNumLCParts);
    
  //info for the user...
  fprintf(stderr,"size of LCParts buffer = %lf MB\n",((double) (sizeof(LCParticle)*wb.MaxTotNumLCParts))/1073741824.0*1024.0);
  fprintf(stderr,"size of LCParticle chunk = %lf MB\n",((double) (sizeof(LCParticle)*wb.ChunkSizeLCParts))/1073741824.0*1024.0);
  
  //prep output files
  fprintf(stderr,"preping ray tracing plane files %ld...\n",wb.NumRayTracingPlanes);
  for(RayTracingPlaneId=0;RayTracingPlaneId<wb.NumRayTracingPlanes;++RayTracingPlaneId)
    {
      if((wb.NumRayTracingPlanes/10) != 0)
	{
	  if(RayTracingPlaneId%(wb.NumRayTracingPlanes/10) == 0)
	    fprintf(stderr,"\tRayTracingPlaneId = %ld of %ld\n",RayTracingPlaneId,wb.NumRayTracingPlanes);
	}
      else
	fprintf(stderr,"\tRayTracingPlaneId = %ld of %ld\n",RayTracingPlaneId,wb.NumRayTracingPlanes);
      
      //open correct file and write HEALPixOrder to it
      sprintf(filename,"%s/%s%04ld.h5",rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,RayTracingPlaneId);
      file_id = H5Fcreate(filename,H5F_ACC_TRUNC,H5P_DEFAULT,fapl);
      assert(file_id >= 0);
      
      //write # of planes
      dims[0] = 1;
      status = H5LTmake_dataset(file_id,"/NumLensPlanes",1,dims,H5T_NATIVE_LONG,&(rayTraceData.NumLensPlanes));
      assert(status >= 0);

      //write maxComvDistance
      dims[0] = 1;
      status = H5LTmake_dataset(file_id,"/MaxComvDistance",1,dims,H5T_NATIVE_DOUBLE,&(rayTraceData.maxComvDistance));
      assert(status >= 0);
      
      //write HEALPixOrder
      dims[0] = 1;
      status = H5LTmake_dataset(file_id,"/HEALPixOrder",1,dims,H5T_NATIVE_LONG,&(wb.HEALPixOrder));
      assert(status >= 0);
      
      //write index table for particle numbers
      dims[0] = wb.NPix;
      status = H5LTmake_dataset(file_id,"/NumLCPartsInPix",1,dims,H5T_NATIVE_LONG,wb.NumLCPartsInPix);
      assert(status >= 0);
      
      //close file
      status = H5Fclose(file_id);
      assert(status >= 0);
    }

  //open the list file
  listfp = fopen(filelist,"r");
  assert(listfp != NULL);
  Nlist = fnumlines(listfp);
  
  //loop through files and put parts in right spot
  for(fileNum=0;fileNum<Nlist;++fileNum)
    {
      //get file name
      fscanf(listfp,"%s\n",filename);
      fprintf(stderr,"reading file (%ld of %ld): %s\n",fileNum+1,Nlist,filename);
      
      //open file and read parts
      infp = fopen(filename,"rb");
      assert(infp != NULL);
      Np = getNumLCPartsFile(infp);
      /*FIXME: comment out code below
	long k;
	k = 0;
      */
      for(i=0;i<Np;++i)
	{
	  if(i%10000000 == 0)
	    fprintf(stderr,"\t%ld of %ld (%.2f percent)\n",i,Np,((double) i)/((double) Np)*100.0);
	  
	  //check if I/O buff exceeds max
	  if(needToWriteRayTracingPlanes(&wb,&RayTracingPlaneIdMaxNumLCParts,&TotNumLCPartsNonZero))
	    {
	      for(j=0;j<wb.NumRayTracingPlanes;++j)
		{
		  if((j == RayTracingPlaneIdMaxNumLCParts || TotNumLCPartsNonZero > rayTraceData.MaxNumLensPlaneInMem) && wb.NumLCPartsUsed[j] > 0)
		    {
		      writeRayTracingPlane(j,&wb);
		    }
		}
	    }
	  
	  //read particle
	  LCPartRead = getLCPartFromFile(i,Np,infp,0);
	  
	  //get radius and bin
	  LCPartRead.px = (float) (LCPartRead.px - rayTraceData.LightConeOriginX);
	  LCPartRead.py = (float) (LCPartRead.py - rayTraceData.LightConeOriginY);
	  LCPartRead.pz = (float) (LCPartRead.pz - rayTraceData.LightConeOriginZ);
	  rad = sqrt(LCPartRead.px*LCPartRead.px + LCPartRead.py*LCPartRead.py + LCPartRead.pz*LCPartRead.pz);
	  RayTracingPlaneId = (long) (rad/dcomvd);
	  
	  ////////////////////////////////////////
	  //FIXME: comment out of the code below
	  /*if(k == 10 || npplanetot == 10*2)
	    break;
	    if(k < 10 && RayTracingPlaneId < wb.NumRayTracingPlanes && RayTracingPlaneId >= 0)
	    {
	    fprintf(stderr,"partid = %ld, pos = %f|%f|%f, vel = %f|%f|%f, mass = %le, rad = %lf, planeNum = %ld\n",LCPartRead.partid,
	    LCPartRead.px,LCPartRead.py,LCPartRead.pz,
	    LCPartRead.vx,LCPartRead.vy,LCPartRead.vz,
	    LCPartRead.mass,rad,RayTracingPlaneId);
	    ++k;
	    //if(i == 1)
	    //exit(1);
	    }
	  */
	  //END of FIXME
	  //////////////////////////////
	  
	  if(RayTracingPlaneId < wb.NumRayTracingPlanes && RayTracingPlaneId >= 0)
	    {
	      //make sure we have mem
	      if(wb.LCParts[RayTracingPlaneId] == NULL)
		{
		  wb.NumLCParts[RayTracingPlaneId] = wb.ChunkSizeLCParts;
		  //fprintf(stderr,"\tallocated mem planeid = %ld, mem frac = %lf\n",RayTracingPlaneId,
		  //((double) (wb.NumLCParts[RayTracingPlaneId]))/((double) wb.MaxTotNumLCParts));
		  wb.LCParts[RayTracingPlaneId] = (LCParticle*)malloc(sizeof(LCParticle)*wb.ChunkSizeLCParts);
		  assert(wb.LCParts[RayTracingPlaneId] != NULL);
		}
	      else if(wb.NumLCPartsUsed[RayTracingPlaneId] >= wb.NumLCParts[RayTracingPlaneId])
		{
		  wb.NumLCParts[RayTracingPlaneId] += wb.ChunkSizeLCParts;
		  //fprintf(stderr,"\treallocated mem planeid = %ld, mem frac = %lf\n",RayTracingPlaneId,
		  //((double) (wb.NumLCParts[RayTracingPlaneId]))/((double) wb.MaxTotNumLCParts));
		  wb.LCParts[RayTracingPlaneId] = (LCParticle*)realloc(wb.LCParts[RayTracingPlaneId],sizeof(LCParticle)*wb.NumLCParts[RayTracingPlaneId]);
		  assert(wb.LCParts[RayTracingPlaneId] != NULL);
		}
	      
	      //fill into vecs
	      wb.LCParts[RayTracingPlaneId][wb.NumLCPartsUsed[RayTracingPlaneId]] = LCPartRead;
	      wb.NumLCPartsUsed[RayTracingPlaneId] += 1;
	      
	      //error check tot # parts
	      ++npplanetot;
	      wb.TotNumLCPartsInPlane[RayTracingPlaneId] += 1;
	    }
	  else
	    {
	      //fprintf(stderr,"weird planeid %ld: rad = %lf \n",RayTracingPlaneId,rad);
	    }
	}
      
      //close LC inputfile
      LCPartRead = getLCPartFromFile(0,Np,infp,1);
      fclose(infp);
    }
  
  //close files
  fclose(listfp);
  
  //write last set of particles
  fprintf(stderr,"writing last set of particles...\n");
  for(j=0;j<wb.NumRayTracingPlanes;++j)
    {
      if(wb.NumLCPartsUsed[j] > 0)
	{
	  writeRayTracingPlane(j,&wb);
	}
    }
  
  //check number of parts  
  npplanetotcheck = 0;
  for(j=0;j<wb.NumRayTracingPlanes;++j)
    npplanetotcheck += wb.TotNumLCPartsInPlane[j];
  fprintf(stderr,"check total number of LC particles: TotNumLCParts = %ld, TotNumLCPartsCheck = %ld\n",npplanetotcheck,npplanetot);
  assert(npplanetot == npplanetotcheck);
  
  //free mem
  freeWriteBuffData(&wb);
  
  status = H5Pclose(fapl);
  assert(status >= 0);
}
