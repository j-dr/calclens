#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>
#include <gsl/gsl_heapsort.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "raytrace.h"

//make a healpix map of the lens plane for debugging
void make_lensplane_map(long planeNum)
{
  long order = 10; //must be <= 13
  assert(order <= 13);
  long Npix = order2npix(order);
  float *map,*totmap;
  
  long FileHEALPixOrder,FileNPix,*NumLCPartsInPix;
  Part *Parts;
  long NumParts;
  char file_name[MAX_FILENAME];
  
  hid_t file_id;
  herr_t status;
  
  long minInd,maxInd,NumIndsPerTask;
  long PeanoIndsToRead,NumPeanoIndsToRead;
  long i,j;
  
  double vec[3];
  double r,theta,phi;
  long ring;
  FILE *fp;

  /* read info about file */  
  sprintf(file_name,"%s/%s%04ld.h5",rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,planeNum);
  file_id = H5Fopen(file_name,H5F_ACC_RDONLY,H5P_DEFAULT);
  if(file_id < 0)
    {
      fprintf(stderr,"%d: lens plane %ld could not be opened!\n",ThisTask,planeNum);
      assert(0);
    }
  status = H5LTread_dataset(file_id,"/HEALPixOrder",H5T_NATIVE_LONG,&FileHEALPixOrder);
  assert(status >= 0);
  FileNPix = order2npix(FileHEALPixOrder);
  NumLCPartsInPix = (long*)malloc(sizeof(long)*FileNPix);
  status = H5LTread_dataset(file_id,"/NumLCPartsInPix",H5T_NATIVE_LONG,NumLCPartsInPix);
  assert(status >= 0);
  status = H5Fclose(file_id);
  assert(status >= 0);
    
  map = (float*)malloc(sizeof(float)*Npix);
  assert(map != NULL);
  totmap = (float*)malloc(sizeof(float)*Npix);
  assert(totmap != NULL);
  
  for(i=0;i<Npix;++i)
    map[i] = 0.0;
  
  NumIndsPerTask = FileNPix/NTasks;
  minInd = ThisTask*NumIndsPerTask;
  maxInd = minInd + NumIndsPerTask - 1;
  if(ThisTask == NTasks-1)
    maxInd = FileNPix-1;
  
  for(i=minInd;i<=maxInd;++i)
    {
      PeanoIndsToRead = i;
      NumPeanoIndsToRead = 1;
      
      readRayTracingPlaneAtPeanoInds_HDF5(planeNum,FileHEALPixOrder,&PeanoIndsToRead,NumPeanoIndsToRead,&Parts,&NumParts);
      //fprintf(stderr,"%d: NumParts = %ld, NumLCPartsInPix = %ld\n",ThisTask,NumParts,NumLCPartsInPix[i]);
      
      assert(NumParts == NumLCPartsInPix[i]);
      
      for(j=0;j<NumParts;++j)
        {
          r = sqrt(Parts[j].pos[0]*Parts[j].pos[0] + 
                   Parts[j].pos[1]*Parts[j].pos[1] + 
	               Parts[j].pos[2]*Parts[j].pos[2]);
          Parts[j].pos[0] /= r;
          Parts[j].pos[1] /= r;
          Parts[j].pos[2] /= r;

          vec[0] = (double) (Parts[j].pos[0]);
          vec[1] = (double) (Parts[j].pos[1]);
          vec[2] = (double) (Parts[j].pos[2]);
          
          vec2ang(vec,&theta,&phi);
          ring = ang2nest(theta,phi,order);
          ring = nest2ring(ring,order);
          
          map[ring] += 1.0;
        }
      
      free(Parts);
    }
  
  free(NumLCPartsInPix);
  
  MPI_Allreduce(map,totmap,(int) Npix,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
  
  if(ThisTask == 0)
    {
      sprintf(file_name,"%s/%s_healpixmap.%ld",rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,planeNum);
      fp = fopen(file_name,"w");
      fwrite(totmap,(size_t) Npix,sizeof(float),fp);
      fclose(fp);
    }
  
  free(map);
  free(totmap);
}

