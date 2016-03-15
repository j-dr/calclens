#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <gsl/gsl_sort_long.h>
#include <gsl/gsl_rng.h>

#include "raytrace.h"
#include "read_lensplanes_pixLC.h"

void readRayTracingPlaneAtPeanoInds_pixLC(long planeNum, long HEALPixOrder, long *PeanoIndsToRead, long NumPeanoIndsToRead, Part **LCParts, long *NumLCParts)
{
  char file_name[MAX_FILENAME];
  long i,ind,j,k,FileHEALPixOrder,*NumLCPartsInPix,FileNPix;
  long *PeanoIndsToReadFromFile,NumPeanoIndsToReadFromFile;
  long KeepLCPart,LCPartPeanoInd;
  double vec[3];
  Part LCPartRead;
  
  // FIXME stub
  NumLCParts = 0;
  *LCParts = NULL;
  
  // open file
  
  /*if(file_id < 0)
    {
      fprintf(stderr,"%d: lens plane '%s' could not be opened!\n",ThisTask,file_name);
      assert(0);
    }
  
#ifdef KEEP_RAND_FRAC 
  if(ThisTask == 0)
    {
      fprintf(stderr,"keeping only 1 of %lg of particles.\n",ThisTask,1.0/RAND_FRAC_TO_KEEP);
      fflush(stderr);
    }
  
  gsl_rng *rng;
  rng = gsl_rng_alloc(gsl_rng_ranlxd2);
#endif
  
  /// read info about file

  status = H5LTread_dataset(file_id,"/HEALPixOrder",H5T_NATIVE_LONG,&FileHEALPixOrder);
  assert(status >= 0);
  FileNPix = order2npix(FileHEALPixOrder);
  NumLCPartsInPix = (long*)malloc(sizeof(long)*FileNPix);
  status = H5LTread_dataset(file_id,"/NumLCPartsInPix",H5T_NATIVE_LONG,NumLCPartsInPix);
  assert(status >= 0);
  //getPeanoIndsToReadFromFile(HEALPixOrder,PeanoIndsToRead,NumPeanoIndsToRead,FileHEALPixOrder,&PeanoIndsToReadFromFile,&NumPeanoIndsToReadFromFile);
  
  *NumLCParts = 0;
  for(i=0;i<NumPeanoIndsToReadFromFile;++i)
    *NumLCParts = *NumLCParts + NumLCPartsInPix[PeanoIndsToReadFromFile[i]];
  
  if(*NumLCParts > 0)
    {
      *LCParts = (Part*)malloc(sizeof(Part)*(*NumLCParts));
      assert(*LCParts != NULL);
      
      ind = 0;
      for(i=0;i<NumPeanoIndsToReadFromFile;++i)
	{
	  if(NumLCPartsInPix[PeanoIndsToReadFromFile[i]] > 0)
	    {
	      sprintf(tablename,"PeanoInd%ld",PeanoIndsToReadFromFile[i]);
	      status = H5TBread_fields_name(file_id,tablename,"px,py,pz,mass",0,NumLCPartsInPix[PeanoIndsToReadFromFile[i]], 
					    dst_size,field_offset,dst_sizes,*LCParts+ind);
	      assert(status >= 0);
	      
#ifdef KEEP_RAND_FRAC 
	      gsl_rng_set(rng,(unsigned long) (PeanoIndsToReadFromFile[i]));
	      j = 0;
	      for(k=0;k<NumLCPartsInPix[PeanoIndsToReadFromFile[i]];++k)
		{
		  if(gsl_rng_uniform(rng) < RAND_FRAC_TO_KEEP)
		    {
		      (*LCParts)[ind+j] = (*LCParts)[ind+k];
		      (*LCParts)[ind+j].mass = (*LCParts)[ind+j].mass/RAND_FRAC_TO_KEEP;
		      ++j;
		    }
		}
	      ind += j;
#else
	      ind += NumLCPartsInPix[PeanoIndsToReadFromFile[i]];
#endif
	    }
	}
      
#ifdef KEEP_RAND_FRAC 
      *NumLCParts = ind;
      if(*NumLCParts > 0)
	{
	  *LCParts = (Part*)realloc(*LCParts,sizeof(Part)*(*NumLCParts));
	  assert(*LCParts != NULL);
	}
      else
	{
	  free(*LCParts);
	  *NumLCParts = 0;
	  *LCParts = NULL;
	}
#endif

  /// if file cells are larger than requested cells, cull extra particles
      if(FileHEALPixOrder < HEALPixOrder && (*NumLCParts) > 0)
	{
	  ind = 0;
	  for(i=0;i<(*NumLCParts);++i)
	    {
	      vec[0] = (double) ((*LCParts)[i].pos[0]);
	      vec[1] = (double) ((*LCParts)[i].pos[1]);
	      vec[2] = (double) ((*LCParts)[i].pos[2]);
	      LCPartPeanoInd = nest2peano(vec2nest(vec,HEALPixOrder),HEALPixOrder);
	      
	      KeepLCPart = 0;
	      for(j=0;j<NumPeanoIndsToRead;++j)
		{
		  if(LCPartPeanoInd == PeanoIndsToRead[j])
		    {
		      KeepLCPart = 1;
		      break;
		    }
		}

	      if(KeepLCPart)
		{
		  (*LCParts)[ind] = (*LCParts)[i];
		  ++ind;
		}
	    }
	  
	  *NumLCParts = ind;
	  if(*NumLCParts > 0)
	    {
	      *LCParts = (Part*)realloc(*LCParts,sizeof(Part)*(*NumLCParts));
	      assert(*LCParts != NULL);
	    }
	  else
	    {
	      free(*LCParts);
	      *NumLCParts = 0;
	      *LCParts = NULL;
	    }
	}
    }
  else
    {
      *NumLCParts = 0;
      *LCParts = NULL;
    }

  // free mem
  free(PeanoIndsToReadFromFile);
  free(NumLCPartsInPix);
#ifdef KEEP_RAND_FRAC 
  gsl_rng_free(rng);
#endif
  
  /// close file
  status = H5Fclose(file_id);
  assert(status >= 0);
*/
}
