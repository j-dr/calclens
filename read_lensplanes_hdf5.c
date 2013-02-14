#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <gsl/gsl_sort_long.h>
#include <gsl/gsl_rng.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#include "raytrace.h"

static void getPeanoIndsToReadFromFile(long HEALPixOrder, long *PeanoIndsToRead, long NumPeanoIndsToRead,
				       long FileHEALPixOrder, long **FilePeanoIndsToRead, long *NumFilePeanoIndsToRead)
{
  /**********************  get inds to read from the file ********************
   *  If the order of the file differs from the order of the requested cells, then we either 
   *    1) have to read in all cells in file which are in the requested cells
   *    2) read in the cell in the file which contains the requested cell and then cull the particles
   */
  
  long FileNPix = order2npix(FileHEALPixOrder);
  long i,j,BaseNest,FileNest,OrderDiff,ind;
  long NinPix,Nextra=1000,*tmp;
  
  *FilePeanoIndsToRead = (long*)malloc(sizeof(long)*FileNPix);
  assert(*FilePeanoIndsToRead != NULL);
  *NumFilePeanoIndsToRead = FileNPix;
  
  if(FileHEALPixOrder > HEALPixOrder) /* Case #1 above: file cells are smaller, so just need to figure out which file cells are in the large cells we have requested */
    {
      OrderDiff = FileHEALPixOrder - HEALPixOrder;
      NinPix = 1;
      NinPix = (1 << (2*OrderDiff));
      
      ind = 0;
      for(i=0;i<NumPeanoIndsToRead;++i)
	{
	  BaseNest = peano2nest(PeanoIndsToRead[i],HEALPixOrder);
	  FileNest = (BaseNest << (2*OrderDiff));
	  
	  for(j=0;j<NinPix;++j)
	    {
	      (*FilePeanoIndsToRead)[ind] = nest2peano(FileNest+j,FileHEALPixOrder);
	      ++ind;
	      
	      if(ind >= (*NumFilePeanoIndsToRead))
		{
		  tmp = (long*)realloc(*FilePeanoIndsToRead,sizeof(long)*((*NumFilePeanoIndsToRead)+Nextra));
		  assert(tmp != NULL);
		  *FilePeanoIndsToRead = tmp;
		  (*NumFilePeanoIndsToRead) = (*NumFilePeanoIndsToRead) + Nextra;
		}
	    }
	}
      
      *NumFilePeanoIndsToRead = ind;
      tmp = (long*)realloc(*FilePeanoIndsToRead,sizeof(long)*(*NumFilePeanoIndsToRead));
      assert(tmp != NULL);
      *FilePeanoIndsToRead = tmp;
    }
  else if(FileHEALPixOrder < HEALPixOrder) /* Case #2 above: file cells are larger, so need to convert requested cells to file cells */
    {
      OrderDiff = HEALPixOrder - FileHEALPixOrder;
      
      ind = 0;
      for(i=0;i<NumPeanoIndsToRead;++i)
	{
	  BaseNest = peano2nest(PeanoIndsToRead[i],HEALPixOrder);
	  FileNest = BaseNest >> (2*OrderDiff); 
	  
	  (*FilePeanoIndsToRead)[ind] = nest2peano(FileNest,FileHEALPixOrder);
	  ++ind;
	  
	  if(ind >= (*NumFilePeanoIndsToRead))
	    {
	      tmp = (long*)realloc(*FilePeanoIndsToRead,sizeof(long)*((*NumFilePeanoIndsToRead)+Nextra));
	      assert(tmp != NULL);
	      *FilePeanoIndsToRead = tmp;
	      (*NumFilePeanoIndsToRead) = (*NumFilePeanoIndsToRead) + Nextra;
	    }
	}
      *NumFilePeanoIndsToRead = ind;
      tmp = (long*)realloc(*FilePeanoIndsToRead,sizeof(long)*(*NumFilePeanoIndsToRead));
      assert(tmp != NULL);
      *FilePeanoIndsToRead = tmp;
    }
  else /*  order is the same, so just copy the inds */
    {
      *NumFilePeanoIndsToRead = NumPeanoIndsToRead;
      *FilePeanoIndsToRead = (long*)realloc(*FilePeanoIndsToRead,sizeof(long)*(*NumFilePeanoIndsToRead));
      assert(*FilePeanoIndsToRead != NULL);
      
      for(j=0;j<NumPeanoIndsToRead;++j)
	(*FilePeanoIndsToRead)[j] = PeanoIndsToRead[j];
    }
  
  /* remove duplicates */
  gsl_sort_long(*FilePeanoIndsToRead,(size_t) 1,(size_t) (*NumFilePeanoIndsToRead));
  ind = 1;
  for(j=1;j<(*NumFilePeanoIndsToRead);++j)
    {
      if((*FilePeanoIndsToRead)[j] != (*FilePeanoIndsToRead)[ind-1])
	{
	  (*FilePeanoIndsToRead)[ind] = (*FilePeanoIndsToRead)[j];
	  ++ind;
	}
    }
  *NumFilePeanoIndsToRead = ind;
  tmp = (long*)realloc(*FilePeanoIndsToRead,sizeof(long)*(*NumFilePeanoIndsToRead));
  assert(tmp != NULL);
  *FilePeanoIndsToRead = tmp;
}

//#define KEEP_RAND_FRAC 
#ifdef KEEP_RAND_FRAC 
#define RAND_FRAC_TO_KEEP (1.0/64.0)
#endif

void readRayTracingPlaneAtPeanoInds(hid_t *file_id, long HEALPixOrder, long *PeanoIndsToRead, long NumPeanoIndsToRead, Part **LCParts, long *NumLCParts)
{
  herr_t status;
  char tablename[MAX_FILENAME];
  long i,ind,j,FileHEALPixOrder,*NumLCPartsInPix,FileNPix;
  long *PeanoIndsToReadFromFile,NumPeanoIndsToReadFromFile;
  long KeepLCPart,LCPartPeanoInd;
  double vec[3];
  Part LCPartRead;
  
#ifdef KEEP_RAND_FRAC 
  if(ThisTask == 0)
    fprintf(stderr,"%d: keeping a random fraction of %le of particles\n",ThisTask,RAND_FRAC_TO_KEEP);
  
  static gsl_rng *rng = NULL;
  if(rng == NULL)
    {
      rng = gsl_rng_alloc(gsl_rng_ranlxd2);
      gsl_rng_set(rng,(unsigned long) (ThisTask+1));
    }
#endif
  
  /* define LCParticle type in HDF5 for table I/O */
  size_t dst_size = sizeof(Part);
  size_t dst_sizes[4] = { sizeof(LCPartRead.pos[0]),
			  sizeof(LCPartRead.pos[1]),
			  sizeof(LCPartRead.pos[2]),
			  sizeof(LCPartRead.mass) };
  size_t field_offset[4] = { HOFFSET(Part,pos[0]),
			     HOFFSET(Part,pos[1]),
			     HOFFSET(Part,pos[2]),
			     HOFFSET(Part,mass) };
  
  /* read info about file */
  status = H5LTread_dataset(*file_id,"/HEALPixOrder",H5T_NATIVE_LONG,&FileHEALPixOrder);
  assert(status >= 0);
  FileNPix = order2npix(FileHEALPixOrder);
  NumLCPartsInPix = (long*)malloc(sizeof(long)*FileNPix);
  status = H5LTread_dataset(*file_id,"/NumLCPartsInPix",H5T_NATIVE_LONG,NumLCPartsInPix);
  assert(status >= 0);
  
  getPeanoIndsToReadFromFile(HEALPixOrder,PeanoIndsToRead,NumPeanoIndsToRead,FileHEALPixOrder,&PeanoIndsToReadFromFile,&NumPeanoIndsToReadFromFile);
  
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
	      status = H5TBread_fields_name(*file_id,tablename,"px,py,pz,mass",0,NumLCPartsInPix[PeanoIndsToReadFromFile[i]], 
					    dst_size,field_offset,dst_sizes,*LCParts+ind);
	      assert(status >= 0);
	      
	      ind += NumLCPartsInPix[PeanoIndsToReadFromFile[i]];
	    }
	}
      
      /* if file cells are larger than requested cells, cull extra particles */
      if(FileHEALPixOrder < HEALPixOrder)
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
      
#ifdef KEEP_RAND_FRAC
      if(*NumLCParts > 0)
	{
	  ind = 0;
	  for(i=0;i<(*NumLCParts);++i)
	    {
	      (*LCParts)[i].mass = (*LCParts)[i].mass/RAND_FRAC_TO_KEEP;
	      if(gsl_rng_uniform(rng) < RAND_FRAC_TO_KEEP)
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
#endif
      
    }
  else
    {
      *NumLCParts = 0;
      *LCParts = NULL;
    }
  
  free(PeanoIndsToReadFromFile);
  free(NumLCPartsInPix);
}
