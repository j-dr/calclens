#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <gsl/gsl_sort_long.h>
#include <gsl/gsl_rng.h>
#include <unistd.h>

#include "raytrace.h"
#include "read_lensplanes_pixLC.h"

struct pixLCheader {
  unsigned long npart;      // number of particles in the present file
  unsigned int indexnside;  // nside value used to sort particles within this file
  unsigned int filenside;   // nside used to break up radial bin this file falls in
  float rmin;               // minimum radius kept from this box
  float rmax;               // maximum radius
  unsigned long npartrad;   // number of particles in this radial bin
  float boxsize;            // in Mpc/h
  double mass;              // particle mass in 1e10 M_sun/h
  double omega0;            // omegaM
  double omegalambda;       // omegaL
  double hubbleparam;       // little 'h'
};

void readRayTracingPlaneAtPeanoInds_pixLC(long planeNum, long HEALPixOrder, long *PeanoIndsToRead, long NumPeanoIndsToRead, Part **LCParts, long *NumLCParts)
{
  char file_name[MAX_FILENAME];
  long i,ind,j,k,FileHEALPixOrder,nest;
  long *PeanoIndsToReadFromFile,NumPeanoIndsToReadFromFile;
  long KeepLCPart,LCPartPeanoInd;
  double vec[3];
  struct pixLCheader head;
  FILE *fp;
  float *pos = NULL;
  long npos = 0;

  if(ThisTask == 0)
    fprintf(stderr,"reading parts from lens plane '%s/%s_%ld_NESTIND'\n",
	    rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,planeNum);
  
#ifdef KEEP_RAND_FRAC 
  if(ThisTask == 0)
    {
      fprintf(stderr,"keeping only 1 of %lg of particles.\n",ThisTask,1.0/RAND_FRAC_TO_KEEP);
      fflush(stderr);
    }
  
  gsl_rng *rng;
  rng = gsl_rng_alloc(gsl_rng_ranlxd2);
#endif
  
  // get file layout
  FileHEALPixOrder = -1;
  for(i=0;i<1000000000;++i) 
    {
      // try to read file name
      sprintf(file_name,"%s/%s_%ld_%ld",rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,planeNum,i);
      fp = fopen(file_name,"r");
      if(fp == NULL)
	continue;
      
      // read header
      j = fread(&head,sizeof(struct pixLCheader),(size_t) 1,fp);
      if(j < 1)
	{
	  fprintf(stderr,"%d: could not read header for lens plane '%s'!\n",ThisTask,file_name);
	  assert(0);	  
	}
      
      // close file
      fclose(fp);
      
      // get order
      FileHEALPixOrder = head.filenside;
      FileHEALPixOrder = nside2order(FileHEALPixOrder);
      break;
    }
  
  // if we get here, could not find a file
  if(FileHEALPixOrder == -1) 
    {
      fprintf(stderr,"%d: could not get healpix order for lens plane %04ld!\n",ThisTask,planeNum);
      assert(0);      
    }
  
  // get peano inds to read from file
  getPeanoIndsToReadFromFile(HEALPixOrder,PeanoIndsToRead,NumPeanoIndsToRead,FileHEALPixOrder,&PeanoIndsToReadFromFile,&NumPeanoIndsToReadFromFile);
  
  // FIXME stub
  *NumLCParts = 0;
  *LCParts = NULL;
  
  ind = 0;
  for(i=0;i<NumPeanoIndsToReadFromFile;++i)
    {
      // get file name
      nest = peano2nest(PeanoIndsToReadFromFile[i],FileHEALPixOrder);
      sprintf(file_name,"%s/%s_%ld_%ld",rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,planeNum,nest);
      
      // file does not exist, move on
      if(access(file_name,F_OK) == -1)
	continue;

      // open file to read
      fp = fopen(file_name,"r");
      if(fp == NULL)
	{
	  fprintf(stderr,"%d: lens plane '%s' could not be opened!\n",ThisTask,file_name);
	  assert(0);
	}

      // read header
      j = fread(&head,sizeof(struct pixLCheader),(size_t) 1,fp);
      if(j < 1)
	{
	  fprintf(stderr,"%d: could not read header for lens plane '%s'!\n",ThisTask,file_name);
	  assert(0);	  
	}
      
      if(head.npart > 0)
	{
	  // realloc
	  if(head.npart > npos)
	    {
	      pos = (float*)realloc(pos,3*head.npart*sizeof(float));
	      if(pos == NULL)
		{
		  fprintf(stderr,"%d: could not realloc pos array for lens plane '%s'!\n",ThisTask,file_name);
		  assert(0);
		}
	    }
	  
	  // skip indexes
	  j = fseek(fp,sizeof(long)*nside2npix(head.indexnside),SEEK_CUR);
	  if(j != 0) 
	    {
	      fprintf(stderr,"%d: could not seek past idx array for lens plane '%s'!\n",ThisTask,file_name);
	      assert(0);
	    }
	  
	  // read data
	  j = fread(pos,3*head.npart*sizeof(float),(size_t) 1,fp);
	  if(j != 1)
	    {
	      fprintf(stderr,"%d: could not read pos array for lens plane '%s'!\n",ThisTask,file_name);
	      assert(0);	  
	    }
	  
	  // realloc output parts
	  if(head.npart + ind > *NumLCParts) 
	    {
	      *NumLCParts = (*NumLCParts) + head.npart*3; // a little buffer
	      *LCParts = (Part*)realloc(*LCParts,sizeof(Part)*(*NumLCParts));
	      assert(*LCParts != NULL);
	    }
	  
	  // put in outputs
#ifdef KEEP_RAND_FRAC 
	  gsl_rng_set(rng,(unsigned long) (PeanoIndsToReadFromFile[i]));
#endif
	  j = 0;
	  for(k=0;k<head.npart;++k)
	    {
#ifdef KEEP_RAND_FRAC
	      if(gsl_rng_uniform(rng) >= RAND_FRAC_TO_KEEP)
		continue;
#endif	      
	      (*LCParts)[ind+j].pos[0] = pos[3*k+0];
	      (*LCParts)[ind+j].pos[1] = pos[3*k+1];
	      (*LCParts)[ind+j].pos[2] = pos[3*k+2];
#ifdef KEEP_RAND_FRAC 
	      (*LCParts)[ind+j].mass = head.mass/RAND_FRAC_TO_KEEP*1e10;
#else
	      (*LCParts)[ind+j].mass = head.mass*1e10;
#endif	      
	      ++j;
	    }
	  ind += j;
	  
	} // if(head.npart > 0)

      // close the file
      fclose(fp);	  
    }  
  
  // do final realloc
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
  
  // if file cells are larger than requested cells, cull extra particles
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
  else
    {
      *NumLCParts = 0;
      *LCParts = NULL;
    }

  // free mem
  free(pos);
  free(PeanoIndsToReadFromFile);
#ifdef KEEP_RAND_FRAC 
  gsl_rng_free(rng);
#endif
}
