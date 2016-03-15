#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <mpi.h>
#include <gsl/gsl_sort_long.h>
#include <ctype.h>
#include <string.h>

#include "raytrace.h"

//routine to do case insens. string comp.
int strcmp_caseinsens(const char *s1, const char *s2)
{
  int N,i,equal;
  
  if(strlen(s1) != strlen(s2))
    return 1;
  N = strlen(s1);

#ifdef DEBUG
#if DEBUG_LEVEL > 2  
  if(ThisTask == 0)
    fprintf(stderr,"s1 = '%s', s2 = '%s'\n",s1,s2);
#endif
#endif

  equal = 0;
  for(i=0;i<N;++i)
    {
#ifdef DEBUG
#if DEBUG_LEVEL > 2
      if(ThisTask == 0)
        fprintf(stderr,"s1[i] = '%c', s2[i] = '%c'\n",tolower(s1[i]),tolower(s2[i]));
#endif
#endif

      if(tolower(s1[i]) != tolower(s2[i]))
        {
          equal = 1;
          break;
        }
    }

#ifdef DEBUG
#if DEBUG_LEVEL > 2  
  if(ThisTask == 0)
    fprintf(stderr,"equal %d (0 is true, 1 is false, weird)\n",equal);
#endif
#endif

  return equal;
}

/* do M to N on peano inds for reading lens planes */
void getPeanoIndsToReadFromFile(long HEALPixOrder, long *PeanoIndsToRead, long NumPeanoIndsToRead,
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

/*gets the number of lines in a file*/
long fnumlines(FILE *fp)
{
  long i=-1;
  char c[5000];
  while(!feof(fp))
    {
      ++i;
      fgets(c,5000,fp);
    }
  rewind(fp);
  return i;
}

FILE *fopen_retry(const char *filename, const char *mode)
{
  int try,Ntry = 10;
  FILE *fp;
  
  //try Ntry times, if opens, return fp
  for(try=0;try<Ntry;++try)
    {
      fp = fopen(filename,mode);
      
      if(fp != NULL)
	return fp;
    }

  //if we get to here, return NULL
  return NULL;
}
