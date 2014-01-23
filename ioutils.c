#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <mpi.h>

#include "raytrace.h"

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
