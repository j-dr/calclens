#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <string.h>
#include <sys/stat.h>

#include "raytrace.h"

static int strcmp_caseinsens(const char *s1, const char *s2);

#define ASSIGN_CONFIG_STR(TAG) if(strcmp_caseinsens(tag,#TAG) == 0) { strcpy(rayTraceData.TAG,val); continue; }
#define ASSIGN_CONFIG_LONG(TAG) if(strcmp_caseinsens(tag,#TAG) == 0) { rayTraceData.TAG = atol(val); continue; }
#define ASSIGN_CONFIG_DOUBLE(TAG) if(strcmp_caseinsens(tag,#TAG) == 0) { rayTraceData.TAG = atof(val); continue; }

void read_config(char *filename)
{
  char cmd[4096];
  FILE *fp;
  char fline[1024];
  int i,len,loc;
  char *tag,*val;
  
  //set defaults
  rayTraceData.NFFT = -1;
  rayTraceData.MaxNFFT = -1;
  rayTraceData.ThreeDPotSnapList[0] = '\0';
  rayTraceData.LengthConvFact = -1.0;
  
  //read file
  fp = fopen(filename,"r");
  if(fp == NULL)
    {
      fprintf(stderr,"Config file '%s' could not be opened!\n",filename);
      assert(0);
    }
  
  while(fgets(fline,1024,fp) != NULL)
    {
      //start at front
      len = strlen(fline);
      loc = 0;
      
      //remove any new lines or tabs and turn them into spaces
      for(i=0;i<len;++i)
	if(fline[i] == '\t' || fline[i] == '\n')
	  fline[i] = ' ';
      
      //if line is comment then skip it
      if(fline[loc] == '#')
	continue;
      
      //skip white space
      while(fline[loc] == ' ' && loc < len)
	++loc;
      
      //make sure line is not pure white space or a weird comment
      if(loc == len || fline[loc] == '#')
	continue;
      
      //get tag and null terminate
      tag = fline + loc;
      while(fline[loc] != ' ' && loc < len)
	++loc;
      fline[loc] = '\0';
      ++loc;
      
      //skip white space
      while(fline[loc] == ' ' && loc < len)
	++loc;
      
      //make sure line is not pure white space or a weird comment
      if(loc == len || fline[loc] == '#')
	{
	  fprintf(stderr,"found tag '%s' without a value in config file '%s'!\n",tag,filename);
	  assert(0);
	}
      
      //get value
      val = fline + loc;
      while(fline[loc] != ' ' && loc < len)
	++loc;
      fline[loc] = '\0';
      
      //now test tag against structr and get value
      ASSIGN_CONFIG_DOUBLE(OmegaM);
      ASSIGN_CONFIG_DOUBLE(maxComvDistance);
      ASSIGN_CONFIG_LONG(NumLensPlanes);
      
      ASSIGN_CONFIG_LONG(MaxNFFT);
      ASSIGN_CONFIG_STR(ThreeDPotSnapList);
      ASSIGN_CONFIG_DOUBLE(LengthConvFact);
      
      //fprintf(stderr,"Tag-value pair ('%s','%s') not found in config file '%s'!\n",tag,val,filename);
      //assert(0);
    }
  
  //close files
  fclose(fp);
  
  //error check
  assert(strlen(rayTraceData.ThreeDPotSnapList) > 0);
  assert(rayTraceData.MaxNFFT > 0);
  assert(rayTraceData.LengthConvFact > 0.0);
}

//routine to do case insens. string comp.
static int strcmp_caseinsens(const char *s1, const char *s2)
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
