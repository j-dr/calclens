#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>
#include <string.h>
#include <sys/stat.h>

#include "raytrace.h"

static int strcmp_caseinsens(const char *s1, const char *s2);

#define ASSIGN_CONFIG_STR(TAG) if(strcmp_caseinsens(tag,#TAG) == 0) { strcpy(rayTraceData.TAG,val); fprintf(usedfp,"%s %s\n",#TAG,rayTraceData.TAG); continue; }
#define ASSIGN_CONFIG_LONG(TAG) if(strcmp_caseinsens(tag,#TAG) == 0) { rayTraceData.TAG = atol(val); fprintf(usedfp,"%s %ld\n",#TAG,rayTraceData.TAG); continue; }
#define ASSIGN_CONFIG_DOUBLE(TAG) if(strcmp_caseinsens(tag,#TAG) == 0) { rayTraceData.TAG = atof(val); fprintf(usedfp,"%s %lg\n",#TAG,rayTraceData.TAG); continue; }

void read_config(char *filename)
{
  char usedfile[MAX_FILENAME];
  char cmd[4096];
  FILE *usedfp,*fp;
  char fline[1024];
  int i,len,loc;
  char *tag,*val;
  
  //set defaults
  int readLightConeOrigin[3] = {0,0,0};
  rayTraceData.MaxNumLensPlaneInMem = -1;
  rayTraceData.LightConePartChunkFactor = -1.0;
  rayTraceData.partMass = -1.0;
  rayTraceData.galRadPointNFWTest = -1.0;
  
  //make output dir
  mkdir(rayTraceData.OutputPath,02755);
  
  //open file to hold usedvalues
  sprintf(usedfile,"%s-usedvalues",filename);
  usedfp = fopen(usedfile,"w");
  assert(usedfp != NULL);
  
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
      ASSIGN_CONFIG_STR(OutputPath);
      ASSIGN_CONFIG_LONG(NumFilesIOInParallel);

      ASSIGN_CONFIG_DOUBLE(OmegaM);
      ASSIGN_CONFIG_DOUBLE(maxComvDistance);
      ASSIGN_CONFIG_LONG(NumLensPlanes);
      ASSIGN_CONFIG_STR(LensPlanePath);
      ASSIGN_CONFIG_STR(LensPlaneName);
      
      ASSIGN_CONFIG_STR(LightConeFileList);
      ASSIGN_CONFIG_STR(LightConeFileType);
      
      if(strcmp_caseinsens(tag,"LightConeOriginX") == 0)
	readLightConeOrigin[0] = 1;
      if(strcmp_caseinsens(tag,"LightConeOriginY") == 0)
	readLightConeOrigin[1] = 1;
      if(strcmp_caseinsens(tag,"LightConeOriginZ") == 0)
	readLightConeOrigin[2] = 1;
      ASSIGN_CONFIG_DOUBLE(LightConeOriginX);
      ASSIGN_CONFIG_DOUBLE(LightConeOriginY);
      ASSIGN_CONFIG_DOUBLE(LightConeOriginZ);

      ASSIGN_CONFIG_LONG(LensPlaneOrder);
      ASSIGN_CONFIG_LONG(NumDivLensPlane);
      ASSIGN_CONFIG_DOUBLE(memBuffSizeInMB);
      
      ASSIGN_CONFIG_LONG(MaxNumLensPlaneInMem);
      ASSIGN_CONFIG_DOUBLE(LightConePartChunkFactor);
      
      ASSIGN_CONFIG_DOUBLE(partMass);
      ASSIGN_CONFIG_DOUBLE(MassConvFact);
      ASSIGN_CONFIG_DOUBLE(LengthConvFact);
      ASSIGN_CONFIG_DOUBLE(VelocityConvFact);
      
      ASSIGN_CONFIG_DOUBLE(raPointMass);
      ASSIGN_CONFIG_DOUBLE(decPointMass);
      ASSIGN_CONFIG_DOUBLE(radPointMass);
      ASSIGN_CONFIG_DOUBLE(galRadPointNFWTest);
      
      //fprintf(stderr,"Tag-value pair ('%s','%s') not found in config file '%s'!\n",tag,val,filename);
      //assert(0);
    }
  
  //close files
  fclose(usedfp);
  fclose(fp);
  
  //copy config file to output folder
  sprintf(cmd,"cp %s %s/lensplane.cfg",usedfile,rayTraceData.OutputPath);
  system(cmd);
  
  //error check
  assert(readLightConeOrigin[0] == 1);
  assert(readLightConeOrigin[1] == 1);
  assert(readLightConeOrigin[2] == 1);
  assert(rayTraceData.MaxNumLensPlaneInMem > 0);
  assert(rayTraceData.LightConePartChunkFactor > 0);
  
  //error check for point mass test
#ifdef POINTMASSTEST
  assert(rayTraceData.galRadPointNFWTest > 0.0);
#endif
#ifdef NFWHALOTEST
  assert(rayTraceData.galRadPointNFWTest > 0.0);
#endif
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
