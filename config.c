#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <fftw3.h>
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
  rayTraceData.HEALPixLensPlaneMapPath[0] = '\0';
  rayTraceData.HEALPixLensPlaneMapName[0] = '\0';
  rayTraceData.HEALPixLensPlaneMapOrder = -1;
  rayTraceData.UseHEALPixLensPlaneMaps = 0;
  rayTraceData.RayOutputName[0] = '\0';
  rayTraceData.GalsFileList[0] = '\0';
  rayTraceData.GalOutputName[0] = '\0';
  rayTraceData.HEALPixRingWeightPath[0] = '\0';
  rayTraceData.HEALPixWindowFunctionPath[0] = '\0';
  rayTraceData.maxRayMemImbalance = 0.25;
  rayTraceData.MGConvFact = -1.0;
  rayTraceData.ComvSmoothingScale = -1.0;
  
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
      ASSIGN_CONFIG_DOUBLE(WallTimeLimit);
      ASSIGN_CONFIG_DOUBLE(WallTimeBetweenRestart);
      
      ASSIGN_CONFIG_STR(OutputPath);
      ASSIGN_CONFIG_STR(RayOutputName);
      ASSIGN_CONFIG_LONG(NumRayOutputFiles);
      ASSIGN_CONFIG_LONG(NumFilesIOInParallel);

      ASSIGN_CONFIG_DOUBLE(OmegaM);
      ASSIGN_CONFIG_DOUBLE(maxComvDistance);
      ASSIGN_CONFIG_LONG(NumLensPlanes);
      ASSIGN_CONFIG_STR(LensPlanePath);
      ASSIGN_CONFIG_STR(LensPlaneName);
      
      ASSIGN_CONFIG_STR(HEALPixLensPlaneMapPath);
      ASSIGN_CONFIG_STR(HEALPixLensPlaneMapName);
      ASSIGN_CONFIG_LONG(HEALPixLensPlaneMapOrder);
      
      ASSIGN_CONFIG_LONG(bundleOrder);
      ASSIGN_CONFIG_LONG(rayOrder);
      ASSIGN_CONFIG_DOUBLE(minRa);
      ASSIGN_CONFIG_DOUBLE(maxRa);
      ASSIGN_CONFIG_DOUBLE(minDec);
      ASSIGN_CONFIG_DOUBLE(maxDec);
      
      ASSIGN_CONFIG_LONG(SHTOrder);
      ASSIGN_CONFIG_STR(HEALPixRingWeightPath);
      ASSIGN_CONFIG_STR(HEALPixWindowFunctionPath);
      
      ASSIGN_CONFIG_DOUBLE(ComvSmoothingScale);
      ASSIGN_CONFIG_DOUBLE(maxRayMemImbalance);
      ASSIGN_CONFIG_DOUBLE(MGConvFact);
      
      ASSIGN_CONFIG_STR(GalsFileList);
      ASSIGN_CONFIG_STR(GalOutputName);
      ASSIGN_CONFIG_LONG(NumGalOutputFiles);

      fprintf(stderr,"Tag-value pair ('%s','%s') not found in config file '%s'!\n",tag,val,filename);
      assert(0);
    }
  
  //close files
  fclose(usedfp);
  fclose(fp);
  
  //copy config file to output folder
  sprintf(cmd,"cp %s %s/raytrace.cfg",usedfile,rayTraceData.OutputPath);
  system(cmd);
  
  //error check
  assert(rayTraceData.maxRayMemImbalance > 0.0);
  
  assert(rayTraceData.ComvSmoothingScale > 0.0);
  rayTraceData.minComvSmoothingScale = rayTraceData.ComvSmoothingScale;
  rayTraceData.maxComvSmoothingScale = rayTraceData.ComvSmoothingScale;
  
  if(strlen(rayTraceData.RayOutputName) > 0)
    {
      assert(rayTraceData.NumFilesIOInParallel <= rayTraceData.NumRayOutputFiles);
      assert(rayTraceData.NumRayOutputFiles <= NTasks);
      assert(rayTraceData.NumRayOutputFiles > 0);
    }
  
  assert(rayTraceData.rayOrder >= rayTraceData.bundleOrder);
  assert(rayTraceData.SHTOrder >= rayTraceData.bundleOrder);
    
  if(strlen(rayTraceData.GalsFileList) > 0)
    {
      assert(strlen(rayTraceData.GalOutputName) > 0);
      assert(rayTraceData.NumFilesIOInParallel <= rayTraceData.NumGalOutputFiles);
      assert(rayTraceData.NumGalOutputFiles <= NTasks);
      assert(rayTraceData.NumGalOutputFiles > 0);
    }
  
  if(strlen(rayTraceData.HEALPixLensPlaneMapPath) > 0 || strlen(rayTraceData.HEALPixLensPlaneMapName) > 0 || rayTraceData.HEALPixLensPlaneMapOrder >= 0)
    {
      assert(strlen(rayTraceData.HEALPixLensPlaneMapPath) > 0);
      assert(strlen(rayTraceData.HEALPixLensPlaneMapName) > 0);
      assert(rayTraceData.HEALPixLensPlaneMapOrder >= 0);
      rayTraceData.UseHEALPixLensPlaneMaps = 1;
    }
  
  /* set gal image search rads */
  rayTraceData.galImageSearchRad = 10.0*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder));
  if(rayTraceData.galImageSearchRad < GRIDSEARCH_RADIUS_ARCMIN/60.0/180.0*M_PI)
    rayTraceData.galImageSearchRad = GRIDSEARCH_RADIUS_ARCMIN/60.0/180.0*M_PI; 
  rayTraceData.galImageSearchRayBufferRad = sqrt(4.0*M_PI/order2npix(rayTraceData.bundleOrder)) + RAYBUFF_RADIUS_ARCMIN/60.0/180.0*M_PI;
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
