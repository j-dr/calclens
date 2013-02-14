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

#include "raytrace.h"

static int strcmp_caseinsens(const char *s1, const char *s2);

void read_config(char *filename)
{
  char usedfile[MAX_FILENAME];
  FILE *usedfp,*fp;
  char fline[1024];
  int i,len,loc;
  char *tag,*val;
  
  //set defaults
  rayTraceData.RayOutputName[0] = '\0';
  rayTraceData.GalsFileList[0] = '\0';
  rayTraceData.GalOutputName[0] = '\0';
  rayTraceData.HEALPixRingWeightPath[0] = '\0';
  rayTraceData.HEALPixWindowFunctionPath[0] = '\0';
#ifdef MAKE_LENSPLANES  
  int readLightConeOrigin[3] = {0,0,0};
#endif
  rayTraceData.MaxNumLensPlaneInMem = -1;
  rayTraceData.LightConePartChunkFactor = -1.0;
  rayTraceData.partMass = -1.0;
  rayTraceData.maxRayMemImbalance = 0.25;
  rayTraceData.galRadPointNFWTest = -1.0;
  rayTraceData.MGConvFact = -1.0;
  rayTraceData.ComvSmoothingScale = -1.0;
  rayTraceData.treeAllocFactor = 2.5;
  rayTraceData.BHCrit = -1.0;
  
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
      if(strcmp_caseinsens(tag,"WallTimeLimit") == 0)
	{
	  rayTraceData.WallTimeLimit = atof(val);
	  fprintf(usedfp,"%s %lf\n","WallTimeLimit",atof(val));
	} 
      else if(strcmp_caseinsens(tag,"WallTimeBetweenRestart") == 0)
	{
	  rayTraceData.WallTimeBetweenRestart = atof(val);
	  fprintf(usedfp,"%s %lf\n","WallTimeBetweenRestart",atof(val));
	} 
      else if(strcmp_caseinsens(tag,"OmegaM") == 0)
	{
	  rayTraceData.OmegaM = atof(val);
	  fprintf(usedfp,"%s %lf\n","OmegaM",atof(val));
	}
      else if(strcmp_caseinsens(tag,"maxComvDistance") == 0)
	{
	  rayTraceData.maxComvDistance = atof(val);
	  fprintf(usedfp,"%s %lf\n","maxComvDistance",atof(val));
	}
      else if(strcmp_caseinsens(tag,"NumLensPlanes") == 0)
	{
	  rayTraceData.NumLensPlanes = atol(val);
	  fprintf(usedfp,"%s %ld\n","NumLensPlanes",atol(val));
	} 
      else if(strcmp_caseinsens(tag,"LensPlanePath") == 0)
	{
	  strcpy(rayTraceData.LensPlanePath,val);
	  fprintf(usedfp,"%s %s\n","LensPlanePath",val);
	}
      else if(strcmp_caseinsens(tag,"LensPlaneName") == 0)
	{
	  strcpy(rayTraceData.LensPlaneName,val);
	  fprintf(usedfp,"%s %s\n","LensPlaneName",val);
	}
      else if(strcmp_caseinsens(tag,"OutputPath") == 0)
	{
	  strcpy(rayTraceData.OutputPath,val);
	  fprintf(usedfp,"%s %s\n","OutputPath",val);
	}
      else if(strcmp_caseinsens(tag,"RayOutputName") == 0)
	{
	  strcpy(rayTraceData.RayOutputName,val);
	  fprintf(usedfp,"%s %s\n","RayOutputName",val);
	}
      else if(strcmp_caseinsens(tag,"NumRayOutputFiles") == 0)
	{
	  rayTraceData.NumRayOutputFiles = atol(val);
	  fprintf(usedfp,"%s %ld\n","NumRayOutputFiles",atol(val));
	} 
      else if(strcmp_caseinsens(tag,"NumFilesIOInParallel") == 0)
	{
	  rayTraceData.NumFilesIOInParallel = atol(val);
	  fprintf(usedfp,"%s %ld\n","NumFilesIOInParallel",atol(val));
	} 
      else if(strcmp_caseinsens(tag,"bundleOrder") == 0)
	{
	  rayTraceData.bundleOrder = atol(val);
	  fprintf(usedfp,"%s %ld\n","bundleOrder",atol(val));
	} 
      else if(strcmp_caseinsens(tag,"rayOrder") == 0)
	{
	  rayTraceData.rayOrder = atol(val);
	  fprintf(usedfp,"%s %ld\n","rayOrder",atol(val));
	} 
      else if(strcmp_caseinsens(tag,"minRa") == 0)
	{
	  rayTraceData.minRa = atof(val);
	  fprintf(usedfp,"%s %lf\n","minRa",atof(val));
	}
      else if(strcmp_caseinsens(tag,"maxRa") == 0)
	{
	  rayTraceData.maxRa = atof(val);
	  fprintf(usedfp,"%s %lf\n","maxRa",atof(val));
	}
      else if(strcmp_caseinsens(tag,"minDec") == 0)
	{
	  rayTraceData.minDec = atof(val);
	  fprintf(usedfp,"%s %lf\n","minDec",atof(val));
	}
      else if(strcmp_caseinsens(tag,"maxDec") == 0)
	{
	  rayTraceData.maxDec = atof(val);
	  fprintf(usedfp,"%s %lf\n","maxDec",atof(val));
	}
      else if(strcmp_caseinsens(tag,"maxRayMemImbalance") == 0)
	{
	  rayTraceData.maxRayMemImbalance = atof(val);
	  fprintf(usedfp,"%s %lf\n","maxRayMemImbalance",atof(val));
	} 
      else if(strcmp_caseinsens(tag,"HEALPixRingWeightPath") == 0)
	{
	  strcpy(rayTraceData.HEALPixRingWeightPath,val);
	  fprintf(usedfp,"%s %s\n","HEALPixRingWeightPath",val);
	} 
      else if(strcmp_caseinsens(tag,"SHTOrder") == 0)
	{
	  rayTraceData.SHTOrder = atol(val);
	  fprintf(usedfp,"%s %ld\n","SHTOrder",atol(val));
	}
      else if(strcmp_caseinsens(tag,"ComvSmoothingScale") == 0)
	{
	  rayTraceData.ComvSmoothingScale = atof(val);
	  fprintf(usedfp,"%s %lf\n","ComvSmoothingScale",atof(val));
	}
      else if(strcmp_caseinsens(tag,"MGConvFact") == 0)
	{
	  rayTraceData.MGConvFact = atof(val);
	  fprintf(usedfp,"%s %lf\n","MGConvFact",atof(val));
	}
      else if(strcmp_caseinsens(tag,"BHCrit") == 0)
	{
	  rayTraceData.MGConvFact = atof(val);
	  fprintf(usedfp,"%s %lf\n","MGConvFact",atof(val));
	}
      else if(strcmp_caseinsens(tag,"GalsFileList") == 0)
	{
	  strcpy(rayTraceData.GalsFileList,val);
	  fprintf(usedfp,"%s %s\n","GalsFileList",val);
	}
      else if(strcmp_caseinsens(tag,"GalOutputName") == 0)
	{
	  strcpy(rayTraceData.GalOutputName,val);
	  fprintf(usedfp,"%s %s\n","GalOutputName",val);
	}
      else if(strcmp_caseinsens(tag,"NumGalOutputFiles") == 0)
	{
	  rayTraceData.NumGalOutputFiles = atol(val);
	  fprintf(usedfp,"%s %ld\n","NumGalOutputFiles",atol(val));
	} 
      else if(strcmp_caseinsens(tag,"LightConeFileList") == 0)
	{
	  strcpy(rayTraceData.LightConeFileList,val);
	  fprintf(usedfp,"%s %s\n","LightConeFileList",val);
	}
      else if(strcmp_caseinsens(tag,"LightConeFileType") == 0)
	{
	  strcpy(rayTraceData.LightConeFileType,val);
	  fprintf(usedfp,"%s %s\n","LightConeFileType",val);
	}
      else if(strcmp_caseinsens(tag,"LightConeOriginX") == 0)
	{
#ifdef MAKE_LENSPLANES  
	  readLightConeOrigin[0] = 1;
#endif
	  rayTraceData.LightConeOriginX = atof(val);
	  fprintf(usedfp,"%s %lf\n","LightConeOriginX",atof(val));
	} 
      else if(strcmp_caseinsens(tag,"LightConeOriginY") == 0)
	{
#ifdef MAKE_LENSPLANES  
	  readLightConeOrigin[1] = 1;
#endif
	  rayTraceData.LightConeOriginY = atof(val);
	  fprintf(usedfp,"%s %lf\n","LightConeOriginY",atof(val));
	} 
      else if(strcmp_caseinsens(tag,"LightConeOriginZ") == 0)
	{
#ifdef MAKE_LENSPLANES  
	  readLightConeOrigin[2] = 1;
#endif
	  rayTraceData.LightConeOriginZ = atof(val);
	  fprintf(usedfp,"%s %lf\n","LightConeOriginZ",atof(val));
	} 
      else if(strcmp_caseinsens(tag,"LensPlaneOrder") == 0)
	{
	  rayTraceData.LensPlaneOrder = atol(val);
	  fprintf(usedfp,"%s %ld\n","LensPlaneOrder",atol(val));
	} 
      else if(strcmp_caseinsens(tag,"NumDivLensPlane") == 0)
	{
	  rayTraceData.NumDivLensPlane = atol(val);
	  fprintf(usedfp,"%s %ld\n","NumDivLensPlane",atol(val));
	} 
      else if(strcmp_caseinsens(tag,"memBuffSizeInMB") == 0)
	{
	  rayTraceData.memBuffSizeInMB = atof(val);
	  fprintf(usedfp,"%s %lf\n","memBuffSizeInMB",atof(val));
	} 
      else if(strcmp_caseinsens(tag,"MaxNumLensPlaneInMem") == 0)
	{
	  rayTraceData.MaxNumLensPlaneInMem = atol(val);
	  fprintf(usedfp,"%s %ld\n","MaxNumLensPlaneInMem",atol(val));
	} 
      else if(strcmp_caseinsens(tag,"LightConePartChunkFactor") == 0)
	{
	  rayTraceData.LightConePartChunkFactor = atof(val);
	  fprintf(usedfp,"%s %lf\n","LightConePartChunkFactor",atof(val));
	} 
      else if(strcmp_caseinsens(tag,"partMass") == 0)
	{
	  rayTraceData.partMass = atof(val);
	  fprintf(usedfp,"%s %le\n","partMass",atof(val));
	}
      else if(strcmp_caseinsens(tag,"MassConvFact") == 0)
	{
	  rayTraceData.MassConvFact = atof(val);
	  fprintf(usedfp,"%s %le\n","MassConvFact",atof(val));
	}
      else if(strcmp_caseinsens(tag,"LengthConvFact") == 0)
	{
	  rayTraceData.LengthConvFact = atof(val);
	  fprintf(usedfp,"%s %lf\n","LengthConvFact",atof(val));
	}
      else if(strcmp_caseinsens(tag,"VelocityConvFact") == 0)
	{
	  rayTraceData.VelocityConvFact = atof(val);
	  fprintf(usedfp,"%s %lf\n","VelocityConvFact",atof(val));
	}
      else if(strcmp_caseinsens(tag,"raPointMass") == 0)
	{
	  rayTraceData.raPointMass = atof(val);
	  fprintf(usedfp,"%s %lf\n","raPointMass",atof(val));
	}
      else if(strcmp_caseinsens(tag,"decPointMass") == 0)
	{
	  rayTraceData.decPointMass = atof(val);
	  fprintf(usedfp,"%s %lf\n","decPointMass",atof(val));
	}
      else if(strcmp_caseinsens(tag,"radPointMass") == 0)
	{
	  rayTraceData.radPointMass = atof(val);
	  fprintf(usedfp,"%s %lf\n","radPointMass",atof(val));
	}
      else if(strcmp_caseinsens(tag,"galRadPointNFWTest") == 0)
	{
	  rayTraceData.galRadPointNFWTest = atof(val);
	  fprintf(usedfp,"%s %lf\n","galRadPointNFWTest",atof(val));
	}
      else
	{
	  fprintf(stderr,"Tag-value pair ('%s','%s') not found in config file '%s'!\n",tag,val,filename);
	  assert(0);
	}
    }
  
  //close files
  fclose(usedfp);
  fclose(fp);
  
  //error check
#ifndef MAKE_LENSPLANES
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
  
#ifdef TREEPM
  assert(rayTraceData.BHCrit > 0.0);
#endif
  
#else
  assert(readLightConeOrigin[0] == 1);
  assert(readLightConeOrigin[1] == 1);
  assert(readLightConeOrigin[2] == 1);
  assert(rayTraceData.MaxNumLensPlaneInMem > 0);
  assert(rayTraceData.LightConePartChunkFactor > 0);
#endif

  /* set gal image search rads */
  rayTraceData.galImageSearchRad = 10.0*sqrt(4.0*M_PI/order2npix(rayTraceData.rayOrder));
  if(rayTraceData.galImageSearchRad < GRIDSEARCH_RADIUS_ARCMIN/60.0/180.0*M_PI)
    rayTraceData.galImageSearchRad = GRIDSEARCH_RADIUS_ARCMIN/60.0/180.0*M_PI; 
  rayTraceData.galImageSearchRayBufferRad = sqrt(4.0*M_PI/order2npix(rayTraceData.bundleOrder)) + RAYBUFF_RADIUS_ARCMIN/60.0/180.0*M_PI;
    
  //error check for point mass test
#ifdef POINTMASSTEST
  if(strlen(rayTraceData.GalOutputName) > 0)
    assert(rayTraceData.galRadPointNFWTest > 0.0);
#endif
#ifdef NFWHALOTEST
  if(strlen(rayTraceData.GalOutputName) > 0)
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
