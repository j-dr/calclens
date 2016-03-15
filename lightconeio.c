#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include "raytrace.h"

long getNumLCPartsFile(FILE *infp)
{
  if(strcmp(rayTraceData.LightConeFileType,"ARTLC") == 0)
    return getNumLCPartsFile_ARTLC(infp);
  else if(strcmp(rayTraceData.LightConeFileType,"GADGET2") == 0)
    return getNumLCPartsFile_GADGET2(infp);
  else if(strcmp(rayTraceData.LightConeFileType,"LGADGET") == 0)
    return getNumLCPartsFile_LGADGET(infp);
  else
    {
      fprintf(stderr,"%d: rayTraceData.LightConeFileType '%s' does not match one of options!\n",ThisTask,rayTraceData.LightConeFileType);
      assert(0);
      return -1;
    }
}

LCParticle getLCPartFromFile(long i, long Np, FILE *infp, int freeBuff)
{
  LCParticle LCPart;
  
  if(strcmp(rayTraceData.LightConeFileType,"ARTLC") == 0)
    return getLCPartFromFile_ARTLC(i,Np,infp,freeBuff);
  else if(strcmp(rayTraceData.LightConeFileType,"GADGET2") == 0)
    return getLCPartFromFile_GADGET2(i,Np,infp,freeBuff);
  else if(strcmp(rayTraceData.LightConeFileType,"LGADGET") == 0)
    return getLCPartFromFile_LGADGET(i,Np,infp,freeBuff);
  else
    {
      LCPart.partid = -1;
      fprintf(stderr,"%d: rayTraceData.LightConeFileType '%s' does not match one of options!\n",ThisTask,rayTraceData.LightConeFileType);
      assert(0);
      return LCPart;
    }
}

long getNumLCPartsFile_LGADGET(FILE *infp)
{
  struct io_header_1
  {
    unsigned int npart[6];      /*!< npart[1] gives the number of particles in the present file, other particle types are ignored */
    double mass[6];             /*!< mass[1] gives the particle mass */
    double time;                /*!< time (=cosmological scale factor) of snapshot */
    double redshift;            /*!< redshift of snapshot */
    int flag_sfr;               /*!< flags whether star formation is used (not available in L-Gadget2) */
    int flag_feedback;          /*!< flags whether feedback from star formation is included */
    unsigned int npartTotal[6]; /*!< npart[1] gives the total number of particles in the run. If this number exceeds 2^32, the npartTotal[2] stores
				  the result of a division of the particle number by 2^32, while npartTotal[1] holds the remainder. */
    int flag_cooling;           /*!< flags whether radiative cooling is included */
    int num_files;              /*!< determines the number of files that are used for a snapshot */
    double BoxSize;             /*!< Simulation box size (in code units) */
    double Omega0;              /*!< matter density */
    double OmegaLambda;         /*!< vacuum energy density */
    double HubbleParam;         /*!< little 'h' */
    int flag_stellarage;        /*!< flags whether the age of newly formed stars is recorded and saved */
    int flag_metals;            /*!< flags whether metal enrichment is included */
    int hashtabsize;            /*!< gives the size of the hashtable belonging to this snapshot file */
    unsigned int npartTotalHighWord[6];  /*!< High word of the total number of particles of each type */
    char fill[60];
  } header1;
  
  int NumPart;
  int k,dummy;
  int files;

#define SKIP fread(&dummy,sizeof(dummy),(size_t) 1,infp);
  
  SKIP;
  fread(&header1,sizeof(header1),(size_t) 1,infp);
  SKIP;
  
  files = header1.num_files;
  for(k=0,NumPart=0;k<6;k++)
    NumPart += header1.npart[k];
    
  if(ThisTask == 0)
    {
      fprintf(stderr,"# of files = %d\n",files);
      fprintf(stderr,"NumPart = %d\n",NumPart);
      fprintf(stderr,"header1.npart[1] = %u\n",header1.npart[1]);
    }
  
#undef SKIP  
  return ((long) NumPart);
}

LCParticle getLCPartFromFile_LGADGET(long i, long Np, FILE *infp, int freeBuff)
{
  static float *LCPartReadVec = NULL;
  LCParticle LCPartRead;
  struct io_header_1
  {
    unsigned int npart[6];      /*!< npart[1] gives the number of particles in the present file, other particle types are ignored */
    double mass[6];             /*!< mass[1] gives the particle mass */
    double time;                /*!< time (=cosmological scale factor) of snapshot */
    double redshift;            /*!< redshift of snapshot */
    int flag_sfr;               /*!< flags whether star formation is used (not available in L-Gadget2) */
    int flag_feedback;          /*!< flags whether feedback from star formation is included */
    unsigned int npartTotal[6]; /*!< npart[1] gives the total number of particles in the run. If this number exceeds 2^32, the npartTotal[2] stores
				  the result of a division of the particle number by 2^32, while npartTotal[1] holds the remainder. */
    int flag_cooling;           /*!< flags whether radiative cooling is included */
    int num_files;              /*!< determines the number of files that are used for a snapshot */
    double BoxSize;             /*!< Simulation box size (in code units) */
    double Omega0;              /*!< matter density */
    double OmegaLambda;         /*!< vacuum energy density */
    double HubbleParam;         /*!< little 'h' */
    int flag_stellarage;        /*!< flags whether the age of newly formed stars is recorded and saved */
    int flag_metals;            /*!< flags whether metal enrichment is included */
    int hashtabsize;            /*!< gives the size of the hashtable belonging to this snapshot file */
    unsigned int npartTotalHighWord[6];  /*!< High word of the total number of particles of each type */
    char fill[60];
  } header1;

  static float mass[6];
  static long masslims[6];
  
  int dummy;
  long NumPart,k;
  
#define SKIP fread(&dummy,sizeof(dummy),(size_t) 1,infp);
  
  if(freeBuff)
    {
      LCPartRead.partid = -1;
      free(LCPartReadVec);
      LCPartReadVec = NULL;
      return LCPartRead;
    }
  else
    {
      //error check!
      assert(i >= 0 && i < Np);
      
      if(LCPartReadVec == NULL)
	{
	  //read header first
	  rewind(infp);
	  
	  SKIP;
	  fread(&header1,sizeof(header1),(size_t) 1,infp);
	  SKIP;
	  
	  for(k=0,NumPart=0;k<6;k++)
	    NumPart += header1.npart[k];
	  assert(NumPart == Np);
	  
	  for(k=0;k<6;++k)
	    mass[k] = (float) (header1.mass[k]*rayTraceData.MassConvFact);
	  
	  masslims[0] = header1.npart[0];
	  for(k=1;k<6;++k)
	    masslims[k] = masslims[k-1] + header1.npart[k];
	  
	  LCPartReadVec = (float*)malloc((size_t) (NumPart*6*sizeof(float)));
	  assert(LCPartReadVec != NULL);
	  
	  //read particles
	  SKIP;
	  fread(LCPartReadVec,(size_t) Np,3*sizeof(float),infp);
	  SKIP;
	  assert(dummy == (int) (Np*3*sizeof(float)));
	  
	  SKIP;
	  fread(LCPartReadVec+Np*3,(size_t) Np,3*sizeof(float),infp);
	  SKIP;
	  assert(dummy == (int) (Np*3*sizeof(float)));
	}
      
      LCPartRead.partid = -1;
      LCPartRead.px = (float) (LCPartReadVec[i*3 + 0]*rayTraceData.LengthConvFact);
      LCPartRead.py = (float) (LCPartReadVec[i*3 + 1]*rayTraceData.LengthConvFact);
      LCPartRead.pz = (float) (LCPartReadVec[i*3 + 2]*rayTraceData.LengthConvFact);
      
      //convert gadget units to comoving peculiar velocity - i.e. peculiar velocity = v*a where v = dx_{comv}/dt
      LCPartRead.vx = (float) (LCPartReadVec[Np*3 + i*3 + 0]*rayTraceData.VelocityConvFact);
      LCPartRead.vy = (float) (LCPartReadVec[Np*3 + i*3 + 1]*rayTraceData.VelocityConvFact);
      LCPartRead.vz = (float) (LCPartReadVec[Np*3 + i*3 + 2]*rayTraceData.VelocityConvFact);
      for(k=0;k<6;++k)
	if(i < masslims[k])
	  {
	    LCPartRead.mass = mass[k];
	    break;
	  }
      
      return LCPartRead;
    }

#undef SKIP
}

long getNumLCPartsFile_GADGET2(FILE *infp)
{
  struct io_header_1
  {
    unsigned int      npart[6];
    double   mass[6];
    double   time;
    double   redshift;
    int      flag_sfr;
    int      flag_feedback;
    unsigned int      npartTotal[6];
    int      flag_cooling;
    int      num_files;
    double   BoxSize;
    double   Omega0;
    double   OmegaLambda;
    double   HubbleParam;
    char     fill[256- 6*4- 6*8- 2*8- 2*4- 6*4- 2*4 - 4*8];  /* fills to 256 Bytes */
  } header1;
  
  int NumPart;
  int k,dummy;
  int files;

#define SKIP fread(&dummy,sizeof(dummy),(size_t) 1,infp);
  
  SKIP;
  fread(&header1,sizeof(header1),(size_t) 1,infp);
  SKIP;
  
  files = header1.num_files;
  for(k=0,NumPart=0;k<6;k++)
    NumPart += header1.npart[k];
    
  if(ThisTask == 0)
    {
      fprintf(stderr,"# of files = %d\n",files);
      fprintf(stderr,"NumPart = %d\n",NumPart);
      fprintf(stderr,"header1.npart[1] = %u\n",header1.npart[1]);
    }
  
#undef SKIP  
  return ((long) NumPart);
}

LCParticle getLCPartFromFile_GADGET2(long i, long Np, FILE *infp, int freeBuff)
{
  static float *LCPartReadVec = NULL;
  LCParticle LCPartRead;
  static struct io_header_1
  {
    unsigned int      npart[6];
    double   mass[6];
    double   time;
    double   redshift;
    int      flag_sfr;
    int      flag_feedback;
    unsigned int      npartTotal[6];
    int      flag_cooling;
    int      num_files;
    double   BoxSize;
    double   Omega0;
    double   OmegaLambda;
    double   HubbleParam;
    char     fill[256- 6*4- 6*8- 2*8- 2*4- 6*4- 2*4 - 4*8];  /* fills to 256 Bytes */
  } header1;
  static double sqrta;
  static float mass[6];
  static long masslims[6];
  
  int dummy;
  long NumPart,k;
  
#define SKIP fread(&dummy,sizeof(dummy),(size_t) 1,infp);
  
  if(freeBuff)
    {
      LCPartRead.partid = -1;
      free(LCPartReadVec);
      LCPartReadVec = NULL;
      return LCPartRead;
    }
  else
    {
      //error check!
      assert(i >= 0 && i < Np);
      
      if(LCPartReadVec == NULL)
	{
	  //read header first
	  rewind(infp);
	  
	  SKIP;
	  fread(&header1,sizeof(header1),(size_t) 1,infp);
	  SKIP;
	  
	  sqrta = sqrt(header1.time);
	  
	  for(k=0,NumPart=0;k<6;k++)
	    NumPart += header1.npart[k];
	  assert(NumPart == Np);
	  
	  for(k=0;k<6;++k)
	    mass[k] = (float) (header1.mass[k]*rayTraceData.MassConvFact);
	  
	  masslims[0] = header1.npart[0];
	  for(k=1;k<6;++k)
	    masslims[k] = masslims[k-1] + header1.npart[k];
	  
	  LCPartReadVec = (float*)malloc((size_t) (NumPart*6*sizeof(float)));
	  assert(LCPartReadVec != NULL);
	  
	  //read particles
	  SKIP;
	  fread(LCPartReadVec,(size_t) Np,3*sizeof(float),infp);
	  SKIP;
	  assert(dummy == (int) (Np*3*sizeof(float)));
	  
	  SKIP;
	  fread(LCPartReadVec+Np*3,(size_t) Np,3*sizeof(float),infp);
	  SKIP;
	  assert(dummy == (int) (Np*3*sizeof(float)));
	}
      
      LCPartRead.partid = -1;
      LCPartRead.px = (float) (LCPartReadVec[i*3 + 0]*rayTraceData.LengthConvFact);
      LCPartRead.py = (float) (LCPartReadVec[i*3 + 1]*rayTraceData.LengthConvFact);
      LCPartRead.pz = (float) (LCPartReadVec[i*3 + 2]*rayTraceData.LengthConvFact);
      
      //convert gadget units to comoving peculiar velocity - i.e. peculiar velocity = v*a where v = dx_{comv}/dt
      LCPartRead.vx = (float) (LCPartReadVec[Np*3 + i*3 + 0]/sqrta*rayTraceData.VelocityConvFact);
      LCPartRead.vy = (float) (LCPartReadVec[Np*3 + i*3 + 1]/sqrta*rayTraceData.VelocityConvFact);
      LCPartRead.vz = (float) (LCPartReadVec[Np*3 + i*3 + 2]/sqrta*rayTraceData.VelocityConvFact);
      for(k=0;k<6;++k)
	if(i < masslims[k])
	  {
	    LCPartRead.mass = mass[k];
	    break;
	  }
      
      return LCPartRead;
    }

#undef SKIP
}

long getNumLCPartsFile_ARTLC(FILE *infp)
{
  int Np;
  fread(&Np,sizeof(int),(size_t) 1,infp);
  return ((long) Np);
}

LCParticle getLCPartFromFile_ARTLC(long i, long Np, FILE *infp, int freeBuff)
{
  static char *LCPartCharReadVec = NULL;
  long sizeofLCPartOnDiskInChar;
  long NumReadBuff = 5000;
  long ind,Npr;
  sizeofLCPartOnDiskInChar = (sizeof(int)+9*sizeof(float))/(sizeof(char));
  LCParticle LCPartRead;
  
  if(freeBuff)
    {
      LCPartRead.partid = -1;
      free(LCPartCharReadVec);
      LCPartCharReadVec = NULL;
      return LCPartRead;
    }
  else
    {
      //error check!
      assert(i >= 0 && i < Np);
      
      if(LCPartCharReadVec == NULL)
	{
	  LCPartCharReadVec = (char*)malloc((size_t) (NumReadBuff*sizeofLCPartOnDiskInChar));
	  assert(LCPartCharReadVec != NULL);
	}
      
      //read particle
      if(i%NumReadBuff == 0)
	{
	  Npr = fread(LCPartCharReadVec,sizeof(char),(size_t) (NumReadBuff*sizeofLCPartOnDiskInChar),infp);
	  assert(Npr != 0);
	}
      
      ind = i%NumReadBuff;
      LCPartRead.partid   = (long) (*((int*)(LCPartCharReadVec+ind*sizeofLCPartOnDiskInChar)));
      LCPartRead.px     = *((float*)(LCPartCharReadVec+ind*sizeofLCPartOnDiskInChar+sizeof(int)+0*sizeof(float)));
      LCPartRead.py     = *((float*)(LCPartCharReadVec+ind*sizeofLCPartOnDiskInChar+sizeof(int)+1*sizeof(float)));
      LCPartRead.pz     = *((float*)(LCPartCharReadVec+ind*sizeofLCPartOnDiskInChar+sizeof(int)+2*sizeof(float)));
      LCPartRead.vx     = *((float*)(LCPartCharReadVec+ind*sizeofLCPartOnDiskInChar+sizeof(int)+3*sizeof(float)));
      LCPartRead.vy     = *((float*)(LCPartCharReadVec+ind*sizeofLCPartOnDiskInChar+sizeof(int)+4*sizeof(float)));
      LCPartRead.vz     = *((float*)(LCPartCharReadVec+ind*sizeofLCPartOnDiskInChar+sizeof(int)+5*sizeof(float)));
      LCPartRead.mass   = (float) (rayTraceData.partMass);
      
      return LCPartRead;
    }
}
