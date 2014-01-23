#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>

#include "raytrace.h"
#include "lgadgetio.h"

//LGadget-2 I/O header for Lightcones
struct io_header_1 {
  unsigned int npart[6];      /*!< npart[1] gives the number of particles in the present file, other particle types are ignored */
  double mass[6];          /*!< mass[1] gives the particle mass */
  double time;             /*!< time (=cosmological scale factor) of snapshot */
  double redshift;         /*!< redshift of snapshot */
  int flag_sfr;       /*!< flags whether star formation is used (not available in L-Gadget2) */
  int flag_feedback;  /*!< flags whether feedback from star formation is included */
  unsigned int npartTotal[6]; /*!< npart[1] gives the total number of particles in the run. If this number exceeds 2^32, the npartTotal[2] stores                                                 
                                the result of a division of the particle number by 2^32, while npartTotal[1] holds the remainder. */
  int flag_cooling;   /*!< flags whether radiative cooling is included */
  int num_files;      /*!< determines the number of files that are used for a snapshot */
  double BoxSize;          /*!< Simulation box size (in code units) */
  double Omega0;           /*!< matter density */
  double OmegaLambda;      /*!< vacuum energy density */
  double HubbleParam;      /*!< little 'h' */
  int flag_stellarage;     /*!< flags whether the age of newly formed stars is recorded and saved */
  int flag_metals;         /*!< flags whether metal enrichment is included */
  int hashtabsize;         /*!< gives the size of the hashtable belonging to this snapshot file */
  //char fill[84];              /*!< fills to 256 Bytes */                                                                                                                         
  unsigned int npartTotalHighWord[6];  /*!< High word of the total number of par                                                                                                          
                                         ticles of each type */
  char fill[60];
};

long get_numparts_LGADGET(char fname[])
{
  FILE *fp;
  struct io_header_1 header1;
  long np;
  int dummy;
#define SKIP fread(&dummy,sizeof(dummy),(size_t) 1,fp);

  fp = fopen_retry(fname,"r");
  if(fp == NULL)
    {
      fprintf(stderr,"%d: could not open file '%s'!\n",ThisTask,fname);
      assert(fp != NULL);
    }

  SKIP;
  fread(&header1,sizeof(header1),(size_t) 1,fp);
  SKIP;
  
  fclose(fp);
  
  np = (long) (header1.npartTotal[1]) + (((long) (header1.npartTotal[2])) << 32);
  
  return np;
}

int get_numfiles_LGADGET(char fname[])
{
  FILE *fp;
  struct io_header_1 header1;
  int dummy;
#define SKIP fread(&dummy,sizeof(dummy),(size_t) 1,fp);

  fp = fopen_retry(fname,"r");
  if(fp == NULL)
    {
      fprintf(stderr,"%d: could not open file '%s'!\n",ThisTask,fname);
      assert(fp != NULL);
    }

  SKIP;
  fread(&header1,sizeof(header1),(size_t) 1,fp);
  SKIP;
  
  fclose(fp);
  
  return header1.num_files;
}

float get_omegam_LGADGET(char fname[])
{
  FILE *fp;
  struct io_header_1 header1;
  int dummy;
#define SKIP fread(&dummy,sizeof(dummy),(size_t) 1,fp);

  fp = fopen_retry(fname,"r");
  if(fp == NULL)
    {
      fprintf(stderr,"%d: could not open file '%s'!\n",ThisTask,fname);
      assert(fp != NULL);
    }

  SKIP;
  fread(&header1,sizeof(header1),(size_t) 1,fp);
  SKIP;
  
  fclose(fp);
  
  return header1.Omega0;
}

float get_scale_factor_LGADGET(char fname[])
{
  FILE *fp;
  struct io_header_1 header1;
  int dummy;
#define SKIP fread(&dummy,sizeof(dummy),(size_t) 1,fp);

  fp = fopen_retry(fname,"r");
  if(fp == NULL)
    {
      fprintf(stderr,"%d: could not open file '%s'!\n",ThisTask,fname);
      assert(fp != NULL);
    }

  SKIP;
  fread(&header1,sizeof(header1),(size_t) 1,fp);
  SKIP;
  
  fclose(fp);
  
  return header1.time;
}

float get_period_length_LGADGET(char fname[])
{
  FILE *fp;
  struct io_header_1 header1;
  int dummy;
#define SKIP fread(&dummy,sizeof(dummy),(size_t) 1,fp);

  fp = fopen_retry(fname,"r");
  if(fp == NULL)
    {
      fprintf(stderr,"%d: could not open file '%s'!\n",ThisTask,fname);
      assert(fp != NULL);
    }

  SKIP;
  fread(&header1,sizeof(header1),(size_t) 1,fp);
  SKIP;
  
  fclose(fp);
  
  return header1.BoxSize;
}

void read_LGADGET(char fname[], float **px, float **py, float **pz, long **id, int *Np)
{
  FILE *fp;
  struct io_header_1 header1;
  int dummy,k;
  float *partChunk;
#define SKIP fread(&dummy,sizeof(dummy),(size_t) 1,fp);

  fp = fopen_retry(fname,"r");
  if(fp == NULL)
    {
      fprintf(stderr,"%d: could not open file '%s'!\n",ThisTask,fname);
      assert(fp != NULL);
    }

  SKIP;
  fread(&header1,sizeof(header1),(size_t) 1,fp);
  SKIP;

  *Np = 0;
  for(k=0;k<5;k++)
    *Np += header1.npart[k];

  *px = (float*)malloc(sizeof(float)*(*Np));
  assert((*px) != NULL);
  *py = (float*)malloc(sizeof(float)*(*Np));
  assert((*py) != NULL);
  *pz = (float*)malloc(sizeof(float)*(*Np));
  assert((*pz) != NULL);
  if(id != NULL)
    {
      *id = (long*)malloc(sizeof(long)*(*Np));
      assert((*id) != NULL);
    }
  
  //read positions                                                                                                                                                                                           
  partChunk = (float*)malloc(sizeof(float)*(*Np)*3);
  assert(partChunk != NULL);
  SKIP;
  assert(dummy == (*Np)*3*sizeof(float));
  fread(partChunk,(size_t) (*Np),3*sizeof(float),fp);
  SKIP;
  assert(dummy == (*Np)*3*sizeof(float));
  for(k=0;k<(*Np);++k)
    {
      (*px)[k] = partChunk[k*3 + 0];
      (*py)[k] = partChunk[k*3 + 1];
      (*pz)[k] = partChunk[k*3 + 2];
      
      while((*px)[k] < 0.0)
	(*px)[k] += header1.BoxSize;
      while((*px)[k] > header1.BoxSize)
	(*px)[k] -= header1.BoxSize;
      
      while((*py)[k] < 0.0)
	(*py)[k] += header1.BoxSize;
      while((*py)[k] > header1.BoxSize)
	(*py)[k] -= header1.BoxSize;
      
      while((*pz)[k] < 0.0)
	(*pz)[k] += header1.BoxSize;
      while((*pz)[k] > header1.BoxSize)
	(*pz)[k] -= header1.BoxSize;
    }
  free(partChunk);
  
  if(id != NULL)
    {
      SKIP;
      assert(dummy == (*Np)*3*sizeof(float));
      fseek(fp,3*sizeof(float)*(*Np),SEEK_CUR);
      SKIP;
      assert(dummy == (*Np)*3*sizeof(float));
      
      SKIP;
      assert(dummy == (*Np)*sizeof(long));
      fread(*id,(size_t) (*Np),sizeof(long),fp);
      SKIP;
      assert(dummy == (*Np)*sizeof(long));
    }
  
#undef SKIP
  
  fclose(fp);
}
