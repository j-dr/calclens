#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>
#include <gsl/gsl_heapsort.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "raytrace.h"

/* outputs the bundle cells with their domain decomp information */
void write_bundlecells2ascii(char fname_base[MAX_FILENAME])
{
  /*typedef struct {
    long nest;
    unsigned int active;
    long Nparts;
    long firstPart;
    long Nrays;
    HEALPixRay *rays;
    long firstMapCell;
    double cpuTime;
  } HEALPixBundleCell;
  */
  
  FILE *fp;
  long i,ring,npix;
  char fname[MAX_FILENAME];
  long mygroup,currgroup,Ngroups;
  
  //first make dir for files
  if(ThisTask == 0)
    {
      sprintf(fname,"%s/%s",rayTraceData.OutputPath,fname_base);
      mkdir(fname,02755);
    }
  
  ////////////////////////////
  MPI_Barrier(MPI_COMM_WORLD);
  ////////////////////////////
  
  //now output the files to 6
  npix = order2npix(rayTraceData.bundleOrder);
  sprintf(fname,"%s/%s/%s.%04d",rayTraceData.OutputPath,fname_base,fname_base,ThisTask);  
  mygroup = ThisTask/rayTraceData.NumFilesIOInParallel;
  Ngroups = NTasks/rayTraceData.NumFilesIOInParallel;
  if(Ngroups*rayTraceData.NumFilesIOInParallel != NTasks)
    ++Ngroups;
  
  for(currgroup=0;currgroup<Ngroups;++currgroup)
    {
      if(mygroup == currgroup)
        {
	  fp = fopen(fname,"w");
	  fprintf(fp,"# nest nside dflags nparts nrays cpuTime\n");
	  for(i=0;i<npix;++i)
	    {
	      fprintf(fp,"%ld %ld %u %ld %ld %le\n",bundleCells[i].nest,order2nside(rayTraceData.bundleOrder),bundleCells[i].active,
		      bundleCells[i].Nparts,bundleCells[i].Nrays,bundleCells[i].cpuTime);
	    }
	  fclose(fp);
	}
      
      //////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      //////////////////////////////
    }
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

/* makes map cells and creates and index through the bundle cells for searching */
void mark_bundlecells(double mapbuffrad, int searchTag, int markTag)
{
  long i;
  long k,*listpix,Nlistpix,NlistpixMax;
  double theta,phi;
  
  listpix = NULL;
  NlistpixMax = 0;
  
  //make the map buffer cells with their bit flags
  if(mapbuffrad >= M_PI)
    {
      for(i=0;i<NbundleCells;++i)
	SETBITFLAG(bundleCells[i].active,markTag);
    }
  else
    {
      for(i=0;i<NbundleCells;++i)
	CLEARBITFLAG(bundleCells[i].active,markTag);
      for(i=0;i<NbundleCells;++i)
	{
	  if(ISSETBITFLAG(bundleCells[i].active,searchTag))
	    {
	      nest2ang(i,&theta,&phi,rayTraceData.bundleOrder);
	      
	      Nlistpix = query_disc_inclusive_nest_fast(theta,phi,mapbuffrad,&listpix,&NlistpixMax,rayTraceData.bundleOrder);
	      
	      for(k=0;k<Nlistpix;++k)
		if(!(ISSETBITFLAG(bundleCells[listpix[k]].active,searchTag)))
		  SETBITFLAG(bundleCells[listpix[k]].active,markTag);
	    }
	}
      
      if(NlistpixMax > 0)
	free(listpix);
    }
}

/* makes map cells and creates and index through the bundle cells for searching */
void alloc_mapcells(int searchTag, int markTag)
{
  long i,j,NumMapCellsPerBundleCell,bundleMapShift,mapNest;
  
  bundleMapShift = 2*(rayTraceData.poissonOrder - rayTraceData.bundleOrder);
  NumMapCellsPerBundleCell = 1;
  NumMapCellsPerBundleCell = (NumMapCellsPerBundleCell << bundleMapShift);
  
  for(i=0;i<NbundleCells;++i)
    {
      CLEARBITFLAG(bundleCells[i].active,MAPBUFF_BUNDLECELL);

      if(ISSETBITFLAG(bundleCells[i].active,markTag))
	SETBITFLAG(bundleCells[i].active,MAPBUFF_BUNDLECELL);
    }
  
  /* make mapCells
     each bundle cell sets a pointer to its first mapCell in the mapCells array
     this can be used for searching for a given mapCell by doing the following
     1) find out which nest index you need at rayOrder
     2) bit shift this index to the bundleOrder
     3) compute offset by bit shifting the bundleCell index back to rayOrder
     4) go to bundleCell with the bit shifted Index and use offset to find mapCell
  */
  NmapCells = 0;
  for(i=0;i<NbundleCells;++i)
    if(ISSETBITFLAG(bundleCells[i].active,searchTag) || ISSETBITFLAG(bundleCells[i].active,MAPBUFF_BUNDLECELL))
      {
        bundleCells[i].firstMapCell = NmapCells;
        NmapCells += NumMapCellsPerBundleCell;
      }
  mapCells = (HEALPixMapCell*)malloc(sizeof(HEALPixMapCell)*NmapCells);
  assert(mapCells != NULL);
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,searchTag) || ISSETBITFLAG(bundleCells[i].active,MAPBUFF_BUNDLECELL))
        {
          mapNest = bundleCells[i].nest << bundleMapShift;
          for(j=0;j<NumMapCellsPerBundleCell;++j)
            {
              mapCells[bundleCells[i].firstMapCell+j].index = mapNest + j;
              mapCells[bundleCells[i].firstMapCell+j].val = 0.0;
            }
        }
    }
}

void free_mapcells(void)
{
  long i;
  
  for(i=0;i<NbundleCells;++i)
    CLEARBITFLAG(bundleCells[i].active,MAPBUFF_BUNDLECELL);
  
  NmapCells = 0;
  
  if(mapCells != NULL)
    {
      free(mapCells);
      mapCells = NULL;
    }
  
  if(mapCellsGradTheta != NULL)
    {
      free(mapCellsGradTheta);
      mapCellsGradTheta = NULL;
    }
  
  if(mapCellsGradPhi != NULL)
    {
      free(mapCellsGradPhi);
      mapCellsGradPhi = NULL;
    }
  
  if(mapCellsGradThetaTheta != NULL)
    {
      free(mapCellsGradThetaTheta);
      mapCellsGradThetaTheta = NULL;
    }

  if(mapCellsGradThetaPhi != NULL)
    {
      free(mapCellsGradThetaPhi);
      mapCellsGradThetaPhi = NULL;
    }
  
  if(mapCellsGradPhiPhi != NULL)
    {
      free(mapCellsGradPhiPhi);
      mapCellsGradPhiPhi = NULL;
    }
}

//make a healpix map of the lens plane for debugging
void make_lensplane_map(long planeNum)
{
  long order = 10; //must be <= 13
  assert(order <= 13);
  long Npix = order2npix(order);
  float *map,*totmap;
  
  long FileHEALPixOrder,FileNPix,*NumLCPartsInPix;
  Part *Parts;
  long NumParts;
  char file_name[MAX_FILENAME];
  
  hid_t file_id;
  herr_t status;
  
  long minInd,maxInd,NumIndsPerTask;
  long PeanoIndsToRead,NumPeanoIndsToRead;
  long i,j;
  
  double vec[3];
  long ring;
  FILE *fp;
  
  sprintf(file_name,"%s/%s%04ld.h5",rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,planeNum);
  file_id = H5Fopen(file_name,H5F_ACC_RDONLY,H5P_DEFAULT);
  if(file_id < 0)
    {
      fprintf(stderr,"%d: lens plane %ld could not be opened!\n",ThisTask,planeNum);
      assert(0);
    }

  /* read info about file */
  status = H5LTread_dataset(file_id,"/HEALPixOrder",H5T_NATIVE_LONG,&FileHEALPixOrder);
  assert(status >= 0);
  FileNPix = order2npix(FileHEALPixOrder);
  NumLCPartsInPix = (long*)malloc(sizeof(long)*FileNPix);
  status = H5LTread_dataset(file_id,"/NumLCPartsInPix",H5T_NATIVE_LONG,NumLCPartsInPix);
  assert(status >= 0);
  
  map = (float*)malloc(sizeof(float)*Npix);
  assert(map != NULL);
  totmap = (float*)malloc(sizeof(float)*Npix);
  assert(totmap != NULL);
  
  for(i=0;i<Npix;++i)
    map[i] = 0.0;
  
  NumIndsPerTask = FileNPix/NTasks;
  minInd = ThisTask*NumIndsPerTask;
  maxInd = minInd + NumIndsPerTask - 1;
  if(ThisTask == NTasks-1)
    maxInd = FileNPix-1;
  
  for(i=minInd;i<=maxInd;++i)
    {
      PeanoIndsToRead = i;
      NumPeanoIndsToRead = 1;
      
      readRayTracingPlaneAtPeanoInds(&file_id,FileHEALPixOrder,&PeanoIndsToRead,NumPeanoIndsToRead,&Parts,&NumParts);
      //fprintf(stderr,"%d: NumParts = %ld, NumLCPartsInPix = %ld\n",ThisTask,NumParts,NumLCPartsInPix[i]);
      
      assert(NumParts == NumLCPartsInPix[i]);
      
      for(j=0;j<NumParts;++j)
        {
          vec[0] = (double) (Parts[j].pos[0]);
          vec[1] = (double) (Parts[j].pos[1]);
          vec[2] = (double) (Parts[j].pos[2]);
                  
          ring = vec2ring(vec,order);
          
          map[ring] += 1.0;
        }
      
      free(Parts);
    }
  
  status = H5Fclose(file_id);
  assert(status >= 0);
  
  free(NumLCPartsInPix);
  
  MPI_Allreduce(map,totmap,(int) Npix,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
  
  if(ThisTask == 0)
    {
      sprintf(file_name,"%s/%s%04ld.healpixmap.dat",rayTraceData.LensPlanePath,rayTraceData.LensPlaneName,planeNum);
      fp = fopen(file_name,"w");
      fwrite(totmap,(size_t) Npix,sizeof(float),fp);
      fclose(fp);
    }
  
  free(map);
  free(totmap);
}

/* returns 1 if (ra,dec) is within radius of a boundary */
int test_vaccell_boundary(double ra, double dec, double radius)
{
  int checkVacCells;
  double decfac = cos(dec/180.0*M_PI);
  
  double mindiff,maxdiff;
  mindiff = (ra-rayTraceData.minRa)/180.0*M_PI;
  while(mindiff > M_PI)
    mindiff = mindiff - 2.0*M_PI;
  while(mindiff < -M_PI)
    mindiff = mindiff + 2.0*M_PI;
  
  maxdiff = (ra-rayTraceData.maxRa)/180.0*M_PI;
  while(maxdiff > M_PI)
    maxdiff = maxdiff - 2.0*M_PI;
  while(maxdiff < -M_PI)
    maxdiff = maxdiff + 2.0*M_PI;
  
  if(fabs(mindiff*decfac) >= radius &&
     fabs(maxdiff*decfac) >= radius &&
     fabs(dec-rayTraceData.minDec)/180.0*M_PI >= radius &&
     fabs(rayTraceData.maxDec-dec)/180.0*M_PI >= radius)
    checkVacCells = 0;
  else
    checkVacCells = 1;
  
  return checkVacCells;
}


/* returns 1 if cell is outside of ra,dec range defined by config file, else returns 0
 */
int test_vaccell(double ra, double dec)
{
  int vaccell;
  
  /* If minra > maxra, then we are wrapping around the circle in the ra direction by definition.*/
  if(rayTraceData.minRa > rayTraceData.maxRa)
    {
      if((ra >= rayTraceData.minRa || ra <= rayTraceData.maxRa) && dec >= rayTraceData.minDec && dec <= rayTraceData.maxDec)
        vaccell = 0;
      else
        vaccell = 1;
    }
  else
    {
      if(ra >= rayTraceData.minRa && ra <= rayTraceData.maxRa && dec >= rayTraceData.minDec && dec <= rayTraceData.maxDec)
        vaccell = 0;
      else
        vaccell = 1;
    }
  
  return vaccell;
}

void alloc_rays(void)
{
  long i,shift,NraysPerBundleCell;
  
  shift = rayTraceData.rayOrder - rayTraceData.bundleOrder;
  shift = 2*shift;
  
  NraysPerBundleCell = 1;
  NraysPerBundleCell = (NraysPerBundleCell << shift);
  
  long NumBuff = 25.0*1024.0*1024.0/sizeof(HEALPixRay)/NraysPerBundleCell;
  if(NumBuff < 10)
    NumBuff = 10;
  MaxNumAllRaysGlobal = ((long) ((1.0 + rayTraceData.maxRayMemImbalance)*NrestrictedPeanoInd/NTasks))*NraysPerBundleCell 
    + NumBuff*NraysPerBundleCell;
  AllRaysGlobal = (HEALPixRay*)malloc(sizeof(HEALPixRay)*MaxNumAllRaysGlobal);
  assert(AllRaysGlobal != NULL);
  NumAllRaysGlobal = 0;
  
  for(i=0;i<NbundleCells;++i) 
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	{
	  if(NumAllRaysGlobal >= MaxNumAllRaysGlobal)
	    {
	      fprintf(stderr,"%d: out of memory for rays! MaxNumAllRaysGlobal = %ld, NumAllRaysGlobal = %ld, max # of bundle cells = %ld, mem imbal = %lf, tot # of bundle cells = %ld\n",
		      ThisTask,MaxNumAllRaysGlobal,NumAllRaysGlobal,MaxNumAllRaysGlobal/NraysPerBundleCell,rayTraceData.maxRayMemImbalance,NrestrictedPeanoInd);
	      MPI_Abort(MPI_COMM_WORLD,111);
	    }

	  bundleCells[i].rays = AllRaysGlobal + NumAllRaysGlobal;
	  bundleCells[i].Nrays = NraysPerBundleCell;
	  NumAllRaysGlobal += NraysPerBundleCell;
	}
    }
}

void init_rays(void)
{
  long i,shift,j,NraysPerBundleCell,rayNest;
  double binL_2 = rayTraceData.maxComvDistance/rayTraceData.NumLensPlanes/2.0;
  
  shift = rayTraceData.rayOrder - rayTraceData.bundleOrder;
  shift = 2*shift;
  
  NraysPerBundleCell = 1;
  NraysPerBundleCell = (NraysPerBundleCell << shift);
  
  for(i=0;i<NbundleCells;++i) 
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	{
	  rayNest = (bundleCells[i].nest << shift);
	  for(j=0;j<NraysPerBundleCell;++j)
	    {
	      bundleCells[i].rays[j].nest = rayNest + j;
	      
	      nest2vec(bundleCells[i].rays[j].nest,bundleCells[i].rays[j].beta,rayTraceData.rayOrder);
	      
	      bundleCells[i].rays[j].n[0] = bundleCells[i].rays[j].beta[0]*binL_2;
	      bundleCells[i].rays[j].n[1] = bundleCells[i].rays[j].beta[1]*binL_2;
	      bundleCells[i].rays[j].n[2] = bundleCells[i].rays[j].beta[2]*binL_2;
	      
	      bundleCells[i].rays[j].A[0] = 1.0; //0+2*0
	      bundleCells[i].rays[j].A[1] = 0.0; //1+2*0
	      bundleCells[i].rays[j].A[2] = 0.0; //0+2*1
	      bundleCells[i].rays[j].A[3] = 1.0; //1+2*1
	      bundleCells[i].rays[j].Aprev[0] = 1.0; //0+2*0
	      bundleCells[i].rays[j].Aprev[1] = 0.0; //1+2*0
	      bundleCells[i].rays[j].Aprev[2] = 0.0; //0+2*1
	      bundleCells[i].rays[j].Aprev[3] = 1.0; //1+2*1

	      bundleCells[i].rays[j].phi = 0.0;
	      
	      bundleCells[i].rays[j].alpha[0] = 0.0;
	      bundleCells[i].rays[j].alpha[1] = 0.0;
	      	      
	      bundleCells[i].rays[j].U[0] = 0.0;
	      bundleCells[i].rays[j].U[1] = 0.0;
	      bundleCells[i].rays[j].U[2] = 0.0;
	      bundleCells[i].rays[j].U[3] = 0.0;
	    }
	}
    }
}

void destroy_rays(void)
{
  free(AllRaysGlobal);
  AllRaysGlobal = NULL;
  NumAllRaysGlobal = 0;
}

void destroy_gals(void)
{
  if(NumSourceGalsGlobal > 0)
    free(SourceGalsGlobal);
  NumSourceGalsGlobal = 0;
  SourceGalsGlobal = NULL;
  if(NumImageGalsGlobal > 0)
    free(ImageGalsGlobal);
  NumImageGalsGlobal = 0;
  ImageGalsGlobal = NULL;
}

void destroy_parts(void)
{
  long i;
  if(lensPlaneParts != NULL || NlensPlaneParts > 0)
    {
      free(lensPlaneParts);
      lensPlaneParts = NULL;
      NlensPlaneParts = 0;
    }
  for(i=0;i<NbundleCells;++i)
    {
      bundleCells[i].Nparts = 0;
      bundleCells[i].firstPart = -1;
    }
}

//does all domain decomp and indexing functions 
//do not change unless you know what you are doing (and even then it might be a good idea not to change this)
void init_bundlecells(void)
{
  long i,j,k,Nlistpix,*listpix;
  int *glbactivemap,*activemap;
  long minind,maxind,dind;
  long *bundlePeanoInds;
  double theta,phi,ra,dec;
  size_t *index;
    
  /* init bundle cells*/
  NbundleCells = order2npix(rayTraceData.bundleOrder);
  bundleCells = (HEALPixBundleCell*)malloc(sizeof(HEALPixBundleCell)*NbundleCells);
  assert(bundleCells != NULL);
  bundleCellsNest2RestrictedPeanoInd = (long*)malloc(sizeof(long)*NbundleCells);
  assert(bundleCellsNest2RestrictedPeanoInd != NULL);
  for(i=0;i<NbundleCells;++i)
    {
      bundleCells[i].nest = i;
      bundleCells[i].active = 0;
      bundleCells[i].Nparts = 0;
      bundleCells[i].firstPart = -1;
      bundleCells[i].Nrays = 0;
      bundleCells[i].rays = NULL;
      bundleCells[i].firstMapCell = -1;
      nest2ang(i,&theta,&phi,rayTraceData.bundleOrder);
    }
  
  /* get active region and build restricted peano index*/
  glbactivemap = (int*)malloc(sizeof(int)*NbundleCells);
  assert(glbactivemap != NULL);
  activemap = (int*)malloc(sizeof(int)*NbundleCells);
  assert(activemap != NULL);
  for(i=0;i<NbundleCells;++i)
    activemap[i] = 0;
    
  /* split rays up between nodes for checking what is in range of ray tracing */
  dind = NbundleCells/NTasks;
  minind = ThisTask*dind;
  maxind = minind + dind;
  if(ThisTask == NTasks-1)
    maxind = NbundleCells;
  for(i=minind;i<maxind;++i) /* checks all rays assigned to this task */
    {
      nest2ang(i,&theta,&phi,rayTraceData.bundleOrder);
      ang2radec(theta,phi,&ra,&dec);
      if(!(test_vaccell(ra,dec)))
	activemap[i] = 1;
    }
  
  /* build a buffer of rays around patch to catch all gals during grid search */
  for(i=minind;i<maxind;++i)
    {
      if(activemap[i] == 1)
	{
	  nest2ang(i,&theta,&phi,rayTraceData.bundleOrder);
	  listpix = NULL;
	  Nlistpix = 0;
	  query_disc_inclusive_nest(theta,phi,rayTraceData.galImageSearchRayBufferRad,&listpix,&Nlistpix,rayTraceData.bundleOrder);
	  
	  for(k=0;k<Nlistpix;++k)
	    {
	      if(activemap[listpix[k]] == 0)
		activemap[listpix[k]] = 2;
	    }
	  
	  if(Nlistpix > 0)
	    free(listpix);
	}
    }
  
  /* do a global all reduce w/ a "logical or" operation to get a global map of all active bundle cells */
  MPI_Allreduce(activemap,glbactivemap,(int) NbundleCells,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
  j = 0;
  for(i=0;i<NbundleCells;++i)
    {
      if(glbactivemap[i])
	{
	  ++j;
	  SETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL);
	}
      else
	bundleCells[i].active = 0;
    }
  free(activemap);
  free(glbactivemap);

  
  if(j < NTasks)
    {
      if(ThisTask == 0)
	{
	  fprintf(stderr,"too few bundle cells (%ld cells, order %ld) for %d tasks!\n",j,rayTraceData.bundleOrder,NTasks);
	  fflush(stderr);
	}
      
      MPI_Abort(MPI_COMM_WORLD,999);
    }
  
  /* build restricted peano index hash vector which covers cells with particles and rays */
  bundlePeanoInds = (long*)malloc(sizeof(long)*NbundleCells);
  assert(bundlePeanoInds != NULL);
  index = (size_t*)malloc(sizeof(size_t)*NbundleCells);
  assert(index != NULL);
  for(i=0;i<NbundleCells;++i)
    bundlePeanoInds[i] = nest2peano(bundleCells[i].nest,rayTraceData.bundleOrder);
  gsl_sort_long_index(index,bundlePeanoInds,(size_t) 1,(size_t) NbundleCells);  
  free(bundlePeanoInds);
  j = 0;
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[index[i]].active,PRIMARY_BUNDLECELL))
	{
	  bundleCellsNest2RestrictedPeanoInd[index[i]] = j;
	  ++j;
	}
      else
	bundleCellsNest2RestrictedPeanoInd[index[i]] = -1;
    }
  free(index);
  NrestrictedPeanoInd = j;
  bundleCellsRestrictedPeanoInd2Nest = (long*)malloc(sizeof(long)*NbundleCells);
  assert(bundleCellsRestrictedPeanoInd2Nest != NULL);
  for(i=0;i<NbundleCells;++i)
    bundleCellsRestrictedPeanoInd2Nest[i] = -1;
  for(i=0;i<NbundleCells;++i)
    {
      if(bundleCellsNest2RestrictedPeanoInd[i] != -1)
	bundleCellsRestrictedPeanoInd2Nest[bundleCellsNest2RestrictedPeanoInd[i]] = i;
    }
  
  //get domain decomp
  firstRestrictedPeanoIndTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(firstRestrictedPeanoIndTasks != NULL);
  lastRestrictedPeanoIndTasks = (long*)malloc(sizeof(long)*NTasks);
  assert(lastRestrictedPeanoIndTasks != NULL);
  
  for(i=0;i<NbundleCells;++i)
    bundleCells[i].cpuTime = 0.0;
  for(i=0;i<NbundleCells;++i)
    if(bundleCellsNest2RestrictedPeanoInd[i] != -1)
      bundleCells[i].cpuTime = 1.0/((double) NrestrictedPeanoInd);
  
  getDomainDecompPerCPU(1);
  for(i=0;i<NbundleCells;++i)
    bundleCells[i].cpuTime = 0.0;
  
  //creates primary domain decomp for full sky particle distribution cells
#ifdef USE_FULLSKY_PARTDIST 
  long NumFullSkyCellsPerTask,NumExtraFullSkyCells;
  long firstFullSkyCell,lastFullSkyCell;
  long nest;
  NumFullSkyCellsPerTask = NbundleCells/NTasks;
  NumExtraFullSkyCells = NbundleCells - NTasks*NumFullSkyCellsPerTask;
  
  if(ThisTask < NumExtraFullSkyCells)
    {
      firstFullSkyCell = ThisTask*NumFullSkyCellsPerTask + ThisTask;
      lastFullSkyCell = firstFullSkyCell + NumFullSkyCellsPerTask + 1 - 1;
    }
  else
    {
      firstFullSkyCell = ThisTask*NumFullSkyCellsPerTask + NumExtraFullSkyCells;
      lastFullSkyCell = firstFullSkyCell + NumFullSkyCellsPerTask - 1;
    }
  
#ifdef DEBUG
#if DEBUG_LEVEL > 0 
  fprintf(stderr,"%d: first,last,num = %ld|%ld|%ld\n",ThisTask,firstFullSkyCell,lastFullSkyCell,lastFullSkyCell-firstFullSkyCell+1);
#endif
#endif
  
  for(i=firstFullSkyCell;i<=lastFullSkyCell;++i)
    {
      nest = peano2nest(i,rayTraceData.bundleOrder);
      SETBITFLAG(bundleCells[nest].active,FULLSKY_PARTDIST_PRIMARY_BUNDLECELL);
    }
#endif /* USE_FULLSKY_PARTDIST */
  
  if(ThisTask == 0)
    {
      fprintf(stderr,"domain decomp has %ld active bundle cells with order %ld.\n",j,rayTraceData.bundleOrder);
      fflush(stderr);
    }
}

void destroy_bundlecells(void)
{
  NbundleCells = 0;
  
  free(bundleCells);
  bundleCells = NULL;
  
  free(bundleCellsNest2RestrictedPeanoInd);
  bundleCellsNest2RestrictedPeanoInd = NULL;
  
  free(bundleCellsRestrictedPeanoInd2Nest);
  bundleCellsRestrictedPeanoInd2Nest = NULL;
  
  free(firstRestrictedPeanoIndTasks);
  firstRestrictedPeanoIndTasks = NULL;
  
  free(lastRestrictedPeanoIndTasks);
  lastRestrictedPeanoIndTasks = NULL;
  
  NrestrictedPeanoInd = 0;
}
