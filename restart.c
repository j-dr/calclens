#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>

#include "raytrace.h"

static void restart_io(int read, int mapnum);
static size_t frw_io(void *p, size_t size, size_t nitems, FILE *fp, int read);

/*
  this routine writes unformatted binary restart files
  for the ray tracing
*/
static void restart_io(int read, long mapnum)
{
  RayTraceData rtd_in;
  long i,mygroup,currgroup,Ngroups;
  FILE *fp;
  char name[MAX_FILENAME];
  char sys[MAX_FILENAME];
  int NTasks_in;
  int fspd,fspd_in;
  long NraysPerBundleCell = (1ll) << (2*(rayTraceData.rayOrder-rayTraceData.bundleOrder));
  int pbc = PRIMARY_BUNDLECELL,pbc_in;
  double time,minTime,maxTime,totTime,avgTime;

#ifdef USE_FULLSKY_PARTDIST
  fspd = 1;
#else
  fspd = 0;
#endif

  if (mapnum != NULL)
  {
    sprintf(name,"%s/rays_%d.%d",rayTraceData.OutputPath,mapnum, ThisTask);

  }
  else
  {
    sprintf(name,"%s/restart.%d",rayTraceData.OutputPath,ThisTask);
    sprintf(sys,"mv %s %s.bak",name,name);
  }

  mygroup = ThisTask/rayTraceData.NumFilesIOInParallel;
  Ngroups = NTasks/rayTraceData.NumFilesIOInParallel;
  if(Ngroups*rayTraceData.NumFilesIOInParallel != NTasks)
    ++Ngroups;

  for(currgroup=0;currgroup<Ngroups;++currgroup)
    {
      if(mygroup == currgroup)
	{
	  time = -MPI_Wtime();

	  //move old files to .bak files if writing
	  if((!read) && (mapnum != NULL))
	    system(sys);

	  if(read)
	    fp = fopen(name,"r");
	  else
	    fp = fopen(name,"w");

	  if(fp == NULL)
	    {
	      fprintf(stderr,"%d: could not open file '%s' for restart routine!\n",ThisTask,name);
	      MPI_Abort(MPI_COMM_WORLD,777);
	    }

	  //err check # of MPI tasks
	  if(!read)
	    NTasks_in = NTasks;
	  frw_io(&NTasks_in,sizeof(int),(size_t) 1,fp,read);
	  if(read && NTasks_in != NTasks)
	    {
	      fprintf(stderr,"%d: restart must use the same # of tasks! (curr,file NTasks = %d|%d)\n",ThisTask,NTasks,NTasks_in);
	      MPI_Abort(MPI_COMM_WORLD,777);
	    }

	  //err check USE_FULLSKY_PARTDIST
	  if(!read)
	    fspd_in = fspd;
	  frw_io(&fspd_in,sizeof(int),(size_t) 1,fp,read);
	  if(read && fspd_in != fspd)
	    {
	      fprintf(stderr,"%d: restart must define same USE_FULLSKY_PARTDIST option! (curr,file USE_FULLSKY_PARTDIST = %d|%d)\n",ThisTask,fspd,fspd_in);
	      MPI_Abort(MPI_COMM_WORLD,777);
	    }

	  //read/write global data struct - error check if reading
	  if(!read)
	    rtd_in = rayTraceData;
	  frw_io(&rtd_in,sizeof(RayTraceData),(size_t) 1,fp,read);
	  if(read &&
	     (rayTraceData.bundleOrder != rtd_in.bundleOrder ||
	      rayTraceData.rayOrder != rtd_in.rayOrder ||
	      rayTraceData.OmegaM != rtd_in.OmegaM ||
	      rayTraceData.maxComvDistance != rtd_in.maxComvDistance ||
	      rayTraceData.NumLensPlanes != rtd_in.NumLensPlanes ||
	      rayTraceData.minRa != rtd_in.minRa || rayTraceData.maxRa != rtd_in.maxRa ||
	      rayTraceData.minDec != rtd_in.minDec || rayTraceData.maxDec != rtd_in.maxDec))
	    {
	      if(rayTraceData.bundleOrder != rtd_in.bundleOrder || rayTraceData.rayOrder != rtd_in.rayOrder)
		fprintf(stderr,"%d: restart must use the same bundle and ray orders! (curr,file bundleOrder = %ld|%ld, curr,file rayOrder = %ld|%ld)\n",
			ThisTask,rayTraceData.bundleOrder,rtd_in.bundleOrder,rayTraceData.rayOrder,rtd_in.rayOrder);

	      if(rayTraceData.OmegaM != rtd_in.OmegaM)
		fprintf(stderr,"%d: restart must use the same OmegaM! (curr,file OmegaM = %lf|%lf)\n",ThisTask,rayTraceData.OmegaM,rtd_in.OmegaM);

	      if(rayTraceData.maxComvDistance != rtd_in.maxComvDistance || rayTraceData.NumLensPlanes != rtd_in.NumLensPlanes)
		fprintf(stderr,"%d: restart must use the same maxComvDistance and NumLensPlanes! (curr,file maxComvDistance = %lf|%lf, curr,file NumLensPlanes = %ld|%ld)\n",
			ThisTask,rayTraceData.maxComvDistance,rtd_in.maxComvDistance,rayTraceData.NumLensPlanes,rtd_in.NumLensPlanes);

	      if(rayTraceData.minRa != rtd_in.minRa || rayTraceData.maxRa != rtd_in.maxRa ||
		 rayTraceData.minDec != rtd_in.minDec || rayTraceData.maxDec != rtd_in.maxDec
		 )
		{
		  fprintf(stderr,"%d: restart must use the same ray domain! (curr,file minRa = %lf|%lf, curr,file maxRa = %lf|%lf, curr,file minDec = %lf|%lf, curr,file maxDec = %lf|%lf)\n",
			  ThisTask,rayTraceData.minRa,rtd_in.minRa,rayTraceData.maxRa,rtd_in.maxRa,
			  rayTraceData.minDec,rtd_in.minDec,rayTraceData.maxDec,rtd_in.maxDec);
		}

	      MPI_Abort(MPI_COMM_WORLD,777);
	    }
	  if(read)
	    rayTraceData.Restart = rtd_in.CurrentPlaneNum;

	  //read/write domain decomp info
	  frw_io(&NbundleCells,sizeof(long),(size_t) 1,fp,read);
	  if(read)
	    {
	      bundleCells = (HEALPixBundleCell*)malloc(sizeof(HEALPixBundleCell)*NbundleCells);
	      assert(bundleCells != NULL);

	      bundleCellsNest2RestrictedPeanoInd = (long*)malloc(sizeof(long)*NbundleCells);
	      assert(bundleCellsNest2RestrictedPeanoInd != NULL);

	      bundleCellsRestrictedPeanoInd2Nest = (long*)malloc(sizeof(long)*NbundleCells);
	      assert(bundleCellsRestrictedPeanoInd2Nest != NULL);
	    }
	  frw_io(bundleCells,(size_t) NbundleCells,sizeof(HEALPixBundleCell),fp,read);
	  frw_io(bundleCellsNest2RestrictedPeanoInd,(size_t) NbundleCells,sizeof(long),fp,read);
	  frw_io(bundleCellsRestrictedPeanoInd2Nest,(size_t) NbundleCells,sizeof(long),fp,read);

	  frw_io(&NrestrictedPeanoInd,sizeof(long),(size_t) 1,fp,read);
	  if(read)
	    {
	      firstRestrictedPeanoIndTasks = (long*)malloc(sizeof(long)*NTasks);
	      assert(firstRestrictedPeanoIndTasks != NULL);

	      lastRestrictedPeanoIndTasks = (long*)malloc(sizeof(long)*NTasks);
	      assert(lastRestrictedPeanoIndTasks != NULL);
	    }
	  frw_io(firstRestrictedPeanoIndTasks,(size_t) NTasks,sizeof(long),fp,read);
	  frw_io(lastRestrictedPeanoIndTasks,(size_t) NTasks,sizeof(long),fp,read);

	  //err check PRIMARY_BUNDLECELL bit flag value
	  if(!read)
	    pbc_in = pbc;
	  frw_io(&pbc_in,sizeof(int),(size_t) 1,fp,read);
	  if(read && pbc_in != pbc)
	    {
	      fprintf(stderr,"%d: restart must use the same PRIMARY_BUNDLECELL bit flag value! (curr,file PRIMARY_BUNDLECELL = %d|%d)\n",
		      ThisTask,pbc,pbc_in);
	      MPI_Abort(MPI_COMM_WORLD,777);
	    }

	  //read/write rays
	  if(read)
	    alloc_rays();
	  for(i=0;i<NbundleCells;++i)
	    if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
	      frw_io(bundleCells[i].rays,(size_t) NraysPerBundleCell,sizeof(HEALPixRay),fp,read);

	  fclose(fp);

	  time += MPI_Wtime();
	}

      //////////////////////////////
      MPI_Barrier(MPI_COMM_WORLD);
      //////////////////////////////
    }

  //get time info
  MPI_Reduce(&time,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&time,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&time,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgTime = totTime/((double) NTasks);

  if(ThisTask == 0)
    fprintf(stderr,"restart file I/O time max,min,avg = %lf|%lf|%lf (%.2f percent).\n\n"
	    ,maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0);
}

static size_t frw_io(void *p, size_t size, size_t nitems, FILE *fp, int read)
{
  size_t nrw;
  if(read)
    nrw = fread(p,size,nitems,fp);
  else
    nrw = fwrite(p,size,nitems,fp);

  if(nrw != nitems)
    {
      fprintf(stderr,"%d: error in restart routine read/write (%d)\n",ThisTask,read);
      MPI_Abort(MPI_COMM_WORLD,777);
    }

  return nrw;
}

void read_restart(void)
{
  restart_io(1, NULL);
}

void write_restart(void)
{
  restart_io(0, NULL);
}

void write_rays(long mapnum)
{
  restart_io(0, mapnum);
}

//remove gals from previous planes not needed during restart
void clean_gals_restart(void)
{
  long i;
  double rad,binL;
  long bind;
  long NumGalsForPreviousPlanes;
  SourceGal *tmpSourceGal;

  binL = rayTraceData.maxComvDistance/rayTraceData.NumLensPlanes;
  NumGalsForPreviousPlanes = 0;
  for(i=NumSourceGalsGlobal-1;i>=0;--i)
    {
      rad = sqrt(SourceGalsGlobal[i].pos[0]*SourceGalsGlobal[i].pos[0] +
                 SourceGalsGlobal[i].pos[1]*SourceGalsGlobal[i].pos[1] +
                 SourceGalsGlobal[i].pos[2]*SourceGalsGlobal[i].pos[2]);
      bind = (long) (rad/binL);

      if(bind < rayTraceData.CurrentPlaneNum)
        ++NumGalsForPreviousPlanes;
      else
	break;
    }

  if(NumGalsForPreviousPlanes > 0)
    {
      //decrement count and realloc
      NumSourceGalsGlobal -= NumGalsForPreviousPlanes;

      if(NumSourceGalsGlobal == 0)
	{
          free(SourceGalsGlobal);
          SourceGalsGlobal = NULL;
          NumSourceGalsGlobal = 0;
        }
      else
	{
          tmpSourceGal = (SourceGal*)realloc(SourceGalsGlobal,sizeof(SourceGal)*NumSourceGalsGlobal);
          assert(tmpSourceGal != NULL);
          SourceGalsGlobal = tmpSourceGal;
        }
    }
}
