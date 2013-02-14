#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>

#include "raytrace.h"

#ifdef DEBUG_IO
static void write_localmap(char name[], HEALPixMapCell *localMapCells, long NumLocalMapCells, int primaryOnly);
#endif

void gridkappadens(double densfact, double backdens)
{
  long oldPoissonOrder = rayTraceData.poissonOrder;
  double minSL;
  long i,j,k,n,m,gridOrder,maxGridOrder;  
  double timeGrid = 0,timeInterp = 0;
  //double minTime,maxTime,totTime,avgTime;
  long shift,queryOrder,queryNest,numQueryPixPerGridPix;
  double gs[HEALPIX_UTILS_MAXORDER+1];
  double times[5];
  
  for(i=0;i<5;++i)
    times[i] = 0.0;
  
  times[3] -= MPI_Wtime();
  for(i=0;i<=HEALPIX_UTILS_MAXORDER;++i)
    gs[i] = sqrt(4.0*M_PI/order2npix(i));
  
  minSL = lensPlaneParts[0].smoothingLength;
  for(i=0;i<NlensPlaneParts;++i)
    {
      if(lensPlaneParts[i].smoothingLength < minSL)
	minSL = lensPlaneParts[i].smoothingLength;
    }
  gridOrder = rayTraceData.bundleOrder;
  while(gs[gridOrder] > minSL/5.0)
    ++gridOrder;
  if(gridOrder > rayTraceData.rayOrder)
    gridOrder = rayTraceData.rayOrder;
  
  MPI_Allreduce(&gridOrder,&maxGridOrder,1,MPI_LONG,MPI_MAX,MPI_COMM_WORLD);
  
  rayTraceData.poissonOrder = maxGridOrder;
  times[3] += MPI_Wtime();

  //start timing here since the Allreduce call generally waits for the last task
  timeGrid -= MPI_Wtime();
  
  long bundleMapShift,mapNest,bundleNest;
  double vec[3],theta,phi;
  double Nwgt,wgt[4];
  double poissonHEALPixArea;
  double gyyMinusgxx,gyyPlusgxx;
  long wgtpix[4];
  long doNotHaveCell;
  long mapinds[4];
  long baseInd;
  double mapbuffrad;
#ifndef USE_FULLSKY_PARTDIST
  double ra,dec;
#endif
  double smoothingRad;
  long *listpix=NULL,Nlistpix=0,Ntotmass;
  double totmass,r,cosdis,nvec[3];
  double *listdens=NULL,*tmp;
  long Nlistdens = 0,NlistpixMax = 0;
  
  times[4] -= MPI_Wtime();
  
  bundleMapShift = 2*(rayTraceData.poissonOrder - rayTraceData.bundleOrder);
  poissonHEALPixArea = 4.0*M_PI/((double) (order2npix(rayTraceData.poissonOrder)));
  
  //mark buffer cells
  mapbuffrad = rayTraceData.maxSL*2.0 + sqrt(4.0*M_PI/order2npix(rayTraceData.bundleOrder)) + GRIDSEARCH_RADIUS_ARCMIN/60.0/180.0*M_PI;
  mark_bundlecells(mapbuffrad,PRIMARY_BUNDLECELL,GRIDKAPPADENS_MAPBUFF_BUNDLECELL);
  
  alloc_mapcells(PRIMARY_BUNDLECELL,GRIDKAPPADENS_MAPBUFF_BUNDLECELL);
  
  times[4] += MPI_Wtime();
  
  //grid parts
  for(i=0;i<NmapCells;++i)
    mapCells[i].val = 0.0;
  for(i=0;i<NbundleCells;++i)
    {
      if((ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[i].active,PARTBUFF_BUNDLECELL)) 
	 && 
	 bundleCells[i].Nparts > 0
	 &&
	 (ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[i].active,MAPBUFF_BUNDLECELL))
	 &&
	 bundleCells[i].firstMapCell >= 0)
	{
	  for(k=0;k<bundleCells[i].Nparts;++k)
	    {
	      times[0] -= MPI_Wtime();
	      vec[0] = (double) (lensPlaneParts[k+bundleCells[i].firstPart].pos[0]);
	      vec[1] = (double) (lensPlaneParts[k+bundleCells[i].firstPart].pos[1]);
	      vec[2] = (double) (lensPlaneParts[k+bundleCells[i].firstPart].pos[2]);
	      r = lensPlaneParts[k+bundleCells[i].firstPart].r;
	      vec2ang(vec,&theta,&phi);
	      smoothingRad = lensPlaneParts[k+bundleCells[i].firstPart].smoothingLength;
	      
	      queryOrder = 0;
	      while(gs[queryOrder] > smoothingRad/20.0 && queryOrder < rayTraceData.poissonOrder)
		++queryOrder;
	      
	      shift = 2*(rayTraceData.poissonOrder-queryOrder);
	      numQueryPixPerGridPix = (1ll) << shift;
	      assert(numQueryPixPerGridPix >= 1);
	      
	      Nlistpix = query_disc_inclusive_nest_fast(theta,phi,smoothingRad,&listpix,&NlistpixMax,queryOrder);
	      
	      if(Nlistdens < Nlistpix)
		{
		  tmp = (double*)realloc(listdens,sizeof(double)*Nlistpix);
		  
		  if(tmp != NULL)
		    {
		      listdens = tmp;
		      Nlistdens = Nlistpix;
		    }
		  else
		    {
		      fprintf(stderr,"%d: could not realloc memory for list dens!\n",ThisTask);
		      MPI_Abort(MPI_COMM_WORLD,123);
		    }
		}
	      times[0] += MPI_Wtime();
	      
	      times[1] -= MPI_Wtime();
	      totmass = 0.0;
	      Ntotmass = 0;
	      for(n=0;n<Nlistpix;++n)
		{
		  nest2vec(listpix[n],nvec,queryOrder);
		  cosdis = (vec[0]*nvec[0] + vec[1]*nvec[1] + vec[2]*nvec[2])/r;
		  listdens[n] = spline_part_dens(cosdis,smoothingRad);
		  if(listdens[n] > 0.0)
		    {
		      listpix[Ntotmass] = listpix[n];
		      listdens[Ntotmass] = listdens[n];
		      totmass += listdens[Ntotmass];
		      ++Ntotmass;
		    }
		}
	      
	      for(n=0;n<Ntotmass;++n)
		{
		  queryNest = listpix[n];
		  
		  for(m=0;m<numQueryPixPerGridPix;++m)
		    {
		      mapNest = (queryNest << shift) + m;
		      bundleNest = (mapNest >> bundleMapShift);
		      j = (bundleNest << bundleMapShift);
		      
		      if( (ISSETBITFLAG(bundleCells[bundleNest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,MAPBUFF_BUNDLECELL))
			  && bundleCells[bundleNest].firstMapCell >= 0)
			{
			  mapCells[bundleCells[bundleNest].firstMapCell+mapNest-j].val += 
			    (float) (listdens[n]/totmass/numQueryPixPerGridPix*lensPlaneParts[k+bundleCells[i].firstPart].mass);
			  
			  assert(mapNest == mapCells[bundleCells[bundleNest].firstMapCell+mapNest-j].index);
			}
		      
		      /* this error no longer applies since we are reading a different range of particle and map cells to save memory
			 else
			 {
			 //this error happens if bit 0 set and bit 1 set cells are no in the cells returned by get_interpol
			 fprintf(stderr,"%d: map buffer zones for patches are not big enough for kappa dens!\n",ThisTask);
			 MPI_Abort(MPI_COMM_WORLD,123);
			 }
		      */
		    }
		}
	      
	      times[1] += MPI_Wtime();
	    }
	}
    }
  
  if(Nlistdens > 0)
    free(listdens);
  
  if(NlistpixMax > 0)
    free(listpix);
  
  times[2] -= MPI_Wtime();
  //do the units and divide by the cell area
  for(i=0;i<NmapCells;++i)
    {
      mapCells[i].val *= (float) (densfact/poissonHEALPixArea);
      
#ifndef USE_FULLSKY_PARTDIST
      nest2ang(mapCells[i].index,&theta,&phi,rayTraceData.poissonOrder);
      ang2radec(theta,phi,&ra,&dec);
      if(test_vaccell_boundary(ra,dec,2.0*mapbuffrad))
	{
	  if(!(test_vaccell(ra,dec)))
	    mapCells[i].val -= (float) backdens;
	  else
	    mapCells[i].val = 0.0;
	}
      else
	mapCells[i].val -= (float) backdens;
#else
       mapCells[i].val -= (float) backdens;
#endif
    }
  times[2] += MPI_Wtime();
  
#ifdef DEBUG_IO
  char name[MAX_FILENAME];
  sprintf(name,"%s/gridkappadens%04ld.%04d",rayTraceData.OutputPath,rayTraceData.CurrentPlaneNum,ThisTask);
  write_localmap(name,mapCells,NmapCells,0);
#endif
  
  timeGrid += MPI_Wtime();
  timeInterp -= MPI_Wtime();
  
  //now set kappa
  Nwgt = 4;
  for(i=0;i<NbundleCells;++i)
    {
      if(ISSETBITFLAG(bundleCells[i].active,PRIMARY_BUNDLECELL))
        {
          for(j=0;j<bundleCells[i].Nrays;++j)
            {
              vec[0] = bundleCells[i].rays[j].n[0];
              vec[1] = bundleCells[i].rays[j].n[1];
              vec[2] = bundleCells[i].rays[j].n[2];
              vec2ang(vec,&theta,&phi);
              get_interpol(theta,phi,wgtpix,wgt,rayTraceData.poissonOrder);

	      gyyPlusgxx = 0.0;
	      
	      for(k=0;k<Nwgt;++k)
                {
                  if(wgtpix[k] >= 0)
                    {
                      doNotHaveCell = 0;

                      mapNest = ring2nest(wgtpix[k],rayTraceData.poissonOrder);
                      bundleNest = (mapNest >> bundleMapShift);
                      baseInd = (bundleNest << bundleMapShift);
                      if(bundleCells[bundleNest].firstMapCell >= 0
                         &&
                         (ISSETBITFLAG(bundleCells[bundleNest].active,PRIMARY_BUNDLECELL) || ISSETBITFLAG(bundleCells[bundleNest].active,MAPBUFF_BUNDLECELL))
                         )
                        {
                          mapinds[k] = bundleCells[bundleNest].firstMapCell + mapNest - baseInd;
                        }
                      else
                        doNotHaveCell = 1;

                      if(doNotHaveCell)
                        {
                          fprintf(stderr,"%d: buffer region for HEALPix map is not big enough for gridding kappa density! - theta,phi = %le|%le, nest = %ld, doNotHaveCell = %ld\n",
                                  ThisTask,theta,phi,bundleCells[i].rays[j].nest,doNotHaveCell);
                          MPI_Abort(MPI_COMM_WORLD,123);
                        }

                      assert(mapCells[mapinds[k]].index == mapNest);
		      
		      gyyPlusgxx += mapCells[mapinds[k]].val*wgt[k];
		    }
		}
	      
	      gyyMinusgxx = bundleCells[i].rays[j].U[3] - bundleCells[i].rays[j].U[0];
	      
	      bundleCells[i].rays[j].U[0] = (gyyPlusgxx - gyyMinusgxx)*0.5;
	      bundleCells[i].rays[j].U[3] = (gyyPlusgxx + gyyMinusgxx)*0.5;
	    }
        }
    }
  
  free_mapcells();
  
  rayTraceData.poissonOrder = oldPoissonOrder;
  
  timeInterp += MPI_Wtime();
  
  /*
  MPI_Reduce(&timeGrid,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&timeGrid,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&timeGrid,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgTime = totTime/((double) NTasks);
  
  if(ThisTask == 0)
    fprintf(stderr,"grid kappa dens time GRID max,min,avg = %lf|%lf|%lf - %.2f percent (total time %lf, order = %ld)\n"
            ,maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,totTime,maxGridOrder);
  
  MPI_Reduce(&timeInterp,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&timeInterp,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&timeInterp,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgTime = totTime/((double) NTasks);
  
  if(ThisTask == 0)
    fprintf(stderr,"grid kappa dens time INTERP max,min,avg = %lf|%lf|%lf - %.2f percent (total time %lf, order = %ld)\n"
            ,maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,totTime,maxGridOrder);
  
  timeInterp += timeGrid;
  
  MPI_Reduce(&timeInterp,&minTime,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(&timeInterp,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&timeInterp,&totTime,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  avgTime = totTime/((double) NTasks);
  
  if(ThisTask == 0)
  fprintf(stderr,"grid kappa dens time max,min,avg = %lf|%lf|%lf - %.2f percent (total time %lf, order = %ld)\n"
  ,maxTime,minTime,avgTime,(maxTime-avgTime)/avgTime*100.0,totTime,maxGridOrder);
  
  if(ThisTask == 0)
  fprintf(stderr,"%d: grid kappa dens time query,griddens,vaccell,prep,initmapcells = %lg|%lg|%lg|%lg|%lg\n",ThisTask,times[0],times[1],times[2],times[3],times[4]);
  
  */
  
#ifdef DEBUG
  if(ThisTask == 0)
    fprintf(stderr,"computed convergence field in %lg seconds. (initmapcells,query,griddens,vaccell,interp = %lg|%lg|%lg|%lg|%lg seconds)\n",
	    timeInterp+timeGrid,times[4],times[0],times[1],times[2],timeInterp);
#else
  if(ThisTask == 0)
    fprintf(stderr,"computed convergence field in %lg seconds.\n",timeInterp+timeGrid);
#endif
  
}

#ifdef DEBUG_IO
static void write_localmap(char name[], HEALPixMapCell *localMapCells, long NumLocalMapCells, int primaryOnly)
{
  long ring,i;
  FILE *fp;
    
  fp = fopen(name,"w");
  assert(fp != NULL);
  ring = order2nside(rayTraceData.poissonOrder);
  fwrite(&ring,(size_t) 1,sizeof(long),fp);
  
  long n;
  long shift,nest;
  shift = 2*(rayTraceData.poissonOrder-rayTraceData.bundleOrder);
  if(primaryOnly)
    {
      n = 0;
      for(i=0;i<NumLocalMapCells;++i)
	{
	  nest = localMapCells[i].index >> shift;
	  if(ISSETBITFLAG(bundleCells[nest].active,PRIMARY_BUNDLECELL))
	    ++n;
	}
      
      fwrite(&n,(size_t) 1,sizeof(long),fp);
      for(i=0;i<NumLocalMapCells;++i)
	{
	  nest = localMapCells[i].index >> shift;
	  if(ISSETBITFLAG(bundleCells[nest].active,PRIMARY_BUNDLECELL))
	    {
	      fwrite(&(localMapCells[i].val),(size_t) 1,sizeof(float),fp);
	      ring = nest2ring(localMapCells[i].index,rayTraceData.poissonOrder);
	      fwrite(&ring,(size_t) 1,sizeof(long),fp);
	    }
	}
    }
  else
    { 
      fwrite(&NumLocalMapCells,(size_t) 1,sizeof(long),fp);
      for(i=0;i<NumLocalMapCells;++i)
	{
	  fwrite(&(localMapCells[i].val),(size_t) 1,sizeof(float),fp);
	  ring = nest2ring(localMapCells[i].index,rayTraceData.poissonOrder);
	  fwrite(&ring,(size_t) 1,sizeof(long),fp);
	}
    }
  
  fclose(fp);

}
#endif
