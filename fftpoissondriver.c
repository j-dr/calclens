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
#include "fftpoissonsolve.h"
#include "gridcellhash.h"

#define WRAPIF(id,N) if(id >= N) id -= N; if(id < 0) id += N;

/* Notes for how to do this

1) compute for each bundle cell the range of grid cells needed
   for now do this step using an array of cells and inthash.c

2) sort cells by index

3) send/recv cells needed from other processors

4) do integral over the cells

*/

void threedpot_poissondriver(long planeNum, char *fbase)
{
  if(ThisTask == 0)
    fprintf(stderr,"FFT Poisson Driver is a stub!\n");

  //get units and lengths  
  double L,a,dL;
  get_units(fbase,&L,&a);
  dL = L/NFFT;
  double binL = (rayTraceData.maxComvDistance)/((double) (rayTraceData.NumLensPlanes));
  int Nint = binL/dL*2;
  double chimin = rayTraceData.planeRad - binL/2.0;
  double chimax = rayTraceData.planeRad + binL/2.0;
  double dchi = (chimax-chimin)/Nint;
  
  //init grid cell hash table
  GridCellHash *gch; = init_gchash();
  
  double vec[3];
  long i,j,k;
  long im1,jm1,km1;
  long ip1,jp1,kp1;
  long id,n,m;
  long di,dj,dk;
  long ii,jj,kk;
  double rad;
  long bind;
  long rind;
  long ind;
  
  long MaxNbundleCells;
  MPI_Allreduce(&NbundleCells,&MaxNbundleCells,1,MPI_LONG,MPI_MAX,MPI_COMM_WORLD);
  
  long Ngbuff = 0;
  GridCell *gbuff = NULL;
  
  int sendTask,recvTask;
  int level,log2NTasks = 0;
  long offset;
  long Nsend,Nrecv;
  MPI_Status Stat;
  while(NTasks > (1 << log2NTasks))
    ++log2NTasks;
  
  for(bind=0;bind<MaxNbundleCells;++bind) {
    //setup gridcell hash
    gch = init_gchash();
    
    //get grid cells needed
    if(bind < NbundleCells) {
      if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL)) {
	for(rind=0;rind<bundleCells[bind].Nrays;++rind) {
	  for(n=0;n<Nint;++n) {
	    //comp 3D loc
	    rad = chimin + n*dchi + 0.5*dchi;
	    vec[0] = bundleCells[bind].rays[rind].n[0]*rad;
	    vec[1] = bundleCells[bind].rays[rind].n[1]*rad;
	    vec[2] = bundleCells[bind].rays[rind].n[2]*rad;
	    
	    for(m=0;m<3;++m) {
	      while(vec[m] < 0)
		vec[m] += L;
	      while(vec[m] >= L)
		vec[m] -= L;
	    }
	    
	    i = (long) (vec[0]/dL);
	    WRAPIF(i,NFFT);
	    
	    j = (long) (vec[1]/dL);
	    WRAPIF(j,NFFT);
	    
	    k = (long) (vec[2]/dL);
	    WRAPIF(k,NFFT);
	    
	    //get all eight cells for interp plus those needed for all of the derivs
	    for(di=-1;di<=2;++di) {
	      ii = i + di;
	      WRAPIF(ii,NFFT);
	      
	      for(dj=-1;dj<=2;++dj) {
		jj = j + dj;
		WRAPIF(jj,NFFT);
		
		for(dk=-1;dk<=2;++dk) {
		  kk = k + dk;
		  WRAPIF(kk,NFFT);
		  
		  id = (ii*NFFT + kk)*NFFT + kk;
		  ind = getid_gchash(gch,id);
		}//for(dk=-1;dk<=2;++dk)
	      }//for(dj=-1;dj<=2;++dj)
	    }//for(di=-1;di<=2;++di)
	  }//for(n=0;n<Nint;++n)
	}//for(rind=0;rind<bundleCells[bind].Nrays;++rind)
      
	assert(gch->NumGridCells > 0);
      }//if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL))
    }//if(bind < NbundleCells)
    
    //sort to get into slab order
    sortcells_gchash(gch);
    
    //do send/recvs to get cells from other processors
    /*algorithm to loop through pairs of tasks linearly
      -lifted from Gadget-2 under GPL (http://www.gnu.org/copyleft/gpl.html)
      -see pm_periodic.c from Gadget-2 at http://www.mpa-garching.mpg.de/gadget/
    */  
    for(level = 0; level < (1 << log2NTasks); level++) {
      // note: for level=0, target is the same task
      sendTask = ThisTask;
      recvTask = ThisTask ^ level;
      if(recvTask < NTasks) { 
	//comp # of cells needed from other processor and offset
	offset = -1;
	Nrecv = 0;
	for(n=0;n<gch->NumGridCells;++n)
	  {
	    if(gch->GridCells[n].id >= TaskN0LocalStart[recvTask]*NFFT*NFFT &&
                 gch->GridCells[n].id < TaskN0LocalStart[recvTask]*NFFT*NFFT + TaskN0Local[recvTask]*NFFT*NFFT
	       && offset < 0)
	      offset = n;
	    if(gch->GridCells[n].id >= TaskN0LocalStart[recvTask]*NFFT*NFFT &&
	       gch->GridCells[n].id < TaskN0LocalStart[recvTask]*NFFT*NFFT + TaskN0Local[recvTask]*NFFT*NFFT)
	      {
		Nrecv += 1;
		id2ijk(gch->GridCells[n].id,NFFT,&i,&j,&k);
		assert(i >= TaskN0LocalStart[recvTask] && i < TaskN0LocalStart[recvTask]+TaskN0Local[recvTask]);
	      }
	    if(gch->GridCells[n].id >= TaskN0LocalStart[recvTask]*NFFT*NFFT + TaskN0Local[recvTask]*NFFT*NFFT)
	      break;
	  }
          
	if(!((offset >= 0 && Nrecv > 0) || (offset == -1 && Nrecv == 0)))
	  fprintf(stderr,"%d->%d Nrecv = %ld, offset = %ld, tot = %ld\n",sendTask,recvTask,Nrecv,offset,gch->NumGridCells);
	assert((offset >= 0 && Nrecv > 0) || (offset == -1 && Nrecv == 0));
	
	if(sendTask != recvTask) {
	  MPI_Sendrecv(&Nrecv,1,MPI_LONG,recvTask,TAG_POTCELL_NUM,
		       &Nsend,1,MPI_LONG,recvTask,TAG_POTCELL_NUM,
		       MPI_COMM_WORLD,&Stat);
	  
	  if(Nrecv > 0 || Nsend > 0) {
	    //get cells to send
	    if(Nsend > Ngbuff) {
	      gbuff = (GridCell*)realloc(gbuff,sizeof(GridCell)*Nsend);
	      assert(gbuff != NULL);
	      Ngbuff = Nsend;
	    }
	    MPI_Sendrecv(gch->GridCells+offset,sizeof(GridCell)*Nrecv,MPI_BYTE,recvTask,TAG_POTCELL_IDS,
			 gbuff,sizeof(GridCell)*Nsend,MPI_BYTE,recvTask,TAG_POTCELL_IDS,
			 MPI_COMM_WORLD,&Stat);
	    
	    //fill cells for other processor
	    for(m=0;m<Nsend;++m) {
	      id2ijk(gbuff[m].id,NFFT,&i,&j,&k);
	      assert(i >= N0LocalStart && i < N0LocalStart+N0Local);
	      gbuff[m].val = fftwrin[((i-N0LocalStart)*NFFT + j) * (2*(NFFT/2+1)) + k];
	    }
	    
	    //send cells to other processor
	    MPI_Sendrecv(gbuff,sizeof(GridCell)*Nsend,MPI_BYTE,recvTask,TAG_POTCELL_VALS,
			 gch->GridCells+offset,sizeof(GridCell)*Nrecv,MPI_BYTE,recvTask,TAG_POTCELL_VALS,
			 MPI_COMM_WORLD,&Stat);
	    
	  }// if(Nrecv > 0 || Nsend > 0)
	}// if(sendTask != recvTask)
	else {
	  //store pot
	  for(m=0;m<Nrecv;++m) {
	    id2ijk(gch->GridCells[m].id,NFFT,&i,&j,&k);
	    assert(i >= N0LocalStart && i < N0LocalStart+N0Local);
	    gch->GridCells[m].val = fftwrin[((i-N0LocalStart)*NFFT + j) * (2*(NFFT/2+1)) + k];
	  }
	  
	}//else for if(sendTask != recvTask) 
      }// if(recvTask < NTasks)
    }// for(level = 0; level < (1 << log2NTasks); level++)
    
    //interp to rays and comp derivs
    if(bind < NbundleCells) {
      if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL)) {
	//make sure buff cells are the same length as gch cells
	Ngbuff = gch->NumGridCells;
	gbuff = (GridCell*)realloc(gbuff,sizeof(GridCell)*Ngbuff);
	assert(gbuff != NULL);
	
	//do second derivs
	for(dind1=0;dind1<3;++dind1)
	  for(dind2=dind1;dind2<3;++dind2) {
	    //comp deriv for this direction
	    //mark cells with no deriv with -1 for id
	    for(m=0;m<gch-NumGridCells;++m)
	      {
		gbuff[m].id = -1;
		id2ijk(gch->GridCells[m].id,NFFT,&i,&j&k);
		
		im1 = i-1;
		WRAPIF(im1,NFFT);
		ip1 = i+1;
		WRAPIF(ip1,NFFT);
		
		jm1 = j-1;
		WRAPIF(jm1,NFFT);
		jp1 = j+1;
		WRAPIF(jp1,NFFT);
		
		km1 = k-1;
		WRAPIF(km1,NFFT);
		kp1 = k+1;
		WRAPIF(kp1,NFFT);
		
		if(dind1 == dind2) {
		  if(dind1 == 0) {
		    //FIXME start here
		    id = (i*NFFT+j)*NFFT+k;
		    ind = getonlyid_gchash(gch,id);
		    if(ind == GCH_INVALID)
		      continue;
		    gbuff[m].val = -2.0*(gch->GridCells[ind].val);
		    
		    id = (im1*NFFT+j)*NFFT+k;
		    ind = getonlyid_gchash(gch,id);
		    if(ind == GCH_INVALID)
		      continue;
		    gbuff[m].val += gch->GridCells[ind].val;
		      
		    id = (ip1*NFFT+j)*NFFT+k;
		    ind = getonlyid_gchash(gch,id);
		    if(ind == GCH_INVALID)
		      continue;
		    gbuff[m].val += gch->GridCells[ind].val;
		    
		    gbuff[m].val /= dL;
		  }
		  else if(dind1 == 1) {
		    
		  }
		  else if(dind1 == 2) {
		    
		  }
		}
		else {
		  //do first direction, then second
		  
		}
	      }
	    
	    //now add part needed to the rays
	    for(rind=0;rind<bundleCells[bind].Nrays;++rind) {
	      //comp projection matrix
	      
	      for(n=0;n<Nint;++n) {
		//do projetcion
		
		//add to ray
		
	      }//for(n=0;n<Nint;++n)
	    }//for(rind=0;rind<bundleCells[bind].Nrays;++rind)
	
	  }// for(dind2=dind1;dind2<3;++dind2)
      }//if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL))
    }//if(bind < NbundleCells)
    
    //clean it all up
    free_gchash(gch);
    if(Ngbuff > 0) {
      Ngbuff = 0;
      free(gbuff);
    }
  
  }//for(bind=0;bind<MaxNbundleCells;++bind)
}

static void get_units(char *fbase, double *L, double *a)
{
  char fname[MAX_FILENAME];

  sprintf(fname,"%s.0",fbase);

  if(ThisTask == 0)
    *L = get_period_length_LGADGET(fname);
  MPI_Bcast(L,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  *L = (*L)*rayTraceData.LengthConvFact;

  if(ThisTask == 0)
    *a = get_scale_factor_LGADGET(fname);
  MPI_Bcast(a,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
}
