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
#include "lgadgetio.h"

#define WRAPIF(id,N) {if(id >= N) id -= N; if(id < 0) id += N;}
#define THREEDIND(i,j,k,N) ((i*N+j)*N+k)

typedef struct {
  char fname[MAX_FILENAME];
  double a;
  double chi;
} NbodySnap;

static void read_snaps(NbodySnap **snaps, long *Nsnaps);
static void get_units(char *fbase, double *L, double *a);

/* Notes for how to do this

1) compute for each bundle cell the range of grid cells needed
   for now do this step using an array of cells and inthash.c

2) sort cells by index

3) send/recv cells needed from other processors

4) do integral over the cells

*/

void threedpot_poissondriver(void)
{
  //make sure compute FFT of correct snap
  static long currFTTsnap = -1;
  static long initFTTsnaps = 1;
  static long Nsnaps;
  static NbodySnap *snaps;
  char fbase[MAX_FILENAME];
  
  if(initFTTsnaps == 1) {
    initFTTsnaps = 0;
    
    read_snaps(&snaps,&Nsnaps);
    
    //init FFTs
    init_ffts();
    alloc_and_plan_ffts();
  }
  
  //get closest snap
  long i;
  long mysnap = 0;
  double dsnap = fabs(snaps[mysnap].chi-rayTraceData.planeRad);
  for(i=0;i<Nsnaps;++i) {
    if(fabs(snaps[i].chi-rayTraceData.planeRad) < dsnap) {
      mysnap = i;
      dsnap = fabs(snaps[i].chi-rayTraceData.planeRad);
    }
  }
  sprintf(fbase,"%s",snaps[mysnap].fname); 
  
  //solve for potential
  double t0;
  if(mysnap != currFTTsnap) {
    
    currFTTsnap = mysnap;
    
    t0 = -MPI_Wtime();
    if(ThisTask == 0) {
      fprintf(stderr,"getting potential for snapshot %ld.\n",currFTTsnap);
      fflush(stderr);
    }
    
    comp_pot_snap(snaps[mysnap].fname);
    
    t0 += MPI_Wtime();
    if(ThisTask == 0) {
      fprintf(stderr,"got potential for snapshot %ld in %lf seconds.\n",currFTTsnap,t0);
      fflush(stderr);
    }
  }  
  
  t0 = -MPI_Wtime();
  if(ThisTask == 0) {
    fprintf(stderr,"doing interp and integral to rays.\n",currFTTsnap);
    fflush(stderr);
  }
  
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
  GridCellHash *gch;
  
  double vec[3];
  long j,k;
  long ip1,jp1,kp1;
  long id,n,m;
  long di,dj,dk;
  long ii,jj,kk;
  double rad;
  long bind;
  long rind;
  long ind;
  long pp[3],pm[3],mp[3],mm[3];
  long indvec[3][3][3];
  double cost,cosp,sint,sinp;
  double theta,phi,r;
  double dx,dy,dz;
  double val,fac1,fac2;
  
  long NumActiveBundleCells;
  long *activeBundleCellInds;
  long MaxNumActiveBundleCells;
  NumActiveBundleCells = 0;
  for(bind=0;bind<NbundleCells;++bind) {
    if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL)) {
      ++NumActiveBundleCells;
    }
  }
  activeBundleCellInds = (long*)malloc(sizeof(long)*NumActiveBundleCells);
  assert(activeBundleCellInds != NULL);
  n = 0;
  for(bind=0;bind<NbundleCells;++bind) {
    if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL)) {
      activeBundleCellInds[n] = bind;
      ++n;
    }
  }
  assert(n == NumActiveBundleCells);
  MPI_Allreduce(&NumActiveBundleCells,&MaxNumActiveBundleCells,1,MPI_LONG,MPI_MAX,MPI_COMM_WORLD);
  long abind;
  
  long Ngbuff = 0;
  GridCell *gbuff = NULL;
  
  int sendTask,recvTask;
  int level,log2NTasks = 0;
  long offset;
  long Nsend,Nrecv;
  MPI_Status Stat;
  while(NTasks > (1 << log2NTasks))
    ++log2NTasks;
  
  for(abind=0;abind<MaxNumActiveBundleCells;++abind) {
    //setup gridcell hash
    gch = init_gchash();
    
    //get index of bundle cell working with
    if(abind < NumActiveBundleCells) {
      bind = activeBundleCellInds[abind];
    }
    
    //get grid cells needed
    if(abind < NumActiveBundleCells) {
      if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL)) {
	for(rind=0;rind<bundleCells[bind].Nrays;++rind) {
	  r = sqrt(bundleCells[bind].rays[rind].n[0]*bundleCells[bind].rays[rind].n[0] + 
		   bundleCells[bind].rays[rind].n[1]*bundleCells[bind].rays[rind].n[1] + 
		   bundleCells[bind].rays[rind].n[2]*bundleCells[bind].rays[rind].n[2]);
	  
	  for(n=0;n<Nint;++n) {
	    //comp 3D loc
	    rad = chimin + n*dchi + 0.5*dchi;
	    
	    vec[0] = bundleCells[bind].rays[rind].n[0]*rad/r;
	    vec[1] = bundleCells[bind].rays[rind].n[1]*rad/r;
	    vec[2] = bundleCells[bind].rays[rind].n[2]*rad/r;
	    
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
		  
		  id = THREEDIND(ii,jj,kk,NFFT);
		  //id = (ii*NFFT + jj)*NFFT + kk;
		  ind = getid_gchash(gch,id);
		}//for(dk=-1;dk<=2;++dk)
	      }//for(dj=-1;dj<=2;++dj)
	    }//for(di=-1;di<=2;++di)
	  }//for(n=0;n<Nint;++n)
	}//for(rind=0;rind<bundleCells[bind].Nrays;++rind)
      
	assert(gch->NumGridCells > 0);
	//fprintf(stderr,"%04ld: using %ld gridCells for pot and derivs.\n",ThisTask,gch->NumGridCells); fflush(stderr); //FIXME
	
      }//if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL))
    }//if(abind < NumActiveBundleCells)
    
    //sort to get into slab order
    //if(abind < NumActiveBundleCells)
    //fprintf(stderr,"%04ld: sorting gridCells for pot and derivs. %ld\n",ThisTask,gch->NumGridCells); fflush(stderr); //FIXME
    sortcells_gchash(gch);
    //if(abind < NumActiveBundleCells)
    //fprintf(stderr,"%04ld: sorted gridCells for pot and derivs. %ld\n",ThisTask,gch->NumGridCells); fflush(stderr); //FIXME
    
    //if(abind < NumActiveBundleCells)
    //fprintf(stderr,"%04ld: getting gridCells for pot and derivs. %ld\n",ThisTask,gch->NumGridCells); fflush(stderr); //FIXME
    
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
	
	if(!((offset >= 0 && Nrecv > 0) || (offset == -1 && Nrecv == 0))) {
	  fprintf(stderr,"%04ld: %d->%d Nrecv = %ld, offset = %ld, tot = %ld\n",ThisTask,sendTask,recvTask,Nrecv,offset,gch->NumGridCells);
	  fflush(stderr);
	}
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
	      
	      //FIXME
	      if(!(i >= N0LocalStart && i < N0LocalStart+N0Local)) {
		fprintf(stderr,"%04ld: send != recv slab assertion going to fail! %s:%d\n",ThisTask,__FILE__,__LINE__);
		fflush(stderr);
	      }
	      
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
	    id2ijk(gch->GridCells[m+offset].id,NFFT,&i,&j,&k);
	    
	    //FIXME
	    if(!(i >= N0LocalStart && i < N0LocalStart+N0Local)) {
	      fprintf(stderr,"%04ld: send == recv slab assertion going to fail! %s:%d\n",ThisTask,__FILE__,__LINE__);
	      fflush(stderr);
	    }
	    
	    assert(i >= N0LocalStart && i < N0LocalStart+N0Local);
	    gch->GridCells[m+offset].val = fftwrin[((i-N0LocalStart)*NFFT + j) * (2*(NFFT/2+1)) + k];
	  }//for(m=0;m<Nrecv;++m
	}//else for if(sendTask != recvTask) 
      }// if(recvTask < NTasks)
    }// for(level = 0; level < (1 << log2NTasks); level++)
    
    //if(abind < NumActiveBundleCells)
    //fprintf(stderr,"%04ld: got gridCells for pot and derivs. %ld\n",ThisTask,gch->NumGridCells); fflush(stderr); //FIXME
    
    //interp to rays and comp derivs
    int dind1,dind2;
    double jac[3][3];
    if(abind < NumActiveBundleCells) {
      if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL)) {
	
	//fprintf(stderr,"%04ld: doing derivs of gridCells for pot and derivs.\n",ThisTask); fflush(stderr); //FIXME
	
	//make sure buff cells are the same length as gch cells
	Ngbuff = gch->NumGridCells;
	gbuff = (GridCell*)realloc(gbuff,sizeof(GridCell)*Ngbuff);
	assert(gbuff != NULL);
	
	//do first derivs
	for(dind1=0;dind1<3;++dind1) {
	  //comp deriv for this direction
	  //mark cells with no deriv with -1 for id
	  for(m=0;m<gch->NumGridCells;++m)
	    {
	      //get ids of nbr cells
	      gbuff[m].id = -1;
	      id2ijk(gch->GridCells[m].id,NFFT,&i,&j,&k);
	      
	      for(di=-1;di<=1;++di) {
		ii = i + di;
		WRAPIF(ii,NFFT);
		for(dj=-1;dj<=1;++dj) {
		  jj = j + dj;
		  WRAPIF(jj,NFFT);
		  for(dk=-1;dk<=1;++dk) {
		    kk = k + dk;
		    WRAPIF(kk,NFFT);
		    
		    indvec[di+1][dj+1][dk+1] = THREEDIND(ii,jj,kk,NFFT);
		    //indvec[di+1][dj+1][dk+1] = (ii*NFFT+jj)*NFFT+kk; 
		  }
		}
	      }
	      
	      //get derivs
	      //build the stencil
	      for(n=0;n<3;++n) {
		if(n == dind1) {
		  pp[n] = 2;
		  pm[n] = 0;
		} 
		else {
		  pp[n] = 1;
		  pm[n] = 1;
		}
	      }
		  
	      //eval stencil parts
	      gbuff[m].val = 0.0;
		  
	      id = indvec[pp[0]][pp[1]][pp[2]];
	      ind = getonlyid_gchash(gch,id);
	      if(ind == GCH_INVALID)
		continue;
	      gbuff[m].val += gch->GridCells[ind].val;
		  
	      id = indvec[pm[0]][pm[1]][pm[2]];
	      ind = getonlyid_gchash(gch,id);
	      if(ind == GCH_INVALID)
		continue;
	      gbuff[m].val -= gch->GridCells[ind].val;
	      
	      gbuff[m].val /= dL;
	      gbuff[m].val /= 2.0;
	      gbuff[m].id = gch->GridCells[m].id;
	    }//for(m=0;m<gch-NumGridCells;++m)
	
	  //now add part needed to the rays
	  for(rind=0;rind<bundleCells[bind].Nrays;++rind) {
	    //comp jacobian matrix
	    vec2ang(bundleCells[bind].rays[rind].n,&theta,&phi);
	    cost = cos(theta);
	    sint = sin(theta);
	    cosp = cos(phi);
	    sinp = sin(phi);
	    
	    jac[0][0] = cosp*cost;
	    jac[0][1] = -sinp;
	    jac[0][2] = cosp*sint;
	    
	    jac[1][0] = sinp*cost;
	    jac[1][1] = cosp;
	    jac[1][2] = sinp*sint;
	    
	    jac[2][0] = -sint;
	    jac[2][1] = 0.0;
	    jac[2][2] = cost;
	    
	    r = sqrt(bundleCells[bind].rays[rind].n[0]*bundleCells[bind].rays[rind].n[0] +
		     bundleCells[bind].rays[rind].n[1]*bundleCells[bind].rays[rind].n[1] +
		     bundleCells[bind].rays[rind].n[2]*bundleCells[bind].rays[rind].n[2]);
	    
	    for(n=0;n<Nint;++n) {
	      //comp 3D loc
	      rad = chimin + n*dchi + 0.5*dchi;
	      
	      vec[0] = bundleCells[bind].rays[rind].n[0]*rad/r;
	      vec[1] = bundleCells[bind].rays[rind].n[1]*rad/r;
	      vec[2] = bundleCells[bind].rays[rind].n[2]*rad/r;
	      
	      for(m=0;m<3;++m) {
		while(vec[m] < 0)
		  vec[m] += L;
		while(vec[m] >= L)
		  vec[m] -= L;
	      }
	      
	      i = (long) (vec[0]/dL);
	      dx = (vec[0] - i*dL)/dL;
	      WRAPIF(i,NFFT);
	      ip1 = i + 1;
	      WRAPIF(ip1,NFFT);
	      
	      j = (long) (vec[1]/dL);
	      dy = (vec[1] - j*dL)/dL;
	      WRAPIF(j,NFFT);
	      jp1 = j + 1;
	      WRAPIF(jp1,NFFT);
	      
	      k = (long) (vec[2]/dL);
	      dz = (vec[2] - k*dL)/dL;
	      WRAPIF(k,NFFT);
	      kp1 = k + 1;
	      WRAPIF(kp1,NFFT);
	      
	      //interp deriv val
	      val = 0.0;
	      
	      //id = (i*NFFT + j)*NFFT + k;
	      id = THREEDIND(i,j,k,NFFT);
	      ind = getonlyid_gchash(gch,id);
	      assert(ind != GCH_INVALID);
	      assert(gbuff[ind].id != -1);
	      assert(gbuff[ind].id == gch->GridCells[ind].id);
	      val += gbuff[ind].val*(1.0 - dx)*(1.0 - dy)*(1.0 - dz);
	      
	      //id = (i*NFFT + j)*NFFT + kp1;
	      id = THREEDIND(i,j,kp1,NFFT);
	      ind = getonlyid_gchash(gch,id);
	      assert(ind != GCH_INVALID);
	      assert(gbuff[ind].id != -1);
	      assert(gbuff[ind].id == gch->GridCells[ind].id);
	      val += gbuff[ind].val*(1.0 - dx)*(1.0 - dy)*dz;
	      
	      //id = (i*NFFT + jp1)*NFFT + k;
	      id = THREEDIND(i,jp1,k,NFFT);
	      ind = getonlyid_gchash(gch,id);
	      assert(ind != GCH_INVALID);
	      assert(gbuff[ind].id != -1);
	      assert(gbuff[ind].id == gch->GridCells[ind].id);
	      val += gbuff[ind].val*(1.0 - dx)*dy*(1.0 - dz);
	      
	      //id = (i*NFFT + jp1)*NFFT + kp1;
	      id = THREEDIND(i,jp1,kp1,NFFT);
	      ind = getonlyid_gchash(gch,id);
	      assert(ind != GCH_INVALID);
	      assert(gbuff[ind].id != -1);
	      assert(gbuff[ind].id == gch->GridCells[ind].id);
	      val += gbuff[ind].val*(1.0 - dx)*dy*dz;
	      
	      
	      //id = (ip1*NFFT + j)*NFFT + k;
	      id = THREEDIND(ip1,j,k,NFFT);
	      ind = getonlyid_gchash(gch,id);
	      assert(ind != GCH_INVALID);
	      assert(gbuff[ind].id != -1);
	      assert(gbuff[ind].id == gch->GridCells[ind].id);
	      val += gbuff[ind].val*dx*(1.0 - dy)*(1.0 - dz);
	      
	      //id = (ip1*NFFT + j)*NFFT + kp1;
	      id = THREEDIND(ip1,j,kp1,NFFT);
	      ind = getonlyid_gchash(gch,id);
	      assert(ind != GCH_INVALID);
	      assert(gbuff[ind].id != -1);
	      assert(gbuff[ind].id == gch->GridCells[ind].id);
	      val += gbuff[ind].val*dx*(1.0 - dy)*dz;
	      
	      //id = (ip1*NFFT + jp1)*NFFT + k;
	      id = THREEDIND(ip1,jp1,k,NFFT);
	      ind = getonlyid_gchash(gch,id);
	      assert(ind != GCH_INVALID);
	      assert(gbuff[ind].id != -1);
	      assert(gbuff[ind].id == gch->GridCells[ind].id);
	      val += gbuff[ind].val*dx*dy*(1.0 - dz);
	      
	      //id = (ip1*NFFT + jp1)*NFFT + kp1;
	      id = THREEDIND(ip1,jp1,kp1,NFFT);
	      ind = getonlyid_gchash(gch,id);
	      assert(ind != GCH_INVALID);
	      assert(gbuff[ind].id != -1);
	      assert(gbuff[ind].id == gch->GridCells[ind].id);
	      val += gbuff[ind].val*dx*dy*dz;
	      
	      //do the projections and add to ray
	      for(ii=0;ii<2;++ii)
		bundleCells[bind].rays[rind].alpha[ii] += val*jac[dind1][ii];
	      
	      //FIXME DEBUG
	      assert(gsl_finite(bundleCells[bind].rays[rind].alpha[0]));
	      assert(gsl_finite(bundleCells[bind].rays[rind].alpha[1]));
	      
	    }//for(n=0;n<Nint;++n)
	  }//for(rind=0;rind<bundleCells[bind].Nrays;++rind)
	}//for(dind1=0;dind1<3;++dind1)
	
	//do second derivs
	for(dind1=0;dind1<3;++dind1)
	  for(dind2=dind1;dind2<3;++dind2) {
	    //comp deriv for this direction
	    //mark cells with no deriv with -1 for id
	    for(m=0;m<gch->NumGridCells;++m)
	      {
		//get ids of nbr cells
		gbuff[m].id = -1;
		id2ijk(gch->GridCells[m].id,NFFT,&i,&j,&k);
		
		for(di=-1;di<=1;++di) {
		  ii = i + di;
		  WRAPIF(ii,NFFT);
		  for(dj=-1;dj<=1;++dj) {
		    jj = j + dj;
		    WRAPIF(jj,NFFT);
		    for(dk=-1;dk<=1;++dk) {
		      kk = k + dk;
		      WRAPIF(kk,NFFT);
		      
		      indvec[di+1][dj+1][dk+1] = THREEDIND(ii,jj,kk,NFFT); //(ii*NFFT+jj)*NFFT+kk; 
		    }
		  }
		}
		
		//get derivs
		if(dind1 == dind2) {
		  //build the stencil
		  for(n=0;n<3;++n) {
		    if(n == dind1) {
		      pp[n] = 2;
		      pm[n] = 0;
		    } 
		    else {
		      pp[n] = 1;
		      pm[n] = 1;
		    }
		  }
		  
		  //eval stencil parts
		  gbuff[m].val = -2.0*(gch->GridCells[m].val);
		  
		  id = indvec[pp[0]][pp[1]][pp[2]];
		  ind = getonlyid_gchash(gch,id);
		  if(ind == GCH_INVALID)
		    continue;
		  gbuff[m].val += gch->GridCells[ind].val;
		  
		  id = indvec[pm[0]][pm[1]][pm[2]];
		  ind = getonlyid_gchash(gch,id);
		  if(ind == GCH_INVALID)
		    continue;
		  gbuff[m].val += gch->GridCells[ind].val;
		  
		  gbuff[m].val /= dL;
		  gbuff[m].val /= dL;
		  gbuff[m].id = gch->GridCells[m].id;
		} 
		else {
		  //build the stencil
		  for(n=0;n<3;++n) {
		    if(n == dind1) {
		      pp[n] = 2;
		      pm[n] = 2;
		      mp[n] = 0;
		      mm[n] = 0;
		    }
		    else if(n == dind2) {
		      pp[n] = 2;
		      pm[n] = 0;
		      mp[n] = 2;
		      mm[n] = 0;
		    }
		    else {
		      pp[n] = 1;
		      pm[n] = 1;
		      mp[n] = 1;
		      mm[n] = 1;
		    }
		  }
		  
		  //eval stencil parts
		  gbuff[n].val = 0.0;
		  
		  id = indvec[pp[0]][pp[1]][pp[2]];
                  ind = getonlyid_gchash(gch,id);
                  if(ind == GCH_INVALID)
                    continue;
                  gbuff[m].val += gch->GridCells[ind].val;
		  
                  id = indvec[pm[0]][pm[1]][pm[2]];
                  ind = getonlyid_gchash(gch,id);
                  if(ind == GCH_INVALID)
                    continue;
                  gbuff[m].val -= gch->GridCells[ind].val;
		  
		  id = indvec[mp[0]][mp[1]][mp[2]];
                  ind = getonlyid_gchash(gch,id);
                  if(ind == GCH_INVALID)
                    continue;
                  gbuff[m].val -= gch->GridCells[ind].val;
		  
		  id = indvec[mm[0]][mm[1]][mm[2]];
                  ind = getonlyid_gchash(gch,id);
                  if(ind == GCH_INVALID)
                    continue;
                  gbuff[m].val += gch->GridCells[ind].val;
		  
                  gbuff[m].val /= dL;
		  gbuff[m].val /= dL;
		  gbuff[m].val /= 2.0;
		  gbuff[m].val /= 2.0;
                  gbuff[m].id = gch->GridCells[m].id;
		}//end of else
		
	      }//for(m=0;m<gch-NumGridCells;++m)
	    
	    //now add part needed to the rays
	    for(rind=0;rind<bundleCells[bind].Nrays;++rind) {
	      //comp jacobian matrix
	      vec2ang(bundleCells[bind].rays[rind].n,&theta,&phi);
	      cost = cos(theta);
	      sint = sin(theta);
	      cosp = cos(phi);
	      sinp = sin(phi);
	      
	      jac[0][0] = cosp*cost;
	      jac[0][1] = -sinp;
	      jac[0][2] = cosp*sint;
	      
	      jac[1][0] = sinp*cost;
	      jac[1][1] = cosp;
	      jac[1][2] = sinp*sint;
	      
	      jac[2][0] = -sint;
	      jac[2][1] = 0.0;
	      jac[2][2] = cost;
	      
	      r = sqrt(bundleCells[bind].rays[rind].n[0]*bundleCells[bind].rays[rind].n[0] +
		       bundleCells[bind].rays[rind].n[1]*bundleCells[bind].rays[rind].n[1] +
		       bundleCells[bind].rays[rind].n[2]*bundleCells[bind].rays[rind].n[2]);
	      
	      for(n=0;n<Nint;++n) {
		//comp 3D loc
		rad = chimin + n*dchi + 0.5*dchi;
		
		vec[0] = bundleCells[bind].rays[rind].n[0]*rad/r;
		vec[1] = bundleCells[bind].rays[rind].n[1]*rad/r;
		vec[2] = bundleCells[bind].rays[rind].n[2]*rad/r;
		
		for(m=0;m<3;++m) {
		  while(vec[m] < 0)
		    vec[m] += L;
		  while(vec[m] >= L)
		    vec[m] -= L;
		}
		
		i = (long) (vec[0]/dL);
		dx = (vec[0] - i*dL)/dL;
		WRAPIF(i,NFFT);
		ip1 = i + 1;
		WRAPIF(ip1,NFFT);
		
		j = (long) (vec[1]/dL);
		dy = (vec[1] - j*dL)/dL;
		WRAPIF(j,NFFT);
		jp1 = j + 1;
		WRAPIF(jp1,NFFT);
		
		k = (long) (vec[2]/dL);
		dz = (vec[2] - k*dL)/dL;
		WRAPIF(k,NFFT);
		kp1 = k + 1;
		WRAPIF(kp1,NFFT);
		
		//interp deriv val
		val = 0.0;
		
		//id = (i*NFFT + j)*NFFT + k;
		id = THREEDIND(i,j,k,NFFT);
		ind = getonlyid_gchash(gch,id);
		assert(ind != GCH_INVALID);
		assert(gbuff[ind].id != -1);
		assert(gbuff[ind].id == gch->GridCells[ind].id);
		val += gbuff[ind].val*(1.0 - dx)*(1.0 - dy)*(1.0 - dz);
		
		//id = (i*NFFT + j)*NFFT + kp1;
		id = THREEDIND(i,j,kp1,NFFT);
		ind = getonlyid_gchash(gch,id);
		assert(ind != GCH_INVALID);
		assert(gbuff[ind].id != -1);
		assert(gbuff[ind].id == gch->GridCells[ind].id);
		val += gbuff[ind].val*(1.0 - dx)*(1.0 - dy)*dz;
		
		//id = (i*NFFT + jp1)*NFFT + k;
		id = THREEDIND(i,jp1,k,NFFT);
		ind = getonlyid_gchash(gch,id);
		assert(ind != GCH_INVALID);
		assert(gbuff[ind].id != -1);
		assert(gbuff[ind].id == gch->GridCells[ind].id);
		val += gbuff[ind].val*(1.0 - dx)*dy*(1.0 - dz);
		
		//id = (i*NFFT + jp1)*NFFT + kp1;
		id = THREEDIND(i,jp1,k,NFFT);
		ind = getonlyid_gchash(gch,id);
		assert(ind != GCH_INVALID);
		assert(gbuff[ind].id != -1);
		assert(gbuff[ind].id == gch->GridCells[ind].id);
		val += gbuff[ind].val*(1.0 - dx)*dy*dz;
		
		//id = (ip1*NFFT + j)*NFFT + k;
		id = THREEDIND(ip1,j,k,NFFT);
		ind = getonlyid_gchash(gch,id);
		assert(ind != GCH_INVALID);
		assert(gbuff[ind].id != -1);
		assert(gbuff[ind].id == gch->GridCells[ind].id);
		val += gbuff[ind].val*dx*(1.0 - dy)*(1.0 - dz);
		
		//id = (ip1*NFFT + j)*NFFT + kp1;
		id = THREEDIND(ip1,j,kp1,NFFT);
		ind = getonlyid_gchash(gch,id);
		assert(ind != GCH_INVALID);
		assert(gbuff[ind].id != -1);
		assert(gbuff[ind].id == gch->GridCells[ind].id);
		val += gbuff[ind].val*dx*(1.0 - dy)*dz;
		
		//id = (ip1*NFFT + jp1)*NFFT + k;
		id = THREEDIND(ip1,jp1,k,NFFT);
		ind = getonlyid_gchash(gch,id);
		assert(ind != GCH_INVALID);
		assert(gbuff[ind].id != -1);
		assert(gbuff[ind].id == gch->GridCells[ind].id);
		val += gbuff[ind].val*dx*dy*(1.0 - dz);
		
		//id = (ip1*NFFT + jp1)*NFFT + kp1;
		id = THREEDIND(ip1,jp1,kp1,NFFT);
		ind = getonlyid_gchash(gch,id);
		assert(ind != GCH_INVALID);
		assert(gbuff[ind].id != -1);
		assert(gbuff[ind].id == gch->GridCells[ind].id);
		val += gbuff[ind].val*dx*dy*dz;
		
		//do the projections and add to ray
		for(ii=0;ii<2;++ii)
		  for(jj=0;jj<2;++jj)
		    bundleCells[bind].rays[rind].U[ii*2+jj] += val*jac[dind1][ii]*jac[dind2][jj];
		
		if(dind1 != dind2) {
		  for(ii=0;ii<2;++ii)
		    for(jj=0;jj<2;++jj)
		      bundleCells[bind].rays[rind].U[ii*2+jj] += val*jac[dind2][ii]*jac[dind1][jj];
		}
		
		//FIXME DEBUG
		assert(gsl_finite(bundleCells[bind].rays[rind].U[0]));
		assert(gsl_finite(bundleCells[bind].rays[rind].U[1]));
		assert(gsl_finite(bundleCells[bind].rays[rind].U[2]));
		assert(gsl_finite(bundleCells[bind].rays[rind].U[3]));
		
	      }//for(n=0;n<Nint;++n)
	    }//for(rind=0;rind<bundleCells[bind].Nrays;++rind)
	
	  }// for(dind2=dind1;dind2<3;++dind2)

	//fprintf(stderr,"%04ld: finished derivs of gridCells for pot and derivs.\n",ThisTask); fflush(stderr); //FIXME
      }//if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL))
    }//if(abind < NumActiveBundleCells)
    
    //get units right
    if(abind < NumActiveBundleCells) {
      if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL)) {
	//fac for second derivs 2.0/CSOL/CSOL*dchi/chi*chi*chi
	fac2 = 2.0/CSOL/CSOL*dchi*rayTraceData.planeRad;
	
	//fac for first derivs 2.0/CSOL/CSOL*dchi/chi*chi
	fac1 = 2.0/CSOL/CSOL*dchi;

	for(rind=0;rind<bundleCells[bind].Nrays;++rind) {
	  for(ii=0;ii<2;++ii)
	    for(jj=0;jj<2;++jj)
	      bundleCells[bind].rays[rind].U[ii*2+jj] *= fac2;
	  
	  for(ii=0;ii<2;++ii)
	    bundleCells[bind].rays[rind].alpha[ii] *= fac1;
	  
	}//for(rind=0;rind<bundleCells[bind].Nrays;++rind)
      }//if(ISSETBITFLAG(bundleCells[bind].active,PRIMARY_BUNDLECELL))
    }//if(abind < NumActiveBundleCells)
    
    //clean it all up
    free_gchash(gch);
    if(Ngbuff > 0) {
      Ngbuff = 0;
      free(gbuff);
      gbuff = NULL;
    }

    //fprintf(stderr,"%04ld: before barrier, abind = %ld\n",ThisTask,abind); fflush(stderr); //FIXME
    //FIXME - extra barrier
    ///////////////////////////////////
    MPI_Barrier(MPI_COMM_WORLD);
    ///////////////////////////////////
        
  }//for(abind=0;abind<MaxNumActiveBundleCells;++abind)
  
  free(activeBundleCellInds);
  
  t0 += MPI_Wtime();
  if(ThisTask == 0) {
    fprintf(stderr,"did interp and integral to rays in %lf seconds.\n",currFTTsnap,t0);
    fflush(stderr);
  }
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

static void read_snaps(NbodySnap **snaps, long *Nsnaps) {
  char line[MAX_FILENAME];
  FILE *fp;
  long n = 0;
  char fname[MAX_FILENAME];
  long nl;
  
  if(ThisTask == 0) {
    fp = fopen(rayTraceData.ThreeDPotSnapList,"r");
    assert(fp != NULL);
    while(fgets(line,1024,fp) != NULL) {
      if(line[0] == '#')
	continue;
      ++n;
    }
    fclose(fp);
    
    *snaps = (NbodySnap*)malloc(sizeof(NbodySnap)*n);
    assert((*snaps) != NULL);
    *Nsnaps = n;
    
    n = 0;
    fp = fopen(rayTraceData.ThreeDPotSnapList,"r");
    assert(fp != NULL);
    while(fgets(line,1024,fp) != NULL) {
      if(line[0] == '#')
	continue;
      assert(n < (*Nsnaps));
      nl = strlen(line);
      line[nl-1] = '\0';
      sprintf((*snaps)[n].fname,"%s",line);
      ++n;
    }
    fclose(fp);
    
    for(n=0;n<(*Nsnaps);++n) {
      sprintf(fname,"%s.0",(*snaps)[n].fname);
      (*snaps)[n].a = get_scale_factor_LGADGET(fname);
      (*snaps)[n].chi = comvdist((*snaps)[n].a);
      //fprintf(stderr,"snap: '%s' chi(%lf) = %lf\n",(*snaps)[n].fname,(*snaps)[n].a,(*snaps)[n].chi); //FIXME
    }
  }//if(ThisTask == 0)
  
  //send to other tasks
  MPI_Bcast(Nsnaps,1,MPI_LONG,0,MPI_COMM_WORLD);
  if(ThisTask != 0) {
    *snaps = (NbodySnap*)malloc(sizeof(NbodySnap)*(*Nsnaps));
    assert((*snaps) != NULL);
  }
  MPI_Bcast(*snaps,sizeof(NbodySnap)*(*Nsnaps),MPI_BYTE,0,MPI_COMM_WORLD);
}

#undef THREEDIND
#undef WARPIF
