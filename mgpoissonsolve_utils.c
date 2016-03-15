#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <mpi.h>

#include "mgpoissonsolve.h"

//define to print out grids during fas call for debugging
//#define PRINT_FASGRIDS

/*timing variables
  0 - pre-smooth
  1 - down (tau correction)
  2 - exact solve at lowest level
  3 - solution correction
  4 - post-smooth
 */
#define NumRunTimesMGSteps 5
static double runTimesMGSteps[NumRunTimesMGSteps] = {0.0, 0.0, 0.0, 0.0, 0.0};

//convergence criterion
#ifdef MGPOISSONSOLVE_FLOAT
#define MGEPS_EXACT 1e-4 //level of L1norm for exact solve at smallest level
#define MGALPHA     1.0 //convergence param from NR in C - ratio of residual to truncation error
#else
#define MGEPS_EXACT 1e-12 //level of L1norm for exact solve at smallest level
#define MGALPHA     0.1 //convergence param from NR in C - ratio of residual to truncation error
#endif
#define MGEPS       1e-8 //level of L1norm for exact solve at smallest level
#define MGFRACEPS   1e-8 //frac error to terminate FAS cycles
#define MGMAXITR    100

//define to weight BCs differently - DO NOT USE!
//#define USE_BNDWGT
#define BNDWGT     0.00

//prototypes
static void smooth_mggrid_tempblock(MGGrid u, MGGrid rhs, long Nsmooth);

double solve_fas_mggrid(MGGridSet *grids, long Nlev, long NumPreSmooth, long NumPostSmooth, long NumOuterCycles, long NumInnerCycles, double convFact)
{
  //double FracL1norm,L1norm;
  long itr;
  double resid,trunc;
  double alpha = MGALPHA;
  //MGGrid uold = copy_mggrid(grids[Nlev-1].u);
  
  if(convFact > 0.0)
    alpha = convFact;
  
  itr = 0;
  do {
    //copy old grid to compute frac error
    //memcpy(uold->grid,grids[Nlev-1].u->grid,sizeof(double)*(grids[Nlev-1].u->N)*(grids[Nlev-1].u->N));
        
    //do poisson solve
    cycle_fas_mggrid(grids,Nlev-1,NumPreSmooth,NumPostSmooth,NumInnerCycles);
    
    //comp norm
    resid = L2norm_mggrid(grids[Nlev-1].u,grids[Nlev-1].rho);
    trunc = truncErr_mggrid(grids[Nlev-1].u,grids[Nlev-2].u,grids[Nlev-2].rho); //the lower level grids are just used as extra storage - BCs are not clobbered 
    //FracL1norm = fracErr_mggrid(grids[Nlev-1].u,uold);
    //L1norm = L1norm_mggrid(grids[Nlev-1].u,grids[Nlev-1].rho);
    
    ++itr;
    
    /*
    fprintf(stderr,"FracL1norm = %le\n",FracL1norm);
    if(itr == 10)
      exit(1);
    */
        
  } while(resid > trunc*alpha && itr < NumOuterCycles);
  //} while((FracL1norm > MGFRACEPS || L1norm > MGEPS) && itr < NumOuterCycles);
  
  //code to check convergence - not needed
  //fprintf(stderr,"converged: itr = %ld, L1Norm = %le, frac err = %le, resid = %le, trunc = %le, resid/trunc = %le (alpha = %le)\n",itr,L1norm,FracL1norm,resid,trunc,resid/trunc,MGALPHA);
  
  //free mem
  //free_mggrid(uold);
    
  //return FracL1norm;
  return resid/trunc;
}

#define MGCACHEOPT

void cycle_fas_mggrid(MGGridSet *grids, long lev, long NumPreSmooth, long NumPostSmooth, long NumInnerCycles)
{
  double L1norm;
  long itr;
#ifdef PRINT_FASGRIDS
  char fname[1024];
#endif
  
  if(lev == 0) //run smooth until convergence
    {
      //fprintf(stderr,"exact solve at level %ld\n",lev);
      runTimesMGSteps[2] -= MPI_Wtime();
      itr = 0;
      do 
	{
	  smooth_mggrid(grids[lev].u,grids[lev].rho,10l);
	  L1norm = L1norm_mggrid(grids[lev].u,grids[lev].rho);
	  ++itr;
	} while(L1norm > MGEPS_EXACT && itr < MGMAXITR);
      runTimesMGSteps[2] += MPI_Wtime();
      
#ifdef PRINT_FASGRIDS
      sprintf(fname,"./outputs/fasgrid_ulev%ld.dat",lev);
      write_mggrid(fname,grids[lev].u);
#endif
    }
  else //do multigrid steps
    {
      //going down
      //fprintf(stderr,"down at level %ld\n",lev);
      
      //pre-smooth
      runTimesMGSteps[0] -= MPI_Wtime();
#ifndef MGCACHEOPT
      smooth_mggrid(grids[lev].u,grids[lev].rho,NumPreSmooth);
#else      
      smooth_mggrid_tempblock(grids[lev].u,grids[lev].rho,NumPreSmooth);
#endif
      runTimesMGSteps[0] += MPI_Wtime();

      //get tau correction
      runTimesMGSteps[1] -= MPI_Wtime();
      resid_restrict_mggrid(grids[lev-1].rho,grids[lev].u,grids[lev].rho);
      runTimesMGSteps[1] += MPI_Wtime();
      
#ifdef PRINT_FASGRIDS      
      sprintf(fname,"./outputs/fasgrid_reslevm1%ld.dat",lev);
      //write_mggrid(fname,rulm1);
      write_mggrid(fname,grids[lev-1].rho);
#endif
      runTimesMGSteps[1] -= MPI_Wtime();
      restrict_mggrid(grids[lev-1].u,grids[lev].u);
      runTimesMGSteps[1] += MPI_Wtime();
      
#ifdef PRINT_FASGRIDS      
      sprintf(fname,"./outputs/fasgrid_ulevm1%ld.dat",lev);
      write_mggrid(fname,grids[lev-1].u);
#endif
      
      runTimesMGSteps[1] -= MPI_Wtime();
      lop_mggrid_plusequal(grids[lev-1].u,grids[lev-1].rho);
      runTimesMGSteps[1] += MPI_Wtime();
      
#ifdef PRINT_FASGRIDS
      sprintf(fname,"./outputs/fasgrid_resrho%ld.dat",lev);
      write_mggrid(fname,grids[lev-1].rho);
#endif
      
      //recursive call - do once for base grid, but NumInnerCycles for any other grid
      if(lev == 1)
	{
	  cycle_fas_mggrid(grids,lev-1,NumPreSmooth,NumPostSmooth,NumInnerCycles);
	}
      else
	{
	  for(itr=0;itr<NumInnerCycles;++itr)
	    cycle_fas_mggrid(grids,lev-1,NumPreSmooth,NumPostSmooth,NumInnerCycles);
	}
            
      //go up
      //fprintf(stderr,"up at level %ld\n",lev);
      
      //apply tau correction
      runTimesMGSteps[3] -= MPI_Wtime();
      restrict_mggrid_minusequal(grids[lev-1].u,grids[lev].u); 
      interp_mggrid_plusequal(grids[lev].u,grids[lev-1].u);  
      runTimesMGSteps[3] += MPI_Wtime();
      
      //post-smoothing
      runTimesMGSteps[4] -= MPI_Wtime();
#ifndef MGCACHEOPT
      smooth_mggrid(grids[lev].u,grids[lev].rho,NumPostSmooth);
#else      
      smooth_mggrid_tempblock(grids[lev].u,grids[lev].rho,NumPostSmooth);
#endif
      runTimesMGSteps[4] += MPI_Wtime();
    }
}

//define to use red-black ordering
#define REDBLACK

static void smooth_mggrid_tempblock(MGGrid u, MGGrid rhs, long Nsmooth)
{
  long i,j,itr,N,n;
  double diag,h2;
  double fac1;
  
  //fprintf(stderr,"using new GS rouitnes!\n");
  
  h2 = u->dL;
  h2 *= h2;
  N = u->N;

#ifdef REDBLACK
  double sf2nm2,sf2nm1,sf2n,sf2np1,sf2np2;
  double dm1,d;
  //fused loop red-black relaxation
  for(itr=0;itr<Nsmooth;++itr)
    {
      //do first row - red cells
      i = 1;
      d = u->diag[i]; //u->sinfacs[2*i] + u->sinfacs[2*i+2] + 2.0/(u->sinfacs[2*i+1]);
      sf2n = u->sinfacs[2*i];
      sf2np1 = u->sinfacs[2*i+1];
      sf2np2 = u->sinfacs[2*i+2];
      fac1 = h2*sf2np1;
      for(j=1;j<N-1;j+=2)
	u->grid[i*N+j] = (sf2n*(u->grid[(i-1)*N+j]) //top
			  + sf2np2*(u->grid[(i+1)*N+j])  //bottom
			  + (u->grid[i*N + j-1])/sf2np1  //left
			  + (u->grid[i*N + j+1])/sf2np1 //right
			  - (fac1)*(rhs->grid[i*N + j]))/d;
      
      //now do rest of rows 
      //do i row red cells, then go back and do i-1 row black cells
      for(i=2;i<N-1;++i)
	{
	  dm1 = d;
	  sf2nm2 = sf2n;
	  sf2nm1 = sf2np1;
	  sf2n = sf2np2;
	  
	  d = u->diag[i];
	  sf2n = u->sinfacs[2*i];
	  sf2np1 = u->sinfacs[2*i+1];
	  sf2np2 = u->sinfacs[2*i+2];
	  
	  //row i red cells
	  fac1 = h2*sf2np1;
	  for(j=1;j<N-1;j+=2)
	    u->grid[i*N+j] = (sf2n*(u->grid[(i-1)*N+j]) //top
			      + sf2np2*(u->grid[(i+1)*N+j])  //bottom
			      + (u->grid[i*N + j-1])/sf2np1  //left
			      + (u->grid[i*N + j+1])/sf2np1 //right
			      - (fac1)*(rhs->grid[i*N + j]))/d;

	  //row i-1 black cells
	  fac1 = h2*sf2nm1; //(u->sinfacs[2*(n-1)+1]);
	  for(j=2;j<N-1;j+=2)
	    u->grid[(i-1)*N+j] = (sf2nm2*(u->grid[((i-1)-1)*N+j]) //top
				  + sf2n*(u->grid[((i-1)+1)*N+j])  //bottom
				  + (u->grid[(i-1)*N + j-1])/sf2nm1  //left
				  + (u->grid[(i-1)*N + j+1])/sf2nm1 //right
				  - (fac1)*(rhs->grid[(i-1)*N + j]))/dm1;
	}
      
      //do last row black cells
      i = N-2;
      fac1 = h2*sf2np1;
      for(j=2;j<N-1;j+=2)
	u->grid[i*N+j] = (sf2n*(u->grid[(i-1)*N+j]) //top
			  + sf2np2*(u->grid[(i+1)*N+j])  //bottom
			  + (u->grid[i*N + j-1])/sf2np1  //left
			  + (u->grid[i*N + j+1])/sf2np1 //right
			  - (fac1)*(rhs->grid[i*N + j]))/d;
    }
#else
  //natural ordering moving window/temporal blocking scheme
  
  //init window
  //fprintf(stderr,"Nsmooth = %ld\n",Nsmooth);
  for(n=Nsmooth;n>=1;--n)
    for(i=1;i<=n;++i)
      {
       	//fprintf(stderr,"i - doing row %ld\n",i);
	
	diag = u->diag[i]; //u->sinfacs[2*i] + u->sinfacs[2*i+2] + 2.0/(u->sinfacs[2*i+1]);
	fac1 = h2*(u->sinfacs[2*i+1]);
	for(j=1;j<N-1;++j)
	  u->grid[i*N+j] = ((u->sinfacs[2*i])*(u->grid[(i-1)*N+j]) //top
			    + (u->sinfacs[2*i+2])*(u->grid[(i+1)*N+j])  //bottom
			    + (u->grid[i*N + j-1])/(u->sinfacs[2*i+1])  //left
			    + (u->grid[i*N + j+1])/(u->sinfacs[2*i+1]) //right
			    - (fac1)*(rhs->grid[i*N + j]))/diag;
      }
  
  //now do rest of rows 
  for(n=2;n<N-1-Nsmooth;++n)
    for(i=n+Nsmooth-1;i>=n;--i)
      {
	//fprintf(stderr,"m - doing row %ld\n",i);
	
	diag = u->diag[i]; //u->sinfacs[2*i] + u->sinfacs[2*i+2] + 2.0/(u->sinfacs[2*i+1]);
	fac1 = h2*(u->sinfacs[2*i+1]);
	for(j=1;j<N-1;++j)
	  u->grid[i*N+j] = ((u->sinfacs[2*i])*(u->grid[(i-1)*N+j]) //top
			    + (u->sinfacs[2*i+2])*(u->grid[(i+1)*N+j])  //bottom
			    + (u->grid[i*N + j-1])/(u->sinfacs[2*i+1])  //left
			    + (u->grid[i*N + j+1])/(u->sinfacs[2*i+1]) //right
			    - (fac1)*(rhs->grid[i*N + j]))/diag;
      } 
  
  //do last rows
  for(n=N-1-Nsmooth;n<N-1;++n)
    for(i=N-2;i>=n;--i)
      {
	//fprintf(stderr,"e - doing row %ld\n",i);
	
	diag = u->diag[i]; //u->sinfacs[2*i] + u->sinfacs[2*i+2] + 2.0/(u->sinfacs[2*i+1]);
	fac1 = h2*(u->sinfacs[2*i+1]);
	for(j=1;j<N-1;++j)
	  u->grid[i*N+j] = ((u->sinfacs[2*i])*(u->grid[(i-1)*N+j]) //top
			    + (u->sinfacs[2*i+2])*(u->grid[(i+1)*N+j])  //bottom
			    + (u->grid[i*N + j-1])/(u->sinfacs[2*i+1])  //left
			    + (u->grid[i*N + j+1])/(u->sinfacs[2*i+1]) //right
			    - (fac1)*(rhs->grid[i*N + j]))/diag;
      } 
  
  //exit(1);
#endif
}

#ifdef REDBLACK
void smooth_mggrid(MGGrid u, MGGrid rhs, long Nsmooth)
{
  long i,j,itr,N;
  double diag,nbrs,h2;
  double fac1;
  
  h2 = u->dL;
  h2 *= h2;
  N = u->N;
  
  for(itr=0;itr<Nsmooth;++itr)
    {
      for(i=1;i<N-1;++i)
        {
	  diag = u->diag[i]; //u->sinfacs[2*i] + u->sinfacs[2*i+2] + 2.0/(u->sinfacs[2*i+1]);
	  fac1 = h2*(u->sinfacs[2*i+1]);
	  
	  for(j=1-(i%2)+1;j<N-1;j+=2)
	    {
#ifdef USE_BNDWGT
	      //top
	      if(i == 1)
		nbrs = (u->sinfacs[2*i])*((u->grid[(i-1)*N+j])*(1.0 - BNDWGT) + (2.0*(u->grid[i*N+j]) - u->grid[(i+1)*N+j])*BNDWGT);
	      else
		nbrs = (u->sinfacs[2*i])*(u->grid[(i-1)*N+j]);
	      
	      //bottom
	      if(i == N-2)
		nbrs += (u->sinfacs[2*i+2])*((u->grid[(i+1)*N+j])*(1.0 - BNDWGT) + (2.0*(u->grid[i*N+j]) - u->grid[(i-1)*N+j])*BNDWGT);
	      else
		nbrs += (u->sinfacs[2*i+2])*(u->grid[(i+1)*N+j]);
	      
	      //left
	      if(j == 1)
		nbrs += ((u->grid[i*N + j-1])*(1.0 - BNDWGT) + (2.0*(u->grid[i*N+j]) - u->grid[i*N+j+1])*BNDWGT)/(u->sinfacs[2*i+1]);
	      else
		nbrs += (u->grid[i*N + j-1])/(u->sinfacs[2*i+1]);
	      
	      //right
	      if(j == N-1)
		nbrs += ((u->grid[i*N + j+1])*(1.0 - BNDWGT) + (2.0*(u->grid[i*N+j]) - u->grid[i*N+j-1])*BNDWGT)/(u->sinfacs[2*i+1]);
	      else
		nbrs += (u->grid[i*N + j+1])/(u->sinfacs[2*i+1]);
#else
	      //top
	      nbrs = (u->sinfacs[2*i])*(u->grid[(i-1)*N+j]);
	      	    
	      //bottom
	      nbrs += (u->sinfacs[2*i+2])*(u->grid[(i+1)*N+j]);
	      	    
	      //left
	      nbrs += (u->grid[i*N + j-1])/(u->sinfacs[2*i+1]);
	      
	      //right
	      nbrs += (u->grid[i*N + j+1])/(u->sinfacs[2*i+1]);
#endif
	      u->grid[i*N+j] = (nbrs - (fac1)*(rhs->grid[i*N + j]))/diag;
	    }
	}
      
      for(i=1;i<N-1;++i)
        {
	  diag = u->diag[i]; //u->sinfacs[2*i] + u->sinfacs[2*i+2] + 2.0/(u->sinfacs[2*i+1]);
	  fac1 = h2*(u->sinfacs[2*i+1]);
	  
	  for(j=(i%2)+1;j<N-1;j+=2)
	    {
#ifdef USE_BNDWGT
	      //top
	      if(i == 1)
		nbrs = (u->sinfacs[2*i])*((u->grid[(i-1)*N+j])*(1.0 - BNDWGT) + (2.0*(u->grid[i*N+j]) - u->grid[(i+1)*N+j])*BNDWGT);
	      else
		nbrs = (u->sinfacs[2*i])*(u->grid[(i-1)*N+j]);
	      	    
	      //bottom
	      if(i == N-2)
		nbrs += (u->sinfacs[2*i+2])*((u->grid[(i+1)*N+j])*(1.0 - BNDWGT) + (2.0*(u->grid[i*N+j]) - u->grid[(i-1)*N+j])*BNDWGT);
	      else
		nbrs += (u->sinfacs[2*i+2])*(u->grid[(i+1)*N+j]);
	      	    
	      //left
	      if(j == 1)
		nbrs += ((u->grid[i*N + j-1])*(1.0 - BNDWGT) + (2.0*(u->grid[i*N+j]) - u->grid[i*N+j+1])*BNDWGT)/(u->sinfacs[2*i+1]);
	      else
		nbrs += (u->grid[i*N + j-1])/(u->sinfacs[2*i+1]);
	      
	      //right
	      if(j == N-1)
		nbrs += ((u->grid[i*N + j+1])*(1.0 - BNDWGT) + (2.0*(u->grid[i*N+j]) - u->grid[i*N+j-1])*BNDWGT)/(u->sinfacs[2*i+1]);
	      else
		nbrs += (u->grid[i*N + j+1])/(u->sinfacs[2*i+1]);
#else   
	      //top
	      nbrs = (u->sinfacs[2*i])*(u->grid[(i-1)*N+j]);
	      	    
	      //bottom
	      nbrs += (u->sinfacs[2*i+2])*(u->grid[(i+1)*N+j]);
	      	    
	      //left
	      nbrs += (u->grid[i*N + j-1])/(u->sinfacs[2*i+1]);
	      
	      //right
	      nbrs += (u->grid[i*N + j+1])/(u->sinfacs[2*i+1]);
#endif	    
	      u->grid[i*N+j] = (nbrs - (fac1)*(rhs->grid[i*N + j]))/diag;
	    }
	}
    }
}
#else
void smooth_mggrid(MGGrid u, MGGrid rhs, long Nsmooth)
{
  long i,j,s,itr,N,tind,bind;
  double diag,nbrs,h2;
  double vol,sinef,sinefup,sinefdown;
  gsl_vector *d,*b,*x,*fs,*es;
    
  h2 = u->dL;
  h2 *= h2;
  N = u->N;
    
  //alloc mem - actual size of grid is N-2
  d = gsl_vector_alloc((size_t) (N-2));
  b = gsl_vector_alloc((size_t) (N-2));
  x = gsl_vector_alloc((size_t) (N-2));
  fs = gsl_vector_alloc((size_t) (N-3));
  es = gsl_vector_alloc((size_t) (N-3));
  
  for(itr=0;itr<Nsmooth;++itr)
    {
      for(s=1;s<=2;++s)
	{
	  for(i=s;i<N-1;i+=2)
	    {
	      tind = i-1;
	      bind = i+1;
	      sinef = u->sinfacs[2*i+1];
	      sinefdown = u->sinfacs[2*i+2];
	      sinefup = u->sinfacs[2*i];
	      vol = h2*sinef;
	      diag = sinefup;
	      diag += sinefdown;
	      diag += 2.0/sinef;
	      
	      for(j=1;j<N-1;++j)
		{
		  gsl_vector_set(d,(size_t) (j-1),-diag);
		  
		  nbrs = 0.0;
		  
		  //top face
		  nbrs += sinefup*(u->grid[tind*N + j]);
		  
		  //bottom face
		  nbrs += sinefdown*(u->grid[bind*N + j]);
		  
		  //left face
		  if(j == 1)
		    nbrs += (u->grid[i*N + j-1])/sinef;
		  else
		    gsl_vector_set(fs,(size_t) (j-2),1.0/sinef);
		  
		  //right face
		  if(j == N-2)
		    nbrs += (u->grid[i*N + j+1])/sinef;
		  else
		    gsl_vector_set(es,(size_t) (j-1),1.0/sinef);
		  
		  gsl_vector_set(b,(size_t) (j-1),vol*(rhs->grid[i*N + j])-nbrs);
		}
	      
	      gsl_linalg_solve_tridiag(d,es,fs,b,x);
	      
	      for(j=1;j<N-1;++j)
		u->grid[i*N + j] = gsl_vector_get(x,(size_t) (j-1));
	    }
	  
	  for(j=s;j<N-1;j+=2)
	    {
	      for(i=1;i<N-1;++i)
		{
		  tind = i-1;
		  bind = i+1;
		  sinef = u->sinfacs[2*i+1];
		  sinefdown = u->sinfacs[2*i+2];
		  sinefup = u->sinfacs[2*i];
		  vol = h2*sinef;
		  diag = sinefup;
		  diag += sinefdown;
		  diag += 2.0/sinef;
		  
		  gsl_vector_set(d,(size_t) (i-1),-diag);
		  
		  nbrs = 0.0;
		  
		  //bottom face
		  if(i == N-2)
		    nbrs += sinefdown*(u->grid[bind*N + j]);
		  else
		    gsl_vector_set(es,(size_t) (i-1),sinefdown);
		  
		  //top face
		  if(i == 1)
		    nbrs += sinefup*(u->grid[tind*N + j]);
		  else
		    gsl_vector_set(fs,(size_t) (i-2),sinefup);
		  
		  //left face
		  nbrs += (u->grid[i*N + j-1])/sinef;
		  
		  //right face
		  nbrs += (u->grid[i*N + j+1])/sinef;
		  
		  gsl_vector_set(b,(size_t) (i-1),vol*(rhs->grid[i*N + j])-nbrs);
		}
	      
	      gsl_linalg_solve_tridiag(d,es,fs,b,x);
	      
	      for(i=1;i<N-1;++i)
		u->grid[i*N + j] = gsl_vector_get(x,(size_t) (i-1));
	    }
	}
    }
  
  gsl_vector_free(d);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_vector_free(fs);
  gsl_vector_free(es);
}
#endif

MGGrid lop_mggrid(MGGrid g)
{
  MGGrid lopg;
  long i,j,tind,bind,lind,rind,N;
  double sinef,sinefdown,sinefup,vol;
  double h2;
  mgfloat *u,*lop;
      
  lopg = copy_mggrid(g);
  zero_mggrid(lopg);
  
  h2 = g->dL;
  h2 *= h2;
  u = g->grid;
  lop = lopg->grid;
  N = g->N;
  
  for(i=1;i<N-1;++i)
    {
      tind = i-1;
      bind = i+1;
      sinef = g->sinfacs[2*i+1];
      sinefdown = g->sinfacs[2*i+2];
      sinefup = g->sinfacs[2*i];
      vol = h2*sinef;
      
      for(j=1;j<N-1;++j)
        {
          lind = j-1;
	  rind = j+1;
          
	  //Laplacian terms
#ifdef USE_BNDWGT
	  if(i == 1)
	    lop[i*N+j] += (sinefup*((u[tind*N + j])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //top face
	  else
	    lop[i*N+j] += (sinefup*(u[tind*N + j] - u[i*N + j])); //top face
	  
	  if(i == N-2)
	    lop[i*N+j] += (sinefdown*((u[bind*N + j])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //bottom face
	  else
	    lop[i*N+j] += (sinefdown*(u[bind*N + j] - u[i*N + j])); //bottom face
	  
	  if(j == 1)
	    lop[i*N+j] += (1.0/sinef*((u[i*N + lind])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //left face      
	  else
	    lop[i*N+j] += (1.0/sinef*(u[i*N + lind] - u[i*N + j])); //left face      
	  
	  if(j == N-2)
	    lop[i*N+j] += (1.0/sinef*((u[i*N + rind])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //right face
	  else
	    lop[i*N+j] += (1.0/sinef*(u[i*N + rind] - u[i*N + j])); //right face
#else	  
	  lop[i*N+j] += (sinefup*(u[tind*N + j] - u[i*N + j])); //top face
	  lop[i*N+j] += (sinefdown*(u[bind*N + j] - u[i*N + j])); //bottom face
	  lop[i*N+j] += (1.0/sinef*(u[i*N + lind] - u[i*N + j])); //left face      
	  lop[i*N+j] += (1.0/sinef*(u[i*N + rind] - u[i*N + j])); //right face
#endif
	  
	  lop[i*N+j] /= vol;
	}
    }
  
  return lopg;
}

void lop_mggrid_plusequal(MGGrid g, MGGrid pg)
{
  long i,j,tind,bind,lind,rind,N;
  double sinef,sinefdown,sinefup,vol;
  double h2;
  mgfloat *u;
  double lop;
  
  h2 = g->dL;
  h2 *= h2;
  u = g->grid;
  N = g->N;
  
  for(i=1;i<N-1;++i)
    {
      tind = i-1;
      bind = i+1;
      sinef = g->sinfacs[2*i+1];
      sinefdown = g->sinfacs[2*i+2];
      sinefup = g->sinfacs[2*i];
      vol = h2*sinef;
      
      for(j=1;j<N-1;++j)
        {
          lind = j-1;
	  rind = j+1;
	  
	  lop = 0.0;
	  
	  //Laplacian terms
#ifdef USE_BNDWGT
	  if(i == 1)
	    lop += (sinefup*((u[tind*N + j])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //top face
	  else
	    lop += (sinefup*(u[tind*N + j] - u[i*N + j])); //top face
	  
	  if(i == N-2)
	    lop += (sinefdown*((u[bind*N + j])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //bottom face
	  else
	    lop += (sinefdown*(u[bind*N + j] - u[i*N + j])); //bottom face
	  
	  if(j == 1)
	    lop += (1.0/sinef*((u[i*N + lind])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //left face      
	  else
	    lop += (1.0/sinef*(u[i*N + lind] - u[i*N + j])); //left face      
	  
	  if(j == N-2)
	    lop += (1.0/sinef*((u[i*N + rind])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //right face
	  else
	    lop += (1.0/sinef*(u[i*N + rind] - u[i*N + j])); //right face
#else	  
	  lop += (sinefup*(u[tind*N + j] - u[i*N + j])); //top face
	  lop += (sinefdown*(u[bind*N + j] - u[i*N + j])); //bottom face
	  lop += (1.0/sinef*(u[i*N + lind] - u[i*N + j])); //left face      
	  lop += (1.0/sinef*(u[i*N + rind] - u[i*N + j])); //right face
#endif
	  
	  lop /= vol;
	  
	  pg->grid[i*N+j] += lop;
	}
    }
}

MGGrid resid_mggrid(MGGrid u, MGGrid rhs)
{
  MGGrid resid;
  long i,j;
  
  resid = lop_mggrid(u);
  
  for(i=1;i<u->N-1;++i)
    for(j=1;j<u->N-1;++j)
      resid->grid[i*u->N+j] = rhs->grid[i*u->N+j] - resid->grid[i*u->N+j];
  
  return resid;
}

double L1norm_mggrid(MGGrid u, MGGrid rhs)
{
  double norm = 0.0;
  long i,j,tind,bind,lind,rind,N;
  double sinef,sinefdown,sinefup,vol;
  double h2;
  double lop;
  
  h2 = u->dL;
  h2 *= h2;
  N = u->N;
  
  for(i=1;i<N-1;++i)
    {
      tind = i-1;
      bind = i+1;
      sinef = u->sinfacs[2*i+1];
      sinefdown = u->sinfacs[2*i+2];
      sinefup = u->sinfacs[2*i];
      vol = h2*sinef;
      
      for(j=1;j<N-1;++j)
	{
	  lind = j-1;
	  rind = j+1;
          
	  lop = 0.0;
          
	  //Laplacian terms
#ifdef USE_BNDWGT
	  if(i == 1)
	    lop += (sinefup*((u->grid[tind*N + j])*(1.0 - BNDWGT) + (u->grid[i*N+j])*BNDWGT - u->grid[i*N + j])); //top face
	  else
	    lop += (sinefup*(u->grid[tind*N + j] - u->grid[i*N + j])); //top face
          
	  if(i == N-2)
	    lop += (sinefdown*((u->grid[bind*N + j])*(1.0 - BNDWGT) + (u->grid[i*N+j])*BNDWGT - u->grid[i*N + j])); //bottom face
	  else
	    lop += (sinefdown*(u->grid[bind*N + j] - u->grid[i*N + j])); //bottom face
          
	  if(j == 1)
	    lop += (1.0/sinef*((u->grid[i*N + lind])*(1.0 - BNDWGT) + (u->grid[i*N+j])*BNDWGT - u->grid[i*N + j])); //left face      
	  else
	    lop += (1.0/sinef*(u->grid[i*N + lind] - u->grid[i*N + j])); //left face      
          
	  if(j == N-2)
	    lop += (1.0/sinef*((u->grid[i*N + rind])*(1.0 - BNDWGT) + (u->grid[i*N+j])*BNDWGT - u->grid[i*N + j])); //right face
	  else
	    lop += (1.0/sinef*(u->grid[i*N + rind] - u->grid[i*N + j])); //right face
#else     
	  lop += (sinefup*(u->grid[tind*N + j] - u->grid[i*N + j])); //top face
	  lop += (sinefdown*(u->grid[bind*N + j] - u->grid[i*N + j])); //bottom face
	  lop += (1.0/sinef*(u->grid[i*N + lind] - u->grid[i*N + j])); //left face      
	  lop += (1.0/sinef*(u->grid[i*N + rind] - u->grid[i*N + j])); //right face
#endif
	  
	  lop /= vol;
          
	  norm += (rhs->grid[i*u->N+j] - lop);
	}
    }
  
  norm /= u->N;
  norm /= u->N;
  
  return norm;
}

double L2norm_mggrid(MGGrid u, MGGrid rhs)
{
  double norm = 0.0;
  long i,j,tind,bind,lind,rind,N;
  double sinef,sinefdown,sinefup,vol;
  double h2;
  double lop;
  
  h2 = u->dL;
  h2 *= h2;
  N = u->N;
  
  for(i=1;i<N-1;++i)
    {
      tind = i-1;
      bind = i+1;
      sinef = u->sinfacs[2*i+1];
      sinefdown = u->sinfacs[2*i+2];
      sinefup = u->sinfacs[2*i];
      vol = h2*sinef;
      
      for(j=1;j<N-1;++j)
	{
	  lind = j-1;
	  rind = j+1;
          
	  lop = 0.0;
          
	  //Laplacian terms
#ifdef USE_BNDWGT
	  if(i == 1)
	    lop += (sinefup*((u->grid[tind*N + j])*(1.0 - BNDWGT) + (u->grid[i*N+j])*BNDWGT - u->grid[i*N + j])); //top face
	  else
	    lop += (sinefup*(u->grid[tind*N + j] - u->grid[i*N + j])); //top face
          
	  if(i == N-2)
	    lop += (sinefdown*((u->grid[bind*N + j])*(1.0 - BNDWGT) + (u->grid[i*N+j])*BNDWGT - u->grid[i*N + j])); //bottom face
	  else
	    lop += (sinefdown*(u->grid[bind*N + j] - u->grid[i*N + j])); //bottom face
          
	  if(j == 1)
	    lop += (1.0/sinef*((u->grid[i*N + lind])*(1.0 - BNDWGT) + (u->grid[i*N+j])*BNDWGT - u->grid[i*N + j])); //left face      
	  else
	    lop += (1.0/sinef*(u->grid[i*N + lind] - u->grid[i*N + j])); //left face      
          
	  if(j == N-2)
	    lop += (1.0/sinef*((u->grid[i*N + rind])*(1.0 - BNDWGT) + (u->grid[i*N+j])*BNDWGT - u->grid[i*N + j])); //right face
	  else
	    lop += (1.0/sinef*(u->grid[i*N + rind] - u->grid[i*N + j])); //right face
#else     
	  lop += (sinefup*(u->grid[tind*N + j] - u->grid[i*N + j])); //top face
	  lop += (sinefdown*(u->grid[bind*N + j] - u->grid[i*N + j])); //bottom face
	  lop += (1.0/sinef*(u->grid[i*N + lind] - u->grid[i*N + j])); //left face      
	  lop += (1.0/sinef*(u->grid[i*N + rind] - u->grid[i*N + j])); //right face
#endif
	  
	  lop /= vol;
          
	  norm += (rhs->grid[i*u->N+j] - lop)*(rhs->grid[i*u->N+j] - lop);
	}
    }
  
  norm /= u->N;
  norm /= u->N;
  norm = sqrt(norm);
  
  return norm;
}

double fracErr_mggrid(MGGrid u, MGGrid uold)
{
  double norm = 0.0;
  long i,j;
    
  for(i=1;i<u->N-1;++i)
    for(j=1;j<u->N-1;++j)
      norm += fabs(u->grid[i*u->N+j]/uold->grid[i*u->N+j] - 1.0);
  
  norm /= u->N;
  norm /= u->N;
  
  return norm;
}

#define INTERPGRID
void interp_mggrid(MGGrid uf, MGGrid uc)
{
  long i,j;
  long ifn,jfn;
  
#ifdef INTERPGRID
  double val,wx,wy;
  long xind,yind,xindp,yindp;
  long xmod,ymod;
  
  //do boundaries first, revert to diect injection for stability
  i = 1;
  ifn = (i-1)/2 + 1;
  for(j=1;j<uf->N-1;++j)
    uf->grid[i*(uf->N) + j] = uc->grid[ifn*(uc->N)+((j-1)/2 + 1)];
  
  i = uf->N-2;
  ifn = (i-1)/2 + 1;
  for(j=1;j<uf->N-1;++j)
    uf->grid[i*(uf->N) + j] = uc->grid[ifn*(uc->N)+((j-1)/2 + 1)];
  
  j = 1;
  jfn = (j-1)/2 + 1;
  for(i=2;i<uf->N-2;++i)
    uf->grid[i*(uf->N) + j] = uc->grid[((i-1)/2 + 1)*(uc->N)+jfn];
  
  j = uf->N-2;
  jfn = (j-1)/2 + 1;
  for(i=2;i<uf->N-2;++i)
    uf->grid[i*(uf->N) + j] = uc->grid[((i-1)/2 + 1)*(uc->N)+jfn];
  
  //do linear interp in middle
  xmod = 1;
  wx = 0.75;
  for(ifn=2;ifn<uf->N-2;++ifn)
    {
      xind = (ifn-1)/2 + xmod;
      xindp = xind + 1;
      wx = 1.0 - wx;
      xmod = 1 - xmod;
      
      ymod = 1;
      wy = 0.75;
      for(jfn=2;jfn<uf->N-2;++jfn)
        {
          yind = (jfn-1)/2 + ymod;
          yindp = yind + 1;
          wy = 1.0 - wy;
          ymod = 1 - ymod;
                  
          val = (uc->grid[xind*(uc->N)+yind])*(1.0 - wx)*(1.0 - wy);
          val += (uc->grid[xind*(uc->N)+yindp])*(1.0 - wx)*wy;
          val += (uc->grid[xindp*(uc->N)+yind])*wx*(1.0 - wy);
          val += (uc->grid[xindp*(uc->N)+yindp])*wx*wy;
          
          uf->grid[ifn*(uf->N)+jfn] = val;
        }
    }
  
  /* OLD CODE - NOT USED
  double val,wx,wy,tf,pf;
  long xind,yind,xindp,yindp;

  //interp
  for(i=1;i<uf->N-1;++i)
    for(j=1;j<uf->N-1;++j)
      {
	tf = i*(uf->dL) + uf->thetaLoc;
	pf = j*(uf->dL) + uf->phiLoc;
	
	xind = (tf - uc->thetaLoc)/(uc->dL);
	yind = (pf - uc->phiLoc)/(uc->dL);
	xindp = xind + 1;
	yindp = yind + 1;
	
	//if at edge, revert to diect injection for stability
	if(xind == 0 || xindp == uc->N-1 || yind == 0 || yindp == uc->N-1)
	  {
	    ifn = (i-1)/2 + 1;
	    jfn = (j-1)/2 + 1;
	    uf->grid[i*(uf->N) + j] = uc->grid[ifn*(uc->N)+jfn];
	  }
	else
	  {
	    wx = (tf - (xind*(uc->dL) + uc->thetaLoc))/(uc->dL);
	    wy = (pf - (yind*(uc->dL) + uc->phiLoc))/(uc->dL);
	    
	    
	    val = (uc->grid[xind*(uc->N)+yind])*(1.0 - wx)*(1.0 - wy);
	    val += (uc->grid[xind*(uc->N)+yindp])*(1.0 - wx)*wy;
	    val += (uc->grid[xindp*(uc->N)+yind])*wx*(1.0 - wy);
	    val += (uc->grid[xindp*(uc->N)+yindp])*wx*wy;
	    
	    uf->grid[i*(uf->N)+j] = val;
	  }
      }
  */
#else
  long m,n;
  
  //direct injection
  for(i=1;i<uc->N-1;++i)
    for(j=1;j<uc->N-1;++j)
      {
	for(n=0;n<2;++n)
	  for(m=0;m<2;++m)
	    {
	      ifn = 2*(i-1) + n + 1;
	      jfn = 2*(j-1) + m + 1;
	      uf->grid[ifn*(uf->N) + jfn] = uc->grid[i*(uc->N)+j];
	    }
      }
#endif
}

void interp_mggrid_plusequal(MGGrid uf, MGGrid uc)
{
  long i,j;
  long ifn,jfn;
  
#ifdef INTERPGRID
  double val,wx,wy;
  long xind,yind,xindp,yindp;
  long xmod,ymod;
  
  //do boundaries first, revert to diect injection for stability
  i = 1;
  ifn = (i-1)/2 + 1;
  for(j=1;j<uf->N-1;++j)
    uf->grid[i*(uf->N) + j] += uc->grid[ifn*(uc->N)+((j-1)/2 + 1)];
  
  i = uf->N-2;
  ifn = (i-1)/2 + 1;
  for(j=1;j<uf->N-1;++j)
    uf->grid[i*(uf->N) + j] += uc->grid[ifn*(uc->N)+((j-1)/2 + 1)];
  
  j = 1;
  jfn = (j-1)/2 + 1;
  for(i=2;i<uf->N-2;++i)
    uf->grid[i*(uf->N) + j] += uc->grid[((i-1)/2 + 1)*(uc->N)+jfn];
  
  j = uf->N-2;
  jfn = (j-1)/2 + 1;
  for(i=2;i<uf->N-2;++i)
    uf->grid[i*(uf->N) + j] += uc->grid[((i-1)/2 + 1)*(uc->N)+jfn];
  
  //do linear interp in middle
  xmod = 1;
  wx = 0.75;
  for(ifn=2;ifn<uf->N-2;++ifn)
    {
      xind = (ifn-1)/2 + xmod;
      xindp = xind + 1;
      wx = 1.0 - wx;
      xmod = 1 - xmod;
      
      ymod = 1;
      wy = 0.75;
      for(jfn=2;jfn<uf->N-2;++jfn)
	{
	  yind = (jfn-1)/2 + ymod;
	  yindp = yind + 1;
	  wy = 1.0 - wy;
	  ymod = 1 - ymod;
	  	  
	  val = (uc->grid[xind*(uc->N)+yind])*(1.0 - wx)*(1.0 - wy);
	  val += (uc->grid[xind*(uc->N)+yindp])*(1.0 - wx)*wy;
	  val += (uc->grid[xindp*(uc->N)+yind])*wx*(1.0 - wy);
	  val += (uc->grid[xindp*(uc->N)+yindp])*wx*wy;
	  
	  uf->grid[ifn*(uf->N)+jfn] += val;
	}
    }
  
  /* OLD CODE - NOT USED
  double val,wx,wy,tf,pf;
  long xind,yind,xindp,yindp;
  
  //interp
  for(i=1;i<uf->N-1;++i)
    for(j=1;j<uf->N-1;++j)
      {
	tf = i*(uf->dL) + uf->thetaLoc;
	pf = j*(uf->dL) + uf->phiLoc;
	
	xind = (tf - uc->thetaLoc)/(uc->dL);
	yind = (pf - uc->phiLoc)/(uc->dL);
	xindp = xind + 1;
	yindp = yind + 1;
	
	//if at edge, revert to diect injection for stability
	if(xind == 0 || xindp == uc->N-1 || yind == 0 || yindp == uc->N-1)
	  {
	    ifn = (i-1)/2 + 1;
	    jfn = (j-1)/2 + 1;
	    uf->grid[i*(uf->N) + j] += uc->grid[ifn*(uc->N)+jfn];
	  }
	else
	  {
	    wx = (tf - (xind*(uc->dL) + uc->thetaLoc))/(uc->dL);
	    wy = (pf - (yind*(uc->dL) + uc->phiLoc))/(uc->dL);
	    
	    val = (uc->grid[xind*(uc->N)+yind])*(1.0 - wx)*(1.0 - wy);
	    val += (uc->grid[xind*(uc->N)+yindp])*(1.0 - wx)*wy;
	    val += (uc->grid[xindp*(uc->N)+yind])*wx*(1.0 - wy);
	    val += (uc->grid[xindp*(uc->N)+yindp])*wx*wy;
	    
	    uf->grid[i*(uf->N)+j] += val;
	  }
      }
  */
#else
  long m,n;
  
  //direct injection
  for(i=1;i<uc->N-1;++i)
    for(j=1;j<uc->N-1;++j)
      {
	for(n=0;n<2;++n)
	  for(m=0;m<2;++m)
	    {
	      ifn = 2*(i-1) + n + 1;
	      jfn = 2*(j-1) + m + 1;
	      uf->grid[ifn*(uf->N) + jfn] += uc->grid[i*(uc->N)+j];
	    }
      }
#endif
}

void restrict_mggrid(MGGrid uc, MGGrid uf)
{
  long i,j,n,m,ifn,jfn;
  double mass,volc;
    
  //take a simple average
  for(i=1;i<uc->N-1;++i)
    {
      volc = 2.0*(uc->cosfacs[i]);
      
      for(j=1;j<uc->N-1;++j)
	{
	  mass = 0.0;
	  for(n=0;n<2;++n)
	    for(m=0;m<2;++m)
	      {
		ifn = 2*(i-1) + n + 1;
		jfn = 2*(j-1) + m + 1;
		mass += (uf->grid[ifn*(uf->N) + jfn])*(uf->cosfacs[ifn]);
	      }
	  
	  uc->grid[i*(uc->N)+j] = mass/volc;
	}
    }
}

void restrict_mggrid_minusequal(MGGrid uc, MGGrid uf)
{
  long i,j,n,m,ifn,jfn;
  double mass,volc;
  //double dL = (uf->dL);
  
  //take a simple average
  for(i=1;i<uc->N-1;++i)
    {
      volc = 2.0*(uc->cosfacs[i]);
      
      for(j=1;j<uc->N-1;++j)
	{
	  mass = 0.0;
	  for(n=0;n<2;++n)
	    for(m=0;m<2;++m)
	      {
		ifn = 2*(i-1) + n + 1;
		jfn = 2*(j-1) + m + 1;
		mass += (uf->grid[ifn*(uf->N) + jfn])*(uf->cosfacs[ifn]);
	      }
	  
	  uc->grid[i*(uc->N)+j] -= mass/volc;
	}
    }
}

void resid_restrict_mggrid(MGGrid uc, MGGrid uf, MGGrid rhof)
{
  long i,j,n,m,ifn,jfn;
  double mass;
  long tind,bind,lind,rind,N;
  double sinef,sinefdown,sinefup,vol;
  double h2;
  mgfloat *u;
  double lop,resid;
  double volc;
  
  h2 = uf->dL;
  h2 *= h2;
  u = uf->grid;
  N = uf->N;
  
  //take a simple average
  for(i=1;i<uc->N-1;++i)
    {
      volc = 2.0*(uc->cosfacs[i]);
      
      for(j=1;j<uc->N-1;++j)
	{
	  mass = 0.0;
	  for(n=0;n<2;++n)
	    {
	      ifn = 2*(i-1) + n + 1;
	      tind = ifn-1;
	      bind = ifn+1;
	      sinef = uf->sinfacs[2*ifn+1];
	      sinefdown = uf->sinfacs[2*ifn+2];
	      sinefup = uf->sinfacs[2*ifn];
	      vol = h2*sinef;
	      
	      for(m=0;m<2;++m)
		{
		  jfn = 2*(j-1) + m + 1;
		  lind = jfn-1;
		  rind = jfn+1;
		  
		  lop = 0.0;
		  
		  //Laplacian terms
#ifdef USE_BNDWGT
		  if(ifn == 1)
		    lop += (sinefup*((u[tind*N + jfn])*(1.0 - BNDWGT) + (u[ifn*N+jfn])*BNDWGT - u[ifn*N + jfn])); //top face
		  else
		    lop += (sinefup*(u[tind*N + jfn] - u[ifn*N + jfn])); //top face
		  
		  if(ifn == N-2)
		    lop += (sinefdown*((u[bind*N + jfn])*(1.0 - BNDWGT) + (u[ifn*N+jfn])*BNDWGT - u[ifn*N + jfn])); //bottom face
		  else
		    lop += (sinefdown*(u[bind*N + jfn] - u[ifn*N + jfn])); //bottom face
		  
		  if(jfn == 1)
		    lop += (1.0/sinef*((u[ifn*N + lind])*(1.0 - BNDWGT) + (u[ifn*N+jfn])*BNDWGT - u[ifn*N + jfn])); //left face      
		  else
		    lop += (1.0/sinef*(u[ifn*N + lind] - u[ifn*N + jfn])); //left face      
		  
		  if(jfn == N-2)
		    lop += (1.0/sinef*((u[ifn*N + rind])*(1.0 - BNDWGT) + (u[ifn*N+jfn])*BNDWGT - u[ifn*N + jfn])); //right face
		  else
		    lop += (1.0/sinef*(u[ifn*N + rind] - u[ifn*N + jfn])); //right face
#else     
		  lop += (sinefup*(u[tind*N + jfn] - u[ifn*N + jfn])); //top face
		  lop += (sinefdown*(u[bind*N + jfn] - u[ifn*N + jfn])); //bottom face
		  lop += (1.0/sinef*(u[ifn*N + lind] - u[ifn*N + jfn])); //left face      
		  lop += (1.0/sinef*(u[ifn*N + rind] - u[ifn*N + jfn])); //right face
#endif
		  
		  lop /= vol;
		  
		  //comp resid
		  resid = rhof->grid[ifn*(uf->N) + jfn] - lop;
		  
		  mass += (resid)*(uf->cosfacs[ifn]);
		}
	    }
	  
	  uc->grid[i*(uc->N)+j] = mass/volc;
	}
    }
}

static void lop_mggrid_into_grid(MGGrid g, MGGrid lopg)
{
  long i,j,tind,bind,lind,rind,N;
  double sinef,sinefdown,sinefup,vol;
  double h2;
  mgfloat *u,*lop;
      
  //lopg = copy_mggrid(g);
  //zero_mggrid(lopg);
  
  h2 = g->dL;
  h2 *= h2;
  u = g->grid;
  lop = lopg->grid;
  N = g->N;
  
  for(i=1;i<N-1;++i)
    {
      tind = i-1;
      bind = i+1;
      sinef = g->sinfacs[2*i+1];
      sinefdown = g->sinfacs[2*i+2];
      sinefup = g->sinfacs[2*i];
      vol = h2*sinef;
      
      for(j=1;j<N-1;++j)
        {
          lind = j-1;
	  rind = j+1;
          
	  lop[i*N+j] = 0.0;
	  
	  //Laplacian terms
#ifdef USE_BNDWGT
	  if(i == 1)
	    lop[i*N+j] += (sinefup*((u[tind*N + j])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //top face
	  else
	    lop[i*N+j] += (sinefup*(u[tind*N + j] - u[i*N + j])); //top face
	  
	  if(i == N-2)
	    lop[i*N+j] += (sinefdown*((u[bind*N + j])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //bottom face
	  else
	    lop[i*N+j] += (sinefdown*(u[bind*N + j] - u[i*N + j])); //bottom face
	  
	  if(j == 1)
	    lop[i*N+j] += (1.0/sinef*((u[i*N + lind])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //left face      
	  else
	    lop[i*N+j] += (1.0/sinef*(u[i*N + lind] - u[i*N + j])); //left face      
	  
	  if(j == N-2)
	    lop[i*N+j] += (1.0/sinef*((u[i*N + rind])*(1.0 - BNDWGT) + (u[i*N+j])*BNDWGT - u[i*N + j])); //right face
	  else
	    lop[i*N+j] += (1.0/sinef*(u[i*N + rind] - u[i*N + j])); //right face
#else	  
	  lop[i*N+j] += (sinefup*(u[tind*N + j] - u[i*N + j])); //top face
	  lop[i*N+j] += (sinefdown*(u[bind*N + j] - u[i*N + j])); //bottom face
	  lop[i*N+j] += (1.0/sinef*(u[i*N + lind] - u[i*N + j])); //left face      
	  lop[i*N+j] += (1.0/sinef*(u[i*N + rind] - u[i*N + j])); //right face
#endif
	  
	  lop[i*N+j] /= vol;
	}
    }
}

double truncErr_mggrid(MGGrid uf, MGGrid uc, MGGrid lopc)
{
  long i,j,n,m,ifn,jfn;
  double mass;
  long tind,bind,lind,rind,N;
  double sinef,sinefdown,sinefup,vol;
  double h2;
  mgfloat *u;
  double lop;
  double volc;
  double norm = 0.0;
  
  restrict_mggrid(uc,uf);
  lop_mggrid_into_grid(uc,lopc);
  
  h2 = uf->dL;
  h2 *= h2;
  u = uf->grid;
  N = uf->N;
  
  //take a simple average
  for(i=1;i<uc->N-1;++i)
    {
      volc = 2.0*(uc->cosfacs[i]);
      
      for(j=1;j<uc->N-1;++j)
	{
	  mass = 0.0;
	  for(n=0;n<2;++n)
	    {
	      ifn = 2*(i-1) + n + 1;
	      tind = ifn-1;
	      bind = ifn+1;
	      sinef = uf->sinfacs[2*ifn+1];
	      sinefdown = uf->sinfacs[2*ifn+2];
	      sinefup = uf->sinfacs[2*ifn];
	      vol = h2*sinef;
	      
	      for(m=0;m<2;++m)
		{
		  jfn = 2*(j-1) + m + 1;
		  lind = jfn-1;
		  rind = jfn+1;
		  
		  lop = 0.0;
		  
		  //Laplacian terms
#ifdef USE_BNDWGT
		  if(ifn == 1)
		    lop += (sinefup*((u[tind*N + jfn])*(1.0 - BNDWGT) + (u[ifn*N+jfn])*BNDWGT - u[ifn*N + jfn])); //top face
		  else
		    lop += (sinefup*(u[tind*N + jfn] - u[ifn*N + jfn])); //top face
		  
		  if(ifn == N-2)
		    lop += (sinefdown*((u[bind*N + jfn])*(1.0 - BNDWGT) + (u[ifn*N+jfn])*BNDWGT - u[ifn*N + jfn])); //bottom face
		  else
		    lop += (sinefdown*(u[bind*N + jfn] - u[ifn*N + jfn])); //bottom face
		  
		  if(jfn == 1)
		    lop += (1.0/sinef*((u[ifn*N + lind])*(1.0 - BNDWGT) + (u[ifn*N+jfn])*BNDWGT - u[ifn*N + jfn])); //left face      
		  else
		    lop += (1.0/sinef*(u[ifn*N + lind] - u[ifn*N + jfn])); //left face      
		  
		  if(jfn == N-2)
		    lop += (1.0/sinef*((u[ifn*N + rind])*(1.0 - BNDWGT) + (u[ifn*N+jfn])*BNDWGT - u[ifn*N + jfn])); //right face
		  else
		    lop += (1.0/sinef*(u[ifn*N + rind] - u[ifn*N + jfn])); //right face
#else     
		  lop += (sinefup*(u[tind*N + jfn] - u[ifn*N + jfn])); //top face
		  lop += (sinefdown*(u[bind*N + jfn] - u[ifn*N + jfn])); //bottom face
		  lop += (1.0/sinef*(u[ifn*N + lind] - u[ifn*N + jfn])); //left face      
		  lop += (1.0/sinef*(u[ifn*N + rind] - u[ifn*N + jfn])); //right face
#endif
		  
		  lop /= vol;
		  
		  mass += (lop)*(uf->cosfacs[ifn]);
		}
	    }
	  
	  lopc->grid[i*(uc->N)+j] -= mass/volc;
	  norm += (lopc->grid[i*(uc->N)+j])*(lopc->grid[i*(uc->N)+j]);
	}
    }
  
  norm /= uc->N;
  norm /= uc->N;
  norm = sqrt(norm);
  
  /* code to check comp above
     double tnorm;
     MGGrid rlopf,lopf,loprf,rf;
     
     lopf = lop_mggrid(uf);
     rlopf = copy_mggrid(uc);
     restrict_mggrid(rlopf,lopf);
     
     rf = copy_mggrid(uc);
     restrict_mggrid(rf,uf);
     loprf = lop_mggrid(rf);
     
     tnorm = 0.0;
     for(i=1;i<uc->N-1;++i)
     for(j=1;j<uc->N-1;++j)
     tnorm += (loprf->grid[i*(uc->N)+j] - rlopf->grid[i*(uc->N)+j])*(loprf->grid[i*(uc->N)+j] - rlopf->grid[i*(uc->N)+j]);
     
     tnorm /= uc->N;
     tnorm /= uc->N;
     tnorm = sqrt(tnorm);
     
     free_mggrid(rlopf);
     free_mggrid(lopf);
     free_mggrid(loprf);
     free_mggrid(rf);
     
     fprintf(stderr,"norm = %le, tnorm = %le\n",norm,tnorm);
  */

  return norm;
}

void zero_mggrid(MGGrid u)
{
  long i,j;
  
  for(i=0;i<u->N;++i)
    for(j=0;j<u->N;++j)
      u->grid[i*(u->N)+j] = 0.0;
}

MGGrid alloc_mggrid(long N, double L)
{
  MGGrid u;
  long i;
  
  //do mem alloc
  long Ntot = N+2;
  u = (MGGrid)malloc(sizeof(_MGGrid) + Ntot*Ntot*sizeof(mgfloat) + (2*Ntot+1)*sizeof(double) + Ntot*sizeof(double) + 5*Ntot*sizeof(double));
  assert(u != NULL);
  
  u->grid = (mgfloat*)(u + 1);
  u->sinfacs = (double*)(u->grid + Ntot*Ntot);
  u->cosfacs = u->sinfacs + 2*Ntot+1;
  u->sintheta = u->cosfacs + Ntot;
  u->costheta = u->sintheta + Ntot;
  u->sinphi = u->costheta + Ntot;
  u->cosphi = u->sinphi + Ntot;
  u->diag = u->cosphi + Ntot;
  
  //assign values
  u->dL = L/N;
  u->L = L + 2*(u->dL);
  
  u->thetaLoc = M_PI/2.0 - (u->L)/2.0 + u->dL/2.0;
  u->phiLoc = -(u->L)/2.0 + u->dL/2.0;
  u->N = Ntot;
  
  long ind,indm;
  double dL_2 = (u->dL)/2.0;
  for(i=0;i<Ntot;++i)
    {
      ind = 2*i+1;
      indm = ind - 1;
      u->sinfacs[ind] = sin(i*(u->dL) + u->thetaLoc);
      u->sinfacs[indm] = sin(i*(u->dL) + u->thetaLoc - dL_2);
      u->cosfacs[i] = cos(i*(u->dL) + u->thetaLoc - (u->dL)/2.0) - cos(i*(u->dL) + u->thetaLoc + (u->dL)/2.0);
      
      u->sintheta[i] = sin(i*(u->dL) + u->thetaLoc);
      u->costheta[i] = cos(i*(u->dL) + u->thetaLoc);
      
      u->sinphi[i] = sin(i*(u->dL) + u->phiLoc);
      u->cosphi[i] = cos(i*(u->dL) + u->phiLoc);
    }
  u->sinfacs[2*Ntot] = sin((N-1)*(u->dL) + u->thetaLoc + dL_2);
  
  //fill in diag
  for(i=1;i<u->N-1;++i)
    u->diag[i] = u->sinfacs[2*i] + u->sinfacs[2*i+2] + 2.0/(u->sinfacs[2*i+1]);
  
  return u;
}

MGGrid copy_mggrid(MGGrid u)
{
  MGGrid c;
  long N = u->N;
  
  //do mem alloc
  c = (MGGrid)malloc(sizeof(_MGGrid) + N*N*sizeof(mgfloat) + (2*N+1)*sizeof(double) + N*sizeof(double) + 5*N*sizeof(double));
  assert(c != NULL);
  *c = *u;
  
  c->grid = (mgfloat*)(c + 1);
  c->sinfacs = (double*)(c->grid + N*N);
  c->cosfacs = c->sinfacs + 2*N+1;
  c->sintheta = c->cosfacs + N;
  c->costheta = c->sintheta + N;
  c->sinphi = c->costheta + N;
  c->cosphi = c->sinphi + N;
  c->diag = c->cosphi + N;
  
  memcpy(c->grid,u->grid,sizeof(mgfloat)*N*N);
  memcpy(c->sinfacs,u->sinfacs,sizeof(double)*(2*N+1));
  memcpy(c->cosfacs,u->cosfacs,sizeof(double)*N);
  
  memcpy(c->sintheta,u->sintheta,sizeof(double)*N);
  memcpy(c->costheta,u->costheta,sizeof(double)*N);
  
  memcpy(c->sinphi,u->sinphi,sizeof(double)*N);
  memcpy(c->cosphi,u->cosphi,sizeof(double)*N);
  
  memcpy(c->diag,u->diag,sizeof(double)*N);
    
  return c;
}

void free_mggrid(MGGrid u)
{
  free(u);
}

void write_mggrid(char fname[], MGGrid u)
{
  FILE *fp;
  
  fp = fopen(fname,"w");
  fwrite(&(u->N),sizeof(long),(size_t) 1,fp);
  fwrite(&(u->L),sizeof(double),(size_t) 1,fp);
  fwrite(u->grid,(size_t) ((u->N)*(u->N)),sizeof(mgfloat),fp);
  fclose(fp);
}

void print_runtimes_mgsteps(void)
{
  fprintf(stderr,"MG run times - pre-smooth,down,exact,up,post-smooth = %lf|%lf|%lf|%lf|%lf\n",
	  runTimesMGSteps[0],runTimesMGSteps[1],runTimesMGSteps[2],runTimesMGSteps[3],runTimesMGSteps[4]);
}


void reset_runtimes_mgsteps(void)
{
  int i;
  for(i=0;i<NumRunTimesMGSteps;++i)
    runTimesMGSteps[i] = 0.0;
}
