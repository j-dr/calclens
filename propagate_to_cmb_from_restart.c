//////////////////////////////////////////////////////////////////////
// author(s): Stefan Hilbert
//////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>

#include <gsl/gsl_errno.h>
// #include <gsl/gsl_math.h>
#include <gsl/gsl_sf_hyperg.h>

#include "raytrace.h"

//////////////////////////////////////////////////////////////////////
// aux function: checked_fopen
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline FILE *checked_fopen( const char *filename, const char *mode)
{
  FILE * fp;
 
  fp = fopen(filename, mode);
  if(!fp)
  {
    fprintf(stderr,"task %d: could not open file '%s'!\n",ThisTask, filename);
    MPI_Abort(MPI_COMM_WORLD,777);
  }
  return fp;
}


//////////////////////////////////////////////////////////////////////
// aux function: checked_fread
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline size_t checked_fread(void *p, size_t size, size_t nitems, FILE *fp)
{
  size_t nrw;
  nrw = fread(p,size,nitems,fp);
  if(nrw != nitems)
  {
    fprintf(stderr,"task %d: error in checked read!\n",ThisTask);
    MPI_Abort(MPI_COMM_WORLD,777);
  }
  return nrw;
}


//////////////////////////////////////////////////////////////////////
// aux function: checked_fread
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline size_t checked_fwrite(void *p, size_t size, size_t nitems, FILE *fp)
{
  size_t nrw;
  nrw = fwrite(p,size,nitems,fp);
  if(nrw != nitems)
  {
    fprintf(stderr,"task %d: error in checked write!\n",ThisTask);
    MPI_Abort(MPI_COMM_WORLD,777);
  }
  return nrw;
}


//////////////////////////////////////////////////////////////////////
// aux function: checked_fignore
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline size_t checked_fignore(void *p, size_t size, size_t nitems, FILE *fp)
{
  size_t n_bytes_to_ignore  = size * nitems;
        
  fprintf(stderr, "debugging: n_bytes_to_ignore = %ld\n", n_bytes_to_ignore);
      
  if(!fseek(fp, n_bytes_to_ignore, SEEK_CUR))
  {
    fprintf(stderr,"task %d: error in checked ignore!\n", ThisTask);
    MPI_Abort(MPI_COMM_WORLD,777);
  }
  return nitems;
}


//////////////////////////////////////////////////////////////////////
// aux functions: bin_number_for_item_number etc.
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline int
bin_number_for_item_number(const int item_number_, const int number_of_bins_, const int number_of_items_)
{ return (item_number_ * number_of_bins_) / number_of_items_; }

static inline int
item_number_begin_for_bin_number(const int bin_number_, const int number_of_bins_, const int number_of_items_)
{ return (bin_number_ * number_of_items_) / number_of_bins_ + (((bin_number_ * number_of_items_) % number_of_bins_) ? 1 : 0); }

static inline int
item_number_end_for_bin_number(const int bin_number_, const int number_of_bins_, const int number_of_items_)
{ return item_number_begin_for_bin_number(bin_number_ + 1, number_of_bins_, number_of_items_); }


//////////////////////////////////////////////////////////////////////
// function: alloc_rays_for_one_restart_file
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
void alloc_rays_for_one_restart_file(const int number_of_restart_files)
{
  long shift,NraysPerBundleCell;
  
  shift = rayTraceData.rayOrder - rayTraceData.bundleOrder;
  shift = 2*shift;
  
  NraysPerBundleCell = 1;
  NraysPerBundleCell = (NraysPerBundleCell << shift);
  
  long NumBuff = 25.0*1024.0*1024.0/sizeof(HEALPixRay)/NraysPerBundleCell;
  if(NumBuff < 10)
    NumBuff = 10;
  MaxNumAllRaysGlobal = ((long) ((1.0 + rayTraceData.maxRayMemImbalance)*NrestrictedPeanoInd / number_of_restart_files))*NraysPerBundleCell + NumBuff*NraysPerBundleCell;
  AllRaysGlobal = (HEALPixRay*)malloc(sizeof(HEALPixRay)*MaxNumAllRaysGlobal);
  assert(AllRaysGlobal != NULL);
}

  
//////////////////////////////////////////////////////////////////////
// function: reset_bundle_cells
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
void reset_bundle_cells(void)
{
  long i;
  long shift,NraysPerBundleCell;
  
  shift = rayTraceData.rayOrder - rayTraceData.bundleOrder;
  shift = 2*shift;
  
  NraysPerBundleCell = 1;
  NraysPerBundleCell = (NraysPerBundleCell << shift);
  
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
 
 
//////////////////////////////////////////////////////////////////////
// function: set_plane_distances
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static void set_plane_distances(void)
{
  double binL = (rayTraceData.maxComvDistance)/((double) (rayTraceData.NumLensPlanes));
  
  //get comv distances
  if(rayTraceData.CurrentPlaneNum - 1 < 0)
    rayTraceData.planeRadMinus1 = 0.0;
  else
    rayTraceData.planeRadMinus1 = (rayTraceData.CurrentPlaneNum - 1.0)*binL + binL/2.0;
  
  rayTraceData.planeRad = rayTraceData.CurrentPlaneNum*binL + binL/2.0;
  
  if(rayTraceData.CurrentPlaneNum+1 == rayTraceData.NumLensPlanes)
    rayTraceData.planeRadPlus1 = rayTraceData.maxComvDistance;
  else
    rayTraceData.planeRadPlus1 = (rayTraceData.CurrentPlaneNum + 1.0)*binL + binL/2.0;
}


////////////////////////////////////////////////////////////////////////
// flat_LambdaCDM_line_of_sight_comoving_distance_for_redshift:
//--------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////
static inline double
flat_LambdaCDM_line_of_sight_comoving_distance_for_redshift_(const double z_, const double Omega_matter_, const double Hubble_distance_ ) 
{
  const double Omega_Lambda_ = 1.  - Omega_matter_;
  const double inv_omlf_     = 1. / (Omega_Lambda_ + (1. + z_) * (1. + z_) * (1. + z_) * Omega_matter_);
  const double result_       = 
  (0.99 < Omega_Lambda_ * inv_omlf_) ?
    Hubble_distance_ * z_ :
    Hubble_distance_ * (  2. * gsl_sf_hyperg_2F1(1./2., 1., 7./6., Omega_Lambda_            )
                        - 2. * gsl_sf_hyperg_2F1(1./2., 1., 7./6., Omega_Lambda_ * inv_omlf_) * sqrt(inv_omlf_) * (1. + z_));
  return result_;
}

static inline double
flat_LambdaCDM_line_of_sight_comoving_distance_for_redshift(const double z_, const double Omega_matter_) 
{ return flat_LambdaCDM_line_of_sight_comoving_distance_for_redshift_(z_, Omega_matter_, 2997.92458 /* value in Mpc, change if using different units */); }


////////////////////////////////////////////////////////////////////////
// fprintf_ray:
//--------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////
static inline int
fprintf_ray(FILE *stream_, const HEALPixRay* ray_)
{
  int ret_ = 0;
  ret_ |= fprintf(stream_, "  ray.nest  = %ld\n"           , ray_->nest                                                    );
  ret_ |= fprintf(stream_, "  ray.n     = %f, %f, %f\n"    , ray_->n[0], ray_->n[1], ray_->n[2]                            );
  ret_ |= fprintf(stream_, "  ray.beta  = %f, %f\n"        , ray_->beta[0], ray_->beta[1]                                  );
  ret_ |= fprintf(stream_, "  ray.alpha = %f, %f\n"        , ray_->alpha[0], ray_->alpha[1]                                );
  ret_ |= fprintf(stream_, "  ray.A     = %f, %f, %f, %f\n", ray_->A[0], ray_->A[1], ray_->A[2], ray_->A[3]                );
  ret_ |= fprintf(stream_, "  ray.Aprev = %f, %f, %f, %f\n", ray_->Aprev[0], ray_->Aprev[1], ray_->Aprev[2], ray_->Aprev[3]);
  ret_ |= fprintf(stream_, "  ray.U     = %f, %f, %f, %f\n", ray_->U[0], ray_->U[1], ray_->U[2], ray_->U[3]                );
  ret_ |= fprintf(stream_, "  ray.phi   = %f\n"            , ray_->phi                                                     );
  return ret_;
}

//////////////////////////////////////////////////////////////////////
// function: propagate_to_cmb_from_restart
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
void propagate_to_cmb_from_restart(void)
{
  RayTraceData rtd;
  long i;
  
  int number_of_restart_files;
  int fspd;

  FILE *fp;
  char restart_file_name          [MAX_FILENAME];
  char temporary_restart_file_name[MAX_FILENAME];
  char sys[MAX_FILENAME];
  char map_filename = "CMB_rays_2048.fits";
  long maporder_ = 11;

//  char cmb_lensing_output_file_name[MAX_FILENAME];
  int pbc;
  long NraysPerBundleCell, bundleCellInd;
//  double time,minTime,maxTime,totTime,avgTime;
  
  // for ThisTask == 0: read first restart file & extract data:
  if(ThisTask==0)
  { 
    fprintf(stdout,"trying to read rays from restart files and propagating them to the CMB...\n");
 
//    sprintf(restart_file_name, "%s/restart.0", rayTraceData.OutputPath);
     sprintf(restart_file_name, "/nfs/slac/g/ki/ki23/des/jderose/BCC/Chinchilla/Herd/Chinchilla-1/calclens/restart.0");
    
    fp = checked_fopen(restart_file_name, "r");
    fprintf(stderr, "debugging: opened file %s\n", restart_file_name);
    
    checked_fread(&number_of_restart_files, sizeof(int), (size_t) 1, fp);
    fprintf(stderr, "debugging: number_of_restart_files = %d\n", number_of_restart_files);
 
    checked_fread(&fspd, sizeof(int),(size_t) 1, fp);
    fprintf(stderr, "debugging: fspd = %d\n", fspd);

    checked_fread(&rtd, sizeof(RayTraceData), (size_t) 1, fp);
    fprintf(stderr, "debugging: rtd.bundleOrder = %ld\n", rtd.bundleOrder);
    fprintf(stderr, "debugging: rtd.rayOrder = %ld\n", rtd.rayOrder);

    rayTraceData.bundleOrder = rtd.bundleOrder;
    rayTraceData.rayOrder    = rtd.rayOrder   ;

    checked_fread(&NbundleCells, sizeof(long), (size_t) 1, fp);
    fprintf(stderr, "debugging: NbundleCells = %ld\n", NbundleCells);
 
    bundleCells = (HEALPixBundleCell*)malloc(sizeof(HEALPixBundleCell)*NbundleCells);
    assert(bundleCells != NULL);
  
    bundleCellsNest2RestrictedPeanoInd = (long*)malloc(sizeof(long)*NbundleCells);
    assert(bundleCellsNest2RestrictedPeanoInd != NULL);
  
    bundleCellsRestrictedPeanoInd2Nest = (long*)malloc(sizeof(long)*NbundleCells);
    assert(bundleCellsRestrictedPeanoInd2Nest != NULL);

    checked_fread(bundleCells                       , sizeof(HEALPixBundleCell), (size_t) NbundleCells, fp);
    checked_fread(bundleCellsNest2RestrictedPeanoInd, sizeof(long)             , (size_t) NbundleCells, fp);
    checked_fread(bundleCellsRestrictedPeanoInd2Nest, sizeof(long)             , (size_t) NbundleCells, fp);
                  
    checked_fread(&NrestrictedPeanoInd, sizeof(long), (size_t) 1, fp);
    fprintf(stderr, "debugging: NrestrictedPeanoInd = %ld\n", NrestrictedPeanoInd);

    fclose(fp);
  } /* for ThisTask == 0: read first restart file & extract data */

  { /* initialize map data */
    long  npix = 12*(1<<maporder_*2);

    A00 = (float*)malloc(sizeof(float)*npix);
    A01 = (float*)malloc(sizeof(float)*npix);
    A10 = (float*)malloc(sizeof(float)*npix);
    A11 = (float*)malloc(sizeof(float)*npix);
    ra  = (float*)malloc(sizeof(float)*npix);
    dec = (float*)malloc(sizeof(float)*npix);
    nest = (long*)malloc(sizeof(long));
  } /* initialize map data */

  { /* distribute information from first restart file */
    MPI_Bcast(&number_of_restart_files , 1, MPI_INT , 0, MPI_COMM_WORLD);
    MPI_Bcast(&rayTraceData.bundleOrder, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rayTraceData.rayOrder   , 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NbundleCells            , 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NrestrictedPeanoInd     , 1, MPI_LONG, 0, MPI_COMM_WORLD);
  } /* distribute information from first restart file */
  
  { /* set up mem structs to hold ray and aux data */
    NraysPerBundleCell = (1ll) << (2*(rayTraceData.rayOrder - rayTraceData.bundleOrder));
   
    if(ThisTask != 0)
    {
      bundleCells = (HEALPixBundleCell*)malloc(sizeof(HEALPixBundleCell)*NbundleCells);
      assert(bundleCells != NULL);
    
      bundleCellsNest2RestrictedPeanoInd = (long*)malloc(sizeof(long)*NbundleCells);
      assert(bundleCellsNest2RestrictedPeanoInd != NULL);
    
      bundleCellsRestrictedPeanoInd2Nest = (long*)malloc(sizeof(long)*NbundleCells);
      assert(bundleCellsRestrictedPeanoInd2Nest != NULL);
    }
   
    firstRestrictedPeanoIndTasks = (long*) malloc(sizeof(long) * number_of_restart_files);
    assert(firstRestrictedPeanoIndTasks != NULL);
      
    lastRestrictedPeanoIndTasks = (long*) malloc(sizeof(long) * number_of_restart_files);
    assert(lastRestrictedPeanoIndTasks != NULL);
    
    alloc_rays_for_one_restart_file(number_of_restart_files);
  } /* set up mem structs to hold ray and aux data */

  // work filewise: (i) read rays, (ii) propagate rays, (iii) write rays
    
  // const int restart_file_number_begin = item_number_begin_for_bin_number(ThisTask, NTasks, number_of_restart_files);
  // const int restart_file_number_end   = item_number_end_for_bin_number  (ThisTask, NTasks, number_of_restart_files);
  
  //debugging:
  int restart_file_number;
  const int restart_file_number_begin = 0;
  const int restart_file_number_end   = 1;
  for(restart_file_number = restart_file_number_begin; restart_file_number < restart_file_number_end; restart_file_number++)
  {
    { /* read rays: */
//      sprintf(restart_file_name, "%s/restart.%d", rayTraceData.OutputPath, restart_file_number);
      sprintf(restart_file_name, "/nfs/slac/g/ki/ki23/des/jderose/BCC/Chinchilla/Herd/Chinchilla-1/calclens/restart.%d", restart_file_number);
    
#ifdef DEBUG
#if DEBUG_LEVEL > 0
      fprintf(stderr,"task %d:trying to read rays from restart file '%s'...\n", ThisTask, restart_file_name);
#endif /* DEBUG_LEVEL > 0 */
#endif /* DEBUG */   

      fp = checked_fopen(restart_file_name, "r");
      
      checked_fread(&number_of_restart_files, sizeof(int), (size_t) 1, fp);
      
      checked_fread(&fspd, sizeof(int),(size_t) 1, fp);
     
      checked_fread(&rtd, sizeof(RayTraceData), (size_t) 1, fp);
     
      rayTraceData.bundleOrder     = rtd.bundleOrder     ;
      rayTraceData.rayOrder        = rtd.rayOrder        ;
      rayTraceData.OmegaM          = rtd.OmegaM          ;
      rayTraceData.maxComvDistance = rtd.maxComvDistance ;
      rayTraceData.NumLensPlanes   = rtd.NumLensPlanes   ;
      rayTraceData.minRa           = rtd.minRa           ;
      rayTraceData.maxRa           = rtd.maxRa           ;
      rayTraceData.minDec          = rtd.minDec          ;
      rayTraceData.maxDec          = rtd.maxDec          ;
      rayTraceData.CurrentPlaneNum = rtd.CurrentPlaneNum ;
//       rayTraceData.Restart         = rtd.CurrentPlaneNum ;
     
      checked_fread(&NbundleCells, sizeof(long), (size_t) 1, fp);
     
      checked_fread(bundleCells                       , sizeof(HEALPixBundleCell), (size_t) NbundleCells, fp);
      checked_fread(bundleCellsNest2RestrictedPeanoInd, sizeof(long)             , (size_t) NbundleCells, fp);
      checked_fread(bundleCellsRestrictedPeanoInd2Nest, sizeof(long)             , (size_t) NbundleCells, fp);
                    
      checked_fread(&NrestrictedPeanoInd, sizeof(long), (size_t) 1, fp);
     
      checked_fread(firstRestrictedPeanoIndTasks, (size_t) number_of_restart_files, sizeof(long), fp);
      checked_fread(lastRestrictedPeanoIndTasks , (size_t) number_of_restart_files, sizeof(long), fp);
     
      checked_fread(&pbc, sizeof(int), (size_t) 1, fp);
     
      reset_bundle_cells();
      
      for(i=0; i < NbundleCells; ++i) 
       if(ISSETBITFLAG(bundleCells[i].active, PRIMARY_BUNDLECELL))
          checked_fread(bundleCells[i].rays, (size_t) NraysPerBundleCell, sizeof(HEALPixRay), fp);
     
      long position_in_file_after_reading = ftell(fp);
      fprintf(stderr, "debugging: position_in_file_after_reading = %ld\n", position_in_file_after_reading);
      
      fclose(fp);
    } /* read rays */

    { /* propagate rays: */
      set_plane_distances();
      const double z_CMB = 1100.;
      double wp_CMB = flat_LambdaCDM_line_of_sight_comoving_distance_for_redshift(z_CMB, rayTraceData.OmegaM);
      fprintf(stderr, "debugging: rayTraceData.planeRadMinus1 = %f\n", rayTraceData.planeRadMinus1);
      fprintf(stderr, "debugging: rayTraceData.planeRad = %f\n", rayTraceData.planeRad);
      fprintf(stderr, "debugging: wp_CMB = %f\n", wp_CMB);
      
      int index_of_first_active_bundle_cell = -1;
      int index_of_last_active_bundle_cell = -1;
      for(bundleCellInd = 0; bundleCellInd < NbundleCells; ++bundleCellInd) 
        if(ISSETBITFLAG(bundleCells[bundleCellInd].active, PRIMARY_BUNDLECELL))
        { 
          index_of_first_active_bundle_cell = bundleCellInd;
          break;
        }
      for(bundleCellInd = NbundleCells; bundleCellInd--; ) 
        if(ISSETBITFLAG(bundleCells[bundleCellInd].active, PRIMARY_BUNDLECELL))
        { 
          index_of_last_active_bundle_cell = bundleCellInd;
          break;
        }
        
      if(index_of_first_active_bundle_cell < 0)
      {  
        fprintf(stderr,"task %d: no active bundle cells \n",ThisTask);
      }
      else
      {
        fprintf(stderr,"first ray before:\n");
        fprintf_ray(stderr, &(bundleCells[index_of_first_active_bundle_cell].rays[0]));
        fprintf(stderr,"last ray before:\n");
        fprintf_ray(stderr, &(bundleCells[index_of_last_active_bundle_cell].rays[0]));
      }
   
      for(bundleCellInd = 0; bundleCellInd < NbundleCells; ++bundleCellInd) 
       if(ISSETBITFLAG(bundleCells[bundleCellInd].active, PRIMARY_BUNDLECELL))
       {
         for(i = 0; i < bundleCells[bundleCellInd].Nrays; ++i)
         {
            bundleCells[bundleCellInd].rays[i].alpha[0] = 0.;
            bundleCells[bundleCellInd].rays[i].alpha[1] = 0.;
            bundleCells[bundleCellInd].rays[i].U[0]     = 0.;
            bundleCells[bundleCellInd].rays[i].U[1]     = 0.;
            bundleCells[bundleCellInd].rays[i].U[2]     = 0.;
            bundleCells[bundleCellInd].rays[i].U[3]     = 0.;
            bundleCells[bundleCellInd].rays[i].phi      = 0.;
         }
         rayprop_sphere(wp_CMB, rayTraceData.planeRad, rayTraceData.planeRadMinus1, bundleCellInd);
       }
       
      if(index_of_first_active_bundle_cell >= 0)
      {
        fprintf(stderr,"first ray after:\n");
        fprintf_ray(stderr, &(bundleCells[index_of_first_active_bundle_cell].rays[0]));
        fprintf(stderr,"last ray after:\n");
        fprintf_ray(stderr, &(bundleCells[index_of_last_active_bundle_cell].rays[0]));
      }
    } /* propagate rays */

    { /* update degraded healpix map */
      updateMap(&bundleCells[bundlecellInd], maporder_,
		&nest, &A00, &A01, &A10, &A11, &ra, &dec);
    }

    { /* write rays: */
    
  //     sprintf(restart_file_name, "/nfs/slac/g/ki/ki23/des/jderose/BCC/Chinchilla/Herd/Chinchilla-1/calclens/restart_rays_at_cmb.%d", restart_file_number);
      sprintf(restart_file_name, "%s/restart_rays_at_cmb.%d", rayTraceData.OutputPath, restart_file_number);
      sprintf(temporary_restart_file_name, "%s.tmp", restart_file_name);
      
      fprintf(stderr, "debugging: restart_file_name = '%s'\n", restart_file_name);
      fprintf(stderr, "debugging: temporary_restart_file_name = '%s'\n", temporary_restart_file_name);


      fp = checked_fopen(temporary_restart_file_name, "w");
      
      checked_fwrite(&number_of_restart_files, sizeof(int), (size_t) 1, fp);
       
      checked_fwrite(&fspd, sizeof(int),(size_t) 1, fp);
      
      checked_fwrite(&rtd, sizeof(RayTraceData), (size_t) 1, fp);
      
      rayTraceData.bundleOrder     = rtd.bundleOrder     ;
      rayTraceData.rayOrder        = rtd.rayOrder        ;
      rayTraceData.OmegaM          = rtd.OmegaM          ;
      rayTraceData.maxComvDistance = rtd.maxComvDistance ;
      rayTraceData.NumLensPlanes   = rtd.NumLensPlanes   ;
      rayTraceData.minRa           = rtd.minRa           ;
      rayTraceData.maxRa           = rtd.maxRa           ;
      rayTraceData.minDec          = rtd.minDec          ;
      rayTraceData.maxDec          = rtd.maxDec          ;
      rayTraceData.CurrentPlaneNum = rtd.CurrentPlaneNum ;
//       rayTraceData.Restart         = rtd.CurrentPlaneNum ;
      
      checked_fwrite(&NbundleCells, sizeof(long), (size_t) 1, fp);
      
      checked_fwrite(bundleCells                       , sizeof(HEALPixBundleCell), (size_t) NbundleCells, fp);
      checked_fwrite(bundleCellsNest2RestrictedPeanoInd, sizeof(long)             , (size_t) NbundleCells, fp);
      checked_fwrite(bundleCellsRestrictedPeanoInd2Nest, sizeof(long)             , (size_t) NbundleCells, fp);
                    
      checked_fwrite(&NrestrictedPeanoInd, sizeof(long), (size_t) 1, fp);
      
      checked_fwrite(firstRestrictedPeanoIndTasks, (size_t) number_of_restart_files, sizeof(long), fp);
      checked_fwrite(lastRestrictedPeanoIndTasks , (size_t) number_of_restart_files, sizeof(long), fp);
      
      checked_fwrite(&pbc, sizeof(int), (size_t) 1, fp);
      
      for(i=0; i < NbundleCells; ++i) 
       if(ISSETBITFLAG(bundleCells[i].active, PRIMARY_BUNDLECELL))
          checked_fwrite(bundleCells[i].rays, (size_t) NraysPerBundleCell, sizeof(HEALPixRay), fp);
  
      long position_in_file_after_writing = ftell(fp);
      fprintf(stderr, "debugging: position_in_file_after_writing = %ld\n", position_in_file_after_writing);
      
      fclose(fp);
      sprintf(sys,"mv %s %s",temporary_restart_file_name, restart_file_name);
      system(sys);
 
    }  /* write rays */
  } /* loop over restart files */

  { /* reduce and write map */
    reduceMap(&nest, &A00, &A01, &A10, &A11, &ra, &dec, npix);
    if (ThisTask==0)
      {
	writeMap(&nest, &A00, &A01, &A10, &A11, &ra, &dec, npix,
		 map_filename);
      }
  } /*finalize map */

  { /* free mem */
    free(bundleCells);
    free(bundleCellsNest2RestrictedPeanoInd);
    free(bundleCellsRestrictedPeanoInd2Nest);
    free(firstRestrictedPeanoIndTasks);
    free(lastRestrictedPeanoIndTasks);
    destroy_rays();
  } /* free mem */
}

