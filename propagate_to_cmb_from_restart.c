//////////////////////////////////////////////////////////////////////
// author(s): Stefan Hilbert, Joe DeRose
//////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <unistd.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>

#include <gsl/gsl_errno.h>
// #include <gsl/gsl_math.h>
#include <gsl/gsl_sf_hyperg.h>

#include "raytrace.h"

#include "checked_io.h"
#include "checked_alloc.h"





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
static inline void
alloc_rays_for_one_restart_file(const int number_of_restart_files_)
{
  long shift_, NraysPerBundleCell_;

  shift_ = rayTraceData.rayOrder - rayTraceData.bundleOrder;
  shift_ = 2 * shift_;

  NraysPerBundleCell_ = 1;
  NraysPerBundleCell_ = (NraysPerBundleCell_ << shift_);

  long NumBuff_ = 25.0*1024.0*1024.0/sizeof(HEALPixRay)/NraysPerBundleCell_;
  if(NumBuff_ < 10)
  { NumBuff_ = 10; }
  MaxNumAllRaysGlobal = ((long) ((1.0 + rayTraceData.maxRayMemImbalance) * NrestrictedPeanoInd / number_of_restart_files_)) * NraysPerBundleCell_ + NumBuff_ * NraysPerBundleCell_;
  AllRaysGlobal = (HEALPixRay*)checked_malloc(sizeof(HEALPixRay) * MaxNumAllRaysGlobal);
}

//////////////////////////////////////////////////////////////////////
// function: alloc_rays_for_one_restart_file
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline void
free_rays()
{
  free(AllRaysGlobal);
  AllRaysGlobal = NULL;
  NumAllRaysGlobal = 0;
}


//////////////////////////////////////////////////////////////////////
// function: reset_bundle_cells
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
void reset_bundle_cells(void)
{
  long i_;
  long shift_, NraysPerBundleCell_;

  shift_ = rayTraceData.rayOrder - rayTraceData.bundleOrder;
  shift_ = 2 * shift_;

  NraysPerBundleCell_ = 1;
  NraysPerBundleCell_ = (NraysPerBundleCell_ << shift_);

  NumAllRaysGlobal = 0;

  for(i_ = 0; i_ < NbundleCells; ++i_)
    {
      if(ISSETBITFLAG(bundleCells[i_].active,PRIMARY_BUNDLECELL))
        {
          if(NumAllRaysGlobal >= MaxNumAllRaysGlobal)
          {
            fprintf(stderr,"%d: out of memory for rays! MaxNumAllRaysGlobal = %ld, NumAllRaysGlobal = %ld, max # of bundle cells = %ld, mem imbal = %lf, tot # of bundle cells = %ld\n",
                    ThisTask, MaxNumAllRaysGlobal, NumAllRaysGlobal, MaxNumAllRaysGlobal / NraysPerBundleCell_, rayTraceData.maxRayMemImbalance, NrestrictedPeanoInd);
            MPI_Abort(MPI_COMM_WORLD, 111);
          }

          bundleCells[i_].rays = AllRaysGlobal + NumAllRaysGlobal;
          bundleCells[i_].Nrays = NraysPerBundleCell_;
          NumAllRaysGlobal += NraysPerBundleCell_;
        }
    }
}


//////////////////////////////////////////////////////////////////////
// function: set_plane_distances
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static void
set_plane_distances(void)
{
  double binL_ = (rayTraceData.maxComvDistance)/((double) (rayTraceData.NumLensPlanes));

  //get comv distances
  if(rayTraceData.CurrentPlaneNum - 1 < 0)
  { rayTraceData.planeRadMinus1 = 0.0; }
  else
  { rayTraceData.planeRadMinus1 = (rayTraceData.CurrentPlaneNum - 1.0)*binL_ + binL_/2.0; }

  rayTraceData.planeRad = rayTraceData.CurrentPlaneNum*binL_ + binL_/2.0;

  if(rayTraceData.CurrentPlaneNum+1 == rayTraceData.NumLensPlanes)
  { rayTraceData.planeRadPlus1 = rayTraceData.maxComvDistance; }
  else
  { rayTraceData.planeRadPlus1 = (rayTraceData.CurrentPlaneNum + 1.0)*binL_ + binL_/2.0; }
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

double
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
  if(ThisTask==0)
  { fprintf(stdout, "trying to read rays from restart files and propagating them to the CMB...\n"); }

  RayTraceData rtd;
  long i;

  int number_of_restart_files;
  int fspd;

  FILE *fp;
  char output_filename          [MAX_FILENAME];
  char temporary_output_filename[MAX_FILENAME];
  char sys                      [MAX_FILENAME];

  const double z_CMB = 1100.;
  double wp_CMB;

  const bool write_restart_files_for_rays_at_cmb = false;
  const bool write_fits_maps_for_rays_at_cmb     = true;
  const bool overwrite_files_for_rays_at_cmb     = true;

//  const long map_order    = 11;
  const long map_order    = 7;
  const long map_n_side   = (1 << map_order);
  const long map_n_pixels = 12 * map_n_side * map_n_side;

  long   *map_pixel_sum_1  ;
  double *map_pixel_sum_A00;
  double *map_pixel_sum_A01;
  double *map_pixel_sum_A10;
  double *map_pixel_sum_A11;
  double *map_pixel_sum_ra ;
  double *map_pixel_sum_dec;

  int pbc;
  long NraysPerBundleCell, bundleCellIndex;

  // for ThisTask == 0: read first restart file & extract data:
  if(ThisTask==0)
  {
    fprintf(stdout, "task %d: trying to read aux data from first restart file...\n", ThisTask);

//    sprintf(output_filename, "%s/restart.0", rayTraceData.OutputPath);
     sprintf(output_filename, "/home/jderose/uscratch/BCC/Chinchilla/Herd/Chinchilla-1/calclens/restart.0");

    fp = checked_fopen(output_filename, "r");
    fprintf(stderr, "debugging: opened file %s\n", output_filename);

    checked_fread(&number_of_restart_files, sizeof(int)         , (size_t) 1, fp);
    fprintf(stderr, "debugging: number_of_restart_files = %d\n", number_of_restart_files);

    checked_fread(&fspd                   , sizeof(int)         , (size_t) 1, fp);
    fprintf(stderr, "debugging: fspd = %d\n", fspd);

    checked_fread(&rtd                    , sizeof(RayTraceData), (size_t) 1, fp);
    fprintf(stderr, "debugging: rtd.bundleOrder = %ld\n", rtd.bundleOrder);
    fprintf(stderr, "debugging: rtd.rayOrder = %ld\n", rtd.rayOrder);

    rayTraceData.bundleOrder = rtd.bundleOrder;
    rayTraceData.rayOrder    = rtd.rayOrder   ;

    checked_fread(&NbundleCells, sizeof(long), (size_t) 1, fp);
    fprintf(stderr, "debugging: NbundleCells = %ld\n", NbundleCells);

    bundleCells                        = (HEALPixBundleCell*)checked_malloc(sizeof(HEALPixBundleCell) * NbundleCells);
    bundleCellsNest2RestrictedPeanoInd = (long*             )checked_malloc(sizeof(long)              * NbundleCells);
    bundleCellsRestrictedPeanoInd2Nest = (long*             )checked_malloc(sizeof(long)              * NbundleCells);

    checked_fread(bundleCells                       , sizeof(HEALPixBundleCell), (size_t) NbundleCells, fp);
    checked_fread(bundleCellsNest2RestrictedPeanoInd, sizeof(long)             , (size_t) NbundleCells, fp);
    checked_fread(bundleCellsRestrictedPeanoInd2Nest, sizeof(long)             , (size_t) NbundleCells, fp);

    checked_fread(&NrestrictedPeanoInd, sizeof(long), (size_t) 1, fp);
    fprintf(stderr, "debugging: NrestrictedPeanoInd = %ld\n", NrestrictedPeanoInd);

    fclose(fp);

    fprintf(stderr, "task %d: finished reading aux data from first restart file.\n", ThisTask);
  } /* for ThisTask == 0: read first restart file & extract data */

  { /* distribute information from first restart file */
    if(ThisTask == 0)
    { fprintf(stdout, "task %d: trying to broadcast aux data from first restart file...\n", ThisTask); }

    MPI_Bcast(&number_of_restart_files , 1, MPI_INT   , 0, MPI_COMM_WORLD);
    MPI_Bcast(&rayTraceData.OmegaM     , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rayTraceData.bundleOrder, 1, MPI_LONG  , 0, MPI_COMM_WORLD);
    MPI_Bcast(&rayTraceData.rayOrder   , 1, MPI_LONG  , 0, MPI_COMM_WORLD);
    MPI_Bcast(&NbundleCells            , 1, MPI_LONG  , 0, MPI_COMM_WORLD);
    MPI_Bcast(&NrestrictedPeanoInd     , 1, MPI_LONG  , 0, MPI_COMM_WORLD);
    if(ThisTask == 0)
    { fprintf(stdout, "task %d: finished broadcast of aux data from first restart file\n", ThisTask); }
  } /* distribute information from first restart file */

  { /* set up mem structs to hold ray and aux data */
    fprintf(stdout, "task %d: allocating memory for raytracing data...\n", ThisTask);

    NraysPerBundleCell = (1ll) << (2*(rayTraceData.rayOrder - rayTraceData.bundleOrder));

    if(ThisTask != 0)
    {
      bundleCells                        = (HEALPixBundleCell*)checked_malloc(sizeof(HEALPixBundleCell)*NbundleCells);
      bundleCellsNest2RestrictedPeanoInd = (long*             )checked_malloc(sizeof(long             )*NbundleCells);
      bundleCellsRestrictedPeanoInd2Nest = (long*             )checked_malloc(sizeof(long             )*NbundleCells);
    }

    firstRestrictedPeanoIndTasks = (long*)   checked_malloc(sizeof(long) * number_of_restart_files);
    lastRestrictedPeanoIndTasks  = (long*)   checked_malloc(sizeof(long) * number_of_restart_files);

    map_pixel_sum_1              = (long*  ) checked_calloc(map_n_pixels, sizeof(long  ));
    map_pixel_sum_A00            = (double*) checked_calloc(map_n_pixels, sizeof(double));
    map_pixel_sum_A01            = (double*) checked_calloc(map_n_pixels, sizeof(double));
    map_pixel_sum_A10            = (double*) checked_calloc(map_n_pixels, sizeof(double));
    map_pixel_sum_A11            = (double*) checked_calloc(map_n_pixels, sizeof(double));
    map_pixel_sum_ra             = (double*) checked_calloc(map_n_pixels, sizeof(double));
    map_pixel_sum_dec            = (double*) checked_calloc(map_n_pixels, sizeof(double));

    alloc_rays_for_one_restart_file(number_of_restart_files);

    fprintf(stdout, "task %d: memory for raytracing data allocated.\n", ThisTask);
  } /* set up mem structs to hold ray and aux data */

  {/* set distance to cmb */
    wp_CMB = flat_LambdaCDM_line_of_sight_comoving_distance_for_redshift(z_CMB, rayTraceData.OmegaM);
//       fprintf(stderr, "debugging: wp_CMB = %f\n", wp_CMB);
  }/* set distance to cmb */

  // work filewise: (i) read rays, (ii) propagate rays, (iii) write rays
  const int restart_file_number_begin = item_number_begin_for_bin_number(ThisTask, NTasks, number_of_restart_files);
  const int restart_file_number_end   = item_number_end_for_bin_number  (ThisTask, NTasks, number_of_restart_files);
  fprintf(stderr, "task %d:trying to read rays from restart files %d to (excl.) %d...\n", ThisTask, restart_file_number_begin, restart_file_number_end);
  int restart_file_number;
  for(restart_file_number = restart_file_number_begin; restart_file_number < restart_file_number_end; restart_file_number++)
  {
    { /* read rays: */
      sprintf(output_filename, "/home/jderose/uscratch/BCC/Chinchilla/Herd/Chinchilla-1/calclens/restart.%d", restart_file_number);

      fprintf(stderr, "task %d: trying to read rays from restart file '%s'...\n", ThisTask, output_filename);

      fp = checked_fopen(output_filename, "r");

      checked_fread(&number_of_restart_files          , sizeof(int)              , (size_t) 1                      , fp);
      checked_fread(&fspd                             , sizeof(int)              , (size_t) 1                      , fp);
      checked_fread(&rtd                              , sizeof(RayTraceData)     , (size_t) 1                      , fp);

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

      checked_fread(&NbundleCells                     , sizeof(long)             , (size_t) 1                      , fp);
      checked_fread(bundleCells                       , sizeof(HEALPixBundleCell), (size_t) NbundleCells           , fp);
      checked_fread(bundleCellsNest2RestrictedPeanoInd, sizeof(long)             , (size_t) NbundleCells           , fp);
      checked_fread(bundleCellsRestrictedPeanoInd2Nest, sizeof(long)             , (size_t) NbundleCells           , fp);
      checked_fread(&NrestrictedPeanoInd              , sizeof(long)             , (size_t) 1                      , fp);
      checked_fread(firstRestrictedPeanoIndTasks      , sizeof(long)             , (size_t) number_of_restart_files, fp);
      checked_fread(lastRestrictedPeanoIndTasks       , sizeof(long)             , (size_t) number_of_restart_files, fp);
      checked_fread(&pbc                              , sizeof(int)              , (size_t) 1                      , fp);

      reset_bundle_cells();

      for(i=0; i < NbundleCells; ++i)
       if(ISSETBITFLAG(bundleCells[i].active, PRIMARY_BUNDLECELL))
          checked_fread(bundleCells[i].rays, (size_t) NraysPerBundleCell, sizeof(HEALPixRay), fp);

      fclose(fp);
    } /* read rays */

    { /* propagate rays: */
      fprintf(stderr, "task %d: propagating rays from restart file %d...\n", ThisTask, restart_file_number);

      set_plane_distances();


      for(bundleCellIndex = 0; bundleCellIndex < NbundleCells; ++bundleCellIndex)
        if(ISSETBITFLAG(bundleCells[bundleCellIndex].active, PRIMARY_BUNDLECELL))
        {
          for(i = 0; i < bundleCells[bundleCellIndex].Nrays; ++i)
          {
            bundleCells[bundleCellIndex].rays[i].alpha[0] = 0.;
            bundleCells[bundleCellIndex].rays[i].alpha[1] = 0.;
            bundleCells[bundleCellIndex].rays[i].U[0]     = 0.;
            bundleCells[bundleCellIndex].rays[i].U[1]     = 0.;
            bundleCells[bundleCellIndex].rays[i].U[2]     = 0.;
            bundleCells[bundleCellIndex].rays[i].U[3]     = 0.;
            bundleCells[bundleCellIndex].rays[i].phi      = 0.;
          }
          rayprop_sphere(wp_CMB, rayTraceData.planeRad, rayTraceData.planeRadMinus1, bundleCellIndex);
          updateLensMap(&bundleCells[bundleCellIndex], map_order, map_pixel_sum_1, map_pixel_sum_A00, map_pixel_sum_A01, map_pixel_sum_A10, map_pixel_sum_A11, map_pixel_sum_ra, map_pixel_sum_dec);
        }


    } /* propagate rays */

    if(write_restart_files_for_rays_at_cmb)
    { /* write rays: */
      fprintf(stderr, "task %d: wrting rays from restart file %d...\n", ThisTask, restart_file_number);
 
      sprintf(output_filename, "%s/restart_rays_at_cmb.%d", rayTraceData.OutputPath, restart_file_number);
      sprintf(temporary_output_filename, "%s.tmp", output_filename);


      fprintf(stderr, "debugging: output_filename = '%s'\n", output_filename);
      fprintf(stderr, "debugging: temporary_output_filename = '%s'\n", temporary_output_filename);

      if(overwrite_files_for_rays_at_cmb || !file_exists(output_filename))
      {
        fp = checked_fopen(temporary_output_filename, "w");

        checked_fwrite(&number_of_restart_files          , sizeof(int)              , (size_t) 1                      , fp);
        checked_fwrite(&fspd                             , sizeof(int)              , (size_t) 1                      , fp);
        checked_fwrite(&rtd                              , sizeof(RayTraceData)     , (size_t) 1                      , fp);

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

        checked_fwrite(&NbundleCells                     , sizeof(long)             , (size_t) 1                      , fp);
        checked_fwrite(bundleCells                       , sizeof(HEALPixBundleCell), (size_t) NbundleCells           , fp);
        checked_fwrite(bundleCellsNest2RestrictedPeanoInd, sizeof(long)             , (size_t) NbundleCells           , fp);
        checked_fwrite(bundleCellsRestrictedPeanoInd2Nest, sizeof(long)             , (size_t) NbundleCells           , fp);
        checked_fwrite(&NrestrictedPeanoInd              , sizeof(long)             , (size_t) 1                      , fp);
        checked_fwrite(firstRestrictedPeanoIndTasks      , sizeof(long)             , (size_t) number_of_restart_files, fp);
        checked_fwrite(lastRestrictedPeanoIndTasks       , sizeof(long)             , (size_t) number_of_restart_files, fp);
        checked_fwrite(&pbc                              , sizeof(int)              , (size_t) 1                      , fp);

        for(i=0; i < NbundleCells; ++i)
         if(ISSETBITFLAG(bundleCells[i].active, PRIMARY_BUNDLECELL))
            checked_fwrite(bundleCells[i].rays, sizeof(HEALPixRay), (size_t) NraysPerBundleCell, fp);

        fclose(fp);
        sprintf(sys,"mv %s %s",temporary_output_filename, output_filename);
        system(sys);
      }
    }  /* write rays */
  } /* loop over restart files */

  if(write_fits_maps_for_rays_at_cmb)
  { /* reduce and write map */
    if (ThisTask==0)
    { fprintf(stderr, "task %d: reducing rays...\n", ThisTask); }

     MPI_ReduceLensMap(map_pixel_sum_1, map_pixel_sum_A00, map_pixel_sum_A01, map_pixel_sum_A10,
                       map_pixel_sum_A11, map_pixel_sum_ra, map_pixel_sum_dec,
                       map_n_pixels, 0);

    if(ThisTask==0)
    { fprintf(stderr, "task %d: reduced rays.\n", ThisTask); }

    if (ThisTask==0)
    {
      sprintf(output_filename, "%s/CMB_convergence_%ld.fits", rayTraceData.OutputPath, map_n_side);
      fprintf(stderr, "task %d: writing convergence map to file '%s'...\n", ThisTask, output_filename);

      if(overwrite_files_for_rays_at_cmb || !file_exists(output_filename))
      {
        sprintf(temporary_output_filename, "%s.tmp", output_filename);

        /* fits_create_file() (in writeHealpixLensMap) seems not to like to overwrite existing files */
        if(file_exists(temporary_output_filename))
        { remove(temporary_output_filename);}

        float* convergence = (float*) checked_malloc (sizeof(float) * map_n_pixels);
        for(i = 0; i < map_n_pixels; i++)
        { convergence[i] = (map_pixel_sum_1[i] <= 0) ? 0.0 : 1.0 - 0.5 * (map_pixel_sum_A00[i] + map_pixel_sum_A11[i]) / map_pixel_sum_1[i]; }
        writeSingleFITSHEALPixLensMap(convergence, map_n_side, temporary_output_filename);
        free(convergence);

        sprintf(sys,"mv %s %s",temporary_output_filename, output_filename);
        system(sys);
      }

      sprintf(output_filename, "%s/CMB_rays_%ld.fits", rayTraceData.OutputPath, map_n_side);
      fprintf(stderr, "task %d: writing ray map to file '%s'...\n", ThisTask, output_filename);

      if(overwrite_files_for_rays_at_cmb || !file_exists(output_filename))
      {
        /* fits_create_file() (in writeLensMap) seems not to like to overwrite existing files */
        if(file_exists(temporary_output_filename))
        { remove(temporary_output_filename); }

        writeFITSHEALPixLensMap(map_pixel_sum_1, map_pixel_sum_A00, map_pixel_sum_A01, map_pixel_sum_A10,
                     map_pixel_sum_A11, map_pixel_sum_ra, map_pixel_sum_dec,
                     map_n_side, temporary_output_filename);

        sprintf(sys,"mv %s %s", temporary_output_filename, output_filename);
        system(sys);
      }
    }
  } /* reduce and write map */

  { /* free mem */
    fprintf(stderr, "task %d: freeing memory...\n", ThisTask);

    free(bundleCells);
    free(bundleCellsNest2RestrictedPeanoInd);
    free(bundleCellsRestrictedPeanoInd2Nest);
    free(firstRestrictedPeanoIndTasks);
    free(lastRestrictedPeanoIndTasks);

    free(map_pixel_sum_1  );
    free(map_pixel_sum_A00);
    free(map_pixel_sum_A01);
    free(map_pixel_sum_A10);
    free(map_pixel_sum_A11);
    free(map_pixel_sum_ra );
    free(map_pixel_sum_dec);

    free_rays();

    fprintf(stderr, "task %d: memory freed.\n", ThisTask);
  } /* free mem */
}
