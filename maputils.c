//////////////////////////////////////////////////////////////////////
// author(s): Joe DeRose, Stefan Hilbert, ...
//////////////////////////////////////////////////////////////////////

#include <mpi.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "fitsio.h"
#include "raytrace.h"
#include "healpix_utils.h"
#include <gsl/gsl_errno.h>
// #include <gsl/gsl_math.h>
#include <gsl/gsl_sf_hyperg.h>

#include "checked_alloc.h"


static inline double
flat_LambdaCDM_los_comoving_distance_for_redshift_(double z_, double Omega_matter_, double Hubble_distance_ )
  {
    const double Omega_Lambda_ = 1.  - Omega_matter_;
    const double inv_omlf_     = 1. / (Omega_Lambda_ + (1. + z_) * (1. + z_) * (1. + z_) * Omega_matter_);
    const double result_       =
      (0.99 < Omega_Lambda_ * inv_omlf_) ?
      Hubble_distance_ * z_ :
      Hubble_distance_ * (  2. * gsl_sf_hyperg_2F1(1./2., 1., 7./6., Omega_Lambda_            )
				+                        - 2. * gsl_sf_hyperg_2F1(1./2., 1., 7./6., Omega_Lambda_ * inv_omlf_) * sqrt(inv_omlf_) * (1. + z_));
    fprintf(stderr, "z : %f\n", z_);
    fprintf(stderr, "omegal : %f\n", Omega_Lambda_);
    fprintf(stderr, "inv_omlf : %f\n", inv_omlf_);    
    fprintf(stderr, "Result : %f\n", result_);
    return result_;

   }
static inline double
flat_LambdaCDM_los_comoving_distance_for_redshift(double z_, double Omega_matter_)
{ return flat_LambdaCDM_los_comoving_distance_for_redshift_(z_, Omega_matter_, 2997.92458 /* value in Mpc, change if using different units */); }


//////////////////////////////////////////////////////////////////////
// aux function: printerror
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline void
printerror(int status)
{
  /*****************************************************/
  /* Print out cfitsio error messages and exit program */
  /*****************************************************/

  if (status)
  {
    fits_report_error(stderr, status); /* print error report */
    exit( status );    /* terminate the program, returning error status */
  }
  return;
}

//////////////////////////////////////////////////////////////////////
// function: getNMaps
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
void
getNMaps(long *NMaps)
{
  FILE *fp;
  fp = fopen(rayTraceData.MapRedshiftList,"r");
  assert(fp != NULL);
  //Get number of Maps
  *NMaps = fnumlines(fp);
  fprintf(stderr, "Number of maps %ld\n", *NMaps);
  fclose(fp);
}

//////////////////////////////////////////////////////////////////////
// function: readMapRedshifts
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
void
readMapRedshifts(double* z_map,  long NMaps)
{
  int i;
  float mz;
  char mzstr[MAX_FILENAME];
  FILE *fp;
  
  fp = fopen(rayTraceData.MapRedshiftList,"r");
  assert(fp != NULL);

  for(i=0;i<NMaps;++i)
  {
    fgets(mzstr,MAX_FILENAME,fp);
    mz = atof(mzstr);
    assert(mz!=0.0);
    z_map[i] = mz;
  }

  fclose(fp);
}

void
getMapLensPlaneNums(int* lp_map, long NMaps)
{
  int i;
  double* z_map;
  double  r;
  double binL = (rayTraceData.maxComvDistance)/((double) (rayTraceData.NumLensPlanes));
  fprintf(stderr, "binL : %f\n", binL);
  z_map = (double*)malloc(sizeof(double)*NMaps);

  readMapRedshifts(z_map, NMaps);

  for (i=0; i<NMaps; i++)
  {
    fprintf(stderr, "Map redshift : %f\n", z_map[i]);
    r = flat_LambdaCDM_los_comoving_distance_for_redshift(
          z_map[i], rayTraceData.OmegaM);
    fprintf(stderr, "Map comoving distance : %f\n", r);
    fprintf(stderr, "Lens plane : %f\n", round(r/binL));
    lp_map[i] = (int) round(r/binL);
  }
}

//////////////////////////////////////////////////////////////////////
// function: updateLensMap
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
void
updateLensMap(HEALPixBundleCell *bundleCell, const long map_order,
              long* map_pixel_sum_1, double *map_pixel_sum_A00, double *map_pixel_sum_A01, double *map_pixel_sum_A10,
              double *map_pixel_sum_A11, double *map_pixel_sum_ra, double *map_pixel_sum_dec)
{
  int i;
  long lpix;
  double theta, phi;

  for (i=0;i<(*bundleCell).Nrays;i++)
  {
    lpix = lower_nest((*bundleCell).rays[i].nest, rayTraceData.rayOrder, map_order);
    //lpix = vec2nest((*bundleCell).rays[i].n, map_order);

    vec2radec((*bundleCell).rays[i].n, &phi, &theta);
    map_pixel_sum_1   [lpix] ++;
    map_pixel_sum_A00 [lpix] += (*bundleCell).rays[i].A[0];
    map_pixel_sum_A01 [lpix] += (*bundleCell).rays[i].A[1];
    map_pixel_sum_A10 [lpix] += (*bundleCell).rays[i].A[2];
    map_pixel_sum_A11 [lpix] += (*bundleCell).rays[i].A[3];
    map_pixel_sum_ra  [lpix] += phi;
    map_pixel_sum_dec [lpix] += theta;

//     if ( i == 0)
//     {
//       fprintf(stderr, "lpix: %ld\n", lpix);
//       fprintf(stderr, "task %d: debugging: map_pixel_sum_1  [lpix] =  %ld,...\n", ThisTask, map_pixel_sum_1   [lpix]);
//       fprintf(stderr, "task %d: debugging: map_pixel_sum_A00[lpix] =  %f,...\n" , ThisTask, map_pixel_sum_A00 [lpix]);
//       fprintf(stderr, "task %d: debugging: map_pixel_sum_A01[lpix] =  %f,...\n" , ThisTask, map_pixel_sum_A01 [lpix]);
//       fprintf(stderr, "task %d: debugging: map_pixel_sum_A10[lpix] =  %f,...\n" , ThisTask, map_pixel_sum_A10 [lpix]);
//       fprintf(stderr, "task %d: debugging: map_pixel_sum_A11[lpix] =  %f,...\n" , ThisTask, map_pixel_sum_A11 [lpix]);
//       fprintf(stderr, "task %d: debugging: map_pixel_sum_ra [lpix] =  %f,...\n" , ThisTask, map_pixel_sum_ra  [lpix]);
//       fprintf(stderr, "task %d: debugging: map_pixel_sum_dec[lpix] =  %f,...\n" , ThisTask, map_pixel_sum_dec [lpix]);
//     }
  }
}


//////////////////////////////////////////////////////////////////////
// function: MPI_ReduceLensMap
//--------------------------------------------------------------------
// now using MPI_IN_PLACE reduction
//////////////////////////////////////////////////////////////////////
void
MPI_ReduceLensMap(long *map_pixel_sum_1, double *map_pixel_sum_A00, double *map_pixel_sum_A01, double *map_pixel_sum_A10, double *map_pixel_sum_A11, double *map_pixel_sum_ra, double *map_pixel_sum_dec,
                  const long map_n_pixels, const int root)
{
  if(NTasks > 1)
  {
    MPI_Reduce((ThisTask == root) ? MPI_IN_PLACE : map_pixel_sum_1  ,  map_pixel_sum_1  , map_n_pixels, MPI_LONG  , MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce((ThisTask == root) ? MPI_IN_PLACE : map_pixel_sum_A00,  map_pixel_sum_A00, map_n_pixels, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce((ThisTask == root) ? MPI_IN_PLACE : map_pixel_sum_A01,  map_pixel_sum_A01, map_n_pixels, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce((ThisTask == root) ? MPI_IN_PLACE : map_pixel_sum_A10,  map_pixel_sum_A10, map_n_pixels, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce((ThisTask == root) ? MPI_IN_PLACE : map_pixel_sum_A11,  map_pixel_sum_A11, map_n_pixels, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce((ThisTask == root) ? MPI_IN_PLACE : map_pixel_sum_ra ,  map_pixel_sum_ra , map_n_pixels, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce((ThisTask == root) ? MPI_IN_PLACE : map_pixel_sum_dec,  map_pixel_sum_dec, map_n_pixels, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
  }
}


//////////////////////////////////////////////////////////////////////
// function: writeFITSHEALPixLensMap
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
void writeFITSHEALPixLensMap(long *map_pixel_sum_1, double *map_pixel_sum_A00, double *map_pixel_sum_A01, double *map_pixel_sum_A10,
                  double *map_pixel_sum_A11, double *map_pixel_sum_ra, double *map_pixel_sum_dec,
                  long map_n_side, const char *filename)
{
  fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
  int status;
  long firstrow, firstelem;

  long map_n_pixels = 12 * map_n_side * map_n_side;
  long firstpix     = 0;
  long lastpix      = map_n_pixels;

  int tfields   = 8;       /* table will have 8 columns */

  char extname[] = "CMB_lensing_map";           /* extension name */

  /* define the name, datatype, and physical units for the 3 columns */
  char *ttype[] = { "NEST_IDX", "N_RAYS", "A00", "A01", "A10", "A11", "ra" , "dec"};
  char *tform[] = { "1J"      , "1J"    , "1D" , "1D" , "1D" , "1D" , "1D" , "1D" };
  char *tunit[] = { "\0"      , "\0"    , "\0" , "\0" , "\0" , "\0" , "DEG", "DEG"};

  status=0;
  fprintf(stderr, "Creating File\n");

  fits_create_file (&fptr, filename, &status);
  printerror( status );

  /* append a new empty binary table onto the FITS file */
  if ( fits_create_tbl( fptr, BINARY_TBL, map_n_pixels, tfields, ttype, tform, tunit, extname, &status) )
  { printerror( status ); }

  fits_write_key    (fptr, TSTRING, "PIXTYPE" , "HEALPIX"  , "HEALPIX Pixelisation"                        , &status);
  fits_write_key    (fptr, TSTRING, "ORDERING", "NESTED  " , "Pixel ordering scheme, either RING or NESTED", &status);
  fits_write_key    (fptr, TLONG  , "NSIDE"   , &map_n_side, "Resolution parameter for HEALPIX"            , &status);
  fits_write_key    (fptr, TLONG  , "FIRSTPIX", &firstpix  , ""                                            , &status);
  fits_write_key    (fptr, TLONG  , "LASTPIX" , &lastpix   , ""                                            , &status);
  fits_write_key    (fptr, TSTRING, "COORDSYS", "C       " , "Pixelisation coordinate system"              , &status);
  fits_write_comment(fptr, "G = Galactic, E = ecliptic, C = celestial = equatorial"                        , &status);
  { printerror( status ); }

  void  * raw_temp = checked_malloc(((sizeof(long ) < sizeof(double)) ? sizeof(double) : sizeof(long)) * map_n_pixels);
  long  * ltemp    = (long  *) raw_temp;
  double* dtemp    = (double*) raw_temp;
  int     i;

  firstrow  = 1;  /* first row in table to write   */
  firstelem = 1;  /* first element in row  (ignored in ASCII tables) */

  fprintf(stderr, "Writing cols\n");
  for (i = 0; i < map_n_pixels; i++) { ltemp[i] = i ; }
  fits_write_col(fptr, TLONG  , 1, firstrow, firstelem, map_n_pixels, ltemp, &status);

  fprintf(stderr, "Writing cols\n");
  for (i = 0; i < map_n_pixels; i++) { ltemp[i] = map_pixel_sum_1[i]; }
  fits_write_col(fptr, TLONG  , 2, firstrow, firstelem, map_n_pixels, ltemp, &status);

  fprintf(stderr, "Writing cols\n");
  for (i = 0; i < map_n_pixels; i++) { dtemp[i] = map_pixel_sum_1[i] <= 0 ? 0. : map_pixel_sum_A00[i] / map_pixel_sum_1[i]; }
  fits_write_col(fptr, TDOUBLE, 3, firstrow, firstelem, map_n_pixels, dtemp, &status);

  fprintf(stderr, "Writing cols\n");
  for (i = 0; i < map_n_pixels; i++) { dtemp[i] = map_pixel_sum_1[i] <= 0 ? 0. :  map_pixel_sum_A01[i] / map_pixel_sum_1[i]; }
  fits_write_col(fptr, TDOUBLE, 4, firstrow, firstelem, map_n_pixels, dtemp, &status);

  fprintf(stderr, "Writing cols\n");
  for (i = 0; i < map_n_pixels; i++) { dtemp[i] = map_pixel_sum_1[i] <= 0 ? 0. :  map_pixel_sum_A10[i] / map_pixel_sum_1[i]; }
  fits_write_col(fptr, TDOUBLE, 5, firstrow, firstelem, map_n_pixels, dtemp, &status);

  fprintf(stderr, "Writing cols\n");
  for (i = 0; i < map_n_pixels; i++) { dtemp[i] = map_pixel_sum_1[i] <= 0 ? 0. :  map_pixel_sum_A11[i] / map_pixel_sum_1[i]; }
  fits_write_col(fptr, TDOUBLE, 6, firstrow, firstelem, map_n_pixels, dtemp, &status);

  fprintf(stderr, "Writing cols\n");
  for (i = 0; i < map_n_pixels; i++) { dtemp[i] = map_pixel_sum_1[i] <= 0 ? 0. :  map_pixel_sum_ra [i] / map_pixel_sum_1[i]; }
  fits_write_col(fptr, TDOUBLE, 7, firstrow, firstelem, map_n_pixels, dtemp, &status);

  fprintf(stderr, "Writing cols\n");
  for (i = 0; i < map_n_pixels; i++) { dtemp[i] = map_pixel_sum_1[i] <= 0 ? 0. :  map_pixel_sum_dec[i] / map_pixel_sum_1[i]; }
  fits_write_col(fptr, TDOUBLE, 8, firstrow, firstelem, map_n_pixels, dtemp, &status);

  free(raw_temp);

  if (fits_close_file(fptr, &status))       /* close the FITS file */
  { printerror( status ); }

  return;
}


//////////////////////////////////////////////////////////////////////
// function: writeSingleFITSHEALPixLensMap
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
void
writeSingleFITSHEALPixLensMap(const float *signal, long nside, const char *filename)
{
  fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
  int status=0, hdutype;

  long naxes[] = {0,0};

  char order[9];                 /* HEALPix ordering */
  char *ttype[] = { "SIGNAL" };
  char *tform[] = { "1E" };
  char *tunit[] = { " " };
  char CoordinateSystemParameterString[9];

  long firstpix = 0;
  long lastpix  = 12 * nside * nside;

  /* create new FITS file */
  fits_create_file  (&fptr, filename, &status);
  fits_create_img   (fptr, SHORT_IMG, 0, naxes, &status);
  fits_write_date   (fptr, &status);
  fits_movabs_hdu   (fptr, 1, &hdutype, &status);
  fits_create_tbl   (fptr, BINARY_TBL, 12L*nside*nside, 1, ttype, tform, tunit, "BINTABLE", &status);

  fits_write_key    (fptr, TSTRING, "PIXTYPE" , "HEALPIX" , "HEALPIX Pixelisation"                        , &status);
  fits_write_key    (fptr, TSTRING, "ORDERING", "NESTED  ", "Pixel ordering scheme, either RING or NESTED", &status);
  fits_write_key    (fptr, TLONG  , "NSIDE"   , &nside    , "Resolution parameter for HEALPIX"            , &status);
  fits_write_key    (fptr, TLONG  , "FIRSTPIX", &firstpix , ""                                            , &status);
  fits_write_key    (fptr, TLONG  , "LASTPIX" , &lastpix  , ""                                            , &status);
  fits_write_key    (fptr, TSTRING, "COORDSYS", "C       ", "Pixelisation coordinate system"              , &status);
  fits_write_comment(fptr, "G = Galactic, E = ecliptic, C = celestial = equatorial"                       , &status);

  fits_write_col(fptr, TFLOAT, 1, 1, 1, 12 * nside * nside, (void *)signal, &status);
  fits_close_file(fptr, &status);
  printerror(status);
}
