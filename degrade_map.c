#include <mpi.h>
#include <assert.h>
#include "fitsio.h"
#include "raytrace.h"
#include "healpix_utils.h"

void printerror( int status)
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

void updateMap(HEALPixBundleCell *bundleCell, long order_,
	       long *nest, double *A00, double *A01, double *A10,
	       double *A11, double *ra, double *dec)
{
  int i,j;
  long lpix;
  double theta, phi;

  for (j=0;j<(*bundleCell).Nrays;j++)
    {
      lpix = lower_nest((*bundleCell).rays[j].nest,
			rayTraceData.rayOrder,
			order_);
	  
      vec2radec((*bundleCell).rays[j].n, &phi, &theta);
      A00[lpix] += (*bundleCell).rays[j].A[0];
      A01[lpix] += (*bundleCell).rays[j].A[1];
      A10[lpix] += (*bundleCell).rays[j].A[2];
      A11[lpix] += (*bundleCell).rays[j].A[3];
      ra[lpix]  += phi;
      dec[lpix] += theta;
      nest[lpix]++;
    }
}

void reduceMap(long *nest, double *A00, double *A01, double *A10,
	       double *A11, double *ra, double *dec, long npix)
{
  MPI_Reduce(MPI_IN_PLACE, nest, npix, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(MPI_IN_PLACE, A00, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(MPI_IN_PLACE, A01, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(MPI_IN_PLACE, A10, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(MPI_IN_PLACE, A11, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(MPI_IN_PLACE, ra, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(MPI_IN_PLACE, dec, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (ThisTask==0)
    {
      for (i=0;i<npix;i++)
	{
	  A00[i] /= nest[i];
	  A01[i] /= nest[i];
	  A10[i] /= nest[i];
	  A11[i] /= nest[i];
	  ra[i] /= nest[i];
	  dec[i] /= nest[i];
	  nest[i] = i;
	}
    }
}

void writeMap(long *nest, double *A00, double *A01, double *A10,
	      double *A11, double *ra, double *dec, long npix,
	      char *filename)
{
  fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
  int status, hdutype;
  long firstrow, firstelem;

  int tfields   = 7;       /* table will have 7 columns */

  char extname[] = "CMB_lensing_map";           /* extension name */

  /* define the name, datatype, and physical units for the 3 columns */
  char *ttype[] = { "NEST", "A00", "A01", "A10", "A11", "RA", "DEC"};
  char *tform[] = { "K","D","D","D","D","D","D"};
  char *tunit[] = { "\0","\0","\0","\0","\0","DEG","DEG",};

  status=0;

  /* open the FITS file containing a primary array and an ASCII table */
  if ( fits_open_file(&fptr, filename, READWRITE, &status) )
    printerror( status );

  if ( fits_movabs_hdu(fptr, 2, &hdutype, &status) ) /* move to 2nd HDU */
    printerror( status );

  /* append a new empty binary table onto the FITS file */
  if ( fits_create_tbl( fptr, BINARY_TBL, npix, tfields, ttype, tform,
			tunit, extname, &status) )
    printerror( status );

  firstrow  = 1;  /* first row in table to write   */
  firstelem = 1;  /* first element in row  (ignored in ASCII tables) */

  /* write names to the first column (character strings) */
  /* write diameters to the second column (longs) */
  /* write density to the third column (floats) */

  fits_write_col(fptr, TLONG, 1, firstrow, firstelem, npix, nest,
		 &status);
  fits_write_col(fptr, TDOUBLE, 2, firstrow, firstelem, npix, A00,
		 &status);
  fits_write_col(fptr, TDOUBLE, 2, firstrow, firstelem, npix, A01,
		 &status);
  fits_write_col(fptr, TDOUBLE, 2, firstrow, firstelem, npix, A10,
		 &status);
  fits_write_col(fptr, TDOUBLE, 2, firstrow, firstelem, npix, A11,
		 &status);
  fits_write_col(fptr, TDOUBLE, 2, firstrow, firstelem, npix, ra,
		 &status);
  fits_write_col(fptr, TDOUBLE, 2, firstrow, firstelem, npix, dec,
		 &status);  

  if ( fits_close_file(fptr, &status) )       /* close the FITS file */
    printerror( status );
  return;
}
