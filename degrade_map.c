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

void updateMap(HEALPixBundleCell *bundleCell, const long order_,
	       long *nest, double *A00, double *A01, double *A10,
	       double *A11, double *ra, double *dec)
{
  int i;
  long lpix;
  double theta, phi;

  for (i=0;i<(*bundleCell).Nrays;i++)
    {
      
      //lpix = lower_nest((*bundleCell).rays[i].nest,
      //		rayTraceData.rayOrder,
      //		order_);
      lpix = vec2nest((*bundleCell).rays[i].n, order_);
      
      if (i==0)
	{
	  fprintf(stderr, "lpix: %ld\n", lpix);
	}
      vec2radec((*bundleCell).rays[i].n, &phi, &theta);
      A00[lpix] += (*bundleCell).rays[i].A[0];
      A01[lpix] += (*bundleCell).rays[i].A[1];
      A10[lpix] += (*bundleCell).rays[i].A[2];
      A11[lpix] += (*bundleCell).rays[i].A[3];
      ra[lpix]  += phi;
      dec[lpix] += theta;
      nest[lpix]++;
    }
}

void reduceMap(long *nest, double *A00, double *A01, double *A10,
	       double *A11, double *ra, double *dec,
	       const long npix)
{
  int i;
  double *dtemp;
  long *ltemp;
  
  dtemp = (double*)malloc(sizeof(double)*npix);
  ltemp = (long*)malloc(sizeof(long)*npix);

  if (dtemp==NULL || ltemp==NULL)
    {
      fprintf(stderr, "Memory allocation failure\n");
    }

  MPI_Reduce(nest, ltemp, npix, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (ThisTask==0)
    {
      memcpy(nest, ltemp, npix*sizeof(long));
    }
  
  MPI_Reduce(A00, dtemp, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (ThisTask==0)
    {
      memcpy(A00, dtemp, npix*sizeof(double));
    }
  
  MPI_Reduce(A01, dtemp, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (ThisTask==0)
    {
      memcpy(A01, dtemp, npix*sizeof(double));
    }
  
  MPI_Reduce(A10, dtemp, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (ThisTask==0)
    {
      memcpy(A10, dtemp, npix*sizeof(double));
    }

  MPI_Reduce(A11, dtemp, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (ThisTask==0)
    {
      memcpy(A11, dtemp, npix*sizeof(double));
    }

  MPI_Reduce(ra, dtemp, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (ThisTask==0)
    {
      memcpy(ra, dtemp, npix*sizeof(double));
    }
  
  MPI_Reduce(dec, dtemp, npix, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (ThisTask==0)
    {
      memcpy(dec, dtemp, npix*sizeof(double));
    }
  
  fprintf(stderr, "Averaging arrays\n");
  if (ThisTask==0)
    {
      for (i=0;i<npix;i++)
	{
	  if (i==(npix-1))
	    {
	      fprintf(stderr, "Last pixel: %ld\n", npix);
	    }
	  A00[i] /= nest[i];
	  A01[i] /= nest[i];
	  A10[i] /= nest[i];
	  A11[i] /= nest[i];
	  ra[i] /= nest[i];
	  dec[i] /= nest[i];
	  nest[i] = i;
	}
    }
  fprintf(stderr, "Done averaging arrays\n");
}

void writeMap(long *nest, double *A00, double *A01, double *A10,
	      double *A11, double *ra, double *dec, const long npix,
	      const char *filename)
{
  fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
  int status, hdutype;
  long firstrow, firstelem;

  int tfields   = 7;       /* table will have 7 columns */

  char extname[] = "CMB_lensing_map";           /* extension name */

  /* define the name, datatype, and physical units for the 3 columns */
  char *ttype[] = { "NEST", "A00", "A01", "A10", "A11", "RA", "DEC"};
  char *tform[] = { "1J","1D","1D","1D","1D","1D","1D"};
  char *tunit[] = { "\0","\0","\0","\0","\0","DEG","DEG",};

  status=0;
  fprintf(stderr, "Creating File\n");
  
  fits_create_file (&fptr, filename, &status);
  printerror( status );

  /* append a new empty binary table onto the FITS file */
  if ( fits_create_tbl( fptr, BINARY_TBL, npix, tfields, ttype, tform,
			tunit, extname, &status) )
    printerror( status );

  firstrow  = 1;  /* first row in table to write   */
  firstelem = 1;  /* first element in row  (ignored in ASCII tables) */

  fprintf(stderr, "Writing cols\n");
  fits_write_col(fptr, TLONG, 1, firstrow, firstelem, npix, nest,
		 &status);
  fprintf(stderr, "Writing cols\n");
  fits_write_col(fptr, TDOUBLE, 2, firstrow, firstelem, npix, A00,
		 &status);
  fprintf(stderr, "Writing cols\n");
  fits_write_col(fptr, TDOUBLE, 3, firstrow, firstelem, npix, A01,
		 &status);
  fprintf(stderr, "Writing cols\n");
  fits_write_col(fptr, TDOUBLE, 4, firstrow, firstelem, npix, A10,
		 &status);
  fprintf(stderr, "Writing cols\n");
  fits_write_col(fptr, TDOUBLE, 5, firstrow, firstelem, npix, A11,
		 &status);
  fprintf(stderr, "Writing cols\n");
  fits_write_col(fptr, TDOUBLE, 6, firstrow, firstelem, npix, ra,
		 &status);
  fprintf(stderr, "Writing cols\n");
  fits_write_col(fptr, TDOUBLE, 7, firstrow, firstelem, npix, dec,
		 &status);  

  if ( fits_close_file(fptr, &status) )       /* close the FITS file */
    printerror( status );
  return;
}
