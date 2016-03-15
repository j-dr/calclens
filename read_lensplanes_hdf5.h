#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <gsl/gsl_sort_long.h>
#include <gsl/gsl_rng.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#include "raytrace.h"

#ifndef _PARTIO_HDF5_
#define _PARTIO_HDF5_

void readRayTracingPlaneAtPeanoInds_HDF5(long planeNum, long HEALPixOrder, long *PeanoIndsToRead, long NumPeanoIndsToRead, Part **LCParts, long *NumLCParts);

#endif /* _PARTIO_HDF5_ */
