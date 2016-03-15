#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <gsl/gsl_sort_long.h>
#include <gsl/gsl_rng.h>

#include "raytrace.h"

#ifndef _PARTIO_PIXLC_
#define _PARTIO_PIXLC_

void readRayTracingPlaneAtPeanoInds_pixLC(long planeNum, long HEALPixOrder, long *PeanoIndsToRead, long NumPeanoIndsToRead, Part **LCParts, long *NumLCParts);

#endif /* _PARTIO_PIXLC_ */
