#include "inthash.h"

#ifdef MEMWATCH
#include "memwatch.h"
#endif

#ifdef USEMEMCHECK
#include <memcheck.h>
#endif

#ifdef DMALLOC
#include <dmalloc.h>
#endif

#ifndef _GCHASH_
#define _GCHASH_

typedef struct {
  long id;
  double val;
} GridCell;

typedef struct {
  GridCell *GridCells;
  long NumGridCellsAlloc;
  long NumGridCells;
  struct inthash *ih;
} GridCellResults;

/* in gridcellhash.c */
long id2ijk(long id, long N, long *i, long *j, long *k);
long getIDhash(struct inthash **ih, long id, GridCellResults *gcr);
int compGridCell(const void *a, const void *b);
GridCellResults *init_gridcellresults(void);
void free_gridcellresults(GridCellResults *gcr);

#endif /* _GCHASH_ */
