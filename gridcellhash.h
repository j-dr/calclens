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
} GridCellHash;

/* in gridcellhash.c */
long id2ijk(long id, long N, long *i, long *j, long *k);
int compGridCell(const void *a, const void *b);
long getid_gchash(GridCellHash *gch, long id);
void minmem_gchash(GridCellHash *gch);
GridCellHash *init_gchash(void);
void free_gchash(GridCellHash *gch);
void sortcells_gchash(GridCellHash *gch);
void destroyhash_gchash(GridCellHash *gch);
void rebuildhash_gchash(GridCellHash *gch);

#endif /* _GCHASH_ */
