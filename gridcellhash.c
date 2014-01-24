#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "inthash.h"
#include "gridcellhash.h"

//support functions
long id2ijk(long id, long N, long *i, long *j, long *k)
{
  //id = (i*N + j)*N + k
  long tmp;
  *k = id%N;
  tmp = (id-(*k))/N;
  *j = tmp%N;
  tmp = tmp - (*j);
  *i = tmp/N;
  
  if(id != ((*i)*N + (*j))*N + (*k))
    {
      fprintf(stderr,"%d: id = %ld, i,j,k = %ld|%ld|%ld\n",ThisTask,id,*i,*j,*k);
      fflush(stderr);
    }
  assert(id == ((*i)*N + (*j))*N + (*k));
}

long getIDhash(struct inthash **ih, long id)
{
  long ind = ih_getint64(*ih,id);
  if(ind == IH_INVALID)
    {
      if(NumGridCells == NumGridCellsAlloc)
	{
	  NumGridCellsAlloc += 10000;
	  GridCells = (GridCell*)realloc(GridCells,sizeof(GridCell)*NumGridCellsAlloc);
	  assert(GridCells != NULL);
	}
      ih_setint64(*ih,id,NumGridCells);
      NumGridCells += 1;
      ind = NumGridCells-1;
      GridCells[ind].id = id;
      GridCells[ind].val = 0.0;
    }
  assert(GridCells[ind].id == id);
  return ind;
}

int compGridCell(const void *a, const void *b) 
{
  GridCell *g1 = (GridCell*)a;
  GridCell *g2 = (GridCell*)b;
  if (g1->id == g2->id)
    return 0;
  else if(g1->id < g2->id)
    return -1;
  else
    return 1;
}

GridCellResults *init_gridcellhash(void)
{
  GridCellResults *gcr;
  gcr = (GridCellResults*)malloc(sizeof(GridCellResults));
  assert(gcr != NULL);
  gcr->ih = 
  return gcr;
}

void free_gridcellhash(GridCellResults *gcr)
{
  free(gcr->GridCells);
  free(gcr);
  gcr = NULL;
}
