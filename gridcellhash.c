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

long getid_gridcellhash(GridCellResults *gcr, long id)
{
  long ind = ih_getint64(gcr->ih,id);
  if(ind == IH_INVALID)
    {
      if(gcr->NumGridCells == gcr->NumGridCellsAlloc)
	{
	  gcr->NumGridCellsAlloc += 10000;
	  gcr->GridCells = (GridCell*)realloc(gcr->GridCells,sizeof(GridCell)*(gcr->NumGridCellsAlloc));
	  assert(gcr->GridCells != NULL);
	}
      ih_setint64(gcr->ih,id,gcr->NumGridCells);
      gcr->NumGridCells += 1;
      ind = gcr->NumGridCells-1;
      gcr->GridCells[ind].id = id;
      gcr->GridCells[ind].val = 0.0;
    }
  assert(gcr->GridCells[ind].id == id);
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

void minmem_gridcellhash(GridCellResults *gcr)
{
  gcr->GridCells = (GridCell*)realloc(gcr->GridCells,sizeof(GridCell)*(gcr->NumGridCells));
  assert(gcr->GridCells != NULL);
  gcr->NumGridCellsAlloc = gcr->NumGridCells;
}

GridCellResults *init_gridcellhash(void)
{
  GridCellResults *gcr;
  gcr = (GridCellResults*)malloc(sizeof(GridCellResults));
  assert(gcr != NULL);
  gcr->ih = new_inthash();
  return gcr;
}

void free_gridcellhash(GridCellResults *gcr)
{
  free(gcr->GridCells);
  free_inthash(gcr->ih);
  free(gcr);
  gcr = NULL;
}
