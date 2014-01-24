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

long getid_gchash(GridCellHash *gch, long id)
{
  long ind = ih_getint64(gch->ih,id);
  if(ind == IH_INVALID)
    {
      if(gch->NumGridCells == gch->NumGridCellsAlloc)
	{
	  gch->NumGridCellsAlloc += 10000;
	  gch->GridCells = (GridCell*)realloc(gch->GridCells,sizeof(GridCell)*(gch->NumGridCellsAlloc));
	  assert(gch->GridCells != NULL);
	}
      ih_setint64(gch->ih,id,gch->NumGridCells);
      gch->NumGridCells += 1;
      ind = gch->NumGridCells-1;
      gch->GridCells[ind].id = id;
      gch->GridCells[ind].val = 0.0;
    }
  assert(gch->GridCells[ind].id == id);
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

void minmem_gchash(GridCellHash *gch)
{
  gch->GridCells = (GridCell*)realloc(gch->GridCells,sizeof(GridCell)*(gch->NumGridCells));
  assert(gch->GridCells != NULL);
  gch->NumGridCellsAlloc = gch->NumGridCells;
}

GridCellHash *init_gchash(void)
{
  GridCellHash *gch;
  gch = (GridCellHash*)malloc(sizeof(GridCellHash));
  assert(gch != NULL);
  gch->ih = new_inthash();
  return gch;
}

void free_gchash(GridCellHash *gch)
{
  free(gch->GridCells);
  free_inthash(gch->ih);
  free(gch);
  gch = NULL;
}
