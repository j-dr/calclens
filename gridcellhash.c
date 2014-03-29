#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "inthash.h"
#include "gridcellhash.h"

void id2ijk(long id, long N, long *i, long *j, long *k)
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
      fprintf(stderr,"id = %ld, i,j,k = %ld|%ld|%ld, != %ld\n",id,*i,*j,*k,((*i)*N + (*j))*N + (*k));
      fflush(stderr);
    }
  assert(id == ((*i)*N + (*j))*N + (*k));
}

long getonlyid_gchash(GridCellHash *gch, long id)
{
  long ind = ih_getint64(gch->ih,id);
  if(ind != IH_INVALID)
    assert(gch->GridCells[ind].id == id);
  return ind;
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

void destroyhash_gchash(GridCellHash *gch)
{
  if(gch->ih != NULL)
    free_inthash(gch->ih);
  gch->ih = NULL;
}

void rebuildhash_gchash(GridCellHash *gch)
{
  long i;
  if(gch->ih != NULL)
    free_inthash(gch->ih);
  gch->ih = new_inthash();
  for(i=0;i<gch->NumGridCells;++i)
    {
      assert(ih_getint64(gch->ih,gch->GridCells[i].id) == IH_INVALID);
      ih_setint64(gch->ih,gch->GridCells[i].id,i);
    }
}

void sortcells_gchash(GridCellHash *gch)
{
  if(gch->NumGridCells > 0)
    {
      qsort(gch->GridCells,gch->NumGridCells,sizeof(GridCell),compGridCell);
      rebuildhash_gchash(gch);
    }
}

GridCellHash *init_gchash(void)
{
  GridCellHash *gch;
  gch = (GridCellHash*)malloc(sizeof(GridCellHash));
  assert(gch != NULL);
  gch->NumGridCells = 0;
  gch->NumGridCellsAlloc = 0;
  gch->GridCells = NULL;
  gch->ih = new_inthash();
  return gch;
}

void free_gchash(GridCellHash *gch)
{
  free(gch->GridCells);
  if(gch->ih != NULL)
    free_inthash(gch->ih);
  free(gch);
}
