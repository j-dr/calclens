//////////////////////////////////////////////////////////////////////
// author(s): Stefan Hilbert
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// header guard:
//////////////////////////////////////////////////////////////////////
#ifndef HEADER_GUARD_FOR_CHECKED_ALLOC_H
#define HEADER_GUARD_FOR_CHECKED_ALLOC_H

//////////////////////////////////////////////////////////////////////
// C libraries:
//////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

//////////////////////////////////////////////////////////////////////
// aux function: checked_fignore
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline void*
checked_malloc(size_t size_)
{
  void* ptr_;
        
  ptr_  = malloc(size_); 
  if(ptr_ == NULL)
  {
    fprintf(stderr,"task %d: failed to allocate %ld bytes of memory in checked malloc!\n", ThisTask, size_);
    MPI_Abort(MPI_COMM_WORLD, 777);
  }
  return ptr_;
}


//////////////////////////////////////////////////////////////////////
// aux function: checked_fignore
//--------------------------------------------------------------------
// why is C std lib so inconsistent with the order of arguments?
// (compare arguement orer in calloc and fread)
//////////////////////////////////////////////////////////////////////
static inline void*
checked_calloc(size_t num_, size_t size_)
{
  void* ptr_;
        
  ptr_  = calloc(num_, size_); 
  if(ptr_ == NULL)
  {
    fprintf(stderr,"task %d: failed to allocate %ld bytes of memory in checked calloc!\n", ThisTask, size_);
    MPI_Abort(MPI_COMM_WORLD, 777);
  }
  return ptr_;
}

//////////////////////////////////////////////////////////////////////
// end of header file
//////////////////////////////////////////////////////////////////////
#endif /* header guard */