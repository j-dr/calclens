//////////////////////////////////////////////////////////////////////
// author(s): Stefan Hilbert
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// header guard:
//////////////////////////////////////////////////////////////////////
#ifndef HEADER_GUARD_FOR_CHECKED_IO_H
#define HEADER_GUARD_FOR_CHECKED_IO_H

//////////////////////////////////////////////////////////////////////
// C libraries:
//////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <mpi.h>

//////////////////////////////////////////////////////////////////////
// aux function: file_exists
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline bool
file_exists( const char* filename_)
{ return (0 == access(filename_, F_OK)); }


//////////////////////////////////////////////////////////////////////
// aux function: checked_fopen
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline FILE*
checked_fopen( const char* filename_, const char* mode_)
{
  FILE* fp_;
 
  fp_ = fopen(filename_, mode_);
  if(!fp_)
  {
    fprintf(stderr,"task %d: could not open file '%s'!\n",ThisTask, filename_);
    MPI_Abort(MPI_COMM_WORLD, 777);
  }
  return fp_;
}


//////////////////////////////////////////////////////////////////////
// aux function: checked_fread
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline size_t
checked_fread(void* p_, size_t size_, size_t n_items_, FILE* fp_)
{
  size_t nrw_;
  nrw_ = fread(p_, size_, n_items_, fp_);
  if(nrw_ != n_items_)
  {
    fprintf(stderr,"task %d: error in checked read!\n", ThisTask);
    MPI_Abort(MPI_COMM_WORLD, 777);
  }
  return nrw_;
}


//////////////////////////////////////////////////////////////////////
// aux function: checked_fread
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline size_t
checked_fwrite(void* p_, size_t size_, size_t n_items_, FILE* fp_)
{
  size_t nrw_;
  nrw_ = fwrite(p_, size_, n_items_, fp_);
  if(nrw_ != n_items_)
  {
    fprintf(stderr,"task %d: error in checked write!\n",ThisTask);
    MPI_Abort(MPI_COMM_WORLD,777);
  }
  return nrw_;
}


//////////////////////////////////////////////////////////////////////
// aux function: checked_fignore
//--------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////
static inline size_t
checked_fignore(void *p, size_t size_, size_t n_items_, FILE* fp_)
{
  size_t n_bytes_to_ignore_  = size_ * n_items_;
        
  fprintf(stderr, "debugging: n_bytes_to_ignore = %ld\n", n_bytes_to_ignore_);
      
  if(!fseek(fp_, n_bytes_to_ignore_, SEEK_CUR))
  {
    fprintf(stderr,"task %d: error in checked ignore!\n", ThisTask);
    MPI_Abort(MPI_COMM_WORLD,777);
  }
  return n_items_;
}

//////////////////////////////////////////////////////////////////////
// end of header file
//////////////////////////////////////////////////////////////////////
#endif /* header guard */
