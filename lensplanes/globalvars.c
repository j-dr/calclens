#include "raytrace.h"

/* global vars are defined here except those in 
   healpix_shtrans.c 
*/

const char *ProfileTagNames[] = {"TotalTime","InitEndLoadBal","MakeLC","MakeLensPlanes"};

RayTraceData rayTraceData;                               /* global struct with all vars from config file */
int ThisTask;                                            /* this task's rank in MPI_COMM_WORLD */
int NTasks;                                              /* number of tasks in MPI_COMM_WORLD */
