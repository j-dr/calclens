#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef MEMWATCH
#include "memwatch.h"
#endif

#ifdef USEMEMCHECK
#include <memcheck.h>
#endif

#ifdef DMALLOC
#include <dmalloc.h>
#endif

#ifndef _PROFILE_
#define _PROFILE_

/*define to enable a time series output*/
//#define PROFILE_TIMESERIES

#define PROFILETAG_TOTTIME         0
#define PROFILETAG_STEPTIME        1
#define PROFILETAG_SHT             2
#define PROFILETAG_SHTSOLVE        3
#define PROFILETAG_MAPSUFFLE       4
#define PROFILETAG_MG              5
#define PROFILETAG_MGSOLVE         6
#define PROFILETAG_RAYIO           7
#define PROFILETAG_PARTIO          8
#define PROFILETAG_RAYPROP         9
#define PROFILETAG_GRIDSEARCH      10
#define PROFILETAG_GALIO           11
#define PROFILETAG_RAYBUFF         12           
#define PROFILETAG_RESTART         13
#define PROFILETAG_INITEND_LOADBAL 14

#define PROFILETAG_GRIDSEARCH_GALMOVE                15  //this tag is a subset of GRIDSEARCH
#define PROFILETAG_GRIDSEARCH_GALGRIDSEARCH          16  //this tag is a subset of GRIDSEARCH
#define PROFILETAG_GRIDSEARCH_IMAGEGALIO             17  //this tag is NOT a subset of GRIDSEARCH, but is a subset of GALIO

#define PROFILETAG_GRIDKAPPADENS  18
#define PROFILETAG_TREEBUILD      19
#define PROFILETAG_TREEWALK       20

#define NUM_PROFILE_TAGS          21

void logProfileTag(int tag);
void printProfileInfo(const char name[], const char *ProfileTagNames[]);
double getTimeProfileTag(int tag);
double getTotTimeProfileTag(int tag);
void resetProfiler(void);
void printStepTimesProfileTags(FILE *fp, long stepNum, const char *ProfileTagNames[]);

#ifdef PROFILE_TIMESERIES
double getTimeProfileTagSeries(int tag);
double getPrevTimeProfileTagSeries(int tag, long NumPrev);
#endif

#endif /* _PROFILE_ */
