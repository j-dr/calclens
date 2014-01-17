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
#define PROFILETAG_INITEND_LOADBAL 1
#define PROFILETAG_MAKE_LC         2
#define PROFILETAG_MAKE_LENSPLANES 3

#define NUM_PROFILE_TAGS           4

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
