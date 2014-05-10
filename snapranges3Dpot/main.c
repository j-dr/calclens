#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <assert.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sort_long.h>
#include <gsl/gsl_ieee_utils.h>

#include "raytrace.h"
#include "../lgadgetio.h"

typedef struct {
  char fname[MAX_FILENAME];
  double a;
  double chi;
} NbodySnap;

static void read_snaps(NbodySnap **snaps, long *Nsnaps);

int main(int argc, char **argv)
{
  /* vars */
  char name[MAX_FILENAME];
  
  read_config(argv[1]);
      
  /* error check  */
  if(ThisTask == 0)
    {
      fprintf(stderr,"----------------------------------------------------------------------------------------------------------------------\n");
      fprintf(stderr,"running snapranges3Dpot.\n");
      fprintf(stderr,"Nplanes = %ld, omegam = %f, max_comvd = %f \n",rayTraceData.NumLensPlanes,rayTraceData.OmegaM,rayTraceData.maxComvDistance);
      fprintf(stderr,"----------------------------------------------------------------------------------------------------------------------\n");
      fprintf(stderr,"\n");
    }
  
  int plane;
  long i;
  long mysnap;
  double dsnap;
  double dchi = rayTraceData.maxComvDistance/rayTraceData.NumLensPlanes;
  NbodySnap *snaps;
  long Nsnaps;
  
  read_snaps(&snaps,&Nsnaps);
  
  fprintf(stdout,"#plane snapshot rmin rmax\n");
  for(plane=0;plane<rayTraceData.NumLensPlanes;++plane) {
    rayTraceData.planeRad = plane*dchi + dchi*0.5;
    
    //get closest snap
    mysnap = 0;
    dsnap = fabs(snaps[mysnap].chi-rayTraceData.planeRad);
    for(i=0;i<Nsnaps;++i) {
      if(fabs(snaps[i].chi-rayTraceData.planeRad) < dsnap) {
	mysnap = i;
	dsnap = fabs(snaps[i].chi-rayTraceData.planeRad);
      }
    }
    
    fprintf(stdout,"%d %s %lf %lf\n",plane,snaps[mysnap].fname,plane*dchi,(plane+1.0)*dchi);
  }
  
  free(snaps);
  
  return 0;
}

static void read_snaps(NbodySnap **snaps, long *Nsnaps) {
  char line[MAX_FILENAME];
  FILE *fp;
  long n = 0;
  char fname[MAX_FILENAME];
  long nl;

  fp = fopen(rayTraceData.ThreeDPotSnapList,"r");
  assert(fp != NULL);
  while(fgets(line,1024,fp) != NULL) {
    if(line[0] == '#')
      continue;
    ++n;
  }
  fclose(fp);
  
  *snaps = (NbodySnap*)malloc(sizeof(NbodySnap)*n);
  assert((*snaps) != NULL);
  *Nsnaps = n;
  
  n = 0;
  fp = fopen(rayTraceData.ThreeDPotSnapList,"r");
  assert(fp != NULL);
  while(fgets(line,1024,fp) != NULL) {
    if(line[0] == '#')
      continue;
    assert(n < (*Nsnaps));
    nl = strlen(line);
    line[nl-1] = '\0';
    sprintf((*snaps)[n].fname,"%s",line);
    ++n;
  }
  fclose(fp);
  
  for(n=0;n<(*Nsnaps);++n) {
    sprintf(fname,"%s.0",(*snaps)[n].fname);
    (*snaps)[n].a = get_scale_factor_LGADGET(fname);
    (*snaps)[n].chi = comvdist((*snaps)[n].a);
  }
}
