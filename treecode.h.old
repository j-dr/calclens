#include "raytrace.h"

#ifndef _TREECODE_
#define _TREECODE_

typedef struct {
  long order;
  long nest;
  long partInd;
  long down;
  long over;
  double mass;
  double vec[3];
  double vecG[3];
  double cosMaxSL;
  long alwaysOpen;
} TreeNode;

#define NumTreeExpFactTable 1000
typedef struct {
  TreeNode *nodes;
  long Nnodes;
  long NnodesAlloc;
  long *links;
  long Nparts;
  Part *parts;
  double sigmaSoftening;
  double thetaSplit2;
  double thetaSplit;
  long baseOrder;
  long NpixBaseOrder;
  long firstNode;
  double nodeCosRCut[HEALPIX_UTILS_MAXORDER+1];
  double cosNodeArcSizesGeomLim[HEALPIX_UTILS_MAXORDER+1];
  double nodeArcSizes[HEALPIX_UTILS_MAXORDER+1];
  double nodeArcSizes2[HEALPIX_UTILS_MAXORDER+1];
  double cosrTable[NumTreeExpFactTable];
  double expAlphaTable[NumTreeExpFactTable];
  double expGammaTable[NumTreeExpFactTable];
  double potTable[NumTreeExpFactTable];
  double dcosr;
  double minCosr;
} _TreeData,*TreeData;

typedef struct {
  long NumInteractTreeWalk;
  long NumInteractTreeWalkNode;
  long Nempty;
  double pot;
  double alpha[2];
  double U[4];
} TreeWalkData;

//in treecode.c
TreeData buildTree(Part *parts, long Nparts, double thetaSplit, long baseOrder);
void destroyTree(TreeData td);
TreeWalkData computePotentialForceShearDirectSummation(double vec[3], TreeData td);
TreeWalkData computePotentialForceShearTree(double vec[3], double BHCrit, TreeData td);

#endif /* _TREECODE_ */
