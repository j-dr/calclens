#include "raytrace.h"

#ifndef _TREECODE_
#define _TREECODE_

#define GET_TREE_STATS

typedef struct {
  long order;
  long nest;
  long child[4];
  long pstart;
  long np;
  double mass;
  double vec[3];
  double vecG[3];  
  //double cosMaxSL;
  //long alwaysOpen;
} TreeNode;

typedef struct {
  TreeNode *nodes;
  long Nnodes;
  long NnodesAlloc;
  double thetaS2;
  double thetaS;
  Part *parts;
  long Nparts;
  double nodeCosRCut[HEALPIX_UTILS_MAXORDER+1];
  double cosNodeArcSizesGeomLim[HEALPIX_UTILS_MAXORDER+1];
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
TreeData buildTree(Part *parts, long Nparts, double thetaSplit);
void destroyTree(TreeData td);
TreeWalkData computePotentialForceShearDirectSummation(double vec[3], TreeData td);
TreeWalkData computePotentialForceShearTree(double vec[3], double BHCrit, TreeData td);

#endif /* _TREECODE_ */
