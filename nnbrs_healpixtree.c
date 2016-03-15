#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <fftw3.h>
#include <mpi.h>
#include <hdf5.h>
#include <gsl/gsl_math.h>

#include "raytrace.h"

#define MIN_NUM_RAYS_PER_HEALPIXTREENODE 40

static int intersectHEALPixTreeNodeDisc(HEALPixTreeNode *node, double nodeSize, double n[3], double radius)
{
  double cosr = node->n[0]*n[0] + node->n[1]*n[1] + node->n[2]*n[2];
  double cosLim;
  
  //FIXME: used a fudge factor here!
  if(radius + nodeSize*2.0 < M_PI)
    cosLim = cos(radius + nodeSize*2.0);
  else
    cosLim = -1.0;
  
  if(cosr >= cosLim)
    return 1;
  else
    return 0;
}

long nnbrsHEALPixTree(double n[3], double radius, double cmvRad, HEALPixRay *rays, HEALPixTreeData *td, NNbrData **NNbrs, long *maxNumNNbrs)
{
  long currNode,currRay;
  long NumNNbrs;
  long NumN = 0,NumR = 0,NumNp = 0;
  NNbrData *tmpNNbrs;
  double cosrad,nlen,nnorm[3];
  double cosradius;
  
  if(radius <= M_PI)
    cosradius = cos(radius);
  else
    cosradius = -1.0;
  
  //fudge for floating point rounding - try to make sure a point will be nnbr of itself 
  cosradius = cosradius - 1.11e-14;
  
  nlen = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  nnorm[0] = n[0]/nlen;
  nnorm[1] = n[1]/nlen;
  nnorm[2] = n[2]/nlen;
  
  if(*maxNumNNbrs == 0)
    {
      *maxNumNNbrs = 1000;
      *NNbrs = (NNbrData*)malloc(sizeof(NNbrData)*(*maxNumNNbrs));
      assert(*NNbrs != NULL);
    }
  
  NumNNbrs = 0;
  currNode = 0;

  while(currNode >= 0)
    {
      ++NumN;
      
#ifdef DEBUG
#if DEBUG_LEVEL > 2
      fprintf(stderr,"%05d: NumN,NumR,NumNp = %ld|%ld|%ld, currNode = %ld, order,nest = %ld|%ld, over,down = %ld|%ld, startRay = %ld\n",ThisTask,
	      NumN,NumR,NumNp,currNode,td->nodes[currNode].order,td->nodes[currNode].nest,td->nodes[currNode].over,td->nodes[currNode].down,td->nodes[currNode].startRay);
#endif
#endif
      
      if(intersectHEALPixTreeNodeDisc(td->nodes+currNode,td->nodeArcSize[td->nodes[currNode].order],nnorm,radius))
	{
	  if(td->nodes[currNode].startRay == -1 && td->nodes[currNode].down == -1)
	    currNode = td->nodes[currNode].over;
	  else if(td->nodes[currNode].startRay == -1 && td->nodes[currNode].down >= 0)
	    currNode = td->nodes[currNode].down;
	  else
	    {
	      ++NumNp;
	      currRay = td->nodes[currNode].startRay;
	      while(currRay >= 0)
		{
		  ++NumR;
		  cosrad = (rays[currRay].n[0]*nnorm[0] + rays[currRay].n[1]*nnorm[1] + rays[currRay].n[2]*nnorm[2])/cmvRad;
		  
		  if(cosrad >= cosradius)
		    {
		      
		      if(NumNNbrs >= (*maxNumNNbrs))
			{
			  (*maxNumNNbrs) += 1000;
			  tmpNNbrs = (NNbrData*)realloc(*NNbrs,sizeof(NNbrData)*(*maxNumNNbrs));
			  assert(tmpNNbrs != NULL);
			  *NNbrs = tmpNNbrs;
			}
		      
		      (*NNbrs)[NumNNbrs].ind = currRay;
		      (*NNbrs)[NumNNbrs].cosrad = cosrad;
		      ++NumNNbrs;
		    }  
		  
		  currRay = td->links[currRay];
		}
	      
	      currNode = td->nodes[currNode].over;
	    }
	}
      else
	currNode = td->nodes[currNode].over;
    }
  
#ifdef DEBUG
#if DEBUG_LEVEL > 2
  fprintf(stderr,"%05d: NumNodesChecked,NumRaysChecked,NumNodesOpened = %ld|%ld|%ld\n",ThisTask,NumN,NumR,NumNp);
#endif
#endif
  
  return NumNNbrs;
}

HEALPixTreeData *buildHEALPixTree(long Nrays, HEALPixRay *rays)
{
  long i;
  HEALPixTreeData *td;
  long *nodeStack;
  long NumNodesInStack,NumNodeStackAlloc;
  long next,curr,nest,currNode;
  long NumHEALPixTreeNodesAlloc;
  long *nestInds,shift;
  HEALPixTreeNode *tmpNode;
  long *tmpLong;
  HEALPixTreeNode child[4];
  long numRaysInChild[4],NumNodesToAdd,NumNodesToRefine,startLink,childLocs[4];
    
  //init base nodes and links to parts
  td = (HEALPixTreeData*)malloc(sizeof(HEALPixTreeData));
  assert(td != NULL);
  td->links = (long*)malloc(sizeof(long)*Nrays);
  assert(td->links != NULL);
  NumHEALPixTreeNodesAlloc = 1000;
  td->nodes = (HEALPixTreeNode*)malloc(sizeof(HEALPixTreeNode)*NumHEALPixTreeNodesAlloc);
  assert(td->nodes != NULL);
  nestInds = (long*)malloc(sizeof(long)*Nrays);
  assert(nestInds != NULL);
  NumNodeStackAlloc = 1000;
  nodeStack = (long*)malloc(sizeof(long)*NumNodeStackAlloc);
  assert(nodeStack != NULL);
  
  for(i=0;i<=HEALPIX_UTILS_MAXORDER;++i)
    td->nodeArcSize[i] = sqrt(4.0*M_PI/order2npix(i));
  
  td->NumNodes = 12;
  for(i=0;i<12;++i)
    {
      td->nodes[i].order = 0;
      td->nodes[i].nest = i;
      nest2vec(i,td->nodes[i].n,0l);
      td->nodes[i].over = i+1;
      td->nodes[i].down = -1;
      td->nodes[i].startRay = -1;
    }
  td->nodes[11].over = -1;
  
  for(i=0;i<Nrays;++i)
    {
      td->links[i] = -1;
      nestInds[i] = vec2nest(rays[i].n,HEALPIX_UTILS_MAXORDER);
    }
  
  shift = 2*HEALPIX_UTILS_MAXORDER;
  for(i=0;i<Nrays;++i)
    {
      nest = (nestInds[i] >> shift);
      td->links[i] = td->nodes[nest].startRay;
      td->nodes[nest].startRay = i;
    }
  
  //refine tree
  NumNodesInStack = 12;
  for(i=0;i<12;++i)
    nodeStack[i] = 11-i;
  while(NumNodesInStack > 0)
    {
      currNode = nodeStack[NumNodesInStack-1];
      --NumNodesInStack;
      
#ifdef DEBUG
#if DEBUG_LEVEL > 2
      fprintf(stderr,"%d: currNode = %ld, order,nest = %ld|%ld, down,over = %ld|%ld, startRay = %ld, theta,phi = %f|%f\n",ThisTask,currNode,
	      td->nodes[currNode].order,td->nodes[currNode].nest,td->nodes[currNode].down,td->nodes[currNode].over,
	      td->nodes[currNode].startRay,td->nodes[currNode].theta/M_PI*180.0,td->nodes[currNode].phi/M_PI*180.0);
#endif
#endif
      
      //make children
      for(i=0;i<4;++i)
	{
	  child[i].order = td->nodes[currNode].order + 1l;
	  child[i].nest = td->nodes[currNode].nest*4l + i;
	  nest2vec(child[i].nest,child[i].n,child[i].order);
	  child[i].down = -1;
	  child[i].startRay = -1;
	  numRaysInChild[i] = 0;
	}
      
      //split up rays
      shift = 2*(HEALPIX_UTILS_MAXORDER - child[0].order);
      NumNodesToAdd = 0;
      curr = td->nodes[currNode].startRay;
      while(curr >= 0)
	{
	  ++NumNodesToAdd;
	  nest = (nestInds[curr] >> shift);
	  for(i=0;i<4;++i)
	    if(nest == child[i].nest)
	      {
		++(numRaysInChild[i]);
		next = td->links[curr];
		td->links[curr] = child[i].startRay;
		child[i].startRay = curr;
		curr = next;
		break;
	      }
	}
      td->nodes[currNode].startRay = -1;
      assert(NumNodesToAdd == numRaysInChild[0]+numRaysInChild[1]+numRaysInChild[2]+numRaysInChild[3]);
      
      //make sure have room to add nodes to nodes array and stack
      NumNodesToAdd = 0;
      for(i=0;i<4;++i)
	if(numRaysInChild[i] > 0)
	  ++NumNodesToAdd;
      
      if(NumNodesToAdd + td->NumNodes >= NumHEALPixTreeNodesAlloc)
	{
	  NumHEALPixTreeNodesAlloc += 1000;
	  tmpNode = (HEALPixTreeNode*)realloc(td->nodes,sizeof(HEALPixTreeNode)*NumHEALPixTreeNodesAlloc);
	  assert(tmpNode != NULL);
	  td->nodes = tmpNode;
	}
      
      NumNodesToRefine = 0;
      for(i=0;i<4;++i)
	if(numRaysInChild[i] > MIN_NUM_RAYS_PER_HEALPIXTREENODE && child[i].order < HEALPIX_UTILS_MAXORDER)
	  ++NumNodesToRefine;
      
      if(NumNodesToRefine + NumNodesInStack >= NumNodeStackAlloc)
	{
	  NumNodeStackAlloc += 1000;
	  tmpLong = (long*)realloc(nodeStack,sizeof(long)*NumNodeStackAlloc);
	  assert(tmpLong != NULL);
	  nodeStack = tmpLong;
	}
      
      //link up nodes
      startLink = 1;
      curr = -1;
      for(i=0;i<4;++i)
	{
	  if(numRaysInChild[i] > 0)
	    {
	      td->nodes[td->NumNodes] = child[i];
	      childLocs[i] = td->NumNodes;
	      if(startLink)
		{
		  startLink = 0;
		  td->nodes[currNode].down = td->NumNodes;
		}
	      else
		td->nodes[curr].over = td->NumNodes;
	      curr = td->NumNodes;
	      ++(td->NumNodes);
	    }
	}
      if(curr >= 0)
	td->nodes[curr].over = td->nodes[currNode].over;
      
      //add nodes to stack
      for(i=3;i>=0;--i)
	{
	  if(numRaysInChild[i] > MIN_NUM_RAYS_PER_HEALPIXTREENODE && child[i].order < HEALPIX_UTILS_MAXORDER)
	    {
	      nodeStack[NumNodesInStack] = childLocs[i];
	      ++NumNodesInStack;
	    }
	}
    }
  
  if(td->NumNodes < NumHEALPixTreeNodesAlloc)
    {
      tmpNode = (HEALPixTreeNode*)realloc(td->nodes,sizeof(HEALPixTreeNode)*(td->NumNodes));
      assert(tmpNode != NULL);
      td->nodes = tmpNode;
    }
  
  free(nestInds);
  free(nodeStack);
  
#ifdef DEBUG
#if DEBUG_LEVEL > 1
  fprintf(stderr,"%d: size of tree = %lf MB, size of tree nodes = %lf MB\n",ThisTask,
	  ((double) (td->NumNodes*sizeof(HEALPixTreeNode)+Nrays*sizeof(long)))/1024.0/1024.0,((double) (td->NumNodes*sizeof(HEALPixTreeNode)))/1024.0/1024.0);
#endif
#endif
  
  return td;
}

void destroyHEALPixTree(HEALPixTreeData *td)
{
  free(td->nodes);
  free(td->links);
  free(td);
  td = NULL;
}

#undef MIN_NUM_RAYS_PER_HEALPIXTREENODE
