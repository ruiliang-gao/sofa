/*
 *  gProximity Library.
 *  
 *  
 *  Copyright (C) 2010 University of North Carolina at Chapel Hill.
 *  All rights reserved.
 *  
 *  Permission to use, copy, modify, and distribute this software and its
 *  documentation for educational, research, and non-profit purposes, without
 *  fee, and without a written agreement is hereby granted, provided that the
 *  above copyright notice, this paragraph, and the following four paragraphs
 *  appear in all copies.
 *  
 *  Permission to incorporate this software into commercial products may be
 *  obtained by contacting the University of North Carolina at Chapel Hill.
 *  
 *  This software program and documentation are copyrighted by the University of
 *  North Carolina at Chapel Hill. The software program and documentation are
 *  supplied "as is", without any accompanying services from the University of
 *  North Carolina at Chapel Hill or the authors. The University of North
 *  Carolina at Chapel Hill and the authors do not warrant that the operation of
 *  the program will be uninterrupted or error-free. The end-user understands
 *  that the program was developed for research purposes and is advised not to
 *  rely exclusively on the program for any reason.
 *  
 *  IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL OR ITS
 *  EMPLOYEES OR THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 *  SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 *  ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE
 *  UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL OR THE AUTHORS HAVE BEEN ADVISED
 *  OF THE POSSIBILITY OF SUCH DAMAGE.
 *  
 *  THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND THE AUTHORS SPECIFICALLY
 *  DISCLAIM ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND ANY
 *  STATUTORY WARRANTY OF NON-INFRINGEMENT. THE SOFTWARE PROVIDED HEREUNDER IS
 *  ON AN "AS IS" BASIS, AND THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL AND
 *  THE AUTHORS HAVE NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
 *  ENHANCEMENTS, OR MODIFICATIONS.
 *  
 *  Please send all BUG REPORTS to:
 *  
 *  geom@cs.unc.edu
 *  
 *  The authors may be contacted via:
 *  
 *  Christian Lauterbach, Qi Mo, Jia Pan and Dinesh Manocha
 *  Dept. of Computer Science
 *  Frederick P. Brooks Jr. Computer Science Bldg.
 *  3175 University of N.C.
 *  Chapel Hill, N.C. 27599-3175
 *  United States of America
 *  
 *  http://gamma.cs.unc.edu/GPUCOL/
 *  
 */

#ifndef __BVH_KERNELS_H__
#define __BVH_KERNELS_H__

#include "cuda_vertex.h"
#include "cuda_aabb.h"
#include "cuda_obb.h"
#include "cuda_aabbnode.h"
#include "cuda_aabbsplit.h"
#include "cuda_vectors.h"
#include "cuda_defs.h"
#include "cuda_workqueue.h"


/*
template <int nSplitThreads, int childNum>
__device__ __inline__ void storeSplit(const unsigned int thid, const unsigned int myOffset, const unsigned int numLeft, const AABBSplit &inSplit, float *localBoxes,
                                      AABBSplit *outputSplits, int *outputSplitMask, AABBNode *tree, const int *triIDs, AABBNode sharedNodeBuffer[2], const int nTrisPerLeaf);

template <int nSplitThreads, int childNum>
__device__ __inline__ void storeSplitLocal(const unsigned int thid, const unsigned int myOffset, const unsigned int numLeft, const AABBSplit &inSplit, float *localBoxes,
        WorkQueueItem *workQueue, int &workQueueIdx, const int *triIDs, const int *globalTriIDs, AABBNode &outputNode, const int nTrisPerLeaf);
*/

__global__ void generateAABBsIndexed(float2 *boxes, int *triIDs, const GPUVertex *vertices, uint3* triangles, unsigned int *zCode,
                                     const unsigned int nTris, const unsigned int array_offset, const float3 sceneBB_min,
                                     const float3 sceneBB_max, const int nVerts = 0);


__global__ void AABBtoOBBbyLevel(AABBNode *treeIn, OBBNode *treeOut, unsigned int startOffset, unsigned int nNodes, const GPUVertex *vertices, const uint3 *triangles, const int vertexOffset = 0);

__global__ void copyGPUVertex(GPUVertex* v1, float* v2, unsigned int nVertices, unsigned int nTargetVertices, bool useMin, unsigned int d);

#endif //__BVH_KERNELS_H__
