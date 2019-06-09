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
 
#ifndef __CUDA_INTERSECT_TREE_H_
#define __CUDA_INTERSECT_TREE_H_

#include <cuda_runtime.h>
#include <stdio.h>

#include <sm_20_atomic_functions.h>

#include "cuda_prefix.h"
#include "cuda_defs.h"
#include "cuda_intersect_nodes.h"
#include "cuda_intersect_tritri.h"

#include "cuda_collision.h"

#include "ObbTreeGPU_LinearAlgebra.cuh"

template <int nTotalProcessors>
static __device__ __inline__ void callAbort(unsigned int *workQueueCounter, const int threadID)
{
	if(threadID == 0)
		atomicInc(workQueueCounter, nTotalProcessors);
}


//#define GPROXIMITY_DEBUG_TRAVERSE_TREE
//#define GPROXIMITY_TRAVERSETREE_STORE_OVERLAPPING_OBB_PAIRS
// nWorkQueueInitItems should be smaller than workQueueCapacity, e.g. 1/2 workQueueCapacity
// CollisionTag: set the tag for collision. (for milestone 0, for ccd -1)
template <class TreeNode, class BV, int workQueueCapacity, int nWorkQueueInitItems, int nThreads>
__global__ void traverseTree(const TreeNode* tree_object1, const GPUVertex* vertex_object1, const uint3* tri_object1,
							 const TreeNode* tree_object2, const GPUVertex* vertex_object2, const uint3* tri_object2,
							 int2* workQueues, unsigned int* workQueueCounts, unsigned int* idleCount, const unsigned int globalWorkQueueCapacity,
                             int2* outputList, unsigned int* outputListIdx, bool* outputLeafList,
#ifdef GPROXIMITY_TRAVERSETREE_STORE_OVERLAPPING_OBB_PAIRS
                             int2* outputOBBList, unsigned int* obbListIdx, const unsigned int obbListCapacity,
#endif
                             Matrix3x3_d* d_modelTransform1, Matrix3x3_d* d_modelTransform2, float3* d_trVector1, float3* d_trVector2, double alarmDistance)
{
	__shared__ int workQueueIdx;
	__shared__ int globalWorkQueueIdx;
	__shared__ int2 localWorkQueue[workQueueCapacity];
	__shared__ unsigned int wantToAbort;

	const int blockOffset = gridDim.x * blockIdx.y + blockIdx.x;
	const int threadOffset = threadIdx.x;

#ifdef GPROXIMITY_DEBUG_TRAVERSE_TREE
            __syncthreads();

            printf("traverseTree called: blockOffset = %i, threadOffset = %i\n", blockOffset, threadOffset);
#endif

	// read local counter
	if(threadOffset == 0)
	{
		globalWorkQueueIdx = workQueueCounts[blockOffset];
		workQueueIdx = min(nWorkQueueInitItems, globalWorkQueueIdx);
		globalWorkQueueIdx -= workQueueIdx;
	}

	__syncthreads();

	if(workQueueIdx == 0)
	{
#ifdef GPROXIMITY_DEBUG_TRAVERSE_TREE
        __syncthreads();

        printf(" callAbort: workQueueIdx = 0, idleCount = %i, threadOffset = %i\n", idleCount, threadOffset);
#endif
		callAbort<workQueueCapacity>(idleCount, threadOffset);
		return;
	}


	{
		int2* globalQueue = &workQueues[blockOffset * globalWorkQueueCapacity + globalWorkQueueIdx];
		int queueOffset = threadOffset;
		while(queueOffset < workQueueIdx * 3)
		{
			((int*)localWorkQueue)[queueOffset] = ((int*)globalQueue)[queueOffset];
			queueOffset += nThreads;
		}
	}

	__syncthreads();

	while(workQueueIdx > 0)
	{
		int2 work_item;
		int nActive = min(nThreads, workQueueIdx);

		work_item.x = -1;
		if(threadOffset < workQueueIdx)
		{
			work_item = localWorkQueue[workQueueIdx - nActive + threadOffset];
		}
		__syncthreads();

		if(threadOffset == 0)
		{
			workQueueIdx -= nActive;
		}
		__syncthreads();

		if(work_item.x >= 0)
		{
			TreeNode node1 = tree_object1[work_item.x];
			TreeNode node2 = tree_object2[work_item.y];

            (*d_modelTransform1).m_row[0].w = (*d_trVector1).x; (*d_modelTransform1).m_row[1].w = (*d_trVector1).y; (*d_modelTransform1).m_row[2].w = (*d_trVector1).z;
            (*d_modelTransform2).m_row[0].w = (*d_trVector2).x; (*d_modelTransform2).m_row[1].w = (*d_trVector2).y; (*d_modelTransform2).m_row[2].w = (*d_trVector2).z;

            Matrix3x3_d d_modelTransform1_tr = getTranspose(*d_modelTransform1);
            Matrix3x3_d d_modelTransform2_tr = getTranspose(*d_modelTransform2);

#if 0
            Matrix3x3_d x_form, b_rel_a, a_rel_b;
            TINV_MUL_T(b_rel_a, *d_modelTransform1, *d_modelTransform2);
            TRANSFORM_INV(a_rel_b,  b_rel_a);

            Matrix3x3_d obbARelTop;
            obbARelTop.m_row[0].x = node1.bbox.axis1.x; obbARelTop.m_row[1].x = node1.bbox.axis2.x; obbARelTop.m_row[2].x = node1.bbox.axis3.x;
            obbARelTop.m_row[0].y = node1.bbox.axis1.y; obbARelTop.m_row[1].y = node1.bbox.axis2.y; obbARelTop.m_row[2].y = node1.bbox.axis3.y;
            obbARelTop.m_row[0].z = node1.bbox.axis1.z; obbARelTop.m_row[1].z = node1.bbox.axis2.z; obbARelTop.m_row[2].z = node1.bbox.axis3.z;

            Matrix3x3_d obbBRelTop;
            obbBRelTop.m_row[0].x = node2.bbox.axis1.x; obbBRelTop.m_row[1].x = node2.bbox.axis2.x; obbBRelTop.m_row[2].x = node2.bbox.axis3.x;
            obbBRelTop.m_row[0].y = node2.bbox.axis1.y; obbBRelTop.m_row[1].y = node2.bbox.axis2.y; obbBRelTop.m_row[2].y = node2.bbox.axis3.y;
            obbBRelTop.m_row[0].z = node2.bbox.axis1.z; obbBRelTop.m_row[1].z = node2.bbox.axis2.z; obbBRelTop.m_row[2].z = node2.bbox.axis3.z;

            obbARelTop.m_row[0].w = node1.bbox.center.x; obbARelTop.m_row[1].w = node1.bbox.center.y; obbARelTop.m_row[2].w = node1.bbox.center.z;
            obbBRelTop.m_row[0].w = node2.bbox.center.x; obbBRelTop.m_row[1].w = node2.bbox.center.y; obbBRelTop.m_row[2].w = node2.bbox.center.z;

            Matrix3x3_d obbARelTop_tr = getTranspose(obbARelTop);
            Matrix3x3_d obbBRelTop_tr = getTranspose(obbBRelTop);

            TR_MULT(x_form, b_rel_a, obbBRelTop);
            bool disjoint = intersectOBB(node1.bbox, node2.bbox, obbARelTop, x_form);
#endif

            {
                node1.bbox.center = mtMul1(*d_modelTransform1, node1.bbox.center);
                node1.bbox.center.x += d_trVector1->x;
                node1.bbox.center.y += d_trVector1->y;
                node1.bbox.center.z += d_trVector1->z;

                node1.bbox.axis1 = mtMul1(*d_modelTransform1, node1.bbox.axis1);
                node1.bbox.axis2 = mtMul1(*d_modelTransform1, node1.bbox.axis2);
                node1.bbox.axis3 = mtMul1(*d_modelTransform1, node1.bbox.axis3);
            }

            {
                node2.bbox.center = mtMul1(*d_modelTransform2, node2.bbox.center);
                node2.bbox.center.x += d_trVector2->x;
                node2.bbox.center.y += d_trVector2->y;
                node2.bbox.center.z += d_trVector2->z;

                node2.bbox.axis1 = mtMul1(*d_modelTransform2, node2.bbox.axis1);
                node2.bbox.axis2 = mtMul1(*d_modelTransform2, node2.bbox.axis2);
                node2.bbox.axis3 = mtMul1(*d_modelTransform2, node2.bbox.axis3);
            }

#ifdef GPROXIMITY_DEBUG_TRAVERSE_TREE
            __syncthreads();

            printf("Intersect OBB %i - %i: center1 = %f,%f,%f; center2 = %f,%f,%f    \n \
                    --> model1 translation = %f,%f,%f; model2 translation = %f,%f,%f \n \
                    --> model1 rotation = %f,%f,%f\n%f,%f,%f\n%f,%f,%f\n \
                    --> model2 rotation = %f,%f,%f\n%f,%f,%f\n%f,%f,%f\n",
                   node1.bbox.idx, node2.bbox.idx,
                   node1.bbox.center.x, node1.bbox.center.y, node1.bbox.center.z,
                   node2.bbox.center.x, node2.bbox.center.y, node2.bbox.center.z,
                   d_trVector1->x, d_trVector1->y, d_trVector1->z, d_trVector2->x, d_trVector2->y, d_trVector2->z,
                   d_modelTransform1->m_row[0].x, d_modelTransform1->m_row[0].y, d_modelTransform1->m_row[0].z,
                   d_modelTransform1->m_row[1].x, d_modelTransform1->m_row[1].y, d_modelTransform1->m_row[1].z,
                   d_modelTransform1->m_row[2].x, d_modelTransform1->m_row[2].y, d_modelTransform1->m_row[2].z,
                   d_modelTransform2->m_row[0].x, d_modelTransform2->m_row[0].y, d_modelTransform2->m_row[0].z,
                   d_modelTransform2->m_row[1].x, d_modelTransform2->m_row[1].y, d_modelTransform2->m_row[1].z,
                   d_modelTransform2->m_row[2].x, d_modelTransform2->m_row[2].y, d_modelTransform2->m_row[2].z);

            __syncthreads();
#endif

            bool intersects = false;
            if (node1.isLeaf() && node2.isLeaf())
                intersects = intersect<BV>(node1.bbox, node2.bbox, true, alarmDistance);
            else
                intersects = intersect<BV>(node1.bbox, node2.bbox, false, alarmDistance);
#ifdef GPROXIMITY_DEBUG_TRAVERSE_TREE
            if (intersects)
                printf("intersects = 1\n");
            else
                printf("intersects = 0\n");

            __syncthreads();
#endif

            if (intersects)
            {

#ifdef GPROXIMITY_DEBUG_TRAVERSE_TREE
                printf("Overlapping OBB pair marked: %i,%i\n", node1.bbox.idx, node2.bbox.idx);
                __syncthreads();
#endif

#ifdef GPROXIMITY_TRAVERSETREE_STORE_OVERLAPPING_OBB_PAIRS
                int obbOutputIndex = atomicInc(obbListIdx, COLLISION_PAIR_CAPACITY - 1);
                outputOBBList[obbOutputIndex] = make_int2(node1.bbox.idx, node2.bbox.idx);
#endif
                if(node1.isLeaf() && node2.isLeaf())    // both leaf
                {

                    int tri1 = node1.getTriID();
                    int tri2 = node2.getTriID();

                    int globalOutputIndex = atomicInc(outputListIdx, COLLISION_PAIR_CAPACITY - 1);
                    outputList[globalOutputIndex] = make_int2(tri1, tri2);

                    if (outputLeafList != NULL)
                    {
                        if (intersects)
                            outputLeafList[globalOutputIndex] = true;
                        else
                            outputLeafList[globalOutputIndex] = false;
                    }
				}
				else if(node2.isLeaf() || (!node1.isLeaf() && (node1.bbox.getSize() > node2.bbox.getSize())))
				{
					int left1 = node1.getLeftChild() + work_item.x;
					int right1 = left1 + 1;

					int localPtr = atomicAdd(&workQueueIdx, 2);

					if(localPtr < workQueueCapacity)
					{
						localWorkQueue[localPtr].x = left1;
						localWorkQueue[localPtr].y = work_item.y;
						localPtr++;

						if(localPtr < workQueueCapacity)
						{
							localWorkQueue[localPtr].x = right1;
							localWorkQueue[localPtr].y = work_item.y;
						}
						else
						{
							int globalPtr = atomicAdd(&globalWorkQueueIdx, 1);
							int2* globalQueue = &workQueues[blockOffset * globalWorkQueueCapacity + globalPtr];

							globalQueue[0].x = right1;
							globalQueue[0].y = work_item.y;
						}
					}
					else
					{
						int globalPtr = atomicAdd(&globalWorkQueueIdx, 2);
						int2* globalQueue = &workQueues[blockOffset * globalWorkQueueCapacity + globalPtr];

						globalQueue[0].x = left1;
						globalQueue[0].y = work_item.y;

						globalQueue[1].x = right1;
						globalQueue[1].y = work_item.y;
					}
				}
				else
				{
					int left2 = node2.getLeftChild() + work_item.y;
					int right2 = left2 + 1;

					int localPtr = atomicAdd(&workQueueIdx, 2);

					if(localPtr < workQueueCapacity)
					{
						localWorkQueue[localPtr].x = work_item.x;
						localWorkQueue[localPtr].y = left2;
						localPtr++;

						if(localPtr < workQueueCapacity)
						{
							localWorkQueue[localPtr].x = work_item.x;
							localWorkQueue[localPtr].y = right2;
						}
						else
						{
							int globalPtr = atomicAdd(&globalWorkQueueIdx, 1);
							int2* globalQueue = &workQueues[blockOffset * globalWorkQueueCapacity + globalPtr];

							globalQueue[0].x = work_item.x;
							globalQueue[0].y = right2;
						}
					}
					else
					{
						int globalPtr = atomicAdd(&globalWorkQueueIdx, 2);
						int2* globalQueue = &workQueues[blockOffset * globalWorkQueueCapacity + globalPtr];

						globalQueue[0].x = work_item.x;
						globalQueue[0].y = left2;

						globalQueue[1].x = work_item.x;
						globalQueue[1].y = right2;
					}
				}
			}
#ifdef GPROXIMITY_DEBUG_TRAVERSE_TREE
            else
            {
                printf("No OBB overlap: %i, %i\n", node1.bbox.idx, node2.bbox.idx);
                __syncthreads();
            }
#endif
            {
                node1.bbox.center.x -= d_trVector1->x;
                node1.bbox.center.y -= d_trVector1->y;
                node1.bbox.center.z -= d_trVector1->z;
                node1.bbox.center = mtMul1(d_modelTransform1_tr, node1.bbox.center);

                node1.bbox.axis1 = mtMul1(d_modelTransform1_tr, node1.bbox.axis1);
                node1.bbox.axis2 = mtMul1(d_modelTransform1_tr, node1.bbox.axis2);
                node1.bbox.axis3 = mtMul1(d_modelTransform1_tr, node1.bbox.axis3);

            }

            {
                node2.bbox.center.x -= d_trVector2->x;
                node2.bbox.center.y -= d_trVector2->y;
                node2.bbox.center.z -= d_trVector2->z;
                node2.bbox.center = mtMul1(d_modelTransform2_tr, node2.bbox.center);

                node2.bbox.axis1 = mtMul1(d_modelTransform2_tr, node2.bbox.axis1);
                node2.bbox.axis2 = mtMul1(d_modelTransform2_tr, node2.bbox.axis2);
                node2.bbox.axis3 = mtMul1(d_modelTransform2_tr, node2.bbox.axis3);

            }
		}

		__syncthreads();

		if((workQueueIdx >= workQueueCapacity - nThreads) || (globalWorkQueueIdx >= QUEUE_SIZE_PER_TASK_GLOBAL - nThreads * 2 - workQueueCapacity) || (workQueueIdx == 0))
		{
			callAbort<workQueueCapacity>(idleCount, threadOffset);
			break;
		}

		if(threadOffset == 0)
		{
			wantToAbort = *idleCount;
		}

		__syncthreads();

		if(wantToAbort > QUEUE_IDLETASKS_FOR_ABORT)
		{
			callAbort<workQueueCapacity>(idleCount, threadOffset);
			break;
		}
	}

	if(threadOffset == 0)
	{
		workQueueIdx = min(workQueueIdx, workQueueCapacity);
		workQueueCounts[blockOffset] = workQueueIdx + globalWorkQueueIdx;
	}
	__syncthreads();

	{
		int queueOffset = threadOffset;
		int2* globalQueue = &workQueues[blockOffset * globalWorkQueueCapacity + globalWorkQueueIdx];
		while(queueOffset < workQueueIdx * 2)
		{
			((int*)globalQueue)[queueOffset] = ((int*)localWorkQueue)[queueOffset];
			queueOffset += nThreads;
		}
	}
	__syncthreads();
}


// oldest balancing (gproximity's)
// 1. no pump
// 2. copy from inqueue to outqueue with only a few threads (large overhead)
template <int nThreads, int nQueues, class WorkElement>
__global__ void balanceWorkList(WorkElement *workQueueIn, WorkElement* workQueueOut, unsigned int* workQueueCount, const unsigned int maxQueueSize, unsigned int *totalEntries, int* balanceSignal)
{
    /*printf("=== balanceWorkList() ===\n");
    __syncthreads();*/

	__shared__ int shared_sum[nThreads];
	__shared__ int shared_queuesizes[nQueues];
	__shared__ int shared_ifbalance;
	const int idx = threadIdx.x;

	// sum on workQueueCount to find total number of entries
	int nSplitsLeft = nQueues, inputOffset = idx;
	shared_sum[idx] = 0;

	if(idx == 0)
	{
		shared_ifbalance = 0;
	}
	__syncthreads();

	while(idx < nSplitsLeft)
	{
		int nQueueElements = workQueueCount[inputOffset];
		shared_queuesizes[inputOffset] = nQueueElements;
		if((nQueueElements < QUEUE_SIZE_PER_TASK_INIT) && (shared_ifbalance == 0)) atomicExch(&shared_ifbalance, 1);
		else if((nQueueElements >=  QUEUE_SIZE_PER_TASK_GLOBAL - TRAVERSAL_THREADS * 4 - QUEUE_SIZE_PER_TASK) && (shared_ifbalance == 0)) atomicExch(&shared_ifbalance, 1);
		shared_sum[idx] += nQueueElements;
		inputOffset += nThreads;
		nSplitsLeft -= nThreads;
	}
	__syncthreads();

    REDUCE(shared_sum, idx, nThreads, +);

    /*do
    {
        for (int r = nThreads/2; r != 0; r /= 2)
        {
            if (idx < r)
            {
                shared_sum[idx] = shared_sum[idx] + shared_sum[idx + r];
            }
            __syncthreads();
        }
    } while (0);*/

	nSplitsLeft = shared_sum[0];
    //printf(" nSplitsLeft = %i\n", shared_sum[0]);

	if(idx == 0)
	{
		*totalEntries = nSplitsLeft;
		*balanceSignal = shared_ifbalance;

        // printf(" ==> nSplitsLeft = %i, balanceSignal = %i <== \n", *totalEntries, *balanceSignal);
	}
	__syncthreads();

	if(shared_ifbalance > 0)
	{
		int nSplitsPerQueue, nWithPlus1;
		nSplitsPerQueue = nSplitsLeft / nQueues;
		nWithPlus1 = nSplitsLeft - nSplitsPerQueue * nQueues;

		inputOffset = 0;
		int outputOffset = 0, inputQueue = -1, inputQueueCount = 0;

		for(int q = 0; q < nQueues; ++q)
		{
			int nSplitsLocal;
			if(q < nWithPlus1)
				nSplitsLocal = nSplitsPerQueue + 1;
			else
				nSplitsLocal = nSplitsPerQueue;

			outputOffset = maxQueueSize * q + idx;

			if(idx == 0)
				workQueueCount[q] = nSplitsLocal;

			while(nSplitsLocal > 0)
			{
				if(inputQueueCount <= 0)
				{
					inputQueue++;
					inputOffset = idx;
					inputQueueCount = shared_queuesizes[inputQueue];
				}
				else
				{
					int splitsToWrite = min(nSplitsLocal, inputQueueCount);
					splitsToWrite = min(splitsToWrite, nThreads);

					if(idx < splitsToWrite)
						workQueueOut[outputOffset] = workQueueIn[inputQueue * maxQueueSize + inputOffset];
					nSplitsLocal -= splitsToWrite;
					inputOffset += splitsToWrite;
					outputOffset += splitsToWrite;
					inputQueueCount -= splitsToWrite;
					nSplitsLeft -= splitsToWrite;
				}
				__syncthreads();
			}
		}
	}
}

//#define GPROXIMITY_DEBUG_TRIANGLE_COLLISION_2
//#define GPROXIMITY_DEBUG_TRIANGLE_COLLISION_MARK_ALL_TRIS_2

__global__ __forceinline__ void triangleCollision(GPUVertex* vertices1, uint3* triIndices1, GPUVertex* vertices2, uint3* triIndices2, int2* pairs, const int nPairs)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

#ifdef GPROXIMITY_DEBUG_TRIANGLE_COLLISION_2
    printf("triangleCollision: threadID = %i, nPairs = %i\n", threadId, nPairs);
#endif

	if(threadId < nPairs)
	{
		float3 u1, u2, u3, v1, v2, v3;
		int2 pair = pairs[threadId];

		uint3 idx = triIndices1[pair.x];
		uint3 idx2 = triIndices2[pair.y];

		u1 = vertices1[idx.x].v;
		u2 = vertices1[idx.y].v;
		u3 = vertices1[idx.z].v;

		v1 = vertices2[idx2.x].v;
		v2 = vertices2[idx2.y].v;
		v3 = vertices2[idx2.z].v;

#ifdef GPROXIMITY_DEBUG_TRIANGLE_COLLISION_2
        printf("triangleCollision: pair = %i,%i, tri1 = %i,%i,%i, tri2 = %i,%i,%i\n", pair.x, pair.y, idx.x, idx.y, idx.z, idx2.x, idx2.y, idx2.z);
#endif

#ifndef GPROXIMITY_DEBUG_TRIANGLE_COLLISION_MARK_ALL_TRIS_2
        // intersect triangles
        if(triangleIntersection(u1, u2, u3, v1, v2, v3))
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_COLLISION_2
            printf("mark as intersecting (1): %i, %i\n", -pairs[threadIdx.x].x, -pairs[threadIdx.x].y);
#endif
            pairs[threadId] = make_int2(-pair.x, -pair.y);
        }

#ifdef GPROXIMITY_DEBUG_TRIANGLE_COLLISION_VERBOSE_TESTS
        if(triangleIntersection_(u1, u2, u3, v1, v2, v3))
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_COLLISION_2
            printf("mark as intersecting (2): %i, %i\n", -pairs[threadIdx.x].x, -pairs[threadIdx.x].y);
#endif
            pairs[threadId] = make_int2(-pair.x, -pair.y);
        }

        if(triangleIntersection2(u1, u2, u3, v1, v2, v3))
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_COLLISION_2
            printf("mark as intersecting (3): %i, %i\n", -pairs[threadIdx.x].x, -pairs[threadIdx.x].y);
#endif
            pairs[threadId] = make_int2(-pair.x, -pair.y);
        }
#endif
#else
        printf("mark as intersecting (ALL switch): %i, %i\n", -pair.x, -pair.y);
        pairs[threadId] = make_int2(-pair.x, -pair.y);
#endif
	}
}

//#define CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
__device__ inline int doIntersectionTrianglePoint(double dist2, double contactDistance, const float3& p1, const float3& p2, const float3& p3, const float3& n, const float3& q, bool swapElems, gProximityDetectionOutput* contacts, int id1, int id2, int contactId, int outputPos, int triangleCaseNumber = -1, unsigned int maxRange = 0, unsigned int* outputIndex = NULL)
{
#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
    printf("=== doIntersectionTrianglePoint %i - %i: writePos = %i\n ===", id1, id2, outputPos);
    __syncthreads();
#endif

#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
    printf("  doIntersectionTrianglePoint %i - %i: p1 = %f,%f,%f; p2 = %f,%f,%f; p3 = %f,%f,%f, q = %f,%f,%f\n", id1, id2, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, q.x, q.y, q.z);
    __syncthreads();
#endif
    const float3 AB = f3v_sub(p2,p1);
    const float3 AC = f3v_sub(p3,p1);
    const float3 AQ = f3v_sub(q, p1);
    Matrix2x2 A;
    float2 b;
    A.m_row[0].x = f3v_dot(AB, AB);
    A.m_row[1].y = f3v_dot(AC, AC);
    A.m_row[0].y = A.m_row[1].x = f3v_dot(AB,AC);
    b.x = f3v_dot(AQ, AB);
    b.y = f3v_dot(AQ, AC);
    const double det = determinant(A);

#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
    printf("  doIntersectionTrianglePoint %i - %i: AB = %f,%f,%f; AC = %f,%f,%f; AQ = %f,%f,%f\n", id1, id2, AB.x, AB.y, AB.z, AC.x, AC.y, AC.z, AQ.x, AQ.y, AQ.z);
    printf("  doIntersectionTrianglePoint %i - %i: b.x = %f, b.y = %f; A[0][0] = %f, A[0][1] = %f, A[1][0] = %f, A[1][1] = %f\n", id1, id2, b.x, b.y, A.m_row[0].x, A.m_row[0].y, A.m_row[1].x, A.m_row[1].y);
    __syncthreads();
#endif

    double alpha = 0.5;
    double beta = 0.5;
    {
        alpha = (b.x * A.m_row[1].y - b.y * A.m_row[0].y) / det;
        beta  = (b.y * A.m_row[0].x - b.x * A.m_row[1].x) / det;

#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
        printf("  doIntersectionTrianglePoint %i - %i: alpha (1) = %f, beta (1) = %f\n", id1, id2, alpha, beta);
#endif
        if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
        {
            // nearest point is on an edge or corner
            // barycentric coordinate on AB
            double pAB = b.x / A.m_row[0].x; // AQ*AB / AB*AB
            // barycentric coordinate on AC
            double pAC = b.y / A.m_row[1].y; // AQ*AB / AB*AB
            if (pAB < 0.000001 && pAC < 0.0000001)
            {
                // closest point is A
                alpha = 0.0;
                beta = 0.0;

#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
                printf("  doIntersectionTrianglePoint %i - %i: pAB = %f, pAC = %f, alpha (2) = %f, beta (2) = %f\n", id1, id2, pAB, pAC, alpha, beta);
#endif
            }            
            else if (pAB < 0.999999 && beta < 0.000001)
            {
                // closest point is on AB
                alpha = pAB;
                beta = 0.0;
#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
                printf("  doIntersectionTrianglePoint %i - %i: pAB = %f, pAC = %f, alpha (3) = %f, beta (3) = %f\n", id1, id2, pAB, pAC, alpha, beta);
#endif
            }
            else if (pAC < 0.999999 && alpha < 0.000001)
            {
                // closest point is on AC
                alpha = 0.0;
                beta = pAC;
#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
                printf("  doIntersectionTrianglePoint %i - %i: pAB = %f, pAC = %f, alpha (4) = %f, beta (4) = %f\n", id1, id2, pAB, pAC, alpha, beta);
#endif
            }
            else
            {
                // barycentric coordinate on BC
                // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
                double pBC = (b.y - b.x + A.m_row[0].x - A.m_row[0].y) / (A.m_row[0].x + A.m_row[1].y - 2 * A.m_row[0].y); // BQ*BC / BC*BC
                if (pBC < 0.000001)
                {
                    // closest point is B
                    alpha = 1.0;
                    beta = 0.0;
#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
                    printf("  doIntersectionTrianglePoint %i - %i: pBC= %f, alpha (5) = %f, beta (5) = %f\n", id1, id2, pBC, alpha, beta);
#endif
                }
                else if (pBC > 0.999999)
                {
                    // closest point is C
                    alpha = 0.0;
                    beta = 1.0;
#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
                    printf("  doIntersectionTrianglePoint %i - %i: pBC= %f, alpha (6) = %f, beta (6) = %f\n", id1, id2, pBC, alpha, beta);
#endif
                }
                else
                {
                    // closest point is on BC
                    alpha = 1.0 - pBC;
                    beta = pBC;
#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
                    printf("  doIntersectionTrianglePoint %i - %i: pBC= %f, alpha (7) = %f, beta (7) = %f\n", id1, id2, pBC, alpha, beta);
#endif
                }
            }
        }
    }

    float3 p, pq;    

    p = f3v_add(f3v_add(p1, f3v_mul1(AB, alpha)), f3v_mul1(AC, beta));
    pq = f3v_sub(q, p);
    double pq_norm2 = f3v_len_squared(pq);

#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
    __syncthreads();
    printf("  doIntersectionTrianglePoint %i - %i: alpha = %f, beta = %f\n", id1, id2, alpha, beta);
    printf("  doIntersectionTrianglePoint %i - %i: p = %f,%f,%f; pq = %f,%f,%f\n", id1, id2, p.x, p.y, p.z, pq.x, pq.y, pq.z);
    __syncthreads();
#endif

#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
    printf("  doIntersectionTrianglePoint %i - %i: Write to pos %i\n", id1, id2, outputPos);
    __syncthreads();
    printf("  doIntersectionTrianglePoint %i - %i: pq_norm2 = %f, dist2 = %f\n", id1, id2, pq_norm2, dist2);
    __syncthreads();
#endif

    bool contactValid = true;
    if (pq_norm2 >= dist2)
    {
#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
        printf("   Contact valid: false\n");
        __syncthreads();
#endif

        contactValid = false;
    }
#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
    printf("   Contact valid: true\n");
    __syncthreads();
#endif

    if (contactValid == false)
    {
        if (outputIndex == NULL)
            contacts->valid[outputPos] = false;

        return 0;
    }

    int validOutputIndex = -1;
    if (outputIndex != NULL)
    {
        validOutputIndex = atomicInc(outputIndex, maxRange);
#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
        printf("   doIntersectionTrianglePoint: validOutputIndex = %i, maxRange = %i\n", validOutputIndex, maxRange);
        __syncthreads();
#endif
    }

    if (outputIndex == NULL)
    {
        contacts->valid[outputPos] = true;

        contacts->contactId[outputPos] = contactId;
        contacts->elems[outputPos] = make_int4(id2, 0, 0, id1);

        contacts->distance[outputPos] = f3v_len(pq);
        contacts->contactType[outputPos] = COLLISION_VERTEX_FACE;
    }
    else
    {
        contacts->valid[validOutputIndex] = true;
        contacts->contactId[validOutputIndex] = contactId;

        contacts->elems[validOutputIndex] = make_int4(id2, 0, 0, id1);
        switch (triangleCaseNumber)
        {
            case 0:
            {
                contacts->elems[validOutputIndex].y = 0;
                contacts->elems[validOutputIndex].z = -1;
                break;
            }
            case 1:
            {
                contacts->elems[validOutputIndex].y = 1;
                contacts->elems[validOutputIndex].z = -1;
                break;
            }
            case 2:
            {
                contacts->elems[validOutputIndex].y = 2;
                contacts->elems[validOutputIndex].z = -1;
                break;
            }
            case 3:
            {
                contacts->elems[validOutputIndex].y = -1;
                contacts->elems[validOutputIndex].z = 0;
                break;
            }
            case 4:
            {
                contacts->elems[validOutputIndex].y = -1;
                contacts->elems[validOutputIndex].z = 1;
                break;
            }
            case 5:
            {
                contacts->elems[validOutputIndex].y = -1;
                contacts->elems[validOutputIndex].z = 2;
            }
            default:
                break;
        }

        contacts->distance[validOutputIndex] = f3v_len(pq);
        contacts->contactType[validOutputIndex] = COLLISION_VERTEX_FACE;
    }

    if (swapElems)
    {
        float3 minus_pq = make_float3(-pq.x, -pq.y, -pq.z);
        if (outputIndex == NULL)
        {
            contacts->point0[outputPos] = q;
            contacts->point1[outputPos] = p;

            contacts->normal[outputPos] = minus_pq;
        }
        else
        {
            contacts->point0[validOutputIndex] = q;
            contacts->point1[validOutputIndex] = p;

            contacts->normal[validOutputIndex] = minus_pq;
        }
    }
    else
    {
        if (outputIndex == NULL)
        {
            contacts->point0[outputPos] = p;
			contacts->point1[outputPos] = q;

			contacts->normal[outputPos] = pq;
        }
        else
        {
            contacts->point0[validOutputIndex] = p;
            contacts->point1[validOutputIndex] = q;

            contacts->normal[validOutputIndex] = pq;
        }
    }

#ifdef CUDA_INTERSECT_TREE_DEBUG_TRIANGLE_POINT
    printf("  ==> new contact created (point-triangle), outputIndex = %i: point0 = %f,%f,%f; point1 = %f,%f,%f, valid = %d; elems: w = %i, x = %i, y = %i, z = %i <==\n",
           validOutputIndex, p.x, p.y, p.z, q.x, q.y, q.z, contacts->valid[validOutputIndex], contacts->elems[validOutputIndex].w, contacts->elems[validOutputIndex].x, contacts->elems[validOutputIndex].y, contacts->elems[validOutputIndex].z);
    __syncthreads();
#endif

    return 1;
}

//#define GPROXIMITY_DEBUG_SEG_NEAREST_POINTS
__device__ __forceinline__
void segNearestPoints(const float3& p0,const float3& p1, const float3& q0,const float3& q1, float3& P, float3& Q,
                                      double& alpha, double& beta)
{
    const float3 AB = f3v_sub(p1, p0);
    const float3 CD = f3v_sub(q1, q0);
    const float3 AC = f3v_sub(q0, p0);

    Matrix2x2 Amat = getZeroMatrix2x2(); //matrix helping us to find the two nearest points lying on the segments of the two segments
    float2 b;

    Amat.m_row[0].x = f3v_dot(AB, AB);
    Amat.m_row[1].y = f3v_dot(CD, CD);
    const float3 minusCD = make_float3(-CD.x, -CD.y, -CD.z);
    Amat.m_row[0].y = Amat.m_row[1].x = f3v_dot(minusCD, AB);

    b.x = f3v_dot(AB,AC);
    b.y = f3v_dot(minusCD,AC);

    const double det = determinant(Amat);

    double AB_norm2 = f3v_len_squared(AB);
    double CD_norm2 = f3v_len_squared(CD);
    alpha = 0.5;
    beta = 0.5;
    //Check that the determinant is not null which would mean that the segment segments are lying on a same plane.
    //in this case we can solve the little system which gives us
    //the two coefficients alpha and beta. We obtain the two nearest points P and Q lying on the segments of the two segments.
    if (det < -1e-6 || det > 1e-6)
    {
        alpha = (b.x * Amat.m_row[1].y - b.y * Amat.m_row[0].y) / det;
        beta  = (b.y * Amat.m_row[0].x - b.x * Amat.m_row[1].x) / det;
    }
    else
    {   //segment segments on a same plane. Here the idea to find the nearest points
        //is to project segment apexes on the other segment.
        //Visual example with semgents AB and CD :
        //            A----------------B
        //                     C----------------D
        //After projection :
        //            A--------c-------B
        //                     C-------b--------D
        //So the nearest points are p and q which are respecively in the middle of cB and Cb:
        //            A--------c---p---B
        //                     C---q---b--------D
        float3 AD = f3v_sub(q1, p0);
        float3 CB = f3v_sub(p1, q0);

        double c_proj= b.x / AB_norm2; //alpha = (AB * AC)/AB_norm2
        double d_proj = f3v_dot(AB, AD) / AB_norm2;
        double a_proj = b.y / CD_norm2; //beta = (-CD*AC)/CD_norm2
        double b_proj= f3v_dot(CD,CB) / CD_norm2;

        if (c_proj >= 0 && c_proj <= 1)
        {   //projection of C on AB is lying on AB
            if(d_proj > 1)
            {              //case :
                           //             A----------------B
                           //                      C---------------D
                alpha = (1.0 + c_proj) / 2.0;
                beta = b_proj / 2.0;
            }
            else if(d_proj < 0)
            {                   //case :
                                //             A----------------B
                                //     D----------------C
                alpha = c_proj / 2.0;
                beta = (1 + a_proj) / 2.0;
            }
            else
            {   //case :
                //             A----------------B
                //                 C------D
                alpha = (c_proj + d_proj) / 2.0;
                beta  = 0.5;
            }
        }
        else if(d_proj >= 0 && d_proj <= 1)
        {
            if(c_proj < 0)
            {              //case :
                           //             A----------------B
                           //     C----------------D
                alpha = d_proj / 2.0;
                beta = (1 + a_proj) / 2.0;
            }
            else
            {    //case :
                 //          A---------------B
                 //                 D-------------C
                alpha = (1 + d_proj) / 2.0;
                beta = b_proj / 2.0;
            }
        }
        else
        {
            if(c_proj * d_proj < 0)
            {                       //case :
                                    //           A--------B
                                    //       D-----------------C
                alpha = 0.5;
                beta = (a_proj + b_proj) / 2.0;
            }
            else
            {
                if(c_proj < 0)
                {              //case :
                               //                    A---------------B
                               // C-------------D
                    alpha = 0;
                }
                else
                {
                    alpha = 1;
                }

                if(a_proj < 0)
                {              //case :
                               // A---------------B
                               //                     C-------------D
                    beta = 0;
                }
                else
                {    //case :
                     //                     A---------------B
                     //   C-------------D
                    beta = 1;
                }
            }
        }

        P = f3v_mul1(f3v_add(p0, AB), alpha);
        Q = f3v_mul1(f3v_add(q0, CD), beta);
#ifdef GPROXIMITY_DEBUG_SEG_NEAREST_POINTS
        printf("GPU case 1: alpha = %f, beta = %f, P = %f,%f,%f, Q = %f,%f,%f\n", alpha, beta, P.x, P.y, P.z, Q.x, Q.y, Q.z);
#endif
        return;
    }

    if(alpha < 0)
    {
        alpha = 0;
        beta = f3v_dot(CD, f3v_sub(p0, q0)) / CD_norm2;
    }
    else if(alpha > 1)
    {
        alpha = 1;
        beta = f3v_dot(CD, f3v_sub(p1, q0)) / CD_norm2;
    }

    if(beta < 0)
    {
        beta = 0;
        alpha = f3v_dot(AB, f3v_sub(q0, p0)) / AB_norm2;
    }
    else if(beta > 1)
    {
        beta = 1;
        alpha = (f3v_dot(AB, f3v_sub(q1, p0))) / AB_norm2;
    }

    if(alpha < 0)
        alpha = 0;
    else if (alpha > 1)
        alpha = 1;

    P = f3v_mul1(f3v_add(p0, AB), alpha);
    Q = f3v_mul1(f3v_add(q0, CD), beta);
#ifdef GPROXIMITY_DEBUG_SEG_NEAREST_POINTS
    printf("GPU case 2: alpha = %f, beta = %f, P = %f,%f,%f, Q = %f,%f,%f\n", alpha, beta, P.x, P.y, P.z, Q.x, Q.y, Q.z);
#endif
}

//#define CUDA_INTERSECT_TREE_DEBUG_LINE_LINE
__device__ inline int doIntersectionLineLine(double dist2, double contactDistance, const float3& p1, const float3& p2, const float3& q1, const float3& q2, gProximityDetectionOutput* contacts, int id1, int id2, int edgeId1, int edgeId2, int contactId, int outputPos, int lineCaseNumber = -1, unsigned int maxRange = 0, unsigned int* outputIndex = NULL)
{
#ifdef CUDA_INTERSECT_TREE_DEBUG_LINE_LINE
    printf("=== doIntersectionLineLine: %i - %i; writePos = %i ===\n", id1, id2, outputPos);
    __syncthreads();
    printf("  p1 = %f,%f,%f; p2 = %f,%f,%f; q1 = %f,%f,%f; q2 = %f,%f,%f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, q1.x, q1.y, q1.z, q2.x, q2.y, q2.z);
    __syncthreads();
    printf("  edge1 = %i; edge2 = %i\n", edgeId1, edgeId2);
    __syncthreads();
#endif

    float3 AB = f3v_sub(p2, p1);
    float3 CD = f3v_sub(q2, q1);
    float3 AC = f3v_sub(q1, p1);

    Matrix2x2 A;
    float2 b;

    A.m_row[0].x = f3v_dot(AB,AB);
    A.m_row[1].y = f3v_dot(CD,CD);

    const float3 minusCD = make_float3(-CD.x, -CD.y, -CD.z);

    A.m_row[0].y = A.m_row[1].x = f3v_dot(minusCD, AB);
    b.x = f3v_dot(AB, AC);
    b.y = f3v_dot(minusCD, AC);
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    if (det < -1.0e-30 || det > 1.0e-30)
    {
        alpha = (b.x * A.m_row[1].y - b.y * A.m_row[0].y) / det;
        beta  = (b.y * A.m_row[0].x - b.x * A.m_row[1].x) / det;
#ifdef CUDA_INTERSECT_TREE_DEBUG_LINE_LINE
        printf("  alpha = %f, beta = %f, det = %f\n", alpha, beta, det);
        __syncthreads();
#endif
        if (alpha < 1e-15 || alpha > (1.0-1e-15) ||
            beta  < 1e-15  || beta  > (1.0-1e-15) )
        {
#ifdef CUDA_INTERSECT_TREE_DEBUG_LINE_LINE
        printf("   Contact valid: false (1)\n");
        __syncthreads();
#endif
            return 0;
        }
    }
    else
    {
        // several possibilities :
        // -one point in common (auto-collision) => return false !
        // -no point in common but line are // => we can continue to test
#ifdef CUDA_INTERSECT_TREE_DEBUG_LINE_LINE
       printf("  WARNING det is null\n");
       __syncthreads();
#endif
    }

    float3 P,Q,PQ;
    //P = f3v_mul1(f3v_add(p1, AB), alpha);
    //Q = f3v_mul1(f3v_add(q1, CD), beta);

    P = f3v_add(p1, f3v_mul1(AB, alpha));
    Q = f3v_add(q1, f3v_mul1(CD, beta));

    PQ = f3v_sub(Q,P);

    float pqNorm2 = f3v_len_squared(PQ);

#ifdef CUDA_INTERSECT_TREE_DEBUG_LINE_LINE
    printf("  pqNorm2 = %f, dist2 = %f\n", pqNorm2, dist2);
    __syncthreads();
    printf("  P = %f,%f,%f; Q = %f,%f,%f\n", P.x, P.y, P.z, Q.x, Q.y, Q.z);
    __syncthreads();
#endif
    if (pqNorm2 >= dist2)
    {
#ifdef CUDA_INTERSECT_TREE_DEBUG_LINE_LINE
        printf("   Contact valid: false (2)\n");
        __syncthreads();
#endif
        if (outputIndex == NULL)
            contacts->valid[outputPos] = false;

        return 0;
    }

#ifdef CUDA_INTERSECT_TREE_DEBUG_LINE_LINE
        printf("   Contact valid: true\n");
        __syncthreads();
#endif

    int validOutputIndex = -1;
    if (outputIndex != NULL)
    {
        validOutputIndex = atomicInc(outputIndex, maxRange);

#ifdef CUDA_INTERSECT_TREE_DEBUG_LINE_LINE
        printf("   doIntersectionLineLine: validOutputIndex = %i, maxRange = %i\n", validOutputIndex, maxRange);
        __syncthreads();
#endif
    }
    if (outputIndex == NULL)
    {
        contacts->valid[outputPos] = true;
        contacts->contactId[outputPos] = contactId;
        contacts->elems[outputPos] = make_int4(id2, edgeId1, edgeId2, id1);
        contacts->point0[outputPos] = P;
        contacts->point1[outputPos] = Q;

        contacts->contactType[outputPos] = COLLISION_LINE_LINE;
        contacts->distance[outputPos] = f3v_len(PQ);

        contacts->normal[outputPos] = PQ;
    }
    else
    {
        contacts->valid[validOutputIndex] = true;
        contacts->contactId[validOutputIndex] = contactId;


        contacts->elems[validOutputIndex] = make_int4(id2, edgeId1, edgeId2, id1);


        switch (lineCaseNumber)
        {
            case 0:
            {
                contacts->elems[validOutputIndex].y = 2;
                contacts->elems[validOutputIndex].z = 2;
                break;
            }
            case 1:
            {
                contacts->elems[validOutputIndex].y = 2;
                contacts->elems[validOutputIndex].z = 0;
                break;
            }
            case 2:
            {
                contacts->elems[validOutputIndex].y = 2;
                contacts->elems[validOutputIndex].z = 1;
                break;
            }
            case 3:
            {
                contacts->elems[validOutputIndex].y = 0;
                contacts->elems[validOutputIndex].z = 2;
                break;
            }
            case 4:
            {
                contacts->elems[validOutputIndex].y = 0;
                contacts->elems[validOutputIndex].z = 0;
                break;
            }
            case 5:
            {
                contacts->elems[validOutputIndex].y = 0;
                contacts->elems[validOutputIndex].z = 1;
                break;
            }
            case 6:
            {
                contacts->elems[validOutputIndex].y = 1;
                contacts->elems[validOutputIndex].z = 2;
                break;
            }
            case 7:
            {
                contacts->elems[validOutputIndex].y = 1;
                contacts->elems[validOutputIndex].z = 0;
                break;
            }
            case 8:
            {
                contacts->elems[validOutputIndex].y = 1;
                contacts->elems[validOutputIndex].z = 1;
                break;
            }
            default:
                break;
        }

        contacts->point0[validOutputIndex] = P;
        contacts->point1[validOutputIndex] = Q;

        contacts->contactType[validOutputIndex] = COLLISION_LINE_LINE;
        contacts->distance[validOutputIndex] = f3v_len(PQ);

        contacts->normal[validOutputIndex] = PQ;
    }

#ifdef CUDA_INTERSECT_TREE_DEBUG_LINE_LINE
    printf("  ==> new contact created (line-line), validOutputIndex = %i: point0 = %f,%f,%f; point1 = %f,%f,%f, valid = %d; elems: w = %i, x = %i, y = %i, z = %i <==\n",
           validOutputIndex, P.x, P.y, P.z, Q.x, Q.y, Q.z, contacts->valid[validOutputIndex], contacts->elems[validOutputIndex].w, contacts->elems[validOutputIndex].x, contacts->elems[validOutputIndex].y, contacts->elems[validOutputIndex].z);
    __syncthreads();
#endif

    return 1;
}

//#define CUDA_INTERSECT_TREE_DEBUG_LINE_POINT
__device__ __forceinline__ int doIntersectionLinePoint(double dist2, double contactDistance, const float3& p1, const float3& p2, const float3& q1, gProximityDetectionOutput* contacts, int id1, int id2, int contactId, int outputPos)
{
#ifdef CUDA_INTERSECT_TREE_DEBUG_LINE_POINT
    printf("=== doIntersectionLinePoint: %i - %i; outputPos = %i ===\n", id1, id2, outputPos);
    __syncthreads();
    printf("  p1 = %f,%f,%f; p2 = %f,%f,%f; q1 = %f,%f,%f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, q1.x, q1.y, q1.z);
    __syncthreads();
#endif

    float3 AB = f3v_sub(p2, p1);
    float3 AP = f3v_sub(q1, p1);

    double norm_AB = f3v_len(AB);
    double norm_AP = f3v_len(AP);

    if (norm_AB < 0.000000000001 * norm_AP)
    {
        return 0;
    }

    double A = f3v_dot(AB,AB);
    double b = f3v_dot(AP,AB);

    double alpha = b / A;

    if (alpha < 0.000001 || alpha > 0.999999)
        return 0;

    float3 P = make_float3(q1.x, q1.y, q1.z);
    float3 AB_alpha = f3v_mul1(AB, alpha);
    float3 Q = f3v_add(p1, AB_alpha);

    float3 PQ = f3v_sub(Q,P);
    float3 QP = f3v_mul1(PQ, -1.0f);

    double PQ_norm2 = f3v_len_squared(PQ);

    if (PQ_norm2 >= dist2)
    {
        contacts->valid[outputPos] = false;
        return 0;
    }

    contacts->valid[outputPos] = true;
    contacts->contactId[outputPos] = contactId;
    contacts->elems[outputPos] = make_int4(id2, -1, -1, id1);
    contacts->point0[outputPos] = make_float3(Q.x, Q.y, Q.z);
    contacts->point1[outputPos] = make_float3(P.x, P.y, P.z);

    contacts->contactType[outputPos] = COLLISION_LINE_POINT;
    contacts->distance[outputPos] = f3v_len(QP);

    contacts->normal[outputPos] = QP;

    return 1;
}


#define GPROXIMITY_TRIANGLE_PAIR_INTERSECTIONS_LINE_LINE
#define GPROXIMITY_TRIANGLE_PAIR_INTERSECTIONS_TRIANGLE_POINT
//#define GPROXIMITY_TRIANGLE_PAIR_INTERSECTIONS_LINE_POINT
//#define DEBUG_TRIANGLE_CONTACT
//#define GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
__device__ __forceinline__ int triangleContact(float3 p1, float3 p2, float3 p3, float3 q1, float3 q2, float3 q3, double alarmDist, double contactDist, int index1, int index2, gProximityDetectionOutput* contacts, int outputPos, int nMaxContacts, int CollisionTestElementsSize)
{
    int writePos = outputPos * CollisionTestElementsSize;
    if (writePos < nMaxContacts)
    {
#ifdef DEBUG_TRIANGLE_CONTACT
        printf("triangleContact: %i - %i to writePos = %i, outputPos = %i\n", index1, index2, writePos, outputPos);
#endif
        const double maxContactDist = alarmDist + (alarmDist - contactDist);
        const double dist2 =  maxContactDist * maxContactDist;
        float3 tri1_edge1 = f3v_sub(p2, p1);
        float3 tri1_edge2 = f3v_sub(p3, p1);
        const float3 pn = f3v_cross(tri1_edge1, tri1_edge2);

        float3 tri2_edge1 = f3v_sub(q2, q1);
        float3 tri2_edge2 = f3v_sub(q3, q1);
        const float3 qn = f3v_cross(tri2_edge1, tri2_edge2);

        const int id1 = index1; // index of contacts involving points in e1
        const int id2 = index2; // index of contacts involving points in e2

        int n = 0;

#ifdef GPROXIMITY_TRIANGLE_PAIR_INTERSECTIONS_TRIANGLE_POINT
        int nTriangleContacts = 0;
        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, q1, q2, q3, qn, p1, true, contacts, id1, id2, id1+0, writePos);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
            contacts->elems[writePos].y = 0;
            contacts->elems[writePos].z = -1;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif

        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, q1, q2, q3, qn, p2, true, contacts, id1, id2, id1+1, writePos + 1);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
            contacts->elems[writePos + 1].y = 1;
            contacts->elems[writePos + 1].z = -1;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif
        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, q1, q2, q3, qn, p3, true, contacts, id1, id2, id1+2, writePos + 2);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
            contacts->elems[writePos + 2].y = 2;
            contacts->elems[writePos + 2].z = -1;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif


        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, p1, p2, p3, pn, q1, false, contacts, id1, id2, id2+0, writePos + 3);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
            contacts->elems[writePos + 3].z = 0;
            contacts->elems[writePos + 3].y = -1;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif


        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, p1, p2, p3, pn, q2, false, contacts, id1, id2, id2+1, writePos + 4);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
            contacts->elems[writePos + 4].z = 1;
            contacts->elems[writePos + 4].y = -1;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif
        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, p1, p2, p3, pn, q3, false, contacts, id1, id2, id2+2, writePos + 5);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
            contacts->elems[writePos + 5].z = 2;
            contacts->elems[writePos + 5].y = -1;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif

#endif

#ifdef GPROXIMITY_TRIANGLE_PAIR_INTERSECTIONS_LINE_LINE
        int nLineContacts = 0;
        nLineContacts = doIntersectionLineLine(dist2, contactDist, p1, p2, q1, q2, contacts, id1, id2, 0, 0, id2+3, writePos + 6);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
            printf("LINE_LINE contact case 1: edges %i - %i; p1 = %f,%f,%f, p2 = %f,%f,%f, q1 = %f,%f,%f, q2 = %f,%f,%f\n", 2, 2, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, q1.x, q1.y, q1.z, q2.x, q2.y, q2.z);
#endif
            contacts->elems[writePos + 6].y = 2;
            contacts->elems[writePos + 6].z = 2;
        }

#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p1, p2, q2, q3, contacts, id1, id2, 0, 1, id2+4, writePos + 7);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
            printf("LINE_LINE contact case 2: edges %i - %i; p1 = %f,%f,%f, p2 = %f,%f,%f, q2 = %f,%f,%f, q3 = %f,%f,%f\n", 2, 0, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, q2.x, q2.y, q2.z, q3.x, q3.y, q3.z);
#endif
            contacts->elems[writePos + 7].y = 2;
            contacts->elems[writePos + 7].z = 0;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p1, p2, q3, q1, contacts, id1, id2, 0, 2, id2+5, writePos + 8);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
            printf("LINE_LINE contact case 3: edges %i - %i; p1 = %f,%f,%f, p2 = %f,%f,%f, q3 = %f,%f,%f, q1 = %f,%f,%f\n", 2, 1, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, q3.x, q3.y, q3.z, q1.x, q1.y, q1.z);
#endif
            contacts->elems[writePos + 8].y = 2;
            contacts->elems[writePos + 8].z = 1;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p2, p3, q1, q2, contacts, id1, id2, 1, 0, id2+6, writePos + 9);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
            printf("LINE_LINE contact case 4: edges %i - %i; p2 = %f,%f,%f, p3 = %f,%f,%f, q1 = %f,%f,%f, q2 = %f,%f,%f\n", 0, 2, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, q1.x, q1.y, q1.z, q2.x, q2.y, q2.z);
#endif
            contacts->elems[writePos + 9].y = 0;
            contacts->elems[writePos + 9].z = 2;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p2, p3, q2, q3, contacts, id1, id2, 1, 1, id2+7, writePos + 10);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
            printf("LINE_LINE contact case 5: edges %i - %i; p2 = %f,%f,%f, p3 = %f,%f,%f, q2 = %f,%f,%f, q3 = %f,%f,%f\n", 0, 0, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, q2.x, q2.y, q2.z, q3.x, q3.y, q3.z);
#endif
            contacts->elems[writePos + 10].y = 0;
            contacts->elems[writePos + 10].z = 0;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif
        nLineContacts = doIntersectionLineLine(dist2, contactDist, p2, p3, q3, q1, contacts, id1, id2, 1, 2, id2+8, writePos + 11);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
            printf("LINE_LINE contact case 6: edges %i - %i; p2 = %f,%f,%f, p3 = %f,%f,%f, q3 = %f,%f,%f, q1 = %f,%f,%f\n", 0, 1, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, q3.x, q3.y, q3.z, q1.x, q1.y, q1.z);
#endif
            contacts->elems[writePos + 11].y = 0;
            contacts->elems[writePos + 11].z = 1;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p3, p1, q1, q2, contacts, id1, id2, 2, 0, id2+9, writePos + 12);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
            printf("LINE_LINE contact case 7: edges %i - %i; p3 = %f,%f,%f, p1 = %f,%f,%f, q1 = %f,%f,%f, q2 = %f,%f,%f\n", 1, 2, p3.x, p3.y, p3.z, p1.x, p1.y, p1.z, q1.x, q1.y, q1.z, q2.x, q2.y, q2.z);
#endif
            contacts->elems[writePos + 12].y = 1;
            contacts->elems[writePos + 12].z = 2;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif
        nLineContacts = doIntersectionLineLine(dist2, contactDist, p3, p1, q2, q3, contacts, id1, id2, 2, 1, id2+10, writePos + 13);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
            printf("LINE_LINE contact case 8: edges %i - %i; p3 = %f,%f,%f, p1 = %f,%f,%f, q2 = %f,%f,%f, q3 = %f,%f,%f\n", 1, 0, p3.x, p3.y, p3.z, p1.x, p1.y, p1.z, q2.x, q2.y, q2.z, q3.x, q3.y, q3.z);
#endif
            contacts->elems[writePos + 13].y = 1;
            contacts->elems[writePos + 13].z = 0;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif
        nLineContacts = doIntersectionLineLine(dist2, contactDist, p3, p1, q3, q1, contacts, id1, id2, 2, 2, id2+11, writePos + 14);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
            printf("LINE_LINE contact case 9: edges %i - %i; p3 = %f,%f,%f, p1 = %f,%f,%f, q3 = %f,%f,%f, q1 = %f,%f,%f\n", 1, 1, p3.x, p3.y, p3.z, p1.x, p1.y, p1.z, q3.x, q3.y, q3.z, q1.x, q1.y, q1.z);
#endif
            contacts->elems[writePos + 14].y = 1;
            contacts->elems[writePos + 14].z = 1;
            n += nLineContacts;
        }
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif

#endif


#ifdef GPROXIMITY_TRIANGLE_PAIR_INTERSECTIONS_LINE_POINT
        int nLinePointContacts = 0;
        nLinePointContacts = doIntersectionLinePoint(dist2, contactDist, p1, p2, q1, contacts, id1, id2, id2+12, writePos + 15);
        if (nLinePointContacts > 0)
        {
            n += nLineContacts;
            contacts->elems[writePos + 15].y = 2;
            contacts->elems[writePos + 15].z = 0;
        }

        nLinePointContacts = doIntersectionLinePoint(dist2, contactDist, p1, p2, q2, contacts, id1, id2, id2+13, writePos + 16);
        if (nLinePointContacts > 0)
        {
            n += nLineContacts;
            contacts->elems[writePos + 16].y = 2;
            contacts->elems[writePos + 16].z = 1;
        }

        nLinePointContacts = doIntersectionLinePoint(dist2, contactDist, p1, p2, q3, contacts, id1, id2, id2+13, writePos + 17);
        if (nLinePointContacts > 0)
        {
            n += nLineContacts;
            contacts->elems[writePos + 17].y = 2;
            contacts->elems[writePos + 17].z = 2;
        }

        nLinePointContacts = doIntersectionLinePoint(dist2, contactDist, p2, p3, q1, contacts, id1, id2, id2+14, writePos + 18);
        if (nLinePointContacts > 0)
        {
            n += nLineContacts;
            contacts->elems[writePos + 18].y = 0;
            contacts->elems[writePos + 18].z = 0;
        }

        nLinePointContacts = doIntersectionLinePoint(dist2, contactDist, p2, p3, q2, contacts, id1, id2, id2+15, writePos + 19);
        if (nLinePointContacts > 0)
        {
            n += nLineContacts;
            contacts->elems[writePos + 19].y = 0;
            contacts->elems[writePos + 19].z = 1;
        }

        nLinePointContacts = doIntersectionLinePoint(dist2, contactDist, p2, p3, q3, contacts, id1, id2, id2+16, writePos + 20);
        if (nLinePointContacts > 0)
        {
            n += nLineContacts;
            contacts->elems[writePos + 20].y = 0;
            contacts->elems[writePos + 20].z = 2;
        }

        nLinePointContacts = doIntersectionLinePoint(dist2, contactDist, p3, p1, q1, contacts, id1, id2, id2+17, writePos + 21);
        if (nLinePointContacts > 0)
        {
            n += nLineContacts;
            contacts->elems[writePos + 21].y = 1;
            contacts->elems[writePos + 21].z = 0;
        }

        nLinePointContacts = doIntersectionLinePoint(dist2, contactDist, p3, p1, q2, contacts, id1, id2, id2+18, writePos + 22);
        if (nLinePointContacts > 0)
        {
            n += nLineContacts;
            contacts->elems[writePos + 22].y = 1;
            contacts->elems[writePos + 22].z = 1;
        }

        nLinePointContacts = doIntersectionLinePoint(dist2, contactDist, p3, p1, q3, contacts, id1, id2, id2+19, writePos + 23);
        if (nLinePointContacts > 0)
        {
            n += nLineContacts;
            contacts->elems[writePos + 23].y = 1;
            contacts->elems[writePos + 23].z = 2;
        }
#endif



#ifdef DEBUG_TRIANGLE_CONTACT
        printf("triangleContact %i - %i: contact points generated = %i\n", index1, index2, n);
#endif
        return n;
    }
    return 0;
}






//#define GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
__global__ __forceinline__ void trianglePairIntersections(GPUVertex *d_vertices1, uint3 *d_triIndices1, GPUVertex *d_vertices2, uint3* d_triIndices2, int2 *pairs, const int nPairs, gProximityDetectionOutput* contacts, const int nMaxContacts, double alarmDistance, double contactDistance)
{
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
    printf("trianglePairIntersections: gridDim=%i,%i,%i, blockIdx=%i,%i,%i, threadIdx=%i,%i,%i\n", gridDim.x, gridDim.y, gridDim.z, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
#endif
    const int threadID = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
    __syncthreads();
    printf(" threadID = %i, nPairs = %i\n", threadID, nPairs);
    __syncthreads();
#endif

    // don't go beyond end of pair list

    if(threadID < nPairs)
    {
        float3 u1, u2, u3, v1, v2, v3;
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        printf(" access pairs[%i]\n", threadID);
        __syncthreads();
#endif
        int2 pair = pairs[threadID];

        pair.x = abs(pair.x);
        pair.y = abs(pair.y);

#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        printf(" pairs[%i] = %i,%i\n", threadID, pair.x, pair.y);
        __syncthreads();

        printf(" access idx[%i]/idx2[%i]\n", pair.x, pair.y);
#endif

        uint3 idx = d_triIndices1[pair.x];
        uint3 idx2 = d_triIndices2[pair.y];

#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
        printf(" access idx[%i] = %i,%i,%i; idx2[%i] = %i,%i,%i\n", idx.x, idx.y, idx.z, idx2.x, idx2.y, idx2.z);
        __syncthreads();
#endif
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        printf(" access vertices u1/u2/u3\n");
#endif
        u1 = d_vertices1[idx.x].v;
        u2 = d_vertices1[idx.y].v;
        u3 = d_vertices1[idx.z].v;

#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
        printf(" accessed vertices u1 = %f,%f,%f/u2 = %f,%f,%f/u3 = %f,%f,%f\n", u1.x, u1.y, u1.z, u2.x, u2.y, u2.z, u3.x, u3.y, u3.z);
        __syncthreads();
        printf(" access vertices v1/v2/v3\n");
#endif

        v1 = d_vertices2[idx2.x].v;
        v2 = d_vertices2[idx2.y].v;
        v3 = d_vertices2[idx2.z].v;
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
        printf(" accessed vertices v1 = %f,%f,%f/v2 = %f,%f,%f/v3 = %f,%f,%f\n", v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z);
        __syncthreads();
#endif

        // IMPORTANT: MUST CORRESPOND TO static int CollisionTestElementsSize = 15 in ObbTreeGPUCollisionDetection_cuda.h !!!
        triangleContact(u1, u2, u3, v1, v2, v3, alarmDistance, contactDistance, pair.x, pair.y, contacts, threadID, nMaxContacts, 15);
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS
        __syncthreads();
#endif
    }
}







//#define DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
#define DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS_OUTPUTINDEX
__device__ __forceinline__ int triangleContact_RangedResults(float3 p1, float3 p2, float3 p3, float3 q1, float3 q2, float3 q3,
                                                             double alarmDist, double contactDist,
                                                             int index1, int index2,
                                                             gProximityDetectionOutput* contacts, int outputPos,
                                                             int nMaxContacts, int CollisionTestElementsSize,
                                                             unsigned int contactOutputRangeStart,
                                                             unsigned int resultRangeMin, unsigned int resultRangeMax,
                                                             bool flipContactElements,
                                                             unsigned int* binOutputIndex = NULL)
{
    int writePos = outputPos * CollisionTestElementsSize;
    if (writePos < nMaxContacts)
    {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        printf("triangleContact_RangedResults: %i - %i to writePos = %i, outputPos = %i\n", index1, index2, writePos, outputPos);
        __syncthreads();
#endif

        int rangedWritePos = writePos - contactOutputRangeStart;

#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        printf(" rangedWritePos = %i, contactOutputRangeStart = %i; resultRangeMin/Max = %i / %i\n", rangedWritePos, contactOutputRangeStart, resultRangeMin, resultRangeMax);
        __syncthreads();
#endif

        if (rangedWritePos >= resultRangeMin &&
            rangedWritePos < resultRangeMax)
        {
            writePos = rangedWritePos;

#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf(" write to rangedWritePos = %i OK\n", rangedWritePos);
            __syncthreads();
#endif
            const double maxContactDist = alarmDist + contactDist;
            const double dist2 =  maxContactDist * maxContactDist;
            float3 tri1_edge1 = f3v_sub(p2, p1);
            float3 tri1_edge2 = f3v_sub(p3, p1);
            const float3 pn = f3v_cross(tri1_edge1, tri1_edge2);

            float3 tri2_edge1 = f3v_sub(q2, q1);
            float3 tri2_edge2 = f3v_sub(q3, q1);
            const float3 qn = f3v_cross(tri2_edge1, tri2_edge2);

            const int id1 = index1; // index of contacts involving points in e1
            const int id2 = index2; // index of contacts involving points in e2

            int n = 0;

#ifdef GPROXIMITY_TRIANGLE_PAIR_INTERSECTIONS_TRIANGLE_POINT
        int nTriangleContacts = 0;
        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, q1, q2, q3, qn, p1, true, contacts, id1, id2, id1+0, writePos, 0, resultRangeMax, binOutputIndex);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            if (binOutputIndex == NULL)
            {
                printf("   TRIANGLE_POINT contact case 1: p0 = %f,%f,%f, p1 = %f,%f,%f, distance = %f, normal = %f,%f,%f\n",
                       contacts->point0[writePos].x, contacts->point0[writePos].y ,contacts->point0[writePos].z,
                       contacts->point1[writePos].x, contacts->point1[writePos].y ,contacts->point1[writePos].z,
                       contacts->distance[writePos],
                       contacts->normal[writePos].x, contacts->normal[writePos].y ,contacts->normal[writePos].z);
            }
            printf(" ===> n = %i, nTriangleContacts = %i\n", n, nTriangleContacts);
#endif
            if (flipContactElements)
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos] = true;
                }
            }
            else
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos] = false;
                }
            }

            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos].y = 0;
                contacts->elems[writePos].z = -1;
            }
        }

        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, q1, q2, q3, qn, p2, true, contacts, id1, id2, id1+1, writePos + 1, 1, resultRangeMax, binOutputIndex);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   TRIANGLE_POINT contact case 2: p0 = %f,%f,%f, p1 = %f,%f,%f, distance = %f, normal = %f,%f,%f\n",
                   contacts->point0[writePos+1].x, contacts->point0[writePos+1].y ,contacts->point0[writePos+1].z,
                   contacts->point1[writePos+1].x, contacts->point1[writePos+1].y ,contacts->point1[writePos+1].z,
                   contacts->distance[writePos+1],
                   contacts->normal[writePos+1].x, contacts->normal[writePos+1].y ,contacts->normal[writePos+1].z);
            printf(" ===> n = %i, nTriangleContacts = %i\n", n, nTriangleContacts);
#endif
            if (flipContactElements)
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos+1] = true;
                }
            }
            else
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos+1] = false;
                }
            }

            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 1].y = 1;
                contacts->elems[writePos + 1].z = -1;
            }
        }

#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif
        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, q1, q2, q3, qn, p3, true, contacts, id1, id2, id1+2, writePos + 2, 2, resultRangeMax, binOutputIndex);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   TRIANGLE_POINT contact case 3: p0 = %f,%f,%f, p1 = %f,%f,%f, distance = %f, normal = %f,%f,%f\n",
                   contacts->point0[writePos+2].x, contacts->point0[writePos+2].y ,contacts->point0[writePos+2].z,
                   contacts->point1[writePos+2].x, contacts->point1[writePos+2].y ,contacts->point1[writePos+2].z,
                   contacts->distance[writePos+2],
                   contacts->normal[writePos+2].x, contacts->normal[writePos+2].y ,contacts->normal[writePos+2].z);
            printf(" ===> n = %i, nTriangleContacts = %i\n", n, nTriangleContacts);
#endif

            if (flipContactElements)
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos+2] = true;
                }
            }
            else
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos+2] = false;
                }
            }

            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 2].y = 2;
                contacts->elems[writePos + 2].z = -1;
            }
        }
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, p1, p2, p3, pn, q1, false, contacts, id1, id2, id2+0, writePos + 3, 3, resultRangeMax, binOutputIndex);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   TRIANGLE_POINT contact case 4: p0 = %f,%f,%f, p1 = %f,%f,%f, distance = %f, normal = %f,%f,%f\n",
                   contacts->point0[writePos+3].x, contacts->point0[writePos+3].y ,contacts->point0[writePos+3].z,
                   contacts->point1[writePos+3].x, contacts->point1[writePos+3].y ,contacts->point1[writePos+3].z,
                   contacts->distance[writePos+3],
                   contacts->normal[writePos+3].x, contacts->normal[writePos+3].y ,contacts->normal[writePos+3].z);
        printf(" ===> n = %i, nTriangleContacts = %i\n", n, nTriangleContacts);
#endif
            if (flipContactElements)
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos+3] = true;
                }
            }
            else
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos+3] = false;
                }
            }

            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 3].y = -1;
                contacts->elems[writePos + 3].z = 0;
            }
        }

        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, p1, p2, p3, pn, q2, false, contacts, id1, id2, id2+1, writePos + 4, 4, resultRangeMax, binOutputIndex);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   TRIANGLE_POINT contact case 5: p0 = %f,%f,%f, p1 = %f,%f,%f, distance = %f, normal = %f,%f,%f\n",
                   contacts->point0[writePos+4].x, contacts->point0[writePos+4].y ,contacts->point0[writePos+4].z,
                   contacts->point1[writePos+4].x, contacts->point1[writePos+4].y ,contacts->point1[writePos+4].z,
                   contacts->distance[writePos+4],
                   contacts->normal[writePos+4].x, contacts->normal[writePos+4].y ,contacts->normal[writePos+4].z);
            printf(" ===> n = %i, nTriangleContacts = %i\n", n, nTriangleContacts);
#endif

            if (flipContactElements)
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos+4] = true;
                }
            }
            else
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos+4] = false;
                }
            }

            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 4].y = -1;
                contacts->elems[writePos + 4].z = 1;
            }
        }

#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

        nTriangleContacts = doIntersectionTrianglePoint(dist2, contactDist, p1, p2, p3, pn, q3, false, contacts, id1, id2, id2+2, writePos + 5, 5, resultRangeMax, binOutputIndex);
        n += nTriangleContacts;
        if (nTriangleContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   TRIANGLE_POINT contact case 6: p0 = %f,%f,%f, p1 = %f,%f,%f, distance = %f, normal = %f,%f,%f\n",
                   contacts->point0[writePos+5].x, contacts->point0[writePos+5].y ,contacts->point0[writePos+5].z,
                   contacts->point1[writePos+5].x, contacts->point1[writePos+5].y ,contacts->point1[writePos+5].z,
                   contacts->distance[writePos+5],
                   contacts->normal[writePos+5].x, contacts->normal[writePos+5].y ,contacts->normal[writePos+5].z);
            printf(" ===> n = %i, nTriangleContacts = %i\n", n, nTriangleContacts);
#endif

            if (flipContactElements)
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos+5] = true;
                }
            }
            else
            {
                if (binOutputIndex == NULL)
                {
                    contacts->swapped[writePos+5] = false;
                }
            }

            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 5].y = -1;
                contacts->elems[writePos + 5].z = 2;
            }
        }
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif


#endif //GPROXIMITY_TRIANGLE_PAIR_INTERSECTIONS_TRIANGLE_POINT

#ifdef GPROXIMITY_TRIANGLE_PAIR_INTERSECTIONS_LINE_LINE
        int nLineContacts = 0;
        nLineContacts = doIntersectionLineLine(dist2, contactDist, p1, p2, q1, q2, contacts, id1, id2, 0, 0, id2+3, writePos + 6, 0, resultRangeMax, binOutputIndex);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("    LINE_LINE contact case 1: edges %i - %i; p1 = %f,%f,%f, p2 = %f,%f,%f, q1 = %f,%f,%f, q2 = %f,%f,%f\n", 2, 2, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, q1.x, q1.y, q1.z, q2.x, q2.y, q2.z);
            printf(" ===> n = %i, nLineContacts = %i\n", n, nLineContacts);
#endif
            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 6].y = 2;
                contacts->elems[writePos + 6].z = 2;
            }
        }

#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p1, p2, q2, q3, contacts, id1, id2, 0, 1, id2+4, writePos + 7, 1, resultRangeMax, binOutputIndex);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   LINE_LINE contact case 2: edges %i - %i; p1 = %f,%f,%f, p2 = %f,%f,%f, q2 = %f,%f,%f, q3 = %f,%f,%f\n", 2, 0, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, q2.x, q2.y, q2.z, q3.x, q3.y, q3.z);
            printf(" ===> n = %i, nLineContacts = %i\n", n, nLineContacts);
#endif
            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 7].y = 2;
                contacts->elems[writePos + 7].z = 0;
            }
        }

#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p1, p2, q3, q1, contacts, id1, id2, 0, 2, id2+5, writePos + 8, 2, resultRangeMax, binOutputIndex);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   LINE_LINE contact case 3: edges %i - %i; p1 = %f,%f,%f, p2 = %f,%f,%f, q3 = %f,%f,%f, q1 = %f,%f,%f\n", 2, 1, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, q3.x, q3.y, q3.z, q1.x, q1.y, q1.z);
            printf(" ===> n = %i, nLineContacts = %i\n", n, nLineContacts);
#endif
            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 8].y = 2;
                contacts->elems[writePos + 8].z = 1;
            }
        }

#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p2, p3, q1, q2, contacts, id1, id2, 1, 0, id2+6, writePos + 9, 3, resultRangeMax, binOutputIndex);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   LINE_LINE contact case 4: edges %i - %i; p2 = %f,%f,%f, p3 = %f,%f,%f, q1 = %f,%f,%f, q2 = %f,%f,%f\n", 0, 2, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, q1.x, q1.y, q1.z, q2.x, q2.y, q2.z);
            printf(" ===> n = %i, nLineContacts = %i\n", n, nLineContacts);
#endif
            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 9].y = 0;
                contacts->elems[writePos + 9].z = 2;
            }
        }
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p2, p3, q2, q3, contacts, id1, id2, 1, 1, id2+7, writePos + 10, 4, resultRangeMax, binOutputIndex);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   LINE_LINE contact case 5: edges %i - %i; p2 = %f,%f,%f, p3 = %f,%f,%f, q2 = %f,%f,%f, q3 = %f,%f,%f\n", 0, 0, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, q2.x, q2.y, q2.z, q3.x, q3.y, q3.z);
            printf(" ===> n = %i, nLineContacts = %i\n", n, nLineContacts);
#endif
            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 10].y = 0;
                contacts->elems[writePos + 10].z = 0;
            }
        }
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p2, p3, q3, q1, contacts, id1, id2, 1, 2, id2+8, writePos + 11, 5, resultRangeMax, binOutputIndex);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   LINE_LINE contact case 6: edges %i - %i; p2 = %f,%f,%f, p3 = %f,%f,%f, q3 = %f,%f,%f, q1 = %f,%f,%f\n", 0, 1, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, q3.x, q3.y, q3.z, q1.x, q1.y, q1.z);
            printf(" ===> n = %i, nLineContacts = %i\n", n, nLineContacts);
#endif
            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 11].y = 0;
                contacts->elems[writePos + 11].z = 1;
            }
        }
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p3, p1, q1, q2, contacts, id1, id2, 2, 0, id2+9, writePos + 12, 6, resultRangeMax, binOutputIndex);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   LINE_LINE contact case 7: edges %i - %i; p3 = %f,%f,%f, p1 = %f,%f,%f, q1 = %f,%f,%f, q2 = %f,%f,%f\n", 1, 2, p3.x, p3.y, p3.z, p1.x, p1.y, p1.z, q1.x, q1.y, q1.z, q2.x, q2.y, q2.z);
            printf(" ===> n = %i, nLineContacts = %i\n", n, nLineContacts);
#endif
            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 12].y = 1;
                contacts->elems[writePos + 12].z = 2;
            }
        }
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p3, p1, q2, q3, contacts, id1, id2, 2, 1, id2+10, writePos + 13, 7, resultRangeMax, binOutputIndex);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   LINE_LINE contact case 8: edges %i - %i; p3 = %f,%f,%f, p1 = %f,%f,%f, q2 = %f,%f,%f, q3 = %f,%f,%f\n", 1, 0, p3.x, p3.y, p3.z, p1.x, p1.y, p1.z, q2.x, q2.y, q2.z, q3.x, q3.y, q3.z);
            printf(" ===> n = %i, nLineContacts = %i\n", n, nLineContacts);
#endif
            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 13].y = 1;
                contacts->elems[writePos + 13].z = 0;
            }
        }
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

        nLineContacts = doIntersectionLineLine(dist2, contactDist, p3, p1, q3, q1, contacts, id1, id2, 2, 2, id2+11, writePos + 14, 8, resultRangeMax, binOutputIndex);
        n += nLineContacts;
        if (nLineContacts > 0)
        {
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("   LINE_LINE contact case 9: edges %i - %i; p3 = %f,%f,%f, p1 = %f,%f,%f, q3 = %f,%f,%f, q1 = %f,%f,%f\n", 1, 1, p3.x, p3.y, p3.z, p1.x, p1.y, p1.z, q3.x, q3.y, q3.z, q1.x, q1.y, q1.z);
            printf(" ===> n = %i, nLineContacts = %i\n", n, nLineContacts);
#endif
            if (binOutputIndex == NULL)
            {
                contacts->elems[writePos + 14].y = 1;
                contacts->elems[writePos + 14].z = 1;
            }
        }
#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
        __syncthreads();
#endif

#endif //GPROXIMITY_TRIANGLE_PAIR_INTERSECTIONS_LINE_LINE

#ifdef DEBUG_TRIANGLE_CONTACT_RANGEDRESULTS
            printf("triangleContact_RangedResults %i - %i at rangedWritePos = %i: contact points generated = %i\n", index1, index2, rangedWritePos, n);
#endif
            return n;
        }
    }
    return 0;
}





//#define GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
__global__ __forceinline__ void trianglePairIntersections_Streamed(GPUVertex *d_vertices1, uint3 *d_triIndices1, GPUVertex *d_vertices2, uint3* d_triIndices2,
                                                          int2 *pairs, const int nPairs, gProximityDetectionOutput* contacts,
                                                          const unsigned int nVertices_1, const unsigned int nTriangles_1,
                                                          const unsigned int nVertices_2, const unsigned int nTriangles_2,
                                                          const unsigned int contactOutputStartIndex,
                                                          unsigned int* binOutputIndex,
                                                          const unsigned int contactRangeBegin, const unsigned int contactRangeEnd,
                                                          const int nMaxContacts, double alarmDistance, double contactDistance)
{
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED_VERBOSE
    printf("trianglePairIntersections_Streamed: gridDim=%i,%i,%i, blockIdx=%i,%i,%i, threadIdx=%i,%i,%i\n", gridDim.x, gridDim.y, gridDim.z, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    __syncthreads();
#endif
    const int threadID = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED_VERBOSE
    printf(" threadID = %i, nPairs = %i, contactRangeBegin = %i, contactRangeEnd = %i\n", threadID, nPairs, contactRangeBegin, contactRangeEnd);
    __syncthreads();
#endif

    // don't go beyond end of pair list
    // also don't go below or above contactRangeBegin and contactRangeEnd
    if(threadID < nPairs)
    {
        //contacts->valid[threadID] = false;

        float3 u1, u2, u3, v1, v2, v3;
#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
        printf(" trianglePairIntersections_Streamed: nTrianges_1 = %i, nTriangles_2 = %i, nVertices_1 = %i, nVertices_2 = %i\n", nTriangles_1, nTriangles_2, nVertices_1, nVertices_2);
        printf(" threadID = %i, nPairs = %i, contactRangeBegin = %i, contactRangeEnd = %i\n", threadID, nPairs, contactRangeBegin, contactRangeEnd);
        printf(" access pairs[%i]\n", threadID);
        __syncthreads();
#endif
        int2 pair = pairs[threadID];

        pair.x = abs(pair.x);
        pair.y = abs(pair.y);

#ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
        printf(" pairs[%i] = %i,%i\n", threadID, pair.x, pair.y);
        __syncthreads();

        printf(" access idx[%i]/idx2[%i]\n", pair.x, pair.y);
#endif

        if (pair.x < nTriangles_1 && pair.y < nTriangles_2)
        {
    #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
            printf(" TEST CASE 1 -- indices in boundary: %i < %i && %i < %i\n", pair.x, nTriangles_1, pair.y, nTriangles_2);
            __syncthreads();
    #endif

            uint3 idx = d_triIndices1[pair.x];
            uint3 idx2 = d_triIndices2[pair.y];

    #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
            printf(" access idx[%i] = %i,%i,%i; idx2[%i] = %i,%i,%i\n", pair.x, idx.x, idx.y, idx.z, pair.y, idx2.x, idx2.y, idx2.z);
            __syncthreads();
            printf(" TEST CASE 1 -- vertices in boundary : idx.x = %i < %i, idx.y = %i < %i, idx.z = %i < %i, idx2.x = %i < %i, idx2.y = %i < %i, idx2.z = %i < %i\n",
                   idx.x, nVertices_1, idx.y, nVertices_1, idx.z, nVertices_1, idx2.x, nVertices_2, idx2.y, nVertices_2, idx2.z, nVertices_2);
            __syncthreads();
    #endif

            if (idx.x < nVertices_1 && idx.y < nVertices_1 && idx.z < nVertices_1 &&
                idx2.x < nVertices_2 && idx2.y < nVertices_2 && idx2.z < nVertices_2)
            {
        #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
                printf(" access vertices u1/u2/u3\n");
                __syncthreads();
        #endif
                u1 = d_vertices1[idx.x].v;
                u2 = d_vertices1[idx.y].v;
                u3 = d_vertices1[idx.z].v;

        #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
                printf(" accessed vertices u1 = %f,%f,%f/u2 = %f,%f,%f/u3 = %f,%f,%f\n", u1.x, u1.y, u1.z, u2.x, u2.y, u2.z, u3.x, u3.y, u3.z);
                __syncthreads();
                printf(" access vertices v1/v2/v3\n");
                __syncthreads();
        #endif

                v1 = d_vertices2[idx2.x].v;
                v2 = d_vertices2[idx2.y].v;
                v3 = d_vertices2[idx2.z].v;
        #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
                printf(" accessed vertices v1 = %f,%f,%f/v2 = %f,%f,%f/v3 = %f,%f,%f\n", v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z);
                __syncthreads();
        #endif

                // IMPORTANT: MUST CORRESPOND TO static int CollisionTestElementsSize = 15 in ObbTreeGPUCollisionDetection_cuda.h !!!
                triangleContact_RangedResults(u1, u2, u3, v1, v2, v3,
                                              alarmDistance, contactDistance, pair.x, pair.y,
                                              contacts, threadID, nMaxContacts, 15,
                                              contactOutputStartIndex,
                                              contactRangeBegin, contactRangeEnd, false, binOutputIndex);
        #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
                __syncthreads();
        #endif
            }
        }
        else if (pair.y < nTriangles_1 && pair.x < nTriangles_2)
        {
    #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
            printf(" TEST CASE 2 -- indices in boundary: %i < %i && %i < %i\n", pair.y, nTriangles_1, pair.x, nTriangles_2);
            __syncthreads();
    #endif
            uint3 idx = d_triIndices2[pair.x];
            uint3 idx2 = d_triIndices1[pair.y];

    #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
            printf(" TEST CASE 2 -- vertices in boundary: idx2.x = %i < %i, idx2.y = %i < %i, idx2.z = %i < %i, idx.x = %i < %i, idx.y = %i < %i, idx.z = %i < %i\n",
                   idx2.x, nVertices_2, idx2.y, nVertices_2, idx2.z, nVertices_2, idx.x, nVertices_1, idx.y, nVertices_1, idx.z, nVertices_1);
            __syncthreads();
            printf(" access idx[%i] = %i,%i,%i; idx2[%i] = %i,%i,%i\n", pair.y, idx.x, idx.y, idx.z, pair.x, idx2.x, idx2.y, idx2.z);
            __syncthreads();
    #endif
            if (idx2.x < nVertices_1 && idx2.y < nVertices_1 && idx2.z < nVertices_1 &&
                idx.x < nVertices_2 && idx.y < nVertices_2 && idx.z < nVertices_2)
            {
        #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
                printf(" access vertices u1/u2/u3\n");
                __syncthreads();
        #endif
                u1 = d_vertices2[idx.x].v;
                u2 = d_vertices2[idx.y].v;
                u3 = d_vertices2[idx.z].v;

        #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
                printf(" accessed vertices u1 = %f,%f,%f/u2 = %f,%f,%f/u3 = %f,%f,%f\n", u1.x, u1.y, u1.z, u2.x, u2.y, u2.z, u3.x, u3.y, u3.z);
                __syncthreads();
                printf(" access vertices v1/v2/v3\n");
                __syncthreads();
        #endif

                v1 = d_vertices1[idx2.x].v;
                v2 = d_vertices1[idx2.y].v;
                v3 = d_vertices1[idx2.z].v;
        #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
                printf(" accessed vertices v1 = %f,%f,%f/v2 = %f,%f,%f/v3 = %f,%f,%f\n", v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z);
                __syncthreads();
        #endif

                // IMPORTANT: MUST CORRESPOND TO static int CollisionTestElementsSize = 15 in ObbTreeGPUCollisionDetection_cuda.h !!!
                triangleContact_RangedResults(u1, u2, u3, v1, v2, v3,
                                              alarmDistance, contactDistance, pair.y, pair.x,
                                              contacts, threadID, nMaxContacts, 15,
                                              contactOutputStartIndex,
                                              contactRangeBegin, contactRangeEnd, true, binOutputIndex);
        #ifdef GPROXIMITY_DEBUG_TRIANGLE_PAIR_INTERSECTIONS_STREAMED
                __syncthreads();
        #endif
            }
        }
    }
}


#endif
