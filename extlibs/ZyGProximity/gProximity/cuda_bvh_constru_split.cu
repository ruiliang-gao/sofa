#include "bvh_kernels_split.cu"

#ifndef STORE_NODES
#define STORE_NODES(threadOffset,treeOffset, tree, buffer) do { if (threadOffset < 16) ((float *)&tree[treeOffset])[threadOffset] = ((float *)buffer)[threadOffset]; } while(0);
#endif

#ifndef POP_QUEUE
#define POP_QUEUE(outItem, queue, queuePtr, threadOffset) do { if (threadOffset < 10) ((int *)&outItem)[threadOffset] = ((int *)&queue[queuePtr])[threadOffset]; queuePtr--; } while(0);
#endif

#ifndef __CUDA_BVH_CONSTRU_SPLIT_STORE_SPLIT__
#define __CUDA_BVH_CONSTRU_SPLIT_STORE_SPLIT__
template <int nSplitThreads>
__global__ void makeSplit(BVHConstructionState BVH, const int nActiveSplits, const int nTrisAtLeafs)
{
        __shared__ int sharedArray[6*nSplitThreads];

        __shared__ float splitPoint;
        __shared__ AABB nodeBB;

        __shared__ int bestAxis;
        __shared__ AABBSplit inSplit;
        __shared__ AABBNode outBuffer[2];

        const int myOffset = gridDim.x * blockIdx.y + blockIdx.x;
        if(myOffset >= nActiveSplits)
                return;

        int threadOffset = threadIdx.y * blockDim.x + threadIdx.x;

        if(threadOffset == 0)
        {
                inSplit = BVH.inputSplits[BVH.splitIndexTable[myOffset]];
        }
        CONDOREMU_SYNC(nSplitThreads > WARP_SIZE);

        int tsz = inSplit.right - inSplit.left + 1;	// total number of triangles

        //
        // Initialization: (primary thread only)
        // Reads in split information and writes it to parent node,
        // sets up the SAH sampling information.
        //

        if(threadOffset == 0)
        {

                nodeBB = BVH.tree[inSplit.myIndex].bbox;

                // find biggest axis
                nodeBB.bb_max.x -= nodeBB.bb_min.x;
                nodeBB.bb_max.y -= nodeBB.bb_min.y;
                nodeBB.bb_max.z -= nodeBB.bb_min.z;

                bestAxis = (nodeBB.bb_max.x > nodeBB.bb_max.y) ? 0 : 1;
                bestAxis = (nodeBB.bb_max.z > max(nodeBB.bb_max.x, nodeBB.bb_max.y)) ? 2 : bestAxis;

                // set splitpoint = object median
                if(bestAxis == 0)
                        splitPoint = nodeBB.bb_min.x + 0.5f * nodeBB.bb_max.x;
                else if(bestAxis == 1)
                        splitPoint = nodeBB.bb_min.y + 0.5f * nodeBB.bb_max.y;
                else
                        splitPoint = nodeBB.bb_min.z + 0.5f * nodeBB.bb_max.z;



                // new child pointer (relative address in bytes):
                BVH.tree[inSplit.myIndex].lChild = ((inSplit.nextIndex - inSplit.myIndex) << 5) | bestAxis;
        }



        //
        // PART 1: sort triangle into left and right half according to
        // split axis and coordinate.
        //

        int numLeft = 0;						// number of primitives on left side


        int nScanTrisLeft = tsz,				// number of primitives still to process
            inputOffsetLeft = inSplit.left,		// start offset in output triID[] to process
            inputOffsetRight = inSplit.right;	// end offset in output triID[] to process

        // make pointers to arrays we need in shared memory
        int *shared_localTriIDs = &sharedArray[0];
        int *shared_left = &sharedArray[2*nSplitThreads];

        while(nScanTrisLeft > 0)
        {
                splitSort_Warp<nSplitThreads>(shared_localTriIDs, shared_left, BVH.triIDs, bestAxis, splitPoint, threadOffset, nScanTrisLeft, inputOffsetLeft, inputOffsetRight, numLeft);
        }


        //
        // PART 2: (primary thread only)
        // Store results in BVH nodes and write out resulting splits
        //

        __syncthreads();

        // special case: subdivision did not work out, just go half/half
        if(numLeft == 0 || numLeft == tsz)
        {
                numLeft = tsz / 2;
        }

        // local shared memory for storing boxes
        float *localBoxes = (float *)sharedArray;

        // LEFT child: calculate BB
        int nBoxesToProcess = numLeft;
        int boxOffset = inSplit.left;

#include "bvh_parallelupdate.cu"

        storeSplit<nSplitThreads, 0>(threadOffset, myOffset, numLeft, inSplit, localBoxes, BVH.outputSplits, BVH.outputSplitMask, BVH.tree, BVH.triIDs, outBuffer, nTrisAtLeafs);
        __syncthreads();

        // RIGHT child: calculate BB
        nBoxesToProcess = tsz - numLeft;
        boxOffset = inSplit.left + numLeft;

#include "bvh_parallelupdate.cu"

        storeSplit<nSplitThreads, 1>(threadOffset, myOffset, numLeft, inSplit, localBoxes, BVH.outputSplits, BVH.outputSplitMask, BVH.tree, BVH.triIDs, outBuffer, nTrisAtLeafs);
        __syncthreads();

        STORE_NODES(threadOffset, inSplit.nextIndex, BVH.tree, outBuffer);
}
#endif //__CUDA_BVH_CONSTRU_SPLIT_STORE_SPLIT__

#ifndef __CUDA_BVH_CONSTRU_SPLIT_STORE_SPLIT_LOCAL__
#define __CUDA_BVH_CONSTRU_SPLIT_STORE_SPLIT_LOCAL__
template <int nSplitThreads, int nMaxTris>
__global__ void makeSplitLocal(BVHConstructionState BVH, const unsigned int nActiveSplits, const int nTrisAtLeafs)
{
        __shared__ int sharedArray[6*nSplitThreads];

        __shared__ float splitPoint;
        __shared__ WorkQueueItem workItem;
        __shared__ int bestAxis;
        __shared__ AABBNode outBuffer[2];
        __shared__ int globalLeft, globalRight;

        // cached data from global memory:
        __shared__ float localBoundingBoxes[nMaxTris*6];
        __shared__ int localTriIDs[nMaxTris];
        __shared__ int localToGlobalTriIDs[nMaxTris];

        // work queue data
        __shared__ WorkQueueItem workQueue[nMaxTris];
        __shared__ int workQueueIdx;

        const int myOffset = gridDim.x * blockIdx.y + blockIdx.x;

        if(myOffset >= nActiveSplits)
                return;

        int threadOffset = threadIdx.y * blockDim.x + threadIdx.x;

        //
        // set up work queue
        //

        // read in initial split information
        if(threadOffset == 0)
        {
                workQueueIdx = 0;
                const AABBSplit tempSplit = BVH.smallSplits[myOffset];
                globalLeft = tempSplit.left;
                globalRight = tempSplit.right;

                workQueue[0].split.left = 0;
                workQueue[0].split.right = tempSplit.right - tempSplit.left;
                workQueue[0].split.myIndex = tempSplit.myIndex;
                workQueue[0].split.nextIndex = tempSplit.nextIndex;
        }

        // read parent node BB
        if(threadOffset < 6)
                ((float *)&workQueue[0].nodeBB)[threadOffset] = ((float *) & BVH.tree[workQueue[0].split.myIndex].bbox)[threadOffset];

        __syncthreads();

        //
        // read in complete geometry information from global memory:
        //
        int startOffset = globalLeft + threadOffset;
        if(startOffset <= globalRight)
        {
                // read triangle ID
                localTriIDs[threadOffset] = threadOffset;
                const int triID = BVH.triIDs[startOffset];
                localToGlobalTriIDs[threadOffset] = triID;

                // read bounding box

                // X
                float2 box = AABB_GET_AXIS(triID, 0);
                localBoundingBoxes[0*nMaxTris + threadOffset] = box.x;
                localBoundingBoxes[3*nMaxTris + threadOffset] = box.y;

                // Y
                box = AABB_GET_AXIS(triID, 1);
                localBoundingBoxes[1*nMaxTris + threadOffset] = box.x;
                localBoundingBoxes[4*nMaxTris + threadOffset] = box.y;

                // Z
                box = AABB_GET_AXIS(triID, 2);
                localBoundingBoxes[2*nMaxTris + threadOffset] = box.x;
                localBoundingBoxes[5*nMaxTris + threadOffset] = box.y;
        }

        //
        // while items in local work queue:
        //
        while(workQueueIdx >= 0)
        {

                // pop last element from work queue
                POP_QUEUE(workItem, workQueue, workQueueIdx, threadOffset);

                int tsz = workItem.split.right - workItem.split.left + 1;	// total number of triangles

                if(tsz > 2)
                {

                        if(threadOffset < 3)  // workItem.nodeBB.bb_max -= workItem.nodeBB.bb_min
                                ((float *)&workItem.nodeBB)[threadOffset + 3] -= ((float *) & workItem.nodeBB)[threadOffset];
                        __syncthreads();

                        if(threadOffset == 0)
                        {

                                //printf("[%d] Split(%d, %d):[%.2f %.2f %.2f]-[%.2f %.2f %.2f]\n", myOffset, inSplit.left, inSplit.right, nodeBB.bb_min.x, nodeBB.bb_min.y, nodeBB.bb_min.z, nodeBB.bb_max.x, nodeBB.bb_max.y, nodeBB.bb_max.z);
                                bestAxis = (workItem.nodeBB.bb_max.x > workItem.nodeBB.bb_max.y) ? 0 : 1;
                                bestAxis = (workItem.nodeBB.bb_max.z > max(workItem.nodeBB.bb_max.x, workItem.nodeBB.bb_max.y)) ? 2 : bestAxis;


                                // set split point = box median
                                if(bestAxis == 0)
                                        splitPoint = workItem.nodeBB.bb_min.x + 0.5f * workItem.nodeBB.bb_max.x;
                                else if(bestAxis == 1)
                                        splitPoint = workItem.nodeBB.bb_min.y + 0.5f * workItem.nodeBB.bb_max.y;
                                else
                                        splitPoint = workItem.nodeBB.bb_min.z + 0.5f * workItem.nodeBB.bb_max.z;

                                // new child pointer (relative address in bytes):
                                BVH.tree[workItem.split.myIndex].lChild = ((workItem.split.nextIndex - workItem.split.myIndex) << 5) | bestAxis;
                        }
                        __syncthreads();


                        //
                        // PART 1: sort triangle into left and right half according to
                        // split axis and coordinate.
                        //

                        int numLeft = splitSortShared_Local(sharedArray, &localTriIDs[workItem.split.left], localBoundingBoxes, nMaxTris, bestAxis, splitPoint, threadOffset, tsz);

                        //
                        // PART 2: (primary thread only)
                        // Store results in BVH nodes and write out resulting splits
                        //


                        if(numLeft == 0 || numLeft == tsz)
                                numLeft = tsz / 2;

                        __syncthreads();

                        // local shared memory for storing boxes
                        float *localBoxes = (float *)sharedArray;

                        // LEFT child: calculate BB
                        int nBoxesToProcess = numLeft;
                        int boxOffset = workItem.split.left;
#include "bvh_parallelupdate_local.cu"
                        storeSplitLocal<nSplitThreads, 0>(threadOffset, myOffset, numLeft, workItem.split, localBoxes, workQueue, workQueueIdx, localTriIDs, localToGlobalTriIDs, outBuffer[0], nTrisAtLeafs);

                        // RIGHT child: calculate BB
                        nBoxesToProcess = tsz - numLeft;
                        boxOffset += numLeft;
#include "bvh_parallelupdate_local.cu"
                        storeSplitLocal<nSplitThreads, 1>(threadOffset, myOffset, numLeft, workItem.split, localBoxes, workQueue, workQueueIdx, localTriIDs, localToGlobalTriIDs, outBuffer[1], nTrisAtLeafs);

                        // save the two nodes off to global tree
                        STORE_NODES(threadOffset, workItem.split.nextIndex, BVH.tree, outBuffer);
                }
                else
                {
                        if(threadOffset < 2)
                        {
                                //
                                // only two primitives in this split:
                                // do not even run the full routine for finding a split, just sort them right here
                                // and make them leaves right away. run for both children in parallel on two threads.
                                //

                                // new child pointer (relative address in bytes):
                                if(threadOffset == 0)
                                        BVH.tree[workItem.split.myIndex].lChild = ((workItem.split.nextIndex - workItem.split.myIndex) << 5);

                                const int index = workItem.split.left + threadOffset;
                                const int triID = localTriIDs[index];

                                // get bounds for this triangle
                                outBuffer[threadOffset].bbox.bb_min.x = localBoundingBoxes[triID];
                                outBuffer[threadOffset].bbox.bb_min.y = localBoundingBoxes[triID +   nMaxTris];
                                outBuffer[threadOffset].bbox.bb_min.z = localBoundingBoxes[triID + 2*nMaxTris];
                                outBuffer[threadOffset].bbox.bb_max.x = localBoundingBoxes[triID + 3*nMaxTris];
                                outBuffer[threadOffset].bbox.bb_max.y = localBoundingBoxes[triID + 4*nMaxTris];
                                outBuffer[threadOffset].bbox.bb_max.z = localBoundingBoxes[triID + 5*nMaxTris];

                                // make leaf:
                                if(nTrisAtLeafs == 0)    // one tri/leaf: store triID
                                {
                                        outBuffer[threadOffset].leafTriID = (localToGlobalTriIDs[triID] << 2) | 3;
                                }
                                else   // version w/ multiple primitives per leaf: store index range
                                {
                                        outBuffer[threadOffset].leafTriID = ((globalLeft + index) << 2) | 3;
                                        outBuffer[threadOffset].right = (globalLeft + index);
                                }
                        }
                        __syncthreads();

                        // save the two nodes off to global tree
                        STORE_NODES(threadOffset, workItem.split.nextIndex, BVH.tree, outBuffer);
                }

                // end of work queue loop, sync for loop condition test
                __syncthreads();
        }

        //
        // done with construction, write back modified index list
        //

        if(nTrisAtLeafs > 0)
        {
                int startOffset = globalLeft + threadOffset;
                int endOffset = globalRight;
                int localOffset = threadOffset;

                while(startOffset <= endOffset)
                {
                        BVH.triIDs[startOffset] = localToGlobalTriIDs[localTriIDs[localOffset]];

                        localOffset += nSplitThreads;
                        startOffset += nSplitThreads;
                }
        }
}
#endif //__CUDA_BVH_CONSTRU_SPLIT_STORE_SPLIT_LOCAL__
