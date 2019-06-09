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

#include "global_objects.h"
#include "cuda_rss_constru.h"
#include "cuda_defs.h"

#include <cutil/cutil.h>
#include <cuda_gl_interop.h>
#include <texture_types.h>
#include <stdio.h>

#include "bvh_kernels.h"
#include "cuda_timer.h"

__host__ void RSSConstructionState::allocate(int nTris, int nVerts)
{
        nElements = nTris;
        nVertices = nVerts;

        GPUMALLOC((void**)&triIDs, sizeof(int) * nElements);

        GPUMALLOC((void**)&zCodes, sizeof(unsigned int) * nElements);
        GPUMALLOC((void**)&d_AABBTexData, nElements * 3 * sizeof(float2));
}

__host__ void RSSConstructionState::freeNonEssential()
{
        if (zCodes)
        {
                CUDA_SAFE_CALL(cudaFree(zCodes));
                zCodes = 0;
        }

        if(d_AABBTexData)
        {
                CUDA_SAFE_CALL(cudaUnbindTexture(g_tex_AABBs));
                CUDA_SAFE_CALL(cudaFree(d_AABBTexData));
                d_AABBTexData = NULL;
        }
}

__host__ void RSSConstructionState::free()
{
        if(triIDs)
        {
                GPUFREE(triIDs);
                triIDs = NULL;
        }
}

__host__ void RSSConstructionState::constructAABBsIndexed(const GPUVertex *d_Vertices, uint3 *d_TriIndices, const float3 &bb_min, const float3 &bb_max)
{
        int nBlocks = (int)ceilf(nElements / (float)AABB_BLOCK_SIZE);
	
        // calculate dimensions for storing all AABBs in a single 2-D texture
        int tex_width = nElements * 3;
        int tex_offset = nElements;
	
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_tex_AABBs_offset, &tex_offset, sizeof(int)));
	
        dim3 grid(nBlocks, 1, 1);
        dim3 threads(AABB_BLOCK_SIZE, 1, 1);
	
        // execute kernel for generating the AABBs from triangles
        generateAABBsIndexed <<< nBlocks, AABB_BLOCK_SIZE  >>>(d_AABBTexData, triIDs, d_Vertices, d_TriIndices, zCodes, nElements, tex_offset, bb_min, bb_max);
	
        cudaThreadSynchronize();
	
        // check if kernel execution generated an error
        CUT_CHECK_ERROR("AABB generation kernel execution failed");
	
        // allocate CUDA array for storing AABBs
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
	
        // set up texture
        g_tex_AABBs.filterMode = cudaFilterModePoint;
        g_tex_AABBs.normalized = false;
        g_tex_AABBs.channelDesc = channelDesc;
        CUDA_SAFE_CALL(cudaBindTexture(NULL, &g_tex_AABBs, d_AABBTexData, &channelDesc, tex_width * sizeof(float2)));
}

__host__ void RSSConstructionState::updateGeometry(const float3& bb_min, const float3& bb_max)
{
        static int triIndexListInitialized = 0;
	
        if(!triIndexListInitialized)
        {
                triIndexListInitialized = 1;
        }
	
        // map VBOs to CUDA addresses
        GPUVertex *d_GL_vertices = NULL;
        uint3 *d_GL_triIndices = NULL;
	
        CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_GL_vertices, bufferVertices));
        CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_GL_triIndices, bufferTriIndices));
	
        // construct AABBs from triangles and store in texture
        constructAABBsIndexed(d_GL_vertices, d_GL_triIndices, bb_min, bb_max);
	
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject(bufferVertices));
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject(bufferTriIndices));
}

#include "radixsort.h"

__host__ void RSSConstructionState::updateGeometry_wobb(float3 &bb_min, float3 &bb_max)
{
        static int triIndexListInitialized = 0;
	
        if(!triIndexListInitialized)
        {
                triIndexListInitialized = 1;
        }
	
        // map VBOs to CUDA addresses
        GPUVertex *d_GL_vertices = NULL;
        uint3 *d_GL_triIndices = NULL;
	
        CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_GL_vertices, bufferVertices));
        CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_GL_triIndices, bufferTriIndices));
	
	
        float* v = NULL;
        GPUMALLOC((void**)&v, sizeof(float) * nVertices);
        int nBlocks = (int)ceilf(nVertices / (float)AABB_BOUNDINGBOX_THREADS);
        dim3 grid(nBlocks, 1, 1);
        dim3 threads(AABB_BOUNDINGBOX_THREADS, 1, 1);
	
        float tmp;
        nvRadixSort::RadixSort radixSort(nVertices, true);
	
        copyGPUVertex <<< grid, threads>>>(d_GL_vertices, v, nVertices, nVertices, true, 0);
        radixSort.sort(v, NULL, nVertices, 32, true);
        FROMGPU(&tmp, v, sizeof(float));
        bb_min.x = tmp;
        FROMGPU(&tmp, v + nVertices - 1, sizeof(float));
        bb_max.x = tmp;
	
        copyGPUVertex <<< grid, threads>>>(d_GL_vertices, v, nVertices, nVertices, true, 1);
        radixSort.sort(v, NULL, nVertices, 32, true);
        FROMGPU(&tmp, v, sizeof(float));
        bb_min.y = tmp;
        FROMGPU(&tmp, v + nVertices - 1, sizeof(float));
        bb_max.y = tmp;
	
        copyGPUVertex <<< grid, threads>>>(d_GL_vertices, v, nVertices, nVertices, true, 2);
        radixSort.sort(v, NULL, nVertices, 32, true);
        FROMGPU(&tmp, v, sizeof(float));
        bb_min.z = tmp;
        FROMGPU(&tmp, v + nVertices - 1, sizeof(float));
        bb_max.z = tmp;
	
	
        GPUFREE(v);
	
	
        // construct AABBs from triangles and store in texture
        constructAABBsIndexed(d_GL_vertices, d_GL_triIndices, bb_min, bb_max);
	
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject(bufferVertices));
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject(bufferTriIndices));
}

__host__ void RSSConstructionState::reorderByLevel(LBVH& lbvh)
{
        unsigned int offsetList[200]; // temporary list of offsets
        BVNode *treeIn, *treeOut;
	
        // allocate trees
        printf("Reordering tree by level...\n");
        treeIn = (BVNode *)malloc(sizeof(BVNode) * lbvh.numNodes);
        treeOut = (BVNode *)malloc(sizeof(BVNode) * (2 * nElements - 1));
	
        // read back tree from GPU
        FROMGPU(treeIn, lbvh.Nodes, sizeof(BVNode) * lbvh.numNodes);
	
        uint2 *nodeList[2];
        unsigned int listSize[2];
        nodeList[0] = (uint2 *)malloc(sizeof(uint2) * nElements);
        nodeList[1] = (uint2 *)malloc(sizeof(uint2) * nElements);
        nodeList[0][0] = make_uint2(0, 0); // init with root node
        listSize[0] = 1;
        nLevels = 0;

        printf("nElements %d\n", nElements);
        printf("numNodes %d\n", lbvh.numNodes);

        // reorder nodes:
        unsigned int listIn = 0, listOut = 1;
        unsigned int curOffset = 0;
        offsetList[0] = 0;
        while (listSize[listIn])
        {
                nLevels++;
                unsigned int childOffset = curOffset + listSize[listIn];
                offsetList[nLevels] = childOffset;
                listSize[listOut] = 0;
		
                // for each node in this level
                for (unsigned int i = 0; i < listSize[listIn]; i++)
                {
                        int offset = nodeList[listIn][i].x;
                        BVNode node = treeIn[offset];
			
                        // set parent pointer
                        node.parent = nodeList[listIn][i].y;
			
                        unsigned int childBegin = 0;
                        unsigned int childEnd = 0;
			
                        if (node.children.end != node.children.begin)
                        {
                                // push back children for next level
                                for (int j = node.children.begin; j < node.children.end; j++)
                                {
                                        nodeList[listOut][listSize[listOut]] = make_uint2(j, curOffset);
                                        if (j == node.children.begin) childBegin = listSize[listOut] + childOffset;
                                        if (j == node.children.end - 1) childEnd = listSize[listOut] + childOffset + 1;
                                        listSize[listOut]++;
                                }
                                node.children.begin = childBegin;
                                node.children.end = childEnd;
                        }
                        node.level = nLevels - 1;
                        treeOut[curOffset++] = node; // write node to new tree
                }
		
                listIn = 1 - listIn;
                listOut = 1 - listOut;
        }

        printf("?");
	
        // copy offset list to GPU
        if (levelOffsetList == 0)
                levelOffsetList = (unsigned int *)malloc((nLevels + 1) * sizeof(unsigned int));
        memcpy(levelOffsetList, offsetList, (nLevels + 1)*sizeof(unsigned int));
        printf("?");
        // copy new tree to GPU
        lbvh.numNodes = curOffset;
        GPUMALLOC((void**)&lbvh_tree, sizeof(BVNode) * curOffset);
        TOGPU(lbvh_tree, treeOut, sizeof(BVNode) * curOffset);
        printf("?");
        printf("Done. %d levels in tree.\n", nLevels);
	
        ::free(treeIn);
        ::free(treeOut);
        ::free(nodeList[0]);
        ::free(nodeList[1]);
}


__host__ double RSSConstructionState::build(bool trisStoredAsEdges, bool dumpStats, bool freeConstructionState)
{
        //how do i do deformable here?
        TimerValue lbvh_construction;
        LBVH lbvh;
        lbvh_construction.start();
        lbvh = gpu_create_tree_rss(this);
        lbvh_construction.stop();
        printf("LBVH time: %f\n", lbvh_construction.getElapsedMs());
        printf("building tree nodes: %i\n", (int)(lbvh.numNodes));
        printf("building tree splits: %i\n", (int)(lbvh.numSplits));
        //if (freeConstructionState)
        //{
        //	freeNonEssential();
        //}

        reorderByLevel(lbvh);

        return lbvh_construction.getElapsed();
}

__host__ void RSSConstructionState::registerGeometryBuffers()
{
        // register shared GL triangle data for CUDA use
        CUDA_SAFE_CALL(cudaGLRegisterBufferObject(bufferVertices));
        CUDA_SAFE_CALL(cudaGLRegisterBufferObject(bufferTriIndices));
}

__host__ void RSSConstructionState::releaseGeometryBuffers()
{
        // unregister shared GL triangle data for CUDA use
        CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(bufferVertices));
        CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(bufferTriIndices));
}

__host__ void RSSConstructionState::mapGeometryBuffers()
{
        if(vertexPointer == NULL)
        {
                CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&vertexPointer, bufferVertices));
        }
        if(triIdxPointer == NULL)
        {
                CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&triIdxPointer, bufferTriIndices));
        }
}

__host__ void RSSConstructionState::unmapGeometryBuffers()
{
        if(vertexPointer)
        {
                CUDA_SAFE_CALL(cudaGLUnmapBufferObject(bufferVertices));
                vertexPointer = NULL;
        }
        if(triIdxPointer)
        {
                CUDA_SAFE_CALL(cudaGLUnmapBufferObject(bufferTriIndices));
                triIdxPointer = NULL;
        }
}
