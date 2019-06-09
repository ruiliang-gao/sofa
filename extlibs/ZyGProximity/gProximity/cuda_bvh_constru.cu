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
 *  Chapel Hill, N.C. 27599-3175
 *  United States of America
 *  
 *  http://gamma.cs.unc.edu/GPUCOL/
 *  
 */
 
#include "cuda_bvh_constru.h"

#include "global_objects.h"

#include "cuda_defs.h"
#include "radixsort.h"
#include "cuda_timer.h"
#include "cuda_make_grid.h"
#include "split_compaction.h"
#include "split_sort.h"

#include <cuda_gl_interop.h>
#include <texture_types.h>
#include <stdio.h>

#include "bvh_kernels.h"

// #include "bvh_kernels_split.cu"

#define STORE_NODES(threadOffset,treeOffset, tree, buffer) do { if (threadOffset < 16) ((float *)&tree[treeOffset])[threadOffset] = ((float *)buffer)[threadOffset]; } while(0);

#define POP_QUEUE(outItem, queue, queuePtr, threadOffset) do { if (threadOffset < 10) ((int *)&outItem)[threadOffset] = ((int *)&queue[queuePtr])[threadOffset]; queuePtr--; } while(0);


void checkBVHSanity(int *indexlist, int g_nTris, bool printList)
{
    printf("######### SANITY CHECK ###############\n");
    bool hasError = false;
	
    int *ids = (int *)malloc(sizeof(int) * g_nTris);
    int *id_hist = (int *)malloc(sizeof(int) * g_nTris);
	
    memset(id_hist, 0, sizeof(int)*g_nTris);
	
    CUDA_SAFE_CALL(cudaMemcpy(ids, indexlist, g_nTris * sizeof(int), cudaMemcpyDeviceToHost));
	
    if(printList)
        printf("[ ");
		
    for(int i = 0; i < g_nTris; i++)
    {
        if(printList)
            printf("%d ", ids[i]);
        if(ids[i] < 0 || ids[i] >= g_nTris)
        {
            printf("Error: invalid ID %d referenced at idx %d!\n", ids[i], i);
            hasError = true;
        }
        else
            id_hist[ids[i]]++;
    }
	
    if(printList)
        printf("]\n");
		
    for(int i = 0; i < g_nTris; i++)
    {
        if(id_hist[i] == 0)
        {
            printf("Error: ID %d not referenced!\n", i);
            hasError = true;
        }
        else if(id_hist[i] > 1)
        {
            printf("Error: ID %d referenced %d times!\n", i, id_hist[i]);
            hasError = true;
        }
    }
	
    free(ids);
    free(id_hist);
    if(hasError)
        getchar();
}



