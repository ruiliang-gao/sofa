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
 
template<unsigned int logB>
__global__ void diff_bits_w_nbr_kernel(
                                                                           unsigned int *codes,
                                                                           size_t N,
                                                                           unsigned int *level,
                                                                           unsigned int *split_pos,
                                                                           unsigned int *spaces)
{
        // My index
        unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        // My right neighbors index
        unsigned int tid_r = tid + 1;

        // This is the mask 0011 1000 0000 0000 0000 0000 0000 0000
        // Used to pick the first 3 bits and then so on and so forth
        // This actually assumes that logB is 3 - needs to be fixed
        //unsigned int mask = 0x38000000;
        //unsigned int mask = 1 << 29; /// changed for binary -- CL
        unsigned int mask = ((1 << logB) - 1) << (30 - logB);

        // We may spawn more threads than we need so that all blocks
        // of the same size - so check if we are within valid range
        if (tid < N)
        {
                unsigned int differPos = 0;

                // Go over every "octet" (octet only if branching factor
                // is 8 - in the general case this would be
                // log_2 (branching factor) set of bits) and see at
                // which octet do the Z codes of two neighboring triangles
                // differ. One important thing to note that in the specific
                // case of 30 bit Z values and 3 bit octets - there are 10
                // octets in the Z value. However I am only going over 9
                // (that is the value of maxDepthToCheck in this case) octets
                // This is because each leaf in Bvh should have all those faces
                // whose Z codes match in the first 9 octets. This will
                // hold in general.
                for (unsigned int i = 0; i < maxDepthToCheck; ++i)
                {
                        // My Z code
                        IndexCode p = codes[tid];

                        // My neighbors Z code
                        IndexCode p_r = codes[tid_r];

                        // Bits in my i-th octet
                        unsigned int l_bits = p & mask;

                        // Bits in my neighbors i-th octet
                        unsigned int r_bits = p_r & mask;

                        // If they differ then we have found the octet in which
                        // me and my neighbor differ.
                        // Else increment the octet number and shift the mask
                        // so that I select the next octet in the next iteration
                        if (l_bits != r_bits)
                        {
                                break;
                        }
                        else
                        {
                                differPos++;
                                mask = mask >> logB;
                        }
                }

                // This is the octet at which Z codes of two neighboring
                // faces differ.
                level[tid] = differPos;

                // Position of the split in the array. This is the index
                // of the left face in each pair
                split_pos[tid] = tid;

                // For each split how many more splits do we have to
                // generate in the lower levels of the tree
                spaces[tid] = (maxDepthToCheck - differPos);
        }
        __syncthreads();
}

template <typename ConstructionState>
void gpu_diff_bits_w_nbr(const ConstructionState *on_device, unsigned int *level,
                                                 unsigned int *split_pos, unsigned int *spaces)
{
        /// CL
        diff_bits_w_nbr_kernel<LOG_NBRANCH> <<< BLOCKING(on_device->nElements - 1, 256) >>>
                (on_device->zCodes, on_device->nElements - 1, level, split_pos, spaces);

        cudaThreadSynchronize();
        check_cuda_error("diff_bits_w_nbr_kernel");

}
