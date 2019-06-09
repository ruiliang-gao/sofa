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
 
#ifndef __CUDA_INTERSECT_NODES_H_
#define __CUDA_INTERSECT_NODES_H_

#include <stdio.h>

#include "cuda_obb.h"
#include "cuda_aabb.h"
#include <vector_functions.h>

#include "ObbTreeGPU_LinearAlgebra.cuh"

#define OBB_ROTATION_MATRIX_EPSILON 0.000001f

template <class BV>
__device__ __inline__ bool intersect(const BV &node1, const BV &node2, bool useMinDimension, double alarmDistance)
{
	return true;
}

// specialization: AABB - AABB intersect
template <>
__device__ __inline__ bool intersect<AABB>(const AABB &node1, const AABB &node2, bool useMinDimension, double alarmDistance)
{
	if((node1.bb_min.x > node2.bb_max.x || node1.bb_min.y > node2.bb_max.y || node1.bb_min.z > node2.bb_max.z)
	        || (node1.bb_max.x < node2.bb_min.x || node1.bb_max.y < node2.bb_min.y || node1.bb_max.z < node2.bb_min.z))
		return false;
	else
		return true;
}

/*#define TR_INV_MULT_ROW0(ab,a,b)                                                                          \
  (ab)[R00] = ((ab)[R11]*(ab)[R22]) - ((ab)[R12]*(ab)[R21]);                                              \
  (ab)[R01] = ((ab)[R12]*(ab)[R20]) - ((ab)[R10]*(ab)[R22]);                                              \
  (ab)[R02] = ((ab)[R10]*(ab)[R21]) - ((ab)[R11]*(ab)[R20]);                                              \
  (ab)[TX]  = (a)[R00]*((b)[TX] - (a)[TX]) + (a)[R10]*((b)[TY] - (a)[TY]) + (a)[R20]*((b)[TZ] - (a)[TZ])
*/

__device__ __inline__ void TR_INV_MULT_ROW0(Matrix3x3_d& ab, const Matrix3x3_d& a, const Matrix3x3_d& b)
{
  (ab).m_row[0].x = ((ab).m_row[1].y * (ab).m_row[2].z) - ((ab).m_row[1].z * (ab).m_row[2].y);
  (ab).m_row[0].y = ((ab).m_row[1].z * (ab).m_row[2].x) - ((ab).m_row[1].x * (ab).m_row[2].z);
  (ab).m_row[0].z = ((ab).m_row[1].x * (ab).m_row[2].y) - ((ab).m_row[1].y * (ab).m_row[2].x);

  (ab).m_row[0].w = (a).m_row[0].x * ((b).m_row[0].w - (a).m_row[0].w) +
                    (a).m_row[1].x * ((b).m_row[1].w - (a).m_row[1].w) +
                    (a).m_row[2].x * ((b).m_row[2].w - (a).m_row[2].w);
}

/*#define TR_INV_MULT_ROW1(ab,a,b)                                                                          \
  (ab)[R10] = (a)[R01]*(b)[R00] + (a)[R11]*(b)[R10] + (a)[R21]*(b)[R20];                                  \
  (ab)[R11] = (a)[R01]*(b)[R01] + (a)[R11]*(b)[R11] + (a)[R21]*(b)[R21];                                  \
  (ab)[R12] = (a)[R01]*(b)[R02] + (a)[R11]*(b)[R12] + (a)[R21]*(b)[R22];                                  \
  (ab)[TY]  = (a)[R01]*((b)[TX] - (a)[TX]) + (a)[R11]*((b)[TY] - (a)[TY]) + (a)[R21]*((b)[TZ] - (a)[TZ])
*/

__device__ __inline__ void TR_INV_MULT_ROW1(Matrix3x3_d& ab, const Matrix3x3_d& a, const Matrix3x3_d& b)
{
    (ab).m_row[1].x = ((a).m_row[0].y * (b).m_row[0].x) + ((a).m_row[1].y * (b).m_row[1].x) + ((a).m_row[2].y * (b).m_row[2].x);
    (ab).m_row[1].y = ((a).m_row[0].y * (b).m_row[0].y) + ((a).m_row[1].y * (b).m_row[1].y) + ((a).m_row[2].y * (b).m_row[2].y);
    (ab).m_row[1].z = ((a).m_row[0].y * (b).m_row[0].z) + ((a).m_row[1].y * (b).m_row[1].z) + ((a).m_row[2].y * (b).m_row[2].z);

    (ab).m_row[1].w = (a).m_row[0].y * ((b).m_row[0].w - (a).m_row[0].w) +
                      (a).m_row[1].y * ((b).m_row[1].w - (a).m_row[1].w) +
                      (a).m_row[2].y * ((b).m_row[2].w - (a).m_row[2].w);
}

/*#define TR_INV_MULT_ROW2(ab,a,b)                                                                        \
  (ab)[R20] = (a)[R02]*(b)[R00] + (a)[R12]*(b)[R10] + (a)[R22]*(b)[R20];                                  \
  (ab)[R21] = (a)[R02]*(b)[R01] + (a)[R12]*(b)[R11] + (a)[R22]*(b)[R21];                                  \
  (ab)[R22] = (a)[R02]*(b)[R02] + (a)[R12]*(b)[R12] + (a)[R22]*(b)[R22];                                  \
  (ab)[TZ]  = (a)[R02]*((b)[TX] - (a)[TX]) + (a)[R12]*((b)[TY] - (a)[TY]) + (a)[R22]*((b)[TZ] - (a)[TZ])
*/

__device__ __inline__ void TR_INV_MULT_ROW2(Matrix3x3_d& ab, const Matrix3x3_d& a, const Matrix3x3_d& b)
{
    (ab).m_row[2].x = ((a).m_row[0].z * (b).m_row[0].x) + ((a).m_row[1].z * (b).m_row[1].x) + ((a).m_row[2].z * (b).m_row[2].x);
    (ab).m_row[2].y = ((a).m_row[0].z * (b).m_row[0].y) + ((a).m_row[1].z * (b).m_row[1].y) + ((a).m_row[2].z * (b).m_row[2].y);
    (ab).m_row[2].z = ((a).m_row[0].z * (b).m_row[0].z) + ((a).m_row[1].z * (b).m_row[1].z) + ((a).m_row[2].z * (b).m_row[2].z);

    (ab).m_row[2].w = (a).m_row[0].z * ((b).m_row[0].w - (a).m_row[0].w) +
                      (a).m_row[1].z * ((b).m_row[1].w - (a).m_row[1].w) +
                      (a).m_row[2].z * ((b).m_row[2].w - (a).m_row[2].w);
}

/*static const Matrix4IndicesMap::value_type matrixHelperIndices[16] =
{
    Matrix4IndicesMap::value_type(R00, std::pair<int,int>(0,0)),
    Matrix4IndicesMap::value_type(R10, std::pair<int,int>(1,0)),
    Matrix4IndicesMap::value_type(R20, std::pair<int,int>(2,0)),
    Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_1, std::pair<int,int>(3,0)),
    Matrix4IndicesMap::value_type(R01, std::pair<int,int>(0,1)),
    Matrix4IndicesMap::value_type(R11, std::pair<int,int>(1,1)),
    Matrix4IndicesMap::value_type(R21, std::pair<int,int>(2,1)),
    Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_2, std::pair<int,int>(3,1)),
    Matrix4IndicesMap::value_type(R02, std::pair<int,int>(0,2)),
    Matrix4IndicesMap::value_type(R12, std::pair<int,int>(1,2)),
    Matrix4IndicesMap::value_type(R22, std::pair<int,int>(2,2)),
    Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_3, std::pair<int,int>(3,2)),

    Matrix4IndicesMap::value_type(TX, std::pair<int,int>(3,0)),
    Matrix4IndicesMap::value_type(TY, std::pair<int,int>(3,1)),
    Matrix4IndicesMap::value_type(TZ, std::pair<int,int>(3,2)),
    Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_4, std::pair<int,int>(3,3))
};*/

#define OBB_EPS 1e-6
__device__ __inline__ bool intersectOBB(const OBB& node1, const OBB& node2, const Matrix3x3_d& a_rel_w, const Matrix3x3_d& b_rel_w)
{
    Matrix3x3_d b_rel_a;
    Matrix3x3_d bf;        // bf = fabs(b_rel_a) + eps
    double t, t2;

    const float3 a = node1.extents;
    const float3 b = node2.extents;

    // Class I tests
    TR_INV_MULT_ROW2(b_rel_a, a_rel_w, b_rel_w);

    bf.m_row[2].x = fabs(b_rel_a.m_row[2].x) + OBB_EPS;
    bf.m_row[2].y = fabs(b_rel_a.m_row[2].y) + OBB_EPS;
    bf.m_row[2].z = fabs(b_rel_a.m_row[2].z) + OBB_EPS;

    // A0 x A1 = A2
    /*t  = b_rel_a[TZ];
      t2 = a[2] + b[0] * bf[R20] + b[1] * bf[R21] + b[2] * bf[R22];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[2].w;
    t2 = a.z + b.x * bf.m_row[2].x +
            b.y * bf.m_row[2].y +
            b.z * bf.m_row[2].z;

    if (t > t2) { return true; }

    TR_INV_MULT_ROW1(b_rel_a, a_rel_w, b_rel_w);

    bf.m_row[1].x = fabs(b_rel_a.m_row[1].x) + OBB_EPS;
    bf.m_row[1].y = fabs(b_rel_a.m_row[1].y) + OBB_EPS;
    bf.m_row[1].z = fabs(b_rel_a.m_row[1].z) + OBB_EPS;

    // A2 x A0 = A1
    /*t  = b_rel_a[TY];
      t2 = a[1] + b[0] * bf[R10] + b[1] * bf[R11] + b[2] * bf[R12];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[1].w;
    t2 = a.y + b.x * bf.m_row[1].x +
            b.y * bf.m_row[1].y +
            b.z * bf.m_row[1].z;
    if (t > t2) { return true; }


    TR_INV_MULT_ROW0(b_rel_a, a_rel_w, b_rel_w);

    bf.m_row[0].x = fabs(b_rel_a.m_row[0].x) + OBB_EPS;
    bf.m_row[0].y = fabs(b_rel_a.m_row[0].y) + OBB_EPS;
    bf.m_row[0].z = fabs(b_rel_a.m_row[0].z) + OBB_EPS;

    // A1 x A2 = A0
    /*t  = b_rel_a[TX];
      t2 = a[0] + b[0] * bf[R00] + b[1] * bf[R01] + b[2] * bf[R02];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[0].w;
    t2 = a.x + b.x * bf.m_row[0].x +
            b.y * bf.m_row[0].y +
            b.z * bf.m_row[0].z;
    if (t > t2) { return true; }

    //assert(is_rot_matrix(b_rel_a));

    // Class II tests

    // B0 x B1 = B2
    /*t  = b_rel_a[TX]*b_rel_a[R02] + b_rel_a[TY]*b_rel_a[R12] + b_rel_a[TZ]*b_rel_a[R22];
        t2 = b[2] + a[0] * bf[R02] + a[1] * bf[R12] + a[2] * bf[R22];
        if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[0].w *
            b_rel_a.m_row[0].z +
            b_rel_a.m_row[1].w *
            b_rel_a.m_row[1].z +
            b_rel_a.m_row[2].w *
            b_rel_a.m_row[2].z;

    t2 = b.z + a.x * bf.m_row[0].z +
            a.y * bf.m_row[1].z +
            a.z * bf.m_row[2].z;

    if (t > t2) { return true; }

    // B2 x B0 = B1
    t  = b_rel_a.m_row[0].w *
            b_rel_a.m_row[0].y +
            b_rel_a.m_row[1].w *
            b_rel_a.m_row[1].y +
            b_rel_a.m_row[2].w *
            b_rel_a.m_row[2].y;

    t2 = b.y + a.x * bf.m_row[0].y +
            a.y * bf.m_row[1].y +
            a.z * bf.m_row[2].y;

    if (t > t2) { return true; }

    // B1 x B2 = B0
    t  = b_rel_a.m_row[0].w *
            b_rel_a.m_row[0].x +
            b_rel_a.m_row[1].w *
            b_rel_a.m_row[1].x +
            b_rel_a.m_row[2].w *
            b_rel_a.m_row[2].x;

    t2 = b.x + a.x * bf.m_row[0].x +
            a.y * bf.m_row[1].x +
            a.z * bf.m_row[2].x;

    if (t > t2) { return true; }

    // Class III tests

    // A0 x B0
    t  = b_rel_a.m_row[2].w *
            b_rel_a.m_row[1].x -
            b_rel_a.m_row[1].w *
            b_rel_a.m_row[2].x;

    t2 = a.y * bf.m_row[2].x +
         a.z * bf.m_row[1].x +
         b.y * bf.m_row[0].z +
         b.z * bf.m_row[0].y;

    if (t > t2) { return true; }

    // A0 x B1
    /*t  = b_rel_a[TZ] * b_rel_a[R11] - b_rel_a[TY] * b_rel_a[R21];
      t2 = a[1] * bf[R21] + a[2] * bf[R11] + b[0] * bf[R02] + b[2] * bf[R00];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[2].w *
         b_rel_a.m_row[1].y -
         b_rel_a.m_row[1].w *
         b_rel_a.m_row[2].y;

    t2 = a.y * bf.m_row[2].y +
         a.z * bf.m_row[1].y +
         b.x * bf.m_row[0].z +
         b.z * bf.m_row[0].x;

    if (t > t2) { return true; }

    // A0 x B2
    /*t  = b_rel_a[TZ] * b_rel_a[R12] - b_rel_a[TY] * b_rel_a[R22];
      t2 = a[1] * bf[R22] + a[2] * bf[R12] + b[0] * bf[R01] + b[1] * bf[R00];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[2].w *
            b_rel_a.m_row[1].z -
            b_rel_a.m_row[1].w *
            b_rel_a.m_row[2].z;

    t2 = a.y * bf.m_row[2].z +
         a.z * bf.m_row[1].z +
         b.x * bf.m_row[0].y +
         b.y * bf.m_row[0].x;

    if (t > t2) { return true; }


    // A1 x B0
    /*t  = b_rel_a[TX] * b_rel_a[R20] - b_rel_a[TZ] * b_rel_a[R00];
      t2 = a[0] * bf[R20] + a[2] * bf[R00] + b[1] * bf[R12] + b[2] * bf[R11];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[0].w *
         b_rel_a.m_row[2].x -
         b_rel_a.m_row[2].w *
         b_rel_a.m_row[0].x;

    t2 = a.x * bf.m_row[2].x +
         a.z * bf.m_row[0].x +
         b.y * bf.m_row[1].z +
         b.z * bf.m_row[1].y;

    if (t > t2) { return true; }

    // A1 x B1
    /*t  = b_rel_a[TX] * b_rel_a[R21] - b_rel_a[TZ] * b_rel_a[R01];
      t2 = a[0] * bf[R21] + a[2] * bf[R01] + b[0] * bf[R12] + b[2] * bf[R10];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[0].w *
         b_rel_a.m_row[2].y -
         b_rel_a.m_row[2].w *
         b_rel_a.m_row[0].y;

    t2 = a.x * bf.m_row[2].y +
         a.z * bf.m_row[0].y +
         b.x * bf.m_row[1].z +
         b.z * bf.m_row[1].x;

    if (t > t2) { return true; }

    // A1 x B2
    /*t  = b_rel_a[TX] * b_rel_a[R22] - b_rel_a[TZ] * b_rel_a[R02];
      t2 = a[0] * bf[R22] + a[2] * bf[R02] + b[0] * bf[R11] + b[1] * bf[R10];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[0].w *
         b_rel_a.m_row[2].z -
         b_rel_a.m_row[2].w *
         b_rel_a.m_row[0].z;

    t2 = a.x * bf.m_row[2].z +
         a.z * bf.m_row[0].z +
         b.x * bf.m_row[1].y +
         b.y * bf.m_row[1].x;

    if (t > t2) { return true; }

    // A2 x B0
    /*t  = b_rel_a[TY] * b_rel_a[R00] - b_rel_a[TX] * b_rel_a[R10];
      t2 = a[0] * bf[R10] + a[1] * bf[R00] + b[1] * bf[R22] + b[2] * bf[R21];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[1].w *
            b_rel_a.m_row[0].x -
            b_rel_a.m_row[0].w *
            b_rel_a.m_row[1].x;

    t2 = a.x * bf.m_row[1].x +
         a.y * bf.m_row[0].x +
         b.y * bf.m_row[2].z +
         b.z * bf.m_row[2].y;

    if (t > t2) { return true; }

    // A2 x B1
    /*t  = b_rel_a[TY] * b_rel_a[R01] - b_rel_a[TX] * b_rel_a[R11];
      t2 = a[0] * bf[R11] + a[1] * bf[R01] + b[0] * bf[R22] + b[2] * bf[R20];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[1].w *
         b_rel_a.m_row[0].y -
         b_rel_a.m_row[0].w *
         b_rel_a.m_row[1].y;

    t2 = a.x * bf.m_row[1].y +
         a.y * bf.m_row[0].y +
         b.x * bf.m_row[2].z +
         b.z * bf.m_row[2].x;

    if (t > t2) { return true; }

    // A2 x B2
    /*t  = b_rel_a[TY] * b_rel_a[R02] - b_rel_a[TX] * b_rel_a[R12];
      t2 = a[0] * bf[R12] + a[1] * bf[R02] + b[0] * bf[R21] + b[1] * bf[R20];
      if (GREATER(t, t2)) { return TRUE; }*/

    t  = b_rel_a.m_row[1].w *
            b_rel_a.m_row[0].z -
            b_rel_a.m_row[0].w *
            b_rel_a.m_row[1].z;

    t2 = a.x * bf.m_row[1].z +
         a.y * bf.m_row[0].z +
         b.x * bf.m_row[2].y +
         b.y * bf.m_row[2].x;

    if (t > t2) { return true; }

    return false;
}

//#define GPROXIMITY_DEBUG_INTERSECT_NODES
template <>
__device__ __inline__ bool intersect<OBB>(const OBB &node1, const OBB &node2, bool useMinDimension, double alarmDistance)
{
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
    __syncthreads();
    printf("intersect<OBB>(%i,%i)\n", node1.idx, node2.idx);
    printf(" node1.extents = %f,%f,%f; node1.axis1 = %f,%f,%f; node1.axis2 = %f,%f,%f; node1.axis3 = %f,%f,%f\n",
           node1.extents.x, node1.extents.y ,node1.extents.z,
           node1.axis1.x, node1.axis1.y ,node1.axis1.z,
           node1.axis2.x, node1.axis2.y ,node1.axis2.z,
           node1.axis3.x, node1.axis3.y ,node1.axis3.z);
    __syncthreads();
    printf(" node2.extents = %f,%f,%f; node2.axis1 = %f,%f,%f; node2.axis2 = %f,%f,%f; node2.axis3 = %f,%f,%f\n",
           node2.extents.x, node2.extents.y ,node2.extents.z,
           node2.axis1.x, node2.axis1.y ,node2.axis1.z,
           node2.axis2.x, node2.axis2.y ,node2.axis2.z,
           node2.axis3.x, node2.axis3.y ,node2.axis3.z);
#endif
    //translation, in parent frame
	float3 v = f3v_sub(node1.center, node2.center);
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
    __syncthreads();
    printf(" parent frame: v = %f,%f,%f\n", v.x, v.y, v.z);
#endif
	//translation, in A's frame
	float3 T = make_float3(f3v_dot(v, node1.axis1), f3v_dot(v, node1.axis2), f3v_dot(v, node1.axis3));
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
    __syncthreads();
    printf(" frame A: T = %f,%f,%f\n", T.x, T.y, T.z);
#endif
	//calculate rotation matrix (B's basis with respect to A', R1)
	float3 R1;
	R1.x = f3v_dot(node1.axis1, node2.axis1);
	R1.y = f3v_dot(node1.axis1, node2.axis2);
	R1.z = f3v_dot(node1.axis1, node2.axis3);
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
    __syncthreads();
    printf(" B relative to A: rotation R1 = %f,%f,%f\n", R1.x, R1.y, R1.z);
#endif
	/*
	ALGORITHM: Use the separating axis test for all 15 potential
	separating axes. If a separating axis could not be found, the two
	boxes overlap.
	*/
	
    float extents1_x = node1.extents.x;
    float extents1_y = node1.extents.y;
    float extents1_z = node1.extents.z;

    float extents2_x = node2.extents.x;
    float extents2_y = node2.extents.y;
    float extents2_z = node2.extents.z;

    if (useMinDimension)
    {
        if (node1.min_dimension == 0)
        {
            extents1_x = node1.min_dimension_val;
            extents1_x += alarmDistance;
        }
        if (node1.min_dimension == 1)
        {
            extents1_y = node1.min_dimension_val;
            extents1_y += alarmDistance;
        }
        if (node1.min_dimension == 2)
        {
            extents1_z = node1.min_dimension_val;
            extents1_z += alarmDistance;
        }

        if (node2.min_dimension == 0)
        {
            extents2_x = node2.min_dimension_val;
            extents2_x += alarmDistance;
        }
        if (node2.min_dimension == 1)
        {
            extents2_y = node2.min_dimension_val;
            extents2_y += alarmDistance;
        }
        if (node2.min_dimension == 2)
        {
            extents2_z = node2.min_dimension_val;
            extents2_z += alarmDistance;
        }
    }

#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" useMinDimension = %i, node1.min_dimension = %i, node1.min_dimension_val = %f, node2.min_dimension = %i, node2.min_dimension_val = %f; extents1_x = %f, extents1_y = %f, extents1_z = %f; extents2_x = %f, extents2_y = %f, extents2_z = %f\n",
                   useMinDimension, node1.min_dimension, node2.min_dimension_val, node1.min_dimension, node2.min_dimension_val, extents1_x, extents1_y, extents1_z, extents2_x, extents2_y, extents2_z);
            __syncthreads();
#endif

	// Axes: A's basis vectors
	{
		float rb;
        rb = extents2_x * fabs(R1.x) + extents2_y * fabs(R1.y) + extents2_z * fabs(R1.z) + OBB_ROTATION_MATRIX_EPSILON * (extents2_x + extents2_y + extents2_z);
        if(fabs(T.x) > (extents1_x + rb))
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" Frame A: separating axis with R1 found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	//calculate rotation matrix (B's basis with respect to A', R2)
	float3 R2;
	R2.x = f3v_dot(node1.axis2, node2.axis1);
	R2.y = f3v_dot(node1.axis2, node2.axis2);
	R2.z = f3v_dot(node1.axis2, node2.axis3);
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
    printf(" B relative to A: rotation R2 = %f,%f,%f\n", R2.x, R2.y, R2.z);
#endif
	{
		float rb;
        rb = extents2_x * fabs(R2.x) + extents2_y * fabs(R2.y) + extents2_z * fabs(R2.z) + OBB_ROTATION_MATRIX_EPSILON * (extents2_x + extents2_y + extents2_z);
        if(fabs(T.y) > (extents1_y + rb))
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" Frame A: separating axis with R2 found\n");
            __syncthreads();
			return false;
#endif
        }
	}
	
	//calculate rotation matrix (B's basis with respect to A', R3)
	float3 R3;
	R3.x = f3v_dot(node1.axis3, node2.axis1);
	R3.y = f3v_dot(node1.axis3, node2.axis2);
	R3.z = f3v_dot(node1.axis3, node2.axis3);

#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
    printf(" B relative to A: rotation R3 = %f,%f,%f\n", R3.x, R3.y, R3.z);
    __syncthreads();
#endif
	{
		float rb;
        rb = extents2_x * fabs(R3.x) + extents2_y * fabs(R3.y) + extents2_z * fabs(R3.z) + OBB_ROTATION_MATRIX_EPSILON * (extents2_x + extents2_y + extents2_z);
        if(fabs(T.z) > (extents1_z + rb))
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" Frame A: separating axis with R3 found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	// Axes: B's basis vectors
	{
		float rb, t;
        rb = extents1_x * fabs(R1.x) + extents1_y * fabs(R2.x) + extents1_z * fabs(R3.x) + OBB_ROTATION_MATRIX_EPSILON * (extents1_x + extents1_y + extents1_z);
		t = fabs(T.x * R1.x + T.y * R2.x + T.z * R3.x);
        if(t > (extents2_x + rb))
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" Frame B: separating axis with R1/2/3.x found\n");
            __syncthreads();
#endif
			return false;
        }
			
        rb = extents1_x * fabs(R1.y) + extents1_y * fabs(R2.y) + extents1_z * fabs(R3.y) + OBB_ROTATION_MATRIX_EPSILON * (extents1_x + extents1_y + extents1_z);
		t = fabs(T.x * R1.y + T.y * R2.y + T.z * R3.y);
        if(t > (extents2_y + rb))
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" Frame B: separating axis with R1/2/3.y found\n");
            __syncthreads();
#endif
			return false;
        }
        rb = extents1_x * fabs(R1.z) + extents1_y * fabs(R2.z) + extents1_z * fabs(R3.z) + OBB_ROTATION_MATRIX_EPSILON * (extents1_x + extents1_y + extents1_z);
		t = fabs(T.x * R1.z + T.y * R2.z + T.z * R3.z);
        if(t > (extents2_z + rb))
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" Frame B: separating axis with R1/2/3.z found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	// Axes: 9 cross products
	
	//L = A0 x B0
	{
		float ra, rb, t;
        ra = extents1_y * fabs(R3.x) + extents1_z * fabs(R2.x) + OBB_ROTATION_MATRIX_EPSILON * (extents1_y + extents1_z);
        rb = extents2_y * fabs(R1.z) + extents2_z * fabs(R1.y) + OBB_ROTATION_MATRIX_EPSILON * (extents2_y + extents2_z);
		t = fabs(T.z * R2.x - T.y * R3.x);
		
        if(t > ra + rb)
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" separating axis with A0xB0 found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	//L = A0 x B1
	{
		float ra, rb, t;
        ra = extents1_y * fabs(R3.y) + extents1_z * fabs(R2.y) + OBB_ROTATION_MATRIX_EPSILON * (extents1_y + extents1_z);
        rb = extents2_x * fabs(R1.z) + extents2_z * fabs(R1.x) + OBB_ROTATION_MATRIX_EPSILON * (extents2_x + extents2_z);
		t = fabs(T.z * R2.y - T.y * R3.y);
		
		if(t > ra + rb)
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" separating axis with A0xB1 found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	//L = A0 x B2
	{
		float ra, rb, t;
        ra = extents1_y * fabs(R3.z) + extents1_z * fabs(R2.z) + OBB_ROTATION_MATRIX_EPSILON * (extents1_y + extents1_z);
        rb = extents2_x * fabs(R1.y) + extents2_y * fabs(R1.x) + OBB_ROTATION_MATRIX_EPSILON * (extents2_x + extents2_y);
		t = fabs(T.z * R2.z - T.y * R3.z);
		
		if(t > ra + rb)
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" separating axis with A0xB2 found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	//L = A1 x B0
	{
		float ra, rb, t;
        ra = extents1_x * fabs(R3.x) + extents1_z * fabs(R1.x) + OBB_ROTATION_MATRIX_EPSILON * (extents1_x + extents1_z);
        rb = extents2_y * fabs(R2.z) + extents2_z * fabs(R2.y) + OBB_ROTATION_MATRIX_EPSILON * (extents2_y + extents2_z);
		t = fabs(T.x * R3.x - T.z * R1.x);
		
		if(t > ra + rb)
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" separating axis with A1xB0 found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	//L = A1 x B1
	{
		float ra, rb, t;
        ra = extents1_x * fabs(R3.y) + extents1_z * fabs(R1.y) + OBB_ROTATION_MATRIX_EPSILON * (extents1_x + extents1_z);
        rb = extents2_x * fabs(R2.z) + extents2_z * fabs(R2.x) + OBB_ROTATION_MATRIX_EPSILON * (extents2_y + extents2_z);
		t = fabs(T.x * R3.y - T.z * R1.y);
		
		if(t > ra + rb)
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" separating axis with A1xB1 found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	//L = A1 x B2
	{
		float ra, rb, t;
        ra = extents1_x * fabs(R3.z) + extents1_z * fabs(R1.z) + OBB_ROTATION_MATRIX_EPSILON * (extents1_x + extents1_z);
        rb = extents2_x * fabs(R2.y) + extents2_y * fabs(R2.x) + OBB_ROTATION_MATRIX_EPSILON * (extents2_x + extents2_y);
		t = fabs(T.x * R3.z - T.z * R1.z);
		
		if(t > ra + rb)
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" separating axis with A1xB2 found\n");
            __syncthreads();
#endif
			return false;
        }
    }
	
	//L = A2 x B0
	{
		float ra, rb, t;
        ra = extents1_x * fabs(R2.x) + extents1_y * fabs(R1.x) + OBB_ROTATION_MATRIX_EPSILON * (extents1_x + extents1_y);
        rb = extents2_y * fabs(R3.z) + extents2_z * fabs(R3.y) + OBB_ROTATION_MATRIX_EPSILON * (extents2_y + extents2_z);
		t = fabs(T.y * R1.x - T.x * R2.x);
		
		if(t > ra + rb)
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" separating axis with A2xB0 found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	//L = A2 x B1
	{
		float ra, rb, t;
        ra = extents1_x * fabs(R2.y) + extents1_y * fabs(R1.y) + OBB_ROTATION_MATRIX_EPSILON * (extents1_x + extents1_y);
        rb = extents2_x * fabs(R3.z) + extents2_z * fabs(R3.x) + OBB_ROTATION_MATRIX_EPSILON * (extents2_x + extents2_z);
		t = fabs(T.y * R1.y - T.x * R2.y);
		
		if(t > ra + rb)
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" separating axis with A2xB1 found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	//L = A2 x B2
	{
		float ra, rb, t;
        ra = extents1_x * fabs(R2.z) + extents1_y * fabs(R1.z) + OBB_ROTATION_MATRIX_EPSILON * (extents1_x + extents1_y);
        rb = extents2_x * fabs(R3.y) + extents2_y * fabs(R3.x) + OBB_ROTATION_MATRIX_EPSILON * (extents2_x + extents2_y);
		t = fabs(T.y * R1.z - T.x * R2.z);
		
		if(t > ra + rb)
        {
#ifdef GPROXIMITY_DEBUG_INTERSECT_NODES
            printf(" separating axis with A2xB2 found\n");
            __syncthreads();
#endif
			return false;
        }
	}
	
	// no separating axis found:
	// the two boxes overlap
	
	return true;
}

#endif
