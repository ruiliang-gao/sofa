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
 
#ifndef __CUDA_COLLISION_H_
#define __CUDA_COLLISION_H_

#include "geometry.h"

#include "cuda_vertex.h"

unsigned int CUDA_trianglePairCollide(GPUVertex* d_vertices1, uint3* d_triIndices1, GPUVertex* d_vertices2, uint3* d_triIndices2, int2* collisionPairs, unsigned int nPairs, std::vector<std::pair<int,int> >& collisionList);

enum gProximityContactType
{
    COLLISION_LINE_LINE,
    COLLISION_VERTEX_FACE,
    COLLISION_LINE_POINT,
    COLLISION_INVALID
};

struct gProximityDetectionOutput
{
    bool* valid;
    bool* swapped;
    /// Pair of colliding elements.
    int4* elems;
    /// ID's for contact points
    int* contactId;

    /// Contact points on the surface of each model. They are expressed in the local coordinate system of the model if any is defined..
    float3* point0;
    float3* point1;

    /// Normal of the contact, pointing outward from the first model
    float3* normal;
    /*
    /// Signed distance (negative if objects are interpenetrating). If using a proximity-based detection, this is the actual distance between the objets minus the specified contact distance.
    */
    double* distance;

    gProximityContactType* contactType;
};

void CUDA_trianglePairIntersect(GPUVertex* d_vertices1, uint3* d_triIndices1, GPUVertex* d_vertices2, uint3* d_triIndices2, int2* collisionPairs, int nPairs,  gProximityDetectionOutput* contactsList, int nMaxContacts, double alarmDistance, double contactDistance);

int CUDA_BVHCollide(ModelInstance* model1, ModelInstance* model2, std::vector<std::pair<int, int> > &collisionList);

#endif
