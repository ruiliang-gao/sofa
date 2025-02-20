/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaMeshCollision/LineModel.h>

namespace sofa::gpu::cuda
{

SOFA_CUDA_ATTRIBUTE_DEPRECATED("v22.06 (PR #2673)", "CudaLineCollisionModel")
CudaDeprecatedAndRemoved CudaLineModel;

using CudaLineCollisionModel = sofa::component::collision::LineCollisionModel<CudaVec3Types>;
using CudaLineCollisionModelf1 = sofa::component::collision::LineCollisionModel<CudaVec3f1Types>;

using CudaLine = sofa::component::collision::TLine<CudaVec3fTypes>;

} // namespace sofa::gpu::cuda
