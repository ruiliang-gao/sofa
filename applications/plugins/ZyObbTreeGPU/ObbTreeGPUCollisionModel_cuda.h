#ifndef OBBTREEGPUCOLLISIONMODEL_CUDA_H
#define OBBTREEGPUCOLLISIONMODEL_CUDA_H

#include "gProximity/ObbTreeGPU_LinearAlgebra.cuh"
#include "ObbTreeGPUCollisionDetection_cuda.h"

void updateInternalGeometry_cuda(ModelInstance* m_gpModel, GPUVertex* transformedVertices,
                                 gpTransform &modelTransform, bool collisionHappens);

void updateInternalGeometry_cuda_streamed(ModelInstance* m_gpModel, GPUVertex* transformedVertices,
                                 gProximityGPUTransform*& modelTransform, cudaStream_t& cudaStream, bool collisionHappens);

#endif // OBBTREEGPUCOLLISIONMODEL_CUDA_H
