#include "ObbTreeGPUCollisionModel_cuda.h"

#include <stdio.h>

#include <gProximity/cuda_make_grid.h>
#include <cutil/cutil.h>

#include <cuda.h>
#include <device_functions.h>

//#define THREEDVERTEXTRANSFORM_DEBUG
__global__ void ThreeDVertexTransform(GPUVertex *inVertices, GPUVertex* outVertices, Matrix3x3_d* transformMatrix, float3* translationVector, int nVertices, bool inverse)
{
    // each thread will compute exactly one number
    // four numbers per vertex [x,y,z,w] means 4 threads will compute exactly
    // one vertex transformation

    // each block must be 1-D, with the number of threads being a multiple of 4 (32 is best)
    // grid of blocks may be 1-D or 2-D to accomodate a very large number of vertices
    // compute the current thread ID
    // Each thread will compute the new vertices[tid]
    int tid = (blockIdx.y * gridDim.x * blockDim.x) +
                    (blockIdx.x * blockDim.x) +
                    (threadIdx.x);

    // perform checking to prevent buffer overruns
    if (tid >= nVertices)
        return;

#ifdef THREEDVERTEXTRANSFORM_DEBUG
    printf("Use rotation matrix: [%f,%f,%f],[%f,%f,%f],[%f,%f,%f]\n",
            transformMatrix->m_row[0].x, transformMatrix->m_row[0].y, transformMatrix->m_row[0].z,
            transformMatrix->m_row[1].x, transformMatrix->m_row[1].y, transformMatrix->m_row[1].z,
            transformMatrix->m_row[2].x, transformMatrix->m_row[2].y, transformMatrix->m_row[2].z);

    printf("Transform vertex %i of %i: %f,%f,%f\n", tid, nVertices, inVertices[tid].v.x, inVertices[tid].v.y, inVertices[tid].v.z);
#endif
    if (!inverse)
    {
        outVertices[tid].v = mtMul1(*transformMatrix, inVertices[tid].v);
        outVertices[tid].v.x += translationVector->x;
        outVertices[tid].v.y += translationVector->y;
        outVertices[tid].v.z += translationVector->z;
    }
    else
    {
        outVertices[tid].v.x -= translationVector->x;
        outVertices[tid].v.y -= translationVector->y;
        outVertices[tid].v.z -= translationVector->z;
        // assumed to be transposed/inversed!
        outVertices[tid].v = mtMul1(*transformMatrix, inVertices[tid].v);
    }
#ifdef THREEDVERTEXTRANSFORM_DEBUG
    printf("New coordinates for vertex %i of %i: %f,%f,%f\n", tid, nVertices, outVertices[tid].v.x, outVertices[tid].v.y, outVertices[tid].v.z);
#endif
}

#define UPDATEINTERNALGEOMETRY_CUDA_DEBUG
void updateInternalGeometry_cuda(ModelInstance* m_gpModel, GPUVertex* transformedVertices,
                                 gpTransform& modelTransform, bool collisionHappens)
{
#ifdef UPDATEINTERNALGEOMETRY_CUDA_DEBUG
    std::cout << "updateInternalGeometry_cuda: " << m_gpModel->nVerts << " vertices to transform." << std::endl;
#endif
    dim3 grids1 = makeGrid((int)ceilf(m_gpModel->nVerts / (float)GENERAL_THREADS));
    dim3 threads(GENERAL_THREADS, 1, 1);

    Matrix3x3_d* d_modelTransform1 = NULL;
    float3* d_trVector1 = NULL;

    Matrix3x3_d h_modelTransform1;

    h_modelTransform1.m_row[0].x = modelTransform.m_R[0][0];
    h_modelTransform1.m_row[0].y = modelTransform.m_R[0][1];
    h_modelTransform1.m_row[0].z = modelTransform.m_R[0][2];
    h_modelTransform1.m_row[1].x = modelTransform.m_R[1][0];
    h_modelTransform1.m_row[1].y = modelTransform.m_R[1][1];
    h_modelTransform1.m_row[1].z = modelTransform.m_R[1][2];
    h_modelTransform1.m_row[2].x = modelTransform.m_R[2][0];
    h_modelTransform1.m_row[2].y = modelTransform.m_R[2][1];
    h_modelTransform1.m_row[2].z = modelTransform.m_R[2][2];

    float3 h_trVector1 = make_float3(modelTransform.m_T[0], modelTransform.m_T[1], modelTransform.m_T[2]);

#ifdef UPDATEINTERNALGEOMETRY_CUDA_DEBUG
    std::cout << " model1 translation from gpTransform = " << modelTransform.m_T[0] << "," << modelTransform.m_T[1] << "," << modelTransform.m_T[2] << std::endl;
    std::cout << " model1 orientation from gpTransform = ["
              << modelTransform.m_R[0][0] << "," << modelTransform.m_R[0][1] << "," << modelTransform.m_R[0][2] << "],["
              << modelTransform.m_R[1][0] << "," << modelTransform.m_R[1][1] << "," << modelTransform.m_R[1][2] << "],["
              << modelTransform.m_R[2][0] << "," << modelTransform.m_R[2][1] << "," << modelTransform.m_R[2][2] << "]"<< std::endl;
#endif

    GPUMALLOC((void**)&d_modelTransform1, sizeof(Matrix3x3_d));
    GPUMALLOC((void**)&d_trVector1, sizeof(float3));

    TOGPU(d_trVector1, &h_trVector1, sizeof(float3));

    TOGPU(d_modelTransform1, &h_modelTransform1, sizeof(Matrix3x3_d));

#ifdef UPDATEINTERNALGEOMETRY_CUDA_DEBUG
    std::cout << " model1 translation for kernel = " << h_trVector1.x << "," << h_trVector1.y << "," << h_trVector1.z << std::endl;
    std::cout << " model1 orientation for kernel = [" << h_modelTransform1.m_row[0].x << "," << h_modelTransform1.m_row[0].y << "," << h_modelTransform1.m_row[0].z << "],[" << h_modelTransform1.m_row[1].x << "," << h_modelTransform1.m_row[1].y << "," << h_modelTransform1.m_row[1].z << "],[" << h_modelTransform1.m_row[2].x << "," << h_modelTransform1.m_row[2].y << "," << h_modelTransform1.m_row[2].z << "]"<< std::endl;
#endif

    if (collisionHappens)
    {
        ThreeDVertexTransform << < grids1, threads >> > ((GPUVertex*)m_gpModel->vertexPointer, (GPUVertex*)m_gpModel->vertexTfPointer, d_modelTransform1, d_trVector1, m_gpModel->nVerts, false);
    }

#ifdef UPDATEINTERNALGEOMETRY_CUDA_DEBUG
    /*if (collisionHappens)
    {
        FROMGPU(transformedVertices, m_gpModel->vertexTfPointer, sizeof(GPUVertex) * m_gpModel->nVerts);

        std::cout << " model vertices transformed: " << std::endl;
        for (int k = 0; k < m_gpModel->nVerts; k++)
        {
            std::cout << " * " << k << ": " << transformedVertices[k].v.x << "," << transformedVertices[k].v.y << "," << transformedVertices[k].v.z << std::endl;
        }
    }*/
#endif

    GPUFREE(d_modelTransform1);
    GPUFREE(d_trVector1);
}

void updateInternalGeometry_cuda_streamed(ModelInstance* m_gpModel, GPUVertex* transformedVertices,
                                 gProximityGPUTransform*& modelTransform, cudaStream_t& cudaStream, bool collisionHappens)
{
#ifdef UPDATEINTERNALGEOMETRY_CUDA_STREAMED_DEBUG
    std::cout << "updateInternalGeometry_cuda_streams: " << m_gpModel->nVerts << " vertices to transform." << std::endl;
#endif
    dim3 grids1 = makeGrid((int)ceilf(m_gpModel->nVerts / (float)GENERAL_THREADS));
    dim3 threads(GENERAL_THREADS, 1, 1);

    if (collisionHappens)
    {
#ifdef UPDATEINTERNALGEOMETRY_CUDA_STREAMED_DEBUG
        Matrix3x3_d h_modelTransform;
        float3 h_trVector;

        FROMGPU(&h_modelTransform, modelTransform->modelOrientation, sizeof(Matrix3x3_d));
        FROMGPU(&h_trVector, modelTransform->modelTranslation, sizeof(float3));

        std::cout << " model translation = " << h_trVector.x << "," << h_trVector.y << "," << h_trVector.z << std::endl;
        std::cout << " model orientation = " << h_modelTransform.m_row[0].x << "," << h_modelTransform.m_row[0].y << "," << h_modelTransform.m_row[0].z
                  << "                     " << h_modelTransform.m_row[1].x << "," << h_modelTransform.m_row[1].y << "," << h_modelTransform.m_row[1].z
                  << "                     " << h_modelTransform.m_row[2].x << "," << h_modelTransform.m_row[2].y << "," << h_modelTransform.m_row[2].z
                  << std::endl;

#endif
        ThreeDVertexTransform <<< grids1, threads, 0, cudaStream >>> ((GPUVertex*)m_gpModel->vertexPointer, (GPUVertex*)m_gpModel->vertexTfPointer, modelTransform->modelOrientation, modelTransform->modelTranslation, m_gpModel->nVerts, false);
	}

    cudaStreamSynchronize(cudaStream);

#ifdef UPDATEINTERNALGEOMETRY_CUDA_STREAMED_DEBUG
    if (collisionHappens)
    {
        FROMGPU(transformedVertices, m_gpModel->vertexTfPointer, sizeof(GPUVertex) * m_gpModel->nVerts);

        std::cout << " model vertices transformed: " << std::endl;
        for (int k = 0; k < m_gpModel->nVerts; k++)
        {
            std::cout << " * " << k << ": " << transformedVertices[k].v.x << "," << transformedVertices[k].v.y << "," << transformedVertices[k].v.z << std::endl;
        }
    }
#endif
}
