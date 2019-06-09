#ifndef OBBTREEGPU_CUDA_DATA_STRUCTURES_H
#define OBBTREEGPU_CUDA_DATA_STRUCTURES_H

//#define USE_TRUNK_CUDA_THRUST

#include <cuda.h>
#include <vector_types.h>

#include <memory>

#include "gProximity/transform.h"
#include "gProximity/cuda_collision.h"
#include "gProximity/ObbTreeGPU_LinearAlgebra.cuh"

// How many contact points are generated per tri-tri test: 6 * Vertex vs. Face + 9 * Edge vs. Edge
static int CollisionTestElementsSize = 15;

struct gProximityWorkerUnit
{
    public:

        gProximityWorkerUnit(unsigned int collisionPairCapacity = COLLISION_PAIR_CAPACITY, unsigned int queueNTasks = QUEUE_NTASKS, unsigned int queueSizePerTaskGlobal = QUEUE_SIZE_PER_TASK_GLOBAL);
        ~gProximityWorkerUnit();

        unsigned int _collisionPairCapacity;
        unsigned int _queueNTasks;
        unsigned int _queueSizePerTaskGlobal;

        unsigned int* d_collisionPairIndex;
        unsigned int* d_collisionSync;

        unsigned int* d_nWorkQueueElements;
        int2* d_collisionPairs;
        bool* d_collisionLeafs;
        int2* d_workQueues, *d_workQueues2;
        unsigned int* d_workQueueCounts;
        int* d_balanceSignal;

        int* d_obbOutputIndex;
        // int2* d_intersectingOBBs;

        unsigned int _nCollidingPairs;

        unsigned int _workerUnitIndex;

        cudaStream_t* _stream;
        //cudaStream_t* _syncStream;
        //cudaEvent_t* _event;

        Matrix3x3_d* d_modelOrientation_1;
        Matrix3x3_d* d_modelOrientation_2;
        float3* d_modelPosition_1;
        float3* d_modelPosition_2;
};

struct gProximityWorkerResultPrivate;

//#define USE_THRUST_HOST_VECTORS_IN_RESULTS
//#define ALLOCATE_RESULT_MEMORY_ON_GPU

struct gProximityWorkerResult
{
    public:

        gProximityWorkerResult(unsigned int maxResults = 64000);
        ~gProximityWorkerResult();

        unsigned int _maxResults;

        bool _blocked;
        unsigned int _resultIndex;
        unsigned int _resultBin;
        unsigned int _outputIndexPosition;

        unsigned int _numResults;

        gProximityDetectionOutput* h_contacts;
        gProximityDetectionOutput* d_contacts;

        unsigned int* d_outputIndex;
        unsigned int h_outputIndex;

        bool* d_valid;
        bool* d_swapped;
        int* d_contactId;
        double* d_distance;
        int4* d_elems;
        float3* d_normal;
        float3* d_point0;
        float3* d_point1;
        gProximityContactType* d_gProximityContactType;

        struct gProximityWorkerResultPrivate* d_ptr;
};

struct gProximityGPUTransform
{
    public:
        gProximityGPUTransform();
        ~gProximityGPUTransform();

        float3* modelTranslation;
        Matrix3x3_d* modelOrientation;
};

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            struct OBBContainer
            {
                public:
                    OBBContainer()
                    {
                        nVerts = nTris = nBVs = 0;
                        vbo_Vertex = vbo_TriIndex = 0;
                        obbTree = NULL;
                        vertexPointer = NULL;
                        triIdxPointer = NULL;
                        modelTransform.set_identity();

                        d_modelTransform = NULL;
                    }

                    OBBContainer(const OBBContainer& other)
                    {
                        nVerts = other.nVerts;
                        nTris = other.nTris;
                        nBVs = other.nBVs;
                        obbTree = other.obbTree;
                        vertexPointer = other.vertexPointer;
                        vertexTfPointer = other.vertexTfPointer;
                        triIdxPointer = other.triIdxPointer;
                        modelTransform = other.modelTransform;

                        d_modelTransform = other.d_modelTransform;
                    }

                    OBBContainer& operator=(const OBBContainer& other)
                    {
                        if (this != &other)
                        {
                            nVerts = other.nVerts;
                            nTris = other.nTris;
                            nBVs = other.nBVs;
                            obbTree = other.obbTree;
                            vertexPointer = other.vertexPointer;
                            vertexTfPointer = other.vertexTfPointer;
                            triIdxPointer = other.triIdxPointer;
                            modelTransform = other.modelTransform;

                            d_modelTransform = other.d_modelTransform;
                        }

                        return *this;
                    }

                    unsigned int nVerts;
                    unsigned int nTris;
                    unsigned int nBVs;

                    gpTransform modelTransform;
                    gProximityGPUTransform* d_modelTransform;

                    void* obbTree;
                    void* vertexPointer;
                    void* vertexTfPointer;
                    void* triIdxPointer;

                    unsigned int vbo_Vertex, vbo_TriIndex; // for OPENGL interoperability
            };
        }
    }
}


#endif //OBBTREEGPU_CUDA_DATA_STRUCTURES_H
