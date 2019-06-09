#include "ObbTreeGPUTriangleCollision_cuda.h"

#include <iomanip>

#include "cutil/cutil.h"
#include "gProximity/cuda_defs.h"
#include "gProximity/cuda_make_grid.h"

#ifdef USE_TRUNK_CUDA_THRUST
#include <thrust_v180/device_vector.h>
#include <thrust_v180/remove.h>
#include <thrust_v180/system/cuda/execution_policy.h>
#else
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/system/cuda/execution_policy.h>
#endif

#include "gProximity/cuda_intersect_tree.h"
#include "gProximity/cuda_intersect_tritri.h"

#include <sofa/helper/AdvancedTimer.h>

struct gProximityWorkerResultPrivate
{
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
    thrust::host_vector<bool, std::allocator<bool> > h_valid;
    thrust::host_vector<int, std::allocator<int> > h_contactId;
    thrust::host_vector<double, std::allocator<double> > h_distance;
    thrust::host_vector<int4, std::allocator<int4> > h_elems;
    thrust::host_vector<float3, std::allocator<float3> > h_point0;
    thrust::host_vector<float3, std::allocator<float3> > h_point1;
    thrust::host_vector<float3, std::allocator<float3> > h_normal;
    thrust::host_vector<gProximityContactType, std::allocator<gProximityContactType> > h_gProximityContactType;
#else
    bool* h_valid;
    int* h_contactId;
    double* h_distance;
    int4* h_elems;
    float3* h_point0;
    float3* h_point1;
    float3* h_normal;
    gProximityContactType* h_gProximityContactType;
#endif
};


// Make predicate easy to write
typedef thrust::tuple< bool, int, int4, double, float3, float3, float3, gProximityContactType > ContactPointTuple;

// Predicate
struct ContactPointValid
{
public:

    /*double contactDistance;

    ContactPointValid(const double& contact_distance = 0.001): contactDistance(contact_distance)
    {}*/

    __host__ __device__ bool operator() (const ContactPointTuple& tup)
    {
        const int valid = (int) thrust::get<0>(tup);
        //printf("valid = %d\n", valid);
        /*if (valid)
        {
        const float3& pt0 = thrust::get<4>( tup );
        const float3& pt1 = thrust::get<5>( tup );
        float3 pt0_pt1 = f3v_sub(pt1, pt0);
        float len = f3v_len(pt0_pt1);

        return (len >= contactDistance || len < 1e-06);
        }
        else
        return true;*/
        return (valid != 1);
    }
};

void ObbTreeGPU_TriangleIntersection(sofa::component::collision::OBBContainer *model1,
    sofa::component::collision::OBBContainer *model2,
    gProximityWorkerUnit* workerUnit,
    gProximityWorkerResult* workerResult,
    double alarmDistance,
    double contactDistance,
    int& nIntersecting)
{
    GPUVertex* d_tfVertices1 = (GPUVertex*)model1->vertexTfPointer;
    GPUVertex* d_tfVertices2 = (GPUVertex*)model2->vertexTfPointer;
    uint3* d_triIndices1 = (uint3*)model1->triIdxPointer;
    uint3* d_triIndices2 = (uint3*)model2->triIdxPointer;

    dim3 grids = makeGrid((int)ceilf(workerUnit->_nCollidingPairs / (float)COLLISION_THREADS * 8));
    dim3 threads(COLLISION_THREADS * 8, 1, 1);

#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG
    std::cout << "CUDA_trianglePairIntersect: " << nPairs << " colliding tri-pairs, grid dim. = " << grids.x << "," << grids.y << "," << grids.z << ", threads dim = " << threads.x << "," << threads.y << "," << threads.z << std::endl;
#endif

    GPUMEMSET(workerResult->d_valid, false, sizeof(bool)* workerResult->_maxResults);
    GPUMEMSET(workerResult->d_contactId, -1, sizeof(int)* workerResult->_maxResults);

    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriangleIntersect");

    trianglePairIntersections << < grids, threads, 0, *(workerUnit->_stream) >> > (d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2,
        workerUnit->d_collisionPairs, workerUnit->_nCollidingPairs, workerResult->d_contacts, workerUnit->_nCollidingPairs * CollisionTestElementsSize, alarmDistance, contactDistance);

    cudaStreamSynchronize(*(workerUnit->_stream));

    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriangleIntersect");

    typedef thrust::device_vector< bool >::iterator  BoolDIter;
    typedef thrust::device_vector< int >::iterator  IntDIter;
    typedef thrust::device_vector< double >::iterator  DoubleDIter;
    typedef thrust::device_vector< int4 >::iterator  Int4DIter;
    typedef thrust::device_vector< float3 >::iterator  Float3DIter;
    typedef thrust::device_vector< gProximityContactType >::iterator gctDIter;

    typedef thrust::tuple< BoolDIter, IntDIter, Int4DIter, DoubleDIter, Float3DIter, Float3DIter, Float3DIter, gctDIter >  ContactPointDIterTuple;
    typedef thrust::zip_iterator< ContactPointDIterTuple >  ZipDIter;

    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_ResultClean");

    thrust::device_ptr<bool> devValidPtr_begin(workerResult->h_contacts->valid);
    thrust::device_ptr<bool> devValidPtr_end(workerResult->h_contacts->valid + workerResult->_maxResults);
    thrust::device_vector<bool> devValidVec(devValidPtr_begin, devValidPtr_end);

    thrust::device_ptr<int> devContactIdPtr_begin(workerResult->h_contacts->contactId);
    thrust::device_ptr<int> devContactIdPtr_end(workerResult->h_contacts->contactId + workerResult->_maxResults);
    thrust::device_vector<int> devContactIdVec(devContactIdPtr_begin, devContactIdPtr_end);

    thrust::device_ptr<int4> devElemsPtr_begin(workerResult->h_contacts->elems);
    thrust::device_ptr<int4> devElemsPtr_end(workerResult->h_contacts->elems + workerResult->_maxResults);
    thrust::device_vector<int4> devElemsVec(devElemsPtr_begin, devElemsPtr_end);

    thrust::device_ptr<double> devDistancePtr_begin(workerResult->h_contacts->distance);
    thrust::device_ptr<double> devDistancePtr_end(workerResult->h_contacts->distance + workerResult->_maxResults);
    thrust::device_vector<double> devDistanceVec(devDistancePtr_begin, devDistancePtr_end);

    thrust::device_ptr<float3> devPoint0Ptr_begin(workerResult->h_contacts->point0);
    thrust::device_ptr<float3> devPoint0Ptr_end(workerResult->h_contacts->point0 + workerResult->_maxResults);
    thrust::device_vector<float3> devPoint0Vec(devPoint0Ptr_begin, devPoint0Ptr_end);

    thrust::device_ptr<float3> devPoint1Ptr_begin(workerResult->h_contacts->point1);
    thrust::device_ptr<float3> devPoint1Ptr_end(workerResult->h_contacts->point1 + workerResult->_maxResults);
    thrust::device_vector<float3> devPoint1Vec(devPoint1Ptr_begin, devPoint1Ptr_end);

    thrust::device_ptr<float3> devNormalPtr_begin(workerResult->h_contacts->normal);
    thrust::device_ptr<float3> devNormalPtr_end(workerResult->h_contacts->normal + workerResult->_maxResults);
    thrust::device_vector<float3> devNormalVec(devNormalPtr_begin, devNormalPtr_end);

    thrust::device_ptr<gProximityContactType> devContactTypePtr_begin(workerResult->h_contacts->contactType);
    thrust::device_ptr<gProximityContactType> devContactTypePtr_end(workerResult->h_contacts->contactType + workerResult->_maxResults);
    thrust::device_vector<gProximityContactType> devContactTypeVec(devContactTypePtr_begin, devContactTypePtr_end);

    ZipDIter contactPointsVecBegin = thrust::make_zip_iterator(thrust::make_tuple(devValidVec.begin(),
        devContactIdVec.begin(),
        devElemsVec.begin(),
        devDistanceVec.begin(),
        devPoint0Vec.begin(),
        devPoint1Vec.begin(),
        devNormalVec.begin(),
        devContactTypeVec.begin()
        ));

    ZipDIter newEnd = thrust::remove_if(contactPointsVecBegin,
        thrust::make_zip_iterator(thrust::make_tuple(devValidVec.end(),
        devContactIdVec.end(),
        devElemsVec.end(),
        devDistanceVec.end(),
        devPoint0Vec.end(),
        devPoint1Vec.end(),
        devNormalVec.end(),
        devContactTypeVec.end()
        )),
        ContactPointValid());

    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_ResultClean");

    ContactPointDIterTuple endTuple = newEnd.get_iterator_tuple();
    IntDIter contactIdVecEnd = thrust::get<1>(endTuple);
    int validElems = contactIdVecEnd - devContactIdVec.begin();

    std::cout << std::endl;
    std::cout << "   Valid elements count = " << validElems << std::endl;
    std::cout << std::endl;

    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_ResultCopy");

#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
    thrust::copy_n(devContactIdVec.begin(), validElems, workerResult->d_ptr->h_contactId.begin());
    thrust::copy_n(devElemsVec.begin(), validElems, workerResult->d_ptr->h_elems.begin());
    thrust::copy_n(devDistanceVec.begin(), validElems, workerResult->d_ptr->h_distance.begin());
    thrust::copy_n(devPoint0Vec.begin(), validElems, workerResult->d_ptr->h_point0.begin());
    thrust::copy_n(devPoint1Vec.begin(), validElems, workerResult->d_ptr->h_point1.begin());
    thrust::copy_n(devNormalVec.begin(), validElems, workerResult->d_ptr->h_normal.begin());
    thrust::copy_n(devContactTypeVec.begin(), validElems, workerResult->d_ptr->h_gProximityContactType.begin());
#else
    FROMGPU(workerResult->d_ptr->h_contactId, thrust::raw_pointer_cast(devContactIdVec.data()), sizeof(int) * validElems);
    FROMGPU(workerResult->d_ptr->h_distance, thrust::raw_pointer_cast(devDistanceVec.data()), sizeof(double) * validElems);
    FROMGPU(workerResult->d_ptr->h_elems, thrust::raw_pointer_cast(devElemsVec.data()), sizeof(int4) * validElems);
    FROMGPU(workerResult->d_ptr->h_normal, thrust::raw_pointer_cast(devNormalVec.data()), sizeof(float3) * validElems);
    FROMGPU(workerResult->d_ptr->h_point0, thrust::raw_pointer_cast(devPoint0Vec.data()), sizeof(float3) * validElems);
    FROMGPU(workerResult->d_ptr->h_point1, thrust::raw_pointer_cast(devPoint1Vec.data()), sizeof(float3) * validElems);
    FROMGPU(workerResult->d_ptr->h_gProximityContactType, thrust::raw_pointer_cast(devContactTypeVec.data()), sizeof(gProximityContactType) * validElems);
#endif

    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_ResultCopy");

    workerResult->_numResults = validElems;

    nIntersecting = validElems;
}

//#define CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
void ObbTreeGPU_TriangleIntersection_Streams(sofa::component::collision::OBBContainer *model1,
    sofa::component::collision::OBBContainer *model2,
    gProximityWorkerUnit* workerUnit,
    std::vector<gProximityWorkerResult*>& workerResults,
    std::vector<std::pair<unsigned int, unsigned int> >& workerResultSizes,
    std::vector<unsigned int>& workerResultStartIndices,
    std::vector<cudaStream_t>& triTestStreams,
    bool cleanResults,
    double alarmDistance,
    double contactDistance,
    int& nIntersecting)
{
    GPUVertex* d_tfVertices1 = (GPUVertex*)model1->vertexTfPointer;
    GPUVertex* d_tfVertices2 = (GPUVertex*)model2->vertexTfPointer;
    uint3* d_triIndices1 = (uint3*)model1->triIdxPointer;
    uint3* d_triIndices2 = (uint3*)model2->triIdxPointer;

#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
    std::cout << "ObbTreeGPU_TriangleIntersection_Streams: " << workerResults.size() << " result units to process." << std::endl;
    std::cout << "  workerUnit index = " << workerUnit->_workerUnitIndex << ", worker unit results = " << workerUnit->_nCollidingPairs << std::endl;
#endif

    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_TriangleIntersection_Streams");

    unsigned int numIterations = workerResults.size();
    std::cout << "  numIterations for bin checks = " << numIterations << std::endl;

    unsigned int numThreads = numIterations;
    /*if (numIterations >= 4)
        numThreads = numIterations / 2;*/

    if (numThreads > 4)
        numThreads = 4;

    std::cout << "  threads for parallel checks = " << numThreads << std::endl;

    unsigned int numStreams = triTestStreams.size();

    dim3 grids = makeGrid((int)ceilf(workerUnit->_nCollidingPairs / (float)COLLISION_THREADS * 8));
    dim3 threads(COLLISION_THREADS * 8, 1, 1);

#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
    std::cout << " * unit " << workerUnit->_workerUnitIndex << ": " << workerUnit->_nCollidingPairs << " colliding tri-pairs, grid dim. = " << grids.x << "," << grids.y << "," << grids.z << ", threads dim = " << threads.x << "," << threads.y << "," << threads.z << std::endl;
#endif

    #pragma omp parallel for num_threads(numThreads)
    for (unsigned int k = 0; k < numIterations; k++)
    {
        if (k < numIterations)
        {
            gProximityWorkerResult* workerResult1 = workerResults[k];
    #ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
            std::cout << "   unit maxResults = " << workerResult1->_maxResults << "; blocked = " << workerResult1->_blocked << std::endl;
            std::cout << "   range start at index = " << workerResultStartIndices[k] << std::endl;
            std::cout << "   range from sizes vector = " << workerResultSizes[k].first << " - " << workerResultSizes[k].second << std::endl;
    #endif
            GPUMEMSET_ASYNC(workerResult1->d_valid, false, sizeof(bool) * workerResult1->_maxResults, *(workerUnit->_stream));
            //GPUMEMSET_ASYNC(workerResult1->d_swapped, false, sizeof(bool) * workerResult1->_maxResults, *(workerUnit->_stream));
            //GPUMEMSET_ASYNC(workerResult1->d_contactId, -1, sizeof(int)* workerResult1->_maxResults, *(workerUnit->_stream));

            sofa::helper::AdvancedTimer::stepBegin("trianglePairIntersections_Streamed k");
            trianglePairIntersections_Streamed <<< grids, threads, 0, triTestStreams[k % numStreams] /**(workerUnit->_stream)*/ >>> (d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2,
                workerUnit->d_collisionPairs, workerUnit->_nCollidingPairs, workerResult1->d_contacts,
                model1->nVerts, model1->nTris,
                model2->nVerts, model2->nTris,
                workerResultStartIndices[k],
                workerResult1->d_outputIndex,
                workerResultSizes[k].first, workerResultSizes[k].second,
                workerUnit->_nCollidingPairs * CollisionTestElementsSize, alarmDistance, contactDistance);

            sofa::helper::AdvancedTimer::stepEnd("trianglePairIntersections_Streamed k");
            //cudaStreamSynchronize(triTestStream);
            //cudaStreamSynchronize(*(workerUnit->_stream));

        }
#if 0
        if (k + 1 < numIterations)
        {
            gProximityWorkerResult* workerResult2 = workerResults[k + 1];
    #ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
            std::cout << "   unit maxResults = " << workerResult2->_maxResults << "; blocked = " << workerResult2->_blocked << std::endl;
            std::cout << "   range start at index = " << workerResultStartIndices[k+1] << std::endl;
            std::cout << "   range from sizes vector = " << workerResultSizes[k+1].first << " - " << workerResultSizes[k+1].second << std::endl;
    #endif
            GPUMEMSET_ASYNC(workerResult2->d_valid, false, sizeof(bool) * workerResult2->_maxResults, *(workerUnit->_stream));
            //GPUMEMSET_ASYNC(workerResult2->d_swapped, false, sizeof(bool) * workerResult2->_maxResults, *(workerUnit->_stream));
            //GPUMEMSET_ASYNC(workerResult2->d_contactId, -1, sizeof(int)* workerResult2->_maxResults, *(workerUnit->_stream));

            sofa::helper::AdvancedTimer::stepBegin("trianglePairIntersections_Streamed k + 1");
            trianglePairIntersections_Streamed <<< grids, threads, 0, triTestStreams[(k + 1) % numStreams] /**(workerUnit->_stream)*/ >>> (d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2,
                workerUnit->d_collisionPairs, workerUnit->_nCollidingPairs, workerResult2->d_contacts,
                model1->nVerts, model1->nTris,
                model2->nVerts, model2->nTris,
                workerResultStartIndices[k+1],
                workerResultSizes[k+1].first, workerResultSizes[k+1].second,
                workerUnit->_nCollidingPairs * CollisionTestElementsSize, alarmDistance, contactDistance);

            sofa::helper::AdvancedTimer::stepEnd("trianglePairIntersections_Streamed k + 1");

            //cudaStreamSynchronize(triTestStream);
            //cudaStreamSynchronize(*(workerUnit->_stream));
        }
        if (k + 2 < numIterations)
        {
            gProximityWorkerResult* workerResult3 = workerResults[k + 2];
    #ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
            std::cout << "   unit maxResults = " << workerResult3->_maxResults << "; blocked = " << workerResult3->_blocked << std::endl;
            std::cout << "   range start at index = " << workerResultStartIndices[k+2] << std::endl;
            std::cout << "   range from sizes vector = " << workerResultSizes[k+2].first << " - " << workerResultSizes[k+2].second << std::endl;
    #endif
            GPUMEMSET_ASYNC(workerResult3->d_valid, false, sizeof(bool) * workerResult3->_maxResults, *(workerUnit->_stream));
            //GPUMEMSET_ASYNC(workerResult3->d_swapped, false, sizeof(bool) * workerResult3->_maxResults, *(workerUnit->_stream));
            //GPUMEMSET_ASYNC(workerResult3->d_contactId, -1, sizeof(int)* workerResult3->_maxResults, *(workerUnit->_stream));

            sofa::helper::AdvancedTimer::stepBegin("trianglePairIntersections_Streamed k + 2");

            trianglePairIntersections_Streamed <<< grids, threads, 0, triTestStreams[(k + 2) % numStreams] /**(workerUnit->_stream)*/ >>> (d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2,
                workerUnit->d_collisionPairs, workerUnit->_nCollidingPairs, workerResult3->d_contacts,
                model1->nVerts, model1->nTris,
                model2->nVerts, model2->nTris,
                workerResultStartIndices[k+2],
                workerResultSizes[k+2].first, workerResultSizes[k+2].second,
                workerUnit->_nCollidingPairs * CollisionTestElementsSize, alarmDistance, contactDistance);

            sofa::helper::AdvancedTimer::stepEnd("trianglePairIntersections_Streamed k + 2");

            //cudaStreamSynchronize(triTestStream);
            //cudaStreamSynchronize(*(workerUnit->_stream));

        }
        if (k + 3 < numIterations)
        {
            gProximityWorkerResult* workerResult4 = workerResults[k + 3];
    #ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
            std::cout << "   unit maxResults = " << workerResult4->_maxResults << "; blocked = " << workerResult4->_blocked << std::endl;
            std::cout << "   range start at index = " << workerResultStartIndices[k+3] << std::endl;
            std::cout << "   range from sizes vector = " << workerResultSizes[k+3].first << " - " << workerResultSizes[k+3].second << std::endl;
    #endif
            GPUMEMSET_ASYNC(workerResult4->d_valid, false, sizeof(bool) * workerResult4->_maxResults, *(workerUnit->_stream));
            //GPUMEMSET_ASYNC(workerResult4->d_swapped, false, sizeof(bool) * workerResult4->_maxResults, *(workerUnit->_stream));
            //GPUMEMSET_ASYNC(workerResult4->d_contactId, -1, sizeof(int)* workerResult4->_maxResults, *(workerUnit->_stream));

            sofa::helper::AdvancedTimer::stepBegin("trianglePairIntersections_Streamed k + 3");

            trianglePairIntersections_Streamed <<< grids, threads, 0, triTestStreams[(k + 3) % numStreams] /**(workerUnit->_stream)*/ >>> (d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2,
                workerUnit->d_collisionPairs, workerUnit->_nCollidingPairs, workerResult4->d_contacts,
                model1->nVerts, model1->nTris,
                model2->nVerts, model2->nTris,
                workerResultStartIndices[k+3],
                workerResultSizes[k+3].first, workerResultSizes[k+3].second,
                workerUnit->_nCollidingPairs * CollisionTestElementsSize, alarmDistance, contactDistance);

            sofa::helper::AdvancedTimer::stepEnd("trianglePairIntersections_Streamed k + 3");

            //cudaStreamSynchronize(triTestStream);
            //cudaStreamSynchronize(*(workerUnit->_stream));

        }
#endif
    }

    /*for (unsigned int k = 0; k < numStreams; k++)
    {
        cudaStreamSynchronize(triTestStreams[k]);
    }*/

#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED_VERTICES
        GPUVertex* vertices_model1 = new GPUVertex[model1->nVerts];
        GPUVertex* vertices_model2 = new GPUVertex[model2->nVerts];

        FROMGPU(vertices_model1, d_tfVertices1, sizeof(GPUVertex) * model1->nVerts);
        FROMGPU(vertices_model2, d_tfVertices2, sizeof(GPUVertex) * model2->nVerts);

        std::cout << "=== Transformed vertices from GPU ===" << std::endl;
        std::cout << " model1: " << std::endl;
        for (unsigned int l = 0; l < model1->nVerts; l++)
        {
            std::cout << " * " << l << ": " << vertices_model1[l].v.x << "," << vertices_model1[l].v.y << "," << vertices_model1[l].v.z << std::endl;
        }
        std::cout << " model2: " << std::endl;
        for (unsigned int l = 0; l < model2->nVerts; l++)
        {
            std::cout << " * " << l << ": " << vertices_model2[l].v.x << "," << vertices_model2[l].v.y << "," << vertices_model2[l].v.z << std::endl;
        }

        uint3* indices_model1 = new uint3[model1->nTris];
        uint3* indices_model2 = new uint3[model2->nTris];

        FROMGPU(indices_model1, d_triIndices1, sizeof(uint3) * model1->nTris);
        FROMGPU(indices_model2, d_triIndices2, sizeof(uint3) * model2->nTris);

        std::cout << "=== Indices from GPU ===" << std::endl;
        std::cout << " model1: " << std::endl;
        for (unsigned int l = 0; l < model1->nTris; l++)
        {
            std::cout << " * " << l << ": " << indices_model1[l].x << "," << indices_model1[l].y << "," << indices_model1[l].z << std::endl;
        }
        std::cout << " model2: " << std::endl;
        for (unsigned int l = 0; l < model2->nTris; l++)
        {
            std::cout << " * " << l << ": " << indices_model2[l].x << "," << indices_model2[l].y << "," << indices_model2[l].z << std::endl;
        }

        delete[] vertices_model1;
        delete[] vertices_model2;
        delete[] indices_model1;
        delete[] indices_model2;
#endif

#if 0
        trianglePairIntersections_Streamed <<< grids, threads, 0, triTestStream /**(workerUnit->_stream)*/ >>> (d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2,
            workerUnit->d_collisionPairs, workerUnit->_nCollidingPairs, workerResult->d_contacts,
            model1->nVerts, model1->nTris,
            model2->nVerts, model2->nTris,
            workerResultStartIndices[k],
            workerResultSizes[k].first, workerResultSizes[k].second,
            workerUnit->_nCollidingPairs * CollisionTestElementsSize, alarmDistance, contactDistance);

        cudaStreamSynchronize(triTestStream);
        //cudaStreamSynchronize(*(workerUnit->_stream));

    #pragma omp parallel for num_threads(numThreads)
    for (unsigned int k = 0; k < numIterations; k++/*= numThreads*/)
    {
        if (k < numIterations)
        {
            sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_PostProcess_TriangleIntersection_Streamed k");

            ObbTreeGPU_PostProcess_TriangleIntersection_Streamed(workerUnit, workerResults[k], triTestStreams[k % numStreams], cleanResults);

            sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_PostProcess_TriangleIntersection_Streamed k");
        }
    }

    cudaStreamSynchronize(*(workerUnit->_stream));

    for (unsigned int k = 0; k < numStreams; k++)
        cudaStreamSynchronize(triTestStreams[k]);

    for (unsigned int k = 0; k < numIterations; k++)
    {
        nIntersecting += workerResults[k]->_numResults;
    }
    std::cout << "  nIntersecting = " << nIntersecting << std::endl;
#endif
    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_TriangleIntersection_Streams");
}

void ObbTreeGPU_PostProcess_TriangleIntersection_Streamed(gProximityWorkerUnit* workerUnit,
                                                          gProximityWorkerResult* workerResult,
                                                          cudaStream_t& triTestStream,
                                                          bool cleanResults)
{
    /*workerResult->d_ptr->h_contactId.clear();
    workerResult->d_ptr->h_distance.clear();
    workerResult->d_ptr->h_elems.clear();
    workerResult->d_ptr->h_gProximityContactType.clear();
    workerResult->d_ptr->h_normal.clear();
    workerResult->d_ptr->h_point0.clear();
    workerResult->d_ptr->h_point1.clear();*/

    typedef thrust::device_vector< bool >::iterator  BoolDIter;
    typedef thrust::device_vector< int >::iterator  IntDIter;
    typedef thrust::device_vector< double >::iterator  DoubleDIter;
    typedef thrust::device_vector< int4 >::iterator  Int4DIter;
    typedef thrust::device_vector< float3 >::iterator  Float3DIter;
    typedef thrust::device_vector< gProximityContactType >::iterator gctDIter;

    typedef thrust::tuple< BoolDIter, IntDIter, Int4DIter, DoubleDIter, Float3DIter, Float3DIter, Float3DIter, gctDIter >  ContactPointDIterTuple;
    typedef thrust::zip_iterator< ContactPointDIterTuple >  ZipDIter;

    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_TriangleIntersection_Streams_ResultClean");

#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED_VERBOSE
    std::cout << " Allocate thrust::device_ptr to valid array on GPU" << std::endl;
#endif

    thrust::device_ptr<bool> devValidPtr_begin(workerResult->h_contacts->valid);
    thrust::device_ptr<bool> devValidPtr_end(workerResult->h_contacts->valid + workerResult->_maxResults);
    thrust::device_vector<bool> devValidVec(devValidPtr_begin, devValidPtr_end);

#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED_VERBOSE
    std::cout << " Allocate thrust::device_ptr to valid array on GPU: DONE" << std::endl;
#endif

#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED_VERBOSE
    std::cout << "   valid vector size = " << devValidVec.size() << std::endl;

    std::cout << "   contact point dump from workerResult arrays:" << std::endl;

    bool* validArray = new bool[workerResult->_maxResults];
    FROMGPU(validArray, workerResult->d_valid, sizeof(bool) * workerResult->_maxResults);

    bool* swappedArray = new bool[workerResult->_maxResults];
    FROMGPU(swappedArray, workerResult->d_swapped, sizeof(bool) * workerResult->_maxResults);

    int* idArray = new int[workerResult->_maxResults];
    FROMGPU(idArray, workerResult->d_contactId, sizeof(int) * workerResult->_maxResults);

    double* distArray = new double[workerResult->_maxResults];
    FROMGPU(distArray, workerResult->d_distance, sizeof(double) * workerResult->_maxResults);

    float3* point0Array = new float3[workerResult->_maxResults];
    FROMGPU(point0Array, workerResult->d_point0, sizeof(float3) * workerResult->_maxResults);

    float3* point1Array = new float3[workerResult->_maxResults];
    FROMGPU(point1Array, workerResult->d_point1, sizeof(float3) * workerResult->_maxResults);

    float3* normalArray = new float3[workerResult->_maxResults];
    FROMGPU(normalArray, workerResult->d_normal, sizeof(float3) * workerResult->_maxResults);

    gProximityContactType* typeArray = new gProximityContactType[workerResult->_maxResults];
    FROMGPU(typeArray, workerResult->d_gProximityContactType, sizeof(gProximityContactType) * workerResult->_maxResults);

    for (int k = 0; k < workerResult->_maxResults; k++)
    {
        if (validArray[k] == true)
            std::cout << "   * " << k << " valid: " << " swapped = " << swappedArray[k] << ", id = " << idArray[k] <<
                         ", point0 = " << point0Array[k].x << "," << point0Array[k].y << "," << point0Array[k].x <<
                         ", point1 = " << point1Array[k].x << "," << point1Array[k].y << "," << point1Array[k].x <<
                         ", normal = " << normalArray[k].x << "," << normalArray[k].y << "," << normalArray[k].x <<
                         ", type = " << typeArray[k] <<
                         ", distance = " << distArray[k] << std::endl;
    }

    delete[] validArray;
    delete[] swappedArray;
    delete[] idArray;
    delete[] distArray;
    delete[] point0Array;
    delete[] point1Array;
    delete[] normalArray;
    delete[] typeArray;

    if (devValidVec.size() > 0)
    {
        thrust::host_vector<bool> validVec = devValidVec;
        std::cout << "   valid entries: ";
        for (unsigned int k = 0; k < validVec.size(); k++)
        {
            if ((int) validVec[k] == 1)
                std::cout << k << ": " << validVec[k] << ";";
        }
        std::cout << std::endl;
    }
#endif

    thrust::device_ptr<int> devContactIdPtr_begin(workerResult->h_contacts->contactId);
    thrust::device_ptr<int> devContactIdPtr_end(workerResult->h_contacts->contactId + workerResult->_maxResults);
    thrust::device_vector<int> devContactIdVec(devContactIdPtr_begin, devContactIdPtr_end);

    thrust::device_ptr<int4> devElemsPtr_begin(workerResult->h_contacts->elems);
    thrust::device_ptr<int4> devElemsPtr_end(workerResult->h_contacts->elems + workerResult->_maxResults);
    thrust::device_vector<int4> devElemsVec(devElemsPtr_begin, devElemsPtr_end);

    thrust::device_ptr<double> devDistancePtr_begin(workerResult->h_contacts->distance);
    thrust::device_ptr<double> devDistancePtr_end(workerResult->h_contacts->distance + workerResult->_maxResults);
    thrust::device_vector<double> devDistanceVec(devDistancePtr_begin, devDistancePtr_end);

    thrust::device_ptr<float3> devPoint0Ptr_begin(workerResult->h_contacts->point0);
    thrust::device_ptr<float3> devPoint0Ptr_end(workerResult->h_contacts->point0 + workerResult->_maxResults);
    thrust::device_vector<float3> devPoint0Vec(devPoint0Ptr_begin, devPoint0Ptr_end);

    thrust::device_ptr<float3> devPoint1Ptr_begin(workerResult->h_contacts->point1);
    thrust::device_ptr<float3> devPoint1Ptr_end(workerResult->h_contacts->point1 + workerResult->_maxResults);
    thrust::device_vector<float3> devPoint1Vec(devPoint1Ptr_begin, devPoint1Ptr_end);

    thrust::device_ptr<float3> devNormalPtr_begin(workerResult->h_contacts->normal);
    thrust::device_ptr<float3> devNormalPtr_end(workerResult->h_contacts->normal + workerResult->_maxResults);
    thrust::device_vector<float3> devNormalVec(devNormalPtr_begin, devNormalPtr_end);

    thrust::device_ptr<gProximityContactType> devContactTypePtr_begin(workerResult->h_contacts->contactType);
    thrust::device_ptr<gProximityContactType> devContactTypePtr_end(workerResult->h_contacts->contactType + workerResult->_maxResults);
    thrust::device_vector<gProximityContactType> devContactTypeVec(devContactTypePtr_begin, devContactTypePtr_end);

    if (cleanResults)
    {
        ZipDIter contactPointsVecBegin = thrust::make_zip_iterator(thrust::make_tuple(devValidVec.begin(),
                                                                                      devContactIdVec.begin(),
                                                                                      devElemsVec.begin(),
                                                                                      devDistanceVec.begin(),
                                                                                      devPoint0Vec.begin(),
                                                                                      devPoint1Vec.begin(),
                                                                                      devNormalVec.begin(),
                                                                                      devContactTypeVec.begin()
                                                                                      ));

        ZipDIter contactPointsVecEnd = thrust::make_zip_iterator(thrust::make_tuple(devValidVec.end(),
                                                                                    devContactIdVec.end(),
                                                                                    devElemsVec.end(),
                                                                                    devDistanceVec.end(),
                                                                                    devPoint0Vec.end(),
                                                                                    devPoint1Vec.end(),
                                                                                    devNormalVec.end(),
                                                                                    devContactTypeVec.end()));

        ZipDIter newEnd = thrust::remove_if(
                    //thrust::cuda::par.on(*(workerUnit->_stream)),
                    contactPointsVecBegin,
                    contactPointsVecEnd,
                    ContactPointValid());

        //cudaStreamSynchronize(*(workerUnit->_stream));

        sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_TriangleIntersection_Streams_ResultClean");

        ContactPointDIterTuple endTuple = newEnd.get_iterator_tuple();
        IntDIter contactIdVecEnd = thrust::get<1>(endTuple);

        int validElems = contactIdVecEnd - devContactIdVec.begin();

    #ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED_VERBOSE
        std::cout << std::endl;
        std::cout << "   Valid elements count = " << validElems << std::endl;
        std::cout << std::endl;
        if (validElems > 0)
        {
            thrust::host_vector<bool> validVec = devValidVec;

            for (unsigned int k = 0; k < validElems; k++)
            {
                std::cout << validVec[k] << ";";
            }
            std::cout << std::endl;
        }
    #endif

        if (validElems > 0)
        {
            try
            {
                sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_ResultCopy");
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_contactId.data()), thrust::raw_pointer_cast(devContactIdVec.data()), sizeof(int) * validElems, triTestStream /**(workerUnit->_stream)*/);
                FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_distance.data()), thrust::raw_pointer_cast(devDistanceVec.data()), sizeof(double) * validElems, triTestStream /**(workerUnit->_stream)*/);
                FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_elems.data()), thrust::raw_pointer_cast(devElemsVec.data()), sizeof(int4) * validElems, triTestStream /**(workerUnit->_stream)*/);
                FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_normal.data()), thrust::raw_pointer_cast(devNormalVec.data()), sizeof(float3) * validElems, triTestStream /**(workerUnit->_stream)*/);
                FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_point0.data()), thrust::raw_pointer_cast(devPoint0Vec.data()), sizeof(float3) * validElems, triTestStream /**(workerUnit->_stream)*/);
                FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_point1.data()), thrust::raw_pointer_cast(devPoint1Vec.data()), sizeof(float3) * validElems, triTestStream /**(workerUnit->_stream)*/);
                FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_gProximityContactType.data()), thrust::raw_pointer_cast(devContactTypeVec.data()), sizeof(gProximityContactType) * validElems, triTestStream /**(workerUnit->_stream)*/);
#else
                FROMGPU(workerResult->d_ptr->h_valid, thrust::raw_pointer_cast(devValidVec.data()), sizeof(bool) * validElems);
                FROMGPU(workerResult->d_ptr->h_contactId, thrust::raw_pointer_cast(devContactIdVec.data()), sizeof(int) * validElems);
                FROMGPU(workerResult->d_ptr->h_distance, thrust::raw_pointer_cast(devDistanceVec.data()), sizeof(double) * validElems);
                FROMGPU(workerResult->d_ptr->h_elems, thrust::raw_pointer_cast(devElemsVec.data()), sizeof(int4) * validElems);
                FROMGPU(workerResult->d_ptr->h_normal, thrust::raw_pointer_cast(devNormalVec.data()), sizeof(float3) * validElems);
                FROMGPU(workerResult->d_ptr->h_point0, thrust::raw_pointer_cast(devPoint0Vec.data()), sizeof(float3) * validElems);
                FROMGPU(workerResult->d_ptr->h_point1, thrust::raw_pointer_cast(devPoint1Vec.data()), sizeof(float3) * validElems);
                FROMGPU(workerResult->d_ptr->h_gProximityContactType, thrust::raw_pointer_cast(devContactTypeVec.data()), sizeof(gProximityContactType) * validElems);
#endif
                //cudaStreamSynchronize(triTestStream/**(workerUnit->_stream)*/);

                workerResult->_numResults = validElems;

                sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_ResultCopy");

    #ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED_VERBOSE
                {
                    for (unsigned int k = 0; k < validElems; k++)
                    {
                        std::cout << " * " << k << ": type = " << workerResult->d_ptr->h_gProximityContactType[k]
                                                << ", id = " << workerResult->d_ptr->h_contactId[k]
                                                << ", distance = " << workerResult->d_ptr->h_distance[k]
                                                << ", point0 = " << workerResult->d_ptr->h_point0[k].x << "," << workerResult->d_ptr->h_point0[k].y << "," << workerResult->d_ptr->h_point0[k].z
                                                << ", point1 = " << workerResult->d_ptr->h_point1[k].x << "," << workerResult->d_ptr->h_point1[k].y << "," << workerResult->d_ptr->h_point1[k].z
                                                << ", normal = " << workerResult->d_ptr->h_normal[k].x << "," << workerResult->d_ptr->h_normal[k].y << "," << workerResult->d_ptr->h_normal[k].z
                                                << ", elems = " << workerResult->d_ptr->h_elems[k].w << "," << workerResult->d_ptr->h_elems[k].x << "," << workerResult->d_ptr->h_elems[k].y << "," << workerResult->d_ptr->h_elems[k].z
                                                << std::endl;
                    }
                    std::cout << std::endl;
                }
    #endif
            }
            catch (thrust::system_error& ex)
            {
                std::cout << "Thrust exception: " << ex.what() << std::endl;
            }
        }
    }
    else
    {
        sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_ResultCopy");
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_valid.data()), thrust::raw_pointer_cast(devValidVec.data()), sizeof(bool) * workerResult->_maxResults, triTestStream /**(workerUnit->_stream)*/);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_contactId.data()), thrust::raw_pointer_cast(devContactIdVec.data()), sizeof(int) * workerResult->_maxResults, /*triTestStream*/ *(workerUnit->_stream));
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_distance.data()), thrust::raw_pointer_cast(devDistanceVec.data()), sizeof(double) * workerResult->_maxResults, triTestStream /**(workerUnit->_stream)*/);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_elems.data()), thrust::raw_pointer_cast(devElemsVec.data()), sizeof(int4) * workerResult->_maxResults, /*triTestStream*/ *(workerUnit->_stream));
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_normal.data()), thrust::raw_pointer_cast(devNormalVec.data()), sizeof(float3) * workerResult->_maxResults, triTestStream /**(workerUnit->_stream)*/);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_point0.data()), thrust::raw_pointer_cast(devPoint0Vec.data()), sizeof(float3) * workerResult->_maxResults, /*triTestStream*/ *(workerUnit->_stream));
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_point1.data()), thrust::raw_pointer_cast(devPoint1Vec.data()), sizeof(float3) * workerResult->_maxResults, triTestStream /**(workerUnit->_stream)*/);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_gProximityContactType.data()), thrust::raw_pointer_cast(devContactTypeVec.data()), sizeof(gProximityContactType) * workerResult->_maxResults, /*triTestStream*/ *(workerUnit->_stream));
#else
        FROMGPU(workerResult->d_ptr->h_valid, thrust::raw_pointer_cast(devValidVec.data()), sizeof(bool) * workerResult->_maxResults);
        FROMGPU(workerResult->d_ptr->h_contactId, thrust::raw_pointer_cast(devContactIdVec.data()), sizeof(int) * workerResult->_maxResults);
        FROMGPU(workerResult->d_ptr->h_distance, thrust::raw_pointer_cast(devDistanceVec.data()), sizeof(double) * workerResult->_maxResults);
        FROMGPU(workerResult->d_ptr->h_elems, thrust::raw_pointer_cast(devElemsVec.data()), sizeof(int4) * workerResult->_maxResults);
        FROMGPU(workerResult->d_ptr->h_normal, thrust::raw_pointer_cast(devNormalVec.data()), sizeof(float3) * workerResult->_maxResults);
        FROMGPU(workerResult->d_ptr->h_point0, thrust::raw_pointer_cast(devPoint0Vec.data()), sizeof(float3) * workerResult->_maxResults);
        FROMGPU(workerResult->d_ptr->h_point1, thrust::raw_pointer_cast(devPoint1Vec.data()), sizeof(float3) * workerResult->_maxResults);
        FROMGPU(workerResult->d_ptr->h_gProximityContactType, thrust::raw_pointer_cast(devContactTypeVec.data()), sizeof(gProximityContactType) * workerResult->_maxResults);
#endif
        workerResult->_numResults = workerResult->_maxResults;

        sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_ResultCopy");
    }
}

void ObbTreeGPU_TriangleIntersection_Streams_Batch(sofa::component::collision::OBBContainer *model1,
    sofa::component::collision::OBBContainer *model2,
    gProximityWorkerUnit* workerUnit,
    std::vector<gProximityWorkerResult*>& workerResults,
    std::vector<std::pair<unsigned int, unsigned int> >& workerResultSizes,
    std::vector<unsigned int>& workerResultStartIndices,
    std::vector<cudaStream_t>& triTestStreams,
    std::vector<cudaEvent_t>& triTestEvents,
    cudaStream_t& mem_stream,
    cudaEvent_t& startEvent,
    cudaEvent_t& stopEvent,
    double alarmDistance,
    double contactDistance,
    int& nIntersecting,
    float& elapsedTime)
{
    GPUVertex* d_tfVertices1 = (GPUVertex*)model1->vertexTfPointer;
    GPUVertex* d_tfVertices2 = (GPUVertex*)model2->vertexTfPointer;
    uint3* d_triIndices1 = (uint3*)model1->triIdxPointer;
    uint3* d_triIndices2 = (uint3*)model2->triIdxPointer;

#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
    std::cout << "ObbTreeGPU_TriangleIntersection_Streams: " << workerResults.size() << " result units to process." << std::endl;
    std::cout << "  workerUnit index = " << workerUnit->_workerUnitIndex << ", worker unit results = " << workerUnit->_nCollidingPairs << std::endl;
#endif

    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_TriangleIntersection_Streams");

    unsigned int numIterations = workerResults.size();
    std::cout << "  numIterations for bin checks = " << numIterations << std::endl;

    unsigned int numThreads = numIterations;

    if (numThreads > 4)
        numThreads = 4;

    std::cout << "  threads for parallel checks = " << numThreads << std::endl;

    unsigned int numStreams = triTestStreams.size();

    dim3 grids = makeGrid((int)ceilf(workerUnit->_nCollidingPairs / (float)COLLISION_THREADS * 8));
    dim3 threads(COLLISION_THREADS * 8, 1, 1);

#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
    std::cout << " * unit " << workerUnit->_workerUnitIndex << ": " << workerUnit->_nCollidingPairs << " colliding tri-pairs, grid dim. = " << grids.x << "," << grids.y << "," << grids.z << ", threads dim = " << threads.x << "," << threads.y << "," << threads.z << std::endl;
#endif
#if 1
    cudaEventRecord(startEvent, 0);
#endif
    sofa::helper::AdvancedTimer::stepBegin("trianglePairIntersections_Streamed");
#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
    std::cout << "   processing triangle result bins: " << numIterations << std::endl;
#endif
    for (unsigned int k = 0; k < numIterations; k++)
    {

        gProximityWorkerResult* workerResult1 = workerResults[k];
#ifdef CUDA_TRIANGLEPAIRINTERSECT_DEBUG_STREAMED
        std::cout << "   - check triangle bin " << k << std::endl;
        std::cout << "     unit maxResults = " << workerResult1->_maxResults << "; blocked = " << workerResult1->_blocked << std::endl;
        std::cout << "     range start at index = " << workerResultStartIndices[k] << std::endl;
        std::cout << "     range from sizes vector = " << workerResultSizes[k].first << " - " << workerResultSizes[k].second << std::endl;
#endif
        //GPUMEMSET_ASYNC(workerResult1->d_valid, false, sizeof(bool) * workerResult1->_maxResults, *(workerUnit->_stream));
        //GPUMEMSET_ASYNC(workerResult1->d_swapped, false, sizeof(bool) * workerResult1->_maxResults, *(workerUnit->_stream));
        //GPUMEMSET_ASYNC(workerResult1->d_contactId, -1, sizeof(int)* workerResult1->_maxResults, *(workerUnit->_stream));
#if 1
        memset(workerResult1->d_ptr->h_valid, 0, sizeof(bool) * workerResult1->_maxResults);

        trianglePairIntersections_Streamed <<< grids, threads, 0, triTestStreams[k % numStreams] /**(workerUnit->_stream)*/ >>> (d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2,
            workerUnit->d_collisionPairs, workerUnit->_nCollidingPairs, workerResult1->d_contacts,
            model1->nVerts, model1->nTris,
            model2->nVerts, model2->nTris,
            workerResultStartIndices[k],
            workerResult1->d_outputIndex,
            workerResultSizes[k].first, workerResultSizes[k].second,
            workerUnit->_nCollidingPairs * CollisionTestElementsSize, alarmDistance, contactDistance);

        /// TODO: triTestEvents mit TriTestStream-Vektor abstimmen!!!
        CUDA_SAFE_CALL(cudaEventRecord(triTestEvents[k], triTestStreams[k % numStreams]));
#endif
    }
#if 1
    for (unsigned int k = 0; k < numIterations; k++)
    {
        cudaStreamSynchronize(triTestStreams[k % numStreams]);
    }

    for (unsigned int k = 0; k < numIterations; k++)
    {
        gProximityWorkerResult* workerResult = workerResults[k];

        cudaStreamWaitEvent(mem_stream, triTestEvents[k], 0);

        FROMGPU_ASYNC(&(workerResult->h_outputIndex), workerResult->d_outputIndex, sizeof(unsigned int), mem_stream);
    }

    sofa::helper::AdvancedTimer::stepEnd("trianglePairIntersections_Streamed");
#if 1
    cudaEventRecord(stopEvent, 0);
    CUDA_SAFE_CALL(cudaEventSynchronize(stopEvent));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

    //std::cout << " elapsed time for tri-tests = " << elapsedTime << " ms" << std::endl;
#endif

    //std::cout << " storing result counts in result bins: " << std::endl;
    for (unsigned int k = 0; k < numIterations; k++)
    {
        //std::cout << "  * unit " << k << ": " << workerResults[k]->h_outputIndex << std::endl;
        //workerResults[k]->_numResults = workerResults[k]->_maxResults;
        nIntersecting += workerResults[k]->h_outputIndex;
    }
    //std::cout << "  nIntersecting = " << nIntersecting << std::endl;
#endif
    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_TriangleIntersection_Streams");
}

void ObbTreeGPU_PostProcess_TriangleIntersection_Streamed_Batch(gProximityWorkerUnit* workerUnit,
                                                                std::vector<gProximityWorkerResult*>& workerResults,
                                                                cudaStream_t& mem_stream)
{
    typedef thrust::device_vector< bool >::iterator  BoolDIter;
    typedef thrust::device_vector< int >::iterator  IntDIter;
    typedef thrust::device_vector< double >::iterator  DoubleDIter;
    typedef thrust::device_vector< int4 >::iterator  Int4DIter;
    typedef thrust::device_vector< float3 >::iterator  Float3DIter;
    typedef thrust::device_vector< gProximityContactType >::iterator gctDIter;

    typedef thrust::tuple< BoolDIter, IntDIter, Int4DIter, DoubleDIter, Float3DIter, Float3DIter, Float3DIter, gctDIter >  ContactPointDIterTuple;
    typedef thrust::zip_iterator< ContactPointDIterTuple >  ZipDIter;

    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_ResultCopy");
    for (unsigned int k = 0; k < workerResults.size(); k++)
    {
        gProximityWorkerResult* workerResult = workerResults[k];

#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_valid.data()), workerResult->h_contacts->valid, sizeof(bool) * workerResult->_maxResults, mem_stream);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_contactId.data()), workerResult->h_contacts->contactId, sizeof(int) * workerResult->_maxResults, *(workerUnit->_stream) /*mem_stream*/);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_distance.data()), workerResult->h_contacts->distance, sizeof(double) * workerResult->_maxResults, mem_stream);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_elems.data()), workerResult->h_contacts->elems, sizeof(int4) * workerResult->_maxResults, *(workerUnit->_stream) /*mem_stream*/);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_normal.data()), workerResult->h_contacts->normal, sizeof(float3) * workerResult->_maxResults, mem_stream);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_point0.data()), workerResult->h_contacts->point0, sizeof(float3) * workerResult->_maxResults, *(workerUnit->_stream) /*mem_stream*/);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_point1.data()), workerResult->h_contacts->point1, sizeof(float3) * workerResult->_maxResults, mem_stream);
        FROMGPU_ASYNC(thrust::raw_pointer_cast(workerResult->d_ptr->h_gProximityContactType.data()), workerResult->h_contacts->contactType, sizeof(gProximityContactType) * workerResult->_maxResults, *(workerUnit->_stream) /*mem_stream*/);
#else
        FROMGPU_ASYNC(workerResult->d_ptr->h_valid, workerResult->d_valid, sizeof(bool) * workerResult->_maxResults, mem_stream);
        FROMGPU_ASYNC(workerResult->d_ptr->h_contactId, workerResult->d_contactId, sizeof(int) * workerResult->_maxResults, mem_stream);
        FROMGPU_ASYNC(workerResult->d_ptr->h_distance, workerResult->d_distance, sizeof(double) * workerResult->_maxResults, mem_stream);
        FROMGPU_ASYNC(workerResult->d_ptr->h_elems, workerResult->d_elems, sizeof(int4) * workerResult->_maxResults, mem_stream);
        FROMGPU_ASYNC(workerResult->d_ptr->h_normal, workerResult->d_normal, sizeof(float3) * workerResult->_maxResults, *(workerUnit->_stream));
        FROMGPU_ASYNC(workerResult->d_ptr->h_point0, workerResult->d_point0, sizeof(float3) * workerResult->_maxResults, *(workerUnit->_stream));
        FROMGPU_ASYNC(workerResult->d_ptr->h_point1, workerResult->d_point1, sizeof(float3) * workerResult->_maxResults, *(workerUnit->_stream));
        FROMGPU_ASYNC(workerResult->d_ptr->h_gProximityContactType, workerResult->d_gProximityContactType, sizeof(gProximityContactType) * workerResult->_maxResults, *(workerUnit->_stream));

#endif
    }
    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_ResultCopy");
}
