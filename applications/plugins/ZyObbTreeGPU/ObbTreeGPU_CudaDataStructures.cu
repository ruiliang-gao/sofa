#include "ObbTreeGPU_CudaDataStructures.h"

#include <cutil/cutil.h>

// Please use this as debug out
// usage: DBG("Some Integer: " << int_var);
#define DBG(x) std::cout << x << std::endl;

// Comment in to disable debug output
//#define DBG(x)

#ifdef USE_TRUNK_CUDA_THRUST
#include <thrust_v180/host_vector.h>
#endif

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
    //float3* h_pointData;
};

gProximityGPUTransform::gProximityGPUTransform()
{
    modelTranslation = NULL;
    modelOrientation = NULL;

    DBG("gProximityGPUTransform::gProximityGPUTransform(): gpu-allocate modelTranslation");
    GPUMALLOC((void**) &modelTranslation, sizeof(float3));
    DBG("gProximityGPUTransform::gProximityGPUTransform(): gpu-allocate modelOrientation");
    GPUMALLOC((void**) &modelOrientation, sizeof(Matrix3x3_d));
}

gProximityGPUTransform::~gProximityGPUTransform()
{
    DBG("gProximityGPUTransform::~gProximityGPUTransform(): gpu-free modelOrientation");
    GPUFREE(modelOrientation);
    DBG("gProximityGPUTransform::~gProximityGPUTransform(): gpu-free modelTranslation");
    GPUFREE(modelTranslation);
}

gProximityWorkerUnit::gProximityWorkerUnit(unsigned int collisionPairCapacity, unsigned int queueNTasks, unsigned int queueSizePerTaskGlobal):
    _collisionPairCapacity(collisionPairCapacity), _queueNTasks(queueNTasks), _queueSizePerTaskGlobal(queueSizePerTaskGlobal)
{
    _nCollidingPairs = 0;

    d_collisionPairIndex = NULL;
    d_collisionSync = NULL;
    d_nWorkQueueElements = NULL;
    d_collisionPairs = NULL;
    d_collisionLeafs = NULL;
    d_workQueues = NULL;
    d_workQueues2 = NULL;
    d_workQueueCounts = NULL;
    d_balanceSignal = NULL;
    // d_intersectingOBBs = NULL;

    d_modelPosition_1 = NULL;
    d_modelPosition_2 = NULL;
    d_modelOrientation_1 = NULL;
    d_modelOrientation_2 = NULL;

    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): Memory available before allocations: " << mem_free << " of " << mem_total);

    DBG(" _queueNTasks = " << _queueNTasks << ", queueNTasks = " << queueNTasks << ", _queueSizePerTaskGlobal = " << _queueSizePerTaskGlobal << ", queueSizePerTaskGlobal" << queueSizePerTaskGlobal);

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): alloc d_collisionPairs, size = " << sizeof(int2) * _collisionPairCapacity);
    GPUMALLOC((void**)&d_collisionPairs, sizeof(int2) * _collisionPairCapacity);

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): alloc d_collisionLeafs, size = " << sizeof(bool) * _collisionPairCapacity);
    GPUMALLOC((void**)&d_collisionLeafs, sizeof(bool) * _collisionPairCapacity);

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): alloc d_collisionPairIndex, size = " << sizeof(int));
    GPUMALLOC((void **)&d_collisionPairIndex, sizeof(int));

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): alloc d_nWorkQueueElements, size = " << sizeof(int));
    GPUMALLOC((void **)&d_nWorkQueueElements, sizeof(int));

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): alloc d_collisionSync, size = " << sizeof(int));
    GPUMALLOC((void **)&d_collisionSync, sizeof(int));

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): alloc d_workQueues, size = " << sizeof(int2) * _queueNTasks * _queueSizePerTaskGlobal);
    GPUMALLOC((void **)&d_workQueues, sizeof(int2) * _queueNTasks * _queueSizePerTaskGlobal);

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): alloc d_workQueues2, size = " << sizeof(int2) * _queueNTasks * _queueSizePerTaskGlobal);
    GPUMALLOC((void **)&d_workQueues2, sizeof(int2) * _queueNTasks * _queueSizePerTaskGlobal);

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): alloc d_workQueueCounts, size = " << sizeof(int) * _queueNTasks * _queueSizePerTaskGlobal);
    GPUMALLOC((void **)&d_workQueueCounts, sizeof(int) * _queueNTasks * _queueSizePerTaskGlobal);

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): alloc d_balanceSignal, size = " << sizeof(int));
    GPUMALLOC((void**)&d_balanceSignal, sizeof(int));

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): alloc d_obbOutputIndex, size = " << sizeof(int));
    GPUMALLOC((void **)&d_obbOutputIndex, sizeof(int));

    GPUMALLOC((void **)& d_modelPosition_1, sizeof(float3));
    GPUMALLOC((void **)& d_modelOrientation_1, sizeof(Matrix3x3_d));

    GPUMALLOC((void **)& d_modelPosition_2, sizeof(float3));
    GPUMALLOC((void **)& d_modelOrientation_2, sizeof(Matrix3x3_d));

    cudaMemGetInfo(&mem_free, &mem_total);

    DBG("gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): Memory available after allocations: " << mem_free << " of " << mem_total);
}

gProximityWorkerUnit::~gProximityWorkerUnit()
{
    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);

    DBG("~gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): Memory available before freeing: " << mem_free << " of " << mem_total);

    GPUFREE(d_workQueueCounts);
    GPUFREE(d_workQueues);
    GPUFREE(d_workQueues2);
    GPUFREE(d_collisionPairs);
    GPUFREE(d_collisionLeafs);
    // GPUFREE(d_intersectingOBBs);
    GPUFREE(d_collisionPairIndex);
    GPUFREE(d_obbOutputIndex);

    GPUFREE(d_nWorkQueueElements);
    GPUFREE(d_collisionSync);
    GPUFREE(d_balanceSignal);

    GPUFREE(d_modelOrientation_1);
    GPUFREE(d_modelOrientation_2);
    GPUFREE(d_modelPosition_1);
    GPUFREE(d_modelPosition_2);

    cudaMemGetInfo(&mem_free, &mem_total);
    DBG("~gProximityWorkerUnit(" << _collisionPairCapacity << "," << _queueNTasks << "," << _queueSizePerTaskGlobal << "): Memory available after freeing: " << mem_free << " of " << mem_total);
}

gProximityWorkerResult::gProximityWorkerResult(unsigned int maxResults): _maxResults(maxResults), _numResults(0)
{
    d_valid = NULL;
    d_contactId = NULL;
    d_distance = NULL;
    d_elems = NULL;
    d_point0 = NULL;
    d_point1 = NULL;
    d_gProximityContactType = NULL;

    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);

    DBG("gProximityWorkerResult(" << _maxResults << "): Memory available before allocations: " << mem_free << " of " << mem_total);

    d_ptr = new gProximityWorkerResultPrivate;

    h_contacts = new gProximityDetectionOutput;

#ifdef ALLOCATE_RESULT_MEMORY_ON_GPU
    DBG("gProximityWorkerResult(" << _maxResults << "): allocate d_valid, size = " << sizeof(bool) * maxResults * CollisionTestElementsSize);
    GPUMALLOC((void **)&(d_valid), sizeof(bool) * maxResults * CollisionTestElementsSize);

    DBG("gProximityWorkerResult(" << _maxResults << "): allocate d_contactId, size = " << sizeof(int) * maxResults * CollisionTestElementsSize);
    GPUMALLOC((void **)&(d_contactId), sizeof(int) * maxResults * CollisionTestElementsSize);

    DBG("gProximityWorkerResult(" << _maxResults << "): allocate d_distance, size = " << sizeof(double) * maxResults * CollisionTestElementsSize);
    GPUMALLOC((void **)&(d_distance), sizeof(double) * maxResults * CollisionTestElementsSize);

    DBG("gProximityWorkerResult(" << _maxResults << "): allocate d_elems, size = " << sizeof(int4) * maxResults * CollisionTestElementsSize);
    GPUMALLOC((void **)&(d_elems), sizeof(int4) * maxResults * CollisionTestElementsSize);

    DBG("gProximityWorkerResult(" << _maxResults << "): allocate d_normal, size = " << sizeof(float3) * maxResults * CollisionTestElementsSize);
    GPUMALLOC((void **)&(d_normal), sizeof(float3) * maxResults * CollisionTestElementsSize);

    DBG("gProximityWorkerResult(" << _maxResults << "): allocate d_point0, size = " << sizeof(float3) * maxResults * CollisionTestElementsSize);
    GPUMALLOC((void **)&(d_point0), sizeof(float3) * maxResults * CollisionTestElementsSize);

    DBG("gProximityWorkerResult(" << _maxResults << "): allocate d_point1, size = " << sizeof(float3) * maxResults * CollisionTestElementsSize);
    GPUMALLOC((void **)&(d_point1), sizeof(float3) * maxResults * CollisionTestElementsSize);

    DBG("gProximityWorkerResult(" << _maxResults << "): allocate d_gProximityContactType, size = " << sizeof(gProximityContactType) * maxResults * CollisionTestElementsSize);
    GPUMALLOC((void **)&(d_gProximityContactType), sizeof(gProximityContactType) * maxResults * CollisionTestElementsSize);
#else
    CUDA_SAFE_CALL( cudaHostAlloc( (void**) &(d_ptr->h_valid), sizeof(bool) * _maxResults * CollisionTestElementsSize, cudaHostAllocMapped ) );
    CUDA_SAFE_CALL( cudaHostGetDevicePointer( &(h_contacts->valid), d_ptr->h_valid, 0 ) );

    CUDA_SAFE_CALL( cudaHostAlloc( (void**) &(d_ptr->h_contactId), sizeof(int) * _maxResults * CollisionTestElementsSize, cudaHostAllocMapped ) );
    CUDA_SAFE_CALL( cudaHostGetDevicePointer( &(h_contacts->contactId), d_ptr->h_contactId, 0 ) );

    CUDA_SAFE_CALL( cudaHostAlloc( (void**) &(d_ptr->h_distance), sizeof(double) * _maxResults * CollisionTestElementsSize, cudaHostAllocMapped ) );
    CUDA_SAFE_CALL( cudaHostGetDevicePointer( &(h_contacts->distance), d_ptr->h_distance, 0 ) );

    CUDA_SAFE_CALL( cudaHostAlloc( (void**) &(d_ptr->h_elems), sizeof(int4) * _maxResults * CollisionTestElementsSize, cudaHostAllocMapped ) );
    CUDA_SAFE_CALL( cudaHostGetDevicePointer( &(h_contacts->elems), d_ptr->h_elems, 0 ) );

    CUDA_SAFE_CALL( cudaHostAlloc( (void**) &(d_ptr->h_normal), sizeof(float3) * _maxResults * CollisionTestElementsSize, cudaHostAllocMapped ) );
    CUDA_SAFE_CALL( cudaHostGetDevicePointer( &(h_contacts->normal), d_ptr->h_normal, 0 ) );

    CUDA_SAFE_CALL( cudaHostAlloc( (void**) &(d_ptr->h_point0), sizeof(float3) * _maxResults * CollisionTestElementsSize, cudaHostAllocMapped ) );
    CUDA_SAFE_CALL( cudaHostGetDevicePointer( &(h_contacts->point0), d_ptr->h_point0, 0 ) );

    CUDA_SAFE_CALL( cudaHostAlloc( (void**) &(d_ptr->h_point1), sizeof(float3) * _maxResults * CollisionTestElementsSize, cudaHostAllocMapped ) );
    CUDA_SAFE_CALL( cudaHostGetDevicePointer( &(h_contacts->point1), d_ptr->h_point1, 0 ) );

    CUDA_SAFE_CALL( cudaHostAlloc( (void**) &(d_ptr->h_gProximityContactType), sizeof(gProximityContactType) * _maxResults * CollisionTestElementsSize, cudaHostAllocMapped ) );
    CUDA_SAFE_CALL( cudaHostGetDevicePointer( &(h_contacts->contactType), d_ptr->h_gProximityContactType, 0 ) );
#endif

    DBG("gProximityWorkerResult(" << _maxResults << "): allocate d_swapped, size = " << sizeof(bool) * maxResults * CollisionTestElementsSize);
    GPUMALLOC((void **)&(d_swapped), sizeof(bool) * maxResults * CollisionTestElementsSize);

#ifdef ALLOCATE_RESULT_MEMORY_ON_GPU
    h_contacts->valid = d_valid;
    h_contacts->contactId = d_contactId;
    h_contacts->distance = d_distance;
    h_contacts->elems = d_elems;
    h_contacts->normal = d_normal;
    h_contacts->point0 = d_point0;
    h_contacts->point1 = d_point1;
    h_contacts->contactType = d_gProximityContactType;
#endif

    h_contacts->swapped = d_swapped;

    GPUMALLOC((void**)& d_contacts, sizeof(gProximityDetectionOutput));
    TOGPU(d_contacts, h_contacts, sizeof(gProximityDetectionOutput));

    GPUMALLOC((void**)& d_outputIndex, sizeof(unsigned int));
    GPUMEMSET(d_outputIndex, 0, sizeof(unsigned int));

    cudaMemGetInfo(&mem_free, &mem_total);
    DBG("gProximityWorkerResult(" << _maxResults << "): Memory available after allocations: " << mem_free << " of " << mem_total);

    _blocked = false;
    _resultIndex = 0;
    _resultBin = 0;

    h_outputIndex = 0;

#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
    d_ptr->h_valid.resize(_maxResults);
    d_ptr->h_contactId.resize(_maxResults);
    d_ptr->h_distance.resize(_maxResults);
    d_ptr->h_elems.resize(_maxResults);
    d_ptr->h_point0.resize(_maxResults);
    d_ptr->h_point1.resize(_maxResults);
    d_ptr->h_normal.resize(_maxResults);
    d_ptr->h_gProximityContactType.resize(_maxResults);
#else

#ifdef ALLOCATE_RESULT_MEMORY_ON_GPU
    d_ptr->h_valid = new bool[_maxResults];
    d_ptr->h_contactId = new int[_maxResults];
    d_ptr->h_distance = new double[_maxResults];
    d_ptr->h_elems = new int4[_maxResults];
    d_ptr->h_normal = new float3[_maxResults];
    d_ptr->h_point0 = new float3[_maxResults];
    d_ptr->h_point1 = new float3[_maxResults];
    d_ptr->h_gProximityContactType = new gProximityContactType[_maxResults];
#endif

#ifdef ALLOCATE_RESULT_MEMORY_ON_GPU
    cudaHostRegister(d_ptr->h_valid, sizeof(bool) * _maxResults, cudaHostRegisterMapped);
    cudaHostRegister(d_ptr->h_contactId, sizeof(int) * _maxResults, cudaHostRegisterMapped);
    cudaHostRegister(d_ptr->h_distance, sizeof(double) * _maxResults, cudaHostRegisterMapped);
    cudaHostRegister(d_ptr->h_elems, sizeof(int4) * _maxResults, cudaHostRegisterMapped);
    cudaHostRegister(d_ptr->h_normal, sizeof(float3) * _maxResults, cudaHostRegisterMapped);
    cudaHostRegister(d_ptr->h_point0, sizeof(float3) * _maxResults, cudaHostRegisterMapped);
    cudaHostRegister(d_ptr->h_point1, sizeof(float3) * _maxResults, cudaHostRegisterMapped);
    cudaHostRegister(d_ptr->h_gProximityContactType, sizeof(gProximityContactType) * _maxResults, cudaHostRegisterMapped);
#endif

#endif
}

gProximityWorkerResult::~gProximityWorkerResult()
{
    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);

    DBG("~gProximityWorkerResult(" << _maxResults << "): Memory available before freeing: " << mem_free << " of " << mem_total);
#ifdef ALLOCATE_RESULT_MEMORY_ON_GPU
    cudaHostUnregister(d_ptr->h_valid);
    delete[] d_ptr->h_valid;

    cudaHostUnregister(d_ptr->h_contactId);
    delete[] d_ptr->h_contactId;

    cudaHostUnregister(d_ptr->h_distance);
    delete[] d_ptr->h_distance;

    cudaHostUnregister(d_ptr->h_elems);
    delete[] d_ptr->h_elems;

    cudaHostUnregister(d_ptr->h_normal);
    delete[] d_ptr->h_normal;

    cudaHostUnregister(d_ptr->h_point0);
    delete[] d_ptr->h_point0;

    cudaHostUnregister(d_ptr->h_point1);
    delete[] d_ptr->h_point1;

    cudaHostUnregister(d_ptr->h_gProximityContactType);
    delete[] d_ptr->h_gProximityContactType;

#endif

#ifdef ALLOCATE_RESULT_MEMORY_ON_GPU
    GPUFREE(d_outputIndex);
    GPUFREE(d_valid);
    GPUFREE(d_contactId);
    GPUFREE(d_distance);
    GPUFREE(d_elems);
    GPUFREE(d_normal);
    GPUFREE(d_point0);
    GPUFREE(d_point1);
    GPUFREE(d_gProximityContactType);
#else
    CUDA_SAFE_CALL(cudaFreeHost(d_ptr->h_valid));
    CUDA_SAFE_CALL(cudaFreeHost(d_ptr->h_contactId));
    CUDA_SAFE_CALL(cudaFreeHost(d_ptr->h_distance));
    CUDA_SAFE_CALL(cudaFreeHost(d_ptr->h_elems));
    CUDA_SAFE_CALL(cudaFreeHost(d_ptr->h_point0));
    CUDA_SAFE_CALL(cudaFreeHost(d_ptr->h_point1));
    CUDA_SAFE_CALL(cudaFreeHost(d_ptr->h_normal));
    CUDA_SAFE_CALL(cudaFreeHost(d_ptr->h_gProximityContactType));

    GPUFREE(d_outputIndex);

    //CUDA_SAFE_CALL(cudaFreeHost(h_outputIndex));
#endif

    GPUFREE(d_swapped);

    GPUFREE(d_contacts);

    delete h_contacts;

    delete d_ptr;

    cudaMemGetInfo(&mem_free, &mem_total);
    DBG("~gProximityWorkerResult(" << _maxResults << "): Memory available after freeing: " << mem_free << " of " << mem_total);
}
