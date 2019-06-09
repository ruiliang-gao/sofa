#include "ObbTreeGPUCollisionDetection.h"

#include <PQP/src/MatVec.h>
#include <PQP/src/GetTime.h>
#include <PQP/include/PQP.h>
#include <PQP/include/BV.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/core/visual/DrawTool.h>
#include <sofa/core/visual/DrawToolGL.h>

#include <sofa/component/collision/DiscreteIntersection.h>
#include <sofa/core/collision/Contact.h>
#include <sofa/component/collision/BaseContactMapper.h>
#include <sofa/component/collision/BarycentricPenalityContact.h>

#include <sofa/component/container/MechanicalObject.h>

#include <GL/gl.h>

#include <boost/algorithm/string/iter_find.hpp>
#include <boost/algorithm/string/finder.hpp>

#include "ObbTreeGPUIntersection.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include <cutil/cutil.h>

#include "ObbTreeGPU_CudaDataStructures.h"
#include "ObbTreeGPUTriangleCollision_cuda.h"
#include "ObbTreeGPUCollisionModel_cuda.h"

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

using namespace sofa::component::collision;
using namespace sofa;
using namespace sofa::component::container;

SOFA_DECL_CLASS(ObbTreeGPUCollisionDetection)

int ObbTreeGPUCollisionDetectionClass = sofa::core::RegisterObject("Collision detection using GPU-based OBB-trees, with fall back to brute-force pair tests")
.add< ObbTreeGPUCollisionDetection >();

//sofa::helper::Creator<sofa::core::collision::Contact::Factory, sofa::component::collision::BarycentricPenalityContact<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types>, Vec3fTypes> > ObbTreeGPUContactClass("default", true);


ObbTreeGPUCollisionDetection::ObbTreeGPUCollisionDetection():
    BruteForceDetection(),
    m_intersection(NULL),
    _totalResultBinCount(0),
    _updateGPUVertexPositions(initData(&_updateGPUVertexPositions, false, "updateGPUVertexPositions", "Update GPU vertex arrays in beginNarrowPhase", true, false)),
    _useStreamedCollisionQueries(initData(&_useStreamedCollisionQueries, false, "useStreamedCollisionQueries", "Use CUDA streams to further parallelize collision queries", true, false)),
    _useDynamicWorkerScheme(initData(&_useDynamicWorkerScheme, false, "useDynamicWorkerScheme", "Use dynamically sized/allocated worker result bins", true, false)),
    _numStreamedWorkerUnits(initData(&_numStreamedWorkerUnits, (unsigned int) 4, "numStreamedWorkerUnits", "Use CUDA streams to further parallelize collision queries", true, false)),
    _numStreamedWorkerResultBins(initData(&_numStreamedWorkerResultBins, (unsigned int) 4, "numStreamedWorkerResultBins", "Use CUDA streams to further parallelize collision queries", true, false)),
    _streamedWorkerResultMinSize(initData(&_streamedWorkerResultMinSize, (unsigned int) 2048, "streamedWorkerResultMinSize", "Use CUDA streams to further parallelize collision queries", true, false)),
    _streamedWorkerResultMaxSize(initData(&_streamedWorkerResultMaxSize, (unsigned int) 16384, "streamedWorkerResultMaxSize", "Use CUDA streams to further parallelize collision queries", true, false))
{

}

ObbTreeGPUCollisionDetection::~ObbTreeGPUCollisionDetection()
{
#ifdef OBBTREE_GPU_STREAMED_COLLISION_QUERIES
    if (_useDynamicWorkerScheme.getValue())
    {
        CUDA_SAFE_CALL(cudaStreamDestroy(_transformStream));
        CUDA_SAFE_CALL(cudaStreamDestroy(_memoryStream));

        for (unsigned int k = 0; k < _obbTreeGPUModels.size(); k++)
        {
            delete _gpuTransforms[k];
        }
        delete [] _gpuTransforms;

        /*for (std::map<std::string, gProximityGPUTransform*>::iterator it = _modelTransforms.begin(); it != _modelTransforms.end(); it++)
        {
            delete it->second;
        }
        _modelTransforms.clear();*/

        std::cout << "ObbTreeGPUCollisionDetection::~ObbTreeGPUCollisionDetection(" << this->getName() << ")" << std::endl;
        std::cout << " destroy tri-test streams" << std::endl;
        for (unsigned int k = 0; k < _numStreamedWorkerResultBins.getValue(); k++)
        {
            CUDA_SAFE_CALL(cudaStreamDestroy(_triTestStreams[k]));
        }
        _triTestStreams.clear();

        for (unsigned int k = 0; k < _totalResultBinCount; k++)
        {
            CUDA_SAFE_CALL(cudaEventDestroy(_triTestEvents[k]));
        }
        _triTestEvents.clear();


        CUDA_SAFE_CALL(cudaEventDestroy(_triTestStartEvent));
        CUDA_SAFE_CALL(cudaEventDestroy(_triTestEndEvent));

        std::cout << " de-allocate streamed worker units: " << _streamedWorkerUnits.size() << std::endl;
        for (unsigned int k = 0; k < _streamedWorkerUnits.size(); k++)
        {
            gProximityWorkerUnit* workerUnit = _streamedWorkerUnits[k];
            if (workerUnit != NULL)
            {
                CUDA_SAFE_CALL(cudaStreamDestroy(_workerStreams[k]));
                CUDA_SAFE_CALL(cudaEventDestroy(_workerEvents[k]));
                delete workerUnit;
            }
        }
        _streamedWorkerUnits.clear();

        CUDA_SAFE_CALL(cudaEventDestroy(_workerStartEvent));
        CUDA_SAFE_CALL(cudaEventDestroy(_workerEndEvent));
        CUDA_SAFE_CALL(cudaEventDestroy(_workerBalanceEvent));

        std::cout << " de-allocate streamed worker results: " << _streamedWorkerResults.size() << std::endl;
        for (std::map<unsigned int, std::vector<gProximityWorkerResult*> >::iterator it = _streamedWorkerResults.begin(); it != _streamedWorkerResults.end(); it++)
        {
            std::vector<gProximityWorkerResult*>& workerResults = it->second;
            for (unsigned int k = 0; k < workerResults.size(); k++)
            {
                delete workerResults[k];
            }
            workerResults.clear();
        }

        CUDA_SAFE_CALL(cudaFreeHost(_outputIndices));

        _streamedWorkerResults.clear();
    }
#endif
}

void ObbTreeGPUCollisionDetection::reset()
{

    for (std::map<std::string, std::pair<GPUVertex*, GPUVertex*> >::iterator it = m_transformedVertices.begin(); it != m_transformedVertices.end(); it++)
    {
        if (it->second.first)
            delete[] it->second.first;

        if (it->second.second)
            delete[] it->second.second;
    }

    m_transformedVertices.clear();
}

void ObbTreeGPUCollisionDetection::init()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d. Supports concurrent kernel execution = %d; asyncEngineCount = %d\n",
               device, deviceProp.major, deviceProp.minor, deviceProp.concurrentKernels, deviceProp.asyncEngineCount);
    }

    BruteForceDetection::init();

    std::vector<sofa::component::collision::OBBTreeGPUDiscreteIntersection* > moV;
    sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::collision::OBBTreeGPUDiscreteIntersection, std::vector<sofa::component::collision::OBBTreeGPUDiscreteIntersection* > > cb(&moV);

    getContext()->getObjects(TClassInfo<sofa::component::collision::OBBTreeGPUDiscreteIntersection>::get(), cb, TagSet(), BaseContext::SearchRoot);

    std::cout << "ObbTreeGPUCollisionDetection::init(): Searched for instances of OBBTreeGPUDiscreteIntersection; found = " << moV.size() << std::endl;
    if (moV.size() == 1)
    {
        //std::cout << " Using: " << moV.begin()->getName() << " of type " << moV.begin()->getTypeName() << std::endl;
        m_intersection = moV[0];
    }

#ifdef OBBTREE_GPU_STREAMED_COLLISION_QUERIES
    m_numOBBTreePairsTested = 0;
    for (unsigned short k = 0; k < 4; k++)
    {
        m_workerResultOccupied_64[k] = false;
        m_workerResultOccupied_128[k] = false;
        m_workerResultOccupied_256[k] = false;

        m_workerUnitOccupied[k] = false;
    }
#endif
}

void ObbTreeGPUCollisionDetection::reinit()
{
    BruteForceDetection::reinit();
    m_obbModels.clear();
    m_pqpModels.clear();

    for (std::map<std::string, std::pair<GPUVertex*, GPUVertex*> >::iterator it = m_transformedVertices.begin(); it != m_transformedVertices.end(); it++)
    {
        if (it->second.first)
            delete[] it->second.first;

        if (it->second.second)
            delete[] it->second.second;
    }

    m_transformedVertices.clear();
#ifdef OBBTREE_GPU_STREAMED_COLLISION_QUERIES
    m_numOBBTreePairsTested = 0;
    for (unsigned short k = 0; k < 4; k++)
    {
        m_workerResultOccupied_64[k] = false;
        m_workerResultOccupied_128[k] = false;
        m_workerResultOccupied_256[k] = false;

        m_workerUnitOccupied[k] = false;

    }
#endif
}

void ObbTreeGPUCollisionDetection::bwdInit()
{
#ifdef OBBTREE_GPU_STREAMED_COLLISION_QUERIES
    if (_useDynamicWorkerScheme.getValue())
    {

        CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&_transformStream, cudaStreamNonBlocking));
        CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&_memoryStream, cudaStreamNonBlocking));

        std::vector<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>* > obbTreeGPUCollisionModels;
        sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>, std::vector<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>* > > obbTreeGPUCollisionModels_cb(&obbTreeGPUCollisionModels);
        getContext()->getObjects(TClassInfo<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types> >::get(), obbTreeGPUCollisionModels_cb, TagSet(), BaseContext::SearchRoot);

        if (obbTreeGPUCollisionModels.size() > 0)
        {
            _gpuTransforms = new gProximityGPUTransform*[obbTreeGPUCollisionModels.size()];

            std::cout << "ObbTreeGPUCollisionDetection::init(): Searched for instances of OBBTreeGPUCollisionModel; found = " << obbTreeGPUCollisionModels.size() << std::endl;
            for (unsigned int k = 0; k < obbTreeGPUCollisionModels.size(); k++)
            {
                _gpuTransforms[k] = new gProximityGPUTransform();

                std::cout << " * " << k << ": " << obbTreeGPUCollisionModels[k]->getName() << std::endl;
                std::cout << "   store in model map" << std::endl;
                //_modelTransforms.insert(std::make_pair(obbTreeGPUCollisionModels[k]->getName(), new gProximityGPUTransform()));
                _obbTreeGPUModels.insert(std::make_pair(obbTreeGPUCollisionModels[k]->getName(), obbTreeGPUCollisionModels[k]));

                Vector3 modelPosition = obbTreeGPUCollisionModels[k]->getCachedModelPosition();
                Matrix3 modelOrientation = obbTreeGPUCollisionModels[k]->getCachedModelOrientation();

                float3 h_modelPosition = make_float3(modelPosition.x(), modelPosition.y(), modelPosition.z());
                Matrix3x3_d h_modelOrientation;

                h_modelOrientation.m_row[0].x = modelOrientation(0,0);
                h_modelOrientation.m_row[0].y = modelOrientation(0,1);
                h_modelOrientation.m_row[0].z = modelOrientation(0,2);
                h_modelOrientation.m_row[1].x = modelOrientation(1,0);
                h_modelOrientation.m_row[1].y = modelOrientation(1,1);
                h_modelOrientation.m_row[1].z = modelOrientation(1,2);
                h_modelOrientation.m_row[2].x = modelOrientation(2,0);
                h_modelOrientation.m_row[2].y = modelOrientation(2,1);
                h_modelOrientation.m_row[2].z = modelOrientation(2,2);

                std::cout << "     model " << obbTreeGPUCollisionModels[k]->getName() << " position    = " << h_modelPosition.x << "," << h_modelPosition.y << "," << h_modelPosition.z << std::endl;
                std::cout << "                                                             orientation = [" << h_modelOrientation.m_row[0].x << "," << h_modelOrientation.m_row[0].y << "," << h_modelOrientation.m_row[0].z << "],[" << h_modelOrientation.m_row[1].x << "," << h_modelOrientation.m_row[1].y << "," << h_modelOrientation.m_row[1].z << "],[" << h_modelOrientation.m_row[2].x << "," << h_modelOrientation.m_row[2].y << "," << h_modelOrientation.m_row[2].z << "]"<< std::endl;

                std::cout << "   Initial position upload to GPU memory" << std::endl;

                gProximityGPUTransform* gpTransform = _gpuTransforms[k];

                TOGPU(gpTransform->modelTranslation, &h_modelPosition, sizeof(float3));
                TOGPU(gpTransform->modelOrientation, &h_modelOrientation, sizeof(Matrix3x3_d));

                float3 h_modelPosition_Reread;
                Matrix3x3_d h_modelOrientation_Reread;

                FROMGPU(&h_modelPosition_Reread, gpTransform->modelTranslation, sizeof(float3));
                std::cout << "   position re-read_2: " << h_modelPosition_Reread.x << "," << h_modelPosition_Reread.y << "," << h_modelPosition_Reread.z << std::endl;

                FROMGPU(&h_modelOrientation_Reread, gpTransform->modelOrientation, sizeof(Matrix3x3_d));
                std::cout << "   orientation re-read_2: [" << h_modelOrientation_Reread.m_row[0].x << "," << h_modelOrientation_Reread.m_row[0].y << "," << h_modelOrientation_Reread.m_row[0].z << "],[" << h_modelOrientation_Reread.m_row[1].x << "," << h_modelOrientation_Reread.m_row[1].y << "," << h_modelOrientation_Reread.m_row[1].z << "],[" << h_modelOrientation_Reread.m_row[2].x << "," << h_modelOrientation_Reread.m_row[2].y << "," << h_modelOrientation_Reread.m_row[2].z << "]"<< std::endl;

                _gpuTransformIndices.insert(std::make_pair(obbTreeGPUCollisionModels[k]->getName(), k));
            }
        }

        std::cout << "ObbTreeGPUCollisionDetection::bwdInit(" << this->getName() << ")" << std::endl;

        if (_useDynamicWorkerScheme.getValue())
        {
            std::cout << " allocate streamed worker units: " << _numStreamedWorkerUnits.getValue() << std::endl;
            _workerStreams.resize(_numStreamedWorkerUnits.getValue());
            _workerEvents.resize(_numStreamedWorkerUnits.getValue());

            for (unsigned int k = 0; k < _numStreamedWorkerUnits.getValue(); k++)
            {
                gProximityWorkerUnit* workerUnit = new gProximityWorkerUnit();
                workerUnit->_workerUnitIndex = k;
                CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&(_workerStreams[k]), cudaStreamNonBlocking));
                workerUnit->_stream = &(_workerStreams[k]);
                //workerUnit->_event = &(_workerEvents[k]);
                _streamedWorkerUnits.push_back(workerUnit);

                CUDA_SAFE_CALL(cudaEventCreateWithFlags(&(_workerEvents[k]), cudaEventDisableTiming));
            }

            /*for (unsigned int k = 0; k < _numStreamedWorkerUnits.getValue(); k++)
            {
                _streamedWorkerUnits[k]->_syncStream = &(_workerStreams[_workerStreams.size() - 1]);
            }*/

            CUDA_SAFE_CALL(cudaEventCreate(&_workerStartEvent));
            CUDA_SAFE_CALL(cudaEventCreate(&_workerEndEvent));
            CUDA_SAFE_CALL(cudaEventCreate(&_workerBalanceEvent));

            _triTestStreams.resize(_numStreamedWorkerResultBins.getValue());

            for (unsigned int k = 0; k < _numStreamedWorkerResultBins.getValue(); k++)
            {
                CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&(_triTestStreams[k]), cudaStreamNonBlocking));
            }

            CUDA_SAFE_CALL(cudaEventCreate(&_triTestStartEvent));
            CUDA_SAFE_CALL(cudaEventCreate(&_triTestEndEvent));

            std::cout << " allocate streamed worker results: " << _numStreamedWorkerResultBins.getValue() << std::endl;
            for (unsigned int k = 0; k < _numStreamedWorkerResultBins.getValue(); k++)
            {
                _streamedWorkerResults.insert(std::make_pair(k, std::vector<gProximityWorkerResult*>()));
            }

            std::cout << "ObbTreeGPUCollisionDetection(" << this->getName() << "): _numStreamedWorkerResultBins = "
                      << _numStreamedWorkerResultBins.getValue() << "; _streamedWorkerResultMinSize = " << _streamedWorkerResultMinSize.getValue()
                      << "; _streamedWorkerResultMaxSize = " << _streamedWorkerResultMaxSize.getValue() << std::endl;
            unsigned int minMaxRatio = _streamedWorkerResultMaxSize.getValue() / _streamedWorkerResultMinSize.getValue();
            unsigned int curBinSize = _streamedWorkerResultMinSize.getValue();


            _totalResultBinCount = 0;
            for (unsigned int k = 0; k < _numStreamedWorkerResultBins.getValue(); k++)
            {
                unsigned int numResultUnitsForBin = _numStreamedWorkerResultBins.getValue() * minMaxRatio;
                std::cout << "  bin level " << k << ": curBinSize = " << curBinSize << ", minMaxRatio = " << minMaxRatio << ", numResultUnitsForBin = " << numResultUnitsForBin << std::endl;
                for (unsigned int l = 0; l < numResultUnitsForBin; l++)
                {
                    gProximityWorkerResult* workerResult = new gProximityWorkerResult(curBinSize);
                    workerResult->_resultIndex = l;
                    workerResult->_resultBin = k;
                    workerResult->_outputIndexPosition = _totalResultBinCount;
                    _streamedWorkerResults[k].push_back(workerResult);
                    _totalResultBinCount++;
                }

                minMaxRatio /= 2;
                if (minMaxRatio == 0)
                    minMaxRatio = 1;

                curBinSize *= 2;
            }

            std::cout << "Allocate tri-test CUDA events: " << _totalResultBinCount << " events in total" << std::endl;
            _triTestEvents.resize(_totalResultBinCount);

            std::cout << "Allocate tri-test result output indices array" << std::endl;
            CUDA_SAFE_CALL( cudaHostAlloc( (void**) &_outputIndices, sizeof(unsigned int*) * _totalResultBinCount, cudaHostAllocMapped ) );

            for (unsigned int k = 0; k < _totalResultBinCount; k++)
            {
                CUDA_SAFE_CALL(cudaEventCreateWithFlags(&(_triTestEvents[k]), cudaEventDisableTiming));
            }
        }
    }
    m_numOBBTreePairsTested = 0;
#endif


    std::vector<ObbTreeGPULocalMinDistance* > lmdNodes;
    sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<ObbTreeGPULocalMinDistance, std::vector<ObbTreeGPULocalMinDistance* > > cb(&lmdNodes);

    getContext()->getObjects(TClassInfo<ObbTreeGPULocalMinDistance>::get(), cb, TagSet(), BaseContext::SearchRoot);

    std::cout << "ObbTreeGPUCollisionDetection::bwdInit(): ObbTreeGPULocalMinDistance instances found: " << lmdNodes.size() << std::endl;
    if (lmdNodes.size() > 0)
    {
        std::cout << " alarmDistance = " << lmdNodes.at(0)->getAlarmDistance() << ", contactDistance = " << lmdNodes.at(0)->getContactDistance() << std::endl;
        m_alarmDistance = lmdNodes.at(0)->getAlarmDistance();
        m_contactDistance = lmdNodes.at(0)->getContactDistance();
    }
    else
    {
        m_alarmDistance = 0.25f;
        m_contactDistance = 0.125f;
    }
}

//#define OBBTREEGPUCOLLISIONDETECTION_BEGINBROADPHASE_DEBUG
void ObbTreeGPUCollisionDetection::beginBroadPhase()
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_BEGINBROADPHASE_DEBUG
    sout << "=== ObbTreeGPUCollisionDetection::beginBroadPhase() ===" << sendl;
#endif
    m_intersectingOBBs.clear();
    m_intersectingTriangles.clear();

    //m_testedModelPairs_AB.clear();
    //m_testedModelPairs_BA.clear();

    m_testedModelPairs.clear();

    for (std::map<std::string, std::pair<GPUVertex*, GPUVertex*> >::iterator it = m_transformedVertices.begin(); it != m_transformedVertices.end(); it++)
    {
        if (it->second.first)
            delete[] it->second.first;

        if (it->second.second)
            delete[] it->second.second;
    }

    m_transformedVertices.clear();
#ifdef OBBTREE_GPU_STREAMED_COLLISION_QUERIES
    m_numOBBTreePairsTested = 0;
    for (unsigned short k = 0; k < 4; k++)
    {
        m_workerResultOccupied_64[k] = false;
        m_workerResultOccupied_128[k] = false;
        m_workerResultOccupied_256[k] = false;

        m_workerUnitOccupied[k] = false;
    }

    _narrowPhasePairs.clear();
#endif
    BruteForceDetection::beginBroadPhase();
}

//#define OBBTREEGPUCOLLISIONDETECTION_ENDBROADPHASE_DEBUG
void ObbTreeGPUCollisionDetection::endBroadPhase()
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_ENDBROADPHASE_DEBUG
    sout << "ObbTreeGPUCollisionDetection::endBroadPhase()" << sendl;
#endif
#ifdef OBBTREE_GPU_STREAMED_COLLISION_QUERIES
    m_numOBBTreePairsTested = 0;
    for (unsigned short k = 0; k < 4; k++)
    {
        m_workerResultOccupied_64[k] = false;
        m_workerResultOccupied_128[k] = false;
        m_workerResultOccupied_256[k] = false;

        m_workerUnitOccupied[k] = false;
    }
#endif
    BruteForceDetection::endBroadPhase();
}

#define OBBTREEGPU_COLLISION_DETECTION_BEGIN_NARROW_PHASE_DEBUG
#define OBBTREEGPU_COLLISION_DETECTION_BEGIN_NARROW_PHASE_DEBUG_VERBOSE
void ObbTreeGPUCollisionDetection::beginNarrowPhase()
{
    std::cout << "=== ObbTreeGPUCollisionDetection::beginNarrowPhase() ===" << std::endl;
    std::cout << std::dec;
    BruteForceDetection::beginNarrowPhase();

    if (_updateGPUVertexPositions.getValue())
    {
        for (std::map<std::string, ObbTreeGPUCollisionModel<Vec3Types>*>::iterator it = _obbTreeGPUModels.begin(); it != _obbTreeGPUModels.end(); it++)
        {
            ObbTreeGPUCollisionModel<Vec3Types>* cm = it->second;
            //gProximityGPUTransform* mt = it->second;

            Vector3 modelPosition = cm->getCachedModelPosition();
            Matrix3 modelOrientation = cm->getCachedModelOrientation();

            bool skipPositionUpdate = false;
            if (!cm->isSimulated() || !cm->isMoving())
            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_BEGIN_NARROW_PHASE_DEBUG
                std::cout << " EXCEPTION HANDLING FOR STATIC COLLISION MODEL " << cm->getName() << std::endl;
#endif
                {
                    MechanicalObject<Vec3Types>* mob = dynamic_cast<MechanicalObject<Vec3Types>*>(cm->getMechanicalState());
                    if (mob)
                    {
                        modelPosition = Vector3(mob->getPosition()[0][0], mob->getPosition()[0][1], mob->getPosition()[0][2]);
                        Quaternion modelRotation(mob->getPosition()[0][3], mob->getPosition()[0][4], mob->getPosition()[0][5], mob->getPosition()[0][6]);
                        modelRotation.toMatrix(modelOrientation);
#ifdef OBBTREEGPU_COLLISION_DETECTION_BEGIN_NARROW_PHASE_DEBUG
                        std::cout << " position = " << modelPosition << std::endl;
                        std::cout << " orientation = " << modelOrientation << std::endl;
#endif
                    }
                    else
                    {
                        skipPositionUpdate = true;
#ifdef OBBTREEGPU_COLLISION_DETECTION_BEGIN_NARROW_PHASE_DEBUG
                        std::cout << "WARNING: SKIP position update for model " << cm->getName() << " (no position query possible from its MechanicalState); please check its definition in the SOFA scene for correctness!" << std::endl;
#endif
                    }
                }
            }

            if (!skipPositionUpdate)
            {
                float3 h_modelPosition = make_float3(modelPosition.x(), modelPosition.y(), modelPosition.z());
                Matrix3x3_d h_modelOrientation;

                h_modelOrientation.m_row[0].x = modelOrientation(0,0);
                h_modelOrientation.m_row[0].y = modelOrientation(0,1);
                h_modelOrientation.m_row[0].z = modelOrientation(0,2);
                h_modelOrientation.m_row[1].x = modelOrientation(1,0);
                h_modelOrientation.m_row[1].y = modelOrientation(1,1);
                h_modelOrientation.m_row[1].z = modelOrientation(1,2);
                h_modelOrientation.m_row[2].x = modelOrientation(2,0);
                h_modelOrientation.m_row[2].y = modelOrientation(2,1);
                h_modelOrientation.m_row[2].z = modelOrientation(2,2);

                TOGPU_ASYNC(_gpuTransforms[_gpuTransformIndices[cm->getName()]]->modelTranslation, &h_modelPosition, sizeof(float3), _transformStream);
                TOGPU_ASYNC(_gpuTransforms[_gpuTransformIndices[cm->getName()]]->modelOrientation, &h_modelOrientation, sizeof(Matrix3x3_d), _transformStream);

#ifdef OBBTREEGPU_COLLISION_DETECTION_BEGIN_NARROW_PHASE_DEBUG_VERBOSE
                std::cout << " * model " << cm->getName() << " position    = " << h_modelPosition.x << "," << h_modelPosition.y << "," << h_modelPosition.z << std::endl;
                std::cout << "                                 orientation = [" << h_modelOrientation.m_row[0].x << "," << h_modelOrientation.m_row[0].y << "," << h_modelOrientation.m_row[0].z << "],[" << h_modelOrientation.m_row[1].x << "," << h_modelOrientation.m_row[1].y << "," << h_modelOrientation.m_row[1].z << "],[" << h_modelOrientation.m_row[2].x << "," << h_modelOrientation.m_row[2].y << "," << h_modelOrientation.m_row[2].z << "]"<< std::endl;

                float3 h_modelPosition_Reread;
                Matrix3x3_d h_modelOrientation_Reread;

                FROMGPU(&h_modelPosition_Reread, _gpuTransforms[_gpuTransformIndices[cm->getName()]]->modelTranslation, sizeof(float3));
                std::cout << "   position re-read_2: " << h_modelPosition_Reread.x << "," << h_modelPosition_Reread.y << "," << h_modelPosition_Reread.z << std::endl;

                FROMGPU(&h_modelOrientation_Reread, _gpuTransforms[_gpuTransformIndices[cm->getName()]]->modelOrientation, sizeof(Matrix3x3_d));
                std::cout << "   orientation re-read_2: [" << h_modelOrientation_Reread.m_row[0].x << "," << h_modelOrientation_Reread.m_row[0].y << "," << h_modelOrientation_Reread.m_row[0].z << "],[" << h_modelOrientation_Reread.m_row[1].x << "," << h_modelOrientation_Reread.m_row[1].y << "," << h_modelOrientation_Reread.m_row[1].z << "],[" << h_modelOrientation_Reread.m_row[2].x << "," << h_modelOrientation_Reread.m_row[2].y << "," << h_modelOrientation_Reread.m_row[2].z << "]"<< std::endl;
#endif


                updateInternalGeometry_cuda_streamed(cm->getModelInstance(), (GPUVertex*) cm->getTransformedVerticesPtr(), _gpuTransforms[_gpuTransformIndices[cm->getName()]], _transformStream, cm->hasModelPositionChanged());
            }
        }
        cudaStreamSynchronize(_transformStream);
    }

    _narrowPhasePairs.clear();
}

#define OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
#define OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
#define OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DO_TRIANGLE_TESTS
void ObbTreeGPUCollisionDetection::endNarrowPhase()
{
    std::cout << std::dec;
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
    std::cout << "=== ObbTreeGPUCollisionDetection::endNarrowPhase(" << this->getName() << ") ===" << std::endl;
    std::cout << " OBB tree pairs to check = " << _narrowPhasePairs.size() << std::endl;
    for (int k = 0; k < _narrowPhasePairs.size(); k++)
    {
        ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = _narrowPhasePairs[k].first._obbCollisionModel;
        ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = _narrowPhasePairs[k].second._obbCollisionModel;
        std::cout << "  - check: " << obbModel1->getName() << " - " << obbModel2->getName() << std::endl;
    }

#endif

    if (_useDynamicWorkerScheme.getValue())
    {
        int numThreads = _numStreamedWorkerUnits.getValue();

        unsigned int numIterations;
        if (_narrowPhasePairs.size() <= _numStreamedWorkerUnits.getValue())
            numIterations = 1;
        else
            numIterations = (_narrowPhasePairs.size() / _numStreamedWorkerUnits.getValue()) + 1;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
        std::cout << "ITERATIONS FOR COLLISION CHECKS: " << numIterations << " _narrowPhasePairs.size() = " << _narrowPhasePairs.size() << "; _numStreamedWorkerUnits.getValue() = " << _numStreamedWorkerUnits.getValue() << std::endl;
#endif
        std::map<unsigned int, int> workerUnitToBinMapping;

        for (unsigned int k = 0; k < _narrowPhasePairs.size(); k++)
            workerUnitToBinMapping.insert(std::make_pair(k, -1));

        for (unsigned int i = 0; i < numIterations; i++)
        {
            std::vector<int> intersectingTriPairCount;
            intersectingTriPairCount.resize(numThreads);

            bool activeTestsRemaining = false;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
            std::cout << " iteration " << i << std::endl;
#endif

            std::vector<OBBContainer*> models1, models2;
            std::vector<gProximityGPUTransform*> transforms1, transforms2;
            std::vector<gProximityWorkerUnit*> workerUnits;

            //#pragma omp parallel for num_threads(numThreads)

            scheduleBVHTraversals(_narrowPhasePairs, i, numThreads, models1, models2, transforms1, transforms2, workerUnits);


            if (workerUnits.size() > 0)
            {
                std::vector<int> intersectingObbs;
                int* workQueueCounts = new int[QUEUE_NTASKS];
                ObbTreeGPU_BVH_Traverse_Streamed_Batch(models1, models2, transforms1, transforms2,
                                                       workerUnits, _workerStreams, m_alarmDistance, m_contactDistance,
                                                       intersectingObbs, workQueueCounts, _memoryStream,
                                                       _workerStartEvent, _workerEndEvent, _workerBalanceEvent);

                delete[] workQueueCounts;

                for (unsigned int k = 0; k < workerUnits.size(); k++)
                {
                    intersectingTriPairCount[k] = intersectingObbs[k];

                    if (intersectingTriPairCount[k] > 0)
                        activeTestsRemaining = true;

                    workerUnitToBinMapping[k] = (k % _numStreamedWorkerUnits.getValue()); //workerUnits[k]->_workerUnitIndex;
                }
            }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DO_TRIANGLE_TESTS
            /// TODO AB HIER Multi-Threading! 1 Worker-Unit + passendes Set aus Results je Thread-Task.
            if (activeTestsRemaining)
            {   
                std::map<unsigned int, unsigned int> requiredSizesPerPairCheck;

                std::map<unsigned int, bool> satisfiablePairChecks;
                std::map<unsigned int, std::pair<OBBModelContainer, OBBModelContainer> > markedPairChecks;
                std::map<unsigned int, int> freeBinsPerBinLevel;

                std::map<unsigned int, std::multimap<unsigned int, unsigned int> > claimedResultBinsPerPairCheck;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
                std::cout << "==> intersecting tri-pairs (" << intersectingTriPairCount.size() << " slots): ";

                for (unsigned int m = 0; m < intersectingTriPairCount.size(); m++)
                {
                    std::cout << m << " = " << intersectingTriPairCount[m] << ";";
                }
                std::cout << std::endl;
#endif

                for (unsigned int k = 0; k < _numStreamedWorkerResultBins.getValue(); k++)
                {
                    freeBinsPerBinLevel[k] = _streamedWorkerResults[k].size();
                    claimedResultBinsPerPairCheck.insert(std::make_pair(k, std::multimap<unsigned int, unsigned int>()));
                }

                for (unsigned int m = 0; m < intersectingTriPairCount.size(); m++)
                {
                    bool triPairCheckAccepted = false;

                    unsigned int triPairCheckIndex = (i * _numStreamedWorkerUnits.getValue()) + m;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
                    std::cout << "  check intersection " << m << " of " << intersectingTriPairCount.size() << " pair-wise intersection tests." << std::endl;
#endif

                    if (intersectingTriPairCount[m] > 0 && triPairCheckIndex < _narrowPhasePairs.size())
                    {
                        std::pair<OBBModelContainer,OBBModelContainer>& triCheckPair = _narrowPhasePairs.at(triPairCheckIndex);

                        ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = triCheckPair.first._obbCollisionModel;
                        ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = triCheckPair.second._obbCollisionModel;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
                        std::cout << " - check triangle intersection for " << obbModel1->getName() << " -- " << obbModel2->getName() << std::endl;
                        std::cout << "   ==> tri-pair check " << m << " maps to worker unit " << workerUnitToBinMapping[m] << std::endl;
                        std::cout << "   ==> intersectingTriPairCount[" << m << "] = " << intersectingTriPairCount[m] << ", result count in worker unit " << workerUnitToBinMapping[m] << " = " << _streamedWorkerUnits[workerUnitToBinMapping[m]]->_nCollidingPairs << std::endl;
#endif

                        int potentialResults = intersectingTriPairCount[m] * CollisionTestElementsSize;

                        requiredSizesPerPairCheck.insert(std::make_pair(m, potentialResults));

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
                        std::cout << "   potentially intersecting tri-pairs = " << intersectingTriPairCount[m] << "; potential contact points from intersecting triangles = " << potentialResults << std::endl;
#endif

                        std::multimap<unsigned int, unsigned int> requiredSlotsPerBinSize;

                        unsigned int curBinSize = _streamedWorkerResultMinSize.getValue();

                        int bestDivSlotPosition = -1;

                        int bestDivSize = _streamedWorkerResultMaxSize.getValue();

                        for (unsigned int l = 0; l < _numStreamedWorkerResultBins.getValue(); l++)
                        {
                            unsigned int sizeDivBinK = potentialResults / curBinSize;
                            unsigned int sizeModBinK = potentialResults % curBinSize;

                            requiredSlotsPerBinSize.insert(std::make_pair(l, sizeDivBinK));

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout << "    bin fitting: DivBin_" << l << " = " << sizeDivBinK << ", ModBin_" << l << " = " << sizeModBinK << std::endl;
#endif

                            if (_streamedWorkerResults[l].size() > 0)
                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << "    required bins for size " << l << " = " << _streamedWorkerResults[l][0]->_maxResults << ": " << sizeDivBinK << " + " << (sizeModBinK == 0 ? "0" : "1") << std::endl;
#endif

                                if (sizeDivBinK > 0 && sizeDivBinK < bestDivSize)
                                {
                                    bestDivSize = sizeDivBinK;
                                    bestDivSlotPosition = l;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                    std::cout << "    new minimum div slot size = " << bestDivSize << ", in slot = " << bestDivSlotPosition << std::endl;
#endif
                                }
                            }

                            curBinSize *= 2;
                        }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout << "   Size requirements for pair-check " << m << ":" << std::endl;

                        for (std::multimap<unsigned int, unsigned int>::const_iterator it = requiredSlotsPerBinSize.begin(); it != requiredSlotsPerBinSize.end(); it++)
                        {
                            std::cout << "   - slot " << it->first << ": " << it->second << " bins required." << std::endl;
                        }

                        std::cout << "   Best div slot location: slot = " << bestDivSlotPosition << ", bins = " << bestDivSize << std::endl;
#endif

                        // Possibility 1: Does our test fit into the best slot size (divided) at once?

                        unsigned int summedResultSize = 0;

                        if (bestDivSlotPosition >= 0)
                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout << "  Alternative 1: Check bestDivSlotPosition = " << bestDivSlotPosition << ", if it can fit our pair test alone." << std::endl;
#endif

                            if (bestDivSize <= freeBinsPerBinLevel[bestDivSlotPosition])
                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << "  bins required = " << bestDivSize << " <= free bins = " << freeBinsPerBinLevel[bestDivSlotPosition] << std::endl;
#endif

                                if (_streamedWorkerResults[bestDivSlotPosition].size() > 0)
                                {
                                    unsigned int freeBinsInSlot = 0, blockedBinsInSlot = 0;
                                    bool resultSizeSatisfied = false;
                                    for (unsigned int n = 0; n < _streamedWorkerResults[bestDivSlotPosition].size(); n++)
                                    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                        std::cout << "    - block " << n << " state: " << _streamedWorkerResults[bestDivSlotPosition][n]->_blocked << ";";
#endif

                                        if (_streamedWorkerResults[bestDivSlotPosition][n]->_blocked == false)
                                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << " free, marking as blocked";
#endif

                                            claimedResultBinsPerPairCheck[bestDivSlotPosition].insert(std::make_pair(m, n));
                                            freeBinsInSlot++;
                                            summedResultSize += _streamedWorkerResults[bestDivSlotPosition][n]->_maxResults;
                                        }
                                        else
                                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << " blocked, continue";
#endif

                                            blockedBinsInSlot++;
                                        }
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                        std::cout << std::endl;
                                        std::cout << "    summedResultSize step " << n << " = " << summedResultSize << std::endl;
#endif

                                        if (summedResultSize >= potentialResults)
                                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << "    found enough bins in slot " << bestDivSlotPosition << " to fit tri-pair test results" << std::endl;
#endif

                                            triPairCheckAccepted = true;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << "    blocking tagged bins" << std::endl;
#endif

                                            for (std::map<unsigned int, unsigned int>::const_iterator bit = claimedResultBinsPerPairCheck[bestDivSlotPosition].begin(); bit != claimedResultBinsPerPairCheck[bestDivSlotPosition].end(); bit++)
                                            {
                                                if (bit->first == m)
                                                {
                                                    if (_streamedWorkerResults[bestDivSlotPosition][bit->second]->_blocked == false)
                                                    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                        std::cout << "     - blocking bin = " << bit->second << "; remaining free on level " << bestDivSlotPosition << " = " << freeBinsPerBinLevel[bestDivSlotPosition] << std::endl;
#endif

                                                        _streamedWorkerResults[bestDivSlotPosition][bit->second]->_blocked = true;
                                                        freeBinsPerBinLevel[bestDivSlotPosition] -= 1;
                                                    }
                                                }
                                            }

                                            resultSizeSatisfied = true;
                                            break;
                                        }
                                    }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                    std::cout << "    Free bins marked as blocked: " << freeBinsInSlot << ", already blocked: " << blockedBinsInSlot << std::endl;
                                    std::cout << "    summedResultSize = " << summedResultSize << " for potentialResults = " << potentialResults << std::endl;
                                    std::cout << "    triPairCheckAccepted = " << triPairCheckAccepted << ", resultSizeSatisfied = " << resultSizeSatisfied << std::endl;

                                    int remainingSize = potentialResults - summedResultSize;
                                    if (remainingSize > 0)
                                    {
                                        std::cout << "   remaining gap to potentialResults = " << remainingSize << std::endl;
                                        std::cout << "   need to fit additional bin(s) to accomodate" << std::endl;
                                    }
#endif

                                    if (!resultSizeSatisfied)
                                    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                        std::cout << "    WARNING: Failed to block enough result bins to satisfy size requirements: summedResultSize = " << summedResultSize << " < " << potentialResults << std::endl;
                                        std::cout << "    Un-blocking already blocked result bins." << std::endl;
#endif

                                        triPairCheckAccepted = false;
                                    }
                                }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << "    triPairCheckAccepted = " << triPairCheckAccepted;
#endif
                                if (triPairCheckAccepted)
                                {
                                    satisfiablePairChecks.insert(std::make_pair(m, true));
                                    markedPairChecks.insert(std::make_pair(m, triCheckPair));

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                    std::cout << "; marking for execution.";
#endif
                                }
                                else
                                {
                                    satisfiablePairChecks.insert(std::make_pair(m, false));
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                    std::cout << "; NOT MARKED FOR EXECUTION.";
#endif
                                }
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << std::endl;
#endif
                            }
                        }

                        // Possibility 2: Divide the test up amongst several bins
                        if (!triPairCheckAccepted)
                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout << "  Alternative 2: Trying to locate a bin combination that could fit." << std::endl;
                            std::cout << "  requiredSlotsPerBinSize.size() = " << requiredSlotsPerBinSize.size() << std::endl;
#endif
                            unsigned int diffToMaxBinSize = std::abs((int) _streamedWorkerResultMaxSize.getValue() - (int) bestDivSize);
                            unsigned int diffToMinBinSize = std::abs((int) _streamedWorkerResultMinSize.getValue() - (int) bestDivSize);

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout << "   difference bestDivSize - max bin size = " << diffToMaxBinSize << ", bestDivSize - min bin size = " << diffToMinBinSize << std::endl;
							if (diffToMaxBinSize < 0 || diffToMinBinSize < 0) {
									std::cerr << "ERROR! diffToMinBinSize (" << diffToMinBinSize <<") and diffToMaxBinSize (" << diffToMaxBinSize << ") should be > 0" << std::endl;
									return;
							}

                            if (diffToMaxBinSize < diffToMinBinSize)
                            {
                                bool resultSizeSatisfied = false;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << "   starting from largest bin size down" << std::endl;
#endif

                                for (std::multimap<unsigned int, unsigned int>::const_reverse_iterator it = requiredSlotsPerBinSize.rbegin(); it != requiredSlotsPerBinSize.rend(); it++)
                                {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                    std::cout << "   --> alternative " << it->first << ": " << it->second << " bins * " << _streamedWorkerResults[it->first][0]->_maxResults << " size = " << (it->second * _streamedWorkerResults[it->first][0]->_maxResults) << std::endl;
#endif

                                    unsigned int freeBinsInSlot = 0, blockedBinsInSlot = 0;
                                    if (freeBinsPerBinLevel[it->first] >= it->second)
                                    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                        std::cout << "      holds enough free bins to fulfill size requirement, trying to block: " << freeBinsPerBinLevel[it->first] << std::endl;
#endif

                                        for (unsigned int r = 0; r < _streamedWorkerResults[it->first].size(); r++)
                                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << "       - block " << r << "state: ";
#endif

                                            if (_streamedWorkerResults[it->first][r]->_blocked == false)
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << " free, marking as blocked";
#endif

                                                freeBinsInSlot++;
                                                claimedResultBinsPerPairCheck[it->first].insert(std::make_pair(m, r));
                                                summedResultSize += _streamedWorkerResults[it->first][r]->_maxResults;
                                            }
                                            else
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "  blocked, skipping";
#endif

                                                blockedBinsInSlot++;
                                            }
                                            std::cout << std::endl;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << "    summedResultSize step " << r << " = " << summedResultSize << std::endl;
#endif

                                            if (summedResultSize >= potentialResults)
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    found enough bins in slot " << it->first << " to fit tri-pair test results" << std::endl;
#endif

                                                triPairCheckAccepted = true;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    blocking tagged bins" << std::endl;
#endif

                                                for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator it = claimedResultBinsPerPairCheck.begin(); it != claimedResultBinsPerPairCheck.end(); it++)
                                                {
                                                    std::multimap<unsigned int, unsigned int>& claimedResultBinInLevel = claimedResultBinsPerPairCheck[it->first];
                                                    for (std::multimap<unsigned int, unsigned int>::const_iterator bit = claimedResultBinInLevel.begin(); bit != claimedResultBinInLevel.end(); bit++)
                                                    {
                                                        if (bit->first == m)
                                                        {
                                                            if (_streamedWorkerResults[it->first][bit->second]->_blocked == false)
                                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                                std::cout << "     - blocking bin = " << bit->second << "; remaining free on level " << it->first << " = " << freeBinsPerBinLevel[it->first] << std::endl;
#endif

                                                                _streamedWorkerResults[it->first][bit->second]->_blocked = true;
                                                                freeBinsPerBinLevel[it->first] -= 1;
                                                            }
                                                        }
                                                    }
                                                }
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    Free bins marked as blocked: " << freeBinsInSlot << ", already blocked: " << blockedBinsInSlot << std::endl;
#endif

                                                resultSizeSatisfied = true;
                                                break;
                                            }
                                        }
                                    }
                                    else
                                    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                        std::cout << "      slot " << it->first << " does not hold sufficient free bins: " << freeBinsPerBinLevel[it->first] << " out of " << it->second << std::endl;
                                        std::cout << "      trying to fit partially" << std::endl;
#endif

                                        for (unsigned int r = 0; r < _streamedWorkerResults[it->first].size(); r++)
                                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << "       - block " << r << " state: ";
#endif

                                            if (_streamedWorkerResults[it->first][r]->_blocked == false)
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << " free, marking as blocked" << std::endl;
#endif

                                                freeBinsInSlot++;
                                                claimedResultBinsPerPairCheck[it->first].insert(std::make_pair(m, r));
                                                summedResultSize += _streamedWorkerResults[it->first][r]->_maxResults;
                                            }
                                            else
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << " blocked, skipping" << std::endl;
#endif

                                                blockedBinsInSlot++;
                                            }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << "    summedResultSize step " << r << " = " << summedResultSize << std::endl;
#endif

                                            if (summedResultSize >= potentialResults)
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    found enough bins in slot " << it->first << " to fit tri-pair test results" << std::endl;
#endif

                                                triPairCheckAccepted = true;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    blocking tagged bins" << std::endl;
#endif

                                                for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator it = claimedResultBinsPerPairCheck.begin(); it != claimedResultBinsPerPairCheck.end(); it++)
                                                {
                                                    std::multimap<unsigned int, unsigned int>& claimedResultBinInLevel = claimedResultBinsPerPairCheck[it->first];
                                                    for (std::multimap<unsigned int, unsigned int>::const_iterator bit = claimedResultBinInLevel.begin(); bit != claimedResultBinInLevel.end(); bit++)
                                                    {
                                                        if (bit->first == m)
                                                        {
                                                            if (_streamedWorkerResults[it->first][bit->second]->_blocked == false)
                                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                                std::cout << "     - blocking bin = " << bit->second << "; remaining free on level " << it->first << " = " << freeBinsPerBinLevel[it->first] << std::endl;
#endif

                                                                _streamedWorkerResults[it->first][bit->second]->_blocked = true;
                                                                freeBinsPerBinLevel[it->first] -= 1;
                                                            }
                                                        }
                                                    }
                                                }
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    Free bins marked as blocked: " << freeBinsInSlot << ", already blocked: " << blockedBinsInSlot << std::endl;
#endif

                                                resultSizeSatisfied = true;
                                                break;
                                            }
                                        }
                                    }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                    std::cout << "   --> alternative " << it->first << " resultSizeSatisfied = " << resultSizeSatisfied << std::endl;
#endif

                                    if (resultSizeSatisfied)
                                    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                        std::cout << "      sufficient number of bins found, stop search" << std::endl;
#endif

                                        break;
                                    }
                                }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << "    summedResultSize = " << summedResultSize << " for potentialResults = " << potentialResults << std::endl;
                                std::cout << "    triPairCheckAccepted = " << triPairCheckAccepted << ", resultSizeSatisfied = " << resultSizeSatisfied << std::endl;
#endif

                                if (!resultSizeSatisfied)
                                {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                    std::cout << "    WARNING: Failed to block enough result bins to satisfy size requirements: summedResultSize = " << summedResultSize << " < " << potentialResults << std::endl;
                                    std::cout << "    Un-blocking already blocked result bins." << std::endl;
#endif

                                    triPairCheckAccepted = false;
                                }
                            }
                            else
                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << "   starting from smallest bin size up" << std::endl;
#endif

                                bool resultSizeSatisfied = false;
                                for (std::multimap<unsigned int, unsigned int>::const_iterator it = requiredSlotsPerBinSize.begin(); it != requiredSlotsPerBinSize.end(); it++)
                                {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                    std::cout << "   --> alternative " << it->first << ": " << it->second << " bins * " << _streamedWorkerResults[it->first][0]->_maxResults << " size = " << (it->second * _streamedWorkerResults[it->first][0]->_maxResults) << std::endl;
#endif

                                    unsigned int freeBinsInSlot = 0, blockedBinsInSlot = 0;
                                    if (freeBinsPerBinLevel[it->first] >= it->second)
                                    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                        std::cout << "      holds enough free bins to fulfill size requirement, trying to block: " << freeBinsPerBinLevel[it->first] << std::endl;
#endif

                                        for (unsigned int r = 0; r < _streamedWorkerResults[it->first].size(); r++)
                                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << "       - block " << r << "state: ";
#endif

                                            if (_streamedWorkerResults[it->first][r]->_blocked == false)
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << " free, marking as blocked";
#endif

                                                freeBinsInSlot++;
                                                claimedResultBinsPerPairCheck[it->first].insert(std::make_pair(m, r));
                                                summedResultSize += _streamedWorkerResults[it->first][r]->_maxResults;
                                            }
                                            else
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout < " blocked, skipping";
#endif

                                                blockedBinsInSlot++;
                                            }
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << std::endl;
                                            std::cout << "    summedResultSize step " << r << " = " << summedResultSize << std::endl;
#endif

                                            if (summedResultSize >= potentialResults)
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    found enough bins in slot " << it->first << " to fit tri-pair test results" << std::endl;
#endif
                                                triPairCheckAccepted = true;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    blocking tagged bins" << std::endl;
#endif

                                                for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator it = claimedResultBinsPerPairCheck.begin(); it != claimedResultBinsPerPairCheck.end(); it++)
                                                {
                                                    std::multimap<unsigned int, unsigned int>& claimedResultBinInLevel = claimedResultBinsPerPairCheck[it->first];
                                                    for (std::multimap<unsigned int, unsigned int>::const_iterator bit = claimedResultBinInLevel.begin(); bit != claimedResultBinInLevel.end(); bit++)
                                                    {
                                                        if (bit->first == m)
                                                        {
                                                            if (_streamedWorkerResults[it->first][bit->second]->_blocked == false)
                                                            {

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                                std::cout << "     - blocking bin = " << bit->second << "; remaining free on level " << it->first << " = " << freeBinsPerBinLevel[it->first] << std::endl;
#endif

                                                                _streamedWorkerResults[it->first][bit->second]->_blocked = true;
                                                                freeBinsPerBinLevel[it->first] -= 1;
                                                            }
                                                        }
                                                    }
                                                }
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    Free bins marked as blocked: " << freeBinsInSlot << ", already blocked: " << blockedBinsInSlot << std::endl;
#endif

                                                resultSizeSatisfied = true;
                                                break;
                                            }
                                        }
                                    }
                                    else
                                    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                        std::cout << "      slot " << it->first << " does not hold sufficient free bins: " << freeBinsPerBinLevel[it->first] << " out of " << it->second << std::endl;
                                        std::cout << "      trying to fit partially" << std::endl;
#endif

                                        for (unsigned int r = 0; r < _streamedWorkerResults[it->first].size(); r++)
                                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << "       - block " << r << " state:";
#endif

                                            if (_streamedWorkerResults[it->first][r]->_blocked == false)
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << " free, marking as blocked" << std::endl;
#endif

                                                freeBinsInSlot++;
                                                claimedResultBinsPerPairCheck[it->first].insert(std::make_pair(m, r));
                                                summedResultSize += _streamedWorkerResults[it->first][r]->_maxResults;
                                            }
                                            else
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << " blocked, skipping" << std::endl;
#endif

                                                blockedBinsInSlot++;
                                            }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                            std::cout << "    summedResultSize step " << r << " = " << summedResultSize << std::endl;
#endif

                                            if (summedResultSize >= potentialResults)
                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    found enough bins in slot " << it->first << " to fit tri-pair test results" << std::endl;
#endif

                                                triPairCheckAccepted = true;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    blocking tagged bins" << std::endl;
#endif

                                                for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator it = claimedResultBinsPerPairCheck.begin(); it != claimedResultBinsPerPairCheck.end(); it++)
                                                {
                                                    std::multimap<unsigned int, unsigned int>& claimedResultBinInLevel = claimedResultBinsPerPairCheck[it->first];
                                                    for (std::multimap<unsigned int, unsigned int>::const_iterator bit = claimedResultBinInLevel.begin(); bit != claimedResultBinInLevel.end(); bit++)
                                                    {
                                                        if (bit->first == m)
                                                        {
                                                            if (_streamedWorkerResults[it->first][bit->second]->_blocked == false)
                                                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                                std::cout << "     - blocking bin = " << bit->second << "; remaining free on level " << it->first << " = " << freeBinsPerBinLevel[it->first] << std::endl;
#endif

                                                                _streamedWorkerResults[it->first][bit->second]->_blocked = true;
                                                                freeBinsPerBinLevel[it->first] -= 1;
                                                            }
                                                        }
                                                    }
                                                }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout << "    Free bins marked as blocked: " << freeBinsInSlot << ", already blocked: " << blockedBinsInSlot << std::endl;
#endif

                                                resultSizeSatisfied = true;
                                                break;
                                            }
                                        }
                                    }
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                    std::cout << "   --> alternative " << it->first << " resultSizeSatisfied = " << resultSizeSatisfied << std::endl;
#endif

                                    if (resultSizeSatisfied)
                                    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                        std::cout << "      sufficient number of bins found, stop search" << std::endl;
#endif
                                        break;
                                    }
                                }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << "    summedResultSize = " << summedResultSize << " for potentialResults = " << potentialResults << std::endl;
                                std::cout << "    triPairCheckAccepted = " << triPairCheckAccepted << ", resultSizeSatisfied = " << resultSizeSatisfied << std::endl;
#endif

                                if (!resultSizeSatisfied)
                                {

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                    std::cout << "    WARNING: Failed to block enough result bins to satisfy size requirements: summedResultSize = " << summedResultSize << " < " << potentialResults << std::endl;
                                    std::cout << "    Un-blocking already blocked result bins." << std::endl;
#endif

                                    triPairCheckAccepted = false;
                                }
                            }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout << "    triPairCheckAccepted = " << triPairCheckAccepted;
#endif

                            if (triPairCheckAccepted)
                            {
                                satisfiablePairChecks.insert(std::make_pair(m, true));
                                markedPairChecks.insert(std::make_pair(m, triCheckPair));

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << "; marking for execution.";
#endif

                            }
                            else
                            {
                                satisfiablePairChecks.insert(std::make_pair(m, false));

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << "; NOT MARKED FOR EXECUTION.";
#endif
                            }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout << std::endl;
#endif
                        }
                    }
                }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout << "=== Remaining free result bins per level after matching ===" << std::endl;
                for (std::map<unsigned int, int>::const_iterator it = freeBinsPerBinLevel.begin(); it != freeBinsPerBinLevel.end(); it++)
                {
                    std::cout << " - level " << it->first << ": " << it->second << " of " << _streamedWorkerResults[it->first].size() << std::endl;
                }

                std::cout << "=== triPairTests requirements === " << std::endl;
                for (std::map<unsigned int, bool>::const_iterator it = satisfiablePairChecks.begin(); it != satisfiablePairChecks.end(); it++)
                {
                    std::cout << " - check " << it->first << " satisfied = " << it->second << std::endl;
                }
#endif

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout << "=== bin occupancy for tasks ===" << std::endl;
#endif

                std::map<unsigned int, std::multimap<unsigned int, unsigned int> > blocksByTaskAndLevel;
                for (unsigned int l = 0; l < intersectingTriPairCount.size(); l++)
                {
                    if (intersectingTriPairCount[l] > 0)
                        blocksByTaskAndLevel.insert(std::make_pair(l, std::multimap<unsigned int, unsigned int>()));
                }

                for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator bit = claimedResultBinsPerPairCheck.begin(); bit != claimedResultBinsPerPairCheck.end(); bit++)
                {
                    std::multimap<unsigned int, unsigned int>& claimedResultBinsPerTask = bit->second;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout << " - block " << bit->first << ": ";
#endif

                    for (std::multimap<unsigned int, unsigned int>::iterator rit = claimedResultBinsPerTask.begin(); rit != claimedResultBinsPerTask.end(); rit++)
                    {
                        unsigned int tmp1 = rit->first;
                        unsigned int tmp2 = bit->first;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout << " task " << rit->first << ": bin " << rit->second << ";";
#endif

                        blocksByTaskAndLevel[tmp1].insert(std::make_pair(tmp2, rit->second));
                    }
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout << std::endl;
#endif
                }

                std::map<unsigned int, std::vector<gProximityWorkerResult*> > freeResultBins;
                std::map<unsigned int, std::vector<std::pair<unsigned int, unsigned int> > > freeResultBinSizes;
                std::map<unsigned int, std::vector<unsigned int> > freeResultBinStartIndices;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout << "=== bin occupancy sorted by tasks ===" << std::endl;
#endif

                for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::const_iterator bit = blocksByTaskAndLevel.begin(); bit != blocksByTaskAndLevel.end(); bit++)
                {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout << "  - task " << bit->first << ": ";
#endif

                    std::vector<gProximityWorkerResult*> taskBins;
                    std::vector<std::pair<unsigned int, unsigned int> > taskBinSizes;
                    std::vector<unsigned int> taskBinStartIndices;

                    unsigned int curEnd = 0;
                    const std::multimap<unsigned int, unsigned int>& blocksOfTask = bit->second;
                    for (std::multimap<unsigned int, unsigned int>::const_iterator task_it = blocksOfTask.begin(); task_it != blocksOfTask.end(); task_it++)
                    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout << "block " << task_it->first << ": bin " << task_it->second << ";";
#endif
                        taskBins.push_back(_streamedWorkerResults[task_it->first][task_it->second]);
                        taskBinSizes.push_back(std::make_pair(0, _streamedWorkerResults[task_it->first][task_it->second]->_maxResults));
                        taskBinStartIndices.push_back(curEnd);
                        curEnd += _streamedWorkerResults[task_it->first][task_it->second]->_maxResults;
                    }
                    std::cout << std::endl;

                    freeResultBins.insert(std::make_pair(bit->first, taskBins));
                    freeResultBinSizes.insert(std::make_pair(bit->first, taskBinSizes));
                    freeResultBinStartIndices.insert(std::make_pair(bit->first, taskBinStartIndices));
                }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout << "=== Result bins after preparation for intersection test call ===" << std::endl;
#endif

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                for (std::map<unsigned int, std::vector<gProximityWorkerResult*> >::iterator it = freeResultBins.begin(); it != freeResultBins.end(); it++)
                {

                    std::cout << " - Task " << it->first << ": " << it->second.size() << " result bins." << std::endl;
                    std::vector<gProximityWorkerResult*>& taskResults = it->second;
                    for (unsigned int k = 0; k < taskResults.size(); k++)
                    {
                        gProximityWorkerResult* wr = taskResults[k];
                        std::cout << "    - result " << k << ": bin " << wr->_resultBin << ", index = " << wr->_resultIndex << ", maxResults = " << wr->_maxResults
                                  << ", range = " << freeResultBinSizes[it->first][k].first << " - " << freeResultBinSizes[it->first][k].second
                                  << ", size = " << freeResultBinStartIndices[it->first][k]
                                  << std::endl;
                    }
                }
#endif

                std::vector<unsigned int> activeTaskIDs;
                std::map<unsigned int, sofa::core::collision::DetectionOutputVector*> detection_outputs_per_task;
                for (std::map<unsigned int, std::vector<gProximityWorkerResult*> >::iterator it = freeResultBins.begin(); it != freeResultBins.end(); it++)
                {
                    activeTaskIDs.push_back(it->first);
                    detection_outputs_per_task.insert(std::make_pair(it->first, (sofa::core::collision::DetectionOutputVector*)NULL));
                }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout << "=== Calls to tri-pair test routine: " << markedPairChecks.size() << " tests to process. ===" << std::endl;
#endif

                unsigned int numActiveTasks = activeTaskIDs.size();
                std::cout << " activeTasks = " << numActiveTasks << std::endl;
                if (numActiveTasks > 0)
                {
                    //#pragma omp parallel for num_threads(numActiveTasks)
                    for (unsigned int t = 0; t < numActiveTasks; t++)
                    {
                        std::cout << " - Task " << activeTaskIDs[t] << std::endl;

                        std::pair<OBBModelContainer,OBBModelContainer>& triCheckPair = markedPairChecks[activeTaskIDs[t]];

                        if (triCheckPair.first._obbCollisionModel == NULL || triCheckPair.second._obbCollisionModel == NULL)
                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout << "   WARNING -- one or both collision model pointers == NULL; this should not happen. triCheckPair.first._obbCollisionModel = " << triCheckPair.first._obbCollisionModel << ", triCheckPair.second._obbCollisionModel = " << triCheckPair.second._obbCollisionModel << std::endl;
#endif

                            continue;
                        }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout << "    check models: " << triCheckPair.first._obbCollisionModel->getName() << " -- " << triCheckPair.second._obbCollisionModel->getName() << std::endl;
#endif
                        OBBContainer& obbTree1 = triCheckPair.first._obbContainer;
                        OBBContainer& obbTree2 = triCheckPair.second._obbContainer;

                        std::vector<gProximityWorkerResult*>& freeResultBins_Task = freeResultBins[activeTaskIDs[t]];
                        std::vector<std::pair<unsigned int, unsigned int> >& freeResultBinSizes_Task = freeResultBinSizes[activeTaskIDs[t]];
                        std::vector<unsigned int>& freeResultBinStartIndices_Task = freeResultBinStartIndices[activeTaskIDs[t]];

                        int nIntersectingTrianglePairs = 0;

                        sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_TriangleIntersection_Streams");

                        float elapsedTime;
                        ObbTreeGPU_TriangleIntersection_Streams_Batch(&obbTree1, &obbTree2,
                                                                        _streamedWorkerUnits[workerUnitToBinMapping[activeTaskIDs[t]]],
                                                                        freeResultBins_Task,
                                                                        freeResultBinSizes_Task,
                                                                        freeResultBinStartIndices_Task,
                                                                        _triTestStreams,
                                                                        _triTestEvents,
                                                                        _memoryStream,
                                                                        _triTestStartEvent,
                                                                        _triTestEndEvent,
                                                                        m_alarmDistance, m_contactDistance, nIntersectingTrianglePairs, elapsedTime
                                                                     );

                        sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_TriangleIntersection_Streams");

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout << "  === Results after tri-tri test: " << nIntersectingTrianglePairs <<  " ===" << std::endl;
#endif

                        unsigned int totalResults = 0;
                        unsigned int writtenResults = 0;
                        for (unsigned int l = 0; l < freeResultBins_Task.size(); l++)
                        {
                            freeResultBins_Task[l]->_numResults = freeResultBins_Task[l]->h_outputIndex;

                            totalResults += freeResultBins_Task[l]->_numResults;
                            writtenResults += freeResultBins_Task[l]->h_outputIndex;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout << "   * results in bin " << l << ": " << freeResultBins_Task[l]->_numResults << ", outputIndex = " << freeResultBins_Task[l]->h_outputIndex << std::endl;
#endif
                        }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout << "   total results = " << totalResults << ", outputIndices summed = " << writtenResults << std::endl;
#endif

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        for (unsigned int l = 0; l < freeResultBins_Task.size(); l++)
                        {
                            std::cout << "    - bin " << l << " results: " << std::endl;
                            gProximityWorkerResult* workerResultUnit = freeResultBins_Task[l];
                            for (int k = 0; k < workerResultUnit->_numResults; k++)
                            {
                                std::cout << "     - " << k << ": valid = " << workerResultUnit->d_ptr->h_valid[k]
                                          << ", id = " << workerResultUnit->d_ptr->h_contactId[k]
                                          << ", type = " << workerResultUnit->d_ptr->h_gProximityContactType[k]
                                          << ", distance = " << workerResultUnit->d_ptr->h_distance[k]
                                          << ", elements = " << workerResultUnit->d_ptr->h_elems[k].w << "," << workerResultUnit->d_ptr->h_elems[k].x << "," << workerResultUnit->d_ptr->h_elems[k].y << "," << workerResultUnit->d_ptr->h_elems[k].z
                                          << ", point0 = " << workerResultUnit->d_ptr->h_point0[k].x << "," << workerResultUnit->d_ptr->h_point0[k].y << "," << workerResultUnit->d_ptr->h_point0[k].z
                                          << ", point1 = " << workerResultUnit->d_ptr->h_point1[k].x << "," << workerResultUnit->d_ptr->h_point1[k].y << "," << workerResultUnit->d_ptr->h_point1[k].z
                                          << ", normal = " << workerResultUnit->d_ptr->h_normal[k].x << "," << workerResultUnit->d_ptr->h_normal[k].y << "," << workerResultUnit->d_ptr->h_normal[k].z
                                          << std::endl;
                            }
                        }
#endif

                        if (totalResults > 0)
                        {
                            ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = triCheckPair.first._obbCollisionModel;
                            ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = triCheckPair.second._obbCollisionModel;

                            sofa::core::collision::DetectionOutputVector*& outputs = getDetectionOutputs(obbModel1, obbModel2);
                            sofa::core::collision::TDetectionOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >* discreteOutputs =
                            m_intersection->getOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, obbModel2, outputs);

                            if (discreteOutputs == NULL)
                            {

                                discreteOutputs = m_intersection->createOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, obbModel2);
                                if (outputs == NULL)
                                {
                                    outputs = dynamic_cast<sofa::core::collision::DetectionOutputVector*>(discreteOutputs);
                                }
                            }

                            if (outputs && discreteOutputs)
                            {
                                const double maxContactDist = m_alarmDistance + (m_alarmDistance - m_contactDistance);
                                const double maxContactDist2 = maxContactDist * maxContactDist;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
                                std::cout << "Add contacts to DetectionOutputVector" << std::endl;
#endif

                                for (unsigned int l = 0; l < freeResultBins_Task.size(); l++)
                                {
                                    gProximityWorkerResult* workerResultUnit = freeResultBins_Task[l];
                                    for (int k = 0; k < workerResultUnit->_numResults; k++)
                                    {
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                        const float3& normalVec = workerResultUnit->d_ptr->h_normal.operator [](k);
#else
                                        const float3& normalVec = workerResultUnit->d_ptr->h_normal[k];
#endif
                                        Vector3 contactNormal(normalVec.x, normalVec.y, normalVec.z);
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                        const double& contactDistance = workerResultUnit->d_ptr->h_distance.operator [](k);
#else
                                        const double& contactDistance = workerResultUnit->d_ptr->h_distance[k];
#endif
                                        if (contactNormal.norm() >= 1e-06)
                                        {
                                            if (contactNormal.norm2() <= maxContactDist2 &&
                                                std::fabs(contactDistance - m_contactDistance) < m_contactDistance)
                                            {
                                                discreteOutputs->resize(discreteOutputs->size()+1);
                                                sofa::core::collision::DetectionOutput *detection = &*(discreteOutputs->end()-1);

#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                                const int& contactId = workerResultUnit->d_ptr->h_contactId.operator [](k);
#else
                                                const int& contactId = workerResultUnit->d_ptr->h_contactId[k];
#endif
                                                detection->id = contactId;

#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                                const float3& point0 = workerResultUnit->d_ptr->h_point0.operator [](k);
                                                const float3& point1 = workerResultUnit->d_ptr->h_point1.operator [](k);
#else
                                                const float3& point0 = workerResultUnit->d_ptr->h_point0[k];
                                                const float3& point1 = workerResultUnit->d_ptr->h_point1[k];
#endif
                                                detection->point[0] = Vector3(point0.x, point0.y, point0.z);
                                                detection->point[1] = Vector3(point1.x, point1.y, point1.z);

                                                detection->normal = contactNormal;

                                                detection->value = detection->normal.norm();
                                                detection->normal /= detection->value;

                                                detection->value -= m_contactDistance;
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                                const gProximityContactType& contactType = workerResultUnit->d_ptr->h_gProximityContactType.operator [](k);
#else
                                                const gProximityContactType& contactType = workerResultUnit->d_ptr->h_gProximityContactType[k];
#endif
                                                detection->contactType = (sofa::core::collision::DetectionOutputContactType) contactType;

#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                                const int4& contactElems = workerResultUnit->d_ptr->h_elems.operator [](k);
#else
                                                const int4& contactElems = workerResultUnit->d_ptr->h_elems[k];
#endif
                                                if (contactType == COLLISION_LINE_LINE)
                                                {
                                                    detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, contactElems.w * 3 + contactElems.y); // << CollisionElementIterator

                                                    detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, contactElems.x * 3 + contactElems.z); // << CollisionElementIterator

                                                    detection->elemFeatures.first = contactElems.y;
                                                    detection->elemFeatures.second = contactElems.z;
                                                }
                                                else if (contactType == COLLISION_VERTEX_FACE)
                                                {
                                                    if (contactElems.z == -1)
                                                    {
                                                        detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, contactElems.x * 3); // << CollisionElementIterator
                                                        detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, contactElems.w * 3 + contactElems.y); // << CollisionElementIterator
                                                        detection->elemFeatures.first = contactElems.y;
                                                    }
                                                    else if (contactElems.y == -1)
                                                    {
                                                        detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, contactElems.x * 3); // << CollisionElementIterator
                                                        detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, contactElems.w * 3 + contactElems.z); // << CollisionElementIterator
                                                        detection->elemFeatures.second = contactElems.z;
                                                    }
                                                }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
                                                std::cout << "  - add from bin " << l << ", index " << k << ": id = " << detection->id  << ", distance = " << detection->value << ", point0 = " << detection->point[0] << ", point1 = " << detection->point[1] << ", normal = " << detection->normal << std::endl;
#endif

                                            }
                                        }
                                    }
                                }

                                std::sort(discreteOutputs->begin(), discreteOutputs->end(), contactTypeCompare);

                                detection_outputs_per_task[activeTaskIDs[t]] = outputs;
                            }
                        }
                    }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
                    std::cout << "=== DetectionOutputVector instances per task: " << detection_outputs_per_task.size() << " elements in map ===" << std::endl;
#endif

                    for (std::map<unsigned int, sofa::core::collision::DetectionOutputVector*>::iterator cit = detection_outputs_per_task.begin(); cit != detection_outputs_per_task.end(); cit++)
                    {
                        if (cit->second != NULL)
                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
                            std::cout << " - task " << cit->first << ": Vector instantiated, " << cit->second->size() << " contact points." << std::endl;
#endif

                            std::pair<OBBModelContainer,OBBModelContainer>& triCheckPair = markedPairChecks[cit->first];

                            if (triCheckPair.first._obbCollisionModel == NULL || triCheckPair.second._obbCollisionModel == NULL)
                            {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
                                std::cout << "   WARNING -- one or both collision model pointers == NULL; this should not happen. triCheckPair.first._obbCollisionModel = " << triCheckPair.first._obbCollisionModel << ", triCheckPair.second._obbCollisionModel = " << triCheckPair.second._obbCollisionModel << std::endl;
#endif

                                continue;
                            }

                            std::pair< core::CollisionModel*, core::CollisionModel* > cm_pair = std::make_pair(triCheckPair.first._obbCollisionModel, triCheckPair.second._obbCollisionModel);
                            DetectionOutputMap::iterator it = getDetectionOutputs().find(cm_pair);

                            if (it == getDetectionOutputs().end())
                            {
                                getDetectionOutputs().insert(std::make_pair(cm_pair, cit->second));
                            }
                            else
                            {
                                getDetectionOutputs()[cm_pair] = cit->second;
                            }
                        }
                        else
                        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
                            std::cout << " - task " << cit->first << ": NOT INSTANTIATED." << std::endl;
#endif
                        }
                     }
                }
            }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
            std::cout << "=== Removing blocked flags from worker results ===" << std::endl;
#endif

            for (std::map<unsigned int, std::vector<gProximityWorkerResult*> >::iterator it = _streamedWorkerResults.begin(); it != _streamedWorkerResults.end(); it++)
            {
                std::vector<gProximityWorkerResult*>& workerResults = it->second;
                unsigned int outputIndexReset = 0;
                for (int u = 0; u < workerResults.size(); u++)
                {
                    workerResults[u]->_blocked = false;
                    TOGPU_ASYNC(workerResults[u]->d_outputIndex, &outputIndexReset, sizeof(unsigned int), _memoryStream);
                    workerResults[u]->h_outputIndex = outputIndexReset;
                }
            }
#endif
        }
    }

    if (!_useDynamicWorkerScheme.getValue())
    {
        for (unsigned int k = 0; k < _narrowPhasePairs.size(); k++)
        {
            OBBContainer& obbTree1 = _narrowPhasePairs[k].first._obbContainer;
            OBBContainer& obbTree2 = _narrowPhasePairs[k].second._obbContainer;
            ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = _narrowPhasePairs[k].first._obbCollisionModel;
            ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = _narrowPhasePairs[k].second._obbCollisionModel;

            std::cout << "  - check: " << obbModel1->getName() << " - " << obbModel2->getName() << std::endl;
            {
                if (m_workerUnitOccupied[m_numOBBTreePairsTested % 4] == false)
                {
                    m_workerUnitOccupied[m_numOBBTreePairsTested % 4] = true;

                    int nIntersectingTriPairs = 0;
                    ObbTreeGPU_BVH_Traverse(&obbTree1, &obbTree2,
                                            m_gProximityWorkerUnits[m_numOBBTreePairsTested % 4],
                                            m_alarmDistance, m_contactDistance, nIntersectingTriPairs);

                    if (nIntersectingTriPairs > 0)
                    {
                        unsigned int maxPossibleResults = (obbTree1.nTris * obbTree2.nTris * CollisionTestElementsSize);
                        int potentialResults = nIntersectingTriPairs * CollisionTestElementsSize;

                        std::cout << "    potentially intersecting tri-pairs = " << nIntersectingTriPairs << "; potential contact points from intersecting triangles = " << potentialResults <<  "; max. possible contact results for this OBB pair = " << maxPossibleResults << " (" << obbTree1.nTris << " * " << obbTree2.nTris << " * " << CollisionTestElementsSize << ")" << std::endl;

                        int sizeDivBin1 = potentialResults / m_resultBinSize1;
                        int sizeDivBin2 = potentialResults / m_resultBinSize2;
                        int sizeDivBin3 = potentialResults / m_resultBinSize3;

                        int sizeModBin1 = potentialResults % m_resultBinSize1;
                        int sizeModBin2 = potentialResults % m_resultBinSize2;
                        int sizeModBin3 = potentialResults % m_resultBinSize3;

                        std::cout << "    bin fitting: DivBin1 = " << sizeDivBin1 << ", DivBin2 = " << sizeDivBin2 << ", DivBin3 = " << sizeDivBin3 << std::endl;
                        std::cout << "              ModBin1 = " << sizeModBin1 << ", ModBin2 = " << sizeModBin2 << ", ModBin3 = " << sizeModBin3 << std::endl;

                        std::cout << "    required bins for size1 = " << m_resultBinSize1 << ": " << sizeDivBin1 << " + " << (sizeModBin1 == 0 ? "0" : "1") << std::endl;
                        std::cout << "    required bins for size2 = " << m_resultBinSize2 << ": " << sizeDivBin2 << " + " << (sizeModBin2 == 0 ? "0" : "1") << std::endl;
                        std::cout << "    required bins for size3 = " << m_resultBinSize3 << ": " << sizeDivBin3 << " + " << (sizeModBin3 == 0 ? "0" : "1") << std::endl;

                        gProximityWorkerResult* workerResultUnit = NULL;
                        if (potentialResults < m_resultBinSize1)
                        {
                            workerResultUnit = m_gProximityWorkerResults_64[m_numOBBTreePairsTested % 4];
                            m_workerResultOccupied_64[m_numOBBTreePairsTested % 4] = true;
                            std::cout << "    use result bin 1, up to " << m_resultBinSize1 << " max. contacts." << std::endl;
                        }
                        else if (potentialResults >= m_resultBinSize1 && potentialResults < m_resultBinSize2)
                        {
                            workerResultUnit = m_gProximityWorkerResults_128[m_numOBBTreePairsTested % 4];
                            m_workerResultOccupied_128[m_numOBBTreePairsTested % 4] = true;
                            std::cout << "    use result bin 2, up to " << m_resultBinSize2 << " max. contacts." << std::endl;
                        }
                        else if (potentialResults > m_resultBinSize2 && potentialResults < m_resultBinSize3)
                        {
                            workerResultUnit = m_gProximityWorkerResults_256[m_numOBBTreePairsTested % 4];
                            m_workerResultOccupied_256[m_numOBBTreePairsTested % 4] = true;
                            std::cout << "    use result bin 3, up to " << m_resultBinSize3 << " max. contacts." << std::endl;
                        }
                        else
                        {
                            std::cout << "    TOO BIG TO FIT in bins: " << potentialResults << std::endl;
                        }

                        if (workerResultUnit != NULL)
                        {
                            int nIntersectingTriangles = 0;
                            ObbTreeGPU_TriangleIntersection(&obbTree1, &obbTree2,
                                                            m_gProximityWorkerUnits[m_numOBBTreePairsTested % 4],
                                                            workerResultUnit,
                                                            m_alarmDistance, m_contactDistance, nIntersectingTriangles);

                            if (workerResultUnit->_numResults > 0)
                            {

                                std::cout << "   Results from triangle intersection call = " << workerResultUnit->_numResults << std::endl;

                                sofa::core::collision::DetectionOutputVector*& outputs = getDetectionOutputs(obbModel1, obbModel2);
                                sofa::core::collision::TDetectionOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >* discreteOutputs =
                                m_intersection->getOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, obbModel2, outputs);

                                if (discreteOutputs == NULL)
                                {

                                    discreteOutputs = m_intersection->createOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, obbModel2);
                                    if (outputs == NULL)
                                    {
                                        outputs = dynamic_cast<sofa::core::collision::DetectionOutputVector*>(discreteOutputs);
                                    }
                                }

                                if (outputs && discreteOutputs)
                                {
                                    const double maxContactDist = m_alarmDistance + (m_alarmDistance - m_contactDistance);
                                    const double maxContactDist2 = maxContactDist * maxContactDist;

                                    for (int k = 0; k < workerResultUnit->_numResults; k++)
                                    {

                                        /*std::cout << " * id = " << workerResultUnit->h_contactId[k] <<
                                                     ", elems = " << workerResultUnit->h_elems[k].w << "/" << workerResultUnit->h_elems[k].x << "/" << workerResultUnit->h_elems[k].y << "/" << workerResultUnit->h_elems[k].z <<
                                                     ", distance " << workerResultUnit->h_distance[k] <<
                                                     ", point0 = " << workerResultUnit->h_point0[k].x << "," << workerResultUnit->h_point0[k].y << "," << workerResultUnit->h_point0[k].z << "," <<
                                                     ", point1 = " << workerResultUnit->h_point1[k].x << "," << workerResultUnit->h_point1[k].y << "," << workerResultUnit->h_point1[k].z << "," <<
                                                     ", normal = " << workerResultUnit->h_normal[k].x << "," << workerResultUnit->h_normal[k].y << "," << workerResultUnit->h_normal[k].z << "," <<
                                                     ", type = " << workerResultUnit->h_gProximityContactType[k]
                                                     << std::endl;*/
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                        const float3& normalVec = workerResultUnit->d_ptr->h_normal.operator [](k);
#else
                                        const float3& normalVec = workerResultUnit->d_ptr->h_normal[k];
#endif
                                        Vector3 contactNormal(normalVec.x, normalVec.y, normalVec.z);
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                        const double& contactDistance = workerResultUnit->d_ptr->h_distance.operator [](k);
#else
                                        const double& contactDistance = workerResultUnit->d_ptr->h_distance[k];
#endif
                                        if (contactNormal.norm() >= 1e-06)
                                        {
                                            if (contactNormal.norm2() <= maxContactDist2 &&
                                                std::fabs(contactDistance - m_contactDistance) < m_contactDistance)
                                            {
                                                discreteOutputs->resize(discreteOutputs->size()+1);
                                                sofa::core::collision::DetectionOutput *detection = &*(discreteOutputs->end()-1);
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                                const int& contactId = workerResultUnit->d_ptr->h_contactId.operator [](k);
#else
                                                const int& contactId = workerResultUnit->d_ptr->h_contactId[k];
#endif
                                                detection->id = contactId;

#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                                const float3& point0 = workerResultUnit->d_ptr->h_point0.operator [](k);
                                                const float3& point1 = workerResultUnit->d_ptr->h_point1.operator [](k);
#else
                                                const float3& point0 = workerResultUnit->d_ptr->h_point0[k];
                                                const float3& point1 = workerResultUnit->d_ptr->h_point1[k];
#endif

                                                detection->point[0] = Vector3(point0.x, point0.y, point0.z);
                                                detection->point[1] = Vector3(point1.x, point1.y, point1.z);

                                                detection->normal = contactNormal;

                                                // Minus contact distance: Testing...
                                                detection->value = detection->normal.norm();
                                                detection->normal /= detection->value;

                                                detection->value -= m_contactDistance;
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                                const gProximityContactType& contactType = workerResultUnit->d_ptr->h_gProximityContactType.operator [](k);
#else
                                                const gProximityContactType& contactType = workerResultUnit->d_ptr->h_gProximityContactType[k];
#endif
                                                detection->contactType = (sofa::core::collision::DetectionOutputContactType) contactType;



#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
                                                const int4& contactElems = workerResultUnit->d_ptr->h_elems.operator [](k);
#else
                                                const int4& contactElems = workerResultUnit->d_ptr->h_elems[k];
#endif
                                                if (contactType == COLLISION_LINE_LINE)
                                                {
                                                    detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, contactElems.w * 3 + contactElems.y); // << CollisionElementIterator

                                                    detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, contactElems.x * 3 + contactElems.z); // << CollisionElementIterator

                                                    detection->elemFeatures.first = contactElems.y;
                                                    detection->elemFeatures.second = contactElems.z;
                                                }
                                                else if (contactType == COLLISION_VERTEX_FACE)
                                                {
                                                    if (contactElems.z == -1)
                                                    {
                                                        detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, contactElems.x * 3); // << CollisionElementIterator
                                                        detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, contactElems.w * 3 + contactElems.y); // << CollisionElementIterator
                                                        detection->elemFeatures.first = contactElems.y;
                                                    }
                                                    else if (contactElems.y == -1)
                                                    {
                                                        detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, contactElems.x * 3); // << CollisionElementIterator
                                                        detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, contactElems.w * 3 + contactElems.z); // << CollisionElementIterator
                                                        detection->elemFeatures.second = contactElems.z;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    std::sort(discreteOutputs->begin(), discreteOutputs->end(), contactTypeCompare);

                                    std::pair< core::CollisionModel*, core::CollisionModel* > cm_pair = std::make_pair(obbModel1, obbModel2);

                                    DetectionOutputMap::iterator it = getDetectionOutputs().find(cm_pair);

                                    std::cout << "   Contact points count = " << discreteOutputs->size() << std::endl;

                                    if (it == getDetectionOutputs().end())
                                    {
                                        getDetectionOutputs().insert(std::make_pair(cm_pair, outputs));
                                    }
                                    else
                                    {
                                        getDetectionOutputs()[cm_pair] = outputs;
                                    }
                                }
                            }
                        }

                        if (potentialResults < m_resultBinSize1)
                        {
                            m_workerResultOccupied_64[m_numOBBTreePairsTested % 4] = false;
                        }
                        else if (potentialResults >= m_resultBinSize1 && potentialResults < m_resultBinSize2)
                        {
                            m_workerResultOccupied_128[m_numOBBTreePairsTested % 4] = false;
                        }
                        else if (potentialResults > m_resultBinSize2 && potentialResults < m_resultBinSize3)
                        {
                            m_workerResultOccupied_256[m_numOBBTreePairsTested % 4] = false;
                        }
                    }

                    m_workerUnitOccupied[m_numOBBTreePairsTested % 4] = false;
                }
            }
            m_numOBBTreePairsTested++;
        }
    }
}


//#define OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODELS
void ObbTreeGPUCollisionDetection::addCollisionModels(const sofa::helper::vector<core::CollisionModel *> v)
{
    if (!this->f_printLog.getValue())
        this->f_printLog.setValue(true);

#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODELS
    sout << "ObbTreeGPUCollisionDetection::addCollisionModels(): " << v.size() << " models." << sendl;
#endif
    for (sofa::helper::vector<core::CollisionModel *>::const_iterator it = v.begin(); it<v.end(); it++)
    {
        bool obbModelFound = false, pqpModelFound = false;
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODELS
        sout << " * Add model " << (*it)->getName() << " of type " << (*it)->getTypeName() << sendl;
#endif
        core::CollisionModel* cmc = (*it);
        do
        {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODELS
            sout << "   examine " << cmc->getName() << ", type " << cmc->getTypeName() << " if it's a ObbTreeGPUCollisionModel" << sendl;
#endif
            ObbTreeGPUCollisionModel<Vec3Types>* obbModel = dynamic_cast<ObbTreeGPUCollisionModel<Vec3Types>*>(cmc);
            if (obbModel)
            {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODELS
                sout << "    it IS." << sendl;
#endif
                addCollisionModel(cmc);
                obbModelFound = true;
                break;
            }
            else
            {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODELS
                sout << "    it IS NOT." << sendl;
#endif
            }
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODELS
            sout << "   examine " << cmc->getName() << ", type " << cmc->getTypeName() << " if it's a ObbTreeCPUCollisionModel" << sendl;
#endif
            ObbTreeCPUCollisionModel<Vec3Types>* pqpModel = dynamic_cast<ObbTreeCPUCollisionModel<Vec3Types>*>(cmc);
            if (pqpModel)
            {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODELS
                sout << "    it IS." << sendl;
#endif
                addCollisionModel(cmc);
                pqpModelFound = true;
                break;
            }
            else
            {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODELS
                sout << "    it IS NOT." << sendl;
#endif
            }
            cmc = cmc->getNext();
        } while (cmc != NULL);

        if (!obbModelFound && !pqpModelFound)
        {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODELS
            sout << "No ObbTreeGPUCollisionModel found in hierarchy starting at " << (*it)->getName() << ", falling back to BruteForceDetection" << sendl;
#endif
            BruteForceDetection::addCollisionModel((*it));
        }
    }
}

#define OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
void ObbTreeGPUCollisionDetection::addCollisionModel(core::CollisionModel *cm)
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
	std::cout << "ObbTreeGPUCollisionDetection::addCollisionModel(" << cm->getName() << "), type = " << cm->getTypeName() << std::endl;
#endif

    if (!this->f_printLog.getValue())
        this->f_printLog.setValue(true);

    if (!cm)
        return;


#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
    sout << "ObbTreeGPUCollisionDetection::addCollisionModel(" << cm->getName() << "), type = " << cm->getTypeName() << sendl;
#endif

    ObbTreeGPUCollisionModel<Vec3Types>* obbModel = dynamic_cast<ObbTreeGPUCollisionModel<Vec3Types>*>(cm);

    ObbTreeCPUCollisionModel<Vec3Types>* pqpModel = dynamic_cast<ObbTreeCPUCollisionModel<Vec3Types>*>(cm);

    if (obbModel)
    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
        sout << "  obbModel = " << obbModel->getName() << " of type " << obbModel->getTypeName() << sendl;
#endif
        bool doGPUObbTest = true;
        if (cm->isSimulated() && cm->getLast()->canCollideWith(cm->getLast()))
        {
            // self collision

#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
			std::cout << " Test for self-collision ability in broad-phase: " << cm->getLast()->getName() << std::endl;
#endif
            bool swapModels = false;
            core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm, cm, swapModels);
            if (intersector != NULL)
            {
                if (intersector->canIntersect(cm->begin(), cm->begin()))
				{
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
					std::cout << " Self-collision capable: " << cm->getLast()->getName() << std::endl;
#endif
                    cmPairs.push_back(std::make_pair(cm, cm));
                }
            }
        }
        for (sofa::helper::vector<core::CollisionModel*>::iterator it = collisionModels.begin(); it != collisionModels.end(); ++it)
        {
            core::CollisionModel* cm2 = *it;

            if (!cm->isSimulated() && !cm2->isSimulated())
			{
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
				std::cout << " simulated cm = " << cm->getName() << ": " << cm->isSimulated() << ", cm2 =  " << cm2->getName() << ": " << cm2->isSimulated() << std::endl;
#endif
                continue;
            }

            // bad idea for sofa standard models. If this define is set, Bolzen/Bohrung scenario detects a contact within one mesh and crashes after the first 'real' contacts are detected.
            if (!keepCollisionBetween(cm->getLast(), cm2->getLast()))
			{
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
				std::cout << " collisions between cm = " << cm->getLast()->getName() << " and " << cm2->getLast()->getName() << " not kept!" << sendl;
#endif
                continue;
            }

            bool swapModels = false;
            core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm, cm2, swapModels);
            if (intersector == NULL)
			{
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
				std::cout << " no suitable intersector between cm = " << cm->getName() << " and " << cm2->getName() << " found!" << std::endl;
#endif
                continue;
            }
            core::CollisionModel* cm1 = (swapModels?cm2:cm);
            cm2 = (swapModels?cm:cm2);

            // Here we assume a single root element is present in both models

#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
			std::cout << " Intersector used for intersectability query: " << intersector->name() << std::endl;
#endif
            if (intersector->canIntersect(cm1->begin(), cm2->begin()))
			{
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
				std::cout << "Broad phase " << cm1->getLast()->getName() << " - " << cm2->getLast()->getName() << std::endl;
#endif
                cmPairs.push_back(std::make_pair(cm1, cm2));
            }
            else
			{
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
				std::cout << " cm1 = " << cm->getName() << " and cm2 = " << cm2->getName() << " can't intersect!" << std::endl;
#endif
                doGPUObbTest = false;
            }
        }

        collisionModels.push_back(cm);

        if (doGPUObbTest)
        {
            std::map<std::string, ObbTreeGPUCollisionModel<Vec3Types>*>::const_iterator mit = m_obbModels.find(obbModel->getName());

            if (mit == m_obbModels.end())
            {
    #ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
                sout << "   registering OBB model " << obbModel->getName() << sendl;
    #endif
                m_obbModels.insert(std::make_pair(obbModel->getName(), obbModel));
            }

            /*for (std::map<std::string, ObbTreeGPUCollisionModel<Vec3Types>*>::iterator it = m_obbModels.begin(); it != m_obbModels.end(); it++)
            {
                ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = it->second;
    #ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
                sout << "   check possible intersection with OBB model " << obbModel2->getName() << sendl;
    #endif
                if (obbModel2->getName() == obbModel->getName())
                {
    #ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
                    sout << "     no self-intersection between OBB-Models supported!" << sendl;
    #endif
                    continue;
                }

                if (obbModel->canCollideWith(obbModel2))
                {
    #ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
                    sout << "    canCollideWith " << obbModel2->getName() << ": OK." << sendl;
    #endif
                    cmPairs.push_back(std::make_pair(obbModel, obbModel2));
                }
    #ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
                else
                {
                    sout << "    canCollideWith " << obbModel2->getName() << ": FAIL." << sendl;
                }
    #endif
            }*/
        }
    }
    else if (pqpModel)
    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
        sout << "  pqpModel = " << pqpModel->getName() << " of type " << pqpModel->getTypeName() << sendl;
#endif
        std::map<std::string, ObbTreeCPUCollisionModel<Vec3Types>*>::const_iterator mit = m_pqpModels.find(pqpModel->getName());

        if (mit == m_pqpModels.end())
        {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
            sout << "   registering PQP model " << pqpModel->getName() << sendl;
#endif
            m_pqpModels.insert(std::make_pair(pqpModel->getName(), pqpModel));
        }

        for (std::map<std::string, ObbTreeCPUCollisionModel<Vec3Types>*>::iterator it = m_pqpModels.begin(); it != m_pqpModels.end(); it++)
        {
            ObbTreeCPUCollisionModel<Vec3Types>* pqpModel2 = it->second;
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
            sout << "   check possible intersection with PQP model " << pqpModel2->getName() << sendl;
#endif
//            if (!pqpModel2->isSimulated())
//            {
//                sout << "     not simulated, aborting" << sendl;
//                continue;
//            }

            if (pqpModel2->getName() == pqpModel->getName())
            {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
                sout << "     no self-intersection between PQP-Models supported!" << sendl;
#endif
                continue;
            }

            if (pqpModel->canCollideWith(pqpModel2))
            {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
                sout << "    canCollideWith " << pqpModel2->getName() << ": OK." << sendl;
#endif
                cmPairs.push_back(std::make_pair(pqpModel, pqpModel2));
            }
            else
            {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
                sout << "    canCollideWith " << pqpModel2->getName() << ": FAIL." << sendl;
#endif
            }
        }
    }
    else
    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
        sout << "Model " << cm->getName() << " is not a ObbTreeGPU model, fallback to BruteForceDetection" << sendl;
#endif
        BruteForceDetection::addCollisionModel(cm);
    }
}

//#define OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONPAIRS
void ObbTreeGPUCollisionDetection::addCollisionPairs(const sofa::helper::vector<std::pair<core::CollisionModel *, core::CollisionModel *> > &v)
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONPAIRS
    sout << "=== ObbTreeGPUCollisionDetection::addCollisionPairs(): " << v.size() << " possible pairs. ===" << sendl;
    int addedPairs = 0;
#endif
	for (sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >::const_iterator it = v.begin(); it != v.end(); it++)
	{
        addCollisionPair(*it);

#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONPAIRS
        sout << " Add: " << it->first->getName() << " -- " << it->second->getName() << sendl;
        addedPairs++;
#endif

    }
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONPAIRS
    sout << "=== ObbTreeGPUCollisionDetection::addCollisionPairs(): " << addedPairs << " pairs added. ===" << sendl;
#endif
}

#include "ObbTree.h"

//#define OBBTREE_GPU_DEBUG_TRANSFORMED_VERTICES
//#define OBBTREE_GPU_COLLISION_DETECTION_DUMP_INTERSECTING_TRIANGLES
#define OBBTREE_GPU_COLLISION_DETECTION_DEBUG
void ObbTreeGPUCollisionDetection::addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*> &cmPair)
{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
    sout << "ObbTreeGPUCollisionDetection::addCollisionPair(" << cmPair.first->getName() << "," << cmPair.second->getName() << ")" << sendl;
    sout << " model types: " << cmPair.first->getTypeName() << " - " << cmPair.second->getTypeName() << sendl;
#endif

    ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = dynamic_cast<ObbTreeGPUCollisionModel<Vec3Types>*>(cmPair.first);
    ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = dynamic_cast<ObbTreeGPUCollisionModel<Vec3Types>*>(cmPair.second);

    ObbTreeCPUCollisionModel<Vec3Types>* pqpModel1 = dynamic_cast<ObbTreeCPUCollisionModel<Vec3Types>*>(cmPair.first);
    ObbTreeCPUCollisionModel<Vec3Types>* pqpModel2 = dynamic_cast<ObbTreeCPUCollisionModel<Vec3Types>*>(cmPair.second);

    if (obbModel1 && obbModel2)
    {
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
        sout << " check using GPU-based implementation" << sendl;
#endif
        std::pair<std::string, std::string> pairCombo1 = std::make_pair(obbModel1->getName(), obbModel2->getName());
        std::pair<std::string, std::string> pairCombo2 = std::make_pair(obbModel2->getName(), obbModel1->getName());

        bool combo1Found = false;
        bool combo2Found = false;

        bool combo1Used = false;
        bool combo2Used = false;

        for (std::vector<std::pair<std::string, std::string> >::const_iterator it = m_testedModelPairs.begin(); it != m_testedModelPairs.end(); it++)
        {
            if (it->first.compare(pairCombo1.first) == 0 && it->second.compare(pairCombo1.second) == 0)
                combo1Found = true;

            if (it->first.compare(pairCombo2.first) == 0 && it->second.compare(pairCombo2.second) == 0)
                combo2Found = true;
        }

        if (!combo1Found)
        {
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
            std::cout << " not tested yet: combo1 = " << pairCombo1.first << " -- " << pairCombo1.second << std::endl;
#endif
            m_testedModelPairs.push_back(pairCombo1);
            combo1Used = true;
        }
        else
        {
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
            std::cout << " already tested: combo1 = " << pairCombo1.first << " -- " << pairCombo1.second << std::endl;
#endif
            return;
        }

        if (!combo2Found)
        {
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
            std::cout << " not tested yet: combo2 = " << pairCombo2.first << " -- " << pairCombo2.second << std::endl;
#endif
            combo2Used = true;
            m_testedModelPairs.push_back(pairCombo2);
        }
        else
        {
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
            std::cout << " already tested: combo2 = " << pairCombo2.first << " -- " << pairCombo2.second << std::endl;
#endif
            return;
        }

        std::vector<std::pair<int,int> > allPairs;
        int* allOBBPairs = NULL;

        struct OBBContainer obbTree1;
        struct OBBContainer obbTree2;

        obbTree1.nVerts = obbModel1->numVertices();
        obbTree2.nVerts = obbModel2->numVertices();
        obbTree1.nTris = obbModel1->numTriangles();
        obbTree2.nTris = obbModel2->numTriangles();
        obbTree1.nBVs = obbModel1->numOBBs();
        obbTree2.nBVs = obbModel2->numOBBs();

        obbTree1.obbTree = obbModel1->obbTree_device();
        obbTree2.obbTree = obbModel2->obbTree_device();
        obbTree1.vertexPointer = obbModel1->vertexPointer_device();
        obbTree2.vertexPointer = obbModel2->vertexPointer_device();

        obbTree1.vertexTfPointer = obbModel1->vertexTfPointer_device();
        obbTree2.vertexTfPointer = obbModel2->vertexTfPointer_device();

        obbTree1.triIdxPointer = obbModel1->triIndexPointer_device();
        obbTree2.triIdxPointer = obbModel2->triIndexPointer_device();

        if (_useStreamedCollisionQueries.getValue())
        {
    #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
            std::cout << "USE STREAMED COLLISION QUERIES" << std::endl;
    #endif

            OBBModelContainer obbContainer1, obbContainer2;
            obbContainer1._obbContainer = obbTree1;
            obbContainer1._obbCollisionModel = obbModel1;
            obbContainer2._obbContainer = obbTree2;
            obbContainer2._obbCollisionModel = obbModel2;

    #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
            std::cout << " combo1Found = " << combo1Found << ", combo1Used = " << combo1Used << ";" <<
                         " combo2Found = " << combo2Found << ", combo2Used = " << combo2Used << std::endl;
    #endif


            if (combo1Used && combo2Used)
            {
                std::cout << "  push combo1; no combo registered yet = " << obbModel1->getName() << " -- " << obbModel2->getName() << std::endl;
                _narrowPhasePairs.push_back(std::make_pair(obbContainer1, obbContainer2));
            }
            else if (combo1Used && !combo2Used)
            {
                std::cout << "  push combo1 = " << obbModel1->getName() << " -- " << obbModel2->getName() << std::endl;
                _narrowPhasePairs.push_back(std::make_pair(obbContainer1, obbContainer2));
            }
            else if (combo2Used && !combo1Used)
            {
                std::cout << "  push combo2 = " << obbModel2->getName() << " -- " << obbModel1->getName() << std::endl;
                _narrowPhasePairs.push_back(std::make_pair(obbContainer2, obbContainer1));
            }
            else
            {
                std::cout << "  WARNING -- combo1/2 used flags not set, skipping: " << obbModel1->getName() << " -- " << obbModel2->getName() << std::endl;
            }
        }
        else
        {
    #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
            std::cout << "USE FULL ALLOCATION ON GPU" << std::endl;
    #endif
            Vector3 modelPos1, modelPos2;
            Matrix3 modelOri1, modelOri2;

            modelPos1 = obbModel1->getCachedPosition();
            modelPos2 = obbModel2->getCachedPosition();

            Quaternion modelQuat1 = obbModel1->getCachedOrientation();
            Quaternion modelQuat2 = obbModel2->getCachedOrientation();
            modelQuat1.toMatrix(modelOri1);
            modelQuat2.toMatrix(modelOri2);

    #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
            std::cout << " obbModel1 position from MechanicalObject = " << modelPos1 << ", orientation = " << modelOri1 << std::endl;
            std::cout << " obbModel2 position from MechanicalObject = " << modelPos2 << ", orientation = " << modelOri2 << std::endl;
    #endif

            obbTree1.modelTransform.m_T[0] = modelPos1.x(); obbTree1.modelTransform.m_T[1] = modelPos1.y(); obbTree1.modelTransform.m_T[2] = modelPos1.z();
            obbTree2.modelTransform.m_T[0] = modelPos2.x(); obbTree2.modelTransform.m_T[1] = modelPos2.y(); obbTree2.modelTransform.m_T[2] = modelPos2.z();

            for (short k = 0; k < 3; k++)
            {
                for (short l = 0; l < 3; l++)
                {
                    obbTree1.modelTransform.m_R[k][l] = modelOri1(k,l);
                    obbTree2.modelTransform.m_R[k][l] = modelOri2(k,l);
                }
            }

            gProximityDetectionOutput* detectedContacts = NULL;
            int nContacts = 0;
            int nIntersecting = 0;
    #ifdef OBBTREE_GPU_COLLISION_DETECTION_FULL_ALLOCATION_DETECTION
            {

        #ifdef OBBTREE_GPU_COLLISION_DETECTION_RECORD_INTERSECTING_OBBS
                int nIntersectingOBBs = 0;
        #endif
        #ifdef OBBTREE_GPU_DEBUG_TRANSFORMED_VERTICES
                GPUVertex* tfVertices1 = new GPUVertex[obbTree1.nVerts];
                GPUVertex* tfVertices2 = new GPUVertex[obbTree2.nVerts];
        #endif


        #ifdef OBBTREE_GPU_DEBUG_TRANSFORMED_VERTICES
            std::cout << " Non-Streams intersecting = " << nIntersecting << std::endl;
        #endif
        #ifdef SOFA_DUMP_VISITOR_INFO
                std::stringstream visitorId;
                visitorId << "ObbTreeGPU_BVHCollide(" << obbModel1->getName() << " - " << obbModel2->getName() << ")";
                simulation::Visitor::printNode("ObbTreeGPU_BVHCOLLIDE");
                sofa::helper::AdvancedTimer::stepBegin(visitorId.str().c_str());
        #endif

                nIntersecting = 0;
                ObbTreeGPU_BVHCollide(&obbTree1, &obbTree2, allPairs
    #ifdef OBBTREE_GPU_COLLISION_DETECTION_RECORD_INTERSECTING_OBBS
                    , &allOBBPairs, &nIntersectingOBBs
    #endif
    #ifdef OBBTREE_GPU_DEBUG_TRANSFORMED_VERTICES
                    , (void**) &tfVertices1, (void**) &tfVertices2
    #endif
                    , &detectedContacts
                    , &nContacts
                    , m_alarmDistance
                    , m_contactDistance
                    , nIntersecting
                );

                std::cout << " Non-Streams intersecting = " << nIntersecting << std::endl;

        #ifdef SOFA_DUMP_VISITOR_INFO
                simulation::Visitor::printCloseNode("ObbTreeGPU_BVHCOLLIDE");
                sofa::helper::AdvancedTimer::stepEnd(visitorId.str().c_str());
        #endif //OBBTREE_GPU_COLLISION_DETECTION_FULL_ALLOCATION_DETECTION
            }
    #endif

    #ifdef OBBTREE_GPU_COLLISION_DETECTION_FULL_ALLOCATION_DETECTION
            {
        #ifdef OBBTREE_GPU_DEBUG_TRANSFORMED_VERTICES
                std::string collisionPairId(obbModel1->getName() + "::" + obbModel2->getName());
                if (m_transformedVertices.find(collisionPairId) == m_transformedVertices.end())
                {
                    m_transformedVertices.insert(std::make_pair(collisionPairId, std::make_pair(tfVertices1, tfVertices2)));
                }
                else
                {
                    delete[] m_transformedVertices[collisionPairId].first;
                    delete[] m_transformedVertices[collisionPairId].second;

                    m_transformedVertices[collisionPairId] = std::make_pair(tfVertices1, tfVertices2);
                }
        #endif
                if (nIntersecting > 0)
                {
        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DUMP_INTERSECTING_TRIANGLES
                    std::cout << "Intersecting triangle pairs: " << nIntersecting << " -- ";
        #endif
        #if 0
                    std::vector<std::pair<int, int> > intersectingTriIndices;

                    for (int k = 0; k < nIntersecting; k++)
                    {
                        intersectingTriIndices.push_back(std::make_pair(std::abs(allPairs[k].first),std::abs(allPairs[k].second)));
        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DUMP_INTERSECTING_TRIANGLES
                        std::cout << std::abs(allPairs[k].first) << " - ";
                        std::cout << std::abs(allPairs[k].second) << "; ";
        #endif
                    }
        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DUMP_INTERSECTING_TRIANGLES
                    std::cout << std::endl;
        #endif
                    m_intersectingTriangles.insert(std::make_pair(collisionPairId, intersectingTriIndices));
        #endif
                    //if (nContacts > 0)
                    {
        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                        std::cout << "=== Contacts recorded: " << nContacts << " ===" << std::endl;
        #endif

                        sofa::core::collision::DetectionOutputVector*& outputs = getDetectionOutputs(obbModel1, obbModel2);
                        sofa::core::collision::TDetectionOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >* discreteOutputs =
                        m_intersection->getOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, obbModel2, outputs);

        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                        std::cout << " outputs = " << outputs << ", discreteOutputs = " << discreteOutputs << std::endl;
        #endif

                        if (discreteOutputs == NULL)
                        {
        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                            std::cout << " no discreteOutputs pointer exists yet; creating output vector via m_intersection" << std::endl;
        #endif

                            discreteOutputs = m_intersection->createOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, obbModel2);
                            if (outputs == NULL)
                            {
        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                                std::cout << " dynamic_cast'ing discreteOutputs for registration in DetectionOutputMap" << std::endl;
        #endif

                                outputs = dynamic_cast<sofa::core::collision::DetectionOutputVector*>(discreteOutputs);
                            }
                        }

                        const double maxContactDist = m_alarmDistance + (m_alarmDistance - m_contactDistance);
                        const double maxContactDist2 = maxContactDist * maxContactDist;

        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                        std::cout << "=== Contact points generated: " << nContacts << " ===" << std::endl;
        #endif

                        for (int k = 0; k < nContacts; k++)
                        {
                            Vector3 contactNormal(detectedContacts->normal[k].x, detectedContacts->normal[k].y, detectedContacts->normal[k].z);

        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                            std::cout << " -> " << k << ": contactNormal = " << contactNormal << ", contactNormal.norm() = " << contactNormal.norm() << "; distance = " << detectedContacts->distance[k] << ", contactDistance = " << m_contactDistance << std::endl;
        #endif

                            if (contactNormal.norm() >= 1e-06)
                            {
                                if (contactNormal.norm2() <= maxContactDist2 &&
                                    std::fabs(detectedContacts->distance[k] - m_contactDistance) < m_contactDistance)
                                {
                                    discreteOutputs->resize(discreteOutputs->size()+1);
                                    sofa::core::collision::DetectionOutput *detection = &*(discreteOutputs->end()-1);

                                    detection->id = detectedContacts->contactId[k];

                                    // Swapped: Testing...: This is legitimate???
                                    /*if (detectedContacts->contactType[k] == COLLISION_LINE_LINE)
                                    {
                                        detection->point[1] = Vector3(detectedContacts->point0[k].x, detectedContacts->point0[k].y, detectedContacts->point0[k].z);
                                        detection->point[0] = Vector3(detectedContacts->point1[k].x, detectedContacts->point1[k].y, detectedContacts->point1[k].z);
                                    }
                                    else if (detectedContacts->contactType[k] == COLLISION_VERTEX_FACE ||
                                             detectedContacts->contactType[k] == COLLISION_LINE_POINT)
                                    {
                                        detection->point[0] = Vector3(detectedContacts->point0[k].x, detectedContacts->point0[k].y, detectedContacts->point0[k].z);
                                        detection->point[1] = Vector3(detectedContacts->point1[k].x, detectedContacts->point1[k].y, detectedContacts->point1[k].z);
                                    }*/

                                    detection->point[0] = Vector3(detectedContacts->point0[k].x, detectedContacts->point0[k].y, detectedContacts->point0[k].z);
                                    detection->point[1] = Vector3(detectedContacts->point1[k].x, detectedContacts->point1[k].y, detectedContacts->point1[k].z);

                                    // Negated: Testing...: THIS WAS A FACEPALM type FAILURE!!!
                                    /*if (detectedContacts->contactType[k] == COLLISION_VERTEX_FACE)
                                        detection->normal = -contactNormal;
                                    else
                                        detection->normal = contactNormal;*/

                                    detection->normal = contactNormal;

                                    // Minus contact distance: Testing...
                                    detection->value = detection->normal.norm();
                                    detection->normal /= detection->value;

                                    detection->value -= m_contactDistance;

                                    detection->contactType = (sofa::core::collision::DetectionOutputContactType) detectedContacts->contactType[k];

        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                                    std::cout << " * " << k << " data: contact id = " << detection->id << ",";
                                    std::cout << " type = " << detection->contactType << " = ";
                                    if (detection->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                                        std::cout << "LINE_LINE: tri1/edge1 = " << detectedContacts->elems[k].w << "/" << detectedContacts->elems[k].y << " -- tri2/edge2 = " << detectedContacts->elems[k].x << "/" << detectedContacts->elems[k].z;
                                    else if (detection->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                                        std::cout << "FACE_VERTEX: tri1 = " << detectedContacts->elems[k].w << ", tri2 = " << detectedContacts->elems[k].x;
                                    else if (detection->contactType == sofa::core::collision::CONTACT_LINE_VERTEX)
                                        std::cout << "LINE_VERTEX: tri1 = " << detectedContacts->elems[k].w << ", tri2 = " << detectedContacts->elems[k].x;

                                    std::cout << "; distance = " << detection->value << "; point0 = " << detection->point[0] << ", point1 = " << detection->point[1] << ", normal = " << detection->normal << std::endl;
        #endif
                                    if (detectedContacts->contactType[k] == COLLISION_LINE_LINE)
                                    {
        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                                        std::cout << "   edge0 index = " << detectedContacts->elems[k].w * 3 + detectedContacts->elems[k].y << " (" << detectedContacts->elems[k].w  << " * 3 + " << detectedContacts->elems[k].y << ")" << std::endl;
        #endif
                                        detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, detectedContacts->elems[k].w * 3 + detectedContacts->elems[k].y); // << CollisionElementIterator

        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                                        std::cout << "   edge1 index = " << detectedContacts->elems[k].x * 3 + detectedContacts->elems[k].z << " (" << detectedContacts->elems[k].x  << " * 3 + " << detectedContacts->elems[k].z << ")" << std::endl;
        #endif

                                        detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, detectedContacts->elems[k].x * 3 + detectedContacts->elems[k].z); // << CollisionElementIterator

                                        detection->elemFeatures.first = detectedContacts->elems[k].y;
                                        detection->elemFeatures.second = detectedContacts->elems[k].z;
                                    }
                                    else if (detectedContacts->contactType[k] == COLLISION_VERTEX_FACE)
                                    {
        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                                        std::cout << "   tri0 = " << detectedContacts->elems[k].w << ", tri1 = " << detectedContacts->elems[k].x << ", vertex0 = " << detectedContacts->elems[k].y << ", vertex1 = " << detectedContacts->elems[k].z << std::endl;
        #endif

                                        if (detectedContacts->elems[k].z == -1)
                                        {
        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                                            std::cout << "   contact with first triangle: elem.first = " << detectedContacts->elems[k].x * 3 << ", elem.second = " << detectedContacts->elems[k].w * 3 + detectedContacts->elems[k].y << std::endl;
        #endif

                                            detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, detectedContacts->elems[k].x * 3); // << CollisionElementIterator
                                            detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, detectedContacts->elems[k].w * 3 + detectedContacts->elems[k].y); // << CollisionElementIterator
                                            detection->elemFeatures.first = detectedContacts->elems[k].y;
                                        }
                                        else if (detectedContacts->elems[k].y == -1)
                                        {

        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                                            std::cout << "   contact with second triangle: elem.first = " << detectedContacts->elems[k].x * 3 << ", elem.second = " << detectedContacts->elems[k].w * 3 + detectedContacts->elems[k].y << std::endl;
        #endif

                                            detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, detectedContacts->elems[k].x * 3); // << CollisionElementIterator
                                            detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, detectedContacts->elems[k].w * 3 + detectedContacts->elems[k].z); // << CollisionElementIterator
                                            detection->elemFeatures.second = detectedContacts->elems[k].z;
                                        }
                                    }
                                    else if (detectedContacts->contactType[k] == COLLISION_LINE_POINT)
                                    {
                                        detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, detectedContacts->elems[k].w * 3 + detectedContacts->elems[k].y); // << CollisionElementIterator
                                        detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, detectedContacts->elems[k].x * 3 + detectedContacts->elems[k].z); // << CollisionElementIterator
                                        detection->elemFeatures.first = detectedContacts->elems[k].y;
                                        detection->elemFeatures.second = detectedContacts->elems[k].z;
                                    }
                                }
                            }
                        }

                        delete[] detectedContacts->contactId;
                        detectedContacts->contactId = NULL;
                        delete[] detectedContacts->contactType;
                        detectedContacts->contactType = NULL;
                        delete[] detectedContacts->distance;
                        detectedContacts->distance = NULL;
                        delete[] detectedContacts->elems;
                        detectedContacts->elems = NULL;
                        delete[] detectedContacts->normal;
                        detectedContacts->normal = NULL;
                        delete[] detectedContacts->point0;
                        detectedContacts->point0 = NULL;
                        delete[] detectedContacts->point1;
                        detectedContacts->point1 = NULL;
                        delete[] detectedContacts->valid;
                        detectedContacts->valid = NULL;

                        delete detectedContacts;
                        detectedContacts = NULL;

                        std::sort(discreteOutputs->begin(), discreteOutputs->end(), contactTypeCompare);

        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                        std::cout << "=== after sorting by contact type ===" << std::endl;
                        for (sofa::core::collision::TDetectionOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >::const_iterator it = discreteOutputs->begin(); it != discreteOutputs->end(); it++)
                        {
                            std::cout << " - " << (*it).contactType << ", point0 = " << (*it).point[0] << " -- " << (*it).point[1] << std::endl;
                        }
        #endif
                        std::pair< core::CollisionModel*, core::CollisionModel* > cm_pair = std::make_pair(obbModel1, obbModel2);

                        DetectionOutputMap::iterator it = getDetectionOutputs().find(cm_pair);

                        if (it == getDetectionOutputs().end())
                        {
                            getDetectionOutputs().insert(std::make_pair(cm_pair, outputs));
                        }
                        else
                        {
                            getDetectionOutputs()[cm_pair] = outputs;
                        }
        #ifdef OBBTREE_GPU_COLLISION_DETECTION_DUMP_CONTACT_POINTS
                        std::cout << " after filling contacts vector: " << std::endl;
                        std::cout << getDetectionOutputs()[cm_pair]->size() << " elements created." << std::endl;
                        for (sofa::core::collision::TDetectionOutputVector<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> >::const_iterator it = discreteOutputs->begin(); it != discreteOutputs->end(); it++)
                        {
                            std::cout << " - Elements: " << it->elem.first.getIndex() << " -- " << it->elem.second.getIndex() << "; distance = " << it->value << "; id = " << it->id << std::endl;
                        }
        #endif
                    }
                }
            }
    #ifdef OBBTREE_GPU_COLLISION_DETECTION_RECORD_INTERSECTING_OBBS
            if (nIntersectingOBBs > 0)
            {
    #ifdef OBBTREE_GPU_COLLISION_DETECTION_DUMP_INTERSECTING_OBBS
                std::cout << "Intersecting OBB pairs: " << nIntersectingOBBs << " -- ";
    #endif
                std::vector<std::pair<int, int> > intersectingOBBIndices;
                for (int k = 0; k < 2 * nIntersectingOBBs; k++)
                {
                    if (k % 2 != 0)
                    {
    #ifdef OBBTREE_GPU_COLLISION_DETECTION_DUMP_INTERSECTING_OBBS
                        std::cout << allOBBPairs[k-1] << "," << allOBBPairs[k] << ";";
    #endif
                        intersectingOBBIndices.push_back(std::make_pair(allOBBPairs[k-1],allOBBPairs[k]));
                    }
                }
                m_intersectingOBBs.insert(std::make_pair(collisionPairId, intersectingOBBIndices));
    #ifdef OBBTREE_GPU_COLLISION_DETECTION_DUMP_INTERSECTING_OBBS
                std::cout << std::endl;
    #endif
            }
            delete[] allOBBPairs;
    #endif

    #endif //OBBTREE_GPU_COLLISION_DETECTION_FULL_ALLOCATION_DETECTION
    }
    }
    else if (pqpModel1 && pqpModel2)
    {
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
         std::cout << "Check using CPU-based implementation" << std::endl;
#endif

         CollideResult colResult;
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
         std::cout << "=== TEST tree1 against tree2 ===" << std::endl;
#endif
         pqpModel1->getObbTree().testOverlap(pqpModel2->getObbTree(), colResult);

#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
         std::cout << "=== RESULTS tree1 against tree2 ===" << std::endl;
         std::cout << " OBB tests done: " << colResult.num_bv_tests << "; tri-box tests done: " << colResult.num_tri_box_tests << ", tri-tri tests done: " << colResult.num_tri_tests << std::endl;

         std::cout << " overlapping obb pairs: " << colResult.num_obb_pairs << std::endl;
         for (int k = 0; k < colResult.NumOBBPairs(); k++)
         {
             std::cout << " - " << colResult.obb_pairs[k].id1 << " -- " << colResult.obb_pairs[k].id2 << std::endl;
         }

         std::cout << " overlapping tri pairs: " << colResult.num_pairs << std::endl;
         for (int k = 0; k < colResult.NumPairs(); k++)
         {
             std::cout << " - " << colResult.pairs[k].id1 << " -- " << colResult.pairs[k].id2 << std::endl;
         }
#endif

         std::string collisionPairId(pqpModel1->getName() + "::" + pqpModel2->getName());

         if (colResult.NumPairs() > 0)
         {
             std::vector<std::pair<int, int> > intersectingTriIndices;
             for (int k = 0; k < colResult.NumPairs(); k++)
             {
                 intersectingTriIndices.push_back(std::make_pair(std::abs(colResult.pairs[k].id1), std::abs(colResult.pairs[k].id2)));
             }
             m_intersectingTriangles.insert(std::make_pair(collisionPairId, intersectingTriIndices));
         }

         if (colResult.NumOBBPairs() > 0)
         {
             std::vector<std::pair<int, int> > intersectingOBBIndices;
             for (int k = 0; k < colResult.NumOBBPairs(); k++)
             {
                 intersectingOBBIndices.push_back(std::make_pair(colResult.obb_pairs[k].id1, colResult.obb_pairs[k].id2));
             }
             m_intersectingOBBs.insert(std::make_pair(collisionPairId, intersectingOBBIndices));
         }

         if (colResult.NumOBBPairs() > 0)
         {
             std::vector<int> emphasizeOBBs1, emphasizeOBBs2;
             for (int k = 0; k < colResult.NumOBBPairs(); k++)
             {
                 emphasizeOBBs1.push_back(colResult.obb_pairs[k].id1);
                 emphasizeOBBs2.push_back(colResult.obb_pairs[k].id2);
             }

             pqpModel1->setEmphasizedIndices(emphasizeOBBs1);
             pqpModel2->setEmphasizedIndices(emphasizeOBBs2);
         }
    }
    else
    {
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
        sout << " check using default BruteForceDetection implementation" << sendl;
#endif
        BruteForceDetection::addCollisionPair(cmPair);
    }
}

#include "BVHDrawHelpers.h"

//#define OBBTREE_GPU_COLLISION_DETECTION_DEBUG_OBBS
//#define OBBTREE_GPU_COLLISION_DETECTION_DEBUG_TRIANGLES
//#define OBBTREE_GPU_COLLISION_DETECTION_DEBUG_TRANSFORMED_VERTICES

void ObbTreeGPUCollisionDetection::draw(const core::visual::VisualParams *vparams)
{
    BruteForceDetection::draw(vparams);

    //std::cout << " draw intersecting triangles: " << m_intersectingTriangles.size() << std::endl;
    //std::cout << " draw intersecting obbs     : " << m_intersectingOBBs.size() << std::endl;

#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG_OBBS
    if (m_intersectingOBBs.size() > 0)
    {
        for (std::map<std::string, std::vector<std::pair<int,int> > >::const_iterator it = m_intersectingOBBs.begin();
             it != m_intersectingOBBs.end(); it++)
        {
            std::string modelName1, modelName2;
            const std::string& modelPairId = it->first;

            //std::cout << " pair: " << modelPairId << std::endl;

            std::vector<std::string> v;
            boost::algorithm::iter_split(v, modelPairId, boost::algorithm::first_finder("::"));

            if (v.size() == 2)
            {
                modelName1 = v.at(0);
                modelName2 = v.at(1);

                ObbTreeGPUCollisionModel<Vec3Types>* model1 = m_obbModels[modelName1];
                ObbTreeGPUCollisionModel<Vec3Types>* model2 = m_obbModels[modelName2];

                //std::cout << " model1 ptr = " << model1 << ", model2 ptr = " << model2 << std::endl;
                if (model1 && model2)
                {
                    Vector3 model1Pos, model2Pos;
                    Matrix3 model1Ori, model2Ori;
                    model1->getPosition(model1Pos);
                    model2->getPosition(model2Pos);
                    model1->getOrientation(model1Ori);
                    model2->getOrientation(model2Ori);

                    Matrix4 model1Mat; model1Mat.identity();
                    Matrix4 model2Mat; model2Mat.identity();

                    for (short k = 0; k < 3; k++)
                    {
                        for (short l = 0; l < 3; l++)
                        {
                            model1Mat(k,l) = model1Ori(k,l);
                            model2Mat(k,l) = model2Ori(k,l);
                        }
                    }
                    std::vector<std::pair<int,int> >& overlappingOBBs = m_intersectingOBBs[modelPairId];

                    //std::cout << " model1 = " << modelName1 << ", model2 = " << modelName2 << "; pairs = " << overlappingOBBs.size() << std::endl;

                    glPushMatrix();
                    glPushAttrib(GL_ENABLE_BIT);
                    glEnable(GL_COLOR_MATERIAL);

                    for (std::vector<std::pair<int, int> >::const_iterator obb_it = overlappingOBBs.begin();
                         obb_it != overlappingOBBs.end(); obb_it++)
                    {
                        BV* obb1 = model1->getPqpModel()->child(obb_it->first);
                        BV* obb2 = model2->getPqpModel()->child(obb_it->second);

                        glTranslated(model1Pos.x(), model1Pos.y(), model1Pos.z());
                        glMultMatrixd(model1Mat.transposed().ptr());

                        glTranslated(obb1->To[0], obb1->To[1], obb1->To[2]);

                        Matrix4 obb1Ori; obb1Ori.identity();
                        for (short k = 0; k < 3; k++)
                            for (short l = 0; l < 3; l++)
                                obb1Ori(k,l) = obb1->R[k][l];

                        glMultMatrixd(obb1Ori.transposed().ptr());

                        Vec4f obbColor1(0,1,0,0.75);
                        if (obb1->Leaf())
                            obbColor1 = Vec4f(1,0,1,0.75f);

                        Vector3 obb1He(obb1->d[0],obb1->d[1],obb1->d[2]);
                        BVHDrawHelpers::drawCoordinateMarkerGL(0.5f, 0.25f);
                        BVHDrawHelpers::drawObbVolume(obb1He, obbColor1, true);

                        glMultMatrixd(obb1Ori.ptr());
                        glTranslated(-obb1->To[0], -obb1->To[1], -obb1->To[2]);
                        glMultMatrixd(model1Mat.ptr());
                        glTranslated(-model1Pos.x(), -model1Pos.y(), -model1Pos.z());

                        glTranslated(model2Pos.x(), model2Pos.y(), model2Pos.z());
                        glMultMatrixd(model2Mat.transposed().ptr());
                        glTranslated(obb2->To[0], obb2->To[1], obb2->To[2]);

                        Matrix4 obb2Ori; obb2Ori.identity();
                        for (short k = 0; k < 3; k++)
                            for (short l = 0; l < 3; l++)
                                obb2Ori(k,l) = obb2->R[k][l];

                        glMultMatrixd(obb2Ori.transposed().ptr());

                        Vec4f obbColor2(1,0,0,0.75);
                        if (obb2->Leaf())
                            obbColor2 = Vec4f(1,1,0,0.75f);

                        Vector3 obb2He(obb2->d[0],obb2->d[1],obb2->d[2]);
                        BVHDrawHelpers::drawCoordinateMarkerGL(0.5f, 0.25f);
                        BVHDrawHelpers::drawObbVolume(obb2He, obbColor2, true);

                        glMultMatrixd(obb2Ori.ptr());

                        glTranslated(-obb2->To[0], -obb2->To[1], -obb2->To[2]);
                        glMultMatrixd(model2Mat.ptr());
                        glTranslated(-model2Pos.x(), -model2Pos.y(), -model2Pos.z());
                    }
                    glPopAttrib();
                    glPopMatrix();
                }
            }
        }
    }
#endif
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG_TRIANGLES
    if (m_intersectingTriangles.size() > 0)
    {

        for (std::map<std::string, std::vector<std::pair<int,int> > >::const_iterator it = m_intersectingTriangles.begin();
             it != m_intersectingTriangles.end(); it++)
        {
            std::string modelName1, modelName2;
            const std::string& modelPairId = it->first;

            std::vector<std::string> v;
            boost::algorithm::iter_split(v, modelPairId, boost::algorithm::first_finder("::"));

            //std::cout << " Draw for model pair: " << modelPairId << std::endl;
            if (v.size() == 2)
            {
                modelName1 = v.at(0);
                modelName2 = v.at(1);

                //std::cout << " model ids: " << modelName1 << "," << modelName2 << std::endl;

                ObbTreeGPUCollisionModel<Vec3Types> *obbModel1 = NULL, *obbModel2 = NULL;
                ObbTreeCPUCollisionModel<Vec3Types> *pqpModel1 = NULL, *pqpModel2 = NULL;
                std::vector<sofa::core::topology::Triangle> intersectingTriangles1, intersectingTriangles2;

                if (m_obbModels.find(modelName1) != m_obbModels.end() && m_obbModels.find(modelName2) != m_obbModels.end())
                {
                    obbModel1 = m_obbModels[modelName1];
                    obbModel2 = m_obbModels[modelName2];
                }
                else if (m_pqpModels.find(modelName1) != m_pqpModels.end() && m_pqpModels.find(modelName2) != m_pqpModels.end())
                {
                    pqpModel1 = m_pqpModels[modelName1];
                    pqpModel2 = m_pqpModels[modelName2];
                }

                {

                    // std::cout << " found both models in OBB Model instance register" << std::endl;

                    Vector3 modelPos1, modelPos2;
                    Matrix3 modelOrientation1, modelOrientation2;
                    Matrix4 modelOri1, modelOri2;
                    modelOri1.identity(); modelOri2.identity();

                    if (obbModel1 != NULL && obbModel2 != NULL)
                    {
                        obbModel1->getPosition(modelPos1);
                        obbModel2->getPosition(modelPos2);

                        obbModel1->getOrientation(modelOrientation1);
                        obbModel2->getOrientation(modelOrientation2);

                        for (int k = 0; k < 3; k++)
                        {
                            for (int l = 0; l < 3; l++)
                            {
                                modelOri1[k][l] = modelOrientation1[k][l];
                                modelOri2[k][l] = modelOrientation2[k][l];
                            }
                        }
                    }

#if 0
                    const sofa::core::objectmodel::BaseData* posData1 = NULL;
                    const sofa::core::objectmodel::BaseData* posData2 = NULL;
                    if (obbModel1 != NULL && obbModel2 != NULL)
                    {
                        posData1 = obbModel1->getMechanicalState()->baseRead(core::ConstVecCoordId::position());
                        posData2 = obbModel2->getMechanicalState()->baseRead(core::ConstVecCoordId::position());
                    }
                    else if (pqpModel1 != NULL && pqpModel2 != NULL)
                    {
                        posData1 = pqpModel1->getObjectMState()->baseRead(core::ConstVecCoordId::position());
                        posData2 = pqpModel2->getObjectMState()->baseRead(core::ConstVecCoordId::position());
                    }

                    if (posData1)
                    {
                        const void* posValues = posData1->getValueVoidPtr();
                        double t0 = posData1->getValueTypeInfo()->getScalarValue(posValues, 0);
                        double t1 = posData1->getValueTypeInfo()->getScalarValue(posValues, 1);
                        double t2 = posData1->getValueTypeInfo()->getScalarValue(posValues, 2);
                        double r0 = posData1->getValueTypeInfo()->getScalarValue(posValues, 3);
                        double r1 = posData1->getValueTypeInfo()->getScalarValue(posValues, 4);
                        double r2 = posData1->getValueTypeInfo()->getScalarValue(posValues, 5);
                        double r3 = posData1->getValueTypeInfo()->getScalarValue(posValues, 6);

                        modelPos1 = Vector3(t0, t1, t2);
                        Quaternion newRot = Quaternion(r0,r1,r2,r3);

                        Matrix3 newOrientation;
                        newRot.toMatrix(newOrientation);

                        for (int k = 0; k < 3; k++)
                        {
                            for (int l = 0; l < 3; l++)
                            {
                                modelOri1[k][l] = newOrientation[k][l];
                            }
                        }
                    }

                    if (posData2)
                    {
                        const void* posValues = posData2->getValueVoidPtr();
                        double t0 = posData2->getValueTypeInfo()->getScalarValue(posValues, 0);
                        double t1 = posData2->getValueTypeInfo()->getScalarValue(posValues, 1);
                        double t2 = posData2->getValueTypeInfo()->getScalarValue(posValues, 2);
                        double r0 = posData2->getValueTypeInfo()->getScalarValue(posValues, 3);
                        double r1 = posData2->getValueTypeInfo()->getScalarValue(posValues, 4);
                        double r2 = posData2->getValueTypeInfo()->getScalarValue(posValues, 5);
                        double r3 = posData2->getValueTypeInfo()->getScalarValue(posValues, 6);

                        modelPos2 = Vector3(t0, t1, t2);
                        Quaternion newRot = Quaternion(r0,r1,r2,r3);

                        Matrix3 newOrientation;
                        newRot.toMatrix(newOrientation);

                        for (int k = 0; k < 3; k++)
                        {
                            for (int l = 0; l < 3; l++)
                            {
                                modelOri2[k][l] = newOrientation[k][l];
                            }
                        }
                    }
#endif
                    const std::vector<std::pair<int,int> >& intersectingTris = it->second;
                    //std::cout << " intersecting triangle pairs recorded: " << intersectingTris.size() << std::endl;
                    for (std::vector<std::pair<int,int> >::const_iterator tri_it = intersectingTris.begin(); tri_it != intersectingTris.end(); tri_it++)
                    {
                        sofa::core::topology::Triangle tri1, tri2;
                        // std::cout << " * " << tri_it->first << ", " << tri_it->second << std::endl;
                        if (obbModel1 != NULL && obbModel1->getTriangle(tri_it->first, tri1) &&
                            obbModel2 != NULL && obbModel2->getTriangle(tri_it->second, tri2))
                        {
                            intersectingTriangles1.push_back(tri1);
                            intersectingTriangles2.push_back(tri2);
                        }

                        if (pqpModel1 != NULL && pqpModel1->getTriangle(tri_it->first, tri1) &&
                            pqpModel2 != NULL && pqpModel2->getTriangle(tri_it->second, tri2))
                        {
                            intersectingTriangles1.push_back(tri1);
                            intersectingTriangles2.push_back(tri2);
                        }
                    }

                    // std::cout << " model1 pos: " << modelPos1 << ", model2 pos: " << modelPos2 << std::endl;
                    // std::cout << " model1 ori: " << modelOri1 << ", model2 ori: " << modelOri2 << std::endl;
                    // std::cout << " triangles in model1: " << intersectingTriangles1.size() << ", in model2: " << intersectingTriangles2.size() << std::endl;
                    if (intersectingTriangles1.size() > 0 && intersectingTriangles2.size() > 0)
                    {

                        glPushMatrix();
                        glPushAttrib(GL_ENABLE_BIT);
                        glEnable(GL_COLOR_MATERIAL);

                        glBegin(GL_LINES);
                        glColor4d(1,1,1,1);
                        glVertex3d(0,0,0);
                        glColor4d(1,0,1,1);
                        glVertex3d(modelPos1.x(), modelPos1.y(), modelPos1.z());
                        glEnd();

                        glTranslated(modelPos1.x(), modelPos1.y(), modelPos1.z());
                        glMultMatrixd(modelOri1.transposed().ptr());

                        glLineWidth(8.0f);
                        glBegin(GL_LINES);
                        for (std::vector<sofa::core::topology::Triangle>::const_iterator it = intersectingTriangles1.begin(); it != intersectingTriangles1.end(); it++)
                        {
                            const sofa::core::topology::Triangle& tri = *it;
                            Vector3 vt1, vt2, vt3;

                            if (obbModel1 != NULL)
                            {
                                obbModel1->getVertex(tri[0], vt1);
                                obbModel1->getVertex(tri[1], vt2);
                                obbModel1->getVertex(tri[2], vt3);
                            }

                            if (pqpModel1 != NULL)
                            {
                                pqpModel1->getVertex(tri[0], vt1);
                                pqpModel1->getVertex(tri[1], vt2);
                                pqpModel1->getVertex(tri[2], vt3);
                            }

                            //std::cout << "   draw: " << tri << ": " << vt1 << ", " << vt2 << ", " << vt3 << std::endl;

                            glColor4d(0,1,0,1);
                            glVertex3d(vt1.x(), vt1.y(), vt1.z());
                            glColor4d(0,1,0,1);
                            glVertex3d(vt2.x(), vt2.y(), vt2.z());
                            glColor4d(0,1,0,1);
                            glVertex3d(vt2.x(), vt2.y(), vt2.z());
                            glColor4d(0,1,0,1);
                            glVertex3d(vt3.x(), vt3.y(), vt3.z());
                            glColor4d(0,1,0,1);
                            glVertex3d(vt3.x(), vt3.y(), vt3.z());
                            glColor4d(0,1,0,1);
                            glVertex3d(vt1.x(), vt1.y(), vt1.z());
                        }
                        glEnd();
                        glLineWidth(1.0f);

                        glPopAttrib();
                        glPopMatrix();

                        glPushMatrix();
                        glPushAttrib(GL_ENABLE_BIT);
                        glEnable(GL_COLOR_MATERIAL);

                        glBegin(GL_LINES);
                        glColor4d(1,1,1,1);
                        glVertex3d(0,0,0);
                        glColor4d(0,1,1,1);
                        glVertex3d(modelPos2.x(), modelPos2.y(), modelPos2.z());
                        glEnd();

                        glTranslated(modelPos2.x(), modelPos2.y(), modelPos2.z());
                        glMultMatrixd(modelOri2.transposed().ptr());

                        glLineWidth(8.0f);
                        glBegin(GL_LINES);
                        for (std::vector<sofa::core::topology::Triangle>::const_iterator it = intersectingTriangles2.begin(); it != intersectingTriangles2.end(); it++)
                        {
                            const sofa::core::topology::Triangle& tri = *it;
                            Vector3 vt1, vt2, vt3;

                            if (obbModel2 != NULL)
                            {
                                obbModel2->getVertex(tri[0], vt1);
                                obbModel2->getVertex(tri[1], vt2);
                                obbModel2->getVertex(tri[2], vt3);
                            }

                            if (pqpModel2 != NULL)
                            {
                                pqpModel2->getVertex(tri[0], vt1);
                                pqpModel2->getVertex(tri[1], vt2);
                                pqpModel2->getVertex(tri[2], vt3);
                            }

                            //std::cout << "   draw: " << tri << ": " << vt1 << ", " << vt2 << ", " << vt3 << std::endl;

                            glColor4d(1,1,1,1);
                            glVertex3d(vt1.x(), vt1.y(), vt1.z());
                            glColor4d(1,1,1,1);
                            glVertex3d(vt2.x(), vt2.y(), vt2.z());
                            glColor4d(1,1,1,1);
                            glVertex3d(vt2.x(), vt2.y(), vt2.z());
                            glColor4d(1,1,1,1);
                            glVertex3d(vt3.x(), vt3.y(), vt3.z());
                            glColor4d(1,1,1,1);
                            glVertex3d(vt3.x(), vt3.y(), vt3.z());
                            glColor4d(1,1,1,1);
                            glVertex3d(vt1.x(), vt1.y(), vt1.z());
                        }
                        glEnd();
                        glLineWidth(1.0f);

                        glPopAttrib();
                        glPopMatrix();
                    }
                }
            }
        }
    }
#endif

#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG_TRANSFORMED_VERTICES
    if (m_transformedVertices.size() > 0)
    {
        for (std::map<std::string, std::pair<GPUVertex*, GPUVertex*> >::iterator it = m_transformedVertices.begin(); it != m_transformedVertices.end(); it++)
        {
            GPUVertex* m1Vertices = it->second.first;
            GPUVertex* m2Vertices = it->second.second;

            std::string modelName1, modelName2;
            const std::string& modelPairId = it->first;

            std::vector<std::string> v;
            boost::algorithm::iter_split(v, modelPairId, boost::algorithm::first_finder("::"));

            //std::cout << " Draw for model pair: " << modelPairId << std::endl;
            if (v.size() == 2)
            {
                modelName1 = v.at(0);
                modelName2 = v.at(1);

                //std::cout << " model ids: " << modelName1 << "," << modelName2 << std::endl;

                ObbTreeGPUCollisionModel<Vec3Types> *obbModel1 = NULL, *obbModel2 = NULL;

                if (m_obbModels.find(modelName1) != m_obbModels.end() && m_obbModels.find(modelName2) != m_obbModels.end())
                {
                    obbModel1 = m_obbModels[modelName1];
                    obbModel2 = m_obbModels[modelName2];

                    glPointSize(5.0f);
                    glPushMatrix();
                    glPushAttrib(GL_ENABLE_BIT);
                    glEnable(GL_COLOR_MATERIAL);

                    std::cout << "model1 vertices: " << obbModel1->numVertices() << std::endl;
                    glBegin(GL_POINTS);
                    for (int i = 0; i < obbModel1->numVertices(); i++)
                    {
                        std::cout << " * " << i << ": " << m1Vertices[i].v.x << "," << m1Vertices[i].v.y << "," << m1Vertices[i].v.z << std::endl;
                        glColor4f(1,0,0,0.75);
                        glVertex3d(m1Vertices[i].v.x, m1Vertices[i].v.y, m1Vertices[i].v.z);
                    }
                    glEnd();

                    glPopAttrib();
                    glPopMatrix();

                    glPushMatrix();
                    glPushAttrib(GL_ENABLE_BIT);
                    glEnable(GL_COLOR_MATERIAL);

                    std::cout << "model2 vertices: " << obbModel2->numVertices() << std::endl;
                    glBegin(GL_POINTS);
                    for (int i = 0; i < obbModel2->numVertices(); i++)
                    {
                        std::cout << " * " << i << ": " << m2Vertices[i].v.x << "," << m2Vertices[i].v.y << "," << m2Vertices[i].v.z << std::endl;
                        glColor4f(1,1,0,0.75);
                        glVertex3d(m2Vertices[i].v.x, m2Vertices[i].v.y, m2Vertices[i].v.z);
                    }
                    glEnd();

                    glPopAttrib();
                    glPopMatrix();
                    glPointSize(1.0f);
                }
            }
        }
    }
#endif
}

void ObbTreeGPUCollisionDetection::scheduleBVHTraversals(std::vector< std::pair<OBBModelContainer,OBBModelContainer> >& narrowPhasePairs,
                                                         unsigned int iteration, unsigned int numSlots,
                                                         std::vector<OBBContainer*> models1, std::vector<OBBContainer*> models2,
                                                         std::vector<gProximityGPUTransform*> transforms1, std::vector<gProximityGPUTransform*> transforms2,
                                                         std::vector<gProximityWorkerUnit*> workerUnits)
{
    for (unsigned int k = 0; k < numSlots; k++)
    {
        int pairCheckIdx = (iteration * _numStreamedWorkerUnits.getValue()) + k;
        std::cout << "  pair check " << pairCheckIdx << std::endl;
        if (pairCheckIdx < narrowPhasePairs.size())
        {
            OBBContainer& obbTree1 = narrowPhasePairs[pairCheckIdx].first._obbContainer;
            OBBContainer& obbTree2 = narrowPhasePairs[pairCheckIdx].second._obbContainer;

            models1.push_back(&obbTree1);
            models2.push_back(&obbTree2);

            ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = narrowPhasePairs[pairCheckIdx].first._obbCollisionModel;
            ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = narrowPhasePairs[pairCheckIdx].second._obbCollisionModel;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG
            std::cout << " - CHECKED OBB MODEL PAIR: " << obbModel1->getName() << " - " << obbModel2->getName() << std::endl;
#endif
            gProximityGPUTransform* modelTr1 = _gpuTransforms[_gpuTransformIndices[obbModel1->getName()]];
            gProximityGPUTransform* modelTr2 = _gpuTransforms[_gpuTransformIndices[obbModel2->getName()]];
            transforms1.push_back(modelTr1);
            transforms2.push_back(modelTr2);

            workerUnits.push_back(_streamedWorkerUnits[k % _numStreamedWorkerUnits.getValue()]);
        }
    }
}

void ObbTreeGPUCollisionDetection::runBVHTraversals()
{

}

void ObbTreeGPUCollisionDetection::partitionResultBins()
{

}

void ObbTreeGPUCollisionDetection::runTriangleTests()
{

}
