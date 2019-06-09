#include "ObbTreeGPUCollisionDetection_Threaded.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include <cutil/cutil.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/DrawTool.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <SofaOpenglVisual/OglModel.h>

#include <sofa/core/collision/Contact.h>
#include <SofaBaseCollision/DiscreteIntersection.h>
#include <SofaBaseCollision/BaseContactMapper.h>
#include <SofaMeshCollision/BarycentricPenalityContact.h>

#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>

#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseMechanics/MechanicalObject.h>

#include "ObbTreeGPU_CudaDataStructures.h"
#include "ObbTreeGPUTriangleCollision_cuda.h"
#include "ObbTreeGPUCollisionModel_cuda.h"

// (FA) Not working under Linux (unresolved symbol issue)
#ifdef _WIN32
#include "obbtree_rulevisualisation.h"
#endif

#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>

#ifdef _WIN32
// (BE)
#include <RobotConnector.h>
using sofa::component::controller::RobotConnector;
#endif

#ifdef _WIN32
namespace sofa
{
namespace component
{
namespace collision
{
class ObbTreeGPUCollisionDetection_Threaded_Private
{
public:
    ObbTree_RuleVisualisation * vis;
};
}
}
}
#endif

using namespace sofa;
using namespace sofa::component::collision;
using namespace sofa::component::container;

using namespace sofa::simulation;

SOFA_DECL_CLASS(ObbTreeGPUCollisionDetection_Threaded)

ObbTreeGPUCollisionDetection_Threaded::ObbTreeGPUCollisionDetection_Threaded():
    BruteForceDetection(),
    m_intersection(NULL),
    _totalResultBinCount(0),
    _numStreamedWorkerUnits(initData(&_numStreamedWorkerUnits, (unsigned int) 4, "numStreamedWorkerUnits", "Number of stream worker units on GPU per worker thread", true, false)),
    _numStreamedWorkerResultBins(initData(&_numStreamedWorkerResultBins, (unsigned int) 4, "numStreamedWorkerResultBins", "Number of result bins (triangle collision checks) on GPU per worker thread", true, false)),
    _streamedWorkerResultMinSize(initData(&_streamedWorkerResultMinSize, (unsigned int) 2048, "streamedWorkerResultMinSize", "Size of smallest result bins", true, true)),
    _streamedWorkerResultMaxSize(initData(&_streamedWorkerResultMaxSize, (unsigned int) 16384, "streamedWorkerResultMaxSize", "Size of largest result bins", true, true)),
    _showTestFramework(initData(&_showTestFramework, (bool)false, "showTestframework", "Show Testframework", true, false)),
    m_numWorkerThreads(initData(&m_numWorkerThreads, 1, "numWorkerThreads", "Number of worker threads", true, true)),
    m_trianglePairSizeRatio(initData(&m_trianglePairSizeRatio, 0.1f, "trianglePairSizeRatio", "Percentage of possible triangle pair results allocated for result bins", true, true)),
    m_fakeGripping_EventSequence(initData(&m_fakeGripping_EventSequence, "fakeGrippingEventSequence", "Sequence of contact events for fake gripping"))
{
    m_scheduler_traversal = NULL;
    m_scheduler_triangles = NULL;

	m_scheduler_cpu_traversal = NULL;

    m_active_FakeGripping_Event = -1;
    m_previous_FakeGripping_Event = -1;

	m_detectionOutputVectors = new NarrowPhaseDetection::DetectionOutputMap();

#ifdef _WIN32
    m_d = new ObbTreeGPUCollisionDetection_Threaded_Private();

    m_d->vis = NULL;

    if (_showTestFramework.getValue()) {
        m_d->vis = new ObbTree_RuleVisualisation();
        m_d->vis->show();
    }
#endif
}

ObbTreeGPUCollisionDetection_Threaded::~ObbTreeGPUCollisionDetection_Threaded()
{
#ifdef _WIN32
    if (m_d) {
        if (m_d->vis) {
            m_d->vis->close();
            delete m_d->vis;
            m_d->vis = NULL;
        }
        delete m_d;
        m_d = NULL;
    }
#endif

    sout << "ObbTreeGPUCollisionDetection_Threaded::~ObbTreeGPUCollisionDetection_Threaded(" << this->getName() << ")" << std::endl;

    m_scheduler_traversal->cleanup();
    m_scheduler_traversal->getScheduler()->stopThreads(true);

    m_scheduler_triangles->cleanup();
    m_scheduler_triangles->getScheduler()->stopThreads(true);

    delete m_scheduler_traversal;
	m_scheduler_traversal = NULL;
    delete m_scheduler_triangles;
	m_scheduler_triangles = NULL;

	m_scheduler_cpu_traversal->cleanup();
	m_scheduler_cpu_traversal->getScheduler()->stopThreads(true);

	delete m_scheduler_cpu_traversal;
	m_scheduler_cpu_traversal = NULL;

    for (size_t k = 0; k < m_traversalTasks.size(); k++)
    {
        delete m_traversalTasks[k];
		m_traversalTasks[k] = NULL;
    }
    m_traversalTasks.clear();

	for (size_t k = 0; k < m_triangleTasks.size(); k++)
    {
        delete m_triangleTasks[k];
		m_triangleTasks[k] = NULL;
    }
    m_triangleTasks.clear();

	for (size_t k = 0; k < m_cpuTraversalTasks.size(); k++)
	{
		delete m_cpuTraversalTasks[k];
		m_cpuTraversalTasks[k] = NULL;
	}
	m_cpuTraversalTasks.clear();

	if (m_detectionOutputVectors != NULL)
	{
		delete m_detectionOutputVectors;
		m_detectionOutputVectors = NULL;
	}

    sout << " call cudaDeviceReset()" << std::endl;
    CUDA_SAFE_CALL(cudaDeviceReset());
}

void ObbTreeGPUCollisionDetection_Threaded::init()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No Cude Device Found. ObbTreeCollisionDetection will not work.\n");
        return;
    }
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d. Supports concurrent kernel execution = %d; asyncEngineCount = %d\n",
               device, deviceProp.major, deviceProp.minor, deviceProp.concurrentKernels, deviceProp.asyncEngineCount);

		// Uncomment only for debugging CUDA calls!
        //CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceBlockingSync));
    }

    BruteForceDetection::init();

	std::cout << "NarrowPhaseDetection DetectionOutputMap instance = " << &(this->m_outputsMap) << std::endl;

    std::vector<sofa::component::collision::OBBTreeGPUDiscreteIntersection* > moV;
    sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::collision::OBBTreeGPUDiscreteIntersection, std::vector<sofa::component::collision::OBBTreeGPUDiscreteIntersection* > > cb(&moV);

	getContext()->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::collision::OBBTreeGPUDiscreteIntersection>::get(), cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchRoot);

    sout << "ObbTreeGPUCollisionDetection_Threaded::init(): Searched for instances of OBBTreeGPUDiscreteIntersection; found = " << moV.size() << std::endl;
    if (moV.size() == 1)
    {
        sout << " Using: " << moV[0]->getName() << " of type " << moV[0]->getTypeName() << std::endl;
        m_intersection = moV[0];
    }

    m_scheduler_traversal = new ObbTreeGPU_MultiThread_Scheduler<BVHTraversalTask>(m_numWorkerThreads.getValue());

    sout << "=== ObbTreeGPUCollisionDetection_Threaded::ObbTreeGPUCollisionDetection_Threaded(" << this->getName() << ") ===" << std::endl;
    sout << " Worker Thread count = " << m_numWorkerThreads.getValue() << std::endl;
    sout << " _numStreamedWorkerUnits = " << _numStreamedWorkerUnits.getValue() << std::endl;
    sout << " _numStreamedWorkerResultBins = " << _numStreamedWorkerResultBins.getValue() << std::endl;
    sout << " _streamedWorkerResultMinSize = " << _streamedWorkerResultMinSize.getValue() << std::endl;
    sout << " _streamedWorkerResultMaxSize = " << _streamedWorkerResultMaxSize.getValue() << std::endl;

    for (int i = 0; i < m_numWorkerThreads.getValue(); i++)
        m_scheduler_traversal->getScheduler()->createWorkerThread(true, "BVHTraversal");

    m_scheduler_traversal->getScheduler()->startThreads();
    m_scheduler_traversal->getScheduler()->pauseThreads();

    m_scheduler_triangles = new ObbTreeGPU_MultiThread_Scheduler<NarrowPhaseGPUTask>(m_numWorkerThreads.getValue());

    for (int i = 0; i < m_numWorkerThreads.getValue(); i++)
        m_scheduler_triangles->getScheduler()->createWorkerThread(true, "TriangleChecks");

    m_scheduler_triangles->getScheduler()->startThreads();
    m_scheduler_triangles->getScheduler()->pauseThreads();

    m_scheduler_traversal->init();
    m_scheduler_triangles->init();

	m_scheduler_cpu_traversal = new ObbTreeGPU_MultiThread_Scheduler<CPUCollisionCheckTask>(m_numWorkerThreads.getValue());
	for (int i = 0; i < m_numWorkerThreads.getValue(); i++)
		m_scheduler_cpu_traversal->getScheduler()->createWorkerThread(true, "CPUTraversal");

	m_scheduler_cpu_traversal->getScheduler()->startThreads();
	m_scheduler_cpu_traversal->getScheduler()->pauseThreads();

	m_scheduler_cpu_traversal->init();
}

void ObbTreeGPUCollisionDetection_Threaded::reinit()
{
    BruteForceDetection::reinit();
    m_obbModels.clear();
}

void ObbTreeGPUCollisionDetection_Threaded::reset()
{
    for (std::map<int, std::pair<std::pair<sofa::simulation::Node*, sofa::simulation::Node*>, sofa::component::mapping::RigidRigidMapping<Rigid3Types, Rigid3Types>::SPtr > >::iterator it = m_activeFakeGrippingRules.begin(); it != m_activeFakeGrippingRules.end(); it++)
    {
        sofa::component::mapping::RigidRigidMapping<Rigid3dTypes, Rigid3dTypes>::SPtr rigidMapping = it->second.second;
        sofa::simulation::Node* rigidMappingParent = it->second.first.first;

        rigidMappingParent->removeObject(rigidMapping);
        rigidMapping->cleanup();
        rigidMapping.reset();
    }

    for (std::map<int, std::pair<std::pair<sofa::simulation::Node*, sofa::simulation::Node*>, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr > >::iterator it = m_activeFakeGrippingRules_Testing_Slaves.begin(); it != m_activeFakeGrippingRules_Testing_Slaves.end(); it++)
    {
        sofa::component::mapping::RigidMapping<Rigid3dTypes, Vec3Types>::SPtr rigidMapping = it->second.second;
        sofa::simulation::Node* rigidMappingParent = it->second.first.first;

        rigidMappingParent->removeObject(rigidMapping);
        rigidMapping->cleanup();
        rigidMapping.reset();
    }

    // TODO Nodes und RigidMappings für Ghosts/Subobjekte resetten!!!

    m_activeFakeGrippingRules.clear();
    m_activeFakeGrippingRules_Testing_Slaves.clear();

    m_active_FakeGripping_Event = -1;
    m_previous_FakeGripping_Event = -1;

    m_fakeGripping_Activated_Rules.clear();

    for (size_t k = 0; k < m_fakeGripping_Events.size(); k++)
        m_fakeGripping_Activated_Rules.insert(std::make_pair(k, false));

    m_fakeGripping_CollisionCheck_Exceptions.clear();
}

unsigned long roundToThousand(unsigned long num)
{
    num += 500;
    unsigned long thousands = (unsigned long)(num / 1000);
    return thousands * 1000;
}

#ifdef _WIN32
void ObbTreeGPUCollisionDetection_Threaded::updateRuleVis() {
    if (m_d && m_d->vis) {
        m_d->vis->updateRules(m_fakeGripping_Events, m_active_FakeGripping_Event, m_previous_FakeGripping_Event);
    }
}
#endif

void ObbTreeGPUCollisionDetection_Threaded::bwdInit()
{
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&_transformStream, cudaStreamNonBlocking));
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&_memoryStream, cudaStreamNonBlocking));

    std::vector<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>* > obbTreeGPUCollisionModels;
    sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>, std::vector<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>* > > obbTreeGPUCollisionModels_cb(&obbTreeGPUCollisionModels);
	getContext()->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types> >::get(), obbTreeGPUCollisionModels_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchRoot);

    int numLargeTraversalUnits = 0, numSmallTraversalUnits = 0;
    unsigned long finalAboveAvgSize = 0, finalBelowAvgSize = 0;

    if (obbTreeGPUCollisionModels.size() > 0)
    {
        _gpuTransforms = new gProximityGPUTransform*[obbTreeGPUCollisionModels.size()];

        sout << "ObbTreeGPUCollisionDetection::init(): Searched for instances of OBBTreeGPUCollisionModel; found = " << obbTreeGPUCollisionModels.size() << std::endl;
        for (unsigned int k = 0; k < obbTreeGPUCollisionModels.size(); k++)
        {
            _gpuTransforms[k] = new gProximityGPUTransform();

            sout << " * " << k << ": " << obbTreeGPUCollisionModels[k]->getName() << std::endl;
            sout << "   store in model map" << std::endl;
            //_modelTransforms.insert(std::make_pair(obbTreeGPUCollisionModels[k]->getName(), new gProximityGPUTransform()));
            m_obbTreeGPUModels.insert(std::make_pair(obbTreeGPUCollisionModels[k]->getName(), obbTreeGPUCollisionModels[k]));

            Vector3 modelPosition = obbTreeGPUCollisionModels[k]->getCachedPosition();
            Matrix3 modelOrientation = obbTreeGPUCollisionModels[k]->getCachedOrientationMatrix();

            float3 h_modelPosition = make_float3(modelPosition.x(), modelPosition.y(), modelPosition.z());
            Matrix3x3_d h_modelOrientation;

            h_modelOrientation.m_row[0].x = modelOrientation(0, 0);
            h_modelOrientation.m_row[0].y = modelOrientation(0, 1);
            h_modelOrientation.m_row[0].z = modelOrientation(0, 2);
            h_modelOrientation.m_row[1].x = modelOrientation(1, 0);
            h_modelOrientation.m_row[1].y = modelOrientation(1, 1);
            h_modelOrientation.m_row[1].z = modelOrientation(1, 2);
            h_modelOrientation.m_row[2].x = modelOrientation(2, 0);
            h_modelOrientation.m_row[2].y = modelOrientation(2, 1);
            h_modelOrientation.m_row[2].z = modelOrientation(2, 2);

            sout << "     model " << obbTreeGPUCollisionModels[k]->getName() << " position    = " << h_modelPosition.x << "," << h_modelPosition.y << "," << h_modelPosition.z << std::endl;
            sout << "                                                             orientation = [" << h_modelOrientation.m_row[0].x << "," << h_modelOrientation.m_row[0].y << "," << h_modelOrientation.m_row[0].z << "],[" << h_modelOrientation.m_row[1].x << "," << h_modelOrientation.m_row[1].y << "," << h_modelOrientation.m_row[1].z << "],[" << h_modelOrientation.m_row[2].x << "," << h_modelOrientation.m_row[2].y << "," << h_modelOrientation.m_row[2].z << "]" << std::endl;

            sout << "   Initial position upload to GPU memory" << std::endl;

            gProximityGPUTransform* gpTransform = _gpuTransforms[k];

            TOGPU(gpTransform->modelTranslation, &h_modelPosition, sizeof(float3));
            TOGPU(gpTransform->modelOrientation, &h_modelOrientation, sizeof(Matrix3x3_d));

            _gpuTransformIndices.insert(std::make_pair(obbTreeGPUCollisionModels[k]->getName(), k));
        }

        sout << "Check all model combinations for number of possible triangle pair intersections" << std::endl;
        unsigned long maxTriPairs = 0;
        std::map<std::pair<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>*, sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>*>, unsigned long> numTriPairCombinations;
        std::vector<unsigned long> numTriPairCombinations_all;
        unsigned long numTriPairCombinations_sum = 0;
        for (unsigned int k = 0; k < obbTreeGPUCollisionModels.size(); k++)
        {
            sout << " model1 at index " << k << std::endl;
            sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>* model_1 = obbTreeGPUCollisionModels[k];
            for (unsigned int l = 0; l < obbTreeGPUCollisionModels.size(); l++)
            {
                sout << "   model2 at index " << l << std::endl;
                sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>* model_2 = obbTreeGPUCollisionModels[l];

                if (model_1->getName().compare(model_2->getName()) == 0)
                    continue;

                std::pair<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>*, sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>*> model_pair_1 = std::make_pair(model_1, model_2);
                std::pair<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>*, sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>*> model_pair_2 = std::make_pair(model_2, model_1);

                if (numTriPairCombinations.find(model_pair_1) != numTriPairCombinations.end())
                    continue;

                if (numTriPairCombinations.find(model_pair_2) != numTriPairCombinations.end())
                    continue;

                unsigned long numTriPairs = model_1->numTriangles() * model_2->numTriangles();

                sout << "  - combination " << model_1->getName() << " - " << model_2->getName() << " = " << numTriPairs << " possible combinations." << std::endl;

                numTriPairCombinations.insert(std::make_pair(std::make_pair(model_1, model_2), numTriPairs));
                numTriPairCombinations_all.push_back(numTriPairs);

                if (numTriPairs > maxTriPairs)
                {
                    sout << "   new max. value = " << numTriPairs << std::endl;
                    maxTriPairs = numTriPairs;
                }

                numTriPairCombinations_sum += numTriPairs;
            }
        }

        sout << "AFTER checking model pair combinations" << std::endl;

        unsigned int numBelowAvg = 0, numAboveAvg = 0;
        unsigned long totalBelowAvgSize = 0, totalAboveAvgSize = 0;

        if (numTriPairCombinations_all.size() > 0)
        {
            std::sort(numTriPairCombinations_all.begin(), numTriPairCombinations_all.end(), std::greater<unsigned long>());
            sout << " max. triangle pair count = " << numTriPairCombinations_all.at(0) << "; min. triangle pair count = " << numTriPairCombinations_all.at(numTriPairCombinations_all.size() - 1) << std::endl;
            sout << " triangle pair counts per model pair combinations: " << std::endl;

            for (std::map<std::pair<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>*, sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>*>, unsigned long>::const_iterator it = numTriPairCombinations.begin(); it != numTriPairCombinations.end(); it++)
            {
                sout << "   - " << it->first.first->getName() << " -- " << it->first.second->getName() << ": " << it->second << std::endl;
            }

            unsigned long avgNumTriangleCombinations = numTriPairCombinations_sum / numTriPairCombinations_all.size();
            for (unsigned int k = 0; k < numTriPairCombinations_all.size(); k++)
            {
                if (numTriPairCombinations_all[k] <= avgNumTriangleCombinations)
                {
                    numBelowAvg++;
                    totalBelowAvgSize += numTriPairCombinations_all[k];
                }
                else
                {
                    numAboveAvg++;
                    totalAboveAvgSize += numTriPairCombinations_all[k];
                }
            }

            sout << "average triangle pair count = " << avgNumTriangleCombinations << "; below avg. = " << numBelowAvg << ", above avg. = " << numAboveAvg << std::endl;
            unsigned int numTotalWorkerUnits = m_numWorkerThreads.getValue() * _numStreamedWorkerUnits.getValue();
            float triangleCombinationSizeRatio = (numAboveAvg * 1.0f) / (numBelowAvg * 1.0f);
            numLargeTraversalUnits = (int)std::ceil(triangleCombinationSizeRatio * numTotalWorkerUnits);
            numSmallTraversalUnits = numTotalWorkerUnits - numLargeTraversalUnits;

            if (numSmallTraversalUnits <= 0)
                numSmallTraversalUnits = 4;

            if (numLargeTraversalUnits <= 0)
                numLargeTraversalUnits = 2;

            sout << "total worker units = " << numTotalWorkerUnits << ", small/large ratio = " << triangleCombinationSizeRatio << ", small units = " << numSmallTraversalUnits << ", large units = " << numLargeTraversalUnits << std::endl;
        }

        if (numBelowAvg == 0)
            numBelowAvg = 1;

        if (numAboveAvg == 0)
            numAboveAvg = 1;

        unsigned long belowAvgSize = totalBelowAvgSize / numBelowAvg;
        unsigned long aboveAvgSize = totalAboveAvgSize / numAboveAvg;
        sout << "avg. size for small units = " << belowAvgSize << "; avg. size for large units = " << aboveAvgSize << std::endl;

        finalBelowAvgSize = m_trianglePairSizeRatio.getValue() * belowAvgSize;
        finalAboveAvgSize = m_trianglePairSizeRatio.getValue() * aboveAvgSize;

        finalBelowAvgSize = roundToThousand(finalBelowAvgSize);
        finalAboveAvgSize = roundToThousand(finalAboveAvgSize);

        if (finalBelowAvgSize == 0)
            finalBelowAvgSize = 1024;

        if (finalAboveAvgSize == 0)
            finalAboveAvgSize = 2048;

        sout << "final avg. size for small units = " << finalBelowAvgSize << "; for large units = " << finalAboveAvgSize << std::endl;
    }

    sout << "ObbTreeGPUCollisionDetection::bwdInit(" << this->getName() << ")" << std::endl;

    std::vector<ObbTreeGPULocalMinDistance* > lmdNodes;
    sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<ObbTreeGPULocalMinDistance, std::vector<ObbTreeGPULocalMinDistance* > > cb(&lmdNodes);

	getContext()->getObjects(sofa::core::objectmodel::TClassInfo<ObbTreeGPULocalMinDistance>::get(), cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchRoot);

    sout << "ObbTreeGPUCollisionDetection::bwdInit(): ObbTreeGPULocalMinDistance instances found: " << lmdNodes.size() << std::endl;
    if (lmdNodes.size() > 0)
    {
        sout << " alarmDistance = " << lmdNodes.at(0)->getAlarmDistance() << ", contactDistance = " << lmdNodes.at(0)->getContactDistance() << std::endl;
        m_alarmDistance = lmdNodes.at(0)->getAlarmDistance();
        m_contactDistance = lmdNodes.at(0)->getContactDistance();
    }
    else
    {
        m_alarmDistance = 0.25f;
        m_contactDistance = 0.125f;
    }

    sout << " allocate streamed worker units: " << _numStreamedWorkerUnits.getValue() << std::endl;
    _workerStreams.resize(_numStreamedWorkerUnits.getValue());
    _workerEvents.resize(_numStreamedWorkerUnits.getValue());

    m_scheduler_traversal->bwdInit();
    m_scheduler_triangles->bwdInit();

    for (int k = 0; k < this->m_numWorkerThreads.getValue(); k++)
    {
		// Useful for something?
        TaskStatus status;

        BVHTraversalTask* traversal_task = new BVHTraversalTask(&status, numSmallTraversalUnits, finalBelowAvgSize, numLargeTraversalUnits, finalAboveAvgSize);

		std::stringstream idStream;
		idStream << "GPU BVH traversal task " << k;
		traversal_task->setTaskID(idStream.str());
		idStream.str("");

        traversal_task->setAlarmDistance(m_alarmDistance);
        traversal_task->setContactDistance(m_alarmDistance); // (BE) use Alarm Distance instead of contact distance
        m_traversalTasks.push_back(traversal_task);

        NarrowPhaseGPUTask* triangle_task = new NarrowPhaseGPUTask(&status, 4, _streamedWorkerResultMinSize.getValue(), _streamedWorkerResultMaxSize.getValue());
        
		idStream << "GPU triangle task " << k;
		triangle_task->setTaskID(idStream.str());
		
		triangle_task->setAlarmDistance(m_alarmDistance);
        triangle_task->setContactDistance(m_alarmDistance);  // (BE) use Alarm Distance instead of contact distance
        m_triangleTasks.push_back(triangle_task);
    }

	for (int k = 0; k < m_numWorkerThreads.getValue(); ++k)
	{
		// Useful for something?
		TaskStatus status;

		std::stringstream idStr;
		idStr << "CPU collision task " << k;
		// No further setup necessary for CPU tasks as of now; assume 8 'worker units' per task
		CPUCollisionCheckTask* cpu_task = new CPUCollisionCheckTask(&status, this, 8);
		cpu_task->setTaskID(idStr.str());
		m_cpuTraversalTasks.push_back(cpu_task);
	}

    std::string fakeGripping_Sequence = m_fakeGripping_EventSequence.getValue();
    if (!fakeGripping_Sequence.empty())
    {
        sout << "Fake-gripping event sequence detected; split events and conditions" << std::endl;
        boost::regex eventSplit(";;;");
        boost::regex eventEntrySplit(";");
        boost::regex coEntrySplit("::");

        boost::sregex_token_iterator i(fakeGripping_Sequence.begin(), fakeGripping_Sequence.end(), eventSplit, -1);
        boost::sregex_token_iterator j;
        sout << " detected events: " << std::endl;
        while (i != j)
        {
            FakeGripping_Event_Container container;
            std::string eventEntryStr = i->str();
            sout << "   - " << *i++ <<  ": ";
            boost::sregex_token_iterator e_i(eventEntryStr.begin(), eventEntryStr.end(), eventEntrySplit, -1);
            boost::sregex_token_iterator e_j;
            unsigned int numTokens = 0;
            while (e_i != e_j)
            {
                if (numTokens == 0)
                {
                    container.activeModel = e_i->str();
                    sout << " active object:" << *e_i << " ";
                }
                else if (numTokens == 1)
                {
                    container.leadingModel = e_i->str();
                    sout << " leading object:" << *e_i << " ";
                }
                else if (numTokens == 2)
                {
                    sout << " colliding objects: " << *e_i << ": ";
                    std::string coll_obj_str = e_i->str();
                    boost::sregex_token_iterator c_i(coll_obj_str.begin(), coll_obj_str.end(), coEntrySplit, -1);
                    boost::sregex_token_iterator c_j;
                    while (c_i != c_j)
                    {
                        container.contactModels.push_back(c_i->str());
                        sout << " " << *c_i++;
                    }
                }
                else if (numTokens == 3)
                {
                    sout << " colliding objects condition: " << *e_i;
                    if (e_i->str().compare("AND") == 0)
                        container.contactCondition = FG_AND;
                    else if (e_i->str().compare("OR") == 0)
                        container.contactCondition = FG_OR;
                }
                *e_i++;
                numTokens++;
            }

            this->m_fakeGripping_Events.push_back(container);

            sout << sendl;
        }
    }

    testOutputFilename.setValue(sofa::helper::system::DataRepository.getFirstPath() + "/" + this->getName() + ".log");
    sout << "testOutputFilename = " << testOutputFilename.getValue() << std::endl;
}

void ObbTreeGPUCollisionDetection_Threaded::addCollisionModels(const sofa::helper::vector<core::CollisionModel *> collisionModels)
{
    if (!this->f_printLog.getValue())
        this->f_printLog.setValue(true);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONMODELS
    sout << "ObbTreeGPUCollisionDetection::addCollisionModels(): " << collisionModels.size() << " models." << std::endl;
#endif
    for (sofa::helper::vector<core::CollisionModel *>::const_iterator it = collisionModels.begin(); it != collisionModels.end(); it++)
    {
        bool obbModelFound = false;
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONMODELS_VERBOSE
        sout << " * Add model " << (*it)->getName() << " of type " << (*it)->getTypeName() << std::endl;
#endif
        core::CollisionModel* cmc = (*it);
        do
        {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONMODELS_VERBOSE
            sout << "   examine " << cmc->getName() << ", type " << cmc->getTypeName() << " if it's a ObbTreeGPUCollisionModel" << std::endl;
#endif
            ObbTreeGPUCollisionModel<Vec3Types>* obbModel = dynamic_cast<ObbTreeGPUCollisionModel<Vec3Types>*>(cmc);
            if (obbModel)
            {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONMODELS_VERBOSE
                sout << "    it IS." << std::endl;
#endif
                addCollisionModel(cmc);
                obbModelFound = true;
                break;
            }
            else
            {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONMODELS_VERBOSE
                sout << "    it IS NOT." << std::endl;
#endif
            }
            cmc = cmc->getNext();
        } while (cmc != NULL);

        if (!obbModelFound)
        {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONMODELS_VERBOSE
            sout << "No ObbTreeGPUCollisionModel found in hierarchy starting at " << (*it)->getName() << ", falling back to BruteForceDetection" << std::endl;
#endif
            BruteForceDetection::addCollisionModel((*it));
        }
    }
}

void ObbTreeGPUCollisionDetection_Threaded::addCollisionModel(core::CollisionModel *cm)
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
    std::cout << "ObbTreeGPUCollisionDetection::addCollisionModel(" << cm->getName() << "), type = " << cm->getTypeName() << std::endl;
#endif

    if (!cm)
        return;

#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
    std::cout << "ObbTreeGPUCollisionDetection::addCollisionModel(" << cm->getName() << "), type = " << cm->getTypeName() << std::endl;
#endif

    ObbTreeGPUCollisionModel<Vec3Types>* obbModel = dynamic_cast<ObbTreeGPUCollisionModel<Vec3Types>*>(cm);

    if (obbModel)
    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
        std::cout << "  obbModel = " << obbModel->getName() << " of type " << obbModel->getTypeName() << std::endl;
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
                std::cout << " collisions between cm = " << cm->getLast()->getName() << " and " << cm2->getLast()->getName() << " not kept!" << std::endl;
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
                std::cout << "   registering OBB model " << obbModel->getName() << std::endl;
#endif
                m_obbModels.insert(std::make_pair(obbModel->getName(), obbModel));
            }
        }
    }
    else
    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
        std::cout << "Model " << cm->getName() << " is not a ObbTreeGPU model, fallback to BruteForceDetection" << std::endl;
#endif
        BruteForceDetection::addCollisionModel(cm);
    }
}

void ObbTreeGPUCollisionDetection_Threaded::addCollisionPairs(const sofa::helper::vector<std::pair<core::CollisionModel *, core::CollisionModel *> > &v)
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONPAIRS
    std::cout << "=== ObbTreeGPUCollisionDetection::addCollisionPairs(): " << v.size() << " possible pairs. ===" << std::endl;
    int addedPairs = 0;
#endif
    for (sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >::const_iterator it = v.begin(); it != v.end(); it++)
    {
        addCollisionPair(*it);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONPAIRS
        std::cout << " Add: " << it->first->getName() << " -- " << it->second->getName() << std::endl;
        addedPairs++;
#endif

    }

#ifndef	OBBTREE_GPU_COLLISION_DETECTION_SEQUENTIAL_CPU_COLLISION_CHECKS
	BruteForceDetection::createDetectionOutputs(this->_narrowPhasePairs_CPU);
	const sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap& detection_outputs = this->getDetectionOutputs();
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONPAIRS
	std::cout << "=== Pre-created detection outputs: " << detection_outputs.size() << " ===" << std::endl;
	for (sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator do_it = detection_outputs.begin(); do_it != detection_outputs.end(); ++do_it)
	{
		std::cout << " - " << do_it->first.first->getName() << " -- " << do_it->first.second->getName() << std::endl;
	}
#endif
#endif

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONPAIRS
    std::cout << "=== ObbTreeGPUCollisionDetection::addCollisionPairs(): " << addedPairs << " pairs added. ===" << std::endl;
#endif
}

void ObbTreeGPUCollisionDetection_Threaded::addCollisionPair(const std::pair<core::CollisionModel *, core::CollisionModel *> &cmPair)
{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
    std::cout << "ObbTreeGPUCollisionDetection::addCollisionPair(" << cmPair.first->getName() << "," << cmPair.second->getName() << ")" << std::endl;
    std::cout << " model types: " << cmPair.first->getTypeName() << " - " << cmPair.second->getTypeName() << std::endl;
#endif

	ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = dynamic_cast<ObbTreeGPUCollisionModel<Vec3Types>*>(cmPair.first);
	ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = dynamic_cast<ObbTreeGPUCollisionModel<Vec3Types>*>(cmPair.second);

	if (obbModel1 && obbModel2)
	{
		bool doGPUObbTest = true;

		bool swapModels = false;
		core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cmPair.first, cmPair.second, swapModels);
		if (intersector == NULL)
		{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
			std::cout << " no suitable intersector between cm = " << cmPair.first->getName() << " and " << cmPair.second->getName() << " found!" << std::endl;
#endif
			doGPUObbTest = false;
		}

		core::CollisionModel* cm1 = (swapModels ? cmPair.second : cmPair.first);
		core::CollisionModel* cm2 = (swapModels ? cmPair.first : cmPair.second);


#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
		std::cout << " Intersector used for intersectability query: " << intersector->name() << std::endl;
#endif
		if (!intersector->canIntersect(cm1->begin(), cm2->begin()))
		{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
			std::cout << " cm1 = " << cm1->getName() << " and cm2 = " << cm2->getName() << " can't intersect; skip narrow-phase check!" << std::endl;
#endif
			doGPUObbTest = false;
		}
		else
		{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
			std::cout << " cm1 = " << cm1->getName() << " and cm2 = " << cm2->getName() << " CAN intersect: Do narrow-phase check!" << std::endl;
#endif
		}

		if (doGPUObbTest)
		{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
			std::cout << " check using GPU-based implementation" << std::endl;
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

			OBBModelContainer obbContainer1, obbContainer2;
			obbContainer1._obbContainer = obbTree1;
			obbContainer1._obbCollisionModel = obbModel1;
			obbContainer2._obbContainer = obbTree2;
			obbContainer2._obbCollisionModel = obbModel2;

#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
			std::cout << " combo1Found = " << combo1Found << ", combo1Used = " << combo1Used << ";" << " combo2Found = " << combo2Found << ", combo2Used = " << combo2Used << std::endl;
#endif

			if (combo1Used && combo2Used)
			{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
				std::cout << "  push combo1; no combo registered yet = " << obbModel1->getName() << " -- " << obbModel2->getName() << std::endl;
#endif
				_narrowPhasePairs.push_back(std::make_pair(obbContainer1, obbContainer2));
			}
			else if (combo1Used && !combo2Used)
			{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
				std::cout << "  push combo1 = " << obbModel1->getName() << " -- " << obbModel2->getName() << std::endl;
#endif

				_narrowPhasePairs.push_back(std::make_pair(obbContainer1, obbContainer2));
			}
			else if (combo2Used && !combo1Used)
			{
				std::cout << "  push combo2 = " << obbModel2->getName() << " -- " << obbModel1->getName() << std::endl;
				_narrowPhasePairs.push_back(std::make_pair(obbContainer2, obbContainer1));
			}
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG            
			else
			{
				std::cout << "  WARNING -- combo1/2 used flags not set, skipping: " << obbModel1->getName() << " -- " << obbModel2->getName() << std::endl;
			}
#endif
		}
	}
	else
	{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
		sout << " BruteForceDetection (CPU-based models)" << cmPair.first->getName() << " (of type " << cmPair.first->getTypeName() << ") -- " << cmPair.second->getName() << " (of type " << cmPair.second->getTypeName() << ")" << sendl;
#endif
		_narrowPhasePairs_CPU.push_back(std::make_pair(cmPair.first, cmPair.second));
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
		sout << "  now registered in CPU pair vector: " << _narrowPhasePairs_CPU.size() << " pairs." << sendl;
#endif
	}
}

void ObbTreeGPUCollisionDetection_Threaded::beginBroadPhase()
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_BEGINBROADPHASE_DEBUG
    std::cout << "=== ObbTreeGPUCollisionDetection_Threaded::beginBroadPhase() ===" << std::endl;
#endif

    BruteForceDetection::beginBroadPhase();
}

void ObbTreeGPUCollisionDetection_Threaded::endBroadPhase()
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDBROADPHASE_DEBUG
    std::cout << "=== ObbTreeGPUCollisionDetection_Threaded::endBroadPhase() ===" << std::endl;
#endif

    BruteForceDetection::endBroadPhase();
}

void ObbTreeGPUCollisionDetection_Threaded::beginNarrowPhase()
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_BEGINNARROWPHASE_DEBUG
    std::cout << "=== ObbTreeGPUCollisionDetection_Threaded::beginNarrowPhase() ===" << std::endl;
#endif

    BruteForceDetection::beginNarrowPhase();

    for (std::map<std::string, ObbTreeGPUCollisionModel<Vec3Types>*>::iterator it = m_obbTreeGPUModels.begin(); it != m_obbTreeGPUModels.end(); it++)
    {
        ObbTreeGPUCollisionModel<Vec3Types>* cm = it->second;

        Vector3 modelPosition = cm->getCachedPosition();
        Matrix3 modelOrientation;
        cm->getCachedOrientation().toMatrix(modelOrientation);

        bool skipPositionUpdate = false;
        if (!cm->isSimulated() || !cm->isMoving())
        {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_BEGINNARROWPHASE_DEBUG
            std::cout << " EXCEPTION HANDLING FOR STATIC COLLISION MODEL " << cm->getName() << std::endl;
#endif
            {
                MechanicalObject<Vec3Types>* mob = dynamic_cast<MechanicalObject<Vec3Types>*>(cm->getMechanicalState());
                if (mob)
                {
                    modelPosition = Vector3(mob->getPosition()[0][0], mob->getPosition()[0][1], mob->getPosition()[0][2]);
                    Quaternion modelRotation(mob->getPosition()[0][3], mob->getPosition()[0][4], mob->getPosition()[0][5], mob->getPosition()[0][6]);
                    modelRotation.toMatrix(modelOrientation);
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_BEGINNARROWPHASE_DEBUG
                    std::cout << " position = " << modelPosition << std::endl;
                    std::cout << " orientation = " << modelOrientation << std::endl;
#endif
                }
                else
                {
                    skipPositionUpdate = true;
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_BEGINNARROWPHASE_DEBUG
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

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_BEGINNARROWPHASE_DEBUG_VERBOSE
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

    m_testedModelPairs.clear();
    _narrowPhasePairs.clear();
	
	_narrowPhasePairs_CPU.clear();

    m_pairChecks.clear();

	if (m_detectionOutputVectors != NULL)
		m_detectionOutputVectors->clear();
}

void ObbTreeGPUCollisionDetection_Threaded::endNarrowPhase()
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << "=== ObbTreeGPUCollisionDetection_Threaded::endNarrowPhase() ===" << std::endl;
    sout << "Run BVH traversal tasks: Using " << m_numWorkerThreads.getValue() << " threads." << std::endl;

    std::cout << "=== ObbTreeGPUCollisionDetection_Threaded::endNarrowPhase(" << this->getTime() << ") ===" << std::endl;
#endif

	m_scheduler_traversal->clearTasks();
    m_scheduler_triangles->clearTasks();

#ifdef OBBTREE_GPU_COLLISION_DETECTION_THREADED_CPU_COLLISION_CHECKS
	m_scheduler_cpu_traversal->clearTasks();
#endif

    int assignedWorkUnits = scheduleBVHTraversals(_narrowPhasePairs, m_numWorkerThreads.getValue(), m_traversalTasks);

#ifdef OBBTREE_GPU_COLLISION_DETECTION_THREADED_CPU_COLLISION_CHECKS
	int assignedCPUChecks = scheduleCPUCollisionChecks(_narrowPhasePairs_CPU, m_outputsMap, m_numWorkerThreads.getValue());
#endif

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG    
	sout << " assigned worker units = " << assignedWorkUnits << ", total pair checks = " << _narrowPhasePairs.size() << std::endl;
#ifdef OBBTREE_GPU_COLLISION_DETECTION_THREADED_CPU_COLLISION_CHECKS
	sout << " assigned CPU traversals = " << assignedCPUChecks << std::endl;
#endif
#endif

#ifdef OBBTREE_GPU_COLLISION_DETECTION_THREADED_CPU_COLLISION_CHECKS
	if (assignedCPUChecks > 0)
	{
		// First add CPU traversals; these can run independently
		for (size_t k = 0; k < m_cpuTraversalTasks.size(); ++k)
		{
			m_scheduler_cpu_traversal->addTask(m_cpuTraversalTasks[k]);
		}

		m_scheduler_cpu_traversal->getScheduler()->distributeTasks();
	}
#endif

	if (assignedWorkUnits > 0)
	{
		// Then add GPU-specific tasks
		for (size_t k = 0; k < m_traversalTasks.size(); ++k)
		{
			BVHTraversalTask* task_k = m_traversalTasks[k];
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
			sout << " addTask(" << k << "): " << task_k->getNumTraversals() << " traversals assigned." << std::endl;
#endif
			m_scheduler_traversal->addTask(m_traversalTasks[k]);
		}


		m_scheduler_traversal->getScheduler()->distributeTasks();

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPUCollisionDetection_Threaded_BVH_Traversal");

		m_scheduler_traversal->getScheduler()->resumeThreads();
		m_scheduler_traversal->runTasks();
	}

#ifdef OBBTREE_GPU_COLLISION_DETECTION_THREADED_CPU_COLLISION_CHECKS
	if (assignedCPUChecks > 0)
	{
		m_scheduler_cpu_traversal->getScheduler()->resumeThreads();
		m_scheduler_cpu_traversal->runTasks();
	}
#endif
	
    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPUCollisionDetection_Threaded_BVH_Traversal");

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << "=== GPU Traversal phase done ===" << std::endl;
    for (unsigned int k = 0; k < m_traversalTasks.size(); k++)
    {
        BVHTraversalTask* task_k = m_traversalTasks[k];
        sout << " - task " << k << ": CUDA runtime = " << task_k->getElapsedTime() << " ms, thread runtime = " << task_k->getElapsedTimeInThread() << "; traversals done = " << task_k->getTraversalCalls() << std::endl;
        const std::vector<int>& taskResults = task_k->getResults();
        sout << "   results: ";
        for (std::vector<int>::const_iterator it = taskResults.begin(); it != taskResults.end(); it++)
        {
            sout << (int)*it << ";";
        }
        sout << sendl;
    }
#endif

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << "PAUSING traversal pool worker threads..." << std::endl;
#endif

#ifdef OBBTREE_GPU_COLLISION_DETECTION_THREADED_CPU_COLLISION_CHECKS
	if (assignedCPUChecks > 0)
	{
		m_scheduler_cpu_traversal->getScheduler()->pauseThreads();
	}
#endif

	if (assignedWorkUnits > 0)
	{
		m_scheduler_traversal->getScheduler()->pauseThreads();
	}

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << "=== Tasks processed by threads ===" << std::endl;
#endif

    m_scheduler_traversal->getScheduler()->dumpProcessedTasks();

    for (unsigned int k = 0; k < m_traversalTasks.size(); k++)
    {
        m_traversalTasks[k]->setFinished(false);
    }

#ifdef OBBTREE_GPU_COLLISION_DETECTION_THREADED_CPU_COLLISION_CHECKS
	for (unsigned int k = 0; k < m_cpuTraversalTasks.size(); k++)
	{
		m_cpuTraversalTasks[k]->setFinished(false);
	}
#endif

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << "=== Adding triangle check tasks ===" << std::endl;
#endif

	/*for (unsigned int k = 0; k < m_traversalTasks.size(); k++)
	{
		BVHTraversalTask* task_k = m_traversalTasks[k];
		const std::vector<int>& taskResults = task_k->getResults();
		for (unsigned int l = 0; l < taskResults.size(); l++)
		{
			if (taskResults[l] > 0)
			{

			}
		}
	}*/

    // Round robin strategy for triangle traversals
    unsigned int triTraversalIdx = 0;
    std::map<unsigned int, unsigned int> addedTriChecksPerTask;
    for (unsigned int k = 0; k < m_triangleTasks.size(); k++)
    {
        addedTriChecksPerTask.insert(std::make_pair(k,0));
    }

    for (unsigned int k = 0; k < m_traversalTasks.size(); k++)
    {
        BVHTraversalTask* task_k = m_traversalTasks[k];
        const std::vector<int>& taskResults = task_k->getResults();
        for (unsigned int l = 0; l < taskResults.size(); l++)
        {
            if (taskResults[l] > 0)
            {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                sout << " - add check with " << taskResults[l] << " potentially overlapping triangles" << std::endl;
#endif

                NarrowPhaseGPUTask* triangle_task = m_triangleTasks[triTraversalIdx % m_triangleTasks.size()];
                gProximityWorkerUnit* taskWorkerUnit = task_k->getWorkerUnit(l);
                OBBContainer* taskContainer1 = task_k->getContainer1(l);
                OBBContainer* taskContainer2 = task_k->getContainer2(l);
                const std::string taskModelName1 = task_k->getModelName1(l);
                const std::string taskModelName2 = task_k->getModelName2(l);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                sout << "   taskWorkerUnit = " << taskWorkerUnit << ", taskContainer1 = " << taskContainer1 << ", taskContainer2 = " << taskContainer2 << std::endl;
#endif

                if (taskWorkerUnit && taskContainer1 && taskContainer2 && !taskModelName1.empty() && !taskModelName2.empty())
                {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                    sout << "   adding task, all required arguments OK" << std::endl;
#endif
                    triangle_task->addTriangleCheck(taskWorkerUnit, taskContainer1, taskContainer2, task_k->getModel1(l), task_k->getModel2(l));
                    triangle_task->setBVHTraversalResult(addedTriChecksPerTask[triTraversalIdx % m_triangleTasks.size()], taskResults[l]);
                    addedTriChecksPerTask[triTraversalIdx % m_triangleTasks.size()] += 1;
                    triTraversalIdx++;
                }
            }
        }
    }

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << " added triangle checks per task: " << std::endl;
    for (std::map<unsigned int, unsigned int>::const_iterator it = addedTriChecksPerTask.begin(); it != addedTriChecksPerTask.end(); it++)
    {
        sout << " - task " << it->first << ": " << it->second << std::endl;
    }
#endif

    for (unsigned int k = 0; k < m_triangleTasks.size(); k++)
    {
        NarrowPhaseGPUTask* task_k = m_triangleTasks[k];

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
        sout << " addTask(" << k << "): Triangle traversal" << std::endl;
#endif

        m_scheduler_triangles->addTask(task_k);
    }

    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPUCollisionDetection_Threaded_Triangle_Detection");

    m_scheduler_triangles->getScheduler()->distributeTasks();

    m_scheduler_triangles->getScheduler()->resumeThreads();
    m_scheduler_triangles->runTasks();

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << "=== running triangle check tasks ===" << std::endl;
#endif

    m_scheduler_triangles->getScheduler()->pauseThreads();

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << "=== triangle check tasks complete ===" << std::endl;
    sout << "=== Tasks processed by threads ===" << std::endl;
    m_scheduler_triangles->getScheduler()->dumpProcessedTasks();
#endif

    for (unsigned int k = 0; k < m_triangleTasks.size(); k++)
    {
        m_triangleTasks[k]->setFinished(false);
    }

    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPUCollisionDetection_Threaded_Triangle_Detection");

    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPUCollisionDetection_Threaded_Contact_Processing");

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << "========================" << std::endl;
    sout << " triangle task results (" << m_triangleTasks.size() << " tasks): " << std::endl;
    sout << "========================" << std::endl;
#endif

    for (unsigned int k = 0; k < m_triangleTasks.size(); k++)
    {
        NarrowPhaseGPUTask* task_k = m_triangleTasks[k];

        const std::vector<float>& elapsedPerTest = task_k->getElapsedTimePerTest();
        std::map<unsigned int, std::vector<int> >& contactIDs = task_k->getContactIDs();
        std::map<unsigned int, std::vector<double> >& contactDistances = task_k->getContactDistances();
        std::map<unsigned int, std::vector<Vector3> >& contactPoints_0 = task_k->getContactPoints_0();
        std::map<unsigned int, std::vector<Vector3> >& contactPoints_1 = task_k->getContactPoints_1();
        std::map<unsigned int, std::vector<Vector3> >& contactNormals = task_k->getContactNormals();
        std::map<unsigned int, std::vector<gProximityContactType> >& contactTypes = task_k->getContactTypes();
        std::map<unsigned int, std::vector<std::pair<int, int> > >& contactElements = task_k->getContactElements();
        std::map<unsigned int, std::vector<std::pair<int, int> > >& contactElementsFeatures = task_k->getContactElementsFeatures();

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
        sout << "  - task " << k << ", id = " << task_k->getTaskID() << ": " << contactIDs.size() << " contacts vectors" << std::endl;
#endif

        if (contactIDs.size() > 0)
        {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
            sout << "     registered pair checks: " << m_pairChecks.size() << std::endl;
            for (std::map<unsigned int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> >::const_iterator it = m_pairChecks.begin(); it != m_pairChecks.end(); it++)
                sout << "       - " << it->first << ": " << it->second.first->getName() << " -- " << it->second.second->getName() << std::endl;

            sout << "    contact ID vectors:" << std::endl;
#endif
            for (std::map<unsigned int, std::vector<int> >::iterator contacts_it = contactIDs.begin(); contacts_it != contactIDs.end(); contacts_it++)
            {
                std::vector<int>& contactIDVec = contacts_it->second;
                if (contactIDVec.size() > 0)
                {

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                    sout << "     result index " << contacts_it->first << ": " << contactIDVec.size() << " contacts"; 					
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG_VERBOSE                    
					sout << " -- ";
					for (std::vector<int>::const_iterator id_it = contactIDVec.begin(); id_it != contactIDVec.end(); id_it++)
						sout << (*id_it) << ";";

                    sout << sendl;
#else
					sout << sendl;
#endif
                    sout << "      for model pair " << task_k->getModelName1(contacts_it->first) << " -- " << task_k->getModelName2(contacts_it->first) << std::endl;
#endif
                    bool allContactDataVectorsFound = true;

                    if (contactDistances.find(contacts_it->first) != contactDistances.end())
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     found matching contactDistances vector: " << contactDistances[contacts_it->first].size() << " entries" << std::endl;
#endif
                    }
                    else
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     no contactDistances vector found for test index " << contacts_it->first << std::endl;
#endif
                        allContactDataVectorsFound = false;
                    }

                    if (contactPoints_0.find(contacts_it->first) != contactPoints_0.end())
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     found matching contactPoints_0 vector: " << contactPoints_0[contacts_it->first].size() << " entries" << std::endl;
#endif
                    }
                    else
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     no contactPoints_0 vector found for test index " << contacts_it->first << std::endl;
#endif
                        allContactDataVectorsFound = false;
                    }

                    if (contactPoints_1.find(contacts_it->first) != contactPoints_1.end())
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     found matching contactPoints_1 vector: " << contactPoints_1[contacts_it->first].size() << " entries" << std::endl;
#endif
                    }
                    else
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     no contactPoints_1 vector found for test index " << contacts_it->first << std::endl;
#endif
                        allContactDataVectorsFound = false;
                    }

                    if (contactNormals.find(contacts_it->first) != contactNormals.end())
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     found matching contactNormals vector: " << contactNormals[contacts_it->first].size() << " entries" << std::endl;
#endif
                    }
                    else
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     no contactNormals vector found for test index " << contacts_it->first << std::endl;
#endif
                        allContactDataVectorsFound = false;
                    }

                    if (contactTypes.find(contacts_it->first) != contactTypes.end())
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     found matching contactTypes vector: " << contactTypes[contacts_it->first].size() << " entries" << std::endl;
#endif
                    }
                    else
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     no contactTypes vector found for test index " << contacts_it->first << std::endl;
#endif
                        allContactDataVectorsFound = false;
                    }

                    if (contactElements.find(contacts_it->first) != contactElements.end())
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     found matching contactElements vector: " << contactElements[contacts_it->first].size() << " entries" << std::endl;
#endif
                    }
                    else
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     no contactElements vector found for test index " << contacts_it->first << std::endl;
#endif
                        allContactDataVectorsFound = false;
                    }

                    if (contactElementsFeatures.find(contacts_it->first) != contactElementsFeatures.end())
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     found matching contactElementsFeatures vector: " << contactElementsFeatures[contacts_it->first].size() << " entries" << std::endl;
#endif
                    }
                    else
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     no contactElementsFeatures vector found for test index " << contacts_it->first << std::endl;
#endif
                        allContactDataVectorsFound = false;
                    }

                    if (allContactDataVectorsFound)
                    {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        sout << "     all required data vectors present; proceed to contact point generation" << std::endl;
#endif
                        ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = task_k->getModel1(contacts_it->first);
                        ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = task_k->getModel2(contacts_it->first);

                        if (obbModel1 && obbModel2)
                        {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                            sout << "     found matching pair check in checks map: " << obbModel1->getName() << " -- " << obbModel2->getName() << std::endl;
#endif
                            std::pair<core::CollisionModel*,core::CollisionModel*> ghostPair(obbModel1,obbModel2);

                            double minDistance;
                            for (unsigned int q = 0; q < contactIDVec.size(); q++)
                            {
                                minDistance = std::min(minDistance, std::fabs(contactDistances[contacts_it->first][q]));
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG_VERBOSE
                                sout << contactDistances[contacts_it->first][q] << " " << std::endl;
#endif
                            }
                            double ghostTolerance = std::max(obbModel1->getGhostObjectTolerance(),obbModel2->getGhostObjectTolerance());
                            if (minDistance <= (ghostTolerance+m_contactDistance))
                            {
                                collidingModels.push_back(ghostPair);
                            }

                            // check for ghost models
							// FA: I suppose the intention is to ignore contact point generation for ghost objects.
                            if (obbModel1->isGhostObject() || obbModel2->isGhostObject())
								continue;

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
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG_VERBOSE
                                sout << "     got DetectionOutputVector instance(s)" << std::endl;
#endif
                                for (unsigned int q = 0; q < contactIDVec.size(); q++)
                                {
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG_VERBOSE
                                    sout << "      - add contact point " << q << ": id = " << contactIDVec[q] << ", type = " << contactTypes[contacts_it->first][q] << ", distance = " << contactDistances[contacts_it->first][q] << ";" <<
                                            "point0 = " << contactPoints_0[contacts_it->first][q] << ", point1 = " << contactPoints_1[contacts_it->first][q] << ", normal = " << contactNormals[contacts_it->first][q] << ";" <<
                                            " elements = " << contactElements[contacts_it->first][q].first << " - " << contactElements[contacts_it->first][q].second << ";" <<
                                            " features = " << contactElementsFeatures[contacts_it->first][q].first << " - " << contactElementsFeatures[contacts_it->first][q].second << ";" <<
                                            std::endl;
#endif

                                    discreteOutputs->resize(discreteOutputs->size() + 1);
                                    sofa::core::collision::DetectionOutput *detection = &*(discreteOutputs->end() - 1);

                                    detection->id = contactIDVec[q];

                                    detection->point[0] = contactPoints_0[contacts_it->first][q];
                                    detection->point[1] = contactPoints_1[contacts_it->first][q];

                                    detection->normal = contactNormals[contacts_it->first][q];

                                    /**** ???? */
                                    detection->value = detection->normal.norm();
                                    detection->normal /= detection->value;

                                    //detection->value -= m_contactDistance; //!? (BE) this looks like nonsence -> commented out
									// FA: Then it does not matter if this was copy/pasted from a default SOFA implementation, does it?

                                    detection->contactType = (sofa::core::collision::DetectionOutputContactType) contactTypes[contacts_it->first][q];

                                    if (detection->contactType == COLLISION_LINE_LINE)
                                    {
                                        detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, contactElements[contacts_it->first][q].first); // << CollisionElementIterator

                                        detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, contactElements[contacts_it->first][q].second); // << CollisionElementIterator

                                        detection->elemFeatures.first = contactElementsFeatures[contacts_it->first][q].first;
                                        detection->elemFeatures.second = contactElementsFeatures[contacts_it->first][q].second;
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG_VERBOSE
                                        sout << "        LINE_LINE contact: elements = " << detection->elem.first.getIndex() << "," << detection->elem.second.getIndex() << ", features = " << detection->elemFeatures.first << "," << detection->elemFeatures.second << std::endl;
#endif
                                    }
                                    else if (detection->contactType == COLLISION_VERTEX_FACE)
                                    {
                                        if (contactElementsFeatures[contacts_it->first][q].second == -1)
                                        {
                                            detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, contactElements[contacts_it->first][q].first); // << CollisionElementIterator
                                            detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, contactElements[contacts_it->first][q].second); // << CollisionElementIterator
                                            detection->elemFeatures.first = contactElementsFeatures[contacts_it->first][q].first;
                                        }
                                        else if (contactElementsFeatures[contacts_it->first][q].first == -1)
                                        {
                                            detection->elem.first = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel1, contactElements[contacts_it->first][q].first); // << CollisionElementIterator
                                            detection->elem.second = sofa::core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(obbModel2, contactElements[contacts_it->first][q].second); // << CollisionElementIterator
                                            detection->elemFeatures.second = contactElementsFeatures[contacts_it->first][q].second;
                                        }

                                        if (detection->elem.first.getIndex() < 0 || detection->elem.second.getIndex() < 0) {
                                            std::cerr << "ERROR: Should not happen" << std::endl;
                                        }
                                    } else {
                                        std::cerr << "ERROR: contact: " << detection->contactType << std::endl;
										// FA: SERIOUSLY???
                                        // exit(0);
										continue;
                                    }
                                }
                            }
                        }
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                        else
                        {
                            sout << "     NO matching pair check located in checks map: This should not happen!" << std::endl;
                        }
#endif
                    }
                }
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
                else
                {
                    sout << "     result index " << contacts_it->first << ": No contacts detected" << std::endl;
                }
#endif
            }
        }
    }

    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPUCollisionDetection_Threaded_Contact_Processing");

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << "==== Iteration summary ====" << sendl;

    sout << " -- recorded pair check combinations --" << std::endl;
    for (std::map<unsigned int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> >::const_iterator it = m_pairChecks.begin(); it != m_pairChecks.end(); it++)
    {
        sout << " - " << it->first << ": " << it->second.first->getName() << " -- " << it->second.second->getName() << std::endl;
    }

    sout << " -- timing statistics for BVH tests --" << std::endl;
    for (unsigned int k = 0; k < m_traversalTasks.size(); k++)
    {
        BVHTraversalTask* task_k = m_traversalTasks[k];
        const std::vector<float>& elapsedPerTest = task_k->getElapsedTimePerTest();

        sout << " - task " << k << " (" << task_k->getTaskID() << "): total = " << (task_k->getElapsedTime() / 1000000.0f) << " ms; traversals = ";
        for (unsigned int m = 0; m < elapsedPerTest.size(); m++)
			sout << elapsedPerTest[m] / 1000000.0f << " ms;";

        sout << sendl;

        sout << "   CPU step duration: ";
        const std::vector<std::pair<std::string, int64_t> >& elapsedCPUSteps = task_k->getElapsedTimeCPUStep();
        float cpuDuration_ms = 0.0f;
        for (std::vector<std::pair<std::string, int64_t> >::const_iterator it = elapsedCPUSteps.begin(); it != elapsedCPUSteps.end(); it++)
        {
            float stepDuration = it->second / 1000000.0f;
            cpuDuration_ms += stepDuration;
            sout << it->first << ": " << stepDuration << " ms;";
        }
        sout << sendl;
        sout << "   Total CPU duration = " << cpuDuration_ms << " ms"<< sendl;
    }
#endif

	sofa::component::visualmodel::OglModel::resetOglObjectStates(0);

#ifdef _WIN32
#if ROBOTCONNECTOR_COLLISION_STOP_ENABLED
	for (unsigned int k = 0; k < m_triangleTasks.size(); k++)
	{
		NarrowPhaseGPUTask* task_k = m_triangleTasks[k];
		std::vector<int> intersections = task_k->getTriangleIntersectionResults();
		for (int i = 0; i < task_k->getResultSize(); i++)
		{
			if (intersections.at(i) > 0)
			{
				sout << sendl << "#collision between " << task_k->getModelName1(i) << " and " << task_k->getModelName2(i) << " : " << intersections.at(i) << sendl;
				sofa::component::visualmodel::OglModel::setOglObjectState(task_k->getModelName1(i), 1);
				sofa::component::visualmodel::OglModel::setOglObjectState(task_k->getModelName2(i), 1);

				// (BE) Check if robot and human collide -> send stop
				// FA: Hard-coded? SERIOUSLY???
				std::string humanMesh = "geo55";
				std::list<std::string> robotMeshes;
				robotMeshes.push_back("geo0");
				robotMeshes.push_back("geo1");
				robotMeshes.push_back("geo2");
				robotMeshes.push_back("geo3");
				robotMeshes.push_back("geo4");
				robotMeshes.push_back("geo5");
				robotMeshes.push_back("geo6");

				std::string c1 = task_k->getModelName1(i);
				std::string c2 = task_k->getModelName2(i);

				if (c1 == humanMesh || c2 == humanMesh)
				{
					std::string collObj = c2 == humanMesh ? c1 : c2;

					if (std::find(robotMeshes.begin(), robotMeshes.end(), collObj) != robotMeshes.end())
					{

						std::vector<RobotConnector*> moV;
						sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<RobotConnector, std::vector<RobotConnector* > > cb(&moV);

						getContext()->getObjects(TClassInfo<RobotConnector>::get(), cb, TagSet(), BaseContext::SearchRoot);
						if (moV.size() > 0) { // (BE) stop robot
							if (((RobotConnector*)moV[0])->isCollissionStopEnabled()) {
								((RobotConnector*)moV[0])->stopProgram();
							}
						}

					}
				}
			}
		}
	}
#endif //ROBOTCONNECTOR_COLLISION_STOP_ENABLED
#endif

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << " -- timing statistics for triangle tests --" << sendl;
    for (unsigned int k = 0; k < m_triangleTasks.size(); k++)
    {
        NarrowPhaseGPUTask* task_k = m_triangleTasks[k];
        const std::vector<float>& elapsedPerTest = task_k->getElapsedTimePerTest();

		sout << " - task " << k << " (" << task_k->getTaskID() << "): total = " << task_k->getElapsedTime() / 1000000.0f << " ms" << sendl;
		if (elapsedPerTest.size() > 0)
		{
			sout << "   bin checks = ";
			for (unsigned int m = 0; m < elapsedPerTest.size(); m++)
				sout << elapsedPerTest[m] / 1000000.0f << " ms;";
		}

        sout << sendl;
    }
#endif

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << " -- tasks processed by traversal scheduler --" << sendl;
    m_scheduler_traversal->dumpProcessedTasks();

    sout << " -- tasks processed by triangles scheduler --" << sendl;
    m_scheduler_triangles->dumpProcessedTasks();

	sout << " -- tasks processed by CPU scheduler --" << sendl;
	m_scheduler_cpu_traversal->dumpProcessedTasks();

	sout << " ---- CPU scheduler task details ----" << sendl;
	for (unsigned int k = 0; k < m_cpuTraversalTasks.size(); k++)
	{
		CPUCollisionCheckTask* task_k = m_cpuTraversalTasks[k];
		const std::vector<float>& elapsedPerTest = task_k->getElapsedTimePerTest();

		sout << " - task " << k << " (" << task_k->getTaskID() << "): total = " << task_k->getElapsedTime() / 1000000.0f << " ms" << sendl;
		
		if (elapsedPerTest.size() > 0)
		{
			sout << "   bin checks: " << sendl;

			for (unsigned int m = 0; m < elapsedPerTest.size(); m++)
			{
				std::vector<std::pair<std::string, std::string> >& assignedPairs = _narrowPhasePairs_CPU_taskAssignment[k];
				sout << "   * pair " << m << ": " << assignedPairs[m].first << " -- " << assignedPairs[m].second << " runtime = " << elapsedPerTest[m] / 1000000.0f << " ms" << sendl;
			}
		}
		sout << sendl;
	}

	sout << " ---- Detection output vectors ----" << sendl;
	for (unsigned int k = 0; k < m_cpuTraversalTasks.size(); k++)
	{
		CPUCollisionCheckTask* task_k = m_cpuTraversalTasks[k];
		std::vector<core::collision::DetectionOutputVector*>& detection_outputs = task_k->getDetectionOutputs();
		std::vector<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& colliding_pairs = task_k->getCollidingPairs();

		sout << " - task " << k << " (" << task_k->getTaskID() << ") detection outputs size =  " << detection_outputs.size() << " for colliding pairs size = " << colliding_pairs.size() << sendl;
		for (size_t l = 0; l < detection_outputs.size(); ++l)
		{
			if (detection_outputs[l] != NULL)
			{
				sout << "  - DetectionOutputVector " << l << " for pair " << colliding_pairs[l].first->getName() << " -- " << colliding_pairs[l].second->getName() << " empty: " << detection_outputs[l]->empty() << ", num.of elements = " << detection_outputs[l]->size() << sendl;
				sout << "    -> Corresponds to: " << this->_narrowPhasePairs_CPU_modelAssociations[colliding_pairs[l]].first << " <-> " << this->_narrowPhasePairs_CPU_modelAssociations[colliding_pairs[l]].second << sendl;
				std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> collision_pair = std::make_pair(colliding_pairs[l].first, colliding_pairs[l].second);
				m_detectionOutputVectors->insert(std::make_pair(collision_pair, detection_outputs[l]));
			}
			/*else
			{
				sout << "  - DetectionOutputVector " << l << " for pair " << colliding_pairs[l].first->getName() << " -- " << colliding_pairs[l].second->getName() << " NOT INSTANTIATED!" << sendl;
			}*/
		}
	}

    sout << "==== Iteration summary ====" << sendl;
#endif


#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
	std::cout << " -- Fake-gripping --" << std::endl;
#endif

	if (m_fakeGripping_Events.size() > 0)
	{
		int rulesChecked = 0;

		std::vector<int> matchingFakeGrippingRules;
		std::map<int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> > matchingFakeGrippingOrModels;
		std::multimap<int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> > matchingFakeGrippingAndModels;

		// FAKE GRIPPING
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
		std::cout << "=========================================================================================" << std::endl;
		std::cout << "FAKE GRIPPING RULE CHECKS BEGIN" << std::endl;
		std::cout << "=========================================================================================" << std::endl;
#endif

		for (std::vector<FakeGripping_Event_Container>::iterator fg_it = m_fakeGripping_Events.begin(); fg_it != m_fakeGripping_Events.end(); fg_it++)
		{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
			std::cout << " check rule " << rulesChecked << ": " << fg_it->activeModel << ", m_active_FakeGripping_Event = " << m_active_FakeGripping_Event << std::endl;
#endif

			bool orCondition_fulfilled = false;
			std::vector<bool> andCondition_fulfilled;
			bool andFulfilled = true;
			bool ruleTransitionDone = false;
			andCondition_fulfilled.resize(fg_it->contactModels.size());
			for (std::map<unsigned int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> >::const_iterator it = m_pairChecks.begin(); it != m_pairChecks.end(); it++)
			{

				ObbTreeGPUCollisionModel<Vec3Types>* model1 = it->second.first;
				ObbTreeGPUCollisionModel<Vec3Types>* model2 = it->second.second;

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
				std::cout << "  * check pair " << model1->getName() << " -- " << model2->getName() << std::endl;
#endif

				if (!hasIntersections(model1->getName(), model2->getName()))
					continue;

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
				std::cout << "  intersection between model1 = " << model1->getName() << " and model2 = " << model2->getName() << std::endl;
#endif

				if (fg_it->activeModel.compare(model1->getName()) == 0)
				{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
					std::cout << "    - model1 = " << model1->getName() << " in collision; fake-gripping event check" << std::endl;
#endif

					for (unsigned int k = 0; k < fg_it->contactModels.size(); k++)
					{
						if (model2->getName().compare(fg_it->contactModels[k]) == 0)
						{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "       in collision with contactModel = " << model2->getName() << std::endl;
#endif

							orCondition_fulfilled = true;
							andCondition_fulfilled[k] = true;
							matchingFakeGrippingOrModels.insert(std::make_pair(rulesChecked, std::make_pair(model1, model2)));
							matchingFakeGrippingAndModels.insert(std::make_pair(rulesChecked, std::make_pair(model1, model2)));
						}
					}
				}
				else if (fg_it->activeModel.compare(model2->getName()) == 0)
				{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
					std::cout << "    - model2 = " << model1->getName() << " in collision; fake-gripping event check" << std::endl;
#endif
					for (unsigned int k = 0; k < fg_it->contactModels.size(); k++)
					{
						if (model1->getName().compare(fg_it->contactModels[k]) == 0)
						{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "       in collision with contactModel = " << model1->getName() << std::endl;
#endif
							orCondition_fulfilled = true;
							andCondition_fulfilled[k] = true;
							matchingFakeGrippingOrModels.insert(std::make_pair(rulesChecked, std::make_pair(model2, model1)));
							matchingFakeGrippingAndModels.insert(std::make_pair(rulesChecked, std::make_pair(model2, model1)));
						}
					}
				}
			}

			if (fg_it->contactCondition == FG_OR)
			{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
				std::cout << "  ==> rule " << rulesChecked << ": orCondition fulfilled = " << orCondition_fulfilled << std::endl;
#endif
				if (orCondition_fulfilled)
				{
					matchingFakeGrippingRules.push_back(rulesChecked);
					ruleTransitionDone = true;
				}
			}
			else if (fg_it->contactCondition == FG_AND)
			{
				for (unsigned int m = 0; m < andCondition_fulfilled.size(); m++)
				{
					if (!andCondition_fulfilled[m])
					{
						andFulfilled = false;
						break;
					}
				}

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
				std::cout << " ==> rule " << rulesChecked << ": andCondition_fulfilled = " << andFulfilled << std::endl;
#endif

				if (andFulfilled)
				{
					matchingFakeGrippingRules.push_back(rulesChecked);
					ruleTransitionDone = true;
				}
			}
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
			else
			{
				std::cout << " ==> rule " << rulesChecked << ": NO CONDITION fulfilled" << std::endl;
			}
#endif
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
			if (andFulfilled || orCondition_fulfilled)
			{
				if (m_active_FakeGripping_Event != -1)
				{
					if (m_previous_FakeGripping_Event != -1)
					{
						if (m_active_FakeGripping_Event > m_previous_FakeGripping_Event && ruleTransitionDone)
						{
							std::cout << "  ==> Fake-Gripping RULE Transition from " << m_previous_FakeGripping_Event << " to " << m_active_FakeGripping_Event << "; THIS REMOVES AN EXISTING RigidMapping and CREATES a RigidMapping" << std::endl;
						}
						else
						{
							std::cout << "  ==> Remaining in fake-gripping rule = " << m_active_FakeGripping_Event << std::endl;
						}
					}
					else
					{
						std::cout << " ==> FIRST Fake-Gripping rule activated: " << m_active_FakeGripping_Event << "; THIS CREATES a RigidMapping" << std::endl;
					}
				}
				else
				{
					std::cout << "  ==> Fake gripping not active" << std::endl;
				}
			}
			else
			{
				std::cout << "  ==> Fake-gripping rule " << rulesChecked << " NOT fulfilled" << std::endl;
			}
#endif

			rulesChecked++;
		}

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
		std::cout << "highest/last rule is no. = " << ((rulesChecked - 1) > 0 ? (rulesChecked - 1) : -1) << std::endl;
#endif

		bool lastRuleFound = false;
		if (matchingFakeGrippingRules.size() > 0)
		{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
			std::cout << "==============================================" << std::endl;
			std::cout << " Matching fake-gripping rules: " << matchingFakeGrippingRules.size() << ", highest in rule sequence = " << matchingFakeGrippingRules.at(matchingFakeGrippingRules.size() - 1) << std::endl;
			std::cout << "==============================================" << std::endl;
#endif

			for (unsigned int k = 0; k < matchingFakeGrippingRules.size(); k++)
			{
				if (matchingFakeGrippingRules[k] == rulesChecked)
					lastRuleFound = true;

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
				std::cout << "  - " << matchingFakeGrippingRules[k] << "; already activated before = " << m_fakeGripping_Activated_Rules[matchingFakeGrippingRules[k]] << ": " << m_fakeGripping_Events[matchingFakeGrippingRules[k]].activeModel << " against ";
				for (unsigned int l = 0; l < m_fakeGripping_Events[matchingFakeGrippingRules[k]].contactModels.size(); l++)
					std::cout << m_fakeGripping_Events[matchingFakeGrippingRules[k]].contactModels[l] << ";";

				std::cout << std::endl;
#endif
			}
		}

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
		std::cout << "highest/last rule active/to activate = " << lastRuleFound << std::endl;

		std::cout << "======================================================================================" << std::endl;
		std::cout << "BEFORE RULE UPDATE: ACTIVE fake gripping rule = " << m_active_FakeGripping_Event << ", PREVIOUS fake gripping rule = " << m_previous_FakeGripping_Event << std::endl;
		std::cout << "======================================================================================" << std::endl;
#endif

		if (matchingFakeGrippingRules.size() > 0)
		{
			bool fakeGrippingRuleChanged = false;
			if (!m_fakeGripping_Activated_Rules[matchingFakeGrippingRules.at(matchingFakeGrippingRules.size() - 1)])
			{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
				std::cout << " FIRST ACTIVATION of new rule " << matchingFakeGrippingRules.at(matchingFakeGrippingRules.size() - 1) << std::endl;
#endif

				m_fakeGripping_Activated_Rules[matchingFakeGrippingRules.at(matchingFakeGrippingRules.size() - 1)] = true;

				m_previous_FakeGripping_Event = m_active_FakeGripping_Event;
				m_active_FakeGripping_Event = matchingFakeGrippingRules.back();

				fakeGrippingRuleChanged = true;
			}
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
			else
			{
				std::cout << " REMAINING in active rule " << m_active_FakeGripping_Event << std::endl;
			}

			std::cout << "======================================================================================" << std::endl;
			std::cout << "AFTER  RULE UPDATE: ACTIVE fake gripping rule = " << m_active_FakeGripping_Event << ", PREVIOUS fake gripping rule = " << m_previous_FakeGripping_Event << ", fakeGrippingRuleChanged = " << fakeGrippingRuleChanged << std::endl;
			std::cout << "======================================================================================" << std::endl;
#endif

			if (fakeGrippingRuleChanged && (m_previous_FakeGripping_Event != m_active_FakeGripping_Event))
			{
				sofa::simulation::Node* node1 = NULL;
				sofa::simulation::Node* node2 = NULL;

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
				std::cout << " SWITCHING ACTIVE RULES: TO m_active_FakeGripping_Event = " << m_active_FakeGripping_Event << ", AWAY FROM m_previous_FakeGripping_Event = " << m_previous_FakeGripping_Event << std::endl;
#endif

				// AND condition
				if (matchingFakeGrippingAndModels.find(m_active_FakeGripping_Event) != matchingFakeGrippingAndModels.end())
				{
					std::pair<std::multimap<int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> >::iterator, std::multimap<int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> >::iterator> andConditionModels;
					andConditionModels = matchingFakeGrippingAndModels.equal_range(m_active_FakeGripping_Event);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG                
					std::cout << " found m_active_FakeGripping_Event in and-condition map; models = ";
					for (std::multimap<int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> >::iterator ac_it = andConditionModels.first; ac_it != andConditionModels.second; ac_it++)
					{
						std::cout << ac_it->second.first->getName() << " -- " << ac_it->second.second->getName() << ";";
					}
					std::cout << std::endl;
#endif

					std::multimap<int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> >::iterator first_and_it = matchingFakeGrippingAndModels.find(m_active_FakeGripping_Event);
					node1 = static_cast<sofa::simulation::Node*>(first_and_it->second.first->getContext());
					node2 = static_cast<sofa::simulation::Node*>(first_and_it->second.second->getContext());

					ObbTreeGPUCollisionModel<Vec3Types>* leadingObject = first_and_it->second.first;
					ObbTreeGPUCollisionModel<Vec3Types>* followingObject = first_and_it->second.second;

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
					std::cout << " ==> Collision models associated with rule: leadingObject = " << leadingObject->getName() << ", followingObject = " << followingObject->getName() << std::endl;
#endif

					m_fakeGripping_CollisionCheck_Exceptions.insert(std::make_pair(leadingObject->getName(), followingObject->getName()));

					sofa::simulation::Node* parent1 = (sofa::simulation::Node*)(node1->getParents()[0]);
					std::vector<sofa::component::container::MechanicalObject<Rigid3Types> *> mobj_1;
					sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::container::MechanicalObject<Rigid3Types>, std::vector<sofa::component::container::MechanicalObject<Rigid3Types>* > > mobj_cb_1(&mobj_1);
					parent1->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::container::MechanicalObject<Rigid3Types> >::get(), mobj_cb_1, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

					std::vector<sofa::component::mass::UniformMass<Rigid3Types, Rigid3dMass> *> m_uniformMass_1;
					sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::mass::UniformMass<Rigid3Types, Rigid3dMass>, std::vector<sofa::component::mass::UniformMass<Rigid3Types, Rigid3dMass>* > > m_uniformMass_cb_1(&m_uniformMass_1);
					parent1->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::mass::UniformMass<Rigid3Types, Rigid3dMass> >::get(), m_uniformMass_cb_1, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

					sofa::simulation::Node* parent2 = (sofa::simulation::Node*)(node2->getParents()[0]);
					std::vector<sofa::component::container::MechanicalObject<Rigid3Types> *> mobj_2;
					sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::container::MechanicalObject<Rigid3Types>, std::vector<sofa::component::container::MechanicalObject<Rigid3Types>* > > mobj_cb_2(&mobj_2);
					parent2->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::container::MechanicalObject<Rigid3Types> >::get(), mobj_cb_2, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

					std::vector<sofa::component::container::MechanicalObject<Vec3Types> *> inner_mobj_1;
					sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::container::MechanicalObject<Vec3Types>, std::vector<sofa::component::container::MechanicalObject<Vec3Types>* > > inner_mobj_cb_1(&inner_mobj_1);
					parent1->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::container::MechanicalObject<Vec3Types> >::get(), inner_mobj_cb_1, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

					std::vector<sofa::component::container::MechanicalObject<Vec3Types> *> inner_mobj_2;
					sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::container::MechanicalObject<Vec3Types>, std::vector<sofa::component::container::MechanicalObject<Vec3Types>* > > inner_mobj_cb_2(&inner_mobj_2);
					parent2->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::container::MechanicalObject<Vec3Types> >::get(), inner_mobj_cb_2, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
					std::cout << "  MechanicalObject<Rigid3Types> instances:" << std::endl;
					std::cout << "    parent1 child[0] = " << mobj_1[0]->getName() << " of type " << mobj_1[0]->getClassName() << std::endl;
					std::cout << "    parent2 child[0] = " << mobj_2[0]->getName() << " of type " << mobj_2[0]->getClassName() << std::endl;

					std::cout << " MechanicalObject<Vec3Types> instances:" << std::endl;
					std::cout << "    parent1 " << parent1->getName() << " size() = " << inner_mobj_1.size() << ": ";
					for (unsigned int q = 0; q < inner_mobj_1.size(); q++)
						std::cout << " " << inner_mobj_1[q]->getName() << ";";

					std::cout << std::endl;

					std::cout << "    parent2 " << parent2->getName() << " size() = " << inner_mobj_2.size() << ": ";
					for (unsigned int q = 0; q < inner_mobj_2.size(); q++)
						std::cout << " " << inner_mobj_2[q]->getName() << ";";

					std::cout << std::endl;

					std::cout << "==========================================================" << std::endl;
					std::cout << " UniformMass<Rigid3Types, Rigid3dMass> instances:" << m_uniformMass_1.size() << std::endl;
					for (unsigned int q = 0; q < m_uniformMass_1.size(); q++)
						std::cout << " " << m_uniformMass_1[q]->getName() << ";";

					std::cout << std::endl;
					std::cout << "==========================================================" << std::endl;
#endif
					// Detach Fake Gripping
					if (m_activeFakeGrippingRules_Testing_Slaves.find(m_previous_FakeGripping_Event) != m_activeFakeGrippingRules_Testing_Slaves.end())
					{
						std::pair <std::multimap<int, std::pair<std::pair<sofa::simulation::Node*, sofa::simulation::Node*>, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr > >::iterator,
							std::multimap<int, std::pair<std::pair<sofa::simulation::Node*, sofa::simulation::Node*>, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr > >::iterator>
							ret = m_activeFakeGrippingRules_Testing_Slaves.equal_range(m_previous_FakeGripping_Event);

						std::pair<std::multimap<int, std::pair<std::pair<sofa::core::State<Rigid3Types>*, sofa::core::State<Vec3Types>* >, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* > >::iterator,
							std::multimap<int, std::pair<std::pair<sofa::core::State<Rigid3Types>*, sofa::core::State<Vec3Types>* >, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* > >::iterator >
							inner_mappings_ret = m_activeFakeGrippingRules_InnerMappings.equal_range(m_previous_FakeGripping_Event);

						MechanicalObject<Rigid3Types> *previousBaseObjectM = (MechanicalObject<Rigid3Types> *) inner_mappings_ret.first->second.first.first;

						sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr prevRigidMappingM = ret.first->second.second; //m_activeFakeGrippingRules_Testing_Slaves[m_previous_FakeGripping_Event].second;

						MechanicalObject<Rigid3Types> *currentLeadingObject = (MechanicalObject<Rigid3Types> *)prevRigidMappingM->getFromModel();
						const Rigid3Types::VecCoord c = currentLeadingObject->getPosition();
						Vec3d currentLeadingObjectPos(c[0][0], c[0][1], c[0][2]);
						Quat currentLeadingObjectRot(c[0][3], c[0][4], c[0][5], c[0][6]);

						Vec3d lastLeadingObjectPos = prevRigidMappingM->lastObjPos.getValue();
						Quat lastLeadingObjectRot = prevRigidMappingM->lastObjRot.getValue();

						Rigid3Types::VecCoord n = previousBaseObjectM->getPosition();

						Vec3d newObjectPos(n[0][0], n[0][1], n[0][2]);
						Quat newObjectRot(n[0][3], n[0][4], n[0][5], n[0][6]);

						// Translate back by laseLeading, rotate back by lastLeading, rotate to currentLeading, translate to currentLeadinng)
						newObjectPos = currentLeadingObjectRot.rotate(lastLeadingObjectRot.inverseRotate(newObjectPos - lastLeadingObjectPos)) + currentLeadingObjectPos;
						// And the same only for the rotation quaternion
						newObjectRot = currentLeadingObjectRot*lastLeadingObjectRot.inverse()*newObjectRot;

						n[0][0] = newObjectPos[0];
						n[0][1] = newObjectPos[1];
						n[0][2] = newObjectPos[2];

						n[0][3] = newObjectRot[0];
						n[0][4] = newObjectRot[1];
						n[0][5] = newObjectRot[2];
						n[0][6] = newObjectRot[3];

						previousBaseObjectM->setPosition(n);
						previousBaseObjectM->setFreePosition(n);

						// REMOVE ALL RigidMappings
						for (std::multimap<int, std::pair<std::pair<sofa::simulation::Node*, sofa::simulation::Node*>, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr > >::iterator fuck_it = ret.first; fuck_it != ret.second; fuck_it++)
						{
							sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr prevRigidMapping = fuck_it->second.second; //m_activeFakeGrippingRules_Testing_Slaves[m_previous_FakeGripping_Event].second;

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "  Remove RigidMapping for previously active rule " << m_previous_FakeGripping_Event << " before switching to new RigidMapping" << std::endl;
							sofa::simulation::Node* prevRigidMappingParent = fuck_it->second.first.first; //m_activeFakeGrippingRules_Testing_Slaves[m_previous_FakeGripping_Event].first.first;

							std::cout << "   remove prevRigidMapping = " << prevRigidMapping->getName() << std::endl;
							std::cout << "   remove from node = " << prevRigidMappingParent->getName() << std::endl;
#endif

							//prevRigidMapping->removeObject(prevRigidMapping->getName());
							prevRigidMapping->cleanup();
							prevRigidMapping.reset();

							if (m_activeFakeGrippingRules_InnerMappings.find(m_previous_FakeGripping_Event) != m_activeFakeGrippingRules_InnerMappings.end())
							{
								std::pair<std::multimap<int, std::pair<std::pair<sofa::core::State<Rigid3Types>*, sofa::core::State<Vec3Types>* >, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* > >::iterator,
									std::multimap<int, std::pair<std::pair<sofa::core::State<Rigid3Types>*, sofa::core::State<Vec3Types>* >, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* > >::iterator >
									inner_mappings_ret = m_activeFakeGrippingRules_InnerMappings.equal_range(m_previous_FakeGripping_Event);

								std::pair<std::multimap<int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> >::iterator, std::multimap<int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> >::iterator> attached_objects_ret =
									m_fakeGripping_AttachedObjects.equal_range(m_previous_FakeGripping_Event);

								std::pair<std::multimap<int, sofa::simulation::Node::SPtr>::iterator, std::multimap<int, sofa::simulation::Node::SPtr>::iterator > attached_nodes_ret =
									m_fakeGripping_AttachedNodes.equal_range(m_previous_FakeGripping_Event);

								std::multimap<int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> >::iterator ato_it = attached_objects_ret.first;
								std::multimap<int, sofa::simulation::Node::SPtr>::iterator atn_it = attached_nodes_ret.first;

								for (std::multimap<int, std::pair<std::pair<sofa::core::State<Rigid3Types>*, sofa::core::State<Vec3Types>* >, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* > >::iterator im_it = inner_mappings_ret.first; im_it != inner_mappings_ret.second; im_it++)
								{
									sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* prevInnerMapping = im_it->second.second; //m_activeFakeGrippingRules_InnerMappings[m_previous_FakeGripping_Event].second;

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG                                
									std::cout << "  restore outer - inner mapping between " << im_it->second.first.first->getName() << " and " << im_it->second.first.second->getName() << ": " << prevInnerMapping->getName() << std::endl;
#endif
									prevInnerMapping->setModels(im_it->second.first.first, im_it->second.first.second);

									sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor(core::MechanicalParams::defaultInstance()).execute(im_it->second.first.first->getContext());
									sofa::simulation::UpdateMappingVisitor(core::ExecParams::defaultInstance()).execute(im_it->second.first.first->getContext());

									//if (m_fakeGripping_AttachedObjects.find(m_previous_FakeGripping_Event) != m_fakeGripping_AttachedObjects.end())
									if (ato_it != attached_objects_ret.second)
									{
										//std::cout << "  remove AttachFromObject and AttachInnerMapping from CollisionModel" << m_fakeGripping_AttachedObjects[m_previous_FakeGripping_Event]->getName() << std::endl;
										//m_fakeGripping_AttachedObjects[m_previous_FakeGripping_Event]->setAttachFromObject(NULL);
										//m_fakeGripping_AttachedObjects[m_previous_FakeGripping_Event]->setAttachFromInnerMapping(NULL);

										if (ato_it->second.first != NULL && ato_it->second.second != NULL)
										{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
											std::cout << "  remove AttachFromObject and AttachInnerMapping from CollisionModel " << ato_it->second.second->getName() << std::endl;
#endif

											ato_it->second.second->setAttachFromObject(NULL);
											ato_it->second.second->setAttachFromInnerMapping(NULL);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
											std::cout << "  remove from m_fakeGripping_CollisionCheck_Exceptions: " << ato_it->second.first->getName() << ", " << ato_it->second.second->getName() << std::endl;
#endif
											m_fakeGripping_CollisionCheck_Exceptions.erase(ato_it->second.first->getName());
											m_fakeGripping_CollisionCheck_Exceptions.erase(ato_it->second.second->getName());
										}
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
										else
										{
											std::cout << "  ERROR: Can't remove AttachFromObject and AttachInnerMapping from CollisionModel; Pointer = NULL, should never happen!" << std::endl;
										}
#endif

										ato_it++;
									}

									if (atn_it != attached_nodes_ret.second)
									{
										if (atn_it->second != NULL)
										{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
											std::cout << "  remove attached node " << atn_it->second->getName() << " from " << prevRigidMappingParent->getName() << std::endl;
#endif
											atn_it->second->detachFromGraph();
											prevRigidMappingParent->removeChild(atn_it->second);

											atn_it->second->setActive(false);
										}
										atn_it++;
									}
								}
							}
						}

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
						std::cout << "  remove rule " << m_previous_FakeGripping_Event << " from active gripping rules map" << std::endl;
#endif
						m_activeFakeGrippingRules_InnerMappings.erase(m_previous_FakeGripping_Event);
						m_activeFakeGrippingRules_Testing_Slaves.erase(m_previous_FakeGripping_Event);

						m_fakeGripping_AttachedObjects.erase(m_previous_FakeGripping_Event);
						m_fakeGripping_AttachedNodes.erase(m_previous_FakeGripping_Event);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
						std::cout << "  removed rule " << m_previous_FakeGripping_Event << " from active gripping rules map" << std::endl;
#endif

						m_fakeGripping_StoredUniformMasses.erase(m_previous_FakeGripping_Event);
					}

					// Attach Fake Gripping
					if (m_activeFakeGrippingRules_Testing_Slaves.find(m_active_FakeGripping_Event) == m_activeFakeGrippingRules_Testing_Slaves.end())
					{
						std::pair<sofa::simulation::Node*, sofa::simulation::Node*> nodePair = std::make_pair(parent1, parent2);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG					
						std::cout << "  Redirect RigidMapping for currently active fake-gripping rule " << m_active_FakeGripping_Event << " to " << mobj_2[0]->getName() << " -- " << inner_mobj_1[0]->getName() << std::endl;
						std::cout << "  inner_mobj_1.size() = " << inner_mobj_1.size() << std::endl;
#endif

						std::map<std::string, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr> createdRigidMappings;
						for (unsigned int p = 0; p < inner_mobj_1.size(); p++)
						{

							std::string mappingName = inner_mobj_1[p]->getName();
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG                        
							std::cout << "  * " << mappingName << std::endl;
#endif

							// Do not duplicate the wrong mappings
							if (boost::algorithm::ends_with(mappingName, "_MechanicalState"))
							{
								continue;
							}

							sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr rigidMapping = sofa::core::objectmodel::New<sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types> >();
							rigidMapping->setModels(mobj_2[0], inner_mobj_1[p]);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "    new rigidMapping fromModel = " << rigidMapping->getFromModel()->getName() << ", toModel = " << rigidMapping->getToModel()->getName() << std::endl;
#endif

							std::stringstream rgNameStream;
							rgNameStream << first_and_it->second.first->getName() << "_" << first_and_it->second.second->getName() << "_" << inner_mobj_1[p]->getName() << "_Gripping";

							rigidMapping->setName(rgNameStream.str());
							rgNameStream << "_Node";

							sofa::simulation::Node::SPtr rigidMappingNode = parent1->createChild(rgNameStream.str());

							/// START FIXED BY BE
							Vector3 obj_pos_1(mobj_1[0]->getPosition()[0][0], mobj_1[0]->getPosition()[0][1], mobj_1[0]->getPosition()[0][2]);
							Vector3 obj_pos_2(mobj_2[0]->getPosition()[0][0], mobj_2[0]->getPosition()[0][1], mobj_2[0]->getPosition()[0][2]);

							Quaternion rotation_1(mobj_1[0]->getPosition()[0][3], mobj_1[0]->getPosition()[0][4], mobj_1[0]->getPosition()[0][5], mobj_1[0]->getPosition()[0][6]);
							Quaternion rotation_2(mobj_2[0]->getPosition()[0][3], mobj_2[0]->getPosition()[0][4], mobj_2[0]->getPosition()[0][5], mobj_2[0]->getPosition()[0][6]);

							Vector3 position_offset = obj_pos_1 - obj_pos_2;
							Vector3 initialOffset = position_offset - obj_pos_1;

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "    obj_pos_1 = " << obj_pos_1 << ", obj_pos_2 = " << obj_pos_2 << std::endl;
							std::cout << "    position_offset = " << position_offset << std::endl;
							std::cout << "    initialOffset = " << initialOffset << std::endl;

							std::cout << "    apply translation/rotation to inner mapping " << std::endl;
#endif

							inner_mobj_1[p]->applyTranslation(-obj_pos_2.x(), -obj_pos_2.y(), -obj_pos_2.z());
							inner_mobj_1[p]->applyRotation(rotation_2.inverse());

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "    apply translation/rotation to inner mapping done" << std::endl;
							std::cout << "    init and add rigidMapping" << std::endl;
#endif

							rigidMapping->init();
							rigidMapping->appliedTranslation.setValue(rotation_2.inverseRotate(position_offset));
							rigidMapping->appliedRotation.setValue(rotation_2.inverse()*rotation_1);

							/// END FIXED BY BE

							std::cout << "  apply translation/rotation to inner mapping done" << std::endl;

							// OLD
							rigidMapping->lastObjPos.setValue(obj_pos_2);
							rigidMapping->lastObjRot.setValue(rotation_2);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "  init and add rigidMapping" << std::endl;
#endif                        
							rigidMapping->init();

							std::string originOfRigidMapping = inner_mobj_1[p]->getName();
							boost::algorithm::replace_all(originOfRigidMapping, "Trans_", "");
							createdRigidMappings.insert(std::make_pair(originOfRigidMapping, rigidMapping));

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "    add rigidMapping to rigidMappingNode" << std::endl;
#endif
							rigidMappingNode->addObject(rigidMapping);
							parent1->addChild(rigidMappingNode);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG						
							std::cout << "    attach rigidMappingNode = " << rigidMappingNode->getName() << " to parent1 = " << parent1->getName() << std::endl;
#endif

							m_fakeGripping_AttachedNodes.insert(std::make_pair(m_active_FakeGripping_Event, rigidMappingNode));

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "    init and add rigidMapping done" << std::endl;
							std::cout << "    set zero velocity for object " << mobj_1[0]->getName() << std::endl;
#endif

							if (m_uniformMass_1.size() > 0)
							{
								std::cout << "    reset UniformMass to nearZero" << std::endl;
								m_uniformMass_1[0]->old_storedMass.setValue(m_uniformMass_1[0]->d_mass.getValue());
								m_uniformMass_1[0]->d_mass.setValue(0.000001);

								m_fakeGripping_StoredUniformMasses.insert(std::make_pair(m_active_FakeGripping_Event, m_uniformMass_1[0]));
							}

							Rigid3Types::VecDeriv* v_data = mobj_1[0]->v.beginEdit();
							(*v_data)[0][0] = 0.0f;
							(*v_data)[0][1] = 0.0f;
							(*v_data)[0][2] = 0.0f;
							(*v_data)[0][3] = 0.0f;
							(*v_data)[0][4] = 0.0f;
							(*v_data)[0][5] = 0.0f;
							mobj_1[0]->v.endEdit();

							mobj_1[0]->vOp(core::ExecParams::defaultInstance(), core::VecId::velocity());
							mobj_1[0]->vOp(core::ExecParams::defaultInstance(), core::VecId::freeVelocity(), core::VecId::velocity());

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "    insert new rigidMapping's in active rules map" << std::endl;
#endif

							m_activeFakeGrippingRules_Testing_Slaves.insert(std::make_pair(m_active_FakeGripping_Event, std::make_pair(nodePair, rigidMapping)));

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << "    createdRigidMappings.size() = " << createdRigidMappings.size() << std::endl;
#endif

						} // for (unsigned int p = 0; p < inner_mobj_1.size(); p++)

						// AB DA RAUS AUS FOR!!!!!
						std::vector<sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>*> inner_mapping_1;
						sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>, std::vector<sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* > > inner_mapping_1_cb(&inner_mapping_1);
						parent1->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types> >::get(), inner_mapping_1_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
						std::cout << " Inner Mappings type RigidMapping<Rigid3Types, Vec3Types> instances:" << std::endl;
						std::cout << "  parent1 " << parent1->getName() << " size() = " << inner_mapping_1.size() << std::endl;
						for (unsigned int q = 0; q < inner_mapping_1.size(); q++)
							std::cout << "    * " << inner_mapping_1[q]->getName() << std::endl;

						std::cout << std::endl;

						std::cout << "  ==> search for inner mapping to store; inner_mapping_1.size() = " << inner_mapping_1.size() << std::endl;
						std::cout << "  ==> createdRigidMappings.size() = " << createdRigidMappings.size() << std::endl;
#endif

						for (size_t q = 0; q < inner_mapping_1.size(); ++q)
						{
							if (inner_mapping_1[q] != NULL)
							{
								sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr rigidMapping = inner_mapping_1[q];
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
								std::cout << "  - possible candidate for inner mapping for later restoring " << q << ": " << inner_mapping_1[q]->getName() << std::endl;
#endif

								if (boost::algorithm::ends_with(inner_mapping_1[q]->getName(), "_Gripping"))
								{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
									std::cout << "      WARNING: THIS IS NOT the right inner mapping; is this a leftover from the last active rule?: " << inner_mapping_1[q]->getName() << std::endl;
#endif
									continue;
								}
								else
								{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
									if (inner_mapping_1[q]->getFromModel() != NULL && inner_mapping_1[q]->getToModel() != NULL)
										std::cout << "     store for later restoration: " << inner_mapping_1[q]->getName() << ", fromModel = " << inner_mapping_1[q]->getFromModel()->getName() << ", toModel = " << inner_mapping_1[q]->getToModel()->getName() << std::endl;
#endif

									if (inner_mapping_1[q]->getContext() != NULL)
									{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
										std::cout << "     search for toModel-associated collision model under node = " << inner_mapping_1[q]->getContext()->getName() << std::endl;
#endif
										sofa::simulation::Node* parentNode = (sofa::simulation::Node*)(inner_mapping_1[q]->getContext());

										std::vector<ObbTreeGPUCollisionModel<Vec3Types>*> col_objs;
										sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<ObbTreeGPUCollisionModel<Vec3Types>, std::vector<ObbTreeGPUCollisionModel<Vec3Types>* > > col_objs_cb(&col_objs);
										parentNode->getObjects(sofa::core::objectmodel::TClassInfo<ObbTreeGPUCollisionModel<Vec3Types> >::get(), col_objs_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
										std::cout << "      found ObbTreeGPUCollisionModels: " << col_objs.size() << std::endl;
#endif

										sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr originalRigidMapping = createdRigidMappings[inner_mapping_1[q]->getName()];

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG									
										std::cout << "      originalRigidMapping = " << originalRigidMapping->getName() << std::endl;
#endif

										if (originalRigidMapping.get() == NULL)
										{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
											serr << "originalRigidMapping == NULL for " << inner_mapping_1[q]->getName() << std::endl;
#endif
											/// TODO: Assertion überprüfen. Bernd???
											//assert(originalRigidMapping.get() != NULL);
											continue;
										}

										if (originalRigidMapping != NULL)
										{
											for (unsigned int t = 0; t < col_objs.size(); t++)
											{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
												std::cout << "      * " << col_objs[t]->getName() << std::endl;
#endif

												col_objs[t]->setAttachFromObject((sofa::component::container::MechanicalObject<Rigid3Types>*)(originalRigidMapping->getFromModel()));
												col_objs[t]->setAttachFromInnerMapping(originalRigidMapping.get());

												m_fakeGripping_CollisionCheck_Exceptions.insert(std::make_pair(leadingObject->getName(), col_objs[t]->getName()));

												m_fakeGripping_AttachedObjects.insert(std::make_pair(m_active_FakeGripping_Event, std::make_pair(leadingObject, col_objs[t])));
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
												std::cout << "        now stored in m_fakeGripping_AttachedObjects: " << m_fakeGripping_AttachedObjects.size() << " collision objects." << std::endl;
#endif
											}
										}
									}
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
									else
									{
										std::cout << "    search for toModel-associated collision model under node: NULL context pointer!!!" << std::endl;
									}
#endif
								}
							}
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							else
							{
								std::cout << "      WARNING: THERE IS A NULL POINTER in inner_mapping_1 that doesn't belong there!!!, q = " << q << std::endl;
							}
#endif
						}

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
						std::cout << "CollisionObjects with AttachedFromObject/InnerMapping" << std::endl;

						std::cout << "  check which inner mapping we need to disable: We got " << inner_mapping_1.size() << " candidates." << std::endl;
#endif
						for (unsigned int q = 0; q < inner_mapping_1.size(); q++)
						{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
							std::cout << " " << inner_mapping_1[q]->getName() << std::endl;
#endif
							if (boost::algorithm::ends_with(inner_mapping_1[q]->getName(), "_Gripping"))
							{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
								std::cout << "  this is a fake-gripping created mapping; skip it: " << inner_mapping_1[q]->getName() << std::endl;
#endif
								continue;
							}
							else
							{
								if ((inner_mapping_1[q]->getFromModel() != NULL) && (inner_mapping_1[q]->getToModel() != NULL))
								{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
									std::cout << "  disable inner mapping " << inner_mapping_1[q]->getName() << "; points from " << inner_mapping_1[q]->getFromModel()->getName() << " to " << inner_mapping_1[q]->getToModel()->getName() << std::endl;
#endif

									std::pair<sofa::core::State<Rigid3Types>*, sofa::core::State<Vec3Types>*> mechObjPair = std::make_pair(inner_mapping_1[q]->getFromModel(), inner_mapping_1[q]->getToModel());
									m_activeFakeGrippingRules_InnerMappings.insert(std::make_pair(m_active_FakeGripping_Event, std::make_pair(mechObjPair, inner_mapping_1[q])));

									// get collision models to check for ghosts
									std::vector< sofa::component::collision::ObbTreeGPUCollisionModel< Vec3Types >* > collision_model_1;

									sofa::core::objectmodel::BaseContext::GetObjectsCallBackT< sofa::component::collision::ObbTreeGPUCollisionModel< Vec3Types >, std::vector< sofa::component::collision::ObbTreeGPUCollisionModel< Vec3Types >* > > collision_model_1_cb(&collision_model_1);

									static_cast<sofa::simulation::Node*>((inner_mapping_1[q]->getContext()))->getObjects(sofa::core::objectmodel::TClassInfo< sofa::component::collision::ObbTreeGPUCollisionModel< Vec3Types > >::get(), collision_model_1_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

									//sout << "  collision_model_1.size() = " << collision_model_1.size() << std::endl;

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
									if (collision_model_1.size() > 1)
									{
										std::cout << "  WARNING! More than one collision model found when looking for ghost model. First model is checked for ghost." << std::endl;
									}
#endif
									if (collision_model_1.size() == 0)
									{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
										std::cout << "  WARNING! No collision model found when looking for ghost model. No mapping is disabled." << std::endl;
#endif
									}
									else
									{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
										std::cout << "  inner mapping for " << collision_model_1.at(0)->getName() << " is to be disabled" << std::endl;
										std::cout << "  disable mapping " << inner_mapping_1[q]->getName() << std::endl;
#endif
										inner_mapping_1[q]->setModels(NULL, NULL);
									}
								}
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
								else
								{
									if (inner_mapping_1[q]->getFromModel() == NULL)
									{
										std::cout << "    inner_mapping_1[" << q << "]->getFromModel() == NULL; mapping = " << inner_mapping_1[q]->getName() << std::endl;
									}
									if (inner_mapping_1[q]->getToModel() == NULL)
									{
										std::cout << "    inner_mapping_1[" << q << "]->getToModel() == NULL; mapping = " << inner_mapping_1[q]->getName() << std::endl;
									}
								}
#endif
							}
						}

						m_fakeGripping_Activated_Rules[m_active_FakeGripping_Event] = true;
					}
				}
				else if (matchingFakeGrippingOrModels.find(m_active_FakeGripping_Event) != matchingFakeGrippingOrModels.end())  // OR condition (Not Implemented)
				{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
					std::cout << " found m_active_FakeGripping_Event in or-condition map: models = " << matchingFakeGrippingOrModels[m_active_FakeGripping_Event].first->getName() << " -- " << matchingFakeGrippingOrModels[m_active_FakeGripping_Event].second->getName() << std::endl;
#endif
				}
				else  // EXIT rule
				{
					// NO-OP
				}
			}
		}

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
		std::cout << "=== END OF FAKE-GRIPPING HANDLING ===" << std::endl;
#endif	
	}

#ifdef _WIN32
    updateRuleVis();
#endif

    for (unsigned int k = 0; k < m_traversalTasks.size(); k++)
    {
        m_traversalTasks[k]->clearWorkList();
    }

    for (unsigned int k = 0; k < m_triangleTasks.size(); k++)
    {
        m_triangleTasks[k]->clearWorkList();
    }

    m_scheduler_traversal->clearTasks();
    m_scheduler_triangles->clearTasks();

#ifdef OBBTREE_GPU_COLLISION_DETECTION_THREADED_CPU_COLLISION_CHECKS
	for (unsigned int k = 0; k < m_cpuTraversalTasks.size(); k++)
	{
		m_cpuTraversalTasks[k]->clearWorkList();
	}

	m_scheduler_cpu_traversal->clearTasks();
#endif

}

void ObbTreeGPUCollisionDetection_Threaded::draw(const core::visual::VisualParams *vparams)
{
    if (m_active_FakeGripping_Event != -1)
    {
        glPushMatrix();

        glPushAttrib(GL_ENABLE_BIT);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);


        glBegin(GL_LINES);
        glColor4d(1, 0, 0, 0.5);
        glVertex3d(0, 0, 0);
        glColor4d(1, 0, 0, 0.5);
        glVertex3d(m_fakeGripping_ModelPos_1.x(), m_fakeGripping_ModelPos_1.y(), m_fakeGripping_ModelPos_1.z());
        glColor4d(0, 1, 0, 0.5);
        glEnd();

        glBegin(GL_LINES);
        glColor4d(0, 1, 0, 0.5);
        glVertex3d(0, 0, 0);
        glColor4d(0, 1, 0, 0.5);
        glVertex3d(m_fakeGripping_ModelPos_2.x(), m_fakeGripping_ModelPos_2.y(), m_fakeGripping_ModelPos_2.z());
        glEnd();

        glBegin(GL_LINES);
        glColor4d(1, 1, 0, 0.5);
        glVertex3d(m_fakeGripping_ModelPos_1.x(), m_fakeGripping_ModelPos_1.y(), m_fakeGripping_ModelPos_1.z());
        glColor4d(1, 1, 0, 0.5);
        glVertex3d(m_fakeGripping_ModelPos_1.x() + m_fakeGripping_initialOffset.x(), m_fakeGripping_ModelPos_1.y() + m_fakeGripping_initialOffset.y(), m_fakeGripping_ModelPos_1.z() + m_fakeGripping_initialOffset.z());
        glEnd();

        glPopAttrib();
        glPopMatrix();
    }

}

int ObbTreeGPUCollisionDetection_Threaded::scheduleBVHTraversals(std::vector< std::pair<OBBModelContainer,OBBModelContainer> >& narrowPhasePairs,
                                                                 unsigned int numSlots,
                                                                 std::vector<BVHTraversalTask*>& bvhTraversals)
{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
    sout << "ObbTreeGPUCollisionDetection_Threaded::scheduleBVHTraversals(): " << narrowPhasePairs.size() << " pair checks, " << numSlots << " slots" << std::endl;
#endif
    int workUnitsAssigned = 0;

    if (narrowPhasePairs.size() == 0)
        return 0;

    for (unsigned int k = 0; k < narrowPhasePairs.size(); k++)
    {
        ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = narrowPhasePairs[k].first._obbCollisionModel;
        ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = narrowPhasePairs[k].second._obbCollisionModel;

        unsigned int traversalIndex = workUnitsAssigned % bvhTraversals.size();

#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG        
		sout << " - Obb model pair " << k << ": " << obbModel1->getName() << " - " << obbModel2->getName() << " bvhTraversal = " << traversalIndex << std::endl;
#endif

        bool addObbPair = true;
        if (m_fakeGripping_CollisionCheck_Exceptions.find(obbModel1->getName()) != m_fakeGripping_CollisionCheck_Exceptions.end())
        {
            std::pair< std::multimap<std::string, std::string>::iterator, std::multimap<std::string, std::string>::iterator > exc_range = m_fakeGripping_CollisionCheck_Exceptions.equal_range(obbModel1->getName());
            for (std::multimap<std::string, std::string>::iterator it_obb_1 = exc_range.first; it_obb_1 != exc_range.second; it_obb_1++)
            {
                if (it_obb_1->second.compare(obbModel2->getName()) == 0)
                {
                    addObbPair = false;
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                    sout << "   obbModel2 = " << obbModel2->getName() << " detected on fake gripping exception list, with obbModel1 = " << obbModel1->getName() << std::endl;
#endif
                    break;
                }
            }
        }

        if (addObbPair && m_fakeGripping_CollisionCheck_Exceptions.find(obbModel2->getName()) != m_fakeGripping_CollisionCheck_Exceptions.end())
        {
            std::pair< std::multimap<std::string, std::string>::iterator, std::multimap<std::string, std::string>::iterator > exc_range = m_fakeGripping_CollisionCheck_Exceptions.equal_range(obbModel2->getName());
            for (std::multimap<std::string, std::string>::iterator it_obb_2 = exc_range.first; it_obb_2 != exc_range.second; it_obb_2++)
            {
                if (it_obb_2->second.compare(obbModel1->getName()) == 0)
                {
                    addObbPair = false;
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
                    sout << "   obbModel1 = " << obbModel1->getName() << " detected on fake gripping exception list, with obbModel2 = " << obbModel2->getName() << std::endl;
#endif
                    break;
                }
            }
        }

#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
        sout << "   add OBB pair = " << addObbPair << std::endl;
#endif

        if (addObbPair)
        {
            OBBContainer& obbTree1 = narrowPhasePairs[k].first._obbContainer;
            OBBContainer& obbTree2 = narrowPhasePairs[k].second._obbContainer;

            gProximityGPUTransform* modelTr1 = _gpuTransforms[_gpuTransformIndices[obbModel1->getName()]];
            gProximityGPUTransform* modelTr2 = _gpuTransforms[_gpuTransformIndices[obbModel2->getName()]];

            bvhTraversals[traversalIndex]->addTraversal(&obbTree1, &obbTree2, modelTr1, modelTr2, obbModel1, obbModel2);

            m_pairChecks.insert(std::make_pair(k /*workUnitsAssigned*/, std::make_pair(obbModel1, obbModel2)));

            workUnitsAssigned++;
        }
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
        sout << "   workUnitsAssigned = " << workUnitsAssigned << std::endl;
#endif
    }

    return workUnitsAssigned;
}

int ObbTreeGPUCollisionDetection_Threaded::scheduleCPUCollisionChecks(std::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& narrowPhasePairs,
																	  sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap& detectionOutputs,
																	  unsigned int numSlots)
{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
	std::cout << "ObbTreeGPUCollisionDetection_Threaded::scheduleCPUCollisionChecks(): " << narrowPhasePairs.size() << " CPU pair checks, " << numSlots << " slots" << std::endl;
#endif
	if (narrowPhasePairs.size() == 0)
		return 0;

	int traversalsAssigned = 0;

	std::map<unsigned int, unsigned int> pairIndex_perTask;

	// Round-robin, baby
	for (size_t k = 0; k < narrowPhasePairs.size(); ++k)
	{
		std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> collisionPair = narrowPhasePairs[k];
		unsigned int traversalIndex = traversalsAssigned % m_scheduler_cpu_traversal->getNumThreads();

		sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap::iterator outputVector = detectionOutputs.find(collisionPair);
		
		if (outputVector != detectionOutputs.end())
		{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
			sout << "  - " << k << ": " << collisionPair.first->getName() << " <-> " << collisionPair.second->getName() << ": Found pre-created DetectionOutputVector, added to task " << traversalIndex << sendl;
#endif
			sofa::core::CollisionModel* model_1 = collisionPair.first;
			sofa::core::CollisionModel* model_2 = collisionPair.second;
			while (model_1->getNext() != NULL)
			{
				model_1 = model_1->getNext();
			}

			while (model_2->getNext() != NULL)
			{
				model_2 = model_2->getNext();
			}

#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
			sout << "   --> Resolves to: " << model_1->getName() << " <-> " << model_2->getName() << sendl;
#endif
			//std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> parentPair = std::make_pair(model_1, model_2);
			// Add what here: BVLevel7 instances (lowest elements of CubeModel)? Or the parent models (Triangle/Line/Point)? Seems: CubeModel is the right choice.
			m_cpuTraversalTasks[traversalIndex]->addCollidingPair(collisionPair, outputVector->second);

			pairIndex_perTask[traversalIndex] += 1;

			traversalsAssigned++;
			_narrowPhasePairs_CPU_taskAssignment[traversalIndex].push_back(std::make_pair(model_1->getName(), model_2->getName()));

			_narrowPhasePairs_CPU_modelAssociations.insert(std::make_pair(collisionPair, std::make_pair(model_1->getName(), model_2->getName())));
		}
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
		else
		{
			std::cout << "  - " << k << ": " << collisionPair.first->getName() << " <-> " << collisionPair.second->getName() << ": NOT ADDED, no pre-created DetectionOutputVector found!!!" << std::endl;
		}
#endif
	}
	return traversalsAssigned;
}

bool ObbTreeGPUCollisionDetection_Threaded::hasBVHOverlaps(std::string model1, std::string model2)
{
    for (unsigned int k = 0; k < m_traversalTasks.size(); k++)
    {
        BVHTraversalTask* task_k = m_traversalTasks[k];
        const std::vector<int>& task_k_results = task_k->getResults();
        for (unsigned int l = 0; l < task_k_results.size(); l++)
        {
            if (task_k_results[l] > 0)
            {
                const std::string& modelName_1 = task_k->getModelName1(l);
                const std::string& modelName_2 = task_k->getModelName2(l);

                if (modelName_1 == model1 && modelName_2 == model2)
                    return true;
            }
        }
    }
    return false;
}

bool ObbTreeGPUCollisionDetection_Threaded::hasIntersections(std::string model1, std::string model2)
{
    for (unsigned int k = 0; k < m_triangleTasks.size(); k++)
    {
        NarrowPhaseGPUTask* task_k = m_triangleTasks[k];
        std::vector<int> intersections = task_k->getTriangleIntersectionResults();
        for (int i = 0; i < task_k->getResultSize(); i++) {
            if (intersections.at(i) > 0) {
                std::string model1Name = task_k->getModelName1(i);
                std::string model2Name = task_k->getModelName2(i);

                if (model1Name == model1 && model2Name == model2) {
                    std::map<unsigned int, std::vector<double> >& distances = task_k->getContactDistances();
                    for (size_t j = 0; j < distances.at(i).size(); j++) {
                        if (distances[i][j] < m_alarmDistance) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

int ObbTreeGPUCollisionDetection_ThreadedClass = sofa::core::RegisterObject("Collision detection using GPU-based OBB-trees (multi-threaded), with fall back to brute-force pair tests")
        .add< ObbTreeGPUCollisionDetection_Threaded >();



