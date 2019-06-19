#include "TruCollisionDetection_Threaded.h"

/*#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include <cutil/cutil.h>*/

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/DrawTool.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <SofaOpenglVisual/OglModel.h>

#include <sofa/core/collision/Contact.h>
#include <SofaBaseCollision/DiscreteIntersection.h>
#include <SofaBaseCollision/BaseContactMapper.h>
#include <SofaMeshCollision/BarycentricPenalityContact.h>

#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>

#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseMechanics/MechanicalObject.h>

/*#include "ObbTreeGPU_CudaDataStructures.h"
#include "ObbTreeGPUTriangleCollision_cuda.h"
#include "ObbTreeGPUCollisionModel_cuda.h"*/

#ifdef _WIN32
namespace sofa
{
namespace component
{
namespace collision
{
class TruCollisionDetection_Threaded_Private
{
	public:
    
};
}
}
}
#endif

using namespace sofa;
using namespace sofa::component::collision;
using namespace sofa::component::container;

using namespace sofa::simulation;

SOFA_DECL_CLASS(TruCollisionDetection_Threaded)

TruCollisionDetection_Threaded::TruCollisionDetection_Threaded():
    sofa::core::objectmodel::BaseObject(),
    //m_intersection(NULL),
    //_totalResultBinCount(0),
    _numStreamedWorkerUnits(initData(&_numStreamedWorkerUnits, (unsigned int) 4, "numStreamedWorkerUnits", "Number of stream worker units on GPU per worker thread", true, false)),
    _numStreamedWorkerResultBins(initData(&_numStreamedWorkerResultBins, (unsigned int) 4, "numStreamedWorkerResultBins", "Number of result bins (triangle collision checks) on GPU per worker thread", true, false)),
    _streamedWorkerResultMinSize(initData(&_streamedWorkerResultMinSize, (unsigned int) 2048, "streamedWorkerResultMinSize", "Size of smallest result bins", true, true)),
    _streamedWorkerResultMaxSize(initData(&_streamedWorkerResultMaxSize, (unsigned int) 16384, "streamedWorkerResultMaxSize", "Size of largest result bins", true, true)),
    m_numWorkerThreads(initData(&m_numWorkerThreads, 1, "numWorkerThreads", "Number of worker threads", true, true))
{
    /*m_scheduler_traversal = NULL;
    m_scheduler_triangles = NULL;

	m_scheduler_cpu_traversal = NULL;*/

    m_d = new TruCollisionDetection_Threaded_Private();

}

TruCollisionDetection_Threaded::~TruCollisionDetection_Threaded()
{
    if (m_d) 
	{
        delete m_d;
        m_d = NULL;
    }

    sout << "TruCollisionDetection_Threaded::~TruCollisionDetection_Threaded(" << this->getName() << ")" << std::endl;

    /*m_scheduler_traversal->cleanup();
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
	m_cpuTraversalTasks.clear();*/
}

void TruCollisionDetection_Threaded::init()
{
    //BruteForceDetection::init();

	/*std::cout << "NarrowPhaseDetection DetectionOutputMap instance = " << &(this->m_outputsMap) << std::endl;

    std::vector<sofa::component::collision::OBBTreeGPUDiscreteIntersection* > moV;
    sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::collision::OBBTreeGPUDiscreteIntersection, std::vector<sofa::component::collision::OBBTreeGPUDiscreteIntersection* > > cb(&moV);

    getContext()->getObjects(TClassInfo<sofa::component::collision::OBBTreeGPUDiscreteIntersection>::get(), cb, TagSet(), BaseContext::SearchRoot);

    sout << "TruCollisionDetection_Threaded::init(): Searched for instances of OBBTreeGPUDiscreteIntersection; found = " << moV.size() << std::endl;
    if (moV.size() == 1)
    {
        sout << " Using: " << moV[0]->getName() << " of type " << moV[0]->getTypeName() << std::endl;
        m_intersection = moV[0];
    }

    m_scheduler_traversal = new ObbTreeGPU_MultiThread_Scheduler<BVHTraversalTask>(m_numWorkerThreads.getValue());

    sout << "=== TruCollisionDetection_Threaded::TruCollisionDetection_Threaded(" << this->getName() << ") ===" << std::endl;
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

	m_scheduler_cpu_traversal->init();*/
}

void TruCollisionDetection_Threaded::reinit()
{
    //BruteForceDetection::reinit();
    //m_obbModels.clear();
}

void TruCollisionDetection_Threaded::reset()
{
    
}

void TruCollisionDetection_Threaded::bwdInit()
{
    /*CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&_transformStream, cudaStreamNonBlocking));
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&_memoryStream, cudaStreamNonBlocking));

    std::vector<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>* > obbTreeGPUCollisionModels;
    sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>, std::vector<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>* > > obbTreeGPUCollisionModels_cb(&obbTreeGPUCollisionModels);
    getContext()->getObjects(TClassInfo<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types> >::get(), obbTreeGPUCollisionModels_cb, TagSet(), BaseContext::SearchRoot);

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

    getContext()->getObjects(TClassInfo<ObbTreeGPULocalMinDistance>::get(), cb, TagSet(), BaseContext::SearchRoot);

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
    sout << "testOutputFilename = " << testOutputFilename.getValue() << std::endl;*/
}

void TruCollisionDetection_Threaded::addCollisionModels(const sofa::helper::vector<core::CollisionModel *> collisionModels)
{
    if (!this->f_printLog.getValue())
        this->f_printLog.setValue(true);

	/*
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
	*/
}

void TruCollisionDetection_Threaded::addCollisionModel(core::CollisionModel *cm)
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
    std::cout << "ObbTreeGPUCollisionDetection::addCollisionModel(" << cm->getName() << "), type = " << cm->getTypeName() << std::endl;
#endif

    if (!cm)
        return;
	/*
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
	*/
}

void TruCollisionDetection_Threaded::addCollisionPairs(const sofa::helper::vector<std::pair<core::CollisionModel *, core::CollisionModel *> > &v)
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

//
//#ifndef	OBBTREE_GPU_COLLISION_DETECTION_SEQUENTIAL_CPU_COLLISION_CHECKS
//	BruteForceDetection::createDetectionOutputs(this->_narrowPhasePairs_CPU);
//	const sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap& detection_outputs = this->getDetectionOutputs();
//#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONPAIRS
//	std::cout << "=== Pre-created detection outputs: " << detection_outputs.size() << " ===" << std::endl;
//	for (sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator do_it = detection_outputs.begin(); do_it != detection_outputs.end(); ++do_it)
//	{
//		std::cout << " - " << do_it->first.first->getName() << " -- " << do_it->first.second->getName() << std::endl;
//	}
//#endif
//#endif

#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONPAIRS
    std::cout << "=== ObbTreeGPUCollisionDetection::addCollisionPairs(): " << addedPairs << " pairs added. ===" << std::endl;
#endif
}

void TruCollisionDetection_Threaded::addCollisionPair(const std::pair<core::CollisionModel *, core::CollisionModel *> &cmPair)
{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
    std::cout << "ObbTreeGPUCollisionDetection::addCollisionPair(" << cmPair.first->getName() << "," << cmPair.second->getName() << ")" << std::endl;
    std::cout << " model types: " << cmPair.first->getTypeName() << " - " << cmPair.second->getTypeName() << std::endl;
#endif

	/*ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = dynamic_cast<ObbTreeGPUCollisionModel<Vec3Types>*>(cmPair.first);
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
	*/
}

void TruCollisionDetection_Threaded::beginBroadPhase()
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_BEGINBROADPHASE_DEBUG
    std::cout << "=== TruCollisionDetection_Threaded::beginBroadPhase() ===" << std::endl;
#endif

    //BruteForceDetection::beginBroadPhase();
}

void TruCollisionDetection_Threaded::endBroadPhase()
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDBROADPHASE_DEBUG
    std::cout << "=== TruCollisionDetection_Threaded::endBroadPhase() ===" << std::endl;
#endif

    //BruteForceDetection::endBroadPhase();
}

void TruCollisionDetection_Threaded::beginNarrowPhase()
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_BEGINNARROWPHASE_DEBUG
    std::cout << "=== TruCollisionDetection_Threaded::beginNarrowPhase() ===" << std::endl;
#endif

    //BruteForceDetection::beginNarrowPhase();

    /*for (std::map<std::string, ObbTreeGPUCollisionModel<Vec3Types>*>::iterator it = m_obbTreeGPUModels.begin(); it != m_obbTreeGPUModels.end(); it++)
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

	*/
}

void TruCollisionDetection_Threaded::endNarrowPhase()
{
#ifdef OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG
    sout << "=== TruCollisionDetection_Threaded::endNarrowPhase() ===" << std::endl;
    sout << "Run BVH traversal tasks: Using " << m_numWorkerThreads.getValue() << " threads." << std::endl;

    std::cout << "=== TruCollisionDetection_Threaded::endNarrowPhase(" << this->getTime() << ") ===" << std::endl;
#endif

}

void TruCollisionDetection_Threaded::draw(const core::visual::VisualParams *vparams)
{

}

int TruCollisionDetection_Threaded::scheduleBVHTraversals(std::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& narrowPhasePairs,
                                                                 unsigned int numSlots
                                                                 /*std::vector<BVHTraversalTask*>& bvhTraversals*/)
{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
    sout << "TruCollisionDetection_Threaded::scheduleBVHTraversals(): " << narrowPhasePairs.size() << " pair checks, " << numSlots << " slots" << std::endl;
#endif
    int workUnitsAssigned = 0;

    if (narrowPhasePairs.size() == 0)
        return 0;

//    for (unsigned int k = 0; k < narrowPhasePairs.size(); k++)
//    {
//        ObbTreeGPUCollisionModel<Vec3Types>* obbModel1 = narrowPhasePairs[k].first._obbCollisionModel;
//        ObbTreeGPUCollisionModel<Vec3Types>* obbModel2 = narrowPhasePairs[k].second._obbCollisionModel;
//
//        unsigned int traversalIndex = workUnitsAssigned % bvhTraversals.size();
//
//#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG        
//		sout << " - Obb model pair " << k << ": " << obbModel1->getName() << " - " << obbModel2->getName() << " bvhTraversal = " << traversalIndex << std::endl;
//#endif
//
//        bool addObbPair = true;
//        if (m_fakeGripping_CollisionCheck_Exceptions.find(obbModel1->getName()) != m_fakeGripping_CollisionCheck_Exceptions.end())
//        {
//            std::pair< std::multimap<std::string, std::string>::iterator, std::multimap<std::string, std::string>::iterator > exc_range = m_fakeGripping_CollisionCheck_Exceptions.equal_range(obbModel1->getName());
//            for (std::multimap<std::string, std::string>::iterator it_obb_1 = exc_range.first; it_obb_1 != exc_range.second; it_obb_1++)
//            {
//                if (it_obb_1->second.compare(obbModel2->getName()) == 0)
//                {
//                    addObbPair = false;
//#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
//                    sout << "   obbModel2 = " << obbModel2->getName() << " detected on fake gripping exception list, with obbModel1 = " << obbModel1->getName() << std::endl;
//#endif
//                    break;
//                }
//            }
//        }
//
//        if (addObbPair && m_fakeGripping_CollisionCheck_Exceptions.find(obbModel2->getName()) != m_fakeGripping_CollisionCheck_Exceptions.end())
//        {
//            std::pair< std::multimap<std::string, std::string>::iterator, std::multimap<std::string, std::string>::iterator > exc_range = m_fakeGripping_CollisionCheck_Exceptions.equal_range(obbModel2->getName());
//            for (std::multimap<std::string, std::string>::iterator it_obb_2 = exc_range.first; it_obb_2 != exc_range.second; it_obb_2++)
//            {
//                if (it_obb_2->second.compare(obbModel1->getName()) == 0)
//                {
//                    addObbPair = false;
//#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
//                    sout << "   obbModel1 = " << obbModel1->getName() << " detected on fake gripping exception list, with obbModel2 = " << obbModel2->getName() << std::endl;
//#endif
//                    break;
//                }
//            }
//        }
//
//#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
//        sout << "   add OBB pair = " << addObbPair << std::endl;
//#endif
//
//        if (addObbPair)
//        {
//            OBBContainer& obbTree1 = narrowPhasePairs[k].first._obbContainer;
//            OBBContainer& obbTree2 = narrowPhasePairs[k].second._obbContainer;
//
//            gProximityGPUTransform* modelTr1 = _gpuTransforms[_gpuTransformIndices[obbModel1->getName()]];
//            gProximityGPUTransform* modelTr2 = _gpuTransforms[_gpuTransformIndices[obbModel2->getName()]];
//
//            bvhTraversals[traversalIndex]->addTraversal(&obbTree1, &obbTree2, modelTr1, modelTr2, obbModel1, obbModel2);
//
//            m_pairChecks.insert(std::make_pair(k /*workUnitsAssigned*/, std::make_pair(obbModel1, obbModel2)));
//
//            workUnitsAssigned++;
//        }
//#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
//        sout << "   workUnitsAssigned = " << workUnitsAssigned << std::endl;
//#endif
//	}

    return workUnitsAssigned;
}

int TruCollisionDetection_Threaded::scheduleCPUCollisionChecks(std::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& narrowPhasePairs,
																	  /*sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap& detectionOutputs,*/
																	  unsigned int numSlots)
{
#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
	std::cout << "TruCollisionDetection_Threaded::scheduleCPUCollisionChecks(): " << narrowPhasePairs.size() << " CPU pair checks, " << numSlots << " slots" << std::endl;
#endif
	if (narrowPhasePairs.size() == 0)
		return 0;

	int traversalsAssigned = 0;

//	std::map<unsigned int, unsigned int> pairIndex_perTask;
//
//	// Round-robin, baby
//	for (size_t k = 0; k < narrowPhasePairs.size(); ++k)
//	{
//		std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> collisionPair = narrowPhasePairs[k];
//		unsigned int traversalIndex = traversalsAssigned % m_scheduler_cpu_traversal->getNumThreads();
//
//		sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap::iterator outputVector = detectionOutputs.find(collisionPair);
//		
//		if (outputVector != detectionOutputs.end())
//		{
//#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
//			sout << "  - " << k << ": " << collisionPair.first->getName() << " <-> " << collisionPair.second->getName() << ": Found pre-created DetectionOutputVector, added to task " << traversalIndex << sendl;
//#endif
//			sofa::core::CollisionModel* model_1 = collisionPair.first;
//			sofa::core::CollisionModel* model_2 = collisionPair.second;
//			while (model_1->getNext() != NULL)
//			{
//				model_1 = model_1->getNext();
//			}
//
//			while (model_2->getNext() != NULL)
//			{
//				model_2 = model_2->getNext();
//			}
//
//#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
//			sout << "   --> Resolves to: " << model_1->getName() << " <-> " << model_2->getName() << sendl;
//#endif
//			//std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> parentPair = std::make_pair(model_1, model_2);
//			// Add what here: BVLevel7 instances (lowest elements of CubeModel)? Or the parent models (Triangle/Line/Point)? Seems: CubeModel is the right choice.
//			m_cpuTraversalTasks[traversalIndex]->addCollidingPair(collisionPair, outputVector->second);
//
//			pairIndex_perTask[traversalIndex] += 1;
//
//			traversalsAssigned++;
//			_narrowPhasePairs_CPU_taskAssignment[traversalIndex].push_back(std::make_pair(model_1->getName(), model_2->getName()));
//
//			_narrowPhasePairs_CPU_modelAssociations.insert(std::make_pair(collisionPair, std::make_pair(model_1->getName(), model_2->getName())));
//		}
//#ifdef OBBTREE_GPU_COLLISION_DETECTION_DEBUG
//		else
//		{
//			std::cout << "  - " << k << ": " << collisionPair.first->getName() << " <-> " << collisionPair.second->getName() << ": NOT ADDED, no pre-created DetectionOutputVector found!!!" << std::endl;
//		}
//#endif
//	}
	return traversalsAssigned;
}

bool TruCollisionDetection_Threaded::hasBVHOverlaps(std::string model1, std::string model2)
{
    /*for (unsigned int k = 0; k < m_traversalTasks.size(); k++)
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
    }*/
    return false;
}

bool TruCollisionDetection_Threaded::hasIntersections(std::string model1, std::string model2)
{
    /*for (unsigned int k = 0; k < m_triangleTasks.size(); k++)
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
    }*/
    return false;
}

int TruCollisionDetection_ThreadedClass = sofa::core::RegisterObject("Collision detection using GPU-based OBB-trees (multi-threaded), with fall back to brute-force pair tests")
        .add< TruCollisionDetection_Threaded >();



