#ifndef OBBTREEGPUCOLLISIONDETECTION_THREADED_H
#define OBBTREEGPUCOLLISIONDETECTION_THREADED_H

#include <ZyObbTreeGPU/config.h>
#include <SofaBaseCollision/BruteForceDetection.h>

#include "ObbTreeGPUCollisionModel.h"

#include "ObbTreeGPUIntersection.h"

#include <device_types.h>
#include "ObbTreeGPUCollisionDetection_cuda.h"

#include "ObbTreeGPU_MultiThread_Tasks.h"
#include "ObbTreeGPU_MultiThread_Scheduler.h"

#include "ObbTreeGPU_MultiThread_CPU_Tasks.h"

#include <sofa/simulation/Node.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaRigid/RigidRigidMapping.h>

// Debug defines
#define OBBTREE_GPU_COLLISION_DETECTION_DEBUG
#define OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONMODELS
//#define OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONMODELS_VERBOSE
#define OBBTREEGPUCOLLISIONDETECTION_DEBUG_ADDCOLLISIONMODEL
#define OBBTREEGPUCOLLISIONDETECTION_THREADED_DEBUG_ADDCOLLISIONPAIRS
#define OBBTREEGPUCOLLISIONDETECTION_THREADED_BEGINBROADPHASE_DEBUG
#define OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDBROADPHASE_DEBUG
#define OBBTREEGPUCOLLISIONDETECTION_THREADED_BEGINNARROWPHASE_DEBUG
#define OBBTREEGPUCOLLISIONDETECTION_THREADED_ENDNARROWPHASE_DEBUG

//#define OBBTREEGPUCOLLISIONDETECTION_USE_CONTACTTYPE_FROM_DETECTIONOUTPUT

// Defining this will force default sequential processing for CPU collision models
// Otherwise, a TaskPool will handle CPU pair checks in parallel
// #define OBBTREE_GPU_COLLISION_DETECTION_SEQUENTIAL_CPU_COLLISION_CHECKS
#define OBBTREE_GPU_COLLISION_DETECTION_THREADED_CPU_COLLISION_CHECKS

namespace sofa
{
namespace component
{
namespace collision
{
class ObbTreeGPUCollisionDetection_Threaded_Private;

enum FakeGripping_Condition
{
    FG_AND,
    FG_OR
};


struct FakeGripping_Event_Container
{
public:
    std::string activeModel;
    std::string leadingModel;
    std::vector<std::string> contactModels;
    FakeGripping_Condition contactCondition;

    FakeGripping_Event_Container() {}

    FakeGripping_Event_Container(const FakeGripping_Event_Container& other)
    {
        if (this != &other)
        {
            activeModel = other.activeModel;
            contactCondition = other.contactCondition;
            leadingModel = other.leadingModel;

            for (unsigned int k = 0; k < other.contactModels.size(); k++)
                contactModels.push_back(other.contactModels[k]);
        }
    }

    FakeGripping_Event_Container& operator=(const FakeGripping_Event_Container& other)
    {
        if (this != &other)
        {
            activeModel = other.activeModel;
            contactCondition = other.contactCondition;
            leadingModel = other.leadingModel;

            for (unsigned int k = 0; k < other.contactModels.size(); k++)
                contactModels.push_back(other.contactModels[k]);
        }
        return *this;
    }
};

class SOFA_OBBTREEGPUPLUGIN_API ObbTreeGPUCollisionDetection_Threaded: public BruteForceDetection
{
public:
    SOFA_CLASS(ObbTreeGPUCollisionDetection_Threaded, BruteForceDetection);

protected:
    ObbTreeGPUCollisionDetection_Threaded();
    ~ObbTreeGPUCollisionDetection_Threaded();

#ifdef _WIN32
    ObbTreeGPUCollisionDetection_Threaded_Private* m_d;
#endif

    struct OBBModelContainer
    {
    public:
        OBBContainer _obbContainer;
        ObbTreeGPUCollisionModel<Vec3Types>* _obbCollisionModel;
    };

    int scheduleBVHTraversals(std::vector< std::pair<OBBModelContainer,OBBModelContainer> >& narrowPhasePairs,
                              unsigned int numSlots,
                              std::vector<BVHTraversalTask*>& bvhTraversals);

    int scheduleCPUCollisionChecks(std::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& narrowPhasePairs, sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap& detectionOutputs, unsigned int numSlots);

    bool hasIntersections(std::string model1, std::string model2);
    bool hasBVHOverlaps(std::string model1, std::string model2);

    // Narrow-phase pairings for parallel processing, either in ObbTreeGPU or in simple parallel CPU model task scheduler
    std::vector< std::pair<OBBModelContainer,OBBModelContainer> > _narrowPhasePairs;
    std::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> > _narrowPhasePairs_CPU;
    std::map<unsigned int, std::vector<std::pair<std::string, std::string> > > _narrowPhasePairs_CPU_taskAssignment;

    std::map<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>, std::pair<std::string, std::string> > _narrowPhasePairs_CPU_modelAssociations;


    std::map<std::string, ObbTreeGPUCollisionModel<Vec3Types>*> m_obbModels;
    std::map<std::string,ObbTreeGPUCollisionModel<Vec3Types>*> m_obbTreeGPUModels;

    std::vector<std::pair<std::string, std::string> >  m_testedModelPairs;

    OBBTreeGPUDiscreteIntersection* m_intersection;
    double m_alarmDistance, m_contactDistance;

    Data<unsigned int> _numStreamedWorkerUnits;
    Data<unsigned int> _numStreamedWorkerResultBins;
    Data<unsigned int> _streamedWorkerResultMinSize;
    Data<unsigned int> _streamedWorkerResultMaxSize;
    Data<bool> _showTestFramework;

    std::vector<gProximityWorkerUnit*> _streamedWorkerUnits;
    std::map<unsigned int, std::vector<gProximityWorkerResult*> > _streamedWorkerResults;

    std::vector<cudaStream_t> _workerStreams;
    std::vector<cudaEvent_t> _workerEvents;
    cudaEvent_t _workerStartEvent, _workerEndEvent, _workerBalanceEvent;

    cudaStream_t _transformStream;
    cudaStream_t _memoryStream;

    std::vector<cudaStream_t> _triTestStreams;
    std::vector<cudaEvent_t> _triTestEvents;
    cudaEvent_t _triTestStartEvent, _triTestEndEvent;

    unsigned int _totalResultBinCount;

    gProximityGPUTransform** _gpuTransforms;
    std::map<std::string, unsigned int> _gpuTransformIndices;

    Data<float> m_trianglePairSizeRatio;

    Data<int> m_numWorkerThreads;

    // Schedulers for GPU-based BVH traversals and triangle pair tests
    ObbTreeGPU_MultiThread_Scheduler<BVHTraversalTask>* m_scheduler_traversal;
    ObbTreeGPU_MultiThread_Scheduler<NarrowPhaseGPUTask>* m_scheduler_triangles;

    // Scheduler for CPU-based, built-in SOFA collision models
    ObbTreeGPU_MultiThread_Scheduler<CPUCollisionCheckTask>* m_scheduler_cpu_traversal;

    Data<std::string> m_fakeGripping_EventSequence;

    std::vector<FakeGripping_Event_Container> m_fakeGripping_Events;
    int m_active_FakeGripping_Event, m_previous_FakeGripping_Event;

    Vector3 m_fakeGripping_ModelPos_1, m_fakeGripping_ModelPos_2;
    Vector3 m_fakeGripping_initialOffset;

    std::map<int, std::pair<std::pair<sofa::simulation::Node*, sofa::simulation::Node*>, sofa::component::mapping::RigidRigidMapping<Rigid3Types, Rigid3Types>::SPtr > > m_activeFakeGrippingRules;

    std::multimap<int, std::pair<std::pair<sofa::simulation::Node*, sofa::simulation::Node*>, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>::SPtr > > m_activeFakeGrippingRules_Testing_Slaves;
    std::multimap<int, std::pair<std::pair<sofa::core::State<Rigid3Types>*, sofa::core::State<Vec3Types>* >, sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* > > m_activeFakeGrippingRules_InnerMappings;

    std::map<int, sofa::component::mass::UniformMass<Rigid3Types, Rigid3dMass>*> m_fakeGripping_StoredUniformMasses;

    std::multimap<int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> > m_fakeGripping_AttachedObjects;
    std::multimap<int, sofa::simulation::Node::SPtr> m_fakeGripping_AttachedNodes;
    std::map<int, bool> m_fakeGripping_Activated_Rules;

    std::multimap<std::string, std::string> m_fakeGripping_CollisionCheck_Exceptions;

    std::vector<std::pair<core::CollisionModel*,core::CollisionModel*> > collidingModels;

    std::vector<BVHTraversalTask*> m_traversalTasks;
    std::vector<NarrowPhaseGPUTask*> m_triangleTasks;

    std::vector<CPUCollisionCheckTask*> m_cpuTraversalTasks;

    std::map<unsigned int, std::pair<ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*> > m_pairChecks;

    NarrowPhaseDetection::DetectionOutputMap* m_detectionOutputVectors;

    // Needed as class member in derived class because the list in the super class is declared private
    sofa::helper::vector<core::CollisionModel*> m_collisionModels;

    void createDetectionOutputs(const std::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& _cmPairs);

    std::ofstream testOutput;
    sofa::core::objectmodel::DataFileName testOutputFilename;

public:
    void init();
    void reinit();

    void bwdInit();

    void reset();

    void addCollisionModels(const sofa::helper::vector<core::CollisionModel *> v);
    void addCollisionModel(core::CollisionModel *cm);
    void addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair);
    void addCollisionPairs(const sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >& v);

    void beginBroadPhase();
    void endBroadPhase();

    void beginNarrowPhase();
    void endNarrowPhase();

    /* for debugging */
    void draw(const core::visual::VisualParams* vparams);

    const std::vector<std::pair<core::CollisionModel*,core::CollisionModel*> >& getCollidingModels() const
    {
        return collidingModels;
    }

    NarrowPhaseDetection::DetectionOutputMap*& getDetectionOutputVectors()
    {
        return m_detectionOutputVectors;
    }

#ifdef _WIN32
private:
    void updateRuleVis();
#endif


};

// inline bool contactTypeCompare(sofa::core::collision::DetectionOutput i, sofa::core::collision::DetectionOutput j) { return (i.contactType < j.contactType); }
}
}
}

#endif // OBBTREEGPUCOLLISIONDETECTION_THREADED_H
