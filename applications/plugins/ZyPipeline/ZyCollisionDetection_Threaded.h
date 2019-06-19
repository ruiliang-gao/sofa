#ifndef OBBTREEGPUCOLLISIONDETECTION_THREADED_H
#define OBBTREEGPUCOLLISIONDETECTION_THREADED_H

#include <config_trupipeline.h>

//#include <SofaBaseCollision/BruteForceDetection.h>

/*#include "ObbTreeGPUCollisionModel.h"

#include "ObbTreeGPUIntersection.h"

#include <device_types.h>
#include "ObbTreeGPUCollisionDetection_cuda.h"

#include "ObbTreeGPU_MultiThread_Tasks.h"
#include "ObbTreeGPU_MultiThread_Scheduler.h"

#include "ObbTreeGPU_MultiThread_CPU_Tasks.h"*/

#include <sofa/simulation/common/Node.h>
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
class TruCollisionDetection_Threaded_Private;

class TRU_PIPELINE_API TruCollisionDetection_Threaded : public sofa::core::objectmodel::BaseObject
{
public:
	SOFA_CLASS(TruCollisionDetection_Threaded, sofa::core::objectmodel::BaseObject);

protected:
    TruCollisionDetection_Threaded();
    ~TruCollisionDetection_Threaded();

#ifdef _WIN32
    TruCollisionDetection_Threaded_Private* m_d;
#endif

    /*struct OBBModelContainer
    {
    public:
        OBBContainer _obbContainer;
        ObbTreeGPUCollisionModel<Vec3Types>* _obbCollisionModel;
    };*/

	int scheduleBVHTraversals(std::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& narrowPhasePairs,
                              unsigned int numSlots
                              /*std::vector<BVHTraversalTask*>& bvhTraversals*/);

	int scheduleCPUCollisionChecks(std::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& narrowPhasePairs /*, sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap& detectionOutputs*/, unsigned int numSlots);

    bool hasIntersections(std::string model1, std::string model2);
    bool hasBVHOverlaps(std::string model1, std::string model2);

	// Narrow-phase pairings for parallel processing, either in ObbTreeGPU or in simple parallel CPU model task scheduler
    //std::vector< std::pair<OBBModelContainer,OBBModelContainer> > _narrowPhasePairs;
	std::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> > _narrowPhasePairs;
	std::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> > _narrowPhasePairs_CPU;
	std::map<unsigned int, std::vector<std::pair<std::string, std::string> > > _narrowPhasePairs_CPU_taskAssignment;

	std::map<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>, std::pair<std::string, std::string> > _narrowPhasePairs_CPU_modelAssociations;


    /*std::map<std::string, ObbTreeGPUCollisionModel<Vec3Types>*> m_obbModels;
    std::map<std::string,ObbTreeGPUCollisionModel<Vec3Types>*> m_obbTreeGPUModels;*/

    std::vector<std::pair<std::string, std::string> >  m_testedModelPairs;
	
    //OBBTreeGPUDiscreteIntersection* m_intersection;
    double m_alarmDistance, m_contactDistance;

    Data<unsigned int> _numStreamedWorkerUnits;
    Data<unsigned int> _numStreamedWorkerResultBins;
    Data<unsigned int> _streamedWorkerResultMinSize;
    Data<unsigned int> _streamedWorkerResultMaxSize;
    Data<bool> _showTestFramework;

    Data<int> m_numWorkerThreads;

	// Schedulers for GPU-based BVH traversals and triangle pair tests
    //ObbTreeGPU_MultiThread_Scheduler<BVHTraversalTask>* m_scheduler_traversal;
    //ObbTreeGPU_MultiThread_Scheduler<NarrowPhaseGPUTask>* m_scheduler_triangles;

	// Scheduler for CPU-based, built-in SOFA collision models
	//ObbTreeGPU_MultiThread_Scheduler<CPUCollisionCheckTask>* m_scheduler_cpu_traversal;

    std::vector<std::pair<core::CollisionModel*,core::CollisionModel*> > collidingModels;

    //std::vector<BVHTraversalTask*> m_traversalTasks;
    //std::vector<NarrowPhaseGPUTask*> m_triangleTasks;
	//std::vector<CPUCollisionCheckTask*> m_cpuTraversalTasks;

    
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
};

}
}
}

#endif // OBBTREEGPUCOLLISIONDETECTION_THREADED_H
