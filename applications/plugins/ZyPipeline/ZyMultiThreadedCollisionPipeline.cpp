#include "ZyMultiThreadedCollisionPipeline.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/visual/VisualParams.h>

#include <SofaUserInteraction/RayModel.h>
#include <sofa/simulation/Node.h>

#include <set>

#ifdef SOFA_DUMP_VISITOR_INFO
#include <sofa/simulation/common/Visitor.h>
#endif

#include <sofa/helper/vector.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/simulation/Simulation.h>

#include <SofaMiscCollision/RuleBasedContactManager.h>

namespace Zyklio
{
    namespace Pipeline
    {
        class ZyMultiThreadedCollisionPipelinePrivate
        {
            public:
                ZyMultiThreadedCollisionPipelinePrivate() {}
                sofa::helper::vector<sofa::component::collision::RuleBasedContactManager*> m_ruleBasedContactManagers;
        };
    }
}

using namespace Zyklio::MultiThreading;
using namespace Zyklio::Pipeline;
using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::core::collision;

SOFA_DECL_CLASS(ZyMultiThreadedCollisionPipeline)

int ZyMultiThreadedCollisionPipelineClass = sofa::core::RegisterObject("TruPhysics pipeline class.")
        .add< ZyMultiThreadedCollisionPipeline >()
        ;

ZyMultiThreadedCollisionPipeline::ZyMultiThreadedCollisionPipeline() :
m_scheduler_updateBVH(NULL),
broadPhaseDetection(NULL),
narrowPhaseDetection(NULL),
intersectionMethod(NULL),
contactManager(NULL),
groupManager(NULL),
m_numWorkerThreads(initData(&m_numWorkerThreads, 4, "numWorkerThreads", "Number of worker threads", true, true))
{
    m_d = new ZyMultiThreadedCollisionPipelinePrivate();
    bVerbose.setValue(true);
}

ZyMultiThreadedCollisionPipeline::ZyMultiThreadedCollisionPipeline(BroadPhaseDetection* bpd, NarrowPhaseDetection* npd, Intersection* is, ContactManager* cm, CollisionGroupManager* cgm) :
    m_scheduler_updateBVH(NULL),
    broadPhaseDetection(bpd),
    intersectionMethod(is),
    contactManager(cm),
    groupManager(cgm),
    m_numWorkerThreads(initData(&m_numWorkerThreads, 4, "numWorkerThreads", "Number of worker threads", true, true))
{
    narrowPhaseDetection = new ZyParallelNarrowPhase(m_numWorkerThreads.getValue());
    m_d = new ZyMultiThreadedCollisionPipelinePrivate();
    bVerbose.setValue(true);
}

ZyMultiThreadedCollisionPipeline::~ZyMultiThreadedCollisionPipeline()
{
    if (m_scheduler_updateBVH != NULL)
    {
        delete m_scheduler_updateBVH;
        m_scheduler_updateBVH = NULL;
    }

    for (size_t k = 0; k < m_cpuBVHUpdateTasks.size(); k++)
    {
        delete m_cpuBVHUpdateTasks[k];
        m_cpuBVHUpdateTasks[k] = NULL;
    }
    m_cpuBVHUpdateTasks.clear();

    if (m_d != NULL)
    {
        delete m_d;
        m_d = NULL;
    }
}

#ifdef SOFA_DUMP_VISITOR_INFO
typedef simulation::Visitor::ctime_t ctime_t;
#endif

void ZyMultiThreadedCollisionPipeline::init()
{
    msg_info("ZyMultiThreadedCollisionPipeline") << "init()";
    if (!m_doInit)
    {
        msg_info("ZyMultiThreadedCollisionPipeline") << " m_doInit = false, return";
        return;
    }
}

void ZyMultiThreadedCollisionPipeline::bwdInit()
{
    msg_info("ZyMultiThreadedCollisionPipeline") <<  "TruPhysicsPipeline::bwdInit()";
    if (!m_doInit)
    {
        msg_info("ZyMultiThreadedCollisionPipeline") <<  " m_doInit = false, return";
        return;
    }

    for (int k = 0; k < m_numWorkerThreads.getValue(); ++k)
    {
        // Useful for something?
        TaskStatus status;

        std::stringstream idStr;
        idStr << "CPU BVH update task " << k;
        // No further setup necessary for CPU tasks as of now; assume 8 'worker units' per task
        CPUBVHUpdateTask* cpu_task = new CPUBVHUpdateTask(&status, 8);
        cpu_task->setTaskID(idStr.str());
        m_cpuBVHUpdateTasks.push_back(cpu_task);
    }
    m_scheduler_updateBVH = new MultiThread_Scheduler<CPUBVHUpdateTask>(m_numWorkerThreads.getValue());

    for (int i = 0; i < m_numWorkerThreads.getValue(); i++)
        m_scheduler_updateBVH->getScheduler()->createWorkerThread(true, "CPU_BVH_Update");

    m_scheduler_updateBVH->getScheduler()->startThreads();
    m_scheduler_updateBVH->getScheduler()->pauseThreads();

    narrowPhaseDetection = new ZyParallelNarrowPhase(m_numWorkerThreads.getValue());

    setActive(false);
}

void ZyMultiThreadedCollisionPipeline::setup(BroadPhaseDetection* broadPhaseDetection, NarrowPhaseDetection* narrowPhaseDetection, Intersection* intersection, ContactManager* contactManager, CollisionGroupManager* groupManager)
{
    this->broadPhaseDetection = broadPhaseDetection;
    SOFA_UNUSED(narrowPhaseDetection);
    this->intersectionMethod = intersection;
    this->contactManager = contactManager;
    this->groupManager = groupManager;
}

// This is the same as in DefaultPipeline, except for the parallelized update of CubeModel instances.
// TODO: Find a better way to encapsulate this, instead of copy/paste from the super class?
void ZyMultiThreadedCollisionPipeline::doCollisionDetection(const sofa::helper::vector<core::CollisionModel*>& collisionModels)
{
    sofa::helper::AdvancedTimer::stepBegin("doCollisionDetection");
    msg_info("ZyMultiThreadedCollisionPipeline") << "DefaultPipeline::doCollisionDetection, compute Bounding Trees";
    // First, we compute a bounding volume for the collision model (for example bounding sphere)
    // or we have loaded a collision model that knows its other model

    sofa::helper::vector<CollisionModel*> vectBoundingVolume;
    {
        sofa::helper::AdvancedTimer::stepBegin("BBox");
#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printNode("ComputeBoundingTree");
#endif
        const bool continuous = intersectionMethod->useContinuous();
        const SReal dt = getContext()->getDt();

        int used_depth = broadPhaseDetection->needsDeepBoundingTree() ? depth.getValue() : 0;

        sofa::helper::vector<CollisionModel*>::const_iterator it;
        const sofa::helper::vector<CollisionModel*>::const_iterator itEnd = collisionModels.end();
        int nActive = 0;

        m_scheduler_updateBVH->clearTasks();

        int updateTasksAssigned = 0;

        for (size_t k = 0; k < m_cpuBVHUpdateTasks.size(); ++k)
        {
            m_cpuBVHUpdateTasks[k]->setContinuous(continuous);
            m_cpuBVHUpdateTasks[k]->setDt(dt);
            m_cpuBVHUpdateTasks[k]->setUsedDepth(used_depth);
        }

        for (it = collisionModels.begin(); it != itEnd; ++it)
        {
            msg_info("ZyMultiThreadedCollisionPipeline") << "DefaultPipeline::doCollisionDetection, consider model " << (*it)->getName();
            if (!(*it)->isActive())
                continue;

            unsigned int updateIndex = updateTasksAssigned % m_scheduler_updateBVH->getNumThreads();

            m_cpuBVHUpdateTasks[updateIndex]->addBVHUpdateTask((*it));

            updateTasksAssigned++;
            ++nActive;
        }

        if (nActive > 0)
        {
            for (size_t k = 0; k < m_cpuBVHUpdateTasks.size(); ++k)
            {
                m_scheduler_updateBVH->addTask(m_cpuBVHUpdateTasks[k]);
            }

            m_scheduler_updateBVH->getScheduler()->distributeTasks();

            m_scheduler_updateBVH->getScheduler()->resumeThreads();
            m_scheduler_updateBVH->runTasks();

            m_scheduler_updateBVH->getScheduler()->pauseThreads();

            for (unsigned int k = 0; k < m_cpuBVHUpdateTasks.size(); k++)
            {
                m_cpuBVHUpdateTasks[k]->setFinished(false);
            }

            // For loop after scheduler finished processing
            for (unsigned int k = 0; k < m_cpuBVHUpdateTasks.size(); k++)
            {
                CPUBVHUpdateTask* task_k = m_cpuBVHUpdateTasks[k];
                std::vector<sofa::core::CollisionModel*>& collision_models = task_k->getCollisionModels();
                for (size_t l = 0; l < collision_models.size(); ++l)
                    vectBoundingVolume.push_back(collision_models[l]->getFirst());
            }
        }

        sofa::helper::AdvancedTimer::stepEnd("BBox");
#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printCloseNode("ComputeBoundingTree");
#endif
        msg_info("ZyMultiThreadedCollisionPipeline") << "DefaultPipeline::doCollisionDetection, Computed " << nActive << " BBoxs; vectBoundingVolume.size() = " << vectBoundingVolume.size();
    }
    // then we start the broad phase
    if (broadPhaseDetection == NULL) return; // can't go further
    msg_info("ZyMultiThreadedCollisionPipeline") << "DefaultPipeline::doCollisionDetection, BroadPhaseDetection " << broadPhaseDetection->getName();
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("BroadPhase");
#endif
    sofa::helper::AdvancedTimer::stepBegin("BroadPhase");
    intersectionMethod->beginBroadPhase();
    broadPhaseDetection->beginBroadPhase();
    broadPhaseDetection->addCollisionModels(vectBoundingVolume);  // detection is done there
    broadPhaseDetection->endBroadPhase();
    intersectionMethod->endBroadPhase();
    sofa::helper::AdvancedTimer::stepEnd("BroadPhase");

    zyUpdateInternalGeometry();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("BroadPhase");
#endif

    // then we start the narrow phase
    if (narrowPhaseDetection == NULL) return; // can't go further
    msg_info("ZyMultiThreadedCollisionPipeline") << "DefaultPipeline::doCollisionDetection, NarrowPhaseDetection " << narrowPhaseDetection->getName();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("NarrowPhase");
#endif
    sofa::helper::AdvancedTimer::stepBegin("NarrowPhase");
    intersectionMethod->beginNarrowPhase();
    narrowPhaseDetection->beginNarrowPhase();
    sofa::helper::vector<std::pair<CollisionModel*, CollisionModel*> >& vectCMPair = broadPhaseDetection->getCollisionModelPairs();
    msg_info("ZyMultiThreadedCollisionPipeline") << "DefaultPipeline::doCollisionDetection, " << vectCMPair.size() << " colliding model pairs";
    narrowPhaseDetection->addCollisionPairs(vectCMPair);
    narrowPhaseDetection->endNarrowPhase();
    intersectionMethod->endNarrowPhase();
    sofa::helper::AdvancedTimer::stepEnd("NarrowPhase");

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("NarrowPhase");
#endif

    sofa::helper::AdvancedTimer::stepEnd("doCollisionDetection");
}

// Almost copy/pasted from DefaultPipeline, but according to the debugger the DefaultPipeline contactManager member is NULL, thus no contacts are generated
// Here however, the contactManager seems to be valid (per init() function); so, WTF?
void ZyMultiThreadedCollisionPipeline::doCollisionResponse()
{
    core::objectmodel::BaseContext* scene = getContext();
    // then we start the creation of contacts
    if (contactManager == NULL)
    {
        serr << "No ContactManager instance set, can't create any collision responses!" << sendl;
        return; // can't go further
    }

    msg_info("ZyMultiThreadedCollisionPipeline") <<  "Create Contacts: " << contactManager->getName() << " using ContactManager type '" << contactManager->getTypeName();
    msg_info("ZyMultiThreadedCollisionPipeline") <<  "NarrowPhaseDetection used: " << narrowPhaseDetection->getName() << " of type '" << narrowPhaseDetection->getTypeName();

    NarrowPhaseDetection::DetectionOutputMap& detection_outputs = narrowPhaseDetection->getMutableDetectionOutputs();
    msg_info("ZyMultiThreadedCollisionPipeline") <<  "DetectionOutputVector count = " << detection_outputs.size();

    NarrowPhaseDetection::DetectionOutputMap copied_outputs;
    for (NarrowPhaseDetection::DetectionOutputMap::iterator do_it = detection_outputs.begin(); do_it != detection_outputs.end(); ++do_it)
    {
        const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>& collision_pair = do_it->first;
        DetectionOutputVector* do_vec = do_it->second;
        msg_info("ZyMultiThreadedCollisionPipeline") <<  " - Collision pair: " << collision_pair.first->getName() << " (type: " << collision_pair.first->getClassName() << ") <-> " << collision_pair.second->getName() << " (type: " << collision_pair.second->getClassName() << ")";
        if (do_vec == NULL)
        {
            msg_info("ZyMultiThreadedCollisionPipeline") <<  "   --> DetectionOutputVector EMTPY/NULL; not passing to ContactManager!";
            continue;
        }
        else
        {
            msg_info("ZyMultiThreadedCollisionPipeline") <<  "   --> Elements in DetectionOutputVector: " << do_vec->size();
        }

        sofa::core::CollisionModel* model_1 = collision_pair.first;
        sofa::core::CollisionModel* model_2 = collision_pair.second;
        while (model_1->getNext() != NULL)
        {
            model_1 = model_1->getNext();
        }

        while (model_2->getNext() != NULL)
        {
            model_2 = model_2->getNext();
        }
        std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> parent_pair = std::make_pair(model_1, model_2);

        msg_info("ZyMultiThreadedCollisionPipeline") <<  "  -> Parent pair: " << parent_pair.first->getName() << " (type: " << parent_pair.first->getClassName() << ") <-> " << parent_pair.second->getName() << " (type: " << parent_pair.second->getClassName() << ")";

        copied_outputs[parent_pair] = do_vec;
    }

    msg_info("ZyMultiThreadedCollisionPipeline") <<  "Copied outputs size = " << copied_outputs.size();

    contactManager->createContacts(copied_outputs);

    // finally we start the creation of collisionGroup

    const sofa::helper::vector<Contact::SPtr>& contacts = contactManager->getContacts();

    // First we remove all contacts with non-simulated objects and directly add them
    sofa::helper::vector<Contact::SPtr> notStaticContacts;


    for (sofa::helper::vector<Contact::SPtr>::const_iterator it = contacts.begin(); it != contacts.end(); ++it)
    {
        Contact::SPtr c = *it;
        if (!c->getCollisionModels().first->isSimulated())
        {
            c->createResponse(c->getCollisionModels().second->getContext());
        }
        else if (!c->getCollisionModels().second->isSimulated())
        {
            c->createResponse(c->getCollisionModels().first->getContext());
        }
        else
            notStaticContacts.push_back(c);
    }


    if (groupManager == NULL)
    {
        msg_info("ZyMultiThreadedCollisionPipeline") << "Linking all contacts to Scene" << sendl;
        for (sofa::helper::vector<Contact::SPtr>::const_iterator it = notStaticContacts.begin(); it != notStaticContacts.end(); ++it)
        {
            (*it)->createResponse(scene);
        }
    }
    else
    {
        msg_info("ZyMultiThreadedCollisionPipeline") << "Create Groups " << groupManager->getName() << sendl;
        groupManager->createGroups(scene, notStaticContacts);
    }
}

void ZyMultiThreadedCollisionPipeline::doCollisionReset()
{
    // Taken from DefaultPipeline
    msg_info("ZyMultiThreadedCollisionPipeline") << "TruPhysicsPipeline::doCollisionReset: Reset collisions" << sendl;

    // clear all contacts
    if (contactManager != NULL)
    {
        const sofa::helper::vector<Contact::SPtr>& contacts = contactManager->getContacts();
        for (sofa::helper::vector<Contact::SPtr>::const_iterator it = contacts.begin(); it != contacts.end(); ++it)
        {
            (*it)->removeResponse();
        }
    }
    // clear all collision groups
    if (groupManager != NULL)
    {
        core::objectmodel::BaseContext* scene = getContext();
        groupManager->clearGroups(scene);
    }
}

void ZyMultiThreadedCollisionPipeline::zyUpdateInternalGeometry()
{
    // update internal geometry here, instead of in the animation loop, so that it is not called unnecessarily
    sofa::helper::vector<std::pair<CollisionModel*, CollisionModel*> >& tmpCMpairs = broadPhaseDetection->getCollisionModelPairs();
    std::set<CollisionModel*> tmpModelSet;
    for (sofa::helper::vector<std::pair<CollisionModel*, CollisionModel*> >::const_iterator it = tmpCMpairs.begin(); it != tmpCMpairs.end(); it++)
    {
        //msg_info("ZyMultiThreadedCollisionPipeline") <<  std::endl << (*it).first->getName() << "," << (*it).second->getName();
        tmpModelSet.insert((*it).first);
        tmpModelSet.insert((*it).second);
    }

    for (std::set<CollisionModel*>::const_iterator it = tmpModelSet.begin(); it != tmpModelSet.end(); it++)
    {
        //msg_info("ZyMultiThreadedCollisionPipeline") <<  std::endl << "---" << "Calling updateInternalGeometry for " << (*it)->getName() << "right now.";
        (*it)->updateInternalGeometry();
    }
}
