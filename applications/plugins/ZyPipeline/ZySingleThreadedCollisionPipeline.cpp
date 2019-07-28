#include "ZySingleThreadedCollisionPipeline.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/collision/CollisionGroupManager.h>

#include <sofa/helper/AdvancedTimer.h>

using namespace Zyklio::Pipeline;
using namespace sofa::component::collision;

SOFA_DECL_CLASS(ZySingleThreadedCollisionPipeline)
int ZySingleThreadedCollisionPipelineClass = sofa::core::RegisterObject("ZyPipelineInterface wrapper for SOFA's extended DefaultPipeline.")
.add< ZySingleThreadedCollisionPipeline >()
;

ZySingleThreadedCollisionPipelineImpl::ZySingleThreadedCollisionPipelineImpl() : PipelineImpl(), depth(initData(&depth, 6, "depth", "Max depth of bounding trees"))
{

}

ZySingleThreadedCollisionPipelineImpl::~ZySingleThreadedCollisionPipelineImpl()
{

}

void ZySingleThreadedCollisionPipelineImpl::computeCollisionReset()
{
    PipelineImpl::computeCollisionReset();
}

void ZySingleThreadedCollisionPipelineImpl::computeCollisionDetection()
{

}

void ZySingleThreadedCollisionPipelineImpl::computeCollisionResponse()
{
    PipelineImpl::computeCollisionResponse();
}

// Copy/paste from DefaultPipeline to satisfy the 'pure virtual' declaration in the Pipeline base class
std::set< std::string > ZySingleThreadedCollisionPipelineImpl::getResponseList() const
{
    std::set< std::string > listResponse;
    sofa::core::collision::Contact::Factory::iterator it;
    for (it = sofa::core::collision::Contact::Factory::getInstance()->begin(); it != sofa::core::collision::Contact::Factory::getInstance()->end(); ++it)
    {
        listResponse.insert(it->first);
    }
    return listResponse;
}

// Nothing is processed here
void ZySingleThreadedCollisionPipelineImpl::doCollisionReset()
{

}

// Nothing is processed here
void ZySingleThreadedCollisionPipelineImpl::doCollisionResponse()
{

}

// Single-threaded collision pair processing is done here
void ZySingleThreadedCollisionPipelineImpl::doCollisionDetection(const sofa::helper::vector<sofa::core::CollisionModel*>& collisionModels)
{
    sofa::helper::AdvancedTimer::stepBegin("doCollisionDetection");

    // First, we compute a bounding volume for the collision model (for example bounding sphere)
    // or we have loaded a collision model that knows its other model

    if (!this->intersectionMethod || !this->narrowPhaseDetection || !this->broadPhaseDetection)
    {
        msg_warning("ZySingleThreadedCollisionPipelineImpl") << "Required member variable not set, collision detection pipeline non-functional!";

        if (!this->intersectionMethod)
            msg_warning("ZySingleThreadedCollisionPipelineImpl") << "intersectionMethod instance invalid.";

        if (!this->broadPhaseDetection)
            msg_warning("ZySingleThreadedCollisionPipelineImpl") << "broadPhaseDetection instance invalid.";

        if (!this->narrowPhaseDetection)
            msg_warning("ZySingleThreadedCollisionPipelineImpl") << "narrowPhaseDetection instance invalid.";

        return;
    }

    msg_info("ZySingleThreadedCollisionPipelineImpl") << "Update BVHs";
    sofa::helper::vector<sofa::core::CollisionModel*> vectBoundingVolume;
    {
        sofa::helper::AdvancedTimer::stepBegin("BBox");

        const bool continuous = this->intersectionMethod->useContinuous();
        const SReal dt = getContext()->getDt();

        sofa::helper::vector<sofa::core::CollisionModel*>::const_iterator it;
        const sofa::helper::vector<sofa::core::CollisionModel*>::const_iterator itEnd = collisionModels.end();
        int nActive = 0;

        for (it = collisionModels.begin(); it != itEnd; ++it)
        {
            msg_info("ZySingleThreadedCollisionPipelineImpl") << "Consider model " << (*it)->getName();
            if (!(*it)->isActive())
                continue;

            int used_depth = this->broadPhaseDetection->needsDeepBoundingTree() ? depth.getValue() : 0;

            if (continuous)
                (*it)->computeContinuousBoundingTree(dt, used_depth);
            else
                (*it)->computeBoundingTree(used_depth);

            vectBoundingVolume.push_back((*it)->getFirst());
            ++nActive;
        }
        sofa::helper::AdvancedTimer::stepEnd("BBox");

        msg_info("ZySingleThreadedCollisionPipelineImpl") << "Computed " << nActive << " bounding volume hierarchies.";
    }
    // then we start the broad phase
    if (this->broadPhaseDetection == NULL)
    {
        msg_warning("ZySingleThreadedCollisionPipelineImpl") << "broadPhaseDetection invalid, collision detection can not proceed!";
        return; // can't go further
    }
    msg_info("ZySingleThreadedCollisionPipelineImpl") << "doCollisionDetection, BroadPhaseDetection " << broadPhaseDetection->getName() << sendl;

    sofa::helper::AdvancedTimer::stepBegin("BroadPhase");
    this->intersectionMethod->beginBroadPhase();
    this->broadPhaseDetection->beginBroadPhase();
    this->broadPhaseDetection->addCollisionModels(vectBoundingVolume);  // detection is done there
    this->broadPhaseDetection->endBroadPhase();
    this->intersectionMethod->endBroadPhase();

    sofa::helper::AdvancedTimer::stepEnd("BroadPhase");

    // then we start the narrow phase
    if (this->narrowPhaseDetection == NULL)
    {
        msg_warning("ZySingleThreadedCollisionPipelineImpl") << "narrowPhaseDetection invalid, collision detection can not proceed!";
        return; // can't go further
    }

    msg_info("ZySingleThreadedCollisionPipelineImpl") << "DefaultPipeline::doCollisionDetection, NarrowPhaseDetection " << narrowPhaseDetection->getName() << sendl;


    sofa::helper::AdvancedTimer::stepBegin("NarrowPhase");
    this->intersectionMethod->beginNarrowPhase();
    this->narrowPhaseDetection->beginNarrowPhase();
    sofa::helper::vector<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& vectCMPair = broadPhaseDetection->getCollisionModelPairs();

    msg_info("ZySingleThreadedCollisionPipelineImpl") << "doCollisionDetection, " << vectCMPair.size() << " colliding model pairs" << sendl;

    this->narrowPhaseDetection->addCollisionPairs(vectCMPair);
    this->narrowPhaseDetection->endNarrowPhase();
    this->intersectionMethod->endNarrowPhase();
    sofa::helper::AdvancedTimer::stepEnd("NarrowPhase");

    sofa::helper::AdvancedTimer::stepEnd("doCollisionDetection");
}


ZySingleThreadedCollisionPipeline::ZySingleThreadedCollisionPipeline():
    pipelineBroadPhaseDetection(nullptr),
    pipelineNarrowPhaseDetection(nullptr),
    pipelineIntersectionMethod(nullptr),
    pipelineContactManager(nullptr),
    pipelineGroupManager(nullptr)
{
    m_pipelineImpl = new ZySingleThreadedCollisionPipelineImpl();
}

ZySingleThreadedCollisionPipeline::~ZySingleThreadedCollisionPipeline()
{
    if (m_pipelineImpl)
    {
        delete m_pipelineImpl;
        m_pipelineImpl = NULL;
    }
}

void ZySingleThreadedCollisionPipeline::init()
{
    if (!m_doInit)
    {
        return;
    }

    m_pipelineImpl->init();
}

void ZySingleThreadedCollisionPipeline::bwdInit()
{
    if (!m_doInit)
    {
        return;
    }

    m_pipelineImpl->bwdInit();

    setActive(false);
}

void ZySingleThreadedCollisionPipeline::setup(BroadPhaseDetection* broadPhaseDetection, NarrowPhaseDetection* narrowPhaseDetection, Intersection* intersection, ContactManager* contactManager, CollisionGroupManager* groupManager)
{
    msg_info("ZySingleThreadedCollisionPipeline") << "setup()";

    if (broadPhaseDetection)
    {
        msg_info("ZySingleThreadedCollisionPipeline") << "broadPhaseDetection: " << broadPhaseDetection->getName();
        this->pipelineBroadPhaseDetection = broadPhaseDetection;
        this->m_pipelineImpl->broadPhaseDetection = broadPhaseDetection;
    }
    else
    {
        msg_warning("ZySingleThreadedCollisionPipeline") << "No valid broadPhaseDetection provided, pipeline is non-functional!";
    }

    if (narrowPhaseDetection)
    {
        msg_info("ZySingleThreadedCollisionPipeline") << "narrowPhaseDetection: " << narrowPhaseDetection->getName();
        this->pipelineNarrowPhaseDetection = narrowPhaseDetection;
        this->m_pipelineImpl->narrowPhaseDetection = narrowPhaseDetection;
    }
    else
    {
        msg_warning("ZySingleThreadedCollisionPipeline") << "No valid narrowPhaseDetection provided, pipeline is non-functional!";
    }

    if (intersection)
    {
        msg_info("ZySingleThreadedCollisionPipeline") << "intersectionMethod: " << intersection->getName();
        this->pipelineIntersectionMethod = intersection;
        this->m_pipelineImpl->intersectionMethod = intersection;
    }
    else
    {
        msg_warning("ZySingleThreadedCollisionPipeline") << "No valid intersectionMethod provided, pipeline is non-functional!";
    }

    if (contactManager)
    {
        msg_info("ZySingleThreadedCollisionPipeline") << "contactManager: " << contactManager->getName();
        this->pipelineContactManager = contactManager;
        this->m_pipelineImpl->contactManager = contactManager;
    }
    else
    {
        msg_warning("ZySingleThreadedCollisionPipeline") << "No valid contactManager provided, pipeline is non-functional!";
    }

    if (groupManager)
    {
        msg_info("ZySingleThreadedCollisionPipeline") << "groupManager: " << groupManager->getName();
        this->pipelineGroupManager = groupManager;
        this->m_pipelineImpl->groupManager = groupManager;
    }
    else
    {
        msg_warning("ZySingleThreadedCollisionPipeline") << "No valid groupManager provided!";
    }
}

// Just forward the calls to PipelineImpl
// And time the function calls
void ZySingleThreadedCollisionPipeline::doCollisionDetection(const sofa::helper::vector<sofa::core::CollisionModel*>& collisionModels)
{
    m_pipelineImpl->doCollisionDetection(collisionModels);
}

void ZySingleThreadedCollisionPipeline::doCollisionResponse()
{
    m_pipelineImpl->doCollisionResponse();
}

void ZySingleThreadedCollisionPipeline::doCollisionReset()
{
    m_pipelineImpl->doCollisionReset();
}

