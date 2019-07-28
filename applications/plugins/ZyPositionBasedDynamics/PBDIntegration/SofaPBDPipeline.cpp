#include "SofaPBDPipeline.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>

#include "SofaPBDPointCollisionModel.h"
#include "SofaPBDLineCollisionModel.h"
#include "SofaPBDTriangleCollisionModel.h"

using namespace sofa::simulation::PBDSimulation;
using namespace sofa::core::collision;

SOFA_DECL_CLASS(SofaPBDPipeline)

int SofaPBDPipelineClass = sofa::core::RegisterObject("Adapter class to integrate SOFA's collision detection with the PBD framework.")
                            .add< SofaPBDPipeline >()
                            .addDescription("Adapter class to integrate SOFA's collision detection with the PBD framework.");

SofaPBDPipeline::SofaPBDPipeline(): ZyPipelineInterface(), depth(initData(&depth, 6, "depth", "Max depth of bounding trees"))
{

}

SofaPBDPipeline::~SofaPBDPipeline()
{

}

void SofaPBDPipeline::setup(BroadPhaseDetection* broadPhaseDetection, NarrowPhaseDetection* narrowPhaseDetection, Intersection* intersection, ContactManager* contactManager, CollisionGroupManager* groupManager)
{
    msg_info("SofaPBDPipeline") << "setup()";

    if (broadPhaseDetection)
    {
        msg_info("SofaPBDPipeline") << "broadPhaseDetection: " << broadPhaseDetection->getName();
        this->pipelineBroadPhaseDetection = broadPhaseDetection;
    }
    else
    {
        msg_warning("SofaPBDPipeline") << "No valid broadPhaseDetection provided, pipeline is non-functional!";
    }

    if (narrowPhaseDetection)
    {
        msg_info("SofaPBDPipeline") << "narrowPhaseDetection: " << narrowPhaseDetection->getName();
        this->pipelineNarrowPhaseDetection = narrowPhaseDetection;
    }
    else
    {
        msg_warning("SofaPBDPipeline") << "No valid narrowPhaseDetection provided, pipeline is non-functional!";
    }

    if (intersection)
    {
        msg_info("SofaPBDPipeline") << "intersectionMethod: " << intersection->getName();
        this->pipelineIntersectionMethod = intersection;
    }
    else
    {
        msg_warning("SofaPBDPipeline") << "No valid intersectionMethod provided, pipeline is non-functional!";
    }

    if (contactManager)
    {
        msg_info("SofaPBDPipeline") << "contactManager: " << contactManager->getName();
        this->pipelineContactManager = contactManager;
    }
    else
    {
        msg_warning("SofaPBDPipeline") << "No valid contactManager provided, pipeline is non-functional!";
    }

    if (groupManager)
    {
        msg_info("SofaPBDPipeline") << "groupManager: " << groupManager->getName();
        this->pipelineGroupManager = groupManager;
    }
    else
    {
        msg_warning("SofaPBDPipeline") << "No valid groupManager provided!";
    }
}

void SofaPBDPipeline::doCollisionReset()
{
    msg_info("SofaPBDPipeline") << "doCollisionReset()";
}

void SofaPBDPipeline::doCollisionDetection(const sofa::helper::vector<core::CollisionModel*>& collisionModels)
{
    msg_info("SofaPBDPipeline") << "doCollisionDetection()";
    msg_info("SofaPBDPipeline") << "Collision model instances in scene: " << collisionModels.size();

    if (!this->pipelineIntersectionMethod || !this->pipelineNarrowPhaseDetection || !this->pipelineBroadPhaseDetection)
    {
        msg_warning("ZySingleThreadedCollisionPipelineImpl") << "Required member variable not set, collision detection pipeline non-functional!";

        if (!this->pipelineIntersectionMethod)
            msg_warning("ZySingleThreadedCollisionPipelineImpl") << "intersectionMethod instance invalid.";

        if (!this->pipelineBroadPhaseDetection)
            msg_warning("ZySingleThreadedCollisionPipelineImpl") << "broadPhaseDetection instance invalid.";

        if (!this->pipelineNarrowPhaseDetection)
            msg_warning("ZySingleThreadedCollisionPipelineImpl") << "narrowPhaseDetection instance invalid.";

        return;
    }

    simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());

    if(root == NULL)
    {
        msg_warning("SofaPBDPipeline") << "Simulation root node is NULL, doing nothing!";
        return;
    }

    std::vector<SofaPBDPointCollisionModel*> pbdPointCollisionModels;
    std::vector<SofaPBDLineCollisionModel*> pbdLineCollisionModels;
    std::vector<SofaPBDTriangleCollisionModel*> pbdTriangleCollisionModels;

    root->getTreeObjects<SofaPBDPointCollisionModel>(&pbdPointCollisionModels);
    root->getTreeObjects<SofaPBDLineCollisionModel>(&pbdLineCollisionModels);
    root->getTreeObjects<SofaPBDTriangleCollisionModel>(&pbdTriangleCollisionModels);

    msg_info("SofaPBDPipeline") << "SofaPBDPointCollisionModel instances in scene: " << pbdPointCollisionModels.size();
    msg_info("SofaPBDPipeline") << "SofaPBDLineCollisionModel instances in scene: " << pbdLineCollisionModels.size();
    msg_info("SofaPBDPipeline") << "SofaPBDTriangleCollisionModel instances in scene: " << pbdTriangleCollisionModels.size();

    msg_info("SofaPBDPipeline") << "Update BVHs";
    sofa::helper::vector<sofa::core::CollisionModel*> vectBoundingVolume;
    {
        sofa::helper::AdvancedTimer::stepBegin("BBox");

        const bool continuous = this->pipelineIntersectionMethod->useContinuous();
        const SReal dt = getContext()->getDt();

        int nActive = 0;

        sofa::helper::vector<sofa::simulation::PBDSimulation::SofaPBDPointCollisionModel*>::const_iterator it_pt;
        const sofa::helper::vector<sofa::simulation::PBDSimulation::SofaPBDPointCollisionModel*>::const_iterator itEnd_pt = pbdPointCollisionModels.end();
        for (it_pt = pbdPointCollisionModels.begin(); it_pt != itEnd_pt; ++it_pt)
        {
            msg_info("SofaPBDPipeline") << "Consider SofaPBDPointCollisionModel " << (*it_pt)->getName();
            if (!(*it_pt)->isActive())
                continue;

            int used_depth = this->pipelineBroadPhaseDetection->needsDeepBoundingTree() ? depth.getValue() : 0;

            if (continuous)
                (*it_pt)->computeContinuousBoundingTree(dt, used_depth);
            else
                (*it_pt)->computeBoundingTree(used_depth);

            vectBoundingVolume.push_back((*it_pt)->getFirst());
            ++nActive;
        }

        sofa::helper::vector<sofa::simulation::PBDSimulation::SofaPBDLineCollisionModel*>::const_iterator it_ln;
        const sofa::helper::vector<sofa::simulation::PBDSimulation::SofaPBDLineCollisionModel*>::const_iterator itEnd_ln = pbdLineCollisionModels.end();
        for (it_ln = pbdLineCollisionModels.begin(); it_ln != itEnd_ln; ++it_ln)
        {
            msg_info("SofaPBDPipeline") << "Consider SofaPBDLineCollisionModel " << (*it_ln)->getName();
            if (!(*it_ln)->isActive())
                continue;

            int used_depth = this->pipelineBroadPhaseDetection->needsDeepBoundingTree() ? depth.getValue() : 0;

            if (continuous)
                (*it_ln)->computeContinuousBoundingTree(dt, used_depth);
            else
                (*it_ln)->computeBoundingTree(used_depth);

            vectBoundingVolume.push_back((*it_ln)->getFirst());
            ++nActive;
        }

        sofa::helper::vector<sofa::simulation::PBDSimulation::SofaPBDTriangleCollisionModel*>::const_iterator it_tr;
        const sofa::helper::vector<sofa::simulation::PBDSimulation::SofaPBDTriangleCollisionModel*>::const_iterator itEnd_tr = pbdTriangleCollisionModels.end();
        for (it_tr = pbdTriangleCollisionModels.begin(); it_tr != itEnd_tr; ++it_tr)
        {
            msg_info("SofaPBDPipeline") << "Consider SofaPBDTriangleCollisionModel " << (*it_tr)->getName();
            if (!(*it_tr)->isActive())
                continue;

            int used_depth = this->pipelineBroadPhaseDetection->needsDeepBoundingTree() ? depth.getValue() : 0;

            if (continuous)
                (*it_tr)->computeContinuousBoundingTree(dt, used_depth);
            else
                (*it_tr)->computeBoundingTree(used_depth);

            vectBoundingVolume.push_back((*it_tr)->getFirst());
            ++nActive;
        }

        sofa::helper::AdvancedTimer::stepEnd("BBox");

        msg_info("ZySingleThreadedCollisionPipelineImpl") << "Computed " << nActive << " bounding volume hierarchies.";
    }
}

void SofaPBDPipeline::doCollisionResponse()
{
    msg_info("SofaPBDPipeline") << "doCollisionResponse()";
}
