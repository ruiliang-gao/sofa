#include "SofaPBDCollisionVisitor.h"
#include "SofaPBDPipeline.h"

using namespace sofa::simulation::PBDSimulation;
using namespace sofa::core;
using namespace sofa::simulation;

SofaPBDCollisionVisitor::SofaPBDCollisionVisitor(sofa::core::collision::Pipeline* pipeline, const core::ExecParams* params, SReal dt)
    : Visitor(params)
    , m_pipeline(pipeline)
    , dt(dt)
    , firstNodeVisited(false)
{

}

SofaPBDCollisionVisitor::SofaPBDCollisionVisitor(sofa::core::collision::Pipeline* pipeline, const core::ExecParams* params)
    : Visitor(params)
    , m_pipeline(pipeline)
    , dt(0)
    , firstNodeVisited(false)
{

}

void SofaPBDCollisionVisitor::processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj)
{
    sofa::helper::AdvancedTimer::stepBegin("Collision detection", obj);

    sofa::helper::AdvancedTimer::stepBegin("BeginCollisionEvent", obj);
    {
        CollisionBeginEvent evBegin;
        PropagateEventVisitor eventPropagation(params, &evBegin);
        eventPropagation.execute(node->getContext());
    }
    sofa::helper::AdvancedTimer::stepEnd("BeginCollisionEvent", obj);

    CollisionVisitor act(this->params);
    node->execute(&act);

    sofa::helper::AdvancedTimer::stepBegin("EndCollisionEvent", obj);
    {
        CollisionEndEvent evEnd;
        PropagateEventVisitor eventPropagation( params, &evEnd);
        eventPropagation.execute(node->getContext());
    }
    sofa::helper::AdvancedTimer::stepEnd("EndCollisionEvent", obj);

    sofa::helper::AdvancedTimer::stepEnd("Collision detection", obj);
}

Visitor::Result SofaPBDCollisionVisitor::processNodeTopDown(simulation::Node* node)
{
    msg_info("SofaPBDCollisionVisitor") << "processNodeTopDown(" << node->getName() << ")";

    if (!node->isActive())
        return Visitor::RESULT_PRUNE;

    if (node->isSleeping())
        return Visitor::RESULT_PRUNE;

    if (dt == 0)
        setDt(node->getDt());
    else
        node->setDt(dt);

    if (node->collisionPipeline != NULL)
    {
        msg_info("SofaPBDCollisionVisitor") << "=======================================================================";
        msg_info("SofaPBDCollisionVisitor") << "Node has a valid collisionPipeline instance: " << node->collisionPipeline->getName() << " of type " << node->collisionPipeline->getTypeName();
        processCollisionPipeline(node, node->collisionPipeline);
        if (node->collisionPipeline->hasTag(sofa::simulation::PBDSimulation::tagPBDCollisionPipeline))
        {
            msg_info("SofaPBDCollisionVisitor") << "Increasing node->collisionPipeline SofaPBDCollisionVisitor visit counter.";
            SofaPBDPipeline* sofaPBDPipeline = dynamic_cast<SofaPBDPipeline*>(node->collisionPipeline.get());
            sofaPBDPipeline->increaseExternalCallCount();
        }
        msg_info("SofaPBDCollisionVisitor") << "=======================================================================";
    }
    else
    {
        msg_info("SofaPBDCollisionVisitor") << "=======================================================================";
        msg_info("SofaPBDCollisionVisitor") << "Node has no valid collisionPipeline instance set. Using instance provided to SofaPBDCollisionVisitor.";
        processCollisionPipeline(node, m_pipeline);
        if (m_pipeline->hasTag(sofa::simulation::PBDSimulation::tagPBDCollisionPipeline))
        {
            msg_info("SofaPBDCollisionVisitor") << "Increasing m_pipeline SofaPBDCollisionVisitor visit counter.";
            SofaPBDPipeline* sofaPBDPipeline = dynamic_cast<SofaPBDPipeline*>(m_pipeline);
            sofaPBDPipeline->increaseExternalCallCount();
        }
        msg_info("SofaPBDCollisionVisitor") << "=======================================================================";
    }

    return RESULT_PRUNE;
}
