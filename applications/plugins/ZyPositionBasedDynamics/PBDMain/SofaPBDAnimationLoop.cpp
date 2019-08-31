#include "SofaPBDAnimationLoop.h"
#include "PBDModels/PBDSimulationModel.h"

#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/Simulation.h>

#include <SofaBaseCollision/DefaultPipeline.h>

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/visual/VisualParams.h>

#include "PBDIntegration/SofaPBDCollisionVisitor.h"

using namespace sofa::simulation::PBDSimulation;

SOFA_DECL_CLASS(SofaPBDAnimationLoop)

int SofaPBDAnimationLoopClass = sofa::core::RegisterObject("DefaultAnimationLoop-derived class to run the PBD simulation loop.")
                            .add< SofaPBDAnimationLoop >()
                            .addDescription("DefaultAnimationLoop-derived class to run the PBD simulation loop.");

SofaPBDAnimationLoop::SofaPBDAnimationLoop(sofa::simulation::Node*& gnode):
    sofa::simulation::DefaultAnimationLoop(gnode), m_simulation(nullptr), m_collisionPipeline(nullptr),
    SUB_STEPS_PER_ITERATION(initData(&SUB_STEPS_PER_ITERATION, 5, "SubStepsPerIteration", "Number of solver substeps per iteration of the simulation"))
{
    m_dt = 0.0;
    m_prevDt = 0.0;
}

SofaPBDAnimationLoop::~SofaPBDAnimationLoop()
{
    if (m_simulation)
    {
        delete m_simulation;
        m_simulation = NULL;
    }
}

void SofaPBDAnimationLoop::setNode(simulation::Node* node)
{
    msg_info("SofaPBDAnimationLoop") << "setNode(" << node->getName() << ")";
    DefaultAnimationLoop::setNode(node);
}

void SofaPBDAnimationLoop::init()
{
    if (!gnode)
        gnode = dynamic_cast<sofa::simulation::Node*>(this->getContext());

    sofa::simulation::Node::SPtr currentRootNode = sofa::simulation::getSimulation()->getCurrentRootNode();
    if (currentRootNode && currentRootNode->collisionPipeline)
    {
        m_collisionPipeline = currentRootNode->collisionPipeline.get();
        msg_info("SofaPBDAnimationLoop") << "currentRootNode has a valid collisionPipeline instance.";
    }
    else
    {
        if (!currentRootNode)
        {
            msg_error("SofaPBDAnimationLoop") << "currentRootNode is a NULL pointer!";
        }
        else
        {
            msg_warning("SofaPBDAnimationLoop") << "currentRootNode has no valid collisionPipeline instance set!";

            BaseObjectDescription desc("DefaultCollisionPipeline", "DefaultPipeline");
            BaseObject::SPtr obj = sofa::core::ObjectFactory::getInstance()->createObject(currentRootNode.get(), &desc);
            if (obj)
            {
                m_collisionPipelineLocal.reset(dynamic_cast<sofa::core::collision::Pipeline*>(obj.get()));
                msg_info("SofaPBDAnimationLoop") << "Instantiated Pipeline object: " << m_collisionPipelineLocal->getName() << " of type " << m_collisionPipelineLocal->getTypeName();
            }
            else
            {
                msg_error("SofaPBDAnimationLoop") << "Failed to instantiate Pipeline object. Collision detection will not be functional!";
            }
        }
    }
    m_context = gnode->getContext();

    m_simulation = new SofaPBDSimulation();
    SofaPBDSimulation::setCurrent(m_simulation);

    m_simulation->init();
}

void SofaPBDAnimationLoop::bwdInit()
{
    m_simulation->bwdInit();
}

void SofaPBDAnimationLoop::step(const sofa::core::ExecParams *params, SReal dt)
{
    if (dt == 0)
        dt = this->gnode->getDt();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    msg_info("SofaPBDAnimationLoop") << "step(" << dt << ")";

    sofa::helper::AdvancedTimer::stepBegin("AnimateBeginEvent");
    {
        AnimateBeginEvent ev(dt);
        PropagateEventVisitor act(params, &ev);
        gnode->execute(act);
    }
    sofa::helper::AdvancedTimer::stepEnd("AnimateBeginEvent");

    SReal startTime = gnode->getTime();

    m_dt = startTime + dt;
    SReal timeDelta = m_dt - m_prevDt;

    msg_info("SofaPBDAnimationLoop") << "startTime = " << startTime << ", timeDelta = " << timeDelta << " (" << startTime << " + " << dt << ")";

    sofa::helper::AdvancedTimer::stepBegin("BehaviorUpdatePositionVisitor");
    BehaviorUpdatePositionVisitor beh(params , dt);
    gnode->execute(beh);
    sofa::helper::AdvancedTimer::stepEnd("BehaviorUpdatePositionVisitor");

    sofa::helper::AdvancedTimer::stepBegin("AnimateVisitor");
    AnimateVisitor act(params, dt);
    gnode->execute(act);
    sofa::helper::AdvancedTimer::stepEnd("AnimateVisitor");

    Real dtPerSubStep = timeDelta / SUB_STEPS_PER_ITERATION.getValue();

    msg_info("SofaPBDAnimationLoop") << "Setting per-substep time delta to: " << dtPerSubStep;
    PBDTimeManager::getCurrent()->setTimeStepSize(dtPerSubStep);

    sofa::helper::AdvancedTimer::stepBegin("StepPBDTimeLoop");
    PBDSimulationModel* model = m_simulation->getModel();
    msg_info("SofaPBDAnimationLoop") << "Sub steps: " << SUB_STEPS_PER_ITERATION.getValue();
    for (unsigned int k = 0; k < SUB_STEPS_PER_ITERATION.getValue(); k++)
    {
        sofa::helper::AdvancedTimer::stepBegin("SofaPBDCollisionVisitor");

        msg_info("SofaPBDAnimationLoop") << "Starting collision detection.";
        if (m_collisionPipeline)
        {
            msg_info("SofaPBDAnimationLoop") << "Using collision pipeline instance from simulation root node: " << m_collisionPipeline->getName();
            SofaPBDCollisionVisitor pbd_col_visitor(m_collisionPipeline, params, dt);
            gnode->execute(pbd_col_visitor);
        }
        else
        {
            msg_info("SofaPBDAnimationLoop") << "Using locally instantiated collision pipeline object: " << m_collisionPipelineLocal->getName();
            SofaPBDCollisionVisitor pbd_col_visitor(m_collisionPipelineLocal.get(), params, dt);
            gnode->execute(pbd_col_visitor);
        }

        sofa::helper::AdvancedTimer::stepEnd("SofaPBDCollisionVisitor");

        msg_info("SofaPBDAnimationLoop") << "Sub-step " << k;
        SofaPBDTimeStep* timeStep = m_simulation->getTimeStep();
        if (timeStep)
        {
            timeStep->step(*model);
        }
        else
        {
            msg_error("SofaPBDAnimationLoop") << "SofaPBDTimeStep instance not initialized, can't step!";
        }
    }
    sofa::helper::AdvancedTimer::stepEnd("StepPBDTimeLoop");

    sofa::helper::AdvancedTimer::stepBegin("UpdateSimulationContextVisitor");
    gnode->setTime(startTime + dt);
    gnode->execute< UpdateSimulationContextVisitor >(params);
    sofa::helper::AdvancedTimer::stepEnd("UpdateSimulationContextVisitor");

    sofa::helper::AdvancedTimer::stepBegin("AnimateBeginEvent");
    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        gnode->execute ( act );
    }
    sofa::helper::AdvancedTimer::stepEnd("AnimateBeginEvent");

    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    gnode->execute< UpdateMappingVisitor >(params);
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params , &ev );
        gnode->execute ( act );
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");

    // TODO: Write a customized UpdateBoundingBoxVisitor that correctly recognizes PBD collision models!
    /*if (!SOFA_NO_UPDATE_BBOX)
    {
        sofa::helper::ScopedAdvancedTimer timer("UpdateBBox");
        gnode->execute< UpdateBoundingBoxVisitor >(params);
    }*/

    m_prevDt = m_dt;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif
}
