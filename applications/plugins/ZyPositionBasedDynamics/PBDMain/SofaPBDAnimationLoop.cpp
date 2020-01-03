#include "SofaPBDAnimationLoop.h"
#include "SimulationModel.h"
#include "Utils/Timing.h"
#include "Utils/Logger.h"

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

using namespace sofa::simulation::PBDSimulation;

INIT_TIMING

INIT_LOGGING

SOFA_DECL_CLASS(SofaPBDAnimationLoop)

int SofaPBDAnimationLoopClass = sofa::core::RegisterObject("DefaultAnimationLoop-derived class to run the PBD simulation loop.")
                            .add< SofaPBDAnimationLoop >()
                            .addDescription("DefaultAnimationLoop-derived class to run the PBD simulation loop.");

SofaPBDAnimationLoop::SofaPBDAnimationLoop(sofa::simulation::Node*& gnode):
    sofa::simulation::DefaultAnimationLoop(gnode), m_simulation(nullptr),
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

SofaPBDSimulation* SofaPBDAnimationLoop::getSimulation()
{
    return m_simulation;
}

void SofaPBDAnimationLoop::setNode(simulation::Node* node)
{
    msg_info("SofaPBDAnimationLoop") << "setNode(" << node->getName() << ")";
    DefaultAnimationLoop::setNode(node);
}

void SofaPBDAnimationLoop::init()
{
    if (!m_simulation)
    {
        msg_info("SofaPBDAnimationLoop") << "Instantiating SofaPBDSimulation instance.";
        m_simulation = new SofaPBDSimulation();
        SofaPBDSimulation::setCurrent(m_simulation);

        m_simulation->init();
    }
}

void SofaPBDAnimationLoop::bwdInit()
{
    if (m_simulation)
    {
        m_simulation->bwdInit();
    }
}

void SofaPBDAnimationLoop::reset()
{
    if (m_simulation)
    {
        m_simulation->reset();
    }
}

void SofaPBDAnimationLoop::cleanup()
{
    if (m_simulation)
    {
        m_simulation->cleanup();
        delete m_simulation;
        m_simulation = nullptr;
    }
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
    TimeManager::getCurrent()->setTimeStepSize(dtPerSubStep);

    sofa::helper::AdvancedTimer::stepBegin("StepPBDTimeLoop");
    msg_info("SofaPBDAnimationLoop") << "Sub steps: " << SUB_STEPS_PER_ITERATION.getValue();
    for (unsigned int k = 0; k < 1 /*SUB_STEPS_PER_ITERATION.getValue()*/; k++)
    {
        msg_info("SofaPBDAnimationLoop") << "Sub-step " << k;
        SofaPBDTimeStepInterface* timeStep = m_simulation->getTimeStep();
        if (timeStep)
        {
            timeStep->step(params, dt);
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
        AnimateEndEvent ev(dt);
        PropagateEventVisitor act(params, &ev);
        gnode->execute(act);
    }
    sofa::helper::AdvancedTimer::stepEnd("AnimateBeginEvent");

    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    gnode->execute<UpdateMappingVisitor>(params);
    {
        UpdateMappingEndEvent ev (dt);
        PropagateEventVisitor act(params , &ev);
        gnode->execute(act);
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");

    // TODO: Write a customized UpdateBoundingBoxVisitor that correctly recognizes PBD collision models!
    /*if (!SOFA_NO_UPDATE_BBOX)
    {
        sofa::helper::ScopedAdvancedTimer timer("UpdateBBox");
        gnode->execute< UpdateBoundingBoxVisitor >(params);
    }*/

    m_prevDt = m_dt;

    msg_info("SofaPBDAnimationLoop") << "==================================================================";
    msg_info("SofaPBDAnimationLoop") << "Simulation times -- SOFA: " << m_dt << "; PBD: " << m_simulation->getTimeStep()->getTime();
    msg_info("SofaPBDAnimationLoop") << "==================================================================";

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif
}

void SofaPBDAnimationLoop::draw(const core::visual::VisualParams* vparams)
{
    if (m_simulation)
        m_simulation->draw(vparams);
}
