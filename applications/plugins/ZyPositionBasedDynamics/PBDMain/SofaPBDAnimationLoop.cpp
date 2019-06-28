#include "SofaPBDAnimationLoop.h"
#include "PBDModels/PBDSimulationModel.h"

#include <sofa/core/ObjectFactory.h>

using namespace sofa::simulation::PBDSimulation;

int SofaPBDAnimationLoopClass = sofa::core::RegisterObject("DefaultAnimationLoop-derived class to run the PBD simulation loop.")
                            .add< SofaPBDAnimationLoop >()
                            .addDescription("DefaultAnimationLoop-derived class to run the PBD simulation loop.");

SofaPBDAnimationLoop::SofaPBDAnimationLoop(sofa::simulation::Node*& gnode): sofa::simulation::DefaultAnimationLoop(gnode), m_simulation(nullptr)
{

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

    m_context = gnode->getContext();

    m_simulation = new SofaPBDSimulation();
}

void SofaPBDAnimationLoop::bwdInit()
{
    m_simulation->init();
}

void SofaPBDAnimationLoop::step(const sofa::core::ExecParams *params, SReal dt)
{
    SOFA_UNUSED(params);
    msg_info("SofaPBDAnimationLoop") << "step(" << dt << ")";
}
