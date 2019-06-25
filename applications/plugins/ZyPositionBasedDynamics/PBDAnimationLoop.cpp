#include "PBDAnimationLoop.h"
#include "PBDSimulationModel.h"

using namespace sofa::simulation::PBDSimulation;

PBDAnimationLoop::PBDAnimationLoop(sofa::simulation::Node* gnode): sofa::simulation::DefaultAnimationLoop(gnode), m_simulation(nullptr), m_simulationModel(nullptr), m_timeStep(nullptr)
{

}

void PBDAnimationLoop::init()
{
    m_simulationModel = new PBDSimulationModel();
}

void PBDAnimationLoop::bwdInit()
{
    m_simulationModel->init();
}



void PBDAnimationLoop::step(const sofa::core::ExecParams *params, SReal dt)
{

}
