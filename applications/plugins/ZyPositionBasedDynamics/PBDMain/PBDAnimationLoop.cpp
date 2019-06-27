#include "PBDAnimationLoop.h"
#include "PBDModels/PBDSimulationModel.h"

using namespace sofa::simulation::PBDSimulation;

PBDAnimationLoop::PBDAnimationLoop(sofa::simulation::Node* gnode): sofa::simulation::DefaultAnimationLoop(gnode), m_simulation(nullptr), m_simulationModel(nullptr), m_timeStep(nullptr)
{

}

PBDAnimationLoop::~PBDAnimationLoop()
{
    if (m_timeStep)
    {
        delete m_timeStep;
        m_timeStep = NULL;
    }

    if (m_simulation)
    {
        delete m_simulation;
        m_simulation = NULL;
    }

    if (m_simulationModel)
    {
        delete m_simulationModel;
        m_simulationModel = NULL;
    }
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
