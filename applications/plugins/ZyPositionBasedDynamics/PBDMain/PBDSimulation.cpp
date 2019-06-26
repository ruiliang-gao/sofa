#include "PBDSimulation.h"
#include "PBDUtils/PBDTimeManager.h"

#include "PBDTimeStep.h"
#include "PBDMain/PBDTimeStepController.h"

// #include "Utils/Timing.h"

using namespace sofa::simulation::PBDSimulation;
using namespace std;

PBDSimulation* PBDSimulation::current = nullptr;

PBDSimulation::PBDSimulation(): sofa::core::objectmodel::BaseObject()
{
    m_timeStep = nullptr;
    m_simulationMethodChanged = nullptr;
}

PBDSimulation::~PBDSimulation()
{
    delete m_timeStep;
    delete PBDTimeManager::getCurrent();

    current = nullptr;
}

PBDSimulation* PBDSimulation::getCurrent()
{
    if (current == nullptr)
    {
        current = new PBDSimulation();
        current->init();
    }
    return current;
}

void PBDSimulation::setCurrent(PBDSimulation* tm)
{
    current = tm;
}

bool PBDSimulation::hasCurrent()
{
    return (current != nullptr);
}

void PBDSimulation::init()
{
    initParameters();
    setSimulationMethod(static_cast<int>(PBDSimulationMethods::PBD));
}

void PBDSimulation::initParameters()
{
    initData(&GRAVITATION, sofa::defaulttype::Vec3d(0, 0, -9.81), "Gravitation", "Vector to define the gravitational acceleration.");
    GRAVITATION.setGroup("Simulation");

    helper::OptionsGroup methodOptions(3, "0 - Position-Based Dynamics (PBD)",
                                       "1 - eXtended Position-Based Dynamics (XPBD)",
                                       "2 - Impulse-Based Dynamic Simulation (IBDS)"
                                       );

    initData(&SIMULATION_METHOD, "Simulation method", "Simulation method");
    SIMULATION_METHOD.setGroup("Simulation");
    methodOptions.setSelectedItem(0);
    SIMULATION_METHOD.setValue(methodOptions);
}

void PBDSimulation::reset()
{
    m_model->reset();
    if (m_timeStep)
        m_timeStep->reset();

    PBDTimeManager::getCurrent()->setTime(static_cast<Real>(0.0));
    PBDTimeManager::getCurrent()->setTimeStepSize(static_cast<Real>(0.005));
}

void PBDSimulation::setSimulationMethod(const int val)
{
    PBDSimulationMethods method = static_cast<PBDSimulationMethods>(val);
    if ((method < PBDSimulationMethods::PBD) || (method >= PBDSimulationMethods::NumSimulationMethods))
        method = PBDSimulationMethods::PBD;

    if ((int) method == SIMULATION_METHOD.getValue().getSelectedId())
        return;

    delete m_timeStep;
    m_timeStep = nullptr;

    SIMULATION_METHOD.setValue(val);

    if (method == PBDSimulationMethods::PBD)
    {
        m_timeStep = new PBDTimeStepController();
        m_timeStep->init();
        PBDTimeManager::getCurrent()->setTimeStepSize(static_cast<Real>(0.005));
    }
    else if (method == PBDSimulationMethods::XPBD)
    {
        msg_info("PBDSimulation") << "XPBD not implemented yet.";
    }
    else if (method == PBDSimulationMethods::IBDS)
    {
        msg_info("PBDSimulation") << "IBDS not implemented yet.";
    }

    if (m_simulationMethodChanged != nullptr)
        m_simulationMethodChanged();
}

void PBDSimulation::setSimulationMethodChangedCallback(std::function<void()> const& callBackFct)
{
    m_simulationMethodChanged = callBackFct;
}
