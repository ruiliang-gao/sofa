#include "SofaPBDSimulation.h"
#include "PBDUtils/PBDTimeManager.h"

#include "PBDMain/SofaPBDTimeStep.h"

// #include "Utils/Timing.h"
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/simulation/Node.h>

#include <sofa/core/ObjectFactory.h>

using namespace sofa::simulation::PBDSimulation;

int SofaPBDSimulationClass = sofa::core::RegisterObject("Wrapper class for the PBD main simulation entry point.")
                            .add< SofaPBDSimulation >()
                            .addDescription("Wrapper class for the PBD main simulation entry point.");
using namespace std;

SofaPBDSimulation* SofaPBDSimulation::current = nullptr;

SofaPBDSimulation::SofaPBDSimulation(BaseContext *context): sofa::core::objectmodel::BaseObject()
{
    m_context = context;
    m_timeStep = nullptr;
    m_model = nullptr;
    m_simulationMethodChanged = nullptr;
}

SofaPBDSimulation::~SofaPBDSimulation()
{
    if (m_timeStep)
    {
        delete m_timeStep;
        m_timeStep = nullptr;
    }

    if (current)
    {
        delete current;
        current = nullptr;
    }

    if (m_model)
    {
        delete m_model;
        m_model = nullptr;
    }

}

SofaPBDSimulation* SofaPBDSimulation::getCurrent()
{
    if (current == nullptr)
    {
        current = new SofaPBDSimulation();
        current->init();
    }
    return current;
}

void SofaPBDSimulation::setCurrent(SofaPBDSimulation* tm)
{
    current = tm;
}

bool SofaPBDSimulation::hasCurrent()
{
    return (current != nullptr);
}

void SofaPBDSimulation::init()
{
    msg_info("PBDSimulation") << "init()";
    m_model = new PBDSimulationModel();
    m_model->init();

    initParameters();
    setSimulationMethod(static_cast<int>(PBDSimulationMethods::PBD));

    m_geometryConverter.reset(new GeometryConversion(m_model));
    m_mechObjConverter.reset(new MechObjConversion(m_model));

    if (!m_context)
        m_context = dynamic_cast<sofa::simulation::Node*>(this->getContext());

    auto topologies = m_context->getObjects<sofa::core::topology::BaseMeshTopology>(BaseContext::SearchDown);
    auto mechanicalObjects = m_context->getObjects<sofa::component::container::MechanicalObject<sofa::defaulttype::Vec3Types>>(BaseContext::SearchDown);

    msg_info("PBDSimulation") << "Found BaseMeshTopology instances: " << topologies.size();
    msg_info("PBDSimulation") << "Found MechanicalObject<sofa::defaulttype::Vec3Types> instances: " << mechanicalObjects.size();

    m_geometryConverter->setMeshTopologies(topologies);
    m_mechObjConverter->setMechanicalObjects(mechanicalObjects);

    bool convGeoResult = m_geometryConverter->convertToPBDObjects();
    bool convMechObjResult = m_mechObjConverter->convertToPBDObjects();

    msg_info("PBDSimulation") << "Converted geometries OK: " << convGeoResult << ", converted MechanicalObjects OK: " << convMechObjResult;
}

void SofaPBDSimulation::initParameters()
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

void SofaPBDSimulation::bwdInit()
{
    msg_info("PBDSimulation") << "bwdInit()";
}

void SofaPBDSimulation::reset()
{
    msg_info("PBDSimulation") << "reset()";

    if (m_model)
        m_model->reset();

    if (m_timeStep)
        m_timeStep->reset();

    PBDTimeManager::getCurrent()->setTime(static_cast<Real>(0.0));
    PBDTimeManager::getCurrent()->setTimeStepSize(static_cast<Real>(0.005));
}

void SofaPBDSimulation::setSimulationMethod(const int val)
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
        m_timeStep = new SofaPBDTimeStep();
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

void SofaPBDSimulation::setSimulationMethodChangedCallback(std::function<void()> const& callBackFct)
{
    m_simulationMethodChanged = callBackFct;
}
