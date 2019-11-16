#include "SofaPBDSimulation.h"
#include "TimeManager.h"

#include "PBDMain/SofaPBDTimeStep.h"

// #include "Utils/Timing.h"
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/simulation/Node.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#include <PBDIntegration/SofaPBDPointCollisionModel.h>
#include <PBDIntegration/SofaPBDLineCollisionModel.h>
#include <PBDIntegration/SofaPBDTriangleCollisionModel.h>

using namespace sofa::simulation::PBDSimulation;

int SofaPBDSimulationClass = sofa::core::RegisterObject("Wrapper class for the PBD main simulation entry point.")
                            .add< SofaPBDSimulation >()
                            .addDescription("Wrapper class for the PBD main simulation entry point.");
using namespace std;

SofaPBDSimulation* SofaPBDSimulation::current = nullptr;

SofaPBDSimulation::SofaPBDSimulation(BaseContext *context): sofa::core::objectmodel::BaseObject(),
    GRAVITATION(initData(&GRAVITATION, sofa::defaulttype::Vec3d(0, -9.81, 0), "Gravitation", "Vector to define the gravitational acceleration."))
{
    m_context = context;
    m_timeStep = nullptr;

    m_rootNode = nullptr;

    msg_info("PBDSimulation") << "Instantiating PBDSimulationModel.";
    m_model = new SimulationModel();

    m_simulationMethodChanged = nullptr;
}

SofaPBDSimulation::~SofaPBDSimulation()
{
    if (m_timeStep)
    {
        m_timeStep->cleanup();
        delete m_timeStep;
        m_timeStep = nullptr;
    }

    if (m_model)
    {
        m_model->cleanup();
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
    msg_info("SofaPBDSimulation") << "init()";

    m_model->init();

    initParameters();

    msg_info("SofaPBDSimulation") << "Gravitation vector set: " << GRAVITATION.getValue();

    msg_info("SofaPBDSimulation") << "setSimulationMethod(" << PBDSimulationMethods::PBD << ")";
    setSimulationMethod(static_cast<int>(PBDSimulationMethods::PBD));

    //m_geometryConverter.reset(new GeometryConversion(m_model));
    //m_mechObjConverter.reset(new MechObjConversion(m_model));

    if (!m_context)
        m_context = dynamic_cast<sofa::simulation::Node*>(this->getContext());

    m_rootNode = sofa::simulation::getSimulation()->getCurrentRootNode();

    if (!m_context)
    {
        msg_info("SofaPBDSimulation") << "No valid context/Node object so far, trying to use current root node.";
        m_context = dynamic_cast<sofa::simulation::Node*>(m_rootNode.get());
    }

    if (m_timeStep)
        m_timeStep->init();

    if (m_rootNode)
    {   
        msg_info("SofaPBDSimulation") << "Searching for PBD collision model wrappers.";

        msg_info("SofaPBDSimulation") << "PointModel wrappers.";
        std::vector<SofaPBDPointCollisionModel*> pbdPointCollisionModels;
        sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<SofaPBDPointCollisionModel, std::vector<SofaPBDPointCollisionModel*> > pbdPtMCb(&pbdPointCollisionModels);
        m_rootNode->getObjects(sofa::core::objectmodel::TClassInfo<SofaPBDPointCollisionModel>::get(), pbdPtMCb, sofa::core::objectmodel::BaseContext::SearchDown);

        msg_info("SofaPBDSimulation") << "LineModel wrappers.";
        std::vector<SofaPBDLineCollisionModel*> pbdLineCollisionModels;
        sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<SofaPBDLineCollisionModel, std::vector<SofaPBDLineCollisionModel*> > pbdLnMCb(&pbdLineCollisionModels);
        m_rootNode->getObjects(sofa::core::objectmodel::TClassInfo<SofaPBDLineCollisionModel>::get(), pbdLnMCb, sofa::core::objectmodel::BaseContext::SearchDown);

        msg_info("SofaPBDSimulation") << "TriangleModel wrappers.";
        std::vector<SofaPBDTriangleCollisionModel*> pbdTriangleCollisionModels;
        sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<SofaPBDTriangleCollisionModel, std::vector<SofaPBDTriangleCollisionModel*> > pbdTriMCb(&pbdTriangleCollisionModels);
        m_rootNode->getObjects(sofa::core::objectmodel::TClassInfo<SofaPBDTriangleCollisionModel>::get(), pbdTriMCb, sofa::core::objectmodel::BaseContext::SearchDown);

        msg_info("SofaPBDSimulation") << "SofaPBDPointCollisionModels in scene: " << pbdPointCollisionModels.size();
        msg_info("SofaPBDSimulation") << "SofaPBDLineCollisionModels in scene: " << pbdLineCollisionModels.size();
        msg_info("SofaPBDSimulation") << "SofaPBDTriangleCollisionModels in scene: " << pbdTriangleCollisionModels.size();

        if (pbdPointCollisionModels.size() > 0)
        {
            for (size_t k = 0; k < pbdPointCollisionModels.size(); k++)
            {
                msg_info("SofaPBDSimulation") << "Init SofaPBDPointCollisionModel " << pbdPointCollisionModels[k]->getName();
                pbdPointCollisionModels[k]->init();
                pbdPointCollisionModels[k]->bwdInit();
            }
        }

        if (pbdLineCollisionModels.size() > 0)
        {
            for (size_t k = 0; k < pbdLineCollisionModels.size(); k++)
            {
                msg_info("SofaPBDSimulation") << "Init SofaPBDLineCollisionModel " << pbdLineCollisionModels[k]->getName();
                pbdLineCollisionModels[k]->init();
                pbdLineCollisionModels[k]->bwdInit();
            }
        }

        if (pbdTriangleCollisionModels.size() > 0)
        {
            for (size_t k = 0; k < pbdTriangleCollisionModels.size(); k++)
            {
                msg_info("SofaPBDSimulation") << "Init SofaPBDTriangleCollisionModel " << pbdTriangleCollisionModels[k]->getName();
                pbdTriangleCollisionModels[k]->init();
                pbdTriangleCollisionModels[k]->bwdInit();
            }
        }
    }
    else
    {
        msg_error("SofaPBDSimulation") << "Invalid context/Node object! Couldn't initialize PBDCollisionModels correctly!";
    }
}

void SofaPBDSimulation::initParameters()
{
    msg_info("SofaPBDSimulation") << "initParameters()";
    GRAVITATION.setGroup("Simulation");
    GRAVITATION.setValue(sofa::defaulttype::Vec3d(0, -9.81, 0));

    msg_info("SofaPBDSimulation") << "Gravitation vector set: " << GRAVITATION.getValue();

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

    if (m_timeStep)
        m_timeStep->bwdInit();

    /*auto topologies = m_context->getObjects<sofa::core::topology::BaseMeshTopology>(BaseContext::SearchDown);
    auto mechanicalObjects = m_context->getObjects<sofa::component::container::MechanicalObject<sofa::defaulttype::Vec3Types>>(BaseContext::SearchDown);

    msg_info("PBDSimulation") << "Found BaseMeshTopology instances: " << topologies.size();
    msg_info("PBDSimulation") << "Found MechanicalObject<sofa::defaulttype::Vec3Types> instances: " << mechanicalObjects.size();

    m_geometryConverter->setMeshTopologies(topologies);
    m_mechObjConverter->setMechanicalObjects(mechanicalObjects);

    bool convGeoResult = m_geometryConverter->convertToPBDObjects();
    bool convMechObjResult = m_mechObjConverter->convertToPBDObjects();

}
    msg_info("PBDSimulation") << "Converted geometries OK: " << convGeoResult << ", converted MechanicalObjects OK: " << convMechObjResult;*/

    SimulationModel::RigidBodyVector &rb = this->m_model->getRigidBodies();
    ParticleData& pd = this->m_model->getParticles();
    OrientationData& od = this->m_model->getOrientations();
    msg_info("SofaPBDSimulation") << "================== Simulation bwdInit ==================";
    msg_info("SofaPBDSimulation") << "Retrieved data arrays from model: rigidBodies = " << rb.size() << ", particles = " << pd.size() << ", particle orientations = " << od.size();

    const size_t numBodies = rb.size();
    for (size_t i = 0; i < numBodies; i++)
    {
        msg_info("SofaPBDSimulation") << "Rigid body " << i << " current position = " << rb[i]->getPosition()[0] << ", " << rb[i]->getPosition()[1] << ", " << rb[i]->getPosition()[2] << ")";
        msg_info("SofaPBDSimulation") << "Rigid body " << i << " current velocity = " << rb[i]->getVelocity()[0] << ", " << rb[i]->getVelocity()[1] << ", " << rb[i]->getVelocity()[2] << ")";
        msg_info("SofaPBDSimulation") << "Rigid body " << i << " current acceler. = " << rb[i]->getAcceleration()[0] << ", " << rb[i]->getAcceleration()[1] << ", " << rb[i]->getAcceleration()[2] << ")";
    }

    const size_t numParticles = pd.size();
    for (size_t i = 0; i < numParticles; i++)
    {
        msg_info("SofaPBDSimulation") << "Particle " << i << " current position = (" << pd.getPosition(i)[0] << ", " << pd.getPosition(i)[1] << ", " << pd.getPosition(i)[2] << ")";
        msg_info("SofaPBDSimulation") << "Particle " << i << " current velocity = (" << pd.getVelocity(i)[0] << ", " << pd.getVelocity(i)[1] << ", " << pd.getVelocity(i)[2] << ")";
        msg_info("SofaPBDSimulation") << "Particle " << i << " current acceler. = (" << pd.getAcceleration(i)[0] << ", " << pd.getAcceleration(i)[1] << ", " << pd.getAcceleration(i)[2] << ")";
    }

    for (size_t i = 0; i < numParticles; i++)
    {
        msg_info("SofaPBDSimulation") << "Particle " << i << " current quaternion = (" << od.getQuaternion(i).x() << ", " << od.getQuaternion(i).y() << ", " << od.getQuaternion(i).z() << ", " << od.getQuaternion(i).w() << ")";
        msg_info("SofaPBDSimulation") << "Particle " << i << " current rot. vel.  = (" << od.getVelocity(i)[0] << ", " << od.getVelocity(i)[1] << ", " << od.getVelocity(i)[2] << ")";
        msg_info("SofaPBDSimulation") << "Particle " << i << " current rot. acc.  = (" << od.getAcceleration(i)[0] << ", " << od.getAcceleration(i)[1] << ", " << od.getAcceleration(i)[2] << ")";
    }
    msg_info("SofaPBDSimulation") << "================== Simulation bwdInit ==================";
}

void SofaPBDSimulation::cleanup()
{
    msg_info("PBDSimulation") << "cleanup()";

    if (m_timeStep)
    {
        m_timeStep->cleanup();
        delete m_timeStep;
        m_timeStep = nullptr;
    }

    if (m_model)
    {
        m_model->cleanup();
        delete m_model;
        m_model = nullptr;
    }
}

void SofaPBDSimulation::reset()
{
    msg_info("PBDSimulation") << "reset()";

    if (m_timeStep)
        m_timeStep->reset();

    if (TimeManager::getCurrent())
    {
        TimeManager::getCurrent()->setTime(static_cast<Real>(0.0));
        TimeManager::getCurrent()->setTimeStepSize(static_cast<Real>(0.005));
    }

    if (m_model)
        m_model->reset();
}

void SofaPBDSimulation::setSimulationMethod(const int val)
{
    msg_info("SofaPBDSimulation") << "setSimulationMethod()";
    PBDSimulationMethods method = static_cast<PBDSimulationMethods>(val);
    if ((method < PBDSimulationMethods::PBD) || (method >= PBDSimulationMethods::NumSimulationMethods))
        method = PBDSimulationMethods::PBD;

    msg_info("SofaPBDSimulation") << "method set: " << val;

    /*if ((int) method == SIMULATION_METHOD.getValue().getSelectedId())
    {
        return;
    }*/

    if (m_timeStep)
    {
        msg_info("SofaPBDSimulation") << "Deleting existing PBDTimeStep instance.";
        delete m_timeStep;
        m_timeStep = nullptr;
    }

    SIMULATION_METHOD.setValue(val);

    if (method == PBDSimulationMethods::PBD)
    {
        msg_info("SofaPBDSimulation") << "Instantiating new PBDTimeStep object.";
        m_timeStep = new SofaPBDTimeStep(this);
        m_timeStep->init();

        Real timeStepSize = static_cast<Real>(0.005);
        msg_info("SofaPBDSimulation") << "Setting time step size to: " << timeStepSize;
        TimeManager::getCurrent()->setTimeStepSize(timeStepSize);
    }
    else if (method == PBDSimulationMethods::XPBD)
    {
        msg_error("PBDSimulation") << "XPBD not implemented yet.";
    }
    else if (method == PBDSimulationMethods::IBDS)
    {
        msg_error("PBDSimulation") << "IBDS not implemented yet.";
    }

    if (m_simulationMethodChanged != nullptr)
        m_simulationMethodChanged();
}

void SofaPBDSimulation::setSimulationMethodChangedCallback(std::function<void()> const& callBackFct)
{
    m_simulationMethodChanged = callBackFct;
}

void SofaPBDSimulation::draw(const core::visual::VisualParams* vparams)
{
    if (m_timeStep)
        m_timeStep->draw(vparams);
}
