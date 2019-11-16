#include "SofaPBDModelBase.h"
#include "PBDMain/SofaPBDSimulation.h"
#include "PBDMain/SofaPBDAnimationLoop.h"

using namespace sofa::simulation::PBDSimulation;

SofaPBDModelBase::SofaPBDModelBase(SimulationModel *model): BaseObject(), m_pbdRigidBody(nullptr)
{
    msg_info("SofaPBDModelBase") << "SimulationModel instance = " << model;
    m_simulationModel = model;
}

SofaPBDModelBase::~SofaPBDModelBase()
{
    if (m_pbdRigidBody)
    {
        // Do not delete PBD rigid body pointer here, this is done when cleaning up the SimulationModel.
        // Just invalidate the pointer.
        m_pbdRigidBody = nullptr;
    }

    // Same goes for the SimulationModel pointer
    if (m_simulationModel)
        m_simulationModel = nullptr;
}

void SofaPBDModelBase::bwdInit()
{
    msg_info("SofaPBDModelBase") << "bwdInit(" << this->getName() << ")";
    if (m_simulationModel == nullptr)
    {
        msg_warning("SofaPBDModelBase") << "SimulationModel pointer is invalid, searching for SofaPBDSimulation instance to retrieve a valid instance from.";
        BaseContext* bc = this->getContext();
        std::vector<SofaPBDAnimationLoop*> pbdSimulationInstances = bc->getObjects<SofaPBDAnimationLoop>(BaseContext::SearchRoot);
        // There is only one SofaPBDSimulation instance expected to be present in the SOFA scene at any time
        if (pbdSimulationInstances.size() == 1)
        {
            if (pbdSimulationInstances[0]->getSimulation())
            {
                m_simulationModel = pbdSimulationInstances[0]->getSimulation()->getModel();
                msg_info("SofaPBDModelBase") << "Retrieved SimulationModel instance from SofaPBDSimulation: " << m_simulationModel;
            }
        }
    }
}

bool SofaPBDModelBase::hasPBDRigidBody() const
{
    return (m_pbdRigidBody != nullptr);
}

const RigidBody* SofaPBDModelBase::getPBDRigidBody() const
{
    return m_pbdRigidBody;
}

RigidBody* SofaPBDModelBase::getPBDRigidBody()
{
    return m_pbdRigidBody;
}

RigidBodyGeometry& SofaPBDModelBase::getRigidBodyGeometry()
{
    static RigidBodyGeometry emptyPBDGeom;
    if (m_pbdRigidBody == nullptr)
    {
        msg_warning("SofaPBDModelBase") << "getRigidBodyGeometry(" << this->getName() << "): Invalid PBDRigidBodyGeometryPointer!";
        return emptyPBDGeom;
    }
    return m_pbdRigidBody->getGeometry();
}

const RigidBodyGeometry& SofaPBDModelBase::getRigidBodyGeometry() const
{
    static RigidBodyGeometry emptyPBDGeom;
    if (m_pbdRigidBody == nullptr)
    {
        msg_warning("SofaPBDModelBase") << "getRigidBodyGeometry(" << this->getName() << "): Invalid PBDRigidBodyGeometryPointer!";
        return emptyPBDGeom;
    }
    return m_pbdRigidBody->getGeometry();
}

void SofaPBDModelBase::resetTransformations()
{

}
