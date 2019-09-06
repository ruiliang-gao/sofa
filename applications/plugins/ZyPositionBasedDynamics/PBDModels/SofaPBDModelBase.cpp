#include "SofaPBDModelBase.h"

using namespace sofa::simulation::PBDSimulation;

SofaPBDModelBase::SofaPBDModelBase(): BaseObject(), m_pbdRigidBody(nullptr)
{

}

SofaPBDModelBase::~SofaPBDModelBase()
{
    if (m_pbdRigidBody)
    {
        delete m_pbdRigidBody;
        m_pbdRigidBody = nullptr;
    }
}

bool SofaPBDModelBase::hasPBDRigidBody() const
{
    return (m_pbdRigidBody != nullptr);
}

const PBDRigidBody* SofaPBDModelBase::getPBDRigidBody() const
{
    return m_pbdRigidBody;
}

PBDRigidBody* SofaPBDModelBase::getPBDRigidBody()
{
    return m_pbdRigidBody;
}

const PBDRigidBodyGeometry& SofaPBDModelBase::getRigidBodyGeometry() const
{
    static PBDRigidBodyGeometry emptyPBDGeom;
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
