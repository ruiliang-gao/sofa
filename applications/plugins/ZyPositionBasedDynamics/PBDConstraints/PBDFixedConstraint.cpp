#include "PBDFixedConstraint.h"
#include <PBDCommon/IdFactory.h>
#include <PBDModels/PBDSimulationModel.h>

using namespace sofa::simulation::PBDSimulation;

int PBDFixedConstraint::TYPE_ID = IDFactory::getId();

PBDFixedConstraint::PBDFixedConstraint(): PBDConstraintBase(1)
{

}

bool PBDFixedConstraint::initConstraint(PBDSimulationModel &model, int rbIndex1, const Vector3r &pos, const Quaternionr& rot)
{
    m_bodies[0] = rbIndex1;
    m_pos = pos;
    m_rot = rot;
    return true;
}

bool PBDFixedConstraint::solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter)
{
    PBDRigidBody* rb = model.getRigidBodies()[m_bodies[0]];
    rb->setPosition(m_pos);
    rb->setOldPosition(m_pos);
    rb->setRotation(m_rot);
    rb->setOldRotation(m_rot);
    rb->setVelocity(Vector3r(0, 0, 0));
    rb->setAcceleration(Vector3r(0, 0, 0));
    return true;
}
