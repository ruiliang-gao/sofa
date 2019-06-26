#include "PBDTimeStep.h"
#include "PBDUtils/PBDTimeManager.h"
#include "PBDSimulation.h"

using namespace sofa::simulation::PBDSimulation;
using namespace std;

PBDTimeStep::PBDTimeStep(): sofa::core::objectmodel::BaseObject()
{

}

PBDTimeStep::~PBDTimeStep(void)
{
}

void PBDTimeStep::init()
{
    initParameters();
}

void PBDTimeStep::initParameters()
{

}

void PBDTimeStep::clearAccelerations(PBDSimulationModel &model)
{
    //////////////////////////////////////////////////////////////////////////
    // rigid body model
    //////////////////////////////////////////////////////////////////////////

    PBDSimulationModel::RigidBodyVector &rb = model.getRigidBodies();
    PBDSimulation *sim = PBDSimulation::getCurrent();
    sofa::defaulttype::Vec3d gravitation = sim->GRAVITATION.getValue();
    Vector3r grav(gravitation.x(), gravitation.y(), gravitation.z());
    for (size_t i = 0; i < rb.size(); i++)
    {
        // Clear accelerations of dynamic particles
        if (rb[i]->getMass() != 0.0)
        {
            Vector3r &a = rb[i]->getAcceleration();
            a = grav;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // particle model
    //////////////////////////////////////////////////////////////////////////

    ParticleData &pd = model.getParticles();
    const unsigned int count = pd.size();
    for (unsigned int i = 0; i < count; i++)
    {
        // Clear accelerations of dynamic particles
        if (pd.getMass(i) != 0.0)
        {
            Vector3r &a = pd.getAcceleration(i);
            a = grav;
        }
    }
}

void PBDTimeStep::reset()
{

}

/// TODO: Interface collision detection via PBDAnimationLoop
void PBDTimeStep::setCollisionDetection(PBDSimulationModel &model, CollisionDetection *cd)
{
    m_collisionDetection = cd;
    // m_collisionDetection->setContactCallback(contactCallbackFunction, &model);
    // m_collisionDetection->setSolidContactCallback(solidContactCallbackFunction, &model);
}

CollisionDetection *PBDTimeStep::getCollisionDetection()
{
    return m_collisionDetection;
}

/*void PBDTimeStep::contactCallbackFunction(const unsigned int contactType, const unsigned int bodyIndex1, const unsigned int bodyIndex2,
    const Vector3r &cp1, const Vector3r &cp2,
    const Vector3r &normal, const Real dist,
    const Real restitutionCoeff, const Real frictionCoeff, void *userData)
{
    SimulationModel *model = (SimulationModel*)userData;
    if (contactType == CollisionDetection::RigidBodyContactType)
        model->addRigidBodyContactConstraint(bodyIndex1, bodyIndex2, cp1, cp2, normal, dist, restitutionCoeff, frictionCoeff);
    else if (contactType == CollisionDetection::ParticleRigidBodyContactType)
        model->addParticleRigidBodyContactConstraint(bodyIndex1, bodyIndex2, cp1, cp2, normal, dist, restitutionCoeff, frictionCoeff);
}

void TimeStep::solidContactCallbackFunction(const unsigned int contactType, const unsigned int bodyIndex1, const unsigned int bodyIndex2,
    const unsigned int tetIndex, const Vector3r &bary,
    const Vector3r &cp1, const Vector3r &cp2,
    const Vector3r &normal, const Real dist,
    const Real restitutionCoeff, const Real frictionCoeff, void *userData)
{
    SimulationModel *model = (SimulationModel*)userData;
    if (contactType == CollisionDetection::ParticleSolidContactType)
        model->addParticleSolidContactConstraint(bodyIndex1, bodyIndex2, tetIndex, bary, cp1, cp2, normal, dist, restitutionCoeff, frictionCoeff);
}*/
