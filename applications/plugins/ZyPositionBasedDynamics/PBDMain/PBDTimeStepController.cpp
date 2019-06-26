#include "PBDTimeStepController.h"

#include <iostream>

#include "PBDUtils/PBDTimeManager.h"
#include "PBDynamics/PositionBasedRigidBodyDynamics.h"
#include "PBDynamics/PBDTimeIntegration.h"
#include "PBDynamics/PositionBasedDynamics.h"
// #include "Utils/Timing.h"

#include "DistanceFieldCollisionDetection.h"

using namespace sofa::simulation::PBDSimulation;
using namespace std;


PBDTimeStepController::PBDTimeStepController()
{
    m_velocityUpdateMethod = 0;
    m_iterations = 0;
    m_iterationsV = 0;
    m_maxIterations = 5;
    m_maxIterationsV = 5;

    m_collisionDetection = NULL;
}

PBDTimeStepController::~PBDTimeStepController(void)
{
}

void PBDTimeStepController::initParameters()
{
    PBDTimeStep::initParameters();

    initData(&MAX_ITERATIONS, 5, "Max. iterations", "Maximal number of iterations of the solver.");
    MAX_ITERATIONS.setGroup("PBD");

    initData(&MAX_ITERATIONS_V, "Max. velocity iterations", "Maximal number of iterations of the velocity solver.");
    MAX_ITERATIONS_V.setGroup("PBD");

    helper::OptionsGroup methodOptions(2, "0 - First Order Update",
                                       "1 - Second Order Update");

    initData(&VELOCITY_UPDATE_METHOD, "Velocity update method", "Velocity method.");
    VELOCITY_UPDATE_METHOD.setGroup("PBD");
    methodOptions.setSelectedItem(0);
    VELOCITY_UPDATE_METHOD.setValue(methodOptions);
}

void PBDTimeStepController::step(PBDSimulationModel &model)
{
    // START_TIMING("simulation step");
    PBDTimeManager *tm = PBDTimeManager::getCurrent ();
    const Real h = tm->getTimeStepSize();

    msg_info("PBDTimeStepController") << "PBDTimeStepController::step(" << h << ")";

    //////////////////////////////////////////////////////////////////////////
    // rigid body model
    //////////////////////////////////////////////////////////////////////////
    clearAccelerations(model);
    PBDSimulationModel::RigidBodyVector &rb = model.getRigidBodies();
    ParticleData &pd = model.getParticles();
    OrientationData &od = model.getOrientations();

    const int numBodies = (int)rb.size();
    const int numParticles = (int) pd.size();
    #pragma omp parallel if(numBodies > MIN_PARALLEL_SIZE) default(shared)
    {
        // #pragma omp for schedule(static) nowait
        for (int i = 0; i < numBodies; i++)
        {
            rb[i]->getLastPosition() = rb[i]->getOldPosition();
            rb[i]->getOldPosition() = rb[i]->getPosition();
            PBDTimeIntegration::semiImplicitEuler(h, rb[i]->getMass(), rb[i]->getPosition(), rb[i]->getVelocity(), rb[i]->getAcceleration());
            rb[i]->getLastRotation() = rb[i]->getOldRotation();
            rb[i]->getOldRotation() = rb[i]->getRotation();
            PBDTimeIntegration::semiImplicitEulerRotation(h, rb[i]->getMass(), rb[i]->getInertiaTensorInverseW(), rb[i]->getRotation(), rb[i]->getAngularVelocity(), rb[i]->getTorque());
            rb[i]->rotationUpdated();
        }

        //////////////////////////////////////////////////////////////////////////
        // particle model
        //////////////////////////////////////////////////////////////////////////
        #pragma omp for schedule(static)
        for (int i = 0; i < (int) pd.size(); i++)
        {
            pd.getLastPosition(i) = pd.getOldPosition(i);
            pd.getOldPosition(i) = pd.getPosition(i);
            PBDTimeIntegration::semiImplicitEuler(h, pd.getMass(i), pd.getPosition(i), pd.getVelocity(i), pd.getAcceleration(i));
        }

        //////////////////////////////////////////////////////////////////////////
        // orientation model
        //////////////////////////////////////////////////////////////////////////
        #pragma omp for schedule(static)
        for (int i = 0; i < (int)od.size(); i++)
        {
            od.getLastQuaternion(i) = od.getOldQuaternion(i);
            od.getOldQuaternion(i) = od.getQuaternion(i);
            PBDTimeIntegration::semiImplicitEulerRotation(h, od.getMass(i), od.getInvMass(i) * Matrix3r::Identity() ,od.getQuaternion(i), od.getVelocity(i), Vector3r(0,0,0));
        }
    }

    // START_TIMING("position constraints projection");
    positionConstraintProjection(model);
    // STOP_TIMING_AVG;

    /// TODO: Move collision detection interface to the PBDAnimationLoop and companion classes
    // Line collision models
    // TODO: Filtering line models and correspondences to particles (start/end points) can be cached/precomputed!
    msg_info("PBDTimeStepController") << "Start collision detection calls: collision detection active = " << (m_collisionDetection != NULL);
    if (m_collisionDetection)
    {
        std::cout << "Update line collision models." << std::endl;
        std::vector<DistanceFieldCollisionDetection::DistanceFieldCollisionLine*> lineCollisionModels;
        std::vector<CollisionDetection::CollisionObject*>& collisionObjects = m_collisionDetection->getCollisionObjects();
        {
            for (size_t k = 0; k < collisionObjects.size(); k++)
            {
                // TODO: Make bodyTypeId match between IdFactory and LineCollisionShape type (34 != 3)
                msg_info("PBDTimeStepController") << "collisonObject[" << k << "] typeId = " << collisionObjects[k]->getTypeId();
                if (collisionObjects[k]->getTypeId() == 34)
                {
                    DistanceFieldCollisionDetection::DistanceFieldCollisionLine* cl = (DistanceFieldCollisionDetection::DistanceFieldCollisionLine*) collisionObjects[k];
                    lineCollisionModels.push_back(cl);
                }
            }
        }

        for (int i = 0; i < numParticles; i++)
        {
            for (size_t m = 0; m < lineCollisionModels.size(); m++)
            {
                if (lineCollisionModels[m]->m_startPointParticleIndex == i)
                {
                    // msg_info("PBDTimeStepController") << "Update start vertex of line " << m;
                    Matrix3r mat(od.getQuaternion(i));
                    lineCollisionModels[m]->updateTransformation(pd.getPosition(i), mat, false);
                }
                if (lineCollisionModels[m]->m_endPointParticleIndex == i)
                {
                    // msg_info("PBDTimeStepController") << "Update end vertex of line " << m;
                    Matrix3r mat(od.getQuaternion(i));
                    lineCollisionModels[m]->updateTransformation(pd.getPosition(i), mat, true);
                }
            }
        }
    }

    #pragma omp parallel if(numBodies > MIN_PARALLEL_SIZE) default(shared)
    {
        // Update velocities
        // #pragma omp for schedule(static) nowait
        for (int i = 0; i < numBodies; i++)
        {
            if (m_velocityUpdateMethod == 0)
            {
                PBDTimeIntegration::velocityUpdateFirstOrder(h, rb[i]->getMass(), rb[i]->getPosition(), rb[i]->getOldPosition(), rb[i]->getVelocity());
                PBDTimeIntegration::angularVelocityUpdateFirstOrder(h, rb[i]->getMass(), rb[i]->getRotation(), rb[i]->getOldRotation(), rb[i]->getAngularVelocity());
            }
            else
            {
                PBDTimeIntegration::velocityUpdateSecondOrder(h, rb[i]->getMass(), rb[i]->getPosition(), rb[i]->getOldPosition(), rb[i]->getLastPosition(), rb[i]->getVelocity());
                PBDTimeIntegration::angularVelocityUpdateSecondOrder(h, rb[i]->getMass(), rb[i]->getRotation(), rb[i]->getOldRotation(), rb[i]->getLastRotation(), rb[i]->getAngularVelocity());
            }
            // update geometry
            rb[i]->getGeometry().updateMeshTransformation(rb[i]->getPosition(), rb[i]->getRotationMatrix());
        }

        // Update velocities
        #pragma omp for schedule(static)
        for (int i = 0; i < (int) pd.size(); i++)
        {
            if (m_velocityUpdateMethod == 0)
                PBDTimeIntegration::velocityUpdateFirstOrder(h, pd.getMass(i), pd.getPosition(i), pd.getOldPosition(i), pd.getVelocity(i));
            else
                PBDTimeIntegration::velocityUpdateSecondOrder(h, pd.getMass(i), pd.getPosition(i), pd.getOldPosition(i), pd.getLastPosition(i), pd.getVelocity(i));
        }

        // Update velocites of orientations
        #pragma omp for schedule(static)
        for (int i = 0; i < (int)od.size(); i++)
        {
            if (m_velocityUpdateMethod == 0)
                PBDTimeIntegration::angularVelocityUpdateFirstOrder(h, od.getMass(i), od.getQuaternion(i), od.getOldQuaternion(i), od.getVelocity(i));
            else
                PBDTimeIntegration::angularVelocityUpdateSecondOrder(h, od.getMass(i), od.getQuaternion(i), od.getOldQuaternion(i), od.getLastQuaternion(i), od.getVelocity(i));
        }
    }

    // TODO: Move collision detection to PBDAnimationLoop and companion classes
    if (m_collisionDetection)
    {
        // START_TIMING("collision detection");
        m_collisionDetection->collisionDetection(model);
        // STOP_TIMING_AVG;
    }

    velocityConstraintProjection(model);

    //////////////////////////////////////////////////////////////////////////
    // update motor joint targets
    //////////////////////////////////////////////////////////////////////////
    PBDSimulationModel::ConstraintVector &constraints = model.getConstraints();
    for (unsigned int i = 0; i < constraints.size(); i++)
    {
        if ((constraints[i]->getTypeId() == TargetAngleMotorHingeJoint::TYPE_ID) ||
            (constraints[i]->getTypeId() == TargetVelocityMotorHingeJoint::TYPE_ID) ||
            (constraints[i]->getTypeId() == TargetPositionMotorSliderJoint::TYPE_ID) ||
            (constraints[i]->getTypeId() == TargetVelocityMotorSliderJoint::TYPE_ID))
        {
            MotorJoint *motor = (MotorJoint*)constraints[i];
            const std::vector<Real> sequence = motor->getTargetSequence();
            if (sequence.size() > 0)
            {
                Real time = tm->getTime();
                const Real sequenceDuration = sequence[sequence.size() - 2] - sequence[0];
                if (motor->getRepeatSequence())
                {
                    while (time > sequenceDuration)
                        time -= sequenceDuration;
                }
                unsigned int index = 0;
                while ((2*index < sequence.size()) && (sequence[2 * index] <= time))
                    index++;

                // linear interpolation
                Real target = 0.0;
                if (2 * index < sequence.size())
                {
                    const Real alpha = (time - sequence[2 * (index - 1)]) / (sequence[2 * index] - sequence[2 * (index - 1)]);
                    target = (static_cast<Real>(1.0) - alpha) * sequence[2 * index - 1] + alpha * sequence[2 * index + 1];
                }
                else
                    target = sequence[sequence.size() - 1];
                motor->setTarget(target);
            }
        }
    }

    // compute new time
    tm->setTime (tm->getTime () + h);
    // STOP_TIMING_AVG;
}

void PBDTimeStepController::reset()
{
    m_iterations = 0;
    m_iterationsV = 0;
    m_maxIterations = 5;
    m_maxIterationsV = 5;
}

void PBDTimeStepController::positionConstraintProjection(PBDSimulationModel &model)
{
    m_iterations = 0;

    // init constraint groups if necessary
    model.initConstraintGroups();

    PBDSimulationModel::RigidBodyVector &rb = model.getRigidBodies();
    PBDSimulationModel::ConstraintVector &constraints = model.getConstraints();
    PBDSimulationModel::ConstraintGroupVector &groups = model.getConstraintGroups();
    PBDSimulationModel::RigidBodyContactConstraintVector &contacts = model.getRigidBodyContactConstraints();
    PBDSimulationModel::ParticleSolidContactConstraintVector &particleTetContacts = model.getParticleSolidContactConstraints();

    // init constraints for this time step if necessary
    for (auto & constraint : constraints)
    {
        constraint->initConstraintBeforeProjection(model);
    }

    while (m_iterations < m_maxIterations)
    {
        for (unsigned int group = 0; group < groups.size(); group++)
        {
            const int groupSize = (int)groups[group].size();
            #pragma omp parallel if(groupSize > MIN_PARALLEL_SIZE) default(shared)
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < groupSize; i++)
                {
                    const unsigned int constraintIndex = groups[group][i];

                    constraints[constraintIndex]->updateConstraint(model);
                    constraints[constraintIndex]->solvePositionConstraint(model, m_iterations);
                }
            }
        }

        for (unsigned int i = 0; i < particleTetContacts.size(); i++)
        {
            particleTetContacts[i].solvePositionConstraint(model, m_iterations);
        }

        m_iterations++;
    }
}


void PBDTimeStepController::velocityConstraintProjection(PBDSimulationModel &model)
{
    m_iterationsV = 0;

    // init constraint groups if necessary
    model.initConstraintGroups();

    PBDSimulationModel::RigidBodyVector &rb = model.getRigidBodies();
    PBDSimulationModel::ConstraintVector &constraints = model.getConstraints();
    PBDSimulationModel::ConstraintGroupVector &groups = model.getConstraintGroups();
    PBDSimulationModel::RigidBodyContactConstraintVector &rigidBodyContacts = model.getRigidBodyContactConstraints();
    PBDSimulationModel::ParticleRigidBodyContactConstraintVector &particleRigidBodyContacts = model.getParticleRigidBodyContactConstraints();
    PBDSimulationModel::ParticleSolidContactConstraintVector &particleTetContacts = model.getParticleSolidContactConstraints();

    for (unsigned int group = 0; group < groups.size(); group++)
    {
        const int groupSize = (int)groups[group].size();
        #pragma omp parallel if(groupSize > MIN_PARALLEL_SIZE) default(shared)
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < groupSize; i++)
            {
                const unsigned int constraintIndex = groups[group][i];
                constraints[constraintIndex]->updateConstraint(model);
            }
        }
    }

    while (m_iterationsV < m_maxIterationsV)
    {
        for (unsigned int group = 0; group < groups.size(); group++)
        {
            const int groupSize = (int)groups[group].size();
            #pragma omp parallel if(groupSize > MIN_PARALLEL_SIZE) default(shared)
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < groupSize; i++)
                {
                    const unsigned int constraintIndex = groups[group][i];
                    constraints[constraintIndex]->solveVelocityConstraint(model, m_iterationsV);
                }
            }
        }

        // solve contacts
        for (unsigned int i = 0; i < rigidBodyContacts.size(); i++)
        {
            rigidBodyContacts[i].solveVelocityConstraint(model, m_iterationsV);
        }
        for (unsigned int i = 0; i < particleRigidBodyContacts.size(); i++)
        {
            particleRigidBodyContacts[i].solveVelocityConstraint(model, m_iterationsV);
        }
        for (unsigned int i = 0; i < particleTetContacts.size(); i++)
        {
            particleTetContacts[i].solveVelocityConstraint(model, m_iterationsV);
        }
        m_iterationsV++;
    }
}



