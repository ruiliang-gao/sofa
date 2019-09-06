#include "SofaPBDTimeStep.h"
#include "PBDUtils/PBDTimeManager.h"
#include "SofaPBDSimulation.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Simulation.h>

#include <sofa/core/visual/VisualParams.h>

#include "PBDynamics/PBDTimeIntegration.h"
#include "DistanceFieldCollisionDetection.h"

#include <PBDIntegration/SofaPBDPointCollisionModel.h>
#include <PBDIntegration/SofaPBDLineCollisionModel.h>
#include <PBDIntegration/SofaPBDTriangleCollisionModel.h>

#include <PBDIntegration/SofaPBDCollisionDetectionOutput.h>

using namespace sofa::defaulttype;
using namespace sofa::simulation::PBDSimulation;

int SofaPBDTimeStepClass = sofa::core::RegisterObject("Wrapper class for the PBD TimeStep.")
                            .add< SofaPBDTimeStep >()
                            .addDescription("Wrapper class for the PBD TimeStep.");
using namespace std;
using namespace sofa::component::collision;

SofaPBDTimeStep::SofaPBDTimeStep(): sofa::core::objectmodel::BaseObject(),
    MAX_ITERATIONS(initData(&MAX_ITERATIONS, 5, "Max. iterations", "Maximal number of iterations of the solver.")),
    MAX_ITERATIONS_V(initData(&MAX_ITERATIONS_V, 5, "Max. velocity iterations", "Maximal number of iterations of the velocity solver."))
{
    m_iterations = 0;
    m_iterationsV = 0;

    m_collisionDetection = NULL;
    m_sofaPBDCollisionDetection = NULL;
}

SofaPBDTimeStep::~SofaPBDTimeStep()
{

}

void SofaPBDTimeStep::init()
{
    msg_info("SofaPBDTimeStep") << "init()";
    initParameters();

    msg_info("SofaPBDTimeStep") << "MAX_ITERATIONS = " << MAX_ITERATIONS.getValue();
    msg_info("SofaPBDTimeStep") << "MAX_ITERATIONS_V = " << MAX_ITERATIONS_V.getValue();
}

void SofaPBDTimeStep::bwdInit()
{
    BaseContext* bc = sofa::simulation::getSimulation()->getCurrentRootNode().get();

    std::vector<SofaPBDBruteForceDetection*> bfdInstances = bc->getObjects<SofaPBDBruteForceDetection>(BaseContext::SearchDown);

    if (bfdInstances.size() > 0)
    {
        msg_info("SofaPBDTimeStep") << "SofaPBDBruteForceDetection instances found: " << bfdInstances.size();
        // Store reference to the first SofaPBDBruteForceDetection found
        m_sofaPBDCollisionDetection = bfdInstances.at(0);
    }
    else
    {
        msg_warning("SofaPBDTimeStep") << "No SofaPBDBruteForceDetection instance found, collision response will be non-functional!";
    }
}

void SofaPBDTimeStep::initParameters()
{
    MAX_ITERATIONS.setGroup("PBD");
    MAX_ITERATIONS_V.setGroup("PBD");

    helper::OptionsGroup methodOptions(2, "0 - First Order Update",
                                       "1 - Second Order Update");

    initData(&VELOCITY_UPDATE_METHOD, "Velocity update method", "Velocity update method.");
    VELOCITY_UPDATE_METHOD.setGroup("PBD");
    methodOptions.setSelectedItem(0);
    VELOCITY_UPDATE_METHOD.setValue(methodOptions);

    helper::OptionsGroup collisionDetectionOptions(2, "0 - use SOFA collision pipeline",
                                                   "1 - use PBD SignedDistanceFieldCollisionDetection");

    initData(&COLLISION_DETECTION_METHOD, "Collision detection method", "Collision Detection method.");
    COLLISION_DETECTION_METHOD.setGroup("PBD");
    collisionDetectionOptions.setSelectedItem(0);
}

void SofaPBDTimeStep::reset()
{
    m_iterations = 0;
    m_iterationsV = 0;
}

void SofaPBDTimeStep::cleanup()
{
    if (m_collisionDetection)
    {
        m_collisionDetection->cleanup();
        delete m_collisionDetection;
        m_collisionDetection = NULL;
    }

    /*if (m_sofaPBDCollisionDetection)
    {
        m_sofaPBDCollisionDetection->cleanup();
    }*/
}

void SofaPBDTimeStep::clearAccelerations(PBDSimulationModel &model)
{
    //////////////////////////////////////////////////////////////////////////
    // rigid body model
    //////////////////////////////////////////////////////////////////////////

    PBDSimulationModel::RigidBodyVector &rb = model.getRigidBodies();
    SofaPBDSimulation *sim = SofaPBDSimulation::getCurrent();
    sofa::defaulttype::Vec3d gravitation = sim->GRAVITATION.getValue();
    Vector3r grav(gravitation.x(), gravitation.y(), gravitation.z());
    for (size_t i = 0; i < rb.size(); i++)
    {
        // Clear accelerations of dynamic particles
        if (rb[i]->getMass() != 0.0)
        {
            msg_info("SofaPBDTimeStep") << "Clearing acceleration of rigid body " << i << ", setting a to: (" << grav[0] << "," << grav[1] << "," << grav[2] << ")";
            Vector3r &a = rb[i]->getAcceleration();
            a = grav;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // particle model
    //////////////////////////////////////////////////////////////////////////

    PBDParticleData &pd = model.getParticles();
    const unsigned int count = pd.size();
    for (unsigned int i = 0; i < count; i++)
    {
        // Clear accelerations of dynamic particles
        if (pd.getMass(i) != 0.0)
        {
            msg_info("SofaPBDTimeStep") << "Clearing acceleration of particle " << i << ", setting a to: (" << grav[0] << "," << grav[1] << "," << grav[2] << ")";
            Vector3r &a = pd.getAcceleration(i);
            a = grav;
        }
    }
}

/// TODO: Interface collision detection via PBDAnimationLoop
void SofaPBDTimeStep::setCollisionDetection(PBDSimulationModel &model, CollisionDetection *cd)
{
    m_collisionDetection = cd;
    /*m_collisionDetection->setContactCallback(contactCallbackFunction, &model);
    m_collisionDetection->setSolidContactCallback(solidContactCallbackFunction, &model);*/
}

CollisionDetection *SofaPBDTimeStep::getCollisionDetection()
{
    return m_collisionDetection;
}

void SofaPBDTimeStep::step(PBDSimulationModel &model)
{
    msg_info("SofaPBDTimeStep") << "SofaPBDTimeStep::step()";
    // START_TIMING("simulation step");
    PBDTimeManager *tm = PBDTimeManager::getCurrent();
    const Real h = tm->getTimeStepSize();

    msg_info("SofaPBDTimeStep") << "Time step: " << h;

    //////////////////////////////////////////////////////////////////////////
    // rigid body models
    //////////////////////////////////////////////////////////////////////////
    msg_info("SofaPBDTimeStep") << "Clearing accelerations.";
    clearAccelerations(model);
    PBDSimulationModel::RigidBodyVector &rb = model.getRigidBodies();
    PBDParticleData &pd = model.getParticles();
    PBDOrientationData &od = model.getOrientations();

    msg_info("SofaPBDTimeStep") << "Retrieved data arrays from model: rigidBodies = " << rb.size() << ", particles = " << pd.size() << ", particle orientations = " << od.size();

    const int numBodies = (int) rb.size();
    const int numParticles = (int) pd.size();
    // #pragma omp parallel if(numBodies > MIN_PARALLEL_SIZE) default(shared)
    {
        // #pragma omp for schedule(static) nowait
        msg_info("SofaPBDTimeStep") << "Updating rigid body free-motion.";
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
        // #pragma omp for schedule(static)
        msg_info("SofaPBDTimeStep") << "Updating particles free-motion: " << pd.size() << " particles.";
        for (int i = 0; i < (int) pd.size(); i++)
        {
            pd.getLastPosition(i) = pd.getOldPosition(i);
            pd.getOldPosition(i) = pd.getPosition(i);
            if (pd.getMass(i) == 0.0)
            {
                msg_info("SofaPBDTimeStep") << "Not updating particle " << i << ", mass is 0.0";
            }
            else
            {
                PBDTimeIntegration::semiImplicitEuler(h, pd.getMass(i), pd.getPosition(i), pd.getVelocity(i), pd.getAcceleration(i));
            }
        }

        //////////////////////////////////////////////////////////////////////////
        // orientation model
        //////////////////////////////////////////////////////////////////////////
        // #pragma omp for schedule(static)
        msg_info("SofaPBDTimeStep") << "Updating particle orientations free-motion.";
        for (int i = 0; i < (int)od.size(); i++)
        {
            od.getLastQuaternion(i) = od.getOldQuaternion(i);
            od.getOldQuaternion(i) = od.getQuaternion(i);
            PBDTimeIntegration::semiImplicitEulerRotation(h, od.getMass(i), od.getInvMass(i) * Matrix3r::Identity() ,od.getQuaternion(i), od.getVelocity(i), Vector3r(0,0,0));
        }
    }

    // START_TIMING("position constraints projection");
    msg_info("SofaPBDTimeStep") << "Calling positionConstraintProjection()";
    positionConstraintProjection(model);
    // STOP_TIMING_AVG;

    /// TODO: Move collision detection interface to the PBDAnimationLoop and companion classes
    // Line collision models
    // TODO: Filtering line models and correspondences to particles (start/end points) can be cached/precomputed!
    msg_info("SofaPBDTimeStep") << "Update line collision shapes: collision detection active = " << (m_collisionDetection != NULL);
    if (m_collisionDetection)
    {
        std::cout << "Update line collision models." << std::endl;
        std::vector<DistanceFieldCollisionDetection::DistanceFieldCollisionLine*> lineCollisionModels;
        std::vector<CollisionDetection::CollisionObject*>& collisionObjects = m_collisionDetection->getCollisionObjects();
        {
            for (size_t k = 0; k < collisionObjects.size(); k++)
            {
                // TODO: Make bodyTypeId match between IdFactory and LineCollisionShape type (34 != 3)
                msg_info("SofaPBDTimeStep") << "collisonObject[" << k << "] typeId = " << collisionObjects[k]->getTypeId();
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
                    // msg_info("SofaPBDTimeStep") << "Update start vertex of line " << m;
                    Matrix3r mat(od.getQuaternion(i));
                    lineCollisionModels[m]->updateTransformation(pd.getPosition(i), mat, false);
                }
                if (lineCollisionModels[m]->m_endPointParticleIndex == i)
                {
                    // msg_info("SofaPBDTimeStep") << "Update end vertex of line " << m;
                    Matrix3r mat(od.getQuaternion(i));
                    lineCollisionModels[m]->updateTransformation(pd.getPosition(i), mat, true);
                }
            }
        }
    }

    // #pragma omp parallel if(numBodies > MIN_PARALLEL_SIZE) default(shared)
    {
        // Update velocities
        // #pragma omp for schedule(static) nowait

        msg_info("SofaPBDTimeStep") << "Updating velocities - rigid bodies";
        for (int i = 0; i < numBodies; i++)
        {
            if (VELOCITY_UPDATE_METHOD.getValue().getSelectedId() == 0)
            {
                if (rb[i]->getMass() != 0.0)
                {
                    msg_info("SofaPBDTimeStep") << "Rigid body " << i << " -- first-order velocity update.";
                    PBDTimeIntegration::velocityUpdateFirstOrder(h, rb[i]->getMass(), rb[i]->getPosition(), rb[i]->getOldPosition(), rb[i]->getVelocity());
                    PBDTimeIntegration::angularVelocityUpdateFirstOrder(h, rb[i]->getMass(), rb[i]->getRotation(), rb[i]->getOldRotation(), rb[i]->getAngularVelocity());
                }
                else
                {
                    msg_info("SofaPBDTimeStep") << "Rigid body " << i << ": Static body, no velocity update (first order).";
                }
            }
            else
            {
                if (rb[i]->getMass() != 0.0)
                {
                    msg_info("SofaPBDTimeStep") << "Rigid body " << i << " -- second-order velocity update.";
                    PBDTimeIntegration::velocityUpdateSecondOrder(h, rb[i]->getMass(), rb[i]->getPosition(), rb[i]->getOldPosition(), rb[i]->getLastPosition(), rb[i]->getVelocity());
                    PBDTimeIntegration::angularVelocityUpdateSecondOrder(h, rb[i]->getMass(), rb[i]->getRotation(), rb[i]->getOldRotation(), rb[i]->getLastRotation(), rb[i]->getAngularVelocity());
                }
                else
                {
                    msg_info("SofaPBDTimeStep") << "Rigid body " << i << ": Static body, no velocity update (second order).";
                }
            }

            // Update transform of rigid body's geometry; only do so for moving bodies (mass != 0.0)
            if (rb[i]->getMass() != 0.0)
            {
                msg_info("SofaPBDTimeStep") << "Rigid body " << i << ": Updating mesh transformation.";
                rb[i]->getGeometry().updateMeshTransformation(rb[i]->getPosition(), rb[i]->getRotationMatrix());
            }
        }

        // Update velocities
        // #pragma omp for schedule(static)

        msg_info("SofaPBDTimeStep") << "Updating velocities - particles: " << pd.size();
        for (int i = 0; i < (int) pd.size(); i++)
        {
            if (VELOCITY_UPDATE_METHOD.getValue().getSelectedId() == 0)
                PBDTimeIntegration::velocityUpdateFirstOrder(h, pd.getMass(i), pd.getPosition(i), pd.getOldPosition(i), pd.getVelocity(i));
            else
                PBDTimeIntegration::velocityUpdateSecondOrder(h, pd.getMass(i), pd.getPosition(i), pd.getOldPosition(i), pd.getLastPosition(i), pd.getVelocity(i));
        }

        // Update velocites of orientations
        // #pragma omp for schedule(static)

        msg_info("SofaPBDTimeStep") << "Updating velocities - particle orientations";
        for (int i = 0; i < (int)od.size(); i++)
        {
            if (VELOCITY_UPDATE_METHOD.getValue().getSelectedId() == 0)
                PBDTimeIntegration::angularVelocityUpdateFirstOrder(h, od.getMass(i), od.getQuaternion(i), od.getOldQuaternion(i), od.getVelocity(i));
            else
                PBDTimeIntegration::angularVelocityUpdateSecondOrder(h, od.getMass(i), od.getQuaternion(i), od.getOldQuaternion(i), od.getLastQuaternion(i), od.getVelocity(i));
        }
    }

    // TODO: Move collision detection to PBDAnimationLoop and companion classes
    msg_info("SofaPBDTimeStep") << "Running collision queries: collision detection method = " << COLLISION_DETECTION_METHOD.getValue().getSelectedId();

    if (COLLISION_DETECTION_METHOD.getValue().getSelectedId() == 0)
    {
        msg_info("SofaPBDTimeStep") << "Using SOFA-integrated collision detection.";
        if (m_sofaPBDCollisionDetection)
        {
            PBDSimulationModel* simModel = SofaPBDSimulation::getCurrent()->getModel();

            simModel->setContactsCleared(true);
            simModel->resetContacts();

            const PBDSimulationModel::RigidBodyVector& rigidBodies = simModel->getRigidBodies();
            simModel->getParticles();
            const PBDSimulationModel::LineModelVector& lineModels = simModel->getLineModels();

            const std::map<std::pair<core::CollisionModel*, core::CollisionModel*>, sofa::helper::vector<sofa::core::collision::SofaPBDCollisionDetectionOutput>>& collisionOutputs = m_sofaPBDCollisionDetection->getCollisionOutputs();

            // Forward rigid vs. rigid contacts to the PBD solver
            for (std::map<std::pair<core::CollisionModel*, core::CollisionModel*>, sofa::helper::vector<sofa::core::collision::SofaPBDCollisionDetectionOutput>>::const_iterator cm_it = collisionOutputs.begin(); cm_it != collisionOutputs.end(); cm_it++)
            {
                msg_info("SofaPBDTimeStep") << cm_it->first.first->getName() << " -- " << cm_it->first.second->getName() << ": " << cm_it->second.size() << " contacts.";

                for (size_t r = 0; r < cm_it->second.size(); r++)
                {
                    // Distinguish between rigid vs. rigid and rigid vs. particle type contacts here - this needs to be added in SofaPBD BruteForceDetection!
                    msg_info("SofaPBDTimeStep") << "Adding contact: ID = " << cm_it->second[r].id
                                                      << ", contactType = " << cm_it->second[r].contactType
                                                      << ", modelPairType = " << cm_it->second[r].modelPairType
                                                      << ", rigidBodyIndices = " << cm_it->second[r].rigidBodyIndices[0] << " -- " << cm_it->second[r].rigidBodyIndices[1]
                                                      << ", point0 = " << cm_it->second[r].point[0]
                                                      << ", point1 = " << cm_it->second[r].point[1]
                                                      << ", normal = " << cm_it->second[r].normal
                                                      << ", features = " << cm_it->second[r].elem.first.getIndex() << " -- " << cm_it->second[r].elem.second.getIndex()
                                                      << ", distance = " << cm_it->second[r].value;


                    if (cm_it->second[r].contactType == sofa::core::collision::PBD_RIGID_RIGID_CONTACT)
                    {
                        msg_info("SofaPBDTimeStep") << "Adding new contact constraint: PBD_RIGID_RIGID_CONTACT";
                        try
                        {
                            Vector3r contactPoint1(cm_it->second[r].point[0].x(), cm_it->second[r].point[0].y(), cm_it->second[r].point[0].z());
                            Vector3r contactPoint2(cm_it->second[r].point[1].x(), cm_it->second[r].point[1].y(), cm_it->second[r].point[1].z());
                            Vector3r normal(cm_it->second[r].normal.x(), cm_it->second[r].normal.y(), cm_it->second[r].normal.z());

                            Real restitutionCoeff = rigidBodies.at(cm_it->second[r].rigidBodyIndices[0])->getRestitutionCoeff() * rigidBodies.at(cm_it->second[r].rigidBodyIndices[1])->getRestitutionCoeff();
                            Real frictionCoeff = rigidBodies.at(cm_it->second[r].rigidBodyIndices[0])->getFrictionCoeff() + rigidBodies.at(cm_it->second[r].rigidBodyIndices[1])->getFrictionCoeff();

                            simModel->addRigidBodyContactConstraint((unsigned int) cm_it->second[r].rigidBodyIndices[0], (unsigned int) cm_it->second[r].rigidBodyIndices[1],
                                    contactPoint1, contactPoint2, normal, Real(cm_it->second[r].value), restitutionCoeff, frictionCoeff);
                        }
                        catch (std::out_of_range& ex)
                        {
                            msg_error("SofaPBDTimeStep") << "Tried accessing rigid body data outside valid index range (RIGID_RIGID_CONTACT): " << ex.what();
                        }

                    }
                    else if (cm_it->second[r].contactType == sofa::core::collision::PBD_RIGID_LINE_CONTACT)
                    {
                        msg_info("SofaPBDTimeStep") << "Adding new contact constraint: PBD_RIGID_LINE_CONTACT";

                        try
                        {
                            Vector3r contactPoint1(cm_it->second[r].point[0].x(), cm_it->second[r].point[0].y(), cm_it->second[r].point[0].z());
                            Vector3r contactPoint2(cm_it->second[r].point[1].x(), cm_it->second[r].point[1].y(), cm_it->second[r].point[1].z());
                            Vector3r normal(cm_it->second[r].normal.x(), cm_it->second[r].normal.y(), cm_it->second[r].normal.z());

                            Real restitutionCoeff = rigidBodies.at(cm_it->second[r].rigidBodyIndices[0])->getRestitutionCoeff() * lineModels.at(cm_it->second[r].rigidBodyIndices[1])->getRestitutionCoeff();
                            Real frictionCoeff = rigidBodies.at(cm_it->second[r].rigidBodyIndices[0])->getFrictionCoeff() * lineModels.at(cm_it->second[r].rigidBodyIndices[1])->getFrictionCoeff();


                            simModel->addParticleRigidBodyContactConstraint(cm_it->second[r].particleIndices[0], cm_it->second[r].rigidBodyIndices[1],
                                    contactPoint1, contactPoint2, normal, Real(cm_it->second[r].value), restitutionCoeff, frictionCoeff);
                        }
                        catch (std::out_of_range& ex)
                        {
                            msg_error("SofaPBDTimeStep") << "Tried accessing rigid body data outside valid index range (RIGID_LINE_CONTACT): " << ex.what();
                        }
                    }
                    else if (cm_it->second[r].contactType == sofa::core::collision::PBD_PARTICLE_RIGID_CONTACT)
                    {
                        msg_warning("SofaPBDTimeStep") << "Not implemented yet: PBD_PARTICLE_RIGID_CONTACT";
                    }
                    else if (cm_it->second[r].contactType == sofa::core::collision::PBD_PARTICLE_SOLID_CONTACT)
                    {
                        msg_warning("SofaPBDTimeStep") << "Not implemented yet: PBD_PARTICLE_SOLID_CONTACT";
                    }
                }
            }
        }
    }
    else
    {
        msg_info("SofaPBDTimeStep") << "Using PBD-integrated collision detection.";
        if (m_collisionDetection)
        {
            // START_TIMING("collision detection");
            m_collisionDetection->collisionDetection(model);
            // STOP_TIMING_AVG;
        }
    }

    msg_info("SofaPBDTimeStep") << "Calling velocityConstraintProjection()";
    velocityConstraintProjection(model);

    //////////////////////////////////////////////////////////////////////////
    // update motor joint targets
    //////////////////////////////////////////////////////////////////////////
    PBDSimulationModel::ConstraintVector &constraints = model.getConstraints();

    msg_info("SofaPBDTimeStep") << "Updating motor joint targets: Constraint count = " << constraints.size();
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
    msg_info("SofaPBDTimeStep") << "Setting new simulation time: " << (tm->getTime() + h);
    tm->setTime(tm->getTime() + h);

    msg_info("SofaPBDTimeStep") << "Particles: " << pd.size();
    for (int i = 0; i < (int) pd.size(); i++)
    {
        msg_info("SofaPBDTimeStep") << "Particle " << i << ": position = (" << pd.getPosition(i)[0] << "," << pd.getPosition(i)[1] << "," << pd.getPosition(i)[2] << "), velocity = (" << pd.getVelocity(i)[0] << "," << pd.getVelocity(i)[1] << "," << pd.getVelocity(i)[2] << ")";
    }

    // STOP_TIMING_AVG;
}

void SofaPBDTimeStep::positionConstraintProjection(PBDSimulationModel &model)
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

    while (m_iterations < MAX_ITERATIONS.getValue())
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


void SofaPBDTimeStep::velocityConstraintProjection(PBDSimulationModel &model)
{
    msg_info("SofaPBDTimeStep") << "velocityConstraintProjection() - MAX_ITERATIONS_V = " << MAX_ITERATIONS_V.getValue();
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

    while (m_iterationsV < MAX_ITERATIONS_V.getValue())
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
        msg_info("SofaPBDTimeStep") << "Solving rigid body - rigid body constraints: " << rigidBodyContacts.size();
        for (unsigned int i = 0; i < rigidBodyContacts.size(); i++)
        {
            msg_info("SofaPBDTimeStep") << "Solve constraint " << i;
            rigidBodyContacts[i].solveVelocityConstraint(model, m_iterationsV);
        }
        msg_info("SofaPBDTimeStep") << "Solving rigid body - particle constraints: " << particleRigidBodyContacts.size();
        for (unsigned int i = 0; i < particleRigidBodyContacts.size(); i++)
        {
            particleRigidBodyContacts[i].solveVelocityConstraint(model, m_iterationsV);
        }
        msg_info("SofaPBDTimeStep") << "Solving tetrahedron - particle constraints: " << particleTetContacts.size();
        for (unsigned int i = 0; i < particleTetContacts.size(); i++)
        {
            particleTetContacts[i].solveVelocityConstraint(model, m_iterationsV);
        }
        m_iterationsV++;
    }
}

void SofaPBDTimeStep::draw(const core::visual::VisualParams* vparams)
{
    /*if (!vparams->displayFlags().getShowBehaviorModels())
        return;*/

    PBDSimulationModel* model = SofaPBDSimulation::getCurrent()->getModel();
    const PBDSimulationModel::RigidBodyContactConstraintVector& rbConstraints = model->getRigidBodyContactConstraints();

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->enableLighting();

    msg_info("SofaPBDTimeStep") << "PBD rigid body constraints count = " << rbConstraints.size();

    if (rbConstraints.size() > 0)
    {
        Vec4f normalColor(0.5f, 0.9f, 0.2f, 0.5f);
        Vec4f tangentColor(0.2f, 0.5f, 0.9f, 0.5f);
        Vec4f linVelColor(0.9f, 0.5f, 0.2f, 0.75f);

        for (size_t k = 0; k < rbConstraints.size(); k++)
        {
            const RigidBodyContactConstraint& rbc = rbConstraints[k];
            const Vector3r &cp1 = rbc.m_constraintInfo.col(0);
            const Vector3r &cp2 = rbc.m_constraintInfo.col(1);
            const Vector3r &normal = rbc.m_constraintInfo.col(2);
            const Vector3r &tangent = rbc.m_constraintInfo.col(3);

            Vector3 contactPoint1(cp1[0], cp1[1], cp1[2]);
            Vector3 contactPoint2(cp2[0], cp2[1], cp2[2]);
            Vector3 normalVector(normal[0], normal[1], normal[2]);
            Vector3 tangentVector(tangent[0], tangent[1], tangent[2]);

            Vector3 rbVelCorrLin1(rbc.m_corrLin_rb1[0], rbc.m_corrLin_rb1[1], rbc.m_corrLin_rb1[2]);
            Vector3 rbVelCorrLin2(rbc.m_corrLin_rb2[0], rbc.m_corrLin_rb2[1], rbc.m_corrLin_rb2[2]);

            vparams->drawTool()->drawSphere(contactPoint1, 0.02f);
            vparams->drawTool()->drawSphere(contactPoint2, 0.02f);

            vparams->drawTool()->drawArrow(contactPoint1, contactPoint1 + normalVector, 0.0025f, normalColor, 8);
            vparams->drawTool()->drawArrow(contactPoint1, contactPoint1 + tangentVector, 0.0025f, tangentColor, 8);

            vparams->drawTool()->drawArrow(contactPoint1, contactPoint1 + rbVelCorrLin1, 0.005f, linVelColor, 8);
            vparams->drawTool()->drawArrow(contactPoint2, contactPoint2 + rbVelCorrLin2, 0.005f, linVelColor, 8);

            vparams->drawTool();
        }
    }

    vparams->drawTool()->disableLighting();
    vparams->drawTool()->restoreLastState();
}

/// TODO: Check for re-entrancy!
/*void SofaPBDTimeStep::contactCallbackFunction(const unsigned int contactType, const unsigned int bodyIndex1, const unsigned int bodyIndex2,
    const Vector3r &cp1, const Vector3r &cp2,
    const Vector3r &normal, const Real dist,
    const Real restitutionCoeff, const Real frictionCoeff, void *userData)
{
    PBDSimulationModel *model = (PBDSimulationModel*)userData;
    if (contactType == CollisionDetection::RigidBodyContactType)
        model->addRigidBodyContactConstraint(bodyIndex1, bodyIndex2, cp1, cp2, normal, dist, restitutionCoeff, frictionCoeff);
    else if (contactType == CollisionDetection::ParticleRigidBodyContactType)
        model->addParticleRigidBodyContactConstraint(bodyIndex1, bodyIndex2, cp1, cp2, normal, dist, restitutionCoeff, frictionCoeff);
}*/

/// TODO: Check for re-entrancy!
/*void SofaPBDTimeStep::solidContactCallbackFunction(const unsigned int contactType, const unsigned int bodyIndex1, const unsigned int bodyIndex2,
    const unsigned int tetIndex, const Vector3r &bary,
    const Vector3r &cp1, const Vector3r &cp2,
    const Vector3r &normal, const Real dist,
    const Real restitutionCoeff, const Real frictionCoeff, void *userData)
{
    PBDSimulationModel *model = (PBDSimulationModel*)userData;
    if (contactType == CollisionDetection::ParticleSolidContactType)
        model->addParticleSolidContactConstraint(bodyIndex1, bodyIndex2, tetIndex, bary, cp1, cp2, normal, dist, restitutionCoeff, frictionCoeff);
}*/
