#include "SofaPBDTimeStep.h"
#include "TimeManager.h"
#include "SofaPBDSimulation.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Simulation.h>

#include <sofa/core/visual/VisualParams.h>

#include "TimeIntegration.h"
#include "DistanceFieldCollisionDetection.h"

#include "SofaPBDSimulation.h"

#include <PBDIntegration/SofaPBDPointCollisionModel.h>
#include <PBDIntegration/SofaPBDLineCollisionModel.h>
#include <PBDIntegration/SofaPBDTriangleCollisionModel.h>

#include <SofaBaseCollision/DefaultPipeline.h>

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/visual/VisualParams.h>

#include <PBDIntegration/SofaPBDCollisionVisitor.h>
#include <PBDIntegration/SofaPBDCollisionDetectionOutput.h>
#include <PBDIntegration/SofaPBDPipeline.h>

using namespace sofa::defaulttype;
using namespace sofa::simulation::PBDSimulation;

int SofaPBDTimeStepClass = sofa::core::RegisterObject("Wrapper class for the PBD TimeStep.")
                            .add< SofaPBDTimeStep >()
                            .addDescription("Wrapper class for the PBD TimeStep.");
using namespace std;
using namespace sofa::component::collision;

SofaPBDTimeStep::SofaPBDTimeStep(SofaPBDSimulation* pbdSimulation): sofa::core::objectmodel::BaseObject(),
    MAX_ITERATIONS(initData(&MAX_ITERATIONS, 5, "Max. iterations", "Maximal number of iterations of the solver.")),
    MAX_ITERATIONS_V(initData(&MAX_ITERATIONS_V, 5, "Max. velocity iterations", "Maximal number of iterations of the velocity solver.")),
    m_simulation(pbdSimulation)
{
    m_iterations = 0;
    m_iterationsV = 0;

    m_collisionDetection = NULL;
    m_sofaPBDCollisionDetection = NULL;
}

SofaPBDTimeStep::~SofaPBDTimeStep()
{

}

double SofaPBDTimeStep::getTime()
{
    return static_cast<double>(TimeManager::getCurrent()->getTime());
}

void SofaPBDTimeStep::init()
{
    msg_info("SofaPBDTimeStep") << "init()";
    initParameters();

    msg_info("SofaPBDTimeStep") << "MAX_ITERATIONS = " << MAX_ITERATIONS.getValue();
    msg_info("SofaPBDTimeStep") << "MAX_ITERATIONS_V = " << MAX_ITERATIONS_V.getValue();

    if (!gnode)
    {
        gnode = dynamic_cast<sofa::simulation::Node*>(this->getContext());

        if (!gnode)
        {
            gnode = sofa::simulation::getSimulation()->getCurrentRootNode().get();
        }
    }

    sofa::simulation::Node::SPtr currentRootNode = sofa::simulation::getSimulation()->getCurrentRootNode();
    if (currentRootNode && currentRootNode->collisionPipeline)
    {
        m_collisionPipeline = currentRootNode->collisionPipeline.get();
        msg_info("SofaPBDAnimationLoop") << "currentRootNode has a valid collisionPipeline instance.";
    }
    else
    {
        if (!currentRootNode)
        {
            msg_error("SofaPBDAnimationLoop") << "currentRootNode is a NULL pointer!";
        }
        else
        {
            msg_warning("SofaPBDAnimationLoop") << "currentRootNode has no valid collisionPipeline instance set!";

            BaseObjectDescription desc("DefaultCollisionPipeline", "DefaultPipeline");
            BaseObject::SPtr obj = sofa::core::ObjectFactory::getInstance()->createObject(currentRootNode.get(), &desc);
            if (obj)
            {
                m_collisionPipelineLocal.reset(dynamic_cast<sofa::core::collision::Pipeline*>(obj.get()));
                msg_info("SofaPBDAnimationLoop") << "Instantiated Pipeline object: " << m_collisionPipelineLocal->getName() << " of type " << m_collisionPipelineLocal->getTypeName();
            }
            else
            {
                msg_error("SofaPBDAnimationLoop") << "Failed to instantiate Pipeline object. Collision detection will not be functional!";
            }
        }
    }
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

    if (m_sofaPBDCollisionDetection)
    {
        m_sofaPBDCollisionDetection->reset();
    }
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

void SofaPBDTimeStep::clearAccelerations(SimulationModel &model)
{
    //////////////////////////////////////////////////////////////////////////
    // rigid body model
    //////////////////////////////////////////////////////////////////////////

    SimulationModel::RigidBodyVector &rb = model.getRigidBodies();
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

    ParticleData &pd = model.getParticles();
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
void SofaPBDTimeStep::setCollisionDetection(SimulationModel &model, CollisionDetection *cd)
{
    m_collisionDetection = cd;
    /*m_collisionDetection->setContactCallback(contactCallbackFunction, &model);
    m_collisionDetection->setSolidContactCallback(solidContactCallbackFunction, &model);*/
}

CollisionDetection *SofaPBDTimeStep::getCollisionDetection()
{
    return m_collisionDetection;
}

void SofaPBDTimeStep::preStep()
{
    SimulationModel* model = m_simulation->getModel();

    // START_TIMING("simulation step");
    TimeManager *tm = TimeManager::getCurrent();
    const Real h = tm->getTimeStepSize();

    msg_info("SofaPBDTimeStep") << "Time step: " << h;

    SimulationModel::RigidBodyVector &rb = model->getRigidBodies();
    ParticleData &pd = model->getParticles();
    OrientationData &od = model->getOrientations();

    msg_info("SofaPBDTimeStep") << "================== PBD preStep (before acc. reset) ==================";
    msg_info("SofaPBDTimeStep") << "Retrieved data arrays from model: rigidBodies = " << rb.size() << ", particles = " << pd.size() << ", particle orientations = " << od.size();

    const int numBodies = (int) rb.size();
    const int numParticles = (int) pd.size();

    for (size_t i = 0; i < numBodies; i++)
    {
        // msg_info("SofaPBDTimeStep") << "Rigid body " << k << ": position = (" << rb[k]->getPosition()[0] << ", " << rb[k]->getPosition()[1] << ", " << rb[k]->getPosition()[2] << "), orientation = (" << rb[k]->getRotation().x() << ", " << rb[k]->getRotation().y() << ", " << rb[k]->getRotation().z() << ", " << rb[k]->getRotation().w() << ")";
        msg_info("SofaPBDTimeStep") << "Rigid body " << i << " current position = " << rb[i]->getPosition()[0] << ", " << rb[i]->getPosition()[1] << ", " << rb[i]->getPosition()[2] << ")";
        msg_info("SofaPBDTimeStep") << "Rigid body " << i << " current velocity = " << rb[i]->getVelocity()[0] << ", " << rb[i]->getVelocity()[1] << ", " << rb[i]->getVelocity()[2] << ")";
        msg_info("SofaPBDTimeStep") << "Rigid body " << i << " current acceler. = " << rb[i]->getAcceleration()[0] << ", " << rb[i]->getAcceleration()[1] << ", " << rb[i]->getAcceleration()[2] << ")";
    }
    msg_info("SofaPBDTimeStep") << "================== PBD preStep (before acc. reset) ==================";

    //////////////////////////////////////////////////////////////////////////
    // rigid body models
    //////////////////////////////////////////////////////////////////////////
    msg_info("SofaPBDTimeStep") << "Clearing accelerations.";
    clearAccelerations(*model);

    #pragma omp parallel if(numBodies > MIN_PARALLEL_SIZE) default(shared)
    {
        msg_info("SofaPBDTimeStep") << "Updating rigid body free-motion.";

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < numBodies; i++)
        {
            msg_info("SofaPBDTimeStep") << "Rigid body " << i << " old position = " << rb[i]->getOldPosition()[0] << ", " << rb[i]->getOldPosition()[1] << ", " << rb[i]->getOldPosition()[2] << ")";
            rb[i]->getLastPosition() = rb[i]->getOldPosition();

            msg_info("SofaPBDTimeStep") << "Rigid body " << i << " current position = " << rb[i]->getPosition()[0] << ", " << rb[i]->getPosition()[1] << ", " << rb[i]->getPosition()[2] << ")";
            msg_info("SofaPBDTimeStep") << "Rigid body " << i << " current velocity = " << rb[i]->getVelocity()[0] << ", " << rb[i]->getVelocity()[1] << ", " << rb[i]->getVelocity()[2] << ")";
            msg_info("SofaPBDTimeStep") << "Rigid body " << i << " current acceler. = " << rb[i]->getAcceleration()[0] << ", " << rb[i]->getAcceleration()[1] << ", " << rb[i]->getAcceleration()[2] << ")";

            rb[i]->getOldPosition() = rb[i]->getPosition();
            TimeIntegration::semiImplicitEuler(h, rb[i]->getMass(), rb[i]->getPosition(), rb[i]->getVelocity(), rb[i]->getAcceleration());

            msg_info("SofaPBDTimeStep") << "Rigid body " << i << " position after semiImplicitEuler = " << rb[i]->getPosition()[0] << ", " << rb[i]->getPosition()[1] << ", " << rb[i]->getPosition()[2] << ")";
            msg_info("SofaPBDTimeStep") << "Rigid body " << i << " velocity after semiImplicitEuler = " << rb[i]->getVelocity()[0] << ", " << rb[i]->getVelocity()[1] << ", " << rb[i]->getVelocity()[2] << ")";
            msg_info("SofaPBDTimeStep") << "Rigid body " << i << " acceler. after semiImplicitEuler = " << rb[i]->getAcceleration()[0] << ", " << rb[i]->getAcceleration()[1] << ", " << rb[i]->getAcceleration()[2] << ")";

            rb[i]->getLastRotation() = rb[i]->getOldRotation();
            rb[i]->getOldRotation() = rb[i]->getRotation();
            TimeIntegration::semiImplicitEulerRotation(h, rb[i]->getMass(), rb[i]->getInertiaTensorInverseW(), rb[i]->getRotation(), rb[i]->getAngularVelocity(), rb[i]->getTorque());
            rb[i]->rotationUpdated();
        }

        //////////////////////////////////////////////////////////////////////////
        // particle model
        //////////////////////////////////////////////////////////////////////////
        msg_info("SofaPBDTimeStep") << "Updating particles free-motion: " << pd.size() << " particles.";

        #pragma omp for schedule(static)
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
                TimeIntegration::semiImplicitEuler(h, pd.getMass(i), pd.getPosition(i), pd.getVelocity(i), pd.getAcceleration(i));
            }
        }

        //////////////////////////////////////////////////////////////////////////
        // orientation model
        //////////////////////////////////////////////////////////////////////////
        msg_info("SofaPBDTimeStep") << "Updating particle orientations free-motion.";

        #pragma omp for schedule(static)
        for (int i = 0; i < (int)od.size(); i++)
        {
            od.getLastQuaternion(i) = od.getOldQuaternion(i);
            od.getOldQuaternion(i) = od.getQuaternion(i);
            TimeIntegration::semiImplicitEulerRotation(h, od.getMass(i), od.getInvMass(i) * Matrix3r::Identity() ,od.getQuaternion(i), od.getVelocity(i), Vector3r(0,0,0));
        }
    }

    msg_info("SofaPBDTimeStep") << "================== PBD preStep (before positionConstraintProjection) ==================";
    msg_info("SofaPBDTimeStep") << "Retrieved data arrays from model: rigidBodies = " << rb.size() << ", particles = " << pd.size() << ", particle orientations = " << od.size();

    for (size_t k = 0; k < numBodies; k++)
    {
        msg_info("SofaPBDTimeStep") << "Rigid body " << k << ": position = (" << rb[k]->getPosition()[0] << ", " << rb[k]->getPosition()[1] << ", " << rb[k]->getPosition()[2] << "), orientation = (" << rb[k]->getRotation().x() << ", " << rb[k]->getRotation().y() << ", " << rb[k]->getRotation().z() << ", " << rb[k]->getRotation().w() << ")";
    }
    msg_info("SofaPBDTimeStep") << "================== PBD preStep (before positionConstraintProjection) ==================";

    // START_TIMING("position constraints projection");
    msg_info("SofaPBDTimeStep") << "Calling positionConstraintProjection()";
    positionConstraintProjection(*model);
    // STOP_TIMING_AVG;

    #pragma omp parallel if(numBodies > MIN_PARALLEL_SIZE) default(shared)
    {
        // Update velocities
        msg_info("SofaPBDTimeStep") << "Updating velocities - rigid bodies";

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < numBodies; i++)
        {
            if (VELOCITY_UPDATE_METHOD.getValue().getSelectedId() == 0)
            {
                if (rb[i]->getMass() != 0.0)
                {
                    msg_info("SofaPBDTimeStep") << "Rigid body " << i << " -- first-order velocity update.";
                    TimeIntegration::velocityUpdateFirstOrder(h, rb[i]->getMass(), rb[i]->getPosition(), rb[i]->getOldPosition(), rb[i]->getVelocity());
                    TimeIntegration::angularVelocityUpdateFirstOrder(h, rb[i]->getMass(), rb[i]->getRotation(), rb[i]->getOldRotation(), rb[i]->getAngularVelocity());
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
                    TimeIntegration::velocityUpdateSecondOrder(h, rb[i]->getMass(), rb[i]->getPosition(), rb[i]->getOldPosition(), rb[i]->getLastPosition(), rb[i]->getVelocity());
                    TimeIntegration::angularVelocityUpdateSecondOrder(h, rb[i]->getMass(), rb[i]->getRotation(), rb[i]->getOldRotation(), rb[i]->getLastRotation(), rb[i]->getAngularVelocity());
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
        msg_info("SofaPBDTimeStep") << "Updating velocities - particles: " << pd.size();

        #pragma omp for schedule(static)
        for (int i = 0; i < (int) pd.size(); i++)
        {
            if (VELOCITY_UPDATE_METHOD.getValue().getSelectedId() == 0)
                TimeIntegration::velocityUpdateFirstOrder(h, pd.getMass(i), pd.getPosition(i), pd.getOldPosition(i), pd.getVelocity(i));
            else
                TimeIntegration::velocityUpdateSecondOrder(h, pd.getMass(i), pd.getPosition(i), pd.getOldPosition(i), pd.getLastPosition(i), pd.getVelocity(i));
        }

        // Update velocites of orientations
        msg_info("SofaPBDTimeStep") << "Updating velocities - particle orientations";

        #pragma omp for schedule(static)
        for (int i = 0; i < (int)od.size(); i++)
        {
            if (VELOCITY_UPDATE_METHOD.getValue().getSelectedId() == 0)
                TimeIntegration::angularVelocityUpdateFirstOrder(h, od.getMass(i), od.getQuaternion(i), od.getOldQuaternion(i), od.getVelocity(i));
            else
                TimeIntegration::angularVelocityUpdateSecondOrder(h, od.getMass(i), od.getQuaternion(i), od.getOldQuaternion(i), od.getLastQuaternion(i), od.getVelocity(i));
        }

        msg_info("SofaPBDTimeStep") << "================== PBD preStep (after velocity updates) ==================";
        msg_info("SofaPBDTimeStep") << "Retrieved data arrays from model: rigidBodies = " << rb.size() << ", particles = " << pd.size() << ", particle orientations = " << od.size();

        for (size_t k = 0; k < numBodies; k++)
        {
            msg_info("SofaPBDTimeStep") << "Rigid body " << k << ": position = (" << rb[k]->getPosition()[0] << ", " << rb[k]->getPosition()[1] << ", " << rb[k]->getPosition()[2] << "), orientation = (" << rb[k]->getRotation().x() << ", " << rb[k]->getRotation().y() << ", " << rb[k]->getRotation().z() << ", " << rb[k]->getRotation().w() << ")";
        }
        msg_info("SofaPBDTimeStep") << "================== PBD preStep (after velocity updates) ==================";
    }
}

void SofaPBDTimeStep::doCollisionDetection(const ExecParams *params, SReal dt)
{
    msg_info("SofaPBDTimeStep") << "Running collision queries: collision detection method = " << COLLISION_DETECTION_METHOD.getValue().getSelectedId();

    if (COLLISION_DETECTION_METHOD.getValue().getSelectedId() == 0)
    {
        msg_info("SofaPBDTimeStep") << "Using SOFA-integrated collision detection.";
        if (m_sofaPBDCollisionDetection)
        {
            sofa::helper::AdvancedTimer::stepBegin("SofaPBDCollisionVisitor");

            msg_info("SofaPBDAnimationLoop") << "Starting collision detection.";
            if (m_collisionPipeline)
            {
                msg_info("SofaPBDAnimationLoop") << "Using collision pipeline instance from simulation root node: " << m_collisionPipeline->getName();
                SofaPBDCollisionVisitor pbd_col_visitor(m_collisionPipeline, params, dt);
                msg_info("SofaPBDAnimationLoop") << "SofaPBDCollisionVisitor instantiated, calling gnode->execute()";
                gnode->execute(pbd_col_visitor);
                msg_info("SofaPBDAnimationLoop") << "SofaPBDCollisionVisitor pass done.";
            }
            else
            {
                msg_info("SofaPBDAnimationLoop") << "Using locally instantiated collision pipeline object: " << m_collisionPipelineLocal->getName();
                SofaPBDCollisionVisitor pbd_col_visitor(m_collisionPipelineLocal.get(), params, dt);
                msg_info("SofaPBDAnimationLoop") << "SofaPBDCollisionVisitor instantiated, calling gnode->execute()";
                gnode->execute(pbd_col_visitor);
                msg_info("SofaPBDAnimationLoop") << "SofaPBDCollisionVisitor pass done.";
            }

            sofa::helper::AdvancedTimer::stepEnd("SofaPBDCollisionVisitor");

            SimulationModel* simModel = SofaPBDSimulation::getCurrent()->getModel();

            simModel->resetContacts();

            const SimulationModel::RigidBodyVector& rigidBodies = simModel->getRigidBodies();
            const ParticleData& pd = simModel->getParticles();
            const SimulationModel::LineModelVector& lineModels = simModel->getLineModels();

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

            msg_info("SofaPBDTimeStep") << "==============================================================";
            if (dynamic_cast<SofaPBDPipeline*>(this->m_collisionPipeline))
            {
                SofaPBDPipeline* sofaPBDPipeline = dynamic_cast<SofaPBDPipeline*>(this->m_collisionPipeline);
                msg_info("SofaPBDTimeStep") << "SofaPBDPipeline (m_collisionPipeline) calls in this iteration: External = " << sofaPBDPipeline->getExternalCallCount() << ", internal = " << sofaPBDPipeline->getInternalCallCount();
                sofaPBDPipeline->resetExternalCallCount();
            }

            if (dynamic_cast<SofaPBDPipeline*>(this->m_collisionPipelineLocal.get()))
            {
                SofaPBDPipeline* sofaPBDPipeline = dynamic_cast<SofaPBDPipeline*>(this->m_collisionPipelineLocal.get());
                msg_info("SofaPBDTimeStep") << "SofaPBDPipeline (m_collisionPipelineLocal) calls in this iteration: External = " << sofaPBDPipeline->getExternalCallCount() << ", internal = " << sofaPBDPipeline->getInternalCallCount();
                sofaPBDPipeline->resetExternalCallCount();
            }
            msg_info("SofaPBDTimeStep") << "==============================================================";
        }
        else
        {
            msg_warning("SofaPBDTimeStep") << "No SOFAPBDBruteForceDetection instance found, collision detection is non-functional!";
        }
    }
    else
    {
        msg_info("SofaPBDTimeStep") << "Using PBD-integrated collision detection.";
        if (m_collisionDetection)
        {
            SimulationModel* model = SofaPBDSimulation::getCurrent()->getModel();
            // START_TIMING("collision detection");
            m_collisionDetection->collisionDetection(*model);
            // STOP_TIMING_AVG;
        }
    }
}

void SofaPBDTimeStep::step(const core::ExecParams *params, SReal dt)
{
    msg_info("SofaPBDTimeStep") << "SofaPBDTimeStep::step()";

    if (!m_simulation)
    {
        msg_warning("SofaPBDTimeStep") << "No valid SofaPBDSimulation instance, aborting!";
        return;
    }

    if (!m_simulation->getModel())
    {
        msg_warning("SofaPBDTimeStep") << "No valid SimulationModel instance, aborting!";
        return;
    }

    SimulationModel::RigidBodyVector &rb = this->m_simulation->getModel()->getRigidBodies();
    ParticleData& pd = this->m_simulation->getModel()->getParticles();
    OrientationData& od = this->m_simulation->getModel()->getOrientations();
    msg_info("SofaPBDTimeStep") << "================== SofaPBDTimeStep at begin of step " << dt << " ==================";
    msg_info("SofaPBDTimeStep") << "Retrieved data arrays from model: rigidBodies = " << rb.size() << ", particles = " << pd.size() << ", particle orientations = " << od.size();

    const size_t numBodies = rb.size();
    for (size_t i = 0; i < numBodies; i++)
    {
        msg_info("SofaPBDTimeStep") << "Rigid body " << i << " current position = " << rb[i]->getPosition()[0] << ", " << rb[i]->getPosition()[1] << ", " << rb[i]->getPosition()[2] << ")";
        msg_info("SofaPBDTimeStep") << "Rigid body " << i << " current velocity = " << rb[i]->getVelocity()[0] << ", " << rb[i]->getVelocity()[1] << ", " << rb[i]->getVelocity()[2] << ")";
        msg_info("SofaPBDTimeStep") << "Rigid body " << i << " current acceler. = " << rb[i]->getAcceleration()[0] << ", " << rb[i]->getAcceleration()[1] << ", " << rb[i]->getAcceleration()[2] << ")";
    }

    const size_t numParticles = pd.size();
    for (size_t i = 0; i < numParticles; i++)
    {
        msg_info("SofaPBDTimeStep") << "Particle " << i << " current position = (" << pd.getPosition(i)[0] << ", " << pd.getPosition(i)[1] << ", " << pd.getPosition(i)[2] << ")";
        msg_info("SofaPBDTimeStep") << "Particle " << i << " current velocity = (" << pd.getVelocity(i)[0] << ", " << pd.getVelocity(i)[1] << ", " << pd.getVelocity(i)[2] << ")";
        msg_info("SofaPBDTimeStep") << "Particle " << i << " current acceler. = (" << pd.getAcceleration(i)[0] << ", " << pd.getAcceleration(i)[1] << ", " << pd.getAcceleration(i)[2] << ")";
    }

    for (size_t i = 0; i < numParticles; i++)
    {
        msg_info("SofaPBDTimeStep") << "Particle " << i << " current quaternion = (" << od.getQuaternion(i).x() << ", " << od.getQuaternion(i).y() << ", " << od.getQuaternion(i).z() << ", " << od.getQuaternion(i).w() << ")";
        msg_info("SofaPBDTimeStep") << "Particle " << i << " current rot. vel.  = (" << od.getVelocity(i)[0] << ", " << od.getVelocity(i)[1] << ", " << od.getVelocity(i)[2] << ")";
        msg_info("SofaPBDTimeStep") << "Particle " << i << " current rot. acc.  = (" << od.getAcceleration(i)[0] << ", " << od.getAcceleration(i)[1] << ", " << od.getAcceleration(i)[2] << ")";

    }

    msg_info("SofaPBDTimeStep") << "================== SofaPBDTimeStep at begin of step " << dt << " ==================";

    this->preStep();
    this->doCollisionDetection(params, dt);
    this->postStep();
}

void SofaPBDTimeStep::postStep()
{
    msg_info("SofaPBDTimeStep") << "Calling velocityConstraintProjection()";

    SimulationModel* model = SofaPBDSimulation::getCurrent()->getModel();
    TimeManager *tm = TimeManager::getCurrent();
    const Real h = tm->getTimeStepSize();

    velocityConstraintProjection(*model);

    //////////////////////////////////////////////////////////////////////////
    // update motor joint targets
    //////////////////////////////////////////////////////////////////////////
    SimulationModel::ConstraintVector &constraints = model->getConstraints();

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

    SimulationModel::RigidBodyVector &rb = model->getRigidBodies();
    ParticleData& pd = model->getParticles();
    OrientationData& od = model->getOrientations();
    msg_info("SofaPBDTimeStep") << "Particles: " << pd.size();

    msg_info("SofaPBDTimeStep") << "================== PBD postStep ==================";
    msg_info("SofaPBDTimeStep") << "Retrieved data arrays from model: rigidBodies = " << rb.size() << ", particles = " << pd.size() << ", particle orientations = " << od.size();

    const int numBodies = (int) rb.size();
    const int numParticles = (int) pd.size();

    for (size_t k = 0; k < numBodies; k++)
    {
        msg_info("SofaPBDTimeStep") << "Rigid body " << k << ": position = (" << rb[k]->getPosition()[0] << ", " << rb[k]->getPosition()[1] << ", " << rb[k]->getPosition()[2] << "), orientation = (" << rb[k]->getRotation().x() << ", " << rb[k]->getRotation().y() << ", " << rb[k]->getRotation().z() << ", " << rb[k]->getRotation().w() << ")";
    }

    for (int i = 0; i < numParticles; i++)
    {
        msg_info("SofaPBDTimeStep") << "Particle " << i << ": position = (" << pd.getPosition(i)[0] << "," << pd.getPosition(i)[1] << "," << pd.getPosition(i)[2] << "), velocity = (" << pd.getVelocity(i)[0] << "," << pd.getVelocity(i)[1] << "," << pd.getVelocity(i)[2] << ")";
    }

    msg_info("SofaPBDTimeStep") << "================== PBD postStep ==================";

    // STOP_TIMING_AVG;
}

void SofaPBDTimeStep::positionConstraintProjection(SimulationModel &model)
{
    m_iterations = 0;

    // init constraint groups if necessary
    model.initConstraintGroups();

    SimulationModel::RigidBodyVector &rb = model.getRigidBodies();
    SimulationModel::ConstraintVector &constraints = model.getConstraints();
    SimulationModel::ConstraintGroupVector &groups = model.getConstraintGroups();
    SimulationModel::RigidBodyContactConstraintVector &contacts = model.getRigidBodyContactConstraints();
    SimulationModel::ParticleSolidContactConstraintVector &particleTetContacts = model.getParticleSolidContactConstraints();

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

void SofaPBDTimeStep::velocityConstraintProjection(SimulationModel &model)
{
    msg_info("SofaPBDTimeStep") << "velocityConstraintProjection() - MAX_ITERATIONS_V = " << MAX_ITERATIONS_V.getValue();
    m_iterationsV = 0;

    // init constraint groups if necessary
    model.initConstraintGroups();

    SimulationModel::RigidBodyVector &rb = model.getRigidBodies();
    SimulationModel::ConstraintVector &constraints = model.getConstraints();
    SimulationModel::ConstraintGroupVector &groups = model.getConstraintGroups();
    SimulationModel::RigidBodyContactConstraintVector &rigidBodyContacts = model.getRigidBodyContactConstraints();
    SimulationModel::ParticleRigidBodyContactConstraintVector &particleRigidBodyContacts = model.getParticleRigidBodyContactConstraints();
    SimulationModel::ParticleSolidContactConstraintVector &particleTetContacts = model.getParticleSolidContactConstraints();

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
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    SimulationModel* model = SofaPBDSimulation::getCurrent()->getModel();
    const SimulationModel::RigidBodyContactConstraintVector& rbConstraints = model->getRigidBodyContactConstraints();

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->enableLighting();

    msg_info("SofaPBDTimeStep") << "PBD rigid body constraints count = " << rbConstraints.size();

    if (rbConstraints.size() > 0)
    {
        Vec4f normalColor(0.5f, 0.9f, 0.2f, 0.25f);
        Vec4f normalColor2(0.5f, 0.9f, 0.2f, 0.75f);
        Vec4f tangentColor(0.2f, 0.5f, 0.9f, 0.25f);
        Vec4f tangentColor2(0.2f, 0.5f, 0.9f, 0.75f);
        Vec4f linVelColor(0.9f, 0.5f, 0.2f, 0.25f);

        std::stringstream oss;
        for (size_t k = 0; k < rbConstraints.size(); k++)
        {
            // constraintInfo contains
            // 0:	contact point in body 0 (global)
            // 1:	contact point in body 1 (global)
            // 2:	contact normal in body 1 (global)
            // 3:	contact tangent (global)
            // 0,4:  1.0 / normal^T * K * normal
            // 1,4: maximal impulse in tangent direction
            // 2,4: goal velocity in normal direction after collision

            const RigidBodyContactConstraint& rbc = rbConstraints[k];
            const Vector3r &cp1 = rbc.m_constraintInfo.col(0);
            const Vector3r &cp2 = rbc.m_constraintInfo.col(1);
            const Vector3r &normal = rbc.m_constraintInfo.col(2);
            const Vector3r &tangent = rbc.m_constraintInfo.col(3);

            const Real sumImpulses = rbc.m_sum_impulses;
            /*const Real frictionImpulse = rbc.m_frictionImpulse;
            const Real corrMagnitude = rbc.m_correctionMagnitude;*/
            const Real maxImpulseTangentDir = rbc.m_constraintInfo(1,4);
            const Real goalVelocityNormalDir = rbc.m_constraintInfo(2,4);

            Vector3 contactPoint1(cp1[0], cp1[1], cp1[2]);
            Vector3 contactPoint2(cp2[0], cp2[1], cp2[2]);
            Vector3 normalVector(normal[0], normal[1], normal[2]);
            Vector3 tangentVector(tangent[0], tangent[1], tangent[2]);

            /*Vector3 rbVelCorrLin1(rbc.m_corrLin_rb1[0], rbc.m_corrLin_rb1[1], rbc.m_corrLin_rb1[2]);
            Vector3 rbVelCorrLin2(rbc.m_corrLin_rb2[0], rbc.m_corrLin_rb2[1], rbc.m_corrLin_rb2[2]);*/

            vparams->drawTool()->drawSphere(contactPoint1, 0.02f);
            vparams->drawTool()->drawSphere(contactPoint2, 0.02f);

            vparams->drawTool()->drawArrow(contactPoint1, contactPoint1 + normalVector, 0.0025f, normalColor, 8);
            vparams->drawTool()->drawArrow(contactPoint1, contactPoint1 + tangentVector, 0.0025f, tangentColor, 8);

            /*vparams->drawTool()->drawArrow(contactPoint1, contactPoint1 + rbVelCorrLin1, 0.005f, linVelColor, 8);
            vparams->drawTool()->drawArrow(contactPoint2, contactPoint2 + rbVelCorrLin2, 0.005f, linVelColor, 8);*/

            Vector3 scaledNormalVector = goalVelocityNormalDir * normalVector;
            vparams->drawTool()->drawArrow(contactPoint1, contactPoint1 + scaledNormalVector, 0.005f, normalColor2, 16);
            vparams->drawTool()->drawArrow(contactPoint1, contactPoint1 + (maxImpulseTangentDir * tangentVector), 0.005f, tangentColor2, 16);

            oss.str("");
            oss << "Constraint " << k << ": v_n = " << goalVelocityNormalDir << ", i_t_max = " << maxImpulseTangentDir << std::endl << " sum_imp. = " << sumImpulses;
            // << ", friction_imp. = " << frictionImpulse << ", corr_mag. = " << corrMagnitude;

            Vector3 labelPos((contactPoint2.x() - contactPoint1.x()) / 2.0f,
                             (contactPoint2.y() - contactPoint1.y()) / 2.0f,
                             (contactPoint2.z() - contactPoint1.z()) / 2.0f);
            vparams->drawTool()->draw3DText((1.0f + (k * 0.05f)) * labelPos, 0.03f, normalColor2, oss.str().c_str());
        }
    }

    vparams->drawTool()->disableLighting();
    vparams->drawTool()->restoreLastState();
}
