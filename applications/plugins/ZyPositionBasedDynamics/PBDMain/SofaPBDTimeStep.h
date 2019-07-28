#ifndef PBDTIMESTEP_H
#define PBDTIMESTEP_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/OptionsGroup.h>

#include "initZyPositionBasedDynamicsPlugin.h"

#include "PBDModels/PBDSimulationModel.h"
#include "CollisionDetection.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            using namespace sofa::simulation::PBDDistanceBasedCD;

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDTimeStep: public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SofaPBDTimeStep, sofa::core::objectmodel::BaseObject);

                    SofaPBDTimeStep();

                    virtual ~SofaPBDTimeStep(void);

                    virtual void init();
                    virtual void reset();
                    virtual void cleanup();

                    virtual void step(PBDSimulationModel &model);

                    void setCollisionDetection(PBDSimulationModel &model, CollisionDetection *cd);
                    CollisionDetection *getCollisionDetection();

                    Data<int> MAX_ITERATIONS;
                    Data<int> MAX_ITERATIONS_V;
                    Data<sofa::helper::OptionsGroup> VELOCITY_UPDATE_METHOD;

                protected:
                    int m_velocityUpdateMethod;
                    unsigned int m_iterations;
                    unsigned int m_iterationsV;
                    unsigned int m_maxIterations;
                    unsigned int m_maxIterationsV;

                    virtual void initParameters();

                    void positionConstraintProjection(PBDSimulationModel &model);
                    void velocityConstraintProjection(PBDSimulationModel &model);

                    CollisionDetection *m_collisionDetection;

                    /** Clear accelerations and add gravitation.
                    */
                    void clearAccelerations(PBDSimulationModel &model);

                    /// TODO: Check for re-entrancy!
                    static void contactCallbackFunction(const unsigned int contactType,
                        const unsigned int bodyIndex1, const unsigned int bodyIndex2,
                        const Vector3r &cp1, const Vector3r &cp2,
                        const Vector3r &normal, const Real dist,
                        const Real restitutionCoeff, const Real frictionCoeff, void *userData);

                    /// TODO: Check for re-entrancy!
                    static void solidContactCallbackFunction(const unsigned int contactType,
                        const unsigned int bodyIndex1, const unsigned int bodyIndex2,
                        const unsigned int tetIndex, const Vector3r &bary,
                        const Vector3r &cp1, const Vector3r &cp2,
                        const Vector3r &normal, const Real dist,
                        const Real restitutionCoeff, const Real frictionCoeff, void *userData);
            };
        }
    }
}

#endif // PBDTIMESTEP_H
