#ifndef PBDTIMESTEP_H
#define PBDTIMESTEP_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/OptionsGroup.h>

#include "initZyPositionBasedDynamicsPlugin.h"

#include "PBDModels/PBDSimulationModel.h"

#include "CollisionDetection.h"
#include "PBDIntegration/SofaPBDBruteForceDetection.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            using namespace sofa::simulation::PBDDistanceBasedCD;
            using namespace sofa::component::collision;

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDTimeStep: public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SofaPBDTimeStep, sofa::core::objectmodel::BaseObject);

                    SofaPBDTimeStep();
                    virtual ~SofaPBDTimeStep();

                    virtual void init();
                    virtual void bwdInit();
                    virtual void reset();
                    virtual void cleanup();

                    void draw(const core::visual::VisualParams*) override;

                    virtual void step(PBDSimulationModel &model);

                    void setCollisionDetection(PBDSimulationModel &model, CollisionDetection *cd);
                    CollisionDetection *getCollisionDetection();

                    Data<int> MAX_ITERATIONS;
                    Data<int> MAX_ITERATIONS_V;
                    Data<sofa::helper::OptionsGroup> VELOCITY_UPDATE_METHOD;
                    Data<sofa::helper::OptionsGroup> COLLISION_DETECTION_METHOD;

                protected:
                    unsigned int m_iterations;
                    unsigned int m_iterationsV;

                    virtual void initParameters();

                    void positionConstraintProjection(PBDSimulationModel &model);
                    void velocityConstraintProjection(PBDSimulationModel &model);

                    // Built-in collision detection of the PBD library
                    CollisionDetection *m_collisionDetection;

                    // Sofa-integrated collision pipeline
                    SofaPBDBruteForceDetection* m_sofaPBDCollisionDetection;

                    /** Clear accelerations and add gravitation.
                    */
                    void clearAccelerations(PBDSimulationModel &model);

                    /// TODO: Check for re-entrancy!
                    /*static void contactCallbackFunction(const unsigned int contactType,
                        const unsigned int bodyIndex1, const unsigned int bodyIndex2,
                        const Vector3r &cp1, const Vector3r &cp2,
                        const Vector3r &normal, const Real dist,
                        const Real restitutionCoeff, const Real frictionCoeff, void *userData);*/

                    /// TODO: Check for re-entrancy!
                    /*static void solidContactCallbackFunction(const unsigned int contactType,
                        const unsigned int bodyIndex1, const unsigned int bodyIndex2,
                        const unsigned int tetIndex, const Vector3r &bary,
                        const Vector3r &cp1, const Vector3r &cp2,
                        const Vector3r &normal, const Real dist,
                        const Real restitutionCoeff, const Real frictionCoeff, void *userData);*/
            };
        }
    }
}

#endif // PBDTIMESTEP_H
