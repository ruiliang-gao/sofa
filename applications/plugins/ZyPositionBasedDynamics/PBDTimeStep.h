#ifndef PBDTIMESTEP_H
#define PBDTIMESTEP_H

#include <sofa/core/objectmodel/BaseObject.h>
#include "initZyPositionBasedDynamicsPlugin.h"

#include "PBDSimulationModel.h"
#include "CollisionDetection.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            using namespace sofa::simulation::PBDDistanceBasedCD;

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API PBDTimeStep: public sofa::core::objectmodel::BaseObject
            {
                public:
                    PBDTimeStep();

                    virtual ~PBDTimeStep(void);

                    virtual void init();
                    virtual void reset();

                    virtual void step(PBDSimulationModel &model) = 0;

                    void setCollisionDetection(PBDSimulationModel &model, CollisionDetection *cd);
                    CollisionDetection *getCollisionDetection();

                protected:
                    virtual void initParameters();

                    CollisionDetection *m_collisionDetection;

                    /** Clear accelerations and add gravitation.
                    */
                    void clearAccelerations(PBDSimulationModel &model);

                    /*static void contactCallbackFunction(const unsigned int contactType,
                        const unsigned int bodyIndex1, const unsigned int bodyIndex2,
                        const Vector3r &cp1, const Vector3r &cp2,
                        const Vector3r &normal, const Real dist,
                        const Real restitutionCoeff, const Real frictionCoeff, void *userData);

                    static void solidContactCallbackFunction(const unsigned int contactType,
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
