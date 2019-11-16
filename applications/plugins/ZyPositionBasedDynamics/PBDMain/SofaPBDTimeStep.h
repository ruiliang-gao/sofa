#ifndef PBDTIMESTEP_H
#define PBDTIMESTEP_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/simulation/Node.h>

#include "initZyPositionBasedDynamicsPlugin.h"

#include "SimulationModel.h"
#include "CollisionDetection.h"

#include "PBDIntegration/SofaPBDBruteForceDetection.h"
#include "PBDMain/SofaPBDTimeStepInterface.h"

#define MIN_PARALLEL_SIZE 64

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            using namespace PBD;
            using namespace sofa::component::collision;

            class SofaPBDSimulation;

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDTimeStep: public SofaPBDTimeStepInterface, public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SofaPBDTimeStep, sofa::core::objectmodel::BaseObject);

                    SofaPBDTimeStep(SofaPBDSimulation* = nullptr);
                    virtual ~SofaPBDTimeStep();

                    virtual void init();
                    virtual void bwdInit();
                    virtual void reset();
                    virtual void cleanup();

                    virtual double getTime() override;

                    void draw(const core::visual::VisualParams*) override;

                    virtual void step(const sofa::core::ExecParams* params, SReal dt);

                    void setCollisionDetection(SimulationModel &model, CollisionDetection *cd);
                    CollisionDetection *getCollisionDetection();

                    Data<int> MAX_ITERATIONS;
                    Data<int> MAX_ITERATIONS_V;
                    Data<sofa::helper::OptionsGroup> VELOCITY_UPDATE_METHOD;
                    Data<sofa::helper::OptionsGroup> COLLISION_DETECTION_METHOD;

                protected:
                    unsigned int m_iterations;
                    unsigned int m_iterationsV;

                    virtual void preStep();
                    virtual void doCollisionDetection(const sofa::core::ExecParams* params, SReal dt);
                    virtual void postStep();

                    virtual void initParameters();

                    void positionConstraintProjection(SimulationModel &model);
                    void velocityConstraintProjection(SimulationModel &model);

                    sofa::simulation::Node* gnode;

                    SofaPBDSimulation* m_simulation;

                    // Built-in collision detection of the PBD library
                    CollisionDetection *m_collisionDetection;

                    // Sofa collision detection pipelines
                    sofa::core::collision::Pipeline* m_collisionPipeline;
                    sofa::core::collision::Pipeline::SPtr m_collisionPipelineLocal;

                    // Sofa-integrated collision pipeline
                    SofaPBDBruteForceDetection* m_sofaPBDCollisionDetection;

                    /** Clear accelerations and add gravitation.
                    */
                    void clearAccelerations(SimulationModel &model);
            };
        }
    }
}

#endif // PBDTIMESTEP_H
