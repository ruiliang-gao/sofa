#ifndef PBDANIMATIONLOOP_H
#define PBDANIMATIONLOOP_H

#include "initZyPositionBasedDynamicsPlugin.h"
#include <framework/sofa/simulation/DefaultAnimationLoop.h>
#include <sofa/simulation/Node.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/behavior/ForceField.h>

#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>

#include "SofaPBDSimulation.h"
#include "PBDUtils/PBDTimeManager.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDAnimationLoop: public sofa::simulation::DefaultAnimationLoop
            {
                public:
                    SOFA_CLASS(SofaPBDAnimationLoop, sofa::simulation::DefaultAnimationLoop);

                    SofaPBDAnimationLoop(sofa::simulation::Node*&);
                    virtual ~SofaPBDAnimationLoop();

                    typedef sofa::simulation::DefaultAnimationLoop Inherit;
                    typedef sofa::core::objectmodel::BaseContext BaseContext;
                    typedef sofa::core::objectmodel::BaseObjectDescription BaseObjectDescription;

                    sofa::core::objectmodel::Data<int> SUB_STEPS_PER_ITERATION;

                    virtual void setNode(simulation::Node*);

                    virtual void init() override;
                    virtual void bwdInit() override;

                    /// perform one animation step
                    /*
                     * Inputs : ExecParams *    -> Execution context
                     *          SReal           -> Time Step
                     *
                     * Output : Compute a single step of the simulation
                     */
                    virtual void step(const sofa::core::ExecParams* params, SReal dt) override;

               protected:
                    SofaPBDSimulation* m_simulation;
                    BaseContext* m_context;

                    sofa::core::collision::Pipeline* m_collisionPipeline;
                    sofa::core::collision::Pipeline::SPtr m_collisionPipelineLocal;

                    Real m_dt;
                    Real m_prevDt;
            };
        }
    }
}
#endif // PBDANIMATIONLOOP_H
