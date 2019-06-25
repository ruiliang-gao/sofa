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

#include "PBDSimulationModel.h"
#include "PBDSimulation.h"
#include "PBDTimeStep.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API PBDAnimationLoop: public sofa::simulation::DefaultAnimationLoop
            {
                public:
                    PBDAnimationLoop(sofa::simulation::Node* = nullptr);

                    virtual void init() override;
                    virtual void bwdInit () override;

                    /// perform one animation step
                    /*
                     * Inputs : ExecParams *    -> Execution context
                     *          SReal           -> Time Step
                     *
                     * Output : Compute a single step of the simulation
                     */
                    virtual void step(const sofa::core::ExecParams* params, SReal dt) override;

               protected:
                    PBDSimulationModel* m_simulationModel;
                    PBDSimulation* m_simulation;
                    PBDTimeStep* m_timeStep;

            };
        }
    }
}
#endif // PBDANIMATIONLOOP_H
