#ifndef SOFAPBDCOLLISIONVISITOR_H
#define SOFAPBDCOLLISIONVISITOR_H

#include "initZyPositionBasedDynamicsPlugin.h"
#include <sofa/simulation/CollisionVisitor.h>

#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>

#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDCollisionVisitor: public Visitor
            {
                public:
                    SofaPBDCollisionVisitor(sofa::core::collision::Pipeline* pipeline, const core::ExecParams* params = core::ExecParams::defaultInstance());
                    SofaPBDCollisionVisitor(sofa::core::collision::Pipeline* pipeline, const core::ExecParams* params, SReal dt);

                    Result processNodeTopDown(simulation::Node* node) override;

                    /// Specify whether this action can be parallelized.
                    bool isThreadSafe() const override { return true; }

                    /// Return a category name for this action.
                    /// Only used for debugging / profiling purposes
                    const char* getCategoryName() const override { return "animate"; }
                    const char* getClassName() const override { return "SofaPBDCollisionVisitor"; }

                    void setDt(SReal v) { dt = v; }
                    SReal getDt() const { return dt; }

                    virtual void processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj);

                protected:
                    SReal dt;
                    bool firstNodeVisited;
                    sofa::core::collision::Pipeline* m_pipeline;
            };
        }
    }
}

#endif // SOFAPBDCOLLISIONVISITOR_H
