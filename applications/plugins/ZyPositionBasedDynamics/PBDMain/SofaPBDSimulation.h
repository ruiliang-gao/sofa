#ifndef PBDSIMULATION_H
#define PBDSIMULATION_H

#include <sofa/core/objectmodel/BaseObject.h>
#include "initZyPositionBasedDynamicsPlugin.h"

#include <Common/Common.h>
#include "SimulationModel.h"
#include "SofaPBDTimeStep.h"

#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/simulation/Node.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            enum PBDSimulationMethods { PBD = 0, XPBD, IBDS, NumSimulationMethods };

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDSimulation: public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SofaPBDSimulation, sofa::core::objectmodel::BaseObject);

                    Data<sofa::defaulttype::Vec3d> GRAVITY;
                    Data<sofa::helper::OptionsGroup> SIMULATION_METHOD;

                    SofaPBDSimulation(sofa::core::objectmodel::BaseContext* context = nullptr);
                    ~SofaPBDSimulation();

                    void init();
                    void bwdInit();
                    void reset();
                    void cleanup();

                    void draw(const core::visual::VisualParams*) override;

                    // Singleton
                    static SofaPBDSimulation* getCurrent ();
                    static void setCurrent(SofaPBDSimulation* tm);
                    static bool hasCurrent();

                    SimulationModel *getModel() { return m_model; }
                    void setModel(SimulationModel *model) { m_model = model; }

                    int getSimulationMethod() const { return SIMULATION_METHOD.getValue().getSelectedId(); }
                    void setSimulationMethod(const int val);

                    void setSimulationMethodChangedCallback(std::function<void()> const& callBackFct);

                    SofaPBDTimeStepInterface* getTimeStep() { return m_timeStep; }
                    void setTimeStep(SofaPBDTimeStepInterface* ts) { m_timeStep = ts; }

                protected:
                    sofa::core::objectmodel::BaseContext* m_context;
                    sofa::simulation::Node::SPtr m_rootNode;

                    SimulationModel *m_model;
                    SofaPBDTimeStepInterface* m_timeStep;
                    std::function<void()> m_simulationMethodChanged;

                    virtual void initParameters();

                private:
                    static SofaPBDSimulation *current;
            };
        }
    }
}

#endif // PBDSIMULATION_H
