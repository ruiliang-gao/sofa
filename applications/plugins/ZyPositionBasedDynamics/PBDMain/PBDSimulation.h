#ifndef PBDSIMULATION_H
#define PBDSIMULATION_H

#include <sofa/core/objectmodel/BaseObject.h>
#include "initZyPositionBasedDynamicsPlugin.h"

#include <PBDCommon/PBDCommon.h>
#include "PBDModels/PBDSimulationModel.h"
#include "PBDTimeStep.h"

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            enum PBDSimulationMethods { PBD = 0, XPBD, IBDS, NumSimulationMethods };

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API PBDSimulation: public sofa::core::objectmodel::BaseObject
            {
                public:
                    Data<sofa::defaulttype::Vec3d> GRAVITATION;
                    Data<sofa::helper::OptionsGroup> SIMULATION_METHOD;

                    PBDSimulation();
                    ~PBDSimulation();

                    void init();
                    void reset();

                    // Singleton
                    static PBDSimulation* getCurrent ();
                    static void setCurrent(PBDSimulation* tm);
                    static bool hasCurrent();

                    PBDSimulationModel *getModel() { return m_model; }
                    void setModel(PBDSimulationModel *model) { m_model = model; }

                    int getSimulationMethod() const { return SIMULATION_METHOD.getValue().getSelectedId(); }
                    void setSimulationMethod(const int val);

                    void setSimulationMethodChangedCallback(std::function<void()> const& callBackFct);

                    PBDTimeStep *getTimeStep() { return m_timeStep; }
                    void setTimeStep(PBDTimeStep *ts) { m_timeStep = ts; }

                protected:
                    PBDSimulationModel *m_model;
                    PBDTimeStep *m_timeStep;
                    std::function<void()> m_simulationMethodChanged;

                    virtual void initParameters();

                private:
                    static PBDSimulation *current;
            };
        }
    }
}

#endif // PBDSIMULATION_H
