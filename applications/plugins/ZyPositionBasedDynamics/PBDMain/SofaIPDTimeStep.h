#ifndef SOFAIPDTIMESTEP_H
#define SOFAIPDTIMESTEP_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/OptionsGroup.h>

#include "initZyPositionBasedDynamicsPlugin.h"
#include "SofaPBDTimeStepInterface.h"

#include "Simulator.hpp"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaIPDTimeStep: public SofaPBDTimeStepInterface, public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SofaIPDTimeStep, sofa::core::objectmodel::BaseObject);

                    SofaIPDTimeStep();
                    virtual ~SofaIPDTimeStep();

                    virtual void init();
                    virtual void bwdInit();
                    virtual void reset();
                    virtual void cleanup();

                    virtual double getTime() override;

                    virtual void draw(const core::visual::VisualParams*);

                    virtual void step();

                protected:
                    std::shared_ptr<IPS::Simulator> m_simulator;
            };
        }
    }
}

#endif // SOFAIPDTIMESTEP_H
