#ifndef SOFAPBDTIMESTEPINTERFACE_H
#define SOFAPBDTIMESTEPINTERFACE_H

#include "initZyPositionBasedDynamicsPlugin.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDTimeStepInterface
            {
                public:
                    SofaPBDTimeStepInterface() {}
                    virtual ~SofaPBDTimeStepInterface() {}

                    virtual void init() = 0;
                    virtual void bwdInit() = 0;
                    virtual void reset() = 0;
                    virtual void cleanup() = 0;

                    virtual double getTime() = 0;

                    virtual void draw(const core::visual::VisualParams*) = 0;

                    virtual void step() = 0;
            };
        }
    }
}

#endif // SOFAPBDTIMESTEPINTERFACE_H
