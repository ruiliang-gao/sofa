#ifndef GEOMETRYCONVERSION_H
#define GEOMETRYCONVERSION_H

#include "initZyPositionBasedDynamicsPlugin.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <PBDModels/PBDSimulationModel.h>

#include <vector>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDCollisionGeometryAdapter: public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SofaPBDCollisionGeometryAdapter, sofa::core::objectmodel::BaseObject);
                    SofaPBDCollisionGeometryAdapter();

                private:
                    sofa::core::topology::BaseMeshTopology* m_topology;
            };
        }
    }
}

#endif // GEOMETRYCONVERSION_H
