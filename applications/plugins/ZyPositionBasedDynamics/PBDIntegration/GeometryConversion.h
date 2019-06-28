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

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API GeometryConversion: public sofa::core::objectmodel::BaseObject
            {
                public:
                    GeometryConversion(PBDSimulationModel*);

                    const std::vector<sofa::core::topology::BaseMeshTopology*>& getMeshTopologies() const { return m_topologies; }
                    void setMeshTopologies(std::vector<sofa::core::topology::BaseMeshTopology*>& topologies) { m_topologies = topologies; }

                    bool convertToPBDObjects();

                private:
                    std::shared_ptr<PBDSimulationModel> m_model;
                    std::vector<sofa::core::topology::BaseMeshTopology*> m_topologies;
            };
        }
    }
}

#endif // GEOMETRYCONVERSION_H
