#include "BVHBottomUpConstruction.h"

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>

using namespace sofa::component::collision;

template <class DataTypes>
BVHBottomUpConstruction<DataTypes>::BVHBottomUpConstruction()
{

}

template <class DataTypes>
void BVHBottomUpConstruction<DataTypes>::init()
{
    msg_info("BVHBottomUpConstruction") << "init(" << this->getName() << ")";
    sofa::core::behavior::BaseMechanicalState* base_mech_state = this->getContext()->getMechanicalState();
    sofa::core::topology::BaseMeshTopology* base_mesh_topology = this->getContext()->getMeshTopology();

    if (base_mech_state != NULL)
    {
        msg_info("BVHBottomUpConstruction") << "Got base mechanical state: " << base_mech_state->getName();
    }

    if (base_mesh_topology != NULL)
    {
        msg_info("BVHBottomUpConstruction") << "Got base mesh topology: " << base_mesh_topology->getName();
    }
}

template <class DataTypes>
void BVHBottomUpConstruction<DataTypes>::cleanup()
{

}

template <class DataTypes>
void BVHBottomUpConstruction<DataTypes>::draw(const core::visual::VisualParams* vparams)
{

}

