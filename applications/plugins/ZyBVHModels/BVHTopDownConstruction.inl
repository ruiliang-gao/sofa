#include "BVHTopDownConstruction.h"

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>

using namespace sofa::component::collision;

template <class DataTypes>
BVHTopDownConstruction<DataTypes>::BVHTopDownConstruction()
{

}

template <class DataTypes>
void BVHTopDownConstruction<DataTypes>::init()
{
	std::cout << "BVHTopDownConstruction<DataTypes>::init(" << this->getName() << ")" << std::endl;
	sofa::core::behavior::BaseMechanicalState* base_mech_state = this->getContext()->getMechanicalState();
	sofa::core::topology::BaseMeshTopology* base_mesh_topology = this->getContext()->getMeshTopology();

	if (base_mech_state != NULL)
	{
		std::cout << " got base mechanical state: " << base_mech_state->getName() << std::endl;
	}

	if (base_mesh_topology != NULL)
	{
		std::cout << " got base mesh topology: " << base_mesh_topology->getName() << std::endl;
	}
}

template <class DataTypes>
void BVHTopDownConstruction<DataTypes>::cleanup()
{

}

template <class DataTypes>
void BVHTopDownConstruction<DataTypes>::draw(const core::visual::VisualParams* vparams)
{

}