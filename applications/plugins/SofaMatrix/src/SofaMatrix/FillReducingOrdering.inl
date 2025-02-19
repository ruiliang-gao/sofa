﻿/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <SofaMatrix/FillReducingOrdering.h>

namespace sofa::component::linearsolver
{

template <class DataTypes>
FillReducingOrdering<DataTypes>::FillReducingOrdering()
    : l_mstate(initLink("mstate", "Mechanical state to reorder"))
    , l_topology(initLink("topology", "Topology to reorder"))
    , d_permutation(initData(&d_permutation, "permutation", "Output vector of indices mapping the reordered vertices to the initial list"))
    , d_invPermutation(initData(&d_invPermutation, "invPermutation", "Output vector of indices mapping the initial vertices to the reordered list"))
    , d_position(initData(&d_position, "position", "Reordered position vector"))
    , d_hexahedra(initData(&d_hexahedra, "hexahedra", "Reordered hexahedra"))
    , d_tetrahedra(initData(&d_tetrahedra, "tetrahedra", "Reordered tetrahedra"))
{
}

template <class DataTypes>
void FillReducingOrdering<DataTypes>::init()
{
    DataEngine::init();

    addOutput(&d_permutation);
    addOutput(&d_invPermutation);
    addOutput(&d_position);
    addOutput(&d_hexahedra);
    addOutput(&d_tetrahedra);

    if (!l_mstate)
    {
        l_mstate.set(dynamic_cast< core::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState()));

        msg_error_when(!l_mstate) << "No compatible MechanicalState found in the current context. "
            "This may be because there is no MechanicalState in the local context, "
            "or because the type is not compatible.";
    }

    if (!l_topology)
    {
        l_topology.set(getContext()->getMeshTopology());

        msg_error_when(!l_topology) << "No mesh topology found in the current context.";
    }

    setDirtyValue();
    reinit();
}

template <class DataTypes>
void FillReducingOrdering<DataTypes>::reinit()
{
    DataEngine::reinit();

    update();
}

template <class DataTypes>
void FillReducingOrdering<DataTypes>::doUpdate()
{
    // The number of elements in the mesh
    idx_t ne = l_topology->getNbHexahedra() + l_topology->getNbTetrahedra();
    // The number of nodes in the mesh.
    idx_t nn = l_mstate->getSize();
    //  The size of the eptr array is n+ 1, where n is the number of elements in the mesh
    std::vector<idx_t> eptr(ne+1);
    // The size of the eind array is of size equal to the sum of the number of nodes in all the elements of the mesh.
    std::vector<idx_t> eind(l_topology->getNbHexahedra() * sofa::geometry::Hexahedron::NumberOfNodes + l_topology->getNbTetrahedra() * sofa::geometry::Tetrahedron::NumberOfNodes);
    idx_t numflag {};

    unsigned int eind_id {};
    unsigned int eptr_id {};

    initializeFromElements(l_topology->getHexahedra(), eptr, eind, eptr_id, eind_id);
    initializeFromElements(l_topology->getTetrahedra(), eptr, eind, eptr_id, eind_id);

    idx_t* xadj { nullptr };
    idx_t* adjncy { nullptr };

    int metis_out = METIS_MeshToNodal(&ne, &nn, eptr.data(), eind.data(), &numflag, &xadj, &adjncy);
    if (metis_out != METIS_OK)
    {
        msg_error() << "Generating the nodal graph of the mesh failed.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    auto perm = helper::getWriteAccessor(d_permutation);
    perm.resize(nn);

    auto invperm = helper::getWriteAccessor(d_invPermutation);
    invperm.resize(nn);

    metis_out = METIS_NodeND(&nn, xadj, adjncy, nullptr, nullptr, perm->data(),invperm->data());

    delete xadj;
    delete adjncy;

    if (metis_out != METIS_OK)
    {
        msg_error() << "Computing fill reducing ordering failed.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // Update topology
    updateElements(l_topology->getHexahedra(), d_hexahedra);
    updateElements(l_topology->getTetrahedra(), d_tetrahedra);

    // Update mechanical state
    auto pos = helper::getWriteOnlyAccessor(d_position);
    pos.clear();

    auto previousPos = l_mstate->readPositions();

    for (unsigned int i = 0; i < previousPos.size(); ++i)
    {
        pos.push_back(previousPos[perm[i]]);
    }
}

template <class DataTypes>
template <class Elements>
void FillReducingOrdering<DataTypes>::initializeFromElements(const Elements& elements, std::vector<idx_t>& eptr, std::vector<idx_t>& eind, unsigned int& eptr_id, unsigned int& eind_id)
{
    for (const auto& element : elements)
    {
        for (int i = 0; i < element.static_size; ++i)
        {
            eind[eind_id++] = element[i];
        }
        eptr[++eptr_id] = eind_id;
    }
}

template <class DataTypes>
template <class Element>
void FillReducingOrdering<DataTypes>::updateElements(
    const sofa::type::vector<Element>& inElementSequence,
    Data<sofa::type::vector<Element>>& outElementSequenceData)
{
    auto elements = helper::getWriteOnlyAccessor(outElementSequenceData);
    elements.clear();

    auto invperm = helper::getReadAccessor(d_invPermutation);

    for (const auto& element : inElementSequence)
    {
        auto newElement = element;
        for (int i = 0; i < element.static_size; ++i)
        {
            newElement[i] = invperm[element[i]];
        }
        elements.push_back(newElement);
    }
}

} // namespace sofa::component::linearsolver
