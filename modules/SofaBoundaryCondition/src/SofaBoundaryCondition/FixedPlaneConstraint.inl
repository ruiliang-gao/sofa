/******************************************************************************
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

#include <SofaBoundaryCondition/FixedPlaneConstraint.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/vector_algorithm.h>

namespace sofa::component::projectiveconstraintset
{

using sofa::helper::WriteAccessor;
using sofa::type::Vec;

/////////////////////////// DEFINITION OF FixedPlaneConstraint /////////////////////////////////////
template <class DataTypes>
FixedPlaneConstraint<DataTypes>::FixedPlaneConstraint()
    : d_direction( initData(&d_direction,"direction","normal direction of the plane"))
    , d_dmin( initData(&d_dmin,(Real)0,"dmin","Minimum plane distance from the origin"))
    , d_dmax( initData(&d_dmax,(Real)0,"dmax","Maximum plane distance from the origin") )
    , d_indices( initData(&d_indices,"indices","Indices of the fixed points"))
    , l_topology(initLink("topology", "link to the topology container"))
{
    m_selectVerticesFromPlanes=false;   
}

template <class DataTypes>
FixedPlaneConstraint<DataTypes>::~FixedPlaneConstraint()
{

}

/// Matrix Integration interface
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::applyConstraint(const MechanicalParams* mparams, const MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    if(const MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(mstate.get()))
    {
        /// Implement plane constraint only when the direction is along the coordinates directions
        // TODO : generalize projection to any direction

        const unsigned int N = Deriv::size();
        Coord dir=d_direction.getValue();
        for (auto& index : d_indices.getValue())
        {
            /// Reset Fixed Row and Col
            for (unsigned int c=0; c<N; ++c)
                if (dir[c]!=0.0)
                    r.matrix->clearRowCol(r.offset + N * index + c);
            /// Set Fixed Vertex
            for (unsigned int c=0; c<N; ++c)
                if (dir[c]!=0.0)
                    r.matrix->set(r.offset + N * index + c, r.offset + N * index + c, 1.0);
        }
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::applyConstraint(const MechanicalParams* mparams,
                                                      BaseVector* vect,
                                                      const MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    const int o = matrix->getGlobalOffset(mstate.get());
    if (o >= 0)
    {
        const unsigned int offset = (unsigned int)o;
        /// Implement plane constraint only when the direction is along the coordinates directions
        // TODO : generalize projection to any direction
        Coord dir=d_direction.getValue();

        const unsigned int N = Deriv::size();

        for (auto& index : d_indices.getValue())
        {
            for (unsigned int c=0; c<N; ++c)
                if (dir[c]!=0.0)
                    vect->clear(offset + N * index + c);
        }
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::addConstraint(Index index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::removeConstraint(Index index)
{
    sofa::type::removeValue(*d_indices.beginEdit(),(unsigned int)index);
    d_indices.endEdit();
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectResponse(const MechanicalParams* mparams, DataVecDeriv& resData)
{
    WriteAccessor<DataVecDeriv> res = resData;
    projectResponseImpl(mparams, res.wref());
}

/// project dx to constrained space (dx models a velocity)
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectVelocity(const MechanicalParams* /*mparams*/, DataVecDeriv& /*vData*/)
{

}

/// project x to constrained space (x models a position)
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectPosition(const MechanicalParams* /*mparams*/, DataVecCoord& /*xData*/)
{

}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned /*offset*/ )
{
    /// clears the rows and columns associated with constrained particles
    const unsigned blockSize = DataTypes::deriv_total_size;

    for(auto& index : d_indices.getValue())
    {
        M->clearRowsCols((index) * blockSize,(index+1) * (blockSize) );
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectJacobianMatrix(const MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    WriteAccessor<DataMatrixDeriv> c = cData;
    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    while (rowIt != rowItEnd)
    {
        projectResponseImpl(mparams, rowIt.row());
        ++rowIt;
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::setDirection(Coord dir)
{
    if (dir.norm2()>0)
    {
        d_direction.setValue(dir);
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::selectVerticesAlongPlane()
{
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();
    for(unsigned int i=0; i<x.size(); ++i)
    {
        if (isPointInPlane(x[i]))
            addConstraint(i);
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::init()
{
    ProjectiveConstraintSet<DataTypes>::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (sofa::core::topology::BaseMeshTopology* _topology = l_topology.get())
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";
        
        // Initialize topological changes support
        d_indices.createTopologyHandler(_topology);
    }
    else
    {
        msg_info() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
    }

    /// test that dmin or dmax are different from zero
    if (d_dmin.getValue()!=d_dmax.getValue())
        m_selectVerticesFromPlanes=true;

    if (m_selectVerticesFromPlanes)
        selectVerticesAlongPlane();
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::setDminAndDmax(const Real _dmin,const Real _dmax)
{
    d_dmin=_dmin;
    d_dmax=_dmax;
    m_selectVerticesFromPlanes=true;
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::draw(const VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();
    vparams->drawTool()->disableLighting();

    type::vector<sofa::type::Vector3> points;
    for(auto& index : d_indices.getValue())
    {
        points.push_back({x[index][0], x[index][1], x[index][2]});
    }

    vparams->drawTool()->drawPoints(points, 10, sofa::type::RGBAColor{1,1.0,0.5,1});
}

/// This function are there to provide kind of type translation to the vector one so we can
/// implement the algorithm as is the different objects where of similar type.
/// this solution is not really satisfactory but for the moment it does the job.
/// A better solution would that all the used types are following the same iterface which
/// requires to touch core sofa classes.
sofa::type::Vec3d& getVec(sofa::defaulttype::Rigid3dTypes::Deriv& i){ return i.getVCenter(); }
sofa::type::Vec3d& getVec(sofa::defaulttype::Rigid3dTypes::Coord& i){ return i.getCenter(); }
const sofa::type::Vec3d& getVec(const sofa::defaulttype::Rigid3dTypes::Coord& i){ return i.getCenter(); }
sofa::type::Vec3d& getVec(sofa::defaulttype::Vec3dTypes::Deriv& i){ return i; }
const sofa::type::Vec3d& getVec(const sofa::defaulttype::Vec3dTypes::Deriv& i){ return i; }
sofa::type::Vec6d& getVec(sofa::defaulttype::Vec6dTypes::Deriv& i){ return i; }
const sofa::type::Vec6d& getVec(const sofa::defaulttype::Vec6dTypes::Deriv& i){ return i; }

sofa::type::Vec3f& getVec(sofa::defaulttype::Rigid3fTypes::Deriv& i){ return i.getVCenter(); }
sofa::type::Vec3f& getVec(sofa::defaulttype::Rigid3fTypes::Coord& i){ return i.getCenter(); }
const sofa::type::Vec3f& getVec(const sofa::defaulttype::Rigid3fTypes::Coord& i){ return i.getCenter(); }
sofa::type::Vec3f& getVec(sofa::defaulttype::Vec3fTypes::Deriv& i){ return i; }
const sofa::type::Vec3f& getVec(const sofa::defaulttype::Vec3fTypes::Deriv& i){ return i; }
sofa::type::Vec6f& getVec(sofa::defaulttype::Vec6fTypes::Deriv& i){ return i; }
const sofa::type::Vec6f& getVec(const sofa::defaulttype::Vec6fTypes::Deriv& i){ return i; }

template<class DataTypes>
bool FixedPlaneConstraint<DataTypes>::isPointInPlane(Coord p) const
{
    Vec<Coord::spatial_dimensions,Real> pos = getVec(p) ;
    Real d=pos*getVec(d_direction.getValue());
    if ((d>d_dmin.getValue())&& (d<d_dmax.getValue()))
        return true;
    else
        return false;
}

template <class DataTypes>
template <class T>
void FixedPlaneConstraint<DataTypes>::projectResponseImpl(const MechanicalParams* mparams, T& res) const
{
    SOFA_UNUSED(mparams);

    Coord dir=d_direction.getValue();
    for (auto& index : d_indices.getValue())
    {
        /// only constraint one projection of the displacement to be zero
        getVec(res[index]) -= getVec(dir) * dot( getVec(res[index]), getVec(dir));
    }
}

} // namespace sofa::component::projectiveconstraintset
