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
#include <SofaBoundaryCondition/config.h>
#include <sofa/config.h>

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/vector.h>
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <set>

namespace sofa::component::projectiveconstraintset
{

/** Attach given particles to their initial positions. This is a temporary class, somehow redundant with FixedConstraint, simplified to avoid the memory leak issue. @todo Remove one of the two classes
*/
template <class DataTypes>
class SOFA_ATTRIBUTE_DEPRECATED__POINTCONSTRAINT() PointConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PointConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef sofa::type::Vector3 Vector3;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef type::vector<unsigned int> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;


    SetIndex f_indices; ///< Indices of the fixed points
    Data<SReal> _drawSize; ///< 0 -> point based rendering, >0 -> radius of spheres
protected:
    PointConstraint();

    virtual ~PointConstraint();
public:

    // -- Constraint interface
    void init() override;

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;
    virtual const sofa::linearalgebra::BaseMatrix* getJ(const core::MechanicalParams* );

    void applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    void draw(const core::visual::VisualParams* vparams) override;


protected :

    /// Matrix used in getJ
    linearalgebra::SparseMatrix<SReal> jacobian;

};


#if  !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_POINTCONSTRAINT_CPP)
extern template class SOFA_SOFABOUNDARYCONDITION_API PointConstraint<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::projectiveconstraintset
