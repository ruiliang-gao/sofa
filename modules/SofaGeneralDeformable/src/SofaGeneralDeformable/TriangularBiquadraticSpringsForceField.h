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

#include <SofaGeneralDeformable/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaBaseTopology/TopologyData.h>

#include <sofa/config.h>
#include <sofa/config/sharedlibrary_defines.h>

#define SOFAGENERALDEFORMABLE_VERSION @PROJECT_VERSION@

#ifdef SOFA_BUILD_SOFAGENERALDEFORMABLE
#  define SOFA_TARGET @PROJECT_NAME@
#  define SOFA_SOFAGENERALDEFORMABLE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_SOFAGENERALDEFORMABLE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace sofa::component::forcefield
{

template<class DataTypes>
class TriangularBiquadraticSpringsForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularBiquadraticSpringsForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    using Index = sofa::Index;

    class Mat3 : public sofa::helper::fixed_array<Deriv,3>
    {
    public:
        Deriv operator*(const Deriv& v)
        {
            return Deriv((*this)[0]*v,(*this)[1]*v,(*this)[2]*v);
        }
        Deriv transposeMultiply(const Deriv& v)
        {
            return Deriv(v[0]*((*this)[0])[0]+v[1]*((*this)[1])[0]+v[2]*((*this)[2][0]),
                    v[0]*((*this)[0][1])+v[1]*((*this)[1][1])+v[2]*((*this)[2][1]),
                    v[0]*((*this)[0][2])+v[1]*((*this)[1][2])+v[2]*((*this)[2][2]));
        }
    };

protected:


    class EdgeRestInformation
    {
    public:
        Real  restSquareLength;	// the rest length
        Real  currentSquareLength; 	// the current edge length
        Real  deltaL2;  // the current unit direction
        Real stiffness;

        EdgeRestInformation()
        {
        }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeRestInformation& /*eri*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeRestInformation& /*eri*/ )
        {
            return in;
        }
    };

    class TriangleRestInformation
    {
    public:
        Real  gamma[3];	// the angular stiffness
        Real stiffness[3]; // the elongation stiffness
        Mat3 DfDx[3]; /// the edge stiffness matrix

        Coord currentNormal;
        Coord lastValidNormal;
        Real area;
        Real restArea;
        Coord areaVector[3];
        Deriv dp[3];
        Real J;

        TriangleRestInformation()
        {
        }
        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleRestInformation& /*tri*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleRestInformation& /*vec*/ )
        {
            return in;
        }
    };

    sofa::component::topology::TriangleData<helper::vector<TriangleRestInformation> > triangleInfo; ///< Internal triangle data
    sofa::component::topology::EdgeData<helper::vector<EdgeRestInformation> > edgeInfo; ///< Internal edge data
    
    Data < VecCoord >  _initialPoints;										///< the intial positions of the points

    bool updateMatrix;

    Data<Real> f_poissonRatio; ///< Poisson ratio in Hooke's law
    Data<Real> f_youngModulus; ///< Young modulus in Hooke's law
    Data<Real> f_dampingRatio; ///< Ratio damping/stiffness
    Data<bool> f_useAngularSprings; ///< whether angular springs should be included

    Data<bool> f_compressible; ///< whether the material is compressible or not
    /**** coefficient that controls how the material can cope with very compressible cases
    must be between 0 and 1 : if 0 then the deformation may diverge for large compression
    if 1 then the material can undergo large compression even inverse elements ***/
    Data<Real> f_stiffnessMatrixRegularizationWeight; ///< Regularization of the Stiffnes Matrix (between 0 and 1)

    Real lambda;  /// first Lame coefficient
    Real mu;    /// second Lame coefficient


    TriangularBiquadraticSpringsForceField();

    virtual ~TriangularBiquadraticSpringsForceField();
public:
    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    virtual Real getLambda() const { return lambda;}
    virtual Real getMu() const { return mu;}

    void setYoungModulus(const double modulus)
    {
        f_youngModulus.setValue((Real)modulus);
    }
    void setPoissonRatio(const double ratio)
    {
        f_poissonRatio.setValue((Real)ratio);
    }

    void draw(const core::visual::VisualParams* vparams) override;
    /// compute lambda and mu based on the Young modulus and Poisson ratio
    void updateLameCoefficients();

    class TRBSEdgeHandler : public sofa::component::topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, sofa::helper::vector<EdgeRestInformation> >
    {
    public:
        typedef typename TriangularBiquadraticSpringsForceField<DataTypes>::EdgeRestInformation EdgeRestInformation;

        TRBSEdgeHandler(TriangularBiquadraticSpringsForceField<DataTypes>* ff,
                sofa::component::topology::EdgeData<sofa::helper::vector<EdgeRestInformation> >* data)
            :sofa::component::topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, sofa::helper::vector<EdgeRestInformation> >(data)
            ,ff(ff)
        {
        }
        void applyCreateFunction(Index, EdgeRestInformation &t, const core::topology::BaseMeshTopology::Edge &,
                const sofa::helper::vector<Index> &, const sofa::helper::vector<double> &);

    protected:
        TriangularBiquadraticSpringsForceField<DataTypes>* ff;
    };

    class TRBSTriangleHandler : public sofa::component::topology::TopologyDataHandler<core::topology::BaseMeshTopology::Triangle,sofa::helper::vector<TriangleRestInformation> >
    {
    public:
        typedef typename TriangularBiquadraticSpringsForceField<DataTypes>::TriangleRestInformation TriangleRestInformation;

        TRBSTriangleHandler(TriangularBiquadraticSpringsForceField<DataTypes>* ff,
                sofa::component::topology::TriangleData<sofa::helper::vector<TriangleRestInformation> >* data)
            :sofa::component::topology::TopologyDataHandler<core::topology::BaseMeshTopology::Triangle,sofa::helper::vector<TriangleRestInformation> >(data)
            ,ff(ff)
        {
        }

        void applyCreateFunction(Index, TriangleRestInformation &t,
                const core::topology::BaseMeshTopology::Triangle &,
                const sofa::helper::vector<Index> &,
                const sofa::helper::vector<double> &);
        void applyDestroyFunction(Index, TriangleRestInformation &);


    protected:
        TriangularBiquadraticSpringsForceField<DataTypes>* ff;
    };

    /// Link to be set to the topology container in the component graph.
    SingleLink<TriangularBiquadraticSpringsForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected :
    TRBSEdgeHandler* edgeHandler;
    TRBSTriangleHandler* triangleHandler;

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* m_topology;

    sofa::component::topology::EdgeData<helper::vector<EdgeRestInformation> > &getEdgeInfo() {return edgeInfo;}

};


#if  !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARBIQUADRATICSPRINGSFORCEFIELD_CPP)

extern template class SOFA_SOFAGENERALDEFORMABLE_API TriangularBiquadraticSpringsForceField<sofa::defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARBIQUADRATICSPRINGSFORCEFIELD_CPP)

} //namespace sofa::component::forcefield
