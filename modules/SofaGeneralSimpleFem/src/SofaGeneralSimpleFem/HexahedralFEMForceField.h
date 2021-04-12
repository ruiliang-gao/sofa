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
#include <SofaGeneralSimpleFem/config.h>

#include <sofa/core/behavior/ForceField.h>

#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TopologyData.h>
#include <SofaBaseTopology/HexahedronSetTopologyModifier.h>
#include <SofaBoundaryCondition/FixedConstraint.h>

namespace sofa::component::forcefield
{

/** Compute Finite Element forces based on hexahedral elements.
*
* Corotational hexahedron from
* @Article{NMPCPF05,
*   author       = "Nesme, Matthieu and Marchal, Maud and Promayon, Emmanuel and Chabanas, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
*   title        = "Physically Realistic Interactive Simulation for Biological Soft Tissues",
*   journal      = "Recent Research Developments in Biomechanics",
*   volume       = "2",
*   year         = "2005",
*   keywords     = "surgical simulation physical animation truth cube",
*   url          = "http://www-evasion.imag.fr/Publications/2005/NMPCPF05"
* }
*
* indices ordering (same as in HexahedronSetTopology):
*
*     Y  7---------6
*     ^ /         /|
*     |/    Z    / |
*     3----^----2  |
*     |   /     |  |
*     |  4------|--5
*     | /       | /
*     |/        |/
*     0---------1-->X
*/
template<class DataTypes>
class HexahedralFEMForceField : virtual public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HexahedralFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef helper::ReadAccessor< DataVecCoord > RDataRefVecCoord;
    typedef helper::WriteAccessor< DataVecDeriv > WDataRefVecDeriv;

    typedef core::topology::BaseMeshTopology::Index Index;
    typedef core::topology::BaseMeshTopology::Hexa Element;
    typedef core::topology::BaseMeshTopology::SeqHexahedra VecElement;

    typedef defaulttype::Vec<24, Real> Displacement;		///< the displacement vector

    typedef defaulttype::Mat<6, 6, Real> MaterialStiffness;	///< the matrix of material stiffness
    typedef helper::vector<MaterialStiffness> VecMaterialStiffness;  ///< a vector of material stiffness matrices
    typedef defaulttype::Mat<24, 24, Real> ElementMass;

    typedef defaulttype::Mat<24, 24, Real> ElementStiffness;
    typedef helper::vector<ElementStiffness> VecElementStiffness;


    enum
    {
        LARGE = 0,   ///< Symbol of large displacements hexahedron solver
        POLAR = 1,   ///< Symbol of polar displacements hexahedron solver
    };

protected:

    typedef defaulttype::Mat<3, 3, Real> Mat33;
    typedef Mat33 Transformation; ///< matrix for rigid transformations like rotations

    typedef std::pair<int,Real> Col_Value;
    typedef helper::vector< Col_Value > CompressedValue;
    typedef helper::vector< CompressedValue > CompressedMatrix;

    /// the information stored for each hexahedron
    class HexahedronInformation
    {
    public:
        /// material stiffness matrices of each hexahedron
        MaterialStiffness materialMatrix;

        // large displacement method
        helper::fixed_array<Coord,8> rotatedInitialElements;

        Transformation rotation; // element rotation at the deformed space
        ElementStiffness stiffness;

        // UF TIPS - SC
         /// Plasticity Deformation
        Real plasticYieldThreshold;
        Real plasticMaxThreshold;
        Real restVolume; //inital rest element volume
        Transformation F_C; //Cell center deformation gradient
        Transformation materialDeformationInverse; ///inverse of (J of reference X over rest x')
        helper::fixed_array<Coord, 8> elementPlasticOffset;/// element plastic offset per vertex at current timestep
        bool needsToUpdateRestMesh;

        HexahedronInformation() {}

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const HexahedronInformation& /*hi*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, HexahedronInformation& /*hi*/ )
        {
            return in;
        }
    };


    HexahedralFEMForceField();
    virtual ~HexahedralFEMForceField();
public:
    void setPoissonRatio(Real val) { this->f_poissonRatio.setValue(val); }

    void setYoungModulus(Real val) { this->f_youngModulus.setValue(val); }

    void setMethod(int val) { method = val; }

    void init() override;
    void reinit() override;

    void addForce (const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    void addDForce (const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    void draw(const core::visual::VisualParams* vparams) override;

protected:

    virtual void computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const defaulttype::Vec<8,Coord> &nodes);
    Mat33 integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  );

    /// compute the hookean material matrix
    void computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio);

    void computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K );


    ////////////// large displacements method
    void initLarge(const int i);
    void computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey);
    virtual void accumulateForceLarge( WDataRefVecDeriv& f, RDataRefVecCoord& p, const int i);

    ////////////// polar decomposition method
    void initPolar(const int i);
    void computeRotationPolar( Transformation &r, defaulttype::Vec<8,Coord> &nodes);
    virtual void accumulateForcePolar( WDataRefVecDeriv& f, RDataRefVecCoord & p, const int i);

public:
    int method;
    Data<std::string> f_method; ///< the computation method of the displacements
    Data<Real> f_poissonRatio;
    Data<Real> f_youngModulus;

    /*********************Plasticity Method Reference************************
   Author    Bargteil, Adam W., et al.
   Title     "A finite element method for animating large viscoplastic flow."
   Journal   ACM transactions on graphics(TOG) 26.3 (2007) : 16 - es.
    ******************Ruiliang(rgao15@ufl.edu)******************************/
    /// Plasticity Material Parameters
    Data<Real> f_plasticMaxThreshold;
    Data<Real> f_plasticYieldThreshold; ///< Plastic Yield Threshold (on the deformation gradient)
    Data<Real> f_plasticCreep; ///< plastic flow rate
    Data<Real> f_hardeningParam;/// work hardening params (refer to K.alpha in the paper)
    Data<bool> f_useVertexPlasticity; //if true, using vertex+center F for plasticity 
    Data<bool> f_preserveElementVolume; //if true: preserve element volume under plasticity deformation
    Data<bool> f_updateElementStiffness; //if true, element stiffness matrix will be updated when necessary
    helper::vector< Transformation > F_V; //Vertex Deformation
    helper::vector< Coord > restStateOffsets;//restState plastic offsets for the current time step
    Data<sofa::helper::vector<Real> > debugData; ///< debugData
    ///Plasticity Related Methods
    Mat33 computeCenterJacobian(const helper::fixed_array<Coord, 8>& coords); ///< Compute the center value jacobian from the input hexahedral element
    Mat33 computeJacobian(const helper::fixed_array<Coord, 8>& coords, Real x, Real y, Real z); ///< Compute the exact jacobian located at the reference coords (x,y,z) \in [-1,1]^3
    Real computeElementVolume(const helper::fixed_array<Coord, 8>& coords); //compute the exact hex element volume
    Real totalVolume;
    Real d_currentVolume; //For Debugging : compute total volume at current timestep
    Data<int> f_debugPlasticMethod; //for debugging
    bool d_UpdateRestStatePerElement = true;
    bool needsToUpdateRestMesh = false;
    void computeF_C(RDataRefVecCoord& p, int i); //compute and store F_C for hex element i
    void computeAllF_V(); //compute and store F_V (vertex deformation gradient) for hex element i
    void updateRestStateLarge();//update the rest state plastic offsets per vertex for the current time step using large method
    void updateRestStatePolar();//update the rest state plastic offsets per vertex, using Polar method

    /// container that stotes all requires information for each hexahedron
    topology::HexahedronData<sofa::helper::vector<HexahedronInformation> > hexahedronInfo;

    class HFFHexahedronHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Hexahedron,sofa::helper::vector<HexahedronInformation> >
    {
    public:
        typedef typename HexahedralFEMForceField<DataTypes>::HexahedronInformation HexahedronInformation;

        HFFHexahedronHandler(HexahedralFEMForceField<DataTypes>* ff, topology::HexahedronData<sofa::helper::vector<HexahedronInformation> >* data )
            :topology::TopologyDataHandler<core::topology::BaseMeshTopology::Hexahedron,sofa::helper::vector<HexahedronInformation> >(data)
            ,ff(ff)
        {
        }

        void applyCreateFunction(Index, HexahedronInformation &t, const core::topology::BaseMeshTopology::Hexahedron &,
                const sofa::helper::vector<Index> &, const sofa::helper::vector<double> &);
    protected:
        HexahedralFEMForceField<DataTypes>* ff;
    };



protected:
    HFFHexahedronHandler* hexahedronHandler;

    //topology::HexahedronSetTopologyContainer* _topology;
    sofa::core::topology::BaseMeshTopology* _topology;
    helper::vector<unsigned int> _fixedIndices;
    defaulttype::Mat<8,3,int> _coef; ///< coef of each vertices to compute the strain stress matrix
};

#if  !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_CPP)
extern template class SOFA_SOFAGENERALSIMPLEFEM_API HexahedralFEMForceField<defaulttype::Vec3Types>;

#endif

} // sofa::component::forcefield
