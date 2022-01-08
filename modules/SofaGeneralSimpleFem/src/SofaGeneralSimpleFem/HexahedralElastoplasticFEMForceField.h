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
#include "HexahedralFEMForceField.h"

#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>

//#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <sofa/core/topology/TopologyData.h>
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
* Plasticity components from
* @Article{
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
class HexahedralElastoplasticFEMForceField : virtual public core::behavior::ForceField<DataTypes>, virtual public HexahedralFEMForceField<DataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(HexahedralElastoplasticFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes), SOFA_TEMPLATE(HexahedralFEMForceField, DataTypes));
    
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

    typedef type::Vec<24, Real> Displacement;		///< the displacement vector

    typedef type::Mat<6, 6, Real> MaterialStiffness;	///< the matrix of material stiffness
    typedef type::vector<MaterialStiffness> VecMaterialStiffness;  ///< a vector of material stiffness matrices
    typedef type::Mat<24, 24, Real> ElementMass;

    typedef type::Mat<24, 24, Real> ElementStiffness;
    typedef type::vector<ElementStiffness> VecElementStiffness;

    enum
    {
        LARGE = 0,   ///< Symbol of large displacements hexahedron solver
        POLAR = 1,   ///< Symbol of polar displacements hexahedron solver
    };

protected:

    typedef type::Mat<3, 3, Real> Mat33;
    typedef Mat33 Transformation; ///< matrix for rigid transformations like rotations

    typedef std::pair<int,Real> Col_Value;
    typedef type::vector< Col_Value > CompressedValue;
    typedef type::vector< CompressedValue > CompressedMatrix;

    /// the information stored for each hexahedron, overwrite this from the base class with extra plasticity parameters
    class HexahedronInformation
    {
    public:
        /// material stiffness matrices of each hexahedron
        MaterialStiffness materialMatrix;

        // large displacement method
        type::fixed_array<Coord,8> rotatedInitialElements;

        // Element Rotation related variables (handles rotational plasticity)
        Transformation rotation;//element rotation at current time. from local frame to world frame
        Transformation lastRotation; //element rotation at the previous timestep. from local frame to world frame
        Transformation restRotation; //rest element rotation. from local frame to world frame
        Real plasticRotationYieldThreshold;
        Real plasticMaxRotationThreshold;
        type::fixed_array<Transformation, 8> vertRotation; //rotation sensor per-vertex
        ElementStiffness stiffness; //element stiffness matrix

        /// Element Deformation related variables (handles symmetric plasticity)
        Real plasticYieldThreshold;
        Real plasticMaxThreshold;
        Real restVolume; //undeformed element volume
        Transformation F_C; //Cell center deformation gradient
        Transformation materialDeformationInverse; ///inverse of Jacobian(= dX/dx' with reference X , rest x')
        type::fixed_array<Coord, 8> elementPlasticOffset;/// element plastic offset per vertex at current timestep
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


    HexahedralElastoplasticFEMForceField();
    virtual ~HexahedralElastoplasticFEMForceField();
//public:
//    void setPoissonRatio(Real val) { this->f_poissonRatio.setValue(val); }
//
//    void setYoungModulus(Real val) { this->f_youngModulus.setValue(val); }
//
//    void setMethod(int val) { method = val; }
//
    void init() override;
    void reinit() override;

    void addForce (const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    void addDForce (const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    //TODO PotentialEnergy will be needed for the plastic optimization approach
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }
//
//    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
//
    void draw(const core::visual::VisualParams* vparams) override;

    //virtual void computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const type::Vec<8,Coord> &nodes);
    //Mat33 integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  );

    ///// compute the hookean material matrix
    //void computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio);

    //void computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K );


    //////////// large displacements methods that overrides 
    void initLarge(const int i);
    //void computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey);
    virtual void accumulateForceLarge( WDataRefVecDeriv& f, RDataRefVecCoord& p, const int i);
    //void updateRestStateLarge();//update the rest state plastic offsets per vertex, using the Large method
    

    ////////////// polar decomposition method
    void initPolar(const int i);
    //void computeRotationPolar( Transformation &r, type::Vec<8,Coord> &nodes);
    virtual void accumulateForcePolar( WDataRefVecDeriv& f, RDataRefVecCoord & p, const int i);
    void updateRestStatePlasticity();//update the rest state plastic offsets per vertex, using the Polar method

public:

    ///////////// Plasticity Material Parameters /////////////
    Data<Real> f_plasticMaxThreshold;
    Data<Real> f_plasticYieldThreshold; ///< Plastic Yield Threshold (on the deformation gradient)
    Data<Real> f_plasticCreep; ///< plastic flow rate
    /// TODO: the plasticRotation Parameters should be automatically set
    Data<Real> f_plasticRotationYieldThreshold; ///< Plastic rotation Yield Threshold
    Data<Real> f_plasticMaxRotationThreshold; ///the maximal plastic rototation per element  
    Data<Real> f_plasticRotationCreep; ///< plastic rotation flow rate
    Data<Real> f_hardeningParam;/// work hardening params (refer to K.alpha in the paper), crucial in maintain stability
    Data<bool> f_useHigherOrderPlasticity; //if true, Use higher order scheme -- mixed vertex + center deformation for plasticity
    Data<bool> f_preserveElementVolume; //if true: preserve element volume under plasticity deformation
    Data<bool> f_updateElementStiffness; //if true, element stiffness matrix will be updated when necessary
    //type::vector< Transformation > F_V; //Vertex Deformation -- global vector -- not used
      type::vector< Coord > restStateOffsets;//restState plastic offsets for the current time step
    

    ///Plasticity Related Methods/Helpers
    Mat33 computeCenterJacobian(const type::fixed_array<Coord, 8>& coords); ///< Compute the center value jacobian from the input hexahedral element
    Mat33 computeJacobian(const type::fixed_array<Coord, 8>& coords, Real x, Real y, Real z); ///< Compute the exact jacobian located at the reference coords (x,y,z) \in [-1,1]^3
    Real computeElementVolume(const type::fixed_array<Coord, 8>& coords); //compute the exact volume of a hex element 
    Real rotDist(Mat33 R1, Mat33 R2);//the distance between two rotation using the "Deviation from Identity" metric; range in[0,2*sqrt(2)]
    Real d_totalVolume;   //initial total volume
    Real d_currentVolume; //total volume at current timestep
    Data<int> f_debugPlasticMethod; //for debugging, choose between various plastic method
    bool d_debugRendering; //for debugging, render debugData field as visual
    Data<sofa::type::vector<Real> > debugData; ///< debugData, for debug rendering
    //bool d_UpdateRestStatePerElement = true;
    bool needsToUpdateRestMesh = false; //needs to do the plasticity update
    

    /// Fracture Related Members
    sofa::component::topology::HexahedronSetTopologyModifier* _hexModifier;
    type::vector<Index> _elemsToBeFractured;
    bool _enabledFracture;

    /// container that stotes all requires information for each hexahedron
    core::topology::HexahedronData<sofa::type::vector<HexahedronInformation> > hexahedronInfo;

    /** Method to create @sa HexahedronInformation when a new hexahedron is created.
    * Will be set as creation callback in the HexahedronData @sa hexahedronInfo
    */
    void createHexahedronInformation(Index, HexahedronInformation& t, const core::topology::BaseMeshTopology::Hexahedron&,
        const sofa::type::vector<Index>&, const sofa::type::vector<double>&);

    /*class HexHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Hexahedron, sofa::type::vector<HexahedronInformation> >
    {
    public:
        typedef typename HexahedralElastoplasticFEMForceField<DataTypes>::HexahedronInformation HexahedronInformation;

        HexHandler(HexahedralElastoplasticFEMForceField<DataTypes>* ff, topology::HexahedronData<sofa::type::vector<HexahedronInformation> >* data)
            :topology::TopologyDataHandler<core::topology::BaseMeshTopology::Hexahedron, sofa::type::vector<HexahedronInformation> >(data)
            , ff(ff)
        {
        }

        void applyCreateFunction(Index, HexahedronInformation& t, const core::topology::BaseMeshTopology::Hexahedron&,
            const sofa::type::vector<Index>&, const sofa::type::vector<double>&);
    protected:
        HexahedralElastoplasticFEMForceField<DataTypes>* ff;
    };*/

protected:
    core::topology::BaseMeshTopology* _topology; ///Use BaseMeshTopology to take care of grid topology as well
    type::vector<unsigned int> _fixedIndices; ///needs to exclude the constrainted nodes from plasticity updates
    //HexHandler* hexahedronHandler;

//    type::Mat<8,3,int> _coef; ///< coef of each vertices to compute the strain stress matrix
};

#if  !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALELASTOPLASTICFEMFORCEFIELD_CPP)
extern template class SOFA_SOFAGENERALSIMPLEFEM_API HexahedralElastoplasticFEMForceField<defaulttype::Vec3Types>;

#endif

} // sofa::component::forcefield
