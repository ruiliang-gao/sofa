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

#include <SofaDeformable/SpringForceField.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Mat.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>


namespace sofa::component::interactionforcefield
{

template<class DataTypes>
class VectorSpringForceField: public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(VectorSpringForceField, DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField, DataTypes));

    using Index = sofa::Index;
    typedef typename core::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    struct Spring
    {
        Real  ks;          ///< spring stiffness
        Real  kd;          ///< damping factor
        Deriv restVector;  ///< rest vector of the spring
        float restLength2; /// square of rest length	

        Spring(Real _ks, Real _kd, Deriv _rl) : ks(_ks), kd(_kd), restVector(_rl)
        {
            restLength2 = _rl[0] * _rl[0] + _rl[1] * _rl[1] + _rl[2] * _rl[2];
        }
        Spring() : ks(1.0), kd(1.0)
        {
        }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const Spring& /*s*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, Spring& /*s*/ )
        {
            return in;
        }
    };
protected:

    SReal m_potentialEnergy;
    /// true if the springs are initialized from the topology
    bool useTopology;
    bool usingMask;
    /// indices in case we don't use the topology
    sofa::type::vector<core::topology::BaseMeshTopology::Edge> edgeArray;

    float m_thresTearing2 = 0; // squared threshold of the deform ratio for tearing the spring, init=0

    void resizeArray(std::size_t n);


public:

    /// where the springs information are stored
    sofa::component::topology::EdgeData<sofa::type::vector<Spring> > springArray;

    class EdgeDataHandler : public sofa::component::topology::TopologyDataHandler< core::topology::BaseMeshTopology::Edge, sofa::type::vector<Spring> >
    {
    public:
        typedef typename VectorSpringForceField<DataTypes>::Spring Spring;
        EdgeDataHandler(VectorSpringForceField<DataTypes>* ff, topology::EdgeData<sofa::type::vector<Spring> >* data)
            :topology::TopologyDataHandler< core::topology::BaseMeshTopology::Edge,sofa::type::vector<Spring> >(data)
            ,ff(ff)
        {

        }

        void applyCreateFunction(Index, Spring &t,
                const core::topology::BaseMeshTopology::Edge &,
                const sofa::type::vector<Index> &, const sofa::type::vector<double> &);
    protected:
        VectorSpringForceField<DataTypes>* ff;

    };
    /// the filename where to load the spring information
    sofa::core::objectmodel::DataFileName m_filename;
    /// By default, assume that all edges have the same stiffness
    Data<double> m_stiffness;
    /// By default, assume that all edges have the same viscosity
    Data<double> m_viscosity;

    Data<bool> m_useTopology; ///< Activate/Desactivate topology mode of the component (springs on each edge)

    /// Link to be set to the topology container in the component graph.
    SingleLink<VectorSpringForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
    
    sofa::component::topology::EdgeSetTopologyContainer* edgeCont;
    sofa::component::topology::EdgeSetGeometryAlgorithms<DataTypes>* edgeGeo;
    sofa::component::topology::EdgeSetTopologyModifier* edgeMod;

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* m_topology;

protected:
    VectorSpringForceField();
    VectorSpringForceField(MechanicalState* _object);
    VectorSpringForceField(MechanicalState* _object1, MechanicalState* _object2);
    virtual ~VectorSpringForceField() override;

    EdgeDataHandler* edgeHandler;

public:
    bool load(const char *filename);

    core::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }

    void init() override;
    void bwdInit() override;

    void createDefaultSprings();

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 ) override;

    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2) override;

    SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&, const DataVecCoord& ) const override { return m_potentialEnergy; }

    Real getStiffness() const
    {
        return Real(m_stiffness.getValue());
    }
    const Real getViscosity() const
    {
        return Real(m_viscosity.getValue());
    }
    const topology::EdgeData<sofa::type::vector<Spring> >& getSpringArray() const
    {
        return springArray;
    }

    void draw(const core::visual::VisualParams* vparams) override;

    // -- Modifiers

    void clear(int reserve=0)
    {
        type::vector<Spring>& springArrayData = *(springArray.beginEdit());
        springArrayData.clear();
        if (reserve) springArrayData.reserve(reserve);
        springArray.endEdit();
        if(!useTopology) edgeArray.clear();
    }

    void addSpring(int m1, int m2, SReal ks, SReal kd, Coord restVector);
    void addSpring(int m1, int m2, SReal ks, SReal kd, Coord restVector, float thresTearing);

    /// forward declaration of the loader class used to read spring information from file
    class Loader;
    friend class Loader;

    void updateForceMask() override;

};

#if  !defined(SOFA_COMPONENT_FORCEFIELD_VECTORSPRINGFORCEFIELD_CPP)
extern template class SOFA_SOFAGENERALDEFORMABLE_API VectorSpringForceField<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::interactionforcefield
