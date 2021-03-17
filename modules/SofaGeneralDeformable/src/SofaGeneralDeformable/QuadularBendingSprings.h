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

#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/fixed_array.h>

#include <map>
#include <set>

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

/**
Bending springs added between vertices of quads sharing a common edge.
The springs connect the vertices not belonging to the common edge. It compresses when the surface bends along the common edge.
*/
template<class DataTypes>
class QuadularBendingSprings : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(QuadularBendingSprings,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    //typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;

    using Index = sofa::Index;

    QuadularBendingSprings();

    ~QuadularBendingSprings();

protected:

    class EdgeInformation
    {
    public:
        Mat DfDx; /// the edge stiffness matrix

        int     m1, m2;  /// the two extremities of the first spring: masses m1 and m2
        int     m3, m4;  /// the two extremities of the second spring: masses m3 and m4

        double  ks;      /// spring stiffness (initialized to the default value)
        double  kd;      /// damping factor (initialized to the default value)

        double  restlength1; /// rest length of the first spring
        double  restlength2; /// rest length of the second spring

        bool is_activated;

        bool is_initialized;

        EdgeInformation(int m1=0, int m2=0, int m3=0, int m4=0, double restlength1=0.0, double restlength2=0.0, bool is_activated=false, bool is_initialized=false)
            : m1(m1), m2(m2), m3(m3), m4(m4), restlength1(restlength1), restlength2(restlength2), is_activated(is_activated), is_initialized(is_initialized)
        {
        }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeInformation& /*ei*/ )
        {
            return in;
        }
    };



public:
    /// Searches quad topology and creates the bending springs
    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;

    SReal getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* d_x */) const override;

    virtual double getKs() const { return f_ks.getValue();}
    virtual double getKd() const { return f_kd.getValue();}

    void setKs(const double ks)
    {
        f_ks.setValue((double)ks);
    }
    void setKd(const double kd)
    {
        f_kd.setValue((double)kd);
    }

    // -- VisualModel interface
    void draw(const core::visual::VisualParams* vparams) override;
    void initTextures() { }
    void update() { }

    sofa::component::topology::EdgeData<sofa::helper::vector<EdgeInformation> > &getEdgeInfo() {return edgeInfo;}


    class EdgeBSHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, helper::vector<EdgeInformation> >
    {
    public:
        typedef typename QuadularBendingSprings<DataTypes>::EdgeInformation EdgeInformation;

        EdgeBSHandler(QuadularBendingSprings<DataTypes>* ff, topology::EdgeData<sofa::helper::vector<EdgeInformation> >* data )
            :topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, sofa::helper::vector<EdgeInformation> >(data)
            ,ff(ff)
        {
        }

        void applyCreateFunction(Index edgeIndex, EdgeInformation& ei,
                const core::topology::BaseMeshTopology::Edge &,
                const sofa::helper::vector< Index > &,
                const sofa::helper::vector< double > &);

        void applyQuadCreation(const sofa::helper::vector<Index> & quadAdded,
                const sofa::helper::vector<core::topology::BaseMeshTopology::Quad> &,
                const sofa::helper::vector<sofa::helper::vector<Index> > &,
                const sofa::helper::vector<sofa::helper::vector<double> > &);

        void applyQuadDestruction(const sofa::helper::vector<Index> & quadRemoved);

        using topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, helper::vector<EdgeInformation> >::ApplyTopologyChange;

        /// Callback to add quads elements.
        void ApplyTopologyChange(const core::topology::QuadsAdded* /*event*/);
        /// Callback to remove quads elements.
        void ApplyTopologyChange(const core::topology::QuadsRemoved* /*event*/);

        void applyPointDestruction(const sofa::helper::vector<Index> &pointIndices);

        void applyPointRenumbering(const sofa::helper::vector<Index> &pointToRenumber);

        /// Callback to remove points elements.
        void ApplyTopologyChange(const core::topology::PointsRemoved* /*event*/);
        /// Callback to renumbering on points elements.
        void ApplyTopologyChange(const core::topology::PointsRenumbering* /*event*/);

    protected:
        QuadularBendingSprings<DataTypes>* ff;
    };

    Data<double> f_ks; ///< uniform stiffness for the all springs
    Data<double> f_kd; ///< uniform damping for the all springs

    /// Link to be set to the topology container in the component graph.
    SingleLink<QuadularBendingSprings<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    sofa::component::topology::EdgeData<sofa::helper::vector<EdgeInformation> > edgeInfo; ///< Internal edge data

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* m_topology;

    /// Handler for subset Data
    EdgeBSHandler* edgeHandler;

    bool updateMatrix;
    SReal m_potentialEnergy;
};


#if  !defined(SOFA_COMPONENT_FORCEFIELD_QUADULARBENDINGSPRINGS_CPP)

extern template class SOFA_SOFAGENERALDEFORMABLE_API QuadularBendingSprings<sofa::defaulttype::Vec3Types>;



#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_QUADULARBENDINGSPRINGS_CPP)

} // namespace sofa::component::forcefield
