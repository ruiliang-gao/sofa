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

#include <map>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

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

/**
Bending springs added between vertices of triangles sharing a common edge.
The springs connect the vertices not belonging to the common edge. It compresses when the surface bends along the common edge.


	@author The SOFA team </www.sofa-framework.org>
*/
template<class DataTypes>
class TriangularBendingSprings : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularBendingSprings, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    //typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    //typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;

    using Index = sofa::Index;

    Data<double> f_ks; ///< uniform stiffness for the all springs
    Data<double> f_kd; ///< uniform damping for the all springs

    /// Link to be set to the topology container in the component graph.
    SingleLink<TriangularBendingSprings<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:

    //Data<double> ks;
    //Data<double> kd;

    class EdgeInformation
    {
    public:
        Mat DfDx; /// the edge stiffness matrix

        int     m1, m2;  /// the two extremities of the spring: masses m1 and m2

        double  ks;      /// spring stiffness (initialized to the default value)
        double  kd;      /// damping factor (initialized to the default value)

        double  restlength; /// rest length of the spring

        bool is_activated;

        bool is_initialized;

        EdgeInformation(int m1=0, int m2=0, /* double ks=getKs(), double kd=getKd(), */ double restlength=0.0, bool is_activated=false, bool is_initialized=false)
            : m1(m1), m2(m2), /* ks(ks), kd(kd), */ restlength(restlength), is_activated(is_activated), is_initialized(is_initialized)
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

    sofa::component::topology::EdgeData<helper::vector<EdgeInformation> > edgeInfo; ///< Internal edge data

    class TriangularBSEdgeHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, helper::vector<EdgeInformation> >
    {
    public:
        typedef typename TriangularBendingSprings<DataTypes>::EdgeInformation EdgeInformation;
        TriangularBSEdgeHandler(TriangularBendingSprings<DataTypes>* _ff, topology::EdgeData<helper::vector<EdgeInformation> >* _data)
            : topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, sofa::helper::vector<EdgeInformation> >(_data), ff(_ff) {}

        void applyCreateFunction(Index edgeIndex,
                EdgeInformation &ei,
                const core::topology::BaseMeshTopology::Edge& ,  const sofa::helper::vector< Index > &,
                const sofa::helper::vector< double >&);

        void applyTriangleCreation(const helper::vector<Index> &triangleAdded,
                const helper::vector<core::topology::BaseMeshTopology::Triangle> & ,
                const helper::vector<helper::vector<Index> > & ,
                const helper::vector<helper::vector<double> > &);

        void applyTriangleDestruction(const helper::vector<Index> &triangleRemoved);

        void applyPointDestruction(const helper::vector<Index> &pointIndices);

        void applyPointRenumbering(const helper::vector<Index> &pointToRenumber);

        using topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, helper::vector<EdgeInformation> >::ApplyTopologyChange;
        /// Callback to add triangles elements.
        void ApplyTopologyChange(const core::topology::TrianglesAdded* /*event*/);
        /// Callback to remove triangles elements.
        void ApplyTopologyChange(const core::topology::TrianglesRemoved* /*event*/);

        /// Callback to remove points elements.
        void ApplyTopologyChange(const core::topology::PointsRemoved* /*event*/);
        /// Callback to renumbering on points elements.
        void ApplyTopologyChange(const core::topology::PointsRenumbering* /*event*/);

    protected:
        TriangularBendingSprings<DataTypes>* ff;
    };    

    bool updateMatrix;
    TriangularBendingSprings(/*double _ks, double _kd*/);
    //TriangularBendingSprings(); //MechanicalState<DataTypes> *mm1 = nullptr, MechanicalState<DataTypes> *mm2 = nullptr);

    virtual ~TriangularBendingSprings();
public:
    /// Searches triangle topology and creates the bending springs
    void init() override;

    void reinit() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& d_x) const override;

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

    void draw(const core::visual::VisualParams* vparams) override;

protected:

    sofa::component::topology::EdgeData<helper::vector<EdgeInformation> > &getEdgeInfo() {return edgeInfo;}

    TriangularBSEdgeHandler* edgeHandler;

    SReal m_potentialEnergy;

    sofa::core::topology::BaseMeshTopology* m_topology;

    //public:
    //Data<double> ks;
    //Data<double> kd;

};

#if  !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_CPP)
extern template class SOFA_SOFAGENERALDEFORMABLE_API TriangularBendingSprings<defaulttype::Vec3Types>;

#endif // !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_CPP)


} // namespace sofa::component::forcefield
