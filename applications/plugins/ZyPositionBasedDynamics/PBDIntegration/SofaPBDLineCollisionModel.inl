#include "SofaPBDLineCollisionModel.h"

#include "PBDMain/SofaPBDSimulation.h"

using namespace sofa::simulation::PBDSimulation;

template<class DataTypes>
TPBDLine<DataTypes>::TPBDLine(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{
    msg_info("TPBDLine") << "model-index -- TPBDLine(" << (model ? model->getName() : "NULL") << ", " << index << ")";
}

template<class DataTypes>
TPBDLine<DataTypes>::TPBDLine(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
    msg_info("TPBDLine") << "TCollisionElementIterator -- TPBDLine(" << (model ? model->getName() : "NULL") << ", " << index << ")";
}

template<class DataTypes>
int TPBDLine<DataTypes>::i1() const
{
    if (this->model->usesPBDRigidBody() && !this->model->usesPBDLineModel())
    {
        msg_info("TPBDLine") << "Use SofaPBDRigidBody.";
        const PBDRigidBodyGeometry& pbdRBGeom = this->model->getPBDRigidBody()->getGeometry();
        const Utilities::PBDIndexedFaceMesh::Edges& edges = pbdRBGeom.getMesh().getEdges();

        if (this->index < edges.size())
            return edges[this->index].m_vert[0];

        return -1;
    }

    if (!this->model->usesPBDRigidBody() && this->model->usesPBDLineModel())
    {
        msg_info("TPBDLine") << "Use SofaPBDLineModel.";
        PBDLineModel::Edges& lm_edges = this->model->getPBDLineModel()->getEdges();

        if (this->index < lm_edges.size())
            return lm_edges[this->index].m_vert[0];

        return -1;
    }

    return -1;
}

template<class DataTypes>
int TPBDLine<DataTypes>::i2() const
{
    if (this->model->usesPBDRigidBody() && !this->model->usesPBDLineModel())
    {
        msg_info("TPBDLine") << "Use SofaPBDRigidBody.";
        const PBDRigidBodyGeometry& pbdRBGeom = this->model->getPBDRigidBody()->getGeometry();
        const Utilities::PBDIndexedFaceMesh::Edges& edges = pbdRBGeom.getMesh().getEdges();

        if (this->index < edges.size())
            return edges[this->index].m_vert[1];

        return -1;
    }

    if (!this->model->usesPBDRigidBody() && this->model->usesPBDLineModel())
    {
        msg_info("TPBDLine") << "Use SofaPBDLineModel.";
        PBDLineModel::Edges& lm_edges = this->model->getPBDLineModel()->getEdges();

        if (this->index < lm_edges.size())
            return lm_edges[this->index].m_vert[1];

        return -1;
    }

    return -1;
}

template<class DataTypes>
const typename DataTypes::Coord TPBDLine<DataTypes>::p1() const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (this->model->usesPBDRigidBody() && !this->model->usesPBDLineModel())
    {
        const PBDRigidBodyGeometry& pbdRBGeom = this->model->getPBDRigidBody()->getGeometry();
        const Utilities::PBDIndexedFaceMesh::Edges& edges = pbdRBGeom.getMesh().getEdges();

        if (this->index < edges.size())
        {
            if (edges[this->index].m_vert[0] < this->model->getPBDRigidBody()->getGeometry().getVertexData().size())
            {
                const Vector3r& pt = this->model->getPBDRigidBody()->getGeometry().getVertexData().getPosition(edges[this->index].m_vert[0]);
                return sofa::defaulttype::Vector3(pt[0], pt[1], pt[2]);
            }
            return zeroVec;
        }

        return zeroVec;
    }

    if (!this->model->usesPBDRigidBody() && this->model->usesPBDLineModel())
    {
        PBDLineModel::Edges& lm_edges = this->model->getPBDLineModel()->getEdges();

        if (this->index < lm_edges.size())
        {
            PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
            const PBDParticleData &pd = model->getParticles();
            if (lm_edges[this->index].m_vert[0] < this->model->getPBDLineModel()->getNumPoints())
            {
                const Vector3r& pt = pd.getPosition(lm_edges[this->index].m_vert[0]);
                return sofa::defaulttype::Vector3(pt[0], pt[1], pt[2]);
            }

            return zeroVec;
        }

        return zeroVec;
    }

    return zeroVec;
}

template<class DataTypes>
const typename DataTypes::Coord TPBDLine<DataTypes>::p2() const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (this->model->usesPBDRigidBody() && !this->model->usesPBDLineModel())
    {
        const PBDRigidBodyGeometry& pbdRBGeom = this->model->getPBDRigidBody()->getGeometry();
        const Utilities::PBDIndexedFaceMesh::Edges& edges = pbdRBGeom.getMesh().getEdges();

        if (this->index < edges.size())
        {
            if (edges[this->index].m_vert[1] < this->model->getPBDRigidBody()->getGeometry().getVertexData().size())
            {
                const Vector3r& pt = this->model->getPBDRigidBody()->getGeometry().getVertexData().getPosition(edges[this->index].m_vert[1]);
                return sofa::defaulttype::Vector3(pt[0], pt[1], pt[2]);
            }

            return zeroVec;
        }

        return zeroVec;
    }

    if (!this->model->usesPBDRigidBody() && this->model->usesPBDLineModel())
    {
        PBDLineModel::Edges& lm_edges = this->model->getPBDLineModel()->getEdges();

        if (this->index < lm_edges.size())
        {
            PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
            const PBDParticleData &pd = model->getParticles();
            if (lm_edges[this->index].m_vert[1] < this->model->getPBDLineModel()->getNumPoints())
            {
                const Vector3r& pt = pd.getPosition(lm_edges[this->index].m_vert[1]);
                return sofa::defaulttype::Vector3(pt[0], pt[1], pt[2]);
            }

            return zeroVec;
        }

        return zeroVec;
    }

    return zeroVec;
}

template<class DataTypes>
const typename DataTypes::Coord TPBDLine<DataTypes>::p1Free() const
{
    return p1();
}

template<class DataTypes>
const typename DataTypes::Coord TPBDLine<DataTypes>::p2Free() const
{
    return p2();
}

template<class DataTypes>
const typename DataTypes::Deriv TPBDLine<DataTypes>::v1() const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (this->model->usesPBDRigidBody() && !this->model->usesPBDLineModel())
    {
        const PBDRigidBodyGeometry& pbdRBGeom = this->model->getPBDRigidBody()->getGeometry();
        const Utilities::PBDIndexedFaceMesh::Edges& edges = pbdRBGeom.getMesh().getEdges();

        if (this->index < edges.size())
        {
            Vector3r tr_v = this->model->getPBDRigidBody()->getVelocity();
            Vector3r rot_v =this->model->getPBDRigidBody()->getAngularVelocity();

            Vector3r rb_pos = this->model->getPBDRigidBody()->getPosition();
            Vector3r pt = this->model->getPBDRigidBody()->getGeometry().getVertexData().getPosition(edges[this->index].m_vert[0]);

            Vector3r pt_in_rb_pos = rb_pos - pt;

            sofa::defaulttype::Vec3 pt_in_rb(pt_in_rb_pos[0], pt_in_rb_pos[1], pt_in_rb_pos[2]);

            sofa::defaulttype::Vec3 tr_vel(tr_v[0], tr_v[1], tr_v[2]);
            sofa::defaulttype::Vec3 rot_vel(rot_v[0], rot_v[1], rot_v[2]);

            sofa::defaulttype::Vec3 pt_in_rb_vel = tr_vel + rot_vel.cross(pt_in_rb);

            return pt_in_rb_vel;
        }

        return zeroVec;
    }

    if (!this->model->usesPBDRigidBody() && this->model->usesPBDLineModel())
    {
        PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
        const PBDParticleData &pd = model->getParticles();

        PBDLineModel::Edges& lm_edges = this->model->getPBDLineModel()->getEdges();
        if (this->index > lm_edges.size())
            return zeroVec;

        if (this->index == lm_edges.size())
        {
            const PBDLineModel::OrientedEdge& edge = lm_edges.back();
            const Vector3r& lpt = pd.getVelocity(edge.m_vert[0]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
        else
        {
            const PBDLineModel::OrientedEdge& edge = lm_edges[this->index];
            const Vector3r& lpt = pd.getVelocity(edge.m_vert[0]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
    }

    return zeroVec;
}

template<class DataTypes>
const typename DataTypes::Deriv TPBDLine<DataTypes>::v2() const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (this->model->usesPBDRigidBody() && !this->model->usesPBDLineModel())
    {
        const PBDRigidBodyGeometry& pbdRBGeom = this->model->getPBDRigidBody()->getGeometry();
        const Utilities::PBDIndexedFaceMesh::Edges& edges = pbdRBGeom.getMesh().getEdges();

        if (this->index < edges.size())
        {
            Vector3r tr_v = this->model->getPBDRigidBody()->getVelocity();
            Vector3r rot_v =this->model->getPBDRigidBody()->getAngularVelocity();

            Vector3r rb_pos = this->model->getPBDRigidBody()->getPosition();
            Vector3r pt = this->model->getPBDRigidBody()->getGeometry().getVertexData().getPosition(edges[this->index].m_vert[0]);

            Vector3r pt_in_rb_pos = rb_pos - pt;

            sofa::defaulttype::Vec3 pt_in_rb(pt_in_rb_pos[0], pt_in_rb_pos[1], pt_in_rb_pos[2]);

            sofa::defaulttype::Vec3 tr_vel(tr_v[0], tr_v[1], tr_v[2]);
            sofa::defaulttype::Vec3 rot_vel(rot_v[0], rot_v[1], rot_v[2]);

            sofa::defaulttype::Vec3 pt_in_rb_vel = tr_vel + rot_vel.cross(pt_in_rb);

            return pt_in_rb_vel;
        }

        return zeroVec;
    }

    if (!this->model->usesPBDRigidBody() && this->model->usesPBDLineModel())
    {
        PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
        const PBDParticleData &pd = model->getParticles();

        PBDLineModel::Edges& lm_edges = this->model->getPBDLineModel()->getEdges();
        if (this->index > lm_edges.size())
            return zeroVec;

        if (this->index == lm_edges.size())
        {
            const PBDLineModel::OrientedEdge& edge = lm_edges.back();
            const Vector3r& lpt = pd.getVelocity(edge.m_vert[0]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
        else
        {
            const PBDLineModel::OrientedEdge& edge = lm_edges[this->index];
            const Vector3r& lpt = pd.getVelocity(edge.m_vert[0]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
    }

    return zeroVec;
}

template<class DataTypes>
int TPBDLine<DataTypes>::flags() const
{
    return this->model->getLineFlags(this->index);
}

template<class DataTypes>
bool TPBDLine<DataTypes>::hasFreePosition() const
{
    return false;
}

template<class DataTypes>
inline bool TPBDLine<DataTypes>::activated(core::CollisionModel *cm) const
{
    return this->model->myActiver->activeLine(this->index, cm);
}
