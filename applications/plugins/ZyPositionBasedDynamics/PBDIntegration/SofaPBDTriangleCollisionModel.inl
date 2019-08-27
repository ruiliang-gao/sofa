#include "SofaPBDTriangleCollisionModel.h"

using namespace sofa::simulation::PBDSimulation;

template<class DataTypes>
inline TPBDTriangle<DataTypes>::TPBDTriangle(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{
    msg_info("TPBDTriangle") << "model-index -- TPBDTriangle(" << (model ? model->getName() : "NULL") << ", " << index << ")";
}

template<class DataTypes>
TPBDTriangle<DataTypes>::TPBDTriangle(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{

}

template<class DataTypes>
TPBDTriangle<DataTypes>::TPBDTriangle(ParentModel* model, int index, helper::ReadAccessor<typename DataTypes::VecCoord>& /*x*/)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{
        msg_info("TPBDTriangle") << "TCollisionElementIterator -- TPBDTriangle(" << (model ? model->getName() : "NULL") << ", " << index << ")";
}

template<class DataTypes>
const typename DataTypes::Coord TPBDTriangle<DataTypes>::p1() const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (this->model->getPBDRigidBody())
    {
        if (this->index < this->model->size)
        {
            int v1Idx = this->model->getVertex1Idx(this->index);
            Vector3r pt1 = this->model->getPBDRigidBody()->getGeometry().getVertexData().getPosition(v1Idx);
            return sofa::defaulttype::Vec3(pt1[0], pt1[1], pt1[2]);
        }
        return zeroVec;
    }

    return zeroVec;
}

template<class DataTypes>
const typename DataTypes::Coord TPBDTriangle<DataTypes>::p2() const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (this->model->getPBDRigidBody())
    {
        if (this->index < this->model->size)
        {
            int v2Idx = this->model->getVertex2Idx(this->index);
            Vector3r pt2 = this->model->getPBDRigidBody()->getGeometry().getVertexData().getPosition(v2Idx);
            return sofa::defaulttype::Vec3(pt2[0], pt2[1], pt2[2]);
        }
        return zeroVec;
    }

    return zeroVec;
}

template<class DataTypes>
const typename DataTypes::Coord TPBDTriangle<DataTypes>::p3() const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (this->model->getPBDRigidBody())
    {
        if (this->index < this->model->size)
        {
            int v3Idx = this->model->getVertex3Idx(this->index);
            Vector3r pt3 = this->model->getPBDRigidBody()->getGeometry().getVertexData().getPosition(v3Idx);
            return sofa::defaulttype::Vec3(pt3[0], pt3[1], pt3[2]);
        }
        return zeroVec;
    }

    return zeroVec;
}

template<class DataTypes>
const typename DataTypes::Coord TPBDTriangle<DataTypes>::p1Free() const
{
    return p1();
}

template<class DataTypes>
const typename DataTypes::Coord TPBDTriangle<DataTypes>::p2Free() const
{
    return p2();
}

template<class DataTypes>
const typename DataTypes::Coord TPBDTriangle<DataTypes>::p3Free() const
{
    return p3();
}

template<class DataTypes>
int TPBDTriangle<DataTypes>::p1Index() const
{
    if (this->index < this->model->size)
    {
        return this->model->getVertex1Idx(this->index);
    }

    return -1;
}

template<class DataTypes>
int TPBDTriangle<DataTypes>::p2Index() const
{
    if (this->index < this->model->size)
    {
        return this->model->getVertex2Idx(this->index);
    }

    return -1;
}

template<class DataTypes>
int TPBDTriangle<DataTypes>::p3Index() const
{
    if (this->index < this->model->size)
    {
        return this->model->getVertex3Idx(this->index);
    }

    return -1;
}

template<class DataTypes>
const typename DataTypes::Deriv TPBDTriangle<DataTypes>::v1() const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (this->model->getPBDRigidBody())
    {
        if (this->index >= this->model->getPBDRigidBody()->getGeometry().getVertexData().size())
            return zeroVec;

        Vector3r tr_v = this->model->getPBDRigidBody()->getVelocity();
        Vector3r rot_v = this->model->getPBDRigidBody()->getAngularVelocity();

        Vector3r rb_pos = this->model->getPBDRigidBody()->getPosition();
        Vector3r pt = this->model->getPBDRigidBody()->getGeometry().getVertexData().getPosition(this->model->getVertex1Idx(this->index));

        Vector3r pt_in_rb_pos = rb_pos - pt;

        Vec3 pt_in_rb(pt_in_rb_pos[0], pt_in_rb_pos[1], pt_in_rb_pos[2]);

        Vec3 tr_vel(tr_v[0], tr_v[1], tr_v[2]);
        Vec3 rot_vel(rot_v[0], rot_v[1], rot_v[2]);

        Vec3 pt_in_rb_vel = tr_vel + rot_vel.cross(pt_in_rb);

        return pt_in_rb_vel;
    }

    return zeroVec;
}

template<class DataTypes>
const typename DataTypes::Deriv TPBDTriangle<DataTypes>::v2() const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (this->model->getPBDRigidBody())
    {
        if (this->index >= this->model->getPBDRigidBody()->getGeometry().getVertexData().size())
            return zeroVec;

        Vector3r tr_v = this->model->getPBDRigidBody()->getVelocity();
        Vector3r rot_v = this->model->getPBDRigidBody()->getAngularVelocity();

        Vector3r rb_pos = this->model->getPBDRigidBody()->getPosition();
        Vector3r pt = this->model->getPBDRigidBody()->getGeometry().getVertexData().getPosition(this->model->getVertex2Idx(this->index));

        Vector3r pt_in_rb_pos = rb_pos - pt;

        Vec3 pt_in_rb(pt_in_rb_pos[0], pt_in_rb_pos[1], pt_in_rb_pos[2]);

        Vec3 tr_vel(tr_v[0], tr_v[1], tr_v[2]);
        Vec3 rot_vel(rot_v[0], rot_v[1], rot_v[2]);

        Vec3 pt_in_rb_vel = tr_vel + rot_vel.cross(pt_in_rb);

        return pt_in_rb_vel;
    }

    return zeroVec;
}

template<class DataTypes>
const typename DataTypes::Deriv TPBDTriangle<DataTypes>::v3() const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (this->model->getPBDRigidBody())
    {
        if (this->index >= this->model->getPBDRigidBody()->getGeometry().getVertexData().size())
            return zeroVec;

        Vector3r tr_v = this->model->getPBDRigidBody()->getVelocity();
        Vector3r rot_v = this->model->getPBDRigidBody()->getAngularVelocity();

        Vector3r rb_pos = this->model->getPBDRigidBody()->getPosition();
        Vector3r pt = this->model->getPBDRigidBody()->getGeometry().getVertexData().getPosition(this->model->getVertex3Idx(this->index));

        Vector3r pt_in_rb_pos = rb_pos - pt;

        Vec3 pt_in_rb(pt_in_rb_pos[0], pt_in_rb_pos[1], pt_in_rb_pos[2]);

        Vec3 tr_vel(tr_v[0], tr_v[1], tr_v[2]);
        Vec3 rot_vel(rot_v[0], rot_v[1], rot_v[2]);

        Vec3 pt_in_rb_vel = tr_vel + rot_vel.cross(pt_in_rb);

        return pt_in_rb_vel;
    }

    return zeroVec;
}

template<class DataTypes>
const typename DataTypes::Deriv TPBDTriangle<DataTypes>::n() const
{
    const sofa::defaulttype::Vec3 pt1 = p1();
    const sofa::defaulttype::Vec3 pt2 = p2();
    const sofa::defaulttype::Vec3 pt3 = p3();

    return (pt2 - pt1).cross(pt3 - pt1).normalized();
}

template<class DataTypes>
typename DataTypes::Deriv TPBDTriangle<DataTypes>::n()
{
    sofa::defaulttype::Vec3 pt1 = p1();
    sofa::defaulttype::Vec3 pt2 = p2();
    sofa::defaulttype::Vec3 pt3 = p3();

    return (pt2 - pt1).cross(pt3 - pt1).normalized();
}

template<class DataTypes>
int TPBDTriangle<DataTypes>::flags() const
{
    return this->model->getTriangleFlags(this->index);
}

template<class DataTypes>
bool TPBDTriangle<DataTypes>::hasFreePosition() const
{
    return false;
}
