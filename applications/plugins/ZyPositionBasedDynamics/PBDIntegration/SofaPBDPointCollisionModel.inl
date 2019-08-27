#include "SofaPBDPointCollisionModel.h"

using namespace sofa::simulation::PBDSimulation;

template<class DataTypes>
inline TPBDPoint<DataTypes>::TPBDPoint(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{
    msg_info("TPBDPoint") << "model-index -- TPBDPoint(" << (model ? model->getName() : "NULL") << ", " << index << ")";
}

template<class DataTypes>
inline TPBDPoint<DataTypes>::TPBDPoint(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
    msg_info("TPBDPoint") << "TCollisionElementIterator -- TPBDPoint(" << (model ? model->getName() : "NULL") << ", " << index << ")";
}

template<class DataTypes>
const typename DataTypes::Coord TPBDPoint<DataTypes>::p() const
{
    //msg_info("TPBDPoint") << "p() -- " << (this->model ? model->getName() : "NULL");
    sofa::defaulttype::Vec3 pt = this->model->getCoord(this->index);
    //msg_info("TPBDPoint") << "Retrieved: " << pt;
    return pt;
}

// PBD doesn't have SOFA's equivalent of "free motion" data?
template<class DataTypes>
const typename DataTypes::Coord TPBDPoint<DataTypes>::pFree() const
{
    return p();
}

template<class DataTypes>
const typename DataTypes::Deriv TPBDPoint<DataTypes>::v() const
{
    return this->model->getDeriv(this->index);
}

// TODO: Vertex normals - available/needed?
template<class DataTypes>
typename DataTypes::Deriv TPBDPoint<DataTypes>::n() const
{
    return Deriv();
}

// PBD doesn't have SOFA's equivalent of "free motion" data?
template<class DataTypes>
bool TPBDPoint<DataTypes>::hasFreePosition() const
{
    return false;
}

template<class DataTypes>
bool TPBDPoint<DataTypes>::activated(core::CollisionModel *cm) const
{
    return this->model->myActiver->activePoint(this->index, cm);
}
