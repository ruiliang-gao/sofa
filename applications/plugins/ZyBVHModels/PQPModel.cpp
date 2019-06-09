#include "initBVHModelsPlugin.h"
#include "PQPModel.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>

using namespace sofa::defaulttype;
using namespace sofa::component::collision;
using namespace sofa;

SOFA_DECL_CLASS(PQPCollisionModel)

#define PQPMODELCOLLISIONNODE_DEBUG

PQPCollisionModelNode::PQPCollisionModelNode(PQPCollisionModel<Vec3Types> *model, int index):
    TCollisionElementIterator<PQPCollisionModel<Vec3Types> >(model, index)
{
    PQPCollisionModel<Vec3Types>* obbTreeModel = static_cast<PQPCollisionModel<Vec3Types>* >(model);

    if (index < obbTreeModel->getNumBVs())
    {
#ifdef PQPMODELCOLLISIONNODE_DEBUG
        std::cout << " Retrieved OBB from model index " << index << std::endl;
#endif
        m_obb = PQPTreeNode(obbTreeModel->getChild(index));
    }
    else
    {
#ifdef PQPMODELCOLLISIONNODE_DEBUG
        std::cout << "No valid OBB set: index " << index << " > " << obbTreeModel->getNumBVs() << std::endl;
#endif
        m_obb = PQPTreeNode(NULL);
    }
}

PQPCollisionModelNode::PQPCollisionModelNode(core::CollisionElementIterator &i):
    TCollisionElementIterator<PQPCollisionModel<Vec3Types> >(static_cast<PQPCollisionModel<Vec3Types> *>(i.getCollisionModel()), i.getIndex())
{
#ifdef PQPMODELCOLLISIONNODE_DEBUG
    std::cout << "PQPCollisionModelNode::PQPCollisionModelNode(" << i.getCollisionModel()->getName() << "," << i.getIndex() << "), constructed via CollisionElementIterator" << std::endl;
#endif
    PQPCollisionModel<Vec3Types>* obbTreeModel = static_cast<PQPCollisionModel<Vec3Types>* >(i.getCollisionModel());

    int triangleId = i.getIndex();
    if (triangleId < obbTreeModel->getNumBVs())
    {
#ifdef PQPMODELCOLLISIONNODE_DEBUG
        std::cout << " Retrieved OBB from model index " << triangleId << std::endl;
#endif
        m_obb = PQPTreeNode(obbTreeModel->getChild(triangleId));
    }
    else
    {
#ifdef PQPMODELCOLLISIONNODE_DEBUG
        std::cout << "No valid OBB set: index " << triangleId << " > " << obbTreeModel->getNumBVs() << std::endl;
#endif
        m_obb = PQPTreeNode(NULL);
    }
}

int PQPCollisionModelClass = sofa::core::RegisterObject("Collision model wrapping the PQP library")
#ifndef SOFA_FLOAT
        .add< PQPCollisionModel<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< PQPCollisionModel<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_BVHMODELSPLUGIN_API PQPCollisionModel<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_BVHMODELSPLUGIN_API PQPCollisionModel<Vec3fTypes>;
#endif //SOFA_DOUBLE
