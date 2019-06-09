/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 *
 *  Created on: 04.05.2014
 *      Author: faichele
 */
#define OBBTREEGPU_COLLISIONMODEL_CPP


#include "initObbTreeGpuPlugin.h"
#include "ObbTreeGPUCollisionModel.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>

SOFA_DECL_CLASS(ObbTreeGPUCollisionModel)

using namespace sofa::defaulttype;
using namespace sofa::component::collision;
using namespace sofa;

ObbTreeGPUCollisionModelNode::ObbTreeGPUCollisionModelNode(ObbTreeGPUCollisionModel<Vec3Types> *model, int index):
    TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(model, index)
{
    ObbTreeGPUCollisionModel<Vec3Types>* obbTreeModel = static_cast<ObbTreeGPUCollisionModel<Vec3Types>* >(model);

    int triangleId = index / 15;
    if (triangleId < obbTreeModel->getPqpModel()->num_bvs)
    {
#ifdef OBBTREEGPUCOLLISIONMODELNODE_DEBUG
        std::cout << " Retrieved OBB from model index " << triangleId << std::endl;
#endif
        m_obb = ObbTreeGPUNode(obbTreeModel->getPqpModel()->child(index));
    }
    else
    {
#ifdef OBBTREEGPUCOLLISIONMODELNODE_DEBUG
        std::cout << "No valid OBB set: index " << triangleId << " > " << obbTreeModel->getPqpModel()->num_bvs << std::endl;
#endif
        m_obb = ObbTreeGPUNode(NULL);
    }
}

ObbTreeGPUCollisionModelNode::ObbTreeGPUCollisionModelNode(core::CollisionElementIterator &i):
    TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >(static_cast<ObbTreeGPUCollisionModel<Vec3Types> *>(i.getCollisionModel()), i.getIndex())
{
#ifdef OBBTREEGPUCOLLISIONMODELNODE_DEBUG
    std::cout << "ObbTreeGPUCollisionModelNode::ObbTreeGPUCollisionModelNode(" << i.getCollisionModel()->getName() << "," << i.getIndex() << "), constructed via CollisionElementIterator" << std::endl;
#endif
    ObbTreeGPUCollisionModel<Vec3Types>* obbTreeModel = static_cast<ObbTreeGPUCollisionModel<Vec3Types>* >(i.getCollisionModel());

    int triangleId = i.getIndex() / 15;
    if (triangleId < obbTreeModel->getPqpModel()->num_bvs)
    {
#ifdef OBBTREEGPUCOLLISIONMODELNODE_DEBUG
        std::cout << " Retrieved OBB from model index " << triangleId << std::endl;
#endif
        m_obb = ObbTreeGPUNode(obbTreeModel->getPqpModel()->child(triangleId));
    }
    else
    {
#ifdef OBBTREEGPUCOLLISIONMODELNODE_DEBUG
        std::cout << "No valid OBB set: index " << triangleId << " > " << obbTreeModel->getPqpModel()->num_bvs << std::endl;
#endif
        m_obb = ObbTreeGPUNode(NULL);
    }
}

int ObbTreeGPUCollisionModelClass = sofa::core::RegisterObject("Collision model wrapping the gProximity library")
#ifndef SOFA_FLOAT
        .add< ObbTreeGPUCollisionModel<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< ObbTreeGPUCollisionModel<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_OBBTREEGPUPLUGIN_API ObbTreeGPUCollisionModel<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_OBBTREEGPUPLUGIN_API ObbTreeGPUCollisionModel<Vec3fTypes>;
#endif //SOFA_DOUBLE


