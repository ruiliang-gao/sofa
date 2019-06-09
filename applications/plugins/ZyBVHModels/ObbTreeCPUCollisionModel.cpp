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
 * ObbTreeCPUCollisionModel.cpp
 *
 *  Created on: 15.05.2014
 *      Author: faichele
 */
#define OBBTREECPU_COLLISIONMODEL_CPP


#include <initBVHModelsPlugin.h>
#include "ObbTreeCPUCollisionModel.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>

SOFA_DECL_CLASS(ObbTreeCPUCollisionModel)

using namespace sofa::defaulttype;
using namespace sofa::component::collision;
using namespace sofa;

ObbTreeCPUCollisionModelNode::ObbTreeCPUCollisionModelNode(ObbTreeCPUCollisionModel<Vec3Types> *model, int index):
    TCollisionElementIterator<ObbTreeCPUCollisionModel<Vec3Types> >(model, index), m_obb(ObbTreeCPUNode())
{
    /*if (_obb)
        std::cout << " Retrieved OBB from model: At " << _obb->center() << ", half-extents: " << _obb->halfExtents() << ", depth: " << _obb->depth() << std::endl;*/
}

ObbTreeCPUCollisionModelNode::ObbTreeCPUCollisionModelNode(core::CollisionElementIterator &i):
    TCollisionElementIterator<ObbTreeCPUCollisionModel<Vec3Types> >(static_cast<ObbTreeCPUCollisionModel<Vec3Types> *>(i.getCollisionModel()), i.getIndex())
{
    //std::cout << "LGCObbBox::LGCObbBox(" << i.getCollisionModel()->getName() << "," << i.getIndex() << "), constructed via CollisionElementIterator" << std::endl;
    m_obb = ObbTreeCPUNode();
}

int ObbTreeCPUCollisionModelClass = sofa::core::RegisterObject("Collision model wrapping the PQP library")
#ifndef SOFA_FLOAT
        .add< ObbTreeCPUCollisionModel<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< ObbTreeCPUCollisionModel<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_BVHMODELSPLUGIN_API ObbTreeCPUCollisionModel<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_BVHMODELSPLUGIN_API ObbTreeCPUCollisionModel<Vec3fTypes>;
#endif //SOFA_DOUBLE


