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
#include <SofaMiscCollision/CapsuleModel.h>
#include <SofaMiscCollision/RigidCapsuleModel.h>
#include <SofaMiscCollision/OBBModel.h>
#include <SofaMiscCollision/OBBContactMapper.h>
#include <SofaMiscCollision/CapsuleContactMapper.h>
#include <SofaMeshCollision/BarycentricPenalityContact.inl>
#include <SofaConstraint/FrictionContact.inl>

using namespace sofa::core::collision;

namespace sofa::component::collision
{

Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleTriangleContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleLinePenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleCapsulePenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleSpherePenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleOBBPenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleTriangleContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleLinePenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleRigidCapsulePenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleRigidCapsulePenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleSpherePenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel> > RigidCapsuleRigidSpherePenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleOBBPenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleCylinderPenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleCylinderPenalityContactClass("PenalityContactForceField", true);

Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleCapsuleFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleTriangleFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleSphereFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > CapsuleRigidSphereFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleRigidCapsuleFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleRigidCapsuleFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleTriangleFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleSphereFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel> > RigidCapsuleRigidSphereFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleOBBFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleOBBFrictionContactClass("FrictionContactConstraint", true);

template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>>;

template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel>;
template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel>;
template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;

} // namespace sofa::component::collision
