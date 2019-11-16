/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/simulation/PipelineImpl.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>

#include <sofa/simulation/Node.h>

#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace simulation
{


using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::core::behavior;
using namespace sofa::core::collision;

PipelineImpl::PipelineImpl()
{
}

PipelineImpl::~PipelineImpl()
{
}

void PipelineImpl::init()
{
    msg_info("PipelineImpl") << "---------------------------------------------------------------------------";
    msg_info("PipelineImpl") << "init()";
    simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());
    if (root == NULL)
    {
        msg_info("PipelineImpl") << "Failed to convert getContext() to sofa::simulation::Node*, trying current simulation root node.";
        root = sofa::simulation::getSimulation()->getCurrentRootNode().get();
        if (root == NULL)
        {
            msg_warning("PipelineImpl") << "Failed to retrieve current simulation root node, PipelineImpl init() failed!";
            return;
        }
    }

    intersectionMethods.clear();
    root->getTreeObjects<Intersection>(&intersectionMethods);

    msg_info("PipelineImpl") << "Intersection instances found in scene: " << intersectionMethods.size();

    intersectionMethod = (intersectionMethods.empty() ? NULL : intersectionMethods[0]);

    broadPhaseDetections.clear();
    root->getTreeObjects<BroadPhaseDetection>(&broadPhaseDetections);

    msg_info("PipelineImpl") << "BroadPhaseDetection instances found in scene: " << broadPhaseDetections.size();

    broadPhaseDetection = (broadPhaseDetections.empty() ? NULL : broadPhaseDetections[0]);

    narrowPhaseDetections.clear();
    root->getTreeObjects<NarrowPhaseDetection>(&narrowPhaseDetections);

    msg_info("PipelineImpl") << "NarrowPhaseDetection instances found in scene: " << narrowPhaseDetections.size();

    narrowPhaseDetection = (narrowPhaseDetections.empty() ? NULL : narrowPhaseDetections[0]);

    contactManagers.clear();
    root->getTreeObjects<ContactManager>(&contactManagers);

    msg_info("PipelineImpl") << "ContactManager instances found in scene: " << contactManagers.size();

    contactManager = (contactManagers.empty() ? NULL : contactManagers[0]);

    groupManagers.clear();
    root->getTreeObjects<CollisionGroupManager>(&groupManagers);

    msg_info("PipelineImpl") << "CollisionGroupManager instances found in scene: " << groupManagers.size();

    groupManager = (groupManagers.empty() ? NULL : groupManagers[0]);

    if (intersectionMethod == NULL)
    {
        msg_warning(this) <<"no intersection component defined. Switching to the DiscreteIntersection component. " << msgendl
                            "To remove this warning, you can add an intersection component to your scene. " << msgendl
                            "More details on the collision pipeline can be found at "
                            "[sofadoc::Collision](https://www.sofa-framework.org/community/doc/using-sofa/specific-components/intersectionmethod/). ";
        sofa::core::objectmodel::BaseObjectDescription discreteIntersectionDesc("Default Intersection","DiscreteIntersection");
        sofa::core::objectmodel::BaseObject::SPtr obj = sofa::core::ObjectFactory::CreateObject(getContext(), &discreteIntersectionDesc);
        intersectionMethod = dynamic_cast<Intersection*>(obj.get());
    }
    msg_info("PipelineImpl") << "---------------------------------------------------------------------------";
}

void PipelineImpl::reset()
{
    computeCollisionReset();
}

void PipelineImpl::computeCollisionReset()
{
    msg_info("PipelineImpl") << "computeCollisionReset()";
    simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());

    if(root == NULL)
    {
        msg_warning("PipelineImpl") << "Simulation root node is NULL, doing nothing!";
        return;
    }

    if (broadPhaseDetection!=NULL && broadPhaseDetection->getIntersectionMethod()!=intersectionMethod)
        broadPhaseDetection->setIntersectionMethod(intersectionMethod);
    if (narrowPhaseDetection!=NULL && narrowPhaseDetection->getIntersectionMethod()!=intersectionMethod)
        narrowPhaseDetection->setIntersectionMethod(intersectionMethod);
    if (contactManager!=NULL && contactManager->getIntersectionMethod()!=intersectionMethod)
        contactManager->setIntersectionMethod(intersectionMethod);

    sofa::helper::AdvancedTimer::stepBegin("CollisionReset");
    msg_info("PipelineImpl") << "Calling doCollisionReset()";
    doCollisionReset();
    sofa::helper::AdvancedTimer::stepEnd("CollisionReset");
}

void PipelineImpl::computeCollisionDetection()
{
    msg_info("PipelineImpl") << "computeCollisionDetection()";
    simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());

    if(root == NULL)
    {
        msg_warning("PipelineImpl") << "Simulation root node is NULL, doing nothing!";
        return;
    }

    std::vector<CollisionModel*> collisionModels;
    root->getTreeObjects<CollisionModel>(&collisionModels);

    msg_info("PipelineImpl") << "CollisionModel instances in scene: " << collisionModels.size();

    msg_info("PipelineImpl") << "Calling doCollisionDetection()";
    doCollisionDetection(collisionModels);
}

void PipelineImpl::computeCollisionResponse()
{
    msg_info("PipelineImpl") << "computeCollisionResponse()";
    simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());

    if (root == NULL)
    {
        msg_warning("PipelineImpl") << "Simulation root node is NULL, doing nothing!";
        return;
    }

    sofa::helper::AdvancedTimer::stepBegin("CollisionResponse");
    msg_info("PipelineImpl") << "Calling doCollisionResponse()";
    doCollisionResponse();
    sofa::helper::AdvancedTimer::stepEnd("CollisionResponse");
}

} // namespace simulation

} // namespace sofa
