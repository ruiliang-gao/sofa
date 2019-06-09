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

#include <ZyObbTreeGPU/config.h>

#include "ObbTreeGPUCollisionModel.h"
#include "ObbTreeGPUIntersection.h"
#include "ObbTreeGPUCollisionDetection_Threaded.h"

namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_OBBTREEGPUPLUGIN_API void initExternalModule();
    SOFA_OBBTREEGPUPLUGIN_API const char* getModuleName();
    SOFA_OBBTREEGPUPLUGIN_API const char* getModuleVersion();
    SOFA_OBBTREEGPUPLUGIN_API const char* getModuleLicense();
    SOFA_OBBTREEGPUPLUGIN_API const char* getModuleDescription();
    SOFA_OBBTREEGPUPLUGIN_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

const char* getModuleName()
{
    return "OBB Tree GPU";
}

const char* getModuleVersion()
{
    return "0.1";
}

const char* getModuleLicense()
{
    return "LGPL";
}


const char* getModuleDescription()
{
    return "GPU based OBB tree for SOFA.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    return "ObbTreeGPUCollisionModel, OBBTreeGPUDiscreteIntersection, ObbTreeGPUCollisionDetection_Threaded";
}



}

}

using namespace sofa::component::collision;

/// Use the SOFA_LINK_CLASS macro for each class, to enable linking on all platforms
SOFA_LINK_CLASS(ObbTreeGPUCollisionModel)
SOFA_LINK_CLASS(ObbTreeGPUCollisionDetection_Threaded)
SOFA_LINK_CLASS(OBBTreeGPUDiscreteIntersection)
