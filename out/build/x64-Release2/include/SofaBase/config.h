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
#ifndef SOFABASE_CONFIG_H
#define SOFABASE_CONFIG_H

#include <sofa/simulation/config.h>

#ifdef SOFA_BUILD_BASE_TOPOLOGY
#  define SOFA_TARGET SofaBaseTopology
#  define SOFA_BASE_TOPOLOGY_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_BASE_TOPOLOGY_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_BASE_LINEAR_SOLVER
#  define SOFA_TARGET SofaBaseLinearSolver
#  define SOFA_BASE_LINEAR_SOLVER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_BASE_LINEAR_SOLVER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_BASE_MECHANICS
#  define SOFA_TARGET SofaBaseMechanics
#  define SOFA_BASE_MECHANICS_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_BASE_MECHANICS_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_BASE_COLLISION
#  define SOFA_TARGET SofaBaseCollision
#  define SOFA_BASE_COLLISION_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_BASE_COLLISION_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_BASE_VISUAL
#  define SOFA_TARGET SofaBaseVisual
#  define SOFA_BASE_VISUAL_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_BASE_VISUAL_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
