/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFAGENERAL_CONFIG_H
#define SOFAGENERAL_CONFIG_H

#include <SofaCommon/config.h>

#define SOFAGENERAL_HAVE_SOFADENSESOLVER 1

#ifdef SOFA_BUILD_GENERAL_DEFORMABLE
#  define SOFA_TARGET SofaGeneralDeformable
#  define SOFA_GENERAL_DEFORMABLE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_DEFORMABLE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_EXPLICIT_ODE_SOLVER
#  define SOFA_TARGET SofaGeneralExplicitOdeSolver
#  define SOFA_GENERAL_EXPLICIT_ODE_SOLVER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_EXPLICIT_ODE_SOLVER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_IMPLICIT_ODE_SOLVER
#  define SOFA_TARGET SofaGeneralImplicitOdeSolver
#  define SOFA_GENERAL_IMPLICIT_ODE_SOLVER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_IMPLICIT_ODE_SOLVER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_LOADER
#  define SOFA_TARGET SofaGeneralLoader
#  define SOFA_GENERAL_LOADER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_LOADER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_MESH_COLLISION
#  define SOFA_TARGET SofaGeneralMeshCollision
#  define SOFA_GENERAL_MESH_COLLISION_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_MESH_COLLISION_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_OBJECT_INTERACTION
#  define SOFA_TARGET SofaGeneralObjectInteraction
#  define SOFA_GENERAL_OBJECT_INTERACTION_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_OBJECT_INTERACTION_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_RIGID
#  define SOFA_TARGET SofaGeneralRigid
#  define SOFA_GENERAL_RIGID_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_RIGID_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_SIMPLE_FEM
#  define SOFA_TARGET SofaGeneralSimpleFem
#  define SOFA_GENERAL_SIMPLE_FEM_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_SIMPLE_FEM_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_VALIDATION
#  define SOFA_TARGET SofaValidation
#  define SOFA_VALIDATION_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_VALIDATION_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_EXPORTER
#  define SOFA_TARGET SofaExporter
#  define SOFA_SOFAEXPORTER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_SOFAEXPORTER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GRAPH_COMPONENT
#  define SOFA_TARGET SofaGraphComponent
#  define SOFA_GRAPH_COMPONENT_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GRAPH_COMPONENT_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_TOPOLOGY_MAPPING
#  define SOFA_TARGET SofaTopologyMapping
#  define SOFA_TOPOLOGY_MAPPING_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_TOPOLOGY_MAPPING_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_BOUNDARY_CONDITION
#  define SOFA_TARGET SofaBoundaryCondition
#  define SOFA_BOUNDARY_CONDITION_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_BOUNDARY_CONDITION_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_LINEAR_SOLVER
#  define SOFA_TARGET SofaGeneralLinearSolver
#  define SOFA_GENERAL_LINEAR_SOLVER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_LINEAR_SOLVER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_ANIMATION_LOOP
#  define SOFA_TARGET SofaGeneralAnimationLoop
#  define SOFA_GENERAL_ANIMATION_LOOP_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_ANIMATION_LOOP_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_ENGINE
#  define SOFA_TARGET SofaGeneralEngine
#  define SOFA_GENERAL_ENGINE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_ENGINE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_TOPOLOGY
#  define SOFA_TARGET SofaGeneralTopology
#  define SOFA_GENERAL_TOPOLOGY_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_TOPOLOGY_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_GENERAL_VISUAL
#  define SOFA_TARGET SofaGeneralVisual
#  define SOFA_GENERAL_VISUAL_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_GENERAL_VISUAL_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_USER_INTERACTION
#  define SOFA_TARGET SofaUserInteraction
#  define SOFA_USER_INTERACTION_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_USER_INTERACTION_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_CONSTRAINT
#  define SOFA_TARGET SofaConstraint
#  define SOFA_CONSTRAINT_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_CONSTRAINT_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_DENSE_SOLVER
#  define SOFA_TARGET SofaDenseSolver
#  define SOFA_DENSE_SOLVER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_DENSE_SOLVER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_PARDISO_SOLVER
#  define SOFA_PARDISO_SOLVER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_PARDISO_SOLVER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
