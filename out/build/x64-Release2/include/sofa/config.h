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
#ifndef SOFA_CONFIG_H
#define SOFA_CONFIG_H

#include <sofa/config/sharedlibrary_defines.h>
#include <sofa/config/build_option_dump_visitor.h>
#include <sofa/config/build_option_opengl.h>
#include <sofa/config/build_option_threading.h>
#include <sofa/config/build_option_bbox.h>

// fixes CGAL plugin build errors (default value: 5)
#define BOOST_PARAMETER_MAX_ARITY 12

/* #undef SOFA_DETECTIONOUTPUT_FREEMOTION */

/* #undef SOFA_FLOAT */
/* #undef SOFA_DOUBLE */

#define SOFA_WITH_FLOAT
#define SOFA_WITH_DOUBLE

/* #undef SOFA_USE_MASK */

#define SOFA_WITH_DEVTOOLS

#ifdef _MSC_VER
#define EIGEN_DONT_ALIGN
#endif // _MSC_VER

#ifdef WIN32
#define UNICODE
#endif // WIN32

#ifdef SOFA_FLOAT
typedef float SReal;
#else
typedef double SReal;
#endif

// The SOFA_EXTERN_TEMPLATE macro was used to control the use of extern templates in Sofa.
// It has been cleaned out in 41e0ab98bbb6e22b053b24e7bbdd31c8636336c9 "[ALL] Remove SOFA_EXTERN_TEMPLATE".
// Macro definition is kept to avoid breaking all external plugins.
#define SOFA_EXTERN_TEMPLATE

#ifdef SOFA_BUILD_HELPER
#   define SOFA_TARGET SofaHelper
#	define SOFA_HELPER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_HELPER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_DEFAULTTYPE
#   define SOFA_TARGET SofaDefaulttype
#	define SOFA_DEFAULTTYPE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_DEFAULTTYPE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_CORE
#   define SOFA_TARGET   SofaCore
#	define SOFA_CORE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_CORE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_SIMULATION_CORE
#   define SOFA_TARGET SofaSimulationCore
#	define SOFA_SIMULATION_CORE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_SIMULATION_CORE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
