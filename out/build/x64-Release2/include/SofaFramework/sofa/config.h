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
#pragma once

#include <sofa/config/sharedlibrary_defines.h>
#include <sofa/config/build_option_dump_visitor.h>
#include <sofa/config/build_option_opengl.h>
#include <sofa/config/build_option_threading.h>
#include <sofa/config/build_option_bbox.h>

#include <cstddef> // For nullptr
#include <limits> // std::numeric_limits<>

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


#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#endif

// snprintf() has been provided since MSVC++ 14 (Visual Studio 2015).  For other
// versions, it is simply #defined to _snprintf().
#if (defined(_MSC_VER) && _MSC_VER < 1900)
#  define snprintf _snprintf
#endif

#ifdef _WIN32
#  include <windows.h>
#endif

#ifdef BOOST_NO_EXCEPTIONS
#  include<exception>

namespace boost
{
    inline void throw_exception(std::exception const & e)
    {
        return;
    }
}
#endif // BOOST_NO_EXCEPTIONS


#ifdef _MSC_VER
#  ifndef _USE_MATH_DEFINES
#    define _USE_MATH_DEFINES 1 // required to get M_PI from math.h
#  endif
// Visual C++ does not include stdint.h
typedef signed __int8		int8_t;
typedef signed __int16		int16_t;
typedef signed __int32		int32_t;
typedef signed __int64		int64_t;
typedef unsigned __int8		uint8_t;
typedef unsigned __int16	uint16_t;
typedef unsigned __int32	uint32_t;
typedef unsigned __int64	uint64_t;
#else
#  include <stdint.h>
#endif

#define sofa_do_concat2(a,b) a ## b
#define sofa_do_concat(a,b) sofa_do_concat2(a,b)
#define sofa_concat(a,b) sofa_do_concat(a,b)

#define sofa_do_tostring(a) #a
#define sofa_tostring(a) sofa_do_tostring(a)

///////////////////// These macros can be used to convert deprecation attributes as error.
#if defined(_MSC_VER)
#define SOFA_BEGIN_DEPRECATION_AS_ERROR __pragma(warning( push )) \
                                        __pragma(warning(error: 4996))
#define SOFA_END_DEPRECATION_AS_ERROR __pragma(warning( pop ))
#elif defined(__GNUC__)
#define SOFA_BEGIN_DEPRECATION_AS_ERROR _Pragma("GCC diagnostic push") \
                                   _Pragma("GCC diagnostic error \"-Wdeprecated-declarations\"")
#define SOFA_END_DEPRECATION_AS_ERROR _Pragma("GCC diagnostic pop")
#else /// clang
#define SOFA_BEGIN_DEPRECATION_AS_ERROR _Pragma("clang diagnostic push") \
                                   _Pragma("clang diagnostic error \"-Wdeprecated-declarations\"")
#define SOFA_END_DEPRECATION_AS_ERROR _Pragma("clang diagnostic pop")
#endif

#ifdef _WIN32
#  define SOFA_PRAGMA_MESSAGE(text) __pragma(message(__FILE__ "(" sofa_tostring(__LINE__) "): " #text))
#  define SOFA_PRAGMA_WARNING(text) __pragma(message(__FILE__ "(" sofa_tostring(__LINE__) "): warning: " #text))
#else
#  define SOFA_DO_PRAGMA(x) _Pragma(#x)
#  define SOFA_PRAGMA_MESSAGE(text) SOFA_DO_PRAGMA(message #text)
#  define SOFA_PRAGMA_WARNING(text) SOFA_DO_PRAGMA(GCC warning #text)
#endif
#define SOFA_DEPRECATED_HEADER(untilVersion, newHeader) SOFA_PRAGMA_WARNING( \
    This header is deprecated and will be removed at SOFA untilVersion.      \
    To fix this warning you must include newHeader instead. )

#define SOFA_DECL_CLASS(name) // extern "C" { int sofa_concat(class_,name) = 0; }
#define SOFA_LINK_CLASS(name) // extern "C" { extern int sofa_concat(class_,name); int sofa_concat(link_,name) = sofa_concat(class_,name); }

// Prevent compiler warnings about 'unused variables'.
// This should be used when a parameter name is needed (e.g. for
// documentation purposes) even if it is not used in the code.
#define SOFA_UNUSED(x) (void)(x)

// utility for debug tracing
#ifdef _MSC_VER
    #define SOFA_CLASS_METHOD ( std::string(this->getClassName()) + "::" + __FUNCTION__ + " " )
#else
    #define SOFA_CLASS_METHOD ( std::string(this->getClassName()) + "::" + __func__ + " " )
#endif


// The SOFA_EXTERN_TEMPLATE macro was used to control the use of extern templates in Sofa.
// It has been cleaned out in 41e0ab98bbb6e22b053b24e7bbdd31c8636336c9 "[ALL] Remove SOFA_EXTERN_TEMPLATE".
// Macro definition is kept to avoid breaking all external plugins.
#define SOFA_EXTERN_TEMPLATE

#ifdef SOFA_BUILD_SOFAFRAMEWORK
#  define SOFA_TARGET SofaFramework
#  define SOFA_SOFAFRAMEWORK_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_SOFAFRAMEWORK_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace sofa
{
	
using Index = uint32_t;
using Size = uint32_t;

constexpr Index InvalidID = (std::numeric_limits<Index>::max)(); 

}


