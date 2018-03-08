#ifndef SOFA_CORE_CONFIG_H
#define SOFA_CORE_CONFIG_H

#include <sofa/config/sharedlibrary_defines.h>

#ifdef SOFA_BUILD_CORE
#   define SOFA_TARGET   SofaCore
#	define SOFA_CORE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_CORE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
