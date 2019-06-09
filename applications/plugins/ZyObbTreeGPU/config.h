#ifndef OBBTREEGPU_CONFIG_H
#define OBBTREEGPU_CONFIG_H

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_OBBTREEGPU
#define SOFA_OBBTREEGPUPLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_OBBTREEGPUPLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifndef _WIN32
#define BOOST_NOINLINE __attribute__((noinline))
#endif

#endif
