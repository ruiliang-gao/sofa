#ifndef INITSENSABLE_H
#define INITSENSABLE_H

#include <sofa/helper/system/config.h>

#ifndef WIN32
#define SOFA_EXPORT_DYNAMIC_LIBRARY
#define SOFA_IMPORT_DYNAMIC_LIBRARY
#define SOFA_SENSABLEPLUGIN_API
#else
#ifdef SOFA_BUILD_SENSABLEPLUGIN
#define SOFA_SENSABLEPLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_SENSABLEPLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif

#ifdef WIN32
// BUGFIX(Jeremie A. 02-05-2009): put OpenHaptics libs here instead of the project file to work around a bug in qmake when there is a space in an environment variable used to look-up a library
#pragma comment(lib,"hl.lib")
#pragma comment(lib,"hd.lib")
#pragma comment(lib,"hdu.lib")
#endif

#endif
