#ifndef ZY_PIPELINE_INIT_H
#define ZY_PIPELINE_INIT_H

#include <sofa/helper/system/config.h>

#ifdef BUILD_ZY_PIPELINE
#  define ZY_PIPELINE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define ZY_PIPELINE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
