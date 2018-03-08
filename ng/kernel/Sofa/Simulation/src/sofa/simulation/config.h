#ifndef SOFA_SIMULATION_CONFIG_H
#define SOFA_SIMULATION_CONFIG_H

#include <sofa/config/sharedlibrary_defines.h>

#ifdef BUILD_TARGET_SOFA_SIMULATION
#   define SOFA_TARGET SofaSimulation
#	define SOFA_SIMULATION_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_SIMULATION_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
