/******************************************************************************
******************************************************************************/
#include <sofa/config.h>


#ifndef WIN32
#define SOFA_EXPORT_DYNAMIC_LIBRARY
#define SOFA_IMPORT_DYNAMIC_LIBRARY
#define SURFLAB_HAPTICDEVICE_API
#else
#ifdef SOFA_BUILD_SURFLABHAPTICDEVICE
#define SURFLAB_HAPTICDEVICE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SURFLAB_HAPTICDEVICE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif

#ifdef WIN32
// BUGFIX(Jeremie A. 02-05-2009): put OpenHaptics libs here instead of the project file to work around a bug in qmake when there is a space in an environment variable used to look-up a library
#pragma comment(lib,"hl.lib")
#pragma comment(lib,"hd.lib")
#pragma comment(lib,"hdu.lib")
#endif


namespace sofa
{

    namespace component
    {

        //Here are just several convenient functions to help user to know what contains the plugin

        extern "C" {
            SURFLAB_HAPTICDEVICE_API void initExternalModule();
            SURFLAB_HAPTICDEVICE_API const char* getModuleName();
            SURFLAB_HAPTICDEVICE_API const char* getModuleVersion();
            SURFLAB_HAPTICDEVICE_API const char* getModuleLicense();
            SURFLAB_HAPTICDEVICE_API const char* getModuleDescription();
            SURFLAB_HAPTICDEVICE_API const char* getModuleComponentList();
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
            return "Plugin Sensable";
        }

        const char* getModuleVersion()
        {
            return "beta 1.1";
        }

        const char* getModuleLicense()
        {
            return "LGPL";
        }

        const char* getModuleDescription()
        {
            return "Force feedback with sensable devices into SOFA Framework";
        }

        const char* getModuleComponentList()
        {
            return "ForceFeedback controllers ";
        }

    }

}


