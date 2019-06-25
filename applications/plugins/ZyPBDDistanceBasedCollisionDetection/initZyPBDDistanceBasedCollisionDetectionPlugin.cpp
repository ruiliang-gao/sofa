#include "initZyPBDDistanceBasedCollisionDetectionPlugin.h"

extern "C" {
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
        return "Distance field collision detection.";
    }

    const char* getModuleVersion()
    {
        return "0.0.1";
    }

    const char* getModuleLicense()
    {
        return "LGPL";
    }

    const char* getModuleDescription()
    {
        return "Distance field-based collision detection according to https://github.com/InteractiveComputerGraphics/PositionBasedDynamics";
    }

    const char* getModuleComponentList()
    {
        return "PBDDinstanceBasedCollisionDetection";
    }
}
