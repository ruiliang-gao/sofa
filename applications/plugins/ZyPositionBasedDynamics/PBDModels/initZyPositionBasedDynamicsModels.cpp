#include "initZyPositionBasedDynamicsModels.h"

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
        return "Position Based Dynamics.";
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
        return "Position based dynamics implementation according to https://github.com/InteractiveComputerGraphics/PositionBasedDynamics, model classes.";
    }

    const char* getModuleComponentList()
    {
        return "SofaPBDLineModel, SofaPBDTriangleModel";
    }
}

SOFA_LINK_CLASS(SofaPBDLineModel)
SOFA_LINK_CLASS(SofaPBDTriangleModel)
