#include "initZyPositionBasedDynamicsPlugin.h"

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
        return "Position based dynamics implementation according to https://github.com/InteractiveComputerGraphics/PositionBasedDynamics";
    }

    const char* getModuleComponentList()
    {
        return "PBDSimulation,PBDSimulationModel,PBDTimeStep";
    }
}
