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
        return "SofaPBDPlugin";
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
        return "SofaPBDSimulationLoop, SofaPBDSimulation, SofaPBDLineModel, SofaPBDRigidBodyModel, SofaPBDTriangleMeshModel, SofaPBDTetrahedronMeshModel, SofaPBDPointCollisionModel, SofaPBDLineCollisionModel, SofaPBDTriangleCollisionModel, PBDCollisionModelsIntersection, PBDBruteForceDetection, SofaPBDPipeline, SofaPBDNarrowPhaseCollisionIntersectors";
    }
}

SOFA_LINK_CLASS(SofaPBDSimulationLoop)
SOFA_LINK_CLASS(SofaPBDSimulation)
SOFA_LINK_CLASS(SofaPBDLineModel)
SOFA_LINK_CLASS(SofaPBDRigidBodyModel)
SOFA_LINK_CLASS(SofaPBDTriangleMeshModel)
SOFA_LINK_CLASS(SofaPBDTetrahedronMeshModel)
SOFA_LINK_CLASS(SofaPBDPointCollisionModel)
SOFA_LINK_CLASS(SofaPBDLineCollisionModel)
SOFA_LINK_CLASS(SofaPBDTriangleCollisionModel)
SOFA_LINK_CLASS(PBDCollisionModelsIntersection)
SOFA_LINK_CLASS(SofaPBDPipeline)
SOFA_LINK_CLASS(SofaPBDNarrowPhaseCollisionIntersectors)
SOFA_LINK_CLASS(PBDBruteForceDetection)
