#include "SofaPBDNarrowPhaseCollisionIntersectors.inl"

#include <sofa/core/ObjectFactory.h>

using namespace sofa::simulation::PBDSimulation;

SOFA_DECL_CLASS(SofaPBDNarrowPhaseCollisionIntersectors)

int SofaPBDNarrowPhaseCollisionIntersectorsClass = sofa::core::RegisterObject("SofaPBDNarrowPhaseCollisionIntersectors")
                            .add< SofaPBDNarrowPhaseCollisionIntersectors >()
                            .addDescription("Intersection computation interface for SOFA PBD plugin.");


SofaPBDNarrowPhaseCollisionIntersectors::SofaPBDNarrowPhaseCollisionIntersectors(): SofaPBDCollisionIntersectorInterface()
{

}

