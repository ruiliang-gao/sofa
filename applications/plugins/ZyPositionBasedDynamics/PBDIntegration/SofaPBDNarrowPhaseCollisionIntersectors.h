#ifndef SOFAPBDNARROWPHASECOLLISIONINTERSECTORS_H
#define SOFAPBDNARROWPHASECOLLISIONINTERSECTORS_H

#include "SofaPBDCollisionIntersectorInterface.h"

#include <SofaMeshCollision/TriangleModel.h>

#include <PBDIntegration/SofaPBDPointCollisionModel.h>
#include <PBDIntegration/SofaPBDLineCollisionModel.h>
#include <PBDIntegration/SofaPBDTriangleCollisionModel.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDNarrowPhaseCollisionIntersectors: public SofaPBDCollisionIntersectorInterface
            {
                public:
                    SOFA_CLASS(SofaPBDNarrowPhaseCollisionIntersectors, SofaPBDCollisionIntersectorInterface);
                    SofaPBDNarrowPhaseCollisionIntersectors();
                    ~SofaPBDNarrowPhaseCollisionIntersectors();

                    int doIntersectionPointPoint(double dist2, const defaulttype::Vector3 &p, const defaulttype::Vector3 &q, core::collision::BaseIntersector::OutputVector *contacts, int id, int indexPoint1, int indexPoint2, sofa::component::collision::PointLocalMinDistanceFilter &f1, sofa::component::collision::PointLocalMinDistanceFilter &f2);
                    int doIntersectionLinePoint(double dist2, const defaulttype::Vector3 &p1, const defaulttype::Vector3 &p2, const defaulttype::Vector3 &q, core::collision::BaseIntersector::OutputVector *contacts, int id, int indexLine1, int indexPoint2, sofa::component::collision::LineLocalMinDistanceFilter &f1, sofa::component::collision::PointLocalMinDistanceFilter &f2, bool swapElems = false);
                    int doIntersectionLineLine(double dist2, const defaulttype::Vector3 &p1, const defaulttype::Vector3 &p2, const defaulttype::Vector3 &q1, const defaulttype::Vector3 &q2, core::collision::BaseIntersector::OutputVector *contacts, int id, int indexLine1, int indexLine2, bool useLMDFilters = false, sofa::component::collision::LineLocalMinDistanceFilter *f1 = nullptr, sofa::component::collision::LineLocalMinDistanceFilter *f2 = nullptr);
                    int doIntersectionTrianglePoint(double dist2, int flags, const defaulttype::Vector3 &p1, const defaulttype::Vector3 &p2, const defaulttype::Vector3 &p3, const defaulttype::Vector3 &n, const defaulttype::Vector3 &q, core::collision::BaseIntersector::OutputVector *contacts, int id, sofa::simulation::PBDSimulation::TPBDTriangle<sofa::defaulttype::Vec3Types> &e1, unsigned int *edgesIndices, int indexPoint2, sofa::component::collision::TriangleLocalMinDistanceFilter &f1, bool swapElems = false);
            };
        }
    }
}


#endif // SOFAPBDNARROWPHASECOLLISIONINTERSECTORS_H
