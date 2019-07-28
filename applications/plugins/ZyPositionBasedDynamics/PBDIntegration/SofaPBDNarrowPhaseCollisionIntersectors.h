#ifndef SOFAPBDNARROWPHASECOLLISIONINTERSECTORS_H
#define SOFAPBDNARROWPHASECOLLISIONINTERSECTORS_H

#include "SofaPBDCollisionIntersectorInterface.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDNarrowPhaseCollisionIntersectors: public SofaPBDCollisionIntersectorInterface
            {
                public:
                    SofaPBDNarrowPhaseCollisionIntersectors();
                    ~SofaPBDNarrowPhaseCollisionIntersectors();

                    template< class TFilter1, class TFilter2 >
                    int doIntersectionLineLine(double dist2, const defaulttype::Vector3 &p1, const defaulttype::Vector3 &p2, const defaulttype::Vector3 &q1, const defaulttype::Vector3 &q2, core::collision::BaseIntersector::OutputVector *contacts, int id, int indexLine1, int indexLine2, TFilter1 &f1, TFilter2 &f2);
                    template< class TFilter1, class TFilter2 >
                    int doIntersectionLinePoint(double dist2, const defaulttype::Vector3 &p1, const defaulttype::Vector3 &p2, const defaulttype::Vector3 &q, core::collision::BaseIntersector::OutputVector *contacts, int id, int indexLine1, int indexPoint2, TFilter1 &f1, TFilter2 &f2, bool swapElems);
                    template< class TFilter1, class TFilter2 >
                    int doIntersectionPointPoint(double dist2, const defaulttype::Vector3 &p, const defaulttype::Vector3 &q, core::collision::BaseIntersector::OutputVector *contacts, int id, int indexPoint1, int indexPoint2, TFilter1 &f1, TFilter2 &f2);
                    template< class TFilter1, class TFilter2 >
                    int doIntersectionTrianglePoint(double dist2, int flags, const defaulttype::Vector3 &p1, const defaulttype::Vector3 &p2, const defaulttype::Vector3 &p3, const defaulttype::Vector3 &n, const defaulttype::Vector3 &q, core::collision::BaseIntersector::OutputVector *contacts, int id, component::collision::Triangle &e1, unsigned int *edgesIndices, int indexPoint2, TFilter1 &f1, TFilter2 &f2, bool swapElems);
            };
        }
    }
}


#endif // SOFAPBDNARROWPHASECOLLISIONINTERSECTORS_H
