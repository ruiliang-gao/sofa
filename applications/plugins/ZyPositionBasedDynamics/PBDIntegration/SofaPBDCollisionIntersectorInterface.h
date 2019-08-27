#ifndef SOFAPBDCOLLISIONINTERSECTORINTERFACE_H
#define SOFAPBDCOLLISIONINTERSECTORINTERFACE_H

#include "initZyPositionBasedDynamicsPlugin.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/collision/Intersection.h>

#include <SofaMeshCollision/PointModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/TriangleModel.h>

#include <SofaMeshCollision/PointLocalMinDistanceFilter.h>
#include <SofaMeshCollision/LineLocalMinDistanceFilter.h>
#include <SofaMeshCollision/TriangleLocalMinDistanceFilter.h>

using namespace sofa;
namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDCollisionIntersectorInterface: public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SofaPBDCollisionIntersectorInterface, sofa::core::objectmodel::BaseObject);

                    SofaPBDCollisionIntersectorInterface(): sofa::core::objectmodel::BaseObject() {}

                    ~SofaPBDCollisionIntersectorInterface() {}

                    // From LMDNewProximityIntersection
                    int doIntersectionLineLine(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q1, const defaulttype::Vector3& q2, sofa::core::collision::BaseIntersector::OutputVector* contacts, int id, int indexLine1, int indexLine2, sofa::component::collision::LineLocalMinDistanceFilter &f1, sofa::component::collision::LineLocalMinDistanceFilter &f2) { return 0; }

                    int doIntersectionLinePoint(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q, sofa::core::collision::BaseIntersector::OutputVector* contacts, int id, int indexLine1, int indexPoint2, sofa::component::collision::LineLocalMinDistanceFilter &f1, sofa::component::collision::PointLocalMinDistanceFilter &f2, bool swapElems = false) { return 0; }

                    int doIntersectionPointPoint(double dist2, const defaulttype::Vector3& p, const defaulttype::Vector3& q, sofa::core::collision::BaseIntersector::OutputVector* contacts, int id, int indexPoint1, int indexPoint2, sofa::component::collision::PointLocalMinDistanceFilter &f1, sofa::component::collision::PointLocalMinDistanceFilter &f2) { return 0; }

                    int doIntersectionTrianglePoint(double dist2, int flags, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& p3, const defaulttype::Vector3& n, const defaulttype::Vector3& q, sofa::core::collision::BaseIntersector::OutputVector* contacts, int id, sofa::component::collision::Triangle &e1, unsigned int *edgesIndices, int indexPoint2, sofa::component::collision::TriangleLocalMinDistanceFilter &f1, sofa::component::collision::PointLocalMinDistanceFilter &f2, bool swapElems = false) { return 0; }
            };
        }
    }
}

#endif // SOFAPBDCOLLISIONINTERSECTORINTERFACE_H
