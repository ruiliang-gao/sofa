#ifndef PBDCOLLISIONMODELSINTERSECTION_H
#define PBDCOLLISIONMODELSINTERSECTION_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <SofaConstraint/LMDNewProximityIntersection.h>

#include "initZyPositionBasedDynamicsPlugin.h"

#include <SofaBaseCollision/BruteForceDetection.h>

#include "SofaPBDPointCollisionModel.h"
#include "SofaPBDLineCollisionModel.h"
#include "SofaPBDTriangleCollisionModel.h"

#include "SofaPBDNarrowPhaseCollisionIntersectors.h"

using namespace sofa::simulation::PBDSimulation;

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API PBDCollisionModelsIntersection : public LMDNewProximityIntersection
            {
                typedef DiscreteIntersection::OutputVector OutputVector;

                public:
                    SOFA_CLASS(PBDCollisionModelsIntersection, LMDNewProximityIntersection);

                    PBDCollisionModelsIntersection(/*DiscreteIntersection* = nullptr*/);

                    void init() override;

                    bool testIntersection(Triangle&, Point&);
                    bool testIntersection(Line&, Line&);
                    bool testIntersection(Triangle&, Line&);
                    bool testIntersection(Triangle&, Triangle&);

                    bool testIntersection(Line&, TPBDPoint<sofa::defaulttype::Vec3Types>&);
                    bool testIntersection(Triangle&, TPBDPoint<sofa::defaulttype::Vec3Types>&);

                    int computeIntersection(Triangle&, Point&, OutputVector*);
                    int computeIntersection(Triangle&, Line&, OutputVector*);
                    int computeIntersection(Triangle&, Triangle&, OutputVector*);

                    int computeIntersection(Line&, Line&, OutputVector*);
                    int computeIntersection(Triangle&, TPBDPoint<sofa::defaulttype::Vector3>&, OutputVector*);
                    int computeIntersection(Line&, TPBDPoint<sofa::defaulttype::Vec3Types>&, OutputVector*);
                    int computeIntersection(Triangle&, TPBDPoint<sofa::defaulttype::Vec3Types>&, OutputVector*);

                    bool testIntersection(TPBDPoint<sofa::defaulttype::Vec3Types>&, TPBDPoint<sofa::defaulttype::Vec3Types>&);
                    bool testIntersection(TPBDLine<sofa::defaulttype::Vec3Types>&, TPBDPoint<sofa::defaulttype::Vec3Types>&);
                    bool testIntersection(TPBDLine<sofa::defaulttype::Vec3Types>&, TPBDLine<sofa::defaulttype::Vec3Types>&);
                    bool testIntersection(TPBDTriangle<sofa::defaulttype::Vec3Types>&, TPBDPoint<sofa::defaulttype::Vec3Types>&);
                    bool testIntersection(TPBDTriangle<sofa::defaulttype::Vec3Types>&, TPBDLine<sofa::defaulttype::Vec3Types>&);
                    bool testIntersection(TPBDTriangle<sofa::defaulttype::Vec3Types>&, TPBDTriangle<sofa::defaulttype::Vec3Types>&);

                    int computeIntersection(TPBDPoint<sofa::defaulttype::Vec3Types>&, TPBDPoint<sofa::defaulttype::Vec3Types>&, OutputVector*);
                    int computeIntersection(TPBDLine<sofa::defaulttype::Vec3Types>&, TPBDPoint<sofa::defaulttype::Vec3Types>&, OutputVector*);
                    int computeIntersection(TPBDLine<sofa::defaulttype::Vec3Types>&, TPBDLine<sofa::defaulttype::Vec3Types>&, OutputVector*);
                    int computeIntersection(TPBDTriangle<sofa::defaulttype::Vec3Types>&, TPBDPoint<sofa::defaulttype::Vec3Types>&, OutputVector*);
                    int computeIntersection(TPBDTriangle<sofa::defaulttype::Vec3Types>&, TPBDLine<sofa::defaulttype::Vec3Types>&, OutputVector*);
                    int computeIntersection(TPBDTriangle<sofa::defaulttype::Vec3Types>&, TPBDTriangle<sofa::defaulttype::Vec3Types>&, OutputVector*);

                protected:
                    DiscreteIntersection* intersection;
                    sofa::simulation::PBDSimulation::SofaPBDNarrowPhaseCollisionIntersectors m_intersector;
            };
        } // namespace collision
    } // namespace component
} // namespace sofa

#endif // PBDCOLLISIONMODELSINTERSECTION_H
