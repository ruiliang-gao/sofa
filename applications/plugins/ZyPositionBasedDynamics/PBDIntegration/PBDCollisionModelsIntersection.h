#ifndef PBDCOLLISIONMODELSINTERSECTION_H
#define PBDCOLLISIONMODELSINTERSECTION_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <SofaConstraint/LMDNewProximityIntersection.h>

#include "initZyPositionBasedDynamicsPlugin.h"

#include "SofaPBDPointCollisionModel.h"
#include "SofaPBDLineCollisionModel.h"
#include "SofaPBDTriangleCollisionModel.h"

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API PBDCollisionModelsIntersection : public BaseProximityIntersection
            {
                typedef DiscreteIntersection::OutputVector OutputVector;

                public:
                    SOFA_CLASS(PBDCollisionModelsIntersection, BaseProximityIntersection);

                    PBDCollisionModelsIntersection(/*DiscreteIntersection* = nullptr*/);

                    void init() override;

                    bool testIntersection(Point&, Point&);
                    bool testIntersection(Line&, Point&);
                    bool testIntersection(Line&, Line&);
                    bool testIntersection(Triangle&, Point&);
                    bool testIntersection(Triangle&, Line&);
                    bool testIntersection(Triangle&, Triangle&);

                    int computeIntersection(Point&, Point&, OutputVector*);
                    int computeIntersection(Line&, Point&, OutputVector*);
                    int computeIntersection(Line&, Line&, OutputVector*);
                    int computeIntersection(Triangle&, Point&, OutputVector*);
                    int computeIntersection(Triangle&, Line&, OutputVector*);
                    int computeIntersection(Triangle&, Triangle&, OutputVector*);

                protected:
                    DiscreteIntersection* intersection;
            };


            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API PBDCollisionModelsLocalMinDistance: public LMDNewProximityIntersection
            {
                std::vector<std::pair<std::string, std::string> > checkedCollisionModels;
                std::vector<std::pair<std::string, std::string> > overlappingCollisionModels;

                public:
                    SOFA_CLASS(PBDCollisionModelsLocalMinDistance, sofa::component::collision::LMDNewProximityIntersection);

                    PBDCollisionModelsLocalMinDistance();

                    virtual void init();

                    virtual void beginBroadPhase();
                    virtual void endBroadPhase();

                    void draw(const core::visual::VisualParams *vparams);
            };
        } // namespace collision
    } // namespace component
} // namespace sofa

#endif // PBDCOLLISIONMODELSINTERSECTION_H
