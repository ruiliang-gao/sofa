#ifndef BVHMODELSINTERSECTION_H
#define BVHMODELSINTERSECTION_H

#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>

#include "initBVHModelsPlugin.h"
#include "PQPModel.h"

#include <components/collision/DistanceGridCollisionModel.h>
#include <SofaBaseCollision/MinProximityIntersection.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <SofaConstraint/LocalMinDistance.h>

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            enum PQP_CHECK_MODE
            {
                PQP_MODE_INTERSECTION = 1,
                PQP_MODE_CLOSEST_POINT = 2,
                PQP_MODE_TOLERANCE = 4
            };
            
            class SOFA_BVHMODELSPLUGIN_API BVHModelsDiscreteIntersection : public sofa::core::objectmodel::BaseObject, public core::collision::BaseIntersector
            {
                typedef DiscreteIntersection::OutputVector OutputVector;

                public:
					SOFA_CLASS(BVHModelsDiscreteIntersection, sofa::core::objectmodel::BaseObject);

                    BVHModelsDiscreteIntersection();
                    BVHModelsDiscreteIntersection(DiscreteIntersection* object, bool addSelf=true);

                    bool testIntersection(PQPCollisionModelNode&, PQPCollisionModelNode&);
                    int computeIntersection(PQPCollisionModelNode& e1, PQPCollisionModelNode& e2, OutputVector* contacts);

                protected:
                    DiscreteIntersection* intersection;
            };


            class SOFA_BVHMODELSPLUGIN_API BVHModelsLocalMinDistance: public LocalMinDistance
            {
                std::ofstream testOutput;
                sofa::core::objectmodel::DataFileName testOutputFilename;

                std::vector<std::pair<std::string, std::string> > checkedCollisionModels;
                std::vector<std::pair<std::string, std::string> > overlappingCollisionModels;

                std::map<std::pair<std::string, std::string>, PQP_CollideResult> collideResults;
                std::map<std::pair<std::string, std::string>, PQP_DistanceResult> distanceResults;
                std::map<std::pair<std::string, std::string>, PQP_ToleranceResult> toleranceResults;

                Data<unsigned int> m_checkMode;

                public:
                    SOFA_CLASS(BVHModelsLocalMinDistance,sofa::component::collision::LocalMinDistance);

                    BVHModelsLocalMinDistance();

                    virtual void init();

                    virtual void beginBroadPhase();
                    virtual void endBroadPhase();

                    void draw(const core::visual::VisualParams *vparams);

                    bool testIntersection(PQPCollisionModelNode&, PQPCollisionModelNode&);
                    int computeIntersection(PQPCollisionModelNode& e1, PQPCollisionModelNode& e2, OutputVector* contacts);
            };
        } // namespace collision
    } // namespace component
} // namespace sofa

#endif // BVHMODELSINTERSECTION_H
