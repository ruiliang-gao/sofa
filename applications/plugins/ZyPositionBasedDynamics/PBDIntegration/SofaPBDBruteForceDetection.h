#ifndef SOFAPBDBRUTEFORCEDETECTION_H
#define SOFAPBDBRUTEFORCEDETECTION_H

#include "initZyPositionBasedDynamicsPlugin.h"

#include <SofaBaseCollision/BruteForceDetection.h>
#include <sofa/core/collision/DetectionOutput.h>

#include "PBDIntegration/SofaPBDCollisionDetectionOutput.h"

namespace sofa
{
    namespace component
    {
        namespace collision
        {

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDBruteForceDetection: public sofa::component::collision::BruteForceDetection
            {
                std::vector<std::pair<std::string, std::string> > checkedCollisionModels;
                std::vector<std::pair<std::string, std::string> > overlappingCollisionModels;

                std::map<std::pair<core::CollisionModel*, core::CollisionModel*>, sofa::helper::vector<sofa::core::collision::SofaPBDCollisionDetectionOutput>> collisionOutputs;

                public:
                    SOFA_CLASS(SofaPBDBruteForceDetection, sofa::component::collision::BruteForceDetection);
                    SofaPBDBruteForceDetection();

                    void addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& coll_pair);

                    virtual void beginBroadPhase();
                    virtual void endBroadPhase();

                    virtual void beginNarrowPhase();
                    virtual void endNarrowPhase();

                    void draw(const core::visual::VisualParams*) override;

                    const std::map<std::pair<core::CollisionModel*, core::CollisionModel*>, sofa::helper::vector<sofa::core::collision::SofaPBDCollisionDetectionOutput>>& getCollisionOutputs() const;
            };
        }
    }
}

#endif // SOFAPBDBRUTEFORCEDETECTION_H
