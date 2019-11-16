#ifndef ZYSINGLETHREADEDCOLLISIONPIPELINE_H
#define ZYSINGLETHREADEDCOLLISIONPIPELINE_H

#include "initZyPipeline.h"

#include <sofa/core/objectmodel/Data.h>
#include <sofa/simulation/PipelineImpl.h>
#include <SofaBaseCollision/DefaultPipeline.h>

#include <boost/shared_ptr.hpp>
#include <ZyPipelineInterface.h>

namespace sofa
{
    namespace core
    {
        class CollisionModel;
        namespace collision
        {
            class BroadPhaseDetection;
            class NarrowPhaseDetection;
            class CollisionGroupManager;
            class ContactManager;
            class Intersection;
        }
    }
}

/**
    This class is a duplicate/wrapper for SOFA's DefaultPipeline, but adhering to the TruPipelineInterface.
    It is intended for two use cases only:
    * Testing it vs. the multi-threaded TruPhysicsPipeline implementation for performance comparisons
    * As a place-holder, if for whatever reason the usual DefaultPipeline is required wrapped in a ZyPipelineInterface implementation
*/
namespace Zyklio
{
    namespace Pipeline
    {
        using namespace sofa;
        using namespace sofa::core::objectmodel;
        using namespace sofa::core::collision;

        class ZY_PIPELINE_API ZySingleThreadedCollisionPipelineImpl : public sofa::simulation::PipelineImpl
        {
            public:
                ZySingleThreadedCollisionPipelineImpl();
                ~ZySingleThreadedCollisionPipelineImpl();

                // Pipeline interface
                void computeCollisionReset();
                void computeCollisionDetection();
                void computeCollisionResponse();
                std::set< std::string > getResponseList() const;

            protected:
                // PipelineImpl interface
                void doCollisionReset();
                void doCollisionResponse();
                void doCollisionDetection(const sofa::helper::vector<sofa::core::CollisionModel*>&);

                Data<int> depth;

                friend class ZySingleThreadedCollisionPipeline;

        };

        class ZY_PIPELINE_API ZySingleThreadedCollisionPipeline : public sofa::core::collision::ZyPipelineInterface
        {
            public:
                SOFA_CLASS(ZySingleThreadedCollisionPipeline, sofa::core::collision::ZyPipelineInterface);

                ZySingleThreadedCollisionPipeline();
                ~ZySingleThreadedCollisionPipeline();

                void init();
                void bwdInit();

                void doCollisionDetection(const sofa::helper::vector<sofa::core::CollisionModel*>& collisionModels);
                void doCollisionResponse();
                void doCollisionReset();

                void setup(BroadPhaseDetection*, NarrowPhaseDetection*, Intersection*, ContactManager*, CollisionGroupManager*);

                bool isDefaultPipeline() const { return true; }

            protected:
                void filterCollisionModelsToProcess(const sofa::helper::vector<core::CollisionModel*>& collisionModels, sofa::helper::vector<core::CollisionModel*>& processedCollisionModels);

                ZySingleThreadedCollisionPipelineImpl* m_pipelineImpl;

                Intersection* pipelineIntersectionMethod;
                BroadPhaseDetection* pipelineBroadPhaseDetection;
                NarrowPhaseDetection* pipelineNarrowPhaseDetection;

                ContactManager* pipelineContactManager;
                CollisionGroupManager* pipelineGroupManager;

        };
    }
}

#endif // ZYSINGLETHREADEDCOLLISIONPIPELINE_H
