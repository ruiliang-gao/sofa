#ifndef ZYMULTITHREADEDCOLLISIONPIPELINE_H
#define ZYMULTITHREADEDCOLLISIONPIPELINE_H

#include "initZyPipeline.h"

#include <sofa/core/objectmodel/Data.h>

#include <sofa/core/objectmodel/DataFileName.h>
#include <iostream>
#include <fstream>
#include <string>

#include "ZyWorkerThreads/MultiThread_Scheduler.h"
#include "Zy_MultiThread_Tasks_BVHTraversal.h"

#include <SofaMiscCollision/RuleBasedContactManager.h>

#include "ZyPipelineInterface.h"
#include "ZyParallelNarrowPhase.h"

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

namespace Zyklio
{
    namespace Pipeline
    {
        using namespace Zyklio::MultiThreading::Collision;
        using namespace sofa;
        using namespace sofa::core::objectmodel;
        using namespace sofa::core::collision;

        class ZyMultiThreadedCollisionPipelinePrivate;
        class ZY_PIPELINE_API ZyMultiThreadedCollisionPipeline : public sofa::core::collision::ZyPipelineInterface
        {
            public:
                SOFA_CLASS(ZyMultiThreadedCollisionPipeline, sofa::core::collision::ZyPipelineInterface);

                Data<bool> bVerbose;
                Data<int> depth;

                virtual void init();
                virtual void bwdInit();

                void doCollisionDetection(const sofa::helper::vector<core::CollisionModel*>& collisionModels);
                void doCollisionResponse();
                void doCollisionReset();

                void setup(BroadPhaseDetection*, NarrowPhaseDetection*, Intersection*, ContactManager*, CollisionGroupManager*);

                ZyMultiThreadedCollisionPipeline();
                ZyMultiThreadedCollisionPipeline(BroadPhaseDetection*, NarrowPhaseDetection*, Intersection*, ContactManager*, CollisionGroupManager*);
                ~ZyMultiThreadedCollisionPipeline();

                bool isDefaultPipeline() const { return true; }

            protected:
                void filterCollisionModelsToProcess(const sofa::helper::vector<core::CollisionModel*>& collisionModels, sofa::helper::vector<core::CollisionModel*>& processedCollisionModels);

                MultiThread_Scheduler<CPUBVHUpdateTask>* m_scheduler_updateBVH;
                std::vector<CPUBVHUpdateTask*> m_cpuBVHUpdateTasks;

                Data<int> m_numWorkerThreads;

                Intersection* intersectionMethod;
                BroadPhaseDetection* broadPhaseDetection;
                ZyParallelNarrowPhase* narrowPhaseDetection;

                ContactManager* contactManager;
                CollisionGroupManager* groupManager;

            private:
                void zyUpdateInternalGeometry();
                ZyMultiThreadedCollisionPipelinePrivate* m_d;
        };
    } // namespace Pipeline
} // namespace Zyklio

#endif // ZYMULTITHREADEDCOLLISIONPIPELINE_H
