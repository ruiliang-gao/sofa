#ifndef ZY_PARALLEL_NARROWPHASE_H
#define ZY_PARALLEL_NARROWPHASE_H

#include <sofa/core/objectmodel/BaseObject.h>

#include "initZyPipeline.h"

#include <sofa/core/collision/NarrowPhaseDetection.h>

#include "ZyWorkerThreads//MultiThread_Scheduler.h"
#include "Zy_MultiThread_Tasks_BVHTraversal.h"

using namespace Zyklio::MultiThreading;
using namespace Zyklio::MultiThreading::Collision;

namespace Zyklio
{
    namespace Pipeline
    {
        class ZY_PIPELINE_API ZyParallelNarrowPhase : public sofa::core::collision::NarrowPhaseDetection
        {
            public:
                ZyParallelNarrowPhase(const unsigned int& = 4);

                ~ZyParallelNarrowPhase();

                /// Clear all the potentially colliding pairs detected in the previous simulation step
                void beginNarrowPhase();

                /// Add a new potentially colliding pair of models
                void addCollisionPair(const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>& cmPair);

                /// Add a new list of potentially colliding pairs of models
                void addCollisionPairs(const sofa::helper::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& v);

                void endNarrowPhase();

                NarrowPhaseDetection::DetectionOutputMap& getMutableDetectionOutputs()
                {
                    return *m_detectionOutputVectors;
                }

                void bwdInit();

            protected:
                unsigned int m_numWorkerThreads;

                sofa::helper::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> > m_narrowPhasePairs;

                std::map<unsigned int, std::vector<std::pair<std::string, std::string> > > m_narrowPhasePairs_TaskAssignment;
                std::map<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>, std::pair<std::string, std::string> > m_narrowPhasePairs_ModelAssociations;

                NarrowPhaseDetection::DetectionOutputMap* m_detectionOutputVectors;

                MultiThread_Scheduler<CPUCollisionCheckTask>* m_scheduler_traverseBVH;
                std::vector<CPUCollisionCheckTask*> m_cpuBVHTraversalTasks;

                /// Search for existing IntersectionMethod instances in the scene graph
                void searchIntersectionMethodInstances();
            };
    }
}

#endif // ZY_PARALLEL_NARROWPHASE_H
