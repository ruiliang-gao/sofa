#ifndef TruPhysics_MULTITHREAD_TASKS_BVHTRAVERSAL_H
#define TruPhysics_MULTITHREAD_TASKS_BVHTRAVERSAL_H

#include <sofa/defaulttype/Vec3Types.h>

#include <ZyWorkerThreads/Tasks.h>
#include <ZyWorkerThreads/WorkerThreadIface.h>

#include <sofa/config.h>
#include <vector>

#include "config_zypipeline_tasks.h"

#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/collision/Intersection.h>

namespace Zyklio
{
	namespace MultiThreading
	{
		namespace Collision
		{
            class ZY_PIPELINE_TASKS_API CPUBVHUpdateTask : public PoolTask
			{
				public:
					virtual bool run(WorkerThreadIface*);

					CPUBVHUpdateTask();
					CPUBVHUpdateTask(const TaskStatus* status, unsigned int numWorkerUnits);

					CPUBVHUpdateTask(const CPUBVHUpdateTask&);
					CPUBVHUpdateTask& operator=(const CPUBVHUpdateTask&);

					~CPUBVHUpdateTask();
			
					void setDt(const SReal);
					void setContinuous(const bool);
					void setUsedDepth(const int);

					void addBVHUpdateTask(sofa::core::CollisionModel*);
					void clearWorkList();

					std::vector<sofa::core::CollisionModel*>& getCollisionModels() { return m_collisionModels; }

					float getElapsedTime() const { return m_elapsedTime; }
					const std::vector<float>& getElapsedTimePerTest() const { return m_elapsedTimePerTest; }

				private:
					unsigned int m_numWorkerUnits;

					int m_usedDepth;
					SReal m_dt;
					bool m_continuous;

					std::vector<sofa::core::CollisionModel*> m_collisionModels;

					float m_elapsedTime;
					std::vector<float> m_elapsedTimePerTest;
			};

			enum CPUCollisionCheckIntersectorType
			{
				MinProximityIntersection_Type,
				NewProximityIntersection_Type,
				LocalMinDistance_Type,
				LMDNewProximityIntersection_Type,
				CPUCollisionCheckIntersector_Type_Default = MinProximityIntersection_Type
			};

			class CPUCollisionCheckTaskPrivate;
            class ZY_PIPELINE_TASKS_API CPUCollisionCheckTask : public PoolTask
			{
				public:
					virtual bool run(WorkerThreadIface*);

					CPUCollisionCheckTask();
					CPUCollisionCheckTask(const TaskStatus* status, unsigned int numWorkerUnits);

					CPUCollisionCheckTask(const CPUCollisionCheckTask&);
					CPUCollisionCheckTask& operator=(const CPUCollisionCheckTask&);

					~CPUCollisionCheckTask();

					void addCollidingPair(const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>&, sofa::core::collision::DetectionOutputVector*&);
					void clearWorkList();

					float getElapsedTime() const { return m_elapsedTime; }
					const std::vector<float>& getElapsedTimePerTest() const { return m_elapsedTimePerTest; }

					std::vector<sofa::core::collision::DetectionOutputVector*>& getDetectionOutputs() { return _detectionOutputs; }
					std::vector<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& getCollidingPairs() { return _collidingPairs; }

				protected:
					std::vector<sofa::core::collision::DetectionOutputVector*> _detectionOutputs;
					std::vector<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> > _collidingPairs;
                    unsigned int m_numWorkerUnits;

					void processCollidingPair(const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>&, sofa::core::collision::DetectionOutputVector*&);


					float m_elapsedTime;
					std::vector<float> m_elapsedTimePerTest;

					CPUCollisionCheckIntersectorType m_intersectorType;
					CPUCollisionCheckTaskPrivate* m_d;
			};

			// Duplicate from BruteForceDetection
            class ZY_PIPELINE_TASKS_API MirrorIntersector : public sofa::core::collision::ElementIntersector
			{
				public:
					sofa::core::collision::ElementIntersector* intersector;

					/// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
					virtual bool canIntersect(sofa::core::CollisionElementIterator elem1, sofa::core::CollisionElementIterator elem2)
					{
						return intersector->canIntersect(elem2, elem1);
					}

					/// Begin intersection tests between two collision models. Return the number of contacts written in the contacts vector.
					/// If the given contacts vector is NULL, then this method should allocate it.
					virtual int beginIntersect(sofa::core::CollisionModel* model1, sofa::core::CollisionModel* model2, sofa::core::collision::DetectionOutputVector*& contacts)
					{
						return intersector->beginIntersect(model2, model1, contacts);
					}

					/// Compute the intersection between 2 elements. Return the number of contacts written in the contacts vector.
					virtual int intersect(sofa::core::CollisionElementIterator elem1, sofa::core::CollisionElementIterator elem2, sofa::core::collision::DetectionOutputVector* contacts)
					{
						return intersector->intersect(elem2, elem1, contacts);
					}

					/// End intersection tests between two collision models. Return the number of contacts written in the contacts vector.
					virtual int endIntersect(sofa::core::CollisionModel* model1, sofa::core::CollisionModel* model2, sofa::core::collision::DetectionOutputVector* contacts)
					{
						return intersector->endIntersect(model2, model1, contacts);
					}

					virtual std::string name() const
					{
						return intersector->name() + std::string("<SWAP>");
					}
			};
		}
	}
}

#endif //TruPhysics_MULTITHREAD_TASKS_BVHTRAVERSAL_H
