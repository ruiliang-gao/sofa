#ifndef OBBTREEGPU_MULTITHREAD_CPU_TASKS_H
#define OBBTREEGPU_MULTITHREAD_CPU_TASKS_H

#include "Tasks.h"

#include <WorkerThreadIface.h>

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>

#include <SofaBaseCollision/BaseProximityIntersection.h>

using namespace sofa::defaulttype;
namespace sofa
{
	namespace component
	{
		namespace collision
		{
			class ObbTreeGPUCollisionDetection_Threaded;
		}
	}
}
using namespace sofa::component::collision;

namespace sofa
{
	namespace component
	{
		namespace collision
		{
			enum CPUCollisionCheckIntersectorType
			{
				MinProximityIntersection_Type,
				NewProximityIntersection_Type,
				LocalMinDistance_Type,
				LMDNewProximityIntersection_Type,
				CPUCollisionCheckIntersector_Type_Default = MinProximityIntersection_Type
			};

			class CPUCollisionCheckTaskPrivate;
			class CPUCollisionCheckTask : public Zyklio::MultiThreading::PoolTask
			{
				public:
					virtual bool run(Zyklio::MultiThreading::WorkerThreadIface*);

					CPUCollisionCheckTask();
					CPUCollisionCheckTask(const Zyklio::MultiThreading::TaskStatus* status, ObbTreeGPUCollisionDetection_Threaded* collisionDetection, unsigned int numWorkerUnits);
					
					CPUCollisionCheckTask(const CPUCollisionCheckTask&);
					CPUCollisionCheckTask& operator=(const CPUCollisionCheckTask&);

					~CPUCollisionCheckTask();

					void addCollidingPair(const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>&, sofa::core::collision::DetectionOutputVector*&);
					void clearWorkList();

					float getElapsedTime() const { return m_elapsedTime; }
					const std::vector<float>& getElapsedTimePerTest() const { return m_elapsedTimePerTest; }

					std::vector<core::collision::DetectionOutputVector*>& getDetectionOutputs() { return _detectionOutputs; }
					std::vector<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& getCollidingPairs() { return _collidingPairs; }

				protected:
					std::vector<sofa::core::collision::DetectionOutputVector*> _detectionOutputs;
					std::vector<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> > _collidingPairs;
					unsigned int _numWorkerUnits;

					ObbTreeGPUCollisionDetection_Threaded* _collisionDetection;

					void processCollidingPair(const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>&, sofa::core::collision::DetectionOutputVector*&);


					float m_elapsedTime;
					std::vector<float> m_elapsedTimePerTest;

					CPUCollisionCheckIntersectorType m_intersectorType;
					CPUCollisionCheckTaskPrivate* m_d;
			};
		}
	}
}

#endif