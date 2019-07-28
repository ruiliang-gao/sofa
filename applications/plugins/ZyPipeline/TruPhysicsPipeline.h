/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef TRUPHYSICS_PIPELINE_TRUPHYSICSPIPELINE_H
#define TRUPHYSICS_PIPELINE_TRUPHYSICSPIPELINE_H

#include "initTruPhysicsPipelinePlugin.h"

#include <sofa/core/objectmodel/Data.h>

#include <sofa/core/objectmodel/DataFileName.h>
#include <iostream>
#include <fstream>
#include <string>

#include "MultiThreading/MultiThread_Scheduler.h"
#include "Pipeline/TruPhysics_MultiThread_Tasks_BVHTraversal.h"

#include <SofaMiscCollision/RuleBasedContactManager.h>

#include <sofa/core/collision/TruPipelineInterface.h>

#include "TruParallelNarrowPhase.h"

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

namespace TruPhysics
{
	namespace Pipeline
	{
		using namespace TruPhysics::MultiThreading::Collision;
		using namespace sofa;
		using namespace sofa::core::collision;

		class TruPhysicsPipelinePrivate;
		class SOFA_TRUPHYSICSPIPELINE_API TruPhysicsPipeline : public sofa::core::collision::TruPipelineInterface
		{
			public:
				SOFA_CLASS(TruPhysicsPipeline, sofa::core::collision::TruPipelineInterface);

				Data<bool> bVerbose;
				Data<int> depth;

				virtual void init();
				virtual void bwdInit();

				void doCollisionDetection(const sofa::helper::vector<core::CollisionModel*>& collisionModels);
				void doCollisionResponse();
				void doCollisionReset();

				void setup(BroadPhaseDetection*, NarrowPhaseDetection*, Intersection*, ContactManager*, CollisionGroupManager*);
										 
				TruPhysicsPipeline();
				TruPhysicsPipeline(BroadPhaseDetection*, NarrowPhaseDetection*, Intersection*, ContactManager*, CollisionGroupManager*);
				~TruPhysicsPipeline();
																		
				bool isDefaultPipeline() const { return true; }

			protected:
				MultiThread_Scheduler<CPUBVHUpdateTask>* m_scheduler_updateBVH;
				std::vector<CPUBVHUpdateTask*> m_cpuBVHUpdateTasks;

				Data<int> m_numWorkerThreads;

				Intersection* intersectionMethod;
				BroadPhaseDetection* broadPhaseDetection;
				TruParallelNarrowPhase* narrowPhaseDetection;

				ContactManager* contactManager;
				CollisionGroupManager* groupManager;

			private:
				void tpUpdateInternalGeometry();
				TruPhysicsPipelinePrivate* m_d;
		};
	} // namespace Pipeline
} // namespace TruPhysics

#endif // TRUPHYSICS_PIPELINE_TRUPHYSICSPIPELINE_H
