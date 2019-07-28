#ifndef SOFA_COMPONENT_COLLISION_ZyPipeline_H
#define SOFA_COMPONENT_COLLISION_ZyPipeline_H

#include "initZyPipeline.h"

#include "ZyPipelineInterface.h"
#include <sofa/simulation/PipelineImpl.h>

namespace sofa
{
	namespace component
	{
		namespace collision
		{
			using namespace sofa::core::collision;

            class ZY_PIPELINE_API ZyPipeline : public sofa::simulation::PipelineImpl
			{
				public:
                    SOFA_CLASS(ZyPipeline, sofa::simulation::PipelineImpl);

                    ZyPipeline();
                    ~ZyPipeline();
								   
					void init();
					void bwdInit();

					void reset();

					std::set< std::string > getResponseList() const;

                    // -- ZyPipelineInterface
					/// Detect new collisions. Note that this step must not modify the simulation graph
					void doCollisionDetection(const sofa::helper::vector<core::CollisionModel*>& collisionModels);
					/// Add collision response in the simulation graph
					void doCollisionResponse();
					/// Remove collision response from last step
					void doCollisionReset();

					sofa::core::collision::ContactManager* getContactManager() { return contactManager; }
					sofa::core::collision::CollisionGroupManager* getGroupManager() { return groupManager; }

					sofa::core::collision::Intersection* getIntersectionMethod() { return intersectionMethod; }

				protected:
					// All pipeline interfaces
                    sofa::helper::vector<ZyPipelineInterface*> m_pipeline_interfaces;
					// Main pipeline interface (responsible for pipeline call sequence); this replaces the DefaultPipeline
                    ZyPipelineInterface::SPtr m_pipeline;
			};
		}
	}
}

#endif //SOFA_COMPONENT_COLLISION_ZyPipeline_H
