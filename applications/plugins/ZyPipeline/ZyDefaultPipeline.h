#ifndef ZY_PIPELINE_DEFAULTPIPELINE_H
#define ZY_PIPELINE_DEFAULTPIPELINE_H

#include <sofa/core/objectmodel/BaseObject.h>

#include "initZyPipeline.h"

#include <sofa/simulation/PipelineImpl.h>
#include <SofaBaseCollision/DefaultPipeline.h>

#include <ZyBasePipeline.h>

namespace Zyklio
{
	namespace Pipeline
	{
		using namespace sofa::component::collision;

		enum PipelineMode
		{
			PIPELINE_MODE_SINGLE_THREADED = 0,
			PIPELINE_MODE_MULTI_THREADED_TASKS = 1,
			PIPELINE_MODE_INVALID = 2,
			PIPELINE_MODE_DEFAULT = PIPELINE_MODE_SINGLE_THREADED
		};

        class ZY_PIPELINE_API ZyDefaultPipeline : public sofa::core::collision::Pipeline
		{
			public:
                SOFA_CLASS(ZyDefaultPipeline, sofa::core::collision::Pipeline);

                ZyDefaultPipeline();
                ~ZyDefaultPipeline();

				void reset();
				void bwdInit();

				/// Remove collision response from last step
				void computeCollisionReset();
				/// Detect new collisions. Note that this step must not modify the simulation graph
				void computeCollisionDetection();
				/// Add collision response in the simulation graph
				void computeCollisionResponse();

				/// get the set of response available with the current collision pipeline
				std::set< std::string > getResponseList() const;
			
			protected:
                void filterCollisionModelsToProcess(const sofa::helper::vector<sofa::core::CollisionModel*>& collisionModels, sofa::helper::vector<sofa::core::CollisionModel*>& processedCollisionModels);

				// -- Pipeline interface
				/// Remove collision response from last step
				void doCollisionReset();
				/// Detect new collisions. Note that this step must not modify the simulation graph
				void doCollisionDetection(const sofa::helper::vector<sofa::core::CollisionModel*>& collisionModels);
				/// Add collision response in the simulation graph
				void doCollisionResponse();

				sofa::helper::vector<DefaultPipeline*> m_default_pipelines;
                sofa::helper::vector<ZyPipeline*> m_zyPipelines;

				PipelineMode m_operationMode;

		};
	}
}

#endif //ZY_PIPELINE_DEFAULTPIPELINE_H
