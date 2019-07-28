#ifndef SOFA_CORE_COLLISION_ZYPIPELINEINTERFACE_H
#define SOFA_CORE_COLLISION_ZYPIPELINEINTERFACE_H

#include "initZyPipeline.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/vector.h>
#include <sofa/core/CollisionModel.h>

#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>

namespace sofa
{
	namespace core
	{
		namespace collision
		{
            class ZY_PIPELINE_API ZyPipelineInterface : public sofa::core::objectmodel::BaseObject
			{
				public:
					SOFA_CLASS(ZyPipelineInterface, sofa::core::objectmodel::BaseObject);

					ZyPipelineInterface();
					~ZyPipelineInterface();

					virtual void reset();

					/// get the set of response available with the current collision pipeline
					virtual std::set< std::string > getResponseList() const;

                    virtual void doCollisionDetection(const sofa::helper::vector<core::CollisionModel*>&) {}
                    virtual void doCollisionResponse() {}
                    virtual void doCollisionReset() {}

                    // TODO: Find a better way to cope with the double inheritance hierarchy (PipelineImpl vs. ZyPipelineInterface)
					virtual void setup(BroadPhaseDetection*, NarrowPhaseDetection*, Intersection*, ContactManager*, CollisionGroupManager*) {}

					virtual bool isDefaultPipeline() const { return false; }

					bool isActive() const;
					void setActive(const bool);

				protected:
					bool m_doInit;
			};
		}
	}
}

#endif //SOFA_CORE_COLLISION_ZYPIPELINEINTERFACE_H
