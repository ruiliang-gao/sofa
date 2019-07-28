#include "ZyBasePipeline.h"

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>

#include <sofa/simulation/Simulation.h>

using namespace sofa;
using namespace sofa::component::collision;

SOFA_DECL_CLASS(ZyPipeline)

int ZyPipelineClass = sofa::core::RegisterObject("Zykl.io collision pipeline wrapper")
.add< ZyPipeline >()
;

ZyPipeline::ZyPipeline() : sofa::simulation::PipelineImpl()
{
	
}


ZyPipeline::~ZyPipeline()
{
}

void ZyPipeline::reset()
{
	PipelineImpl::reset();

	if (m_pipeline)
	{
		m_pipeline->reset();

		for (size_t k = 0; k < m_pipeline_interfaces.size(); ++k)
			m_pipeline_interfaces[k]->reset();
	}
}

void ZyPipeline::init()
{
	PipelineImpl::init();
    msg_info("ZyBasePipeline") << "--- PipelineImpl members after init ---";
    msg_info("ZyBasePipeline") << " narrowPhaseDetection: memory location = " << this->narrowPhaseDetection;
	if (this->narrowPhaseDetection)
        msg_info("ZyBasePipeline") << "  named " << this->narrowPhaseDetection->getName() << " of type " << this->narrowPhaseDetection->getTypeName();

    msg_info("ZyBasePipeline") << " broadPhaseDetection: memory location = " << this->broadPhaseDetection;
	if (this->broadPhaseDetection)
        msg_info("ZyBasePipeline") << "  named " << this->broadPhaseDetection->getName() << " of type " << this->broadPhaseDetection->getTypeName();

    msg_info("ZyBasePipeline") << " intersectionMethod: memory location = " << this->intersectionMethod;
	if (this->intersectionMethod)
        msg_info("ZyBasePipeline") << "  named " << this->intersectionMethod->getName() << " of type " << this->intersectionMethod->getTypeName();

    msg_info("ZyBasePipeline") << " contactManager: memory location = " << this->contactManager;
	if (this->contactManager)
        msg_info("ZyBasePipeline") << "  named " << this->contactManager->getName() << " of type " << this->contactManager->getTypeName();

    msg_info("ZyBasePipeline") << " groupManager: memory location = " << this->groupManager;
	if (this->groupManager)
        msg_info("ZyBasePipeline") << "  named " << this->groupManager->getName() << " of type " << this->groupManager->getTypeName();

    msg_info("ZyBasePipeline") << "--- PipelineImpl members after init ---";
}

void ZyPipeline::bwdInit()
{
    msg_info("ZyBasePipeline") << "ZyPipeline::bwdInit()";
	sofa::simulation::PipelineImpl::bwdInit();
	
	sofa::simulation::Node::SPtr root = sofa::simulation::getSimulation()->getCurrentRootNode();	
	if (root)
	{
		m_pipeline_interfaces.clear();
        root->getTreeObjects<ZyPipelineInterface>(&m_pipeline_interfaces);
		
        msg_info("ZyBasePipeline") << "ZyPipelineInterface instances in SOFA scene graph: " << m_pipeline_interfaces.size();

		if (m_pipeline_interfaces.size() > 0)
		{
            sofa::helper::vector<ZyPipelineInterface*> temp_pipeline_interfaces;

			for (size_t k = 0; k < m_pipeline_interfaces.size(); ++k)
			{
				temp_pipeline_interfaces.push_back(m_pipeline_interfaces[k]);
			}
			m_pipeline_interfaces.clear();

			for (size_t k = 0; k < temp_pipeline_interfaces.size(); ++k)
			{	
				// Do delayed init operations now
				temp_pipeline_interfaces[k]->setActive(true);
				temp_pipeline_interfaces[k]->init();

				temp_pipeline_interfaces[k]->setup(this->broadPhaseDetection, this->narrowPhaseDetection, this->intersectionMethod, this->contactManager, this->groupManager);

				temp_pipeline_interfaces[k]->bwdInit();

				// Call setActive(true) twice: init() might reset it to false.
				temp_pipeline_interfaces[k]->setActive(true);

                msg_info("ZyBasePipeline") << " - object " << k << ": " << temp_pipeline_interfaces[k]->getName() << " of type " << temp_pipeline_interfaces[k]->getTypeName();

				if (temp_pipeline_interfaces[k]->isDefaultPipeline())
				{
					m_pipeline = temp_pipeline_interfaces[k];
                    msg_info("ZyBasePipeline") << "  -> claims to be our 'DefaultPipeline'";
				}
				else
				{
					m_pipeline_interfaces.push_back(temp_pipeline_interfaces[k]);
                    msg_info("ZyBasePipeline") << "  -> Normal ZyPipelineInterface implementation";
				}
			}
		}
	}
}

std::set< std::string > ZyPipeline::getResponseList() const
{
	std::set< std::string > listResponse;
	core::collision::Contact::Factory::iterator it;
	for (it = core::collision::Contact::Factory::getInstance()->begin(); it != core::collision::Contact::Factory::getInstance()->end(); ++it)
	{
		listResponse.insert(it->first);
	}
	return listResponse;
}

void ZyPipeline::doCollisionDetection(const sofa::helper::vector<core::CollisionModel*>& collisionModels)
{
    msg_info("ZyPipeline") << "doCollisionDetection()";
	if (m_pipeline)
	{
        msg_info("ZyPipeline") << "m_pipeline instance valid: " << m_pipeline->getName();
        msg_info("ZyPipeline") << "CollisionModel instances in scene: " << collisionModels.size();

		m_pipeline->doCollisionDetection(collisionModels);

		for (size_t k = 0; k < m_pipeline_interfaces.size(); ++k)
			m_pipeline_interfaces[k]->doCollisionDetection(collisionModels);
	}
    else
    {
        msg_warning("ZyPipeline") << "m_pipeline instance INVALID! Collision detection not functional.";
    }
}

void ZyPipeline::doCollisionResponse()
{
    msg_info("ZyPipeline") << "doCollisionResponse()";
	if (m_pipeline)
	{
        msg_info("ZyPipeline") << "m_pipeline instance valid: " << m_pipeline->getName();
		m_pipeline->doCollisionResponse();

		for (size_t k = 0; k < m_pipeline_interfaces.size(); ++k)
			m_pipeline_interfaces[k]->doCollisionResponse();
	}
    else
    {
        msg_warning("ZyPipeline") << "m_pipeline instance INVALID! Collision response not functional.";
    }
}

void ZyPipeline::doCollisionReset()
{
    msg_info("ZyPipeline") << "doCollisionReset()";
	if (m_pipeline)
	{
        msg_info("ZyPipeline") << "m_pipeline instance valid: " << m_pipeline->getName();
		m_pipeline->doCollisionReset();

		for (size_t k = 0; k < m_pipeline_interfaces.size(); ++k)
			m_pipeline_interfaces[k]->doCollisionReset();
	}
    else
    {
        msg_warning("ZyPipeline") << "m_pipeline instance INVALID! Collision reset not functional.";
    }
}
