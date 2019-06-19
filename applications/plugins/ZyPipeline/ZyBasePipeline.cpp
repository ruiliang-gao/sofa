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

int ZyPipelineClass = sofa::core::RegisterObject("TruPhysics pipeline wrapper")
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
	std::cout << "--- PipelineImpl members after init ---" << std::endl;
	std::cout << " narrowPhaseDetection: memory location = " << this->narrowPhaseDetection << std::endl;
	if (this->narrowPhaseDetection)
		std::cout << "  named " << this->narrowPhaseDetection->getName() << " of type " << this->narrowPhaseDetection->getTypeName() << std::endl;

	std::cout << " broadPhaseDetection: memory location = " << this->broadPhaseDetection << std::endl;
	if (this->broadPhaseDetection)
		std::cout << "  named " << this->broadPhaseDetection->getName() << " of type " << this->broadPhaseDetection->getTypeName() << std::endl;

	std::cout << " intersectionMethod: memory location = " << this->intersectionMethod << std::endl;
	if (this->intersectionMethod)
		std::cout << "  named " << this->intersectionMethod->getName() << " of type " << this->intersectionMethod->getTypeName() << std::endl;

	std::cout << " contactManager: memory location = " << this->contactManager << std::endl;
	if (this->contactManager)
		std::cout << "  named " << this->contactManager->getName() << " of type " << this->contactManager->getTypeName() << std::endl;

	std::cout << " groupManager: memory location = " << this->groupManager << std::endl;
	if (this->groupManager)
		std::cout << "  named " << this->groupManager->getName() << " of type " << this->groupManager->getTypeName() << std::endl;

	std::cout << "--- PipelineImpl members after init ---" << std::endl;
}

void ZyPipeline::bwdInit()
{
    std::cout << "ZyPipeline::bwdInit()" << std::endl;
	sofa::simulation::PipelineImpl::bwdInit();
	
	sofa::simulation::Node::SPtr root = sofa::simulation::getSimulation()->getCurrentRootNode();	
	if (root)
	{
		m_pipeline_interfaces.clear();
        root->getTreeObjects<ZyPipelineInterface>(&m_pipeline_interfaces);
		
        std::cout << "ZyPipelineInterface instances in SOFA scene graph: " << m_pipeline_interfaces.size() << std::endl;

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

				std::cout << " - object " << k << ": " << temp_pipeline_interfaces[k]->getName() << " of type " << temp_pipeline_interfaces[k]->getTypeName() << std::endl;

				if (temp_pipeline_interfaces[k]->isDefaultPipeline())
				{
					m_pipeline = temp_pipeline_interfaces[k];
					std::cout << "  -> claims to be our 'DefaultPipeline'" << std::endl;
				}
				else
				{
					m_pipeline_interfaces.push_back(temp_pipeline_interfaces[k]);
                    std::cout << "  -> Normal ZyPipelineInterface implementation" << std::endl;
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
	if (m_pipeline)
	{
		m_pipeline->doCollisionDetection(collisionModels);

		for (size_t k = 0; k < m_pipeline_interfaces.size(); ++k)
			m_pipeline_interfaces[k]->doCollisionDetection(collisionModels);
	}
}

void ZyPipeline::doCollisionResponse()
{
	if (m_pipeline)
	{
		m_pipeline->doCollisionResponse();

		for (size_t k = 0; k < m_pipeline_interfaces.size(); ++k)
			m_pipeline_interfaces[k]->doCollisionResponse();
	}
}

void ZyPipeline::doCollisionReset()
{
	if (m_pipeline)
	{
		m_pipeline->doCollisionReset();

		for (size_t k = 0; k < m_pipeline_interfaces.size(); ++k)
			m_pipeline_interfaces[k]->doCollisionReset();
	}
}
