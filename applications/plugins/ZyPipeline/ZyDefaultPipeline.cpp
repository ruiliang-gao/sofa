#include "ZyDefaultPipeline.h"

#include <sofa/core/collision/Contact.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>

#include <sofa/core/ObjectFactory.h>
#include <SofaSimulationGraph/DAGNode.h>

using namespace Zyklio::Pipeline;

SOFA_DECL_CLASS(TruDefaultPipeline)
int ZyDefaultPipelineClass = sofa::core::RegisterObject("TruPhysics DefaultPipeline alternative")
.add< ZyDefaultPipeline >()
;

/*
template <class NodeType>
TruPhysicsNodeSearchVisitor<NodeType>::TruPhysicsNodeSearchVisitor(const sofa::core::ExecParams* params) : TruPhysicsVisitor(params)
{

}

template <class NodeType>
void TruPhysicsNodeSearchVisitor<NodeType>::doFwd(sofa::simulation::Node* node)
{
	std::cout << "  node " << node->getName() << " of type " << node->getTypeName() << ", className = " << node->getClassName() << std::endl;
	BaseObject* baseObject = dynamic_cast<BaseObject*>(node);
	std::cout << "  BaseObject " << baseObject->getName() << " of type " << baseObject->getTypeName() << ", className = " << baseObject->getClassName() << std::endl;

	NodeType* nodeObject = dynamic_cast<NodeType*>(baseObject);
	std::cout << "  NodeType " << nodeObject->getName() << " of type " << nodeObject->getTypeName() << ", className = " << nodeObject->getClassName() << std::endl;

	if (nodeObject)
	{
		m_foundObjects.push_back(nodeObject);
	}
}
*/

ZyDefaultPipeline::ZyDefaultPipeline() : sofa::core::collision::Pipeline(), m_operationMode(PIPELINE_MODE_DEFAULT)
{
	std::cout << "TruDefaultPipeline::TruDefaultPipeline()" << std::endl;
}

ZyDefaultPipeline::~ZyDefaultPipeline()
{
	std::cout << "TruDefaultPipeline::~TruDefaultPipeline()" << std::endl;
}

void ZyDefaultPipeline::reset()
{
	if (m_operationMode == PIPELINE_MODE_INVALID)
		return;

	if (m_operationMode == PIPELINE_MODE_SINGLE_THREADED)
	{
		m_default_pipelines[0]->reset();
	}
	else if (m_operationMode == PIPELINE_MODE_MULTI_THREADED_TASKS)
	{
        m_zyPipelines[0]->reset();
	}
}

void ZyDefaultPipeline::bwdInit()
{
    std::cout << "ZyPipeline::bwdInit()" << std::endl;
	sofa::simulation::Node::SPtr root = sofa::simulation::getSimulation()->getCurrentRootNode();
	if (root)
	{
		sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<DefaultPipeline, std::vector<DefaultPipeline* > > dp_cb(&m_default_pipelines);
		getContext()->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::collision::DefaultPipeline>::get(), dp_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchRoot);

        sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<ZyPipeline, std::vector<ZyPipeline* > > tp_cb(&m_zyPipelines);
        getContext()->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::collision::ZyPipeline>::get(), tp_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchRoot);

		std::cout << " DefaultPipeline instances (visitor search): " << m_default_pipelines.size() << std::endl;
        std::cout << " ZyPipeline     instances (visitor search): " << m_zyPipelines.size() << std::endl;
		
        if (m_zyPipelines.size() > 0)
		{
            if (m_zyPipelines.size() == 1)
			{
                std::cout << " One ZyPipeline instance detected; using multi-threaded ZyPipeline instance." << std::endl;
				m_operationMode = PIPELINE_MODE_MULTI_THREADED_TASKS;
			}
			else
			{
                std::cout << " More than one ZyPipeline in scene graph. Should not happen! Assuming first found ZyPipeline instance works." << std::endl;
				m_operationMode = PIPELINE_MODE_MULTI_THREADED_TASKS;
			}
		}
		else
		{
			if (m_default_pipelines.size() == 0)
			{
                std::cout << " Neither DefaultPipeline nor ZyPipeline instances found in scene graph: Set PIPELINE_MODE_INVALID" << std::endl;
				m_operationMode = PIPELINE_MODE_INVALID;
			}
			else if (m_default_pipelines.size() == 1)
			{
				std::cout << " One DefaultPipeline instance detected; using single-threaded DefaultPipeline instance." << std::endl;
				m_operationMode = PIPELINE_MODE_SINGLE_THREADED;
			}
			else
			{
				std::cout << " More than one DefaultPipeline in scene graph. Should not happen! Assuming first found DefaultPipeline instance works." << std::endl;
				m_operationMode = PIPELINE_MODE_SINGLE_THREADED;
			}
		}
	}
}

void ZyDefaultPipeline::computeCollisionReset()
{
	if (m_operationMode == PIPELINE_MODE_INVALID)
		return;

	if (m_operationMode == PIPELINE_MODE_SINGLE_THREADED)
	{
		m_default_pipelines[0]->computeCollisionReset();
	}
	else if (m_operationMode == PIPELINE_MODE_MULTI_THREADED_TASKS)
	{
        m_zyPipelines[0]->computeCollisionReset();
	}
}

void ZyDefaultPipeline::computeCollisionDetection()
{
	if (m_operationMode == PIPELINE_MODE_INVALID)
		return;

	if (m_operationMode == PIPELINE_MODE_SINGLE_THREADED)
	{
		m_default_pipelines[0]->computeCollisionDetection();
	}
	else if (m_operationMode == PIPELINE_MODE_MULTI_THREADED_TASKS)
	{
        m_zyPipelines[0]->computeCollisionDetection();
	}
}

void ZyDefaultPipeline::computeCollisionResponse()
{
	if (m_operationMode == PIPELINE_MODE_INVALID)
		return;

	if (m_operationMode == PIPELINE_MODE_SINGLE_THREADED)
	{
		m_default_pipelines[0]->computeCollisionResponse();
	}
	else if (m_operationMode == PIPELINE_MODE_MULTI_THREADED_TASKS)
	{
        m_zyPipelines[0]->computeCollisionResponse();
	}
}

// Borrowed from DefaultPipeline; small copy/paste maneuver should be OK
std::set< std::string > ZyDefaultPipeline::getResponseList() const
{
	std::set< std::string > listResponse;
	sofa::core::collision::Contact::Factory::iterator it;
	for (it = sofa::core::collision::Contact::Factory::getInstance()->begin(); it != sofa::core::collision::Contact::Factory::getInstance()->end(); ++it)
	{
		listResponse.insert(it->first);
	}
	return listResponse;
}

void ZyDefaultPipeline::doCollisionReset()
{
	if (m_operationMode == PIPELINE_MODE_INVALID)
		return;

	if (m_operationMode == PIPELINE_MODE_SINGLE_THREADED)
	{
		m_default_pipelines[0]->doCollisionReset();
	}
	else if (m_operationMode == PIPELINE_MODE_MULTI_THREADED_TASKS)
	{
        m_zyPipelines[0]->doCollisionReset();
	}
}

void ZyDefaultPipeline::doCollisionDetection(const sofa::helper::vector<sofa::core::CollisionModel*>& collisionModels)
{
	if (m_operationMode == PIPELINE_MODE_INVALID)
		return;

	if (m_operationMode == PIPELINE_MODE_SINGLE_THREADED)
	{
		m_default_pipelines[0]->doCollisionDetection(collisionModels);
	}
	else if (m_operationMode == PIPELINE_MODE_MULTI_THREADED_TASKS)
	{
        m_zyPipelines[0]->doCollisionDetection(collisionModels);
	}
}

void ZyDefaultPipeline::doCollisionResponse()
{
	if (m_operationMode == PIPELINE_MODE_INVALID)
		return;

	if (m_operationMode == PIPELINE_MODE_SINGLE_THREADED)
	{
		m_default_pipelines[0]->doCollisionResponse();
	}
	else if (m_operationMode == PIPELINE_MODE_MULTI_THREADED_TASKS)
	{
        m_zyPipelines[0]->doCollisionResponse();
	}
}
