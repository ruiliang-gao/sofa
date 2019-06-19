#include "ZyPipelineInterface.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Contact.h>

namespace sofa
{
	namespace core
	{
		namespace collision
		{
            SOFA_DECL_CLASS(ZyPipelineInterface)

			using namespace sofa;

            ZyPipelineInterface::ZyPipelineInterface() : sofa::core::objectmodel::BaseObject(), m_doInit(false)
			{

			}
			
            ZyPipelineInterface::~ZyPipelineInterface()
			{

			}

            void ZyPipelineInterface::reset()
			{
				m_doInit = false;
			}

            bool ZyPipelineInterface::isActive() const
			{
				return m_doInit;
			}

            void ZyPipelineInterface::setActive(const bool val)
			{
				m_doInit = val;
			}
			
            std::set< std::string > ZyPipelineInterface::getResponseList() const
			{
				std::set< std::string > listResponse;
				sofa::core::collision::Contact::Factory::iterator it;
				for (it = sofa::core::collision::Contact::Factory::getInstance()->begin(); it != sofa::core::collision::Contact::Factory::getInstance()->end(); ++it)
				{
					listResponse.insert(it->first);
				}
				return listResponse;
			}

            int ZyPipelineInterfaceClass = sofa::core::RegisterObject("Zykl.io Pipeline interface")
                .add< ZyPipelineInterface >();
		}
	}
}
