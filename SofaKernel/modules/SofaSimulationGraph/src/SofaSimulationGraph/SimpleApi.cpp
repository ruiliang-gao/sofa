/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/Node.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;

#include <SofaSimulationGraph/DAGNode.h>
using sofa::simulation::graph::DAGNode;

#include <SofaSimulationGraph/SimpleApi.h>
using sofa::core::objectmodel::BaseObjectDescription ;

#include <sofa/simulation/XMLPrintVisitor.h>
using sofa::simulation::XMLPrintVisitor ;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

namespace sofa::simpleapi
{

bool importPlugin(const std::string& name)
{
    return PluginManager::getInstance().loadPlugin(name) ;
}

void dumpScene(Node::SPtr root)
{
    XMLPrintVisitor p(sofa::core::execparams::defaultInstance(), std::cout) ;
    p.execute(root.get()) ;
}

Simulation::SPtr createSimulation(const std::string& type)
{
    if(type!="DAG")
    {
        msg_error("SimpleApi") << "Unable to create simulation of type '"<<type<<"'. Supported type is ['DAG']";
        return nullptr ;
    }

    return new simulation::graph::DAGSimulation() ;
}


Node::SPtr createRootNode(Simulation::SPtr s, const std::string& name,
                                              const std::map<std::string, std::string>& params)
{
    Node::SPtr root = s->createNewNode(name) ;

    BaseObjectDescription desc(name.c_str(), "Node");
    for(auto& kv : params)
    {
        desc.setAttribute(kv.first.c_str(), kv.second);
    }
    root->parse(&desc) ;

    return root ;
}

BaseObject::SPtr createObject(Node::SPtr parent, BaseObjectDescription& desc)
{
    /// Create the object.
    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(parent.get(), &desc);
    if (obj==nullptr)
    {
        std::stringstream msg;
        msg << "Component '" << desc.getName() << "' of type '" << desc.getAttribute("type","") << "' failed:" << msgendl ;
        for (std::vector< std::string >::const_iterator it = desc.getErrors().begin(); it != desc.getErrors().end(); ++it)
            msg << " " << *it << msgendl ;
        msg_error(parent.get()) << msg.str() ;
        return nullptr;
    }

    return obj ;
}

BaseObject::SPtr createObject(Node::SPtr parent, const std::string& type, const std::map<std::string, std::string>& params)
{
    /// temporarily, the name is set to the type name.
    /// if a "name" parameter is provided, it will overwrite it.
    BaseObjectDescription desc(type.c_str(),type.c_str());
    for(auto& kv : params)
    {
        desc.setAttribute(kv.first.c_str(), kv.second);
    }

    return createObject(parent, desc);
}

Node::SPtr createChild(Node::SPtr& node, const std::string& name, const std::map<std::string, std::string>& params)
{
    BaseObjectDescription desc(name.c_str(), "Node");
    for(auto& kv : params)
    {
        desc.setAttribute(kv.first.c_str(), kv.second);
    }
    return createChild(node, desc);
}

Node::SPtr createChild(Node::SPtr node, BaseObjectDescription& desc)
{
    Node::SPtr tmp = node->createChild(desc.getName());
    tmp->parse(&desc);
    return tmp;
}

Node::SPtr createNode(const std::string& name)
{
    return core::objectmodel::New<DAGNode>(name);
}

} // namespace sofa::simpleapi
