#ifndef SOFAPBDSEARCHNODEVISITOR_H
#define SOFAPBDSEARCHNODEVISITOR_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ExecParams.h>
#include <framework/sofa/simulation/Visitor.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            template <class NodeType>
            class SofaPBDNodeSearchVisitor: public sofa::simulation::Visitor
            {
                public:
                    SofaPBDNodeSearchVisitor(const sofa::core::ExecParams* params);
                    void doFwd(sofa::simulation::Node* node);

                protected:
                    std::vector<sofa::core::objectmodel::BaseObject*> m_foundObjects;
            };
        }
    }
}

using namespace sofa::core::objectmodel;
using namespace sofa::simulation::PBDSimulation;

template <class NodeType>
SofaPBDNodeSearchVisitor<NodeType>::SofaPBDNodeSearchVisitor(const sofa::core::ExecParams* params)
{

}

template <class NodeType>
void SofaPBDNodeSearchVisitor<NodeType>::doFwd(sofa::simulation::Node* node)
{
    msg_info("ZyDefaultPipeline") << "Node " << node->getName() << " of type " << node->getTypeName() << ", className = " << node->getClassName();
    BaseObject* baseObject = dynamic_cast<BaseObject*>(node);
    msg_info("ZyDefaultPipeline") << "BaseObject " << baseObject->getName() << " of type " << baseObject->getTypeName() << ", className = " << baseObject->getClassName();

    NodeType* nodeObject = dynamic_cast<NodeType*>(baseObject);
    msg_info("ZyDefaultPipeline") << "  NodeType " << nodeObject->getName() << " of type " << nodeObject->getTypeName() << ", className = " << nodeObject->getClassName();

    if (nodeObject)
    {
        m_foundObjects.push_back(nodeObject);
    }
}

#endif // SOFAPBDSEARCHNODEVISITOR_H
