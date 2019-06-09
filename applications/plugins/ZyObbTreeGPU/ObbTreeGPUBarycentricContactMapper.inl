#ifndef OBBTREEGPUBARYCENTRICCONTACTMAPPER_INL
#define OBBTREEGPUBARYCENTRICCONTACTMAPPER_INL

#include "ObbTreeGPUBarycentricContactMapper.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <iostream>

namespace sofa
{
    namespace component
    {
        namespace collision
        {
#if 0
            template < class TCollisionModel, class DataTypes, bool barycentricMapping >
            void ObbTreeGPUBarycentricContactMapper<TCollisionModel,DataTypes,barycentricMapping>::cleanup()
            {
                if (mappingBarycentric != NULL || mappingRigid != NULL)
                {
                    simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
                    if (parent!=NULL)
                    {
                        if (barycentricMapping)
                        {
                            simulation::Node::SPtr child = dynamic_cast<simulation::Node*>(mappingBarycentric->getContext());
                            child->detachFromGraph();
                            child->execute<simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
                            child.reset();
                        }
                        else
                        {
                            simulation::Node::SPtr child = dynamic_cast<simulation::Node*>(mappingRigid->getContext());
                            child->detachFromGraph();
                            child->execute<simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
                            child.reset();
                        }

                        if (barycentricMapping)
                            mappingBarycentric.reset();
                        else
                            mappingRigid.reset();
                    }
                }
            }

            template < class TCollisionModel, class DataTypes, bool barycentricMapping >
            typename ObbTreeGPUBarycentricContactMapper<TCollisionModel,DataTypes,barycentricMapping>::MMechanicalState* ObbTreeGPUBarycentricContactMapper<TCollisionModel,DataTypes,barycentricMapping>::createMapping(const char* name)
            {
                if (model==NULL)
                    return NULL;

                std::cout << "ObbTreeGPUBarycentricContactMapper<TCollisionModel,DataTypes> model = " << model->getName() << " of type " << model->getTypeName() << std::endl;

                simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
                if (parent==NULL)
                {
                    std::cerr << "ERROR: BarycentricContactMapper only works for scenegraph scenes.\n";
                    return NULL;
                }

                //std::cout << "BarycentricContactMapper<TCollisionModel,DataTypes> parent = " << parent->getName() << " of type " << parent->getTypeName() << std::endl;
                //std::cout << "BarycentricContactMapper<TCollisionModel,DataTypes> model mechanicalState = " << model->getMechanicalState()->getName() << " of type " << model->getMechanicalState()->getTypeName() << std::endl;

                simulation::Node::SPtr child = parent->createChild(name);

                typename MMechanicalObject::SPtr mstate = sofa::core::objectmodel::New<MMechanicalObject>();
                child->addObject(mstate);
                mstate->useMask.setValue(true);
                //mapping = new MMapping(model->getMechanicalState(), mstate, model->getMeshTopology());
                //mapper = mapping->getMapper();

                if (barycentricMapping)
                {
                    mapperBarycentric = sofa::core::objectmodel::New<mapping::BarycentricMapperMeshTopology<InDataTypes, typename ObbTreeGPUBarycentricContactMapper::DataTypes> >(model->getMeshTopology(), (topology::PointSetTopologyContainer*)NULL, &model->getObjectMState()->forceMask, &mstate->forceMask);
                    mappingBarycentric = sofa::core::objectmodel::New<MMappingBarycentric>(model->getMState(), mstate.get(), mapperBarycentric);
                }
                else
                {
                    mapperRigid  = sofa::core::objectmodel::New<mapping::BarycentricMapperMeshTopology<InDataTypes, typename ObbTreeGPUBarycentricContactMapper::DataTypes> >(model->getMeshTopology(), (topology::PointSetTopologyContainer*)NULL, &model->getObjectMState()->forceMask, &mstate->forceMask);
                    mappingRigid = sofa::core::objectmodel::New<MMappingRigid>(model->getMState(), mstate.get(), mapperRigid);
                }
                // std::cout << " model mech. state posVec size = " << 0 << std::endl;

                std::stringstream idStr;
                ++_mappingsCreated;

                idStr.str("");
                idStr << parent->getName() << " mstate container no. " << _mappingsCreated;
                child->setName(idStr.str());

                idStr.str("");
                idStr << parent->getName() << " mstate no. " << _mappingsCreated;
                mstate->setName(idStr.str());

                if (barycentricMapping)
                {
                    idStr << parent->getName() << " .. " << "BarycentricContactMapper mapping no. " << _mappingsCreated;
                    mappingBarycentric->setName(idStr.str());

                    idStr.str("");
                    idStr << parent->getName() << " .. " << "BarycentricContactMapper mapper no. " << _mappingsCreated;
                    mapperBarycentric->setName(idStr.str());

                    child->addObject(mappingBarycentric);
                }
                else
                {
                    idStr << parent->getName() << " .. " << "RigidContactMapper mapping no. " << _mappingsCreated;
                    mappingRigid->setName(idStr.str());

                    idStr.str("");
                    idStr << parent->getName() << " .. " << "RigidContactMapper mapper no. " << _mappingsCreated;
                    mapperRigid->setName(idStr.str());

                    child->addObject(mappingRigid);
                }

                return mstate.get();
            }
#endif
        } // namespace collision
    } // namespace component
} // namespace sofa

#endif // OBBTREEGPUBARYCENTRICCONTACTMAPPER_INL
