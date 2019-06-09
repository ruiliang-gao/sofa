#ifndef OBBTREEGPUBARYCENTRICCONTACTMAPPER_H
#define OBBTREEGPUBARYCENTRICCONTACTMAPPER_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/Factory.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/mapping/SubsetMapping.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>

#include <sofa/component/collision/BaseContactMapper.h>
#include <sofa/component/collision/RigidContactMapper.h>
#include <sofa/component/collision/BarycentricContactMapper.h>

#include "ObbTreeGPUCollisionModel.h"
#include <sofa/component/collision/CubeModel.h>

#include <sofa/component/mapping/IdentityMapping.h>
#include <iostream>


namespace sofa
{
    namespace component
    {
        namespace collision
        {

            using namespace sofa::defaulttype;

#if 1
            template <class TVec3Types>
            class ContactMapper<ObbTreeGPUCollisionModel<TVec3Types>, TVec3Types > : public RigidContactMapper<ObbTreeGPUCollisionModel<TVec3Types>, TVec3Types >
            {
                typedef typename TVec3Types::Real Real;
                typedef typename TVec3Types::Coord Coord;
                public:
                    int addPoint(const typename TVec3Types::Coord & P, int index,typename TVec3Types::Real & r)
                    {
                        /*RigidSphere e(this->model, index);
                        const typename ObbTreeGPUCollisionModel::DataTypes::Coord & rCenter = e.rigidCenter();
                        const typename TVec3Types::Coord & cP = P - rCenter.getCenter();
                        const Quaternion & ori = rCenter.getOrientation();

                        //r = e.r();

                        return RigidContactMapper<ObbTreeGPUCollisionModel,TVec3Types >::addPoint(ori.inverseRotate(cP),index,r);*/
                        return 0;
                    }

                    int addPointB(const Vector3 &P, int elementId, Real &r)
                    {
                        return -1;
                    }
            };
#endif
#if 0
            template<class DataTypes>
            class ContactMapper<ObbTreeGPUCollisionModel<DataTypes>, DataTypes> : public BarycentricContactMapper<ObbTreeGPUCollisionModel<DataTypes>, DataTypes>
            {
            public:
                typedef typename DataTypes::Real Real;
                typedef typename DataTypes::Coord Coord;
                int addPoint(const Coord& P, int index, Real&)
                {
                    /*
                    int nbt = this->model->getMeshTopology()->getNbTriangles();
                    if (index < nbt)
                        return this->mapper->createPointInTriangle(P, index, this->model->getMechanicalState()->getX());
                    else
                    {
                        int qindex = (index - nbt)/2;
                        int nbq = this->model->getMeshTopology()->getNbQuads();
                        if (qindex < nbq)
                            return this->mapper->createPointInQuad(P, qindex, this->model->getMechanicalState()->getX());
                        else
                        {
                            std::cerr << "ContactMapper<TriangleMeshModel>: ERROR invalid contact element index "<<index<<" on a topology with "<<nbt<<" triangles and "<<nbq<<" quads."<<std::endl;
                            std::cerr << "model="<<this->model->getName()<<" size="<<this->model->getSize()<<std::endl;
                            return -1;
                        }
                    }*/
                    return 0;
                }
                int addPointB(const Coord& P, int index, Real& /*r*/, const Vector3& baryP)
                {

                    /*int nbt = this->model->getMeshTopology()->getNbTriangles();
                    if (index < nbt)
                        return this->mapper->addPointInTriangle(index, baryP.ptr());
                    else
                    {
                        // TODO: barycentric coordinates usage for quads
                        int qindex = (index - nbt)/2;
                        int nbq = this->model->getMeshTopology()->getNbQuads();
                        if (qindex < nbq)
                            return this->mapper->createPointInQuad(P, qindex, this->model->getMechanicalState()->getX());
                        else
                        {
                            std::cerr << "ContactMapper<TriangleMeshModel>: ERROR invalid contact element index "<<index<<" on a topology with "<<nbt<<" triangles and "<<nbq<<" quads."<<std::endl;
                            std::cerr << "model="<<this->model->getName()<<" size="<<this->model->getSize()<<std::endl;
                            return -1;
                        }
                    }*/
                    return -1;
                }

                inline int addPointB(const Coord& P, int index, Real& r ){return addPoint(P,index,r);}

            };
#endif
#if 0
            /// Base class for all mappers using ObbTreeGPUBarycentricContactMapper
            template < class TCollisionModel, class DataTypes, bool barycentricMapping = false >
            class ObbTreeGPUBarycentricContactMapper : public BaseContactMapper<DataTypes>
            {
            public:
                typedef typename DataTypes::Real Real;
                typedef typename DataTypes::Coord Coord;
                typedef TCollisionModel MCollisionModel;
                typedef typename MCollisionModel::InDataTypes InDataTypes;
                typedef typename MCollisionModel::Topology InTopology;
                typedef core::behavior::MechanicalState< InDataTypes> InMechanicalState;
                typedef core::behavior::MechanicalState<  typename ObbTreeGPUBarycentricContactMapper::DataTypes> MMechanicalState;
                typedef component::container::MechanicalObject<typename ObbTreeGPUBarycentricContactMapper::DataTypes> MMechanicalObject;

                typedef mapping::BarycentricMapping< InDataTypes, typename ObbTreeGPUBarycentricContactMapper::DataTypes > MMappingBarycentric;
                typedef mapping::TopologyBarycentricMapper<InDataTypes, typename ObbTreeGPUBarycentricContactMapper::DataTypes> MMapperBarycentric;

                typedef sofa::component::mapping::RigidMapping< InDataTypes, typename ObbTreeGPUBarycentricContactMapper::DataTypes > MMappingRigid;
                typedef sofa::component::mapping::TopologyBarycentricMapper<InDataTypes, typename ObbTreeGPUBarycentricContactMapper::DataTypes> MMapperRigid;

                MCollisionModel* model;
                typename MMappingBarycentric::SPtr mappingBarycentric;
                typename MMapperBarycentric::SPtr mapperBarycentric;

                typename MMappingRigid::SPtr mappingRigid;
                typename MMapperRigid::SPtr mapperRigid;

                unsigned long _mappingsCreated;

                ObbTreeGPUBarycentricContactMapper()
                    : model(NULL), mappingRigid(NULL), mapperRigid(NULL), mappingBarycentric(NULL), mapperBarycentric(NULL), _mappingsCreated(0)
                {
                }

                void setCollisionModel(MCollisionModel* model)
                {
                    this->model = model;
                }

                void cleanup();

                /// Added virtual for LGC, to derive LGC contact mapping class
                virtual MMechanicalState* createMapping(const char* name="contactPoints");

                void resize(int size)
                {
                    if (barycentricMapping)
                    {
                        if (mapperBarycentric != NULL && mappingBarycentric != NULL)
                        {
                            mapperBarycentric->clear();
                            mappingBarycentric->getMechTo()[0]->resize(size);
                        }
                    }
                    else
                    {
                        if (mapperRigid != NULL && mappingRigid != NULL)
                        {
                            mapperRigid->clear();
                            mappingRigid->getMechTo()[0]->resize(size);
                        }
                    }
                }

                void update()
                {
                    if (barycentricMapping)
                    {
                        if (mappingBarycentric != NULL)
                        {
                            core::BaseMapping* map = mappingBarycentric.get();
                            map->apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::position(), core::ConstVecCoordId::position());
                            map->applyJ(core::MechanicalParams::defaultInstance(), core::VecDerivId::velocity(), core::ConstVecDerivId::velocity());
                        }
                    }
                    else
                    {
                        if (mappingRigid != NULL)
                        {
                            core::BaseMapping* map = mappingRigid.get();
                            map->apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::position(), core::ConstVecCoordId::position());
                            map->applyJ(core::MechanicalParams::defaultInstance(), core::VecDerivId::velocity(), core::ConstVecDerivId::velocity());
                        }
                    }
                }

                void updateXfree()
                {
                    if (barycentricMapping)
                    {
                        if (mappingBarycentric != NULL)
                        {
                            core::BaseMapping* map = mappingBarycentric.get();
                            map->apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::freePosition(), core::ConstVecCoordId::freePosition());
                            map->applyJ(core::MechanicalParams::defaultInstance(), core::VecDerivId::freeVelocity(), core::ConstVecDerivId::freeVelocity());
                        }
                    }
                    else
                    {
                        if (mappingRigid != NULL)
                        {
                            core::BaseMapping* map = mappingRigid.get();
                            map->apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::freePosition(), core::ConstVecCoordId::freePosition());
                            map->applyJ(core::MechanicalParams::defaultInstance(), core::VecDerivId::freeVelocity(), core::ConstVecDerivId::freeVelocity());
                        }
                    }
                }
            };

            /// Mapper for ObbTreeGPUCollisionModel
            template<class DataTypes>
            class ContactMapper<ObbTreeGPUCollisionModel<Vec3Types>, DataTypes> : public ObbTreeGPUBarycentricContactMapper<ObbTreeGPUCollisionModel<Vec3Types>, DataTypes>
            {
                public:
                    typedef typename DataTypes::Real Real;
                    typedef typename DataTypes::Coord Coord;

                    bool supportsPointsWithIndices() const { return true; }

                    int addPoint(const Coord& P, int index, Real&)
                    {
#ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        std::cout << "ObbTreeGPUBarycentricContactMapper<ObbTreeGPUCollisionModel,DataTypes>::addPoint(" << P << "," << index << "); " << " CollisionModel is-a " << this->model->getClassName() << std::endl;
#endif
                        int triangleId = index/* / 15*/;
#ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        int featureId = index % 15;
                        std::cout << " triangleId = " << triangleId << ", featureId = " << featureId << std::endl;
#endif
                        ObbTreeGPUCollisionModel<Vec3Types>* obbModel = static_cast<ObbTreeGPUCollisionModel<Vec3Types>*>(this->model);
                        if ((unsigned long) triangleId < obbModel->numTriangles())
                        {
#ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                            std::cout << " index = " << triangleId << " < numTriangles = " << obbModel->numTriangles() << std::endl;
#endif
                            /// @TODO!!!
                            /// return this->mapper->createPointInTriangle(P, triangleId, this->model->getMState()->getX());

                        }
#ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        std::cout << " index = " << index << " beyond range numTriangles = " << obbModel->numTriangles() << std::endl;
#endif
                        return 0;
                    }

                    /*int addPointWithIndex(const Coord& P, int index, uint64_t elementIndex, Real&)
                    {
                        int returnValue = -1;
                        std::cout << "LGCBarycentricContactMapper<LGCObbModel,DataTypes>::addPointWithIndex(" << P << "," << index << "," << elementIndex << ")" << std::endl;
                        std::cout << " CollisionModel is-a " << this->model->getClassName() << std::endl;
                        LGCObbModel* obbModel = static_cast<LGCObbModel*>(this->model);
                        LGCObb<Vec3Types>* obb = 0;
                        if ((unsigned long) index < obbModel->numChildren())
                        {
                            obb = obbModel->obbChild(index);
                        }
                        else
                        {
                            obb = obbModel->parentObb();
                        }

                        if (obb != NULL)
                        {
                            std::cout << " contact point maps to OBB " << *obb << ", facet indices range is " << obb->minFacetRange() << " -> " << obb->maxFacetRange() << std::endl;
                            if (elementIndex >= obb->minFacetRange() && elementIndex <= obb->maxFacetRange())
                            {
                                std::cout << " elementIndex " << elementIndex << " falls into obb's facet range" << std::endl;
                                if (elementIndex < (uint64_t) this->model->getMeshTopology()->getNbTriangles())
                                    returnValue = this->mapper->createPointInTriangle(P, elementIndex, this->model->getMechanicalState()->getX());
                            }
                            else
                            {
                                std::cout << " WARNING: elementIndex " << elementIndex << " does NOT fall into cluster's facet range: " << obb->minFacetRange() << " -- " << obb->maxFacetRange() << std::endl;
                            }
                        }
                        else
                        {
                            std::cout << " WARNING: Failed to retrieve a valid OBB with index " << index << " from LGCObbModel " << obbModel->getName() << std::endl;
                        }
                        std::cout << " return value: " << returnValue << std::endl;
                        return returnValue;
                    }*/

            };

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_OBBTREEGPU)
extern template class SOFA_OBBTREEGPUPLUGIN_API ContactMapper<ObbTreeGPUCollisionModel<> >;
#endif
#endif
        } // namespace collision
    } // namespace component
} // namespace sofa


#endif // OBBTREEGPUBARYCENTRICCONTACTMAPPER_H

