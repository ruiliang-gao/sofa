#ifndef OBBTREEGPUFRICTIONCONTACT_H
#define OBBTREEGPUFRICTIONCONTACT_H

#include <sofa/core/collision/Contact.h>
#include <sofa/core/collision/Intersection.h>
#include <SofaBaseMechanics/BarycentricMapping.h>
#include <SofaObjectInteraction/PenalityContactForceField.h>
#include <sofa/helper/Factory.h>

#include <SofaRigid/RigidMapping.h>

#include <SofaBaseCollision/BaseContactMapper.h>
#include <SofaMeshCollision/IdentityContactMapper.h>
#include <SofaMeshCollision/RigidContactMapper.h>
#include <SofaMeshCollision/BarycentricContactMapper.h>

#include <SofaConstraint/UnilateralInteractionConstraint.h>

#include "ObbTreeGPUCollisionModel.h"

namespace sofa
{
    namespace component
    {
        namespace collision
        {

            //#define OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
            //#define OBBTREEGPUBARYCENTRICCONTACTMAPPER_SUPPRESS_CONTACT_RESPONSE
            //#define OBBTREEGPUBARYCENTRICCONTACTMAPPER_SUPPRESS_LINE_LINE_CONTACTS
            
            // dummy contact mapper class 
            // this must exist, so that the contact mappers for SOFA CPU-contact models (mapper1_default and mapper2_default) can be declared as ContactMapper 
            template <class DataTypes>
            class ContactMapper<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>, DataTypes> : public BarycentricContactMapper<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>, DataTypes>{
                typedef typename DataTypes::Real Real;
                typedef typename DataTypes::Coord Coord;

            public:
                int addPoint(const Coord& P, int index, Real& r){
                    std::cout << "ERRROR! ContactMapper<sofa::component::collision::ObbTreeGPUCollisionModel<Vec3Types>, DataTypes> should never be called." << std::endl;
                    return -1;
                }
            };
            // 

            template < class TCollisionModel, class DataTypes >
            class TestContactMapper : public BarycentricContactMapper<TCollisionModel, DataTypes>
            {
            public:
                typedef typename DataTypes::Real Real;
                typedef typename DataTypes::Coord Coord;
                typedef typename DataTypes::VecCoord VecCoord;
                typedef typename DataTypes::VecDeriv VecDeriv;
                typedef BarycentricContactMapper<TCollisionModel, DataTypes> Inherit;
                typedef typename Inherit::MMechanicalState MMechanicalState;
                typedef typename Inherit::MCollisionModel MCollisionModel;

                int addPoint(const Coord &P, int index, Real& r, const sofa::core::collision::DetectionOutputContactType& type)
                {
                    if (type == sofa::core::collision::CONTACT_LINE_LINE)
                    {
#ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        std::cout << "   ObbTreeGPUContactMapper<ObbTreeGPUCollisionModel,DataTypes>::addPoint(" << P << "," << index << "," << r << ") as LINE_LINE" << std::endl;
#endif
                        /*int edgeIdx1 = index % 3;
                        int triIdx1 = index / 3;
                        const sofa::core::topology::BaseMeshTopology::EdgesInTriangle& e1 = this->model->getMeshTopology()->getEdgesInTriangle(triIdx1);
                        int targetEdge = e1[edgeIdx1];*/

                        int nbe = this->model->getMeshTopology()->getNbEdges();
#ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        std::cout << "   edge index = " << index << "; edges in topology: " << nbe << std::endl;
#endif

#ifndef OBBTREEGPUBARYCENTRICCONTACTMAPPER_SUPPRESS_LINE_LINE_CONTACTS
                        if (index < nbe)
                            return this->mapper->createPointInLine(P, index, &this->model->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue());
#endif
                    }
                    else if (type == sofa::core::collision::CONTACT_FACE_VERTEX)
                    {
                        int nbt = this->model->getMeshTopology()->getNbTriangles();
    #ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        std::cout << "   ObbTreeGPUContactMapper<ObbTreeGPUCollisionModel,DataTypes>::addPoint(" << P << "," << index << "," << r << ") as VERTEX_FACE; nbt = " << nbt << std::endl;
    #endif
    #ifndef OBBTREEGPUBARYCENTRICCONTACTMAPPER_SUPPRESS_CONTACT_RESPONSE
                        if (index < nbt)
                            return this->mapper->createPointInTriangle(P, index, &this->model->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue());
    #ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        else
                            std::cout << " index " << index << " >= " << nbt << ", NOT ADDING this contact point!" << std::endl;
    #endif
    #endif
                    }
                    return -1;
                }

                int addPointB(const Coord &P, int index, Real& r, const Vector3& baryP, const sofa::core::collision::DetectionOutputContactType& type)
                {
                    if (type == sofa::core::collision::CONTACT_LINE_LINE)
                    {
#ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        std::cout << "   ObbTreeGPUContactMapper<ObbTreeGPUCollisionModel,DataTypes>::addPointB(" << P << "," << index << "," << r << "," << baryP << ") as LINE_LINE" << std::endl;
#endif
#ifndef OBBTREEGPUBARYCENTRICCONTACTMAPPER_SUPPRESS_LINE_LINE_CONTACTS
                        int edgeIdx1 = index % 3;
                        int triIdx1 = index / 3;

                        int nbe = this->model->getMeshTopology()->getNbEdges();

                        if ((triIdx1 * 3) + edgeIdx1 < nbe)
                            return this->mapper->addPointInLine((triIdx1 * 3) + edgeIdx1, baryP.ptr());
#endif
                    }
                    else if (type == sofa::core::collision::CONTACT_FACE_VERTEX)
                    {
                        int nbt = this->model->getMeshTopology()->getNbTriangles();
    #ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        std::cout << "   ObbTreeGPUContactMapper<ObbTreeGPUCollisionModel,DataTypes>::addPointB(" << P << "," << index << "," << r << "," << baryP << ") as VERTEX_FACE; nbt = " << nbt << std::endl;
    #endif
                        if (index < nbt)
                            return this->mapper->addPointInTriangle(index, baryP.ptr());
                    }
                    return -1;
                }

                inline int addPointB(const Coord& P, int index, Real& r, const sofa::core::collision::DetectionOutputContactType& type){ return addPoint(P,index,r,type); }
#if 1
                int addPointWithIndex(const Coord& P, int index, int featureId, const sofa::core::collision::DetectionOutputContactType& type)
                {
                    if (type == sofa::core::collision::CONTACT_LINE_LINE)
                    {
#ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        std::cout << "   ObbTreeGPUContactMapper<ObbTreeGPUCollisionModel,DataTypes>::addPointWithIndex(" << P << "," << index << "," << featureId << ") as LINE_LINE" << std::endl;
#endif
                        int nbe = this->model->getMeshTopology()->getNbEdges();
#ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        std::cout << "  edge in triangle " << index << ", edge id = " << featureId << "; edges in topology: " << nbe << std::endl;
#endif
                        const sofa::core::topology::BaseMeshTopology::EdgesInTriangle& e1 = this->model->getMeshTopology()->getEdgesInTriangle(index);
                        int targetEdge = e1[featureId];

#ifndef OBBTREEGPUBARYCENTRICCONTACTMAPPER_SUPPRESS_LINE_LINE_CONTACTS
                        if ((index * 3) + featureId < nbe)
                            return this->mapper->createPointInLine(P, targetEdge, this->model->getMechanicalState()->getX());
#endif
                    }
                    else if (type == sofa::core::collision::CONTACT_FACE_VERTEX)
                    {
                        int nbt = this->model->getMeshTopology()->getNbTriangles();
    #ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        std::cout << "   ObbTreeGPUContactMapper<ObbTreeGPUCollisionModel,DataTypes>::addPointWithIndex(" << P << "," << index << "," << featureId << ") as VERTEX_FACE; nbt = " << nbt << std::endl;
    #endif
    #ifndef OBBTREEGPUBARYCENTRICCONTACTMAPPER_SUPPRESS_CONTACT_RESPONSE
                        if (index < nbt)
                        {
                            return this->mapper->createPointInTriangle(P, index, this->model->getMechanicalState()->getX());
                        }
    #ifdef OBBTREEGPUBARYCENTRICCONTACTMAPPER_DEBUG
                        else
                        {
                            std::cout << " index " << index << " >= " << nbt << ", NOT ADDING this contact point!" << std::endl;
                        }
    #endif
    #endif
                    }
                    return -1;
                }

                int addPointWithIndexB(const Coord& P, int index, int featureId, const sofa::core::collision::DetectionOutputContactType& type)
                {

                    return 0;
                }
#endif
            };
        }
    }
}

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace sofa::defaulttype;

            class SOFA_CONSTRAINT_API ObbTreeGPUContactIdentifier
            {
            public:
                ObbTreeGPUContactIdentifier()
                {
                    if (!availableId.empty())
                    {
                        id = availableId.front();
                        availableId.pop_front();
                    }
                    else
                        id = cpt++;

                    //	sout << id << sendl;
                }

                virtual ~ObbTreeGPUContactIdentifier()
                {
                    availableId.push_back(id);
                }

            protected:
                static sofa::core::collision::DetectionOutput::ContactId cpt;
                sofa::core::collision::DetectionOutput::ContactId id;
                static std::list<sofa::core::collision::DetectionOutput::ContactId> availableId;
            };

//#define OBBTREEGPUFRICTIONCONTACT_DEBUG
#define OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
#define OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
#define OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
//#define OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
            template <class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types >
            class ObbTreeGPUFrictionContact : public core::collision::Contact, public ObbTreeGPUContactIdentifier
            {
            public:
                SOFA_CLASS(SOFA_TEMPLATE3(ObbTreeGPUFrictionContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes), core::collision::Contact);
                typedef TCollisionModel1 CollisionModel1;
                typedef TCollisionModel2 CollisionModel2;
                typedef core::collision::Intersection Intersection;
                typedef ResponseDataTypes DataTypes1;
                typedef ResponseDataTypes DataTypes2;

                typedef core::behavior::MechanicalState<DataTypes1> MechanicalState1;
                typedef core::behavior::MechanicalState<DataTypes2> MechanicalState2;
                typedef typename CollisionModel1::Element CollisionElement1;
                typedef typename CollisionModel2::Element CollisionElement2;
                typedef core::collision::DetectionOutputVector OutputVector;
                typedef core::collision::TDetectionOutputVector<CollisionModel1, CollisionModel2> TOutputVector;

            protected:
                TCollisionModel1* model1;
                TCollisionModel2* model2;

                MechanicalState1* mmodel1;
                MechanicalState1* mmodel2;

                Intersection* intersectionMethod;
                bool selfCollision; ///< true if model1==model2 (in this case, only mapper1 is used)
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                //TestContactMapper<CollisionModel1, DataTypes1> mapper1_default;
                //TestContactMapper<CollisionModel2, DataTypes2> mapper2_default;
                ContactMapper<CollisionModel1, DataTypes1> mapper1_default;
                ContactMapper<CollisionModel2, DataTypes2> mapper2_default;
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                TestContactMapper<CollisionModel1, DataTypes1> mapper1_gpu_ll;
                TestContactMapper<CollisionModel2, DataTypes2> mapper2_gpu_ll;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                TestContactMapper<CollisionModel1, DataTypes1> mapper1_gpu_lv;
                TestContactMapper<CollisionModel2, DataTypes2> mapper2_gpu_lv;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                TestContactMapper<CollisionModel1, DataTypes1> mapper1_gpu_vf;
                TestContactMapper<CollisionModel2, DataTypes2> mapper2_gpu_vf;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                typename sofa::component::constraintset::UnilateralInteractionConstraint<ResponseDataTypes>::SPtr m_constraintGPU_ll;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                typename sofa::component::constraintset::UnilateralInteractionConstraint<ResponseDataTypes>::SPtr m_constraintGPU_lv;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                typename sofa::component::constraintset::UnilateralInteractionConstraint<ResponseDataTypes>::SPtr m_constraintGPU_vf;
#endif


#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                typename sofa::component::constraintset::UnilateralInteractionConstraint<ResponseDataTypes>::SPtr m_defaultConstraint;
#endif
                core::objectmodel::BaseContext* parent;

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                unsigned int m_numLineLineContacts;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                unsigned int m_numLineVertexContacts;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                unsigned int m_numVertexFaceContacts;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                unsigned int m_numOtherContacts;
#endif

                Data<double> mu, tol;
                std::vector< sofa::core::collision::DetectionOutput* > contacts;

                typedef std::vector< sofa::core::collision::DetectionOutput* > manifold;
                typedef std::vector< manifold > manifoldVector;

                typedef std::pair< manifold, Vector3 > clusterManifold;
                typedef std::vector< clusterManifold > clusterManifoldVector;

                clusterManifoldVector dynamicManifoldVector;

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                manifoldVector faceVertexManifoldVector;

#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                manifoldVector lineLineManifoldVector;
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                manifoldVector lineVertexManifoldVector;
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                manifoldVector defaultManifoldVector;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                std::vector< std::pair< std::pair<int, int>, double > > mappedContacts;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                std::vector< std::pair< std::pair<int, int>, double > > mappedContacts_VertexFace;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                std::vector< std::pair< std::pair<int, int>, double > > mappedContacts_LineVertex;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                std::vector< std::pair< std::pair<int, int>, double > > mappedContacts_LineLine;
#endif

                void activateMappers();

                void setInteractionTags(MechanicalState1* mstate1, MechanicalState2* mstate2,
                    typename sofa::component::constraintset::UnilateralInteractionConstraint<ResponseDataTypes>* constraint);

                ObbTreeGPUFrictionContact() {}

                ObbTreeGPUFrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
                virtual ~ObbTreeGPUFrictionContact();

                    inline long cantorPolynomia(sofa::core::collision::DetectionOutput::ContactId x, sofa::core::collision::DetectionOutput::ContactId y)
                    {
                        // Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
                        return (long)(((x+y)*(x+y)+3*x+y)/2);
                    }

                std::ofstream testOutput;

                unsigned long m_numContacts;

            public:
                void cleanup();

                std::pair<core::CollisionModel*, core::CollisionModel*> getCollisionModels() { return std::make_pair(model1, model2); }

                void setDetectionOutputs(OutputVector* outputs);

                void createResponse(core::objectmodel::BaseContext* group);

                void removeResponse();

                void draw(const core::visual::VisualParams*);

                sofa::core::objectmodel::DataFileName testOutputFilename;

                //void setIntersectionOccurred(bool newSCRValue); // for intersection detection (now obsolete, hopefully. See implementation for more information.)
            };

        }
    }
}

#endif // OBBTREEGPUFRICTIONCONTACT_H
