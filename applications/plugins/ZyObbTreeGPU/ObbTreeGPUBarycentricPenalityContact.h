#ifndef SOFA_COMPONENT_COLLISION_OBBTREEGPUBARYCENTRICPENALITYCONTACT_H
#define SOFA_COMPONENT_COLLISION_OBBTREEGPUBARYCENTRICPENALITYCONTACT_H

#include <sofa/core/collision/Contact.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/interactionforcefield/PenalityContactForceField.h>
#include <sofa/helper/Factory.h>

#include <sofa/component/collision/BarycentricPenalityContact.h>
#include <sofa/component/collision/FrictionContact.h>

#include <sofa/component/collision/IdentityContactMapper.h>
//#include <sofa/component/collision/RigidContactMapper.inl>

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace sofa::defaulttype;
            using namespace core::collision;

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types >
            class ObbTreeGPUBarycentricPenalityContact : public BarycentricPenalityContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes>
            {
                public:
                    SOFA_CLASS(SOFA_TEMPLATE3(ObbTreeGPUBarycentricPenalityContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes), core::collision::Contact);

                    typedef BarycentricPenalityContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes> Base;

                    typedef typename Base::CollisionModel1 CollisionModel1;
                    typedef typename Base::CollisionModel2 CollisionModel2;
                    typedef core::collision::Intersection Intersection;
                    typedef core::collision::DetectionOutputVector OutputVector;
                    typedef core::collision::TDetectionOutputVector<CollisionModel1,CollisionModel2> TOutputVector;
                    typedef ResponseDataTypes DataTypes1;
                    typedef ResponseDataTypes DataTypes2;
                    typedef core::behavior::MechanicalState<DataTypes1> MechanicalState1;
                    typedef core::behavior::MechanicalState<DataTypes2> MechanicalState2;
                    typedef typename CollisionModel1::Element CollisionElement1;
                    typedef typename CollisionModel2::Element CollisionElement2;

                    typedef interactionforcefield::PenalityContactForceField<ResponseDataTypes> ResponseForceField;

                protected:
                    ObbTreeGPUBarycentricPenalityContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
                    virtual ~ObbTreeGPUBarycentricPenalityContact();

                public:
                    virtual void cleanup();

                    virtual void setDetectionOutputs(OutputVector* outputs);

                    virtual void createResponse(core::objectmodel::BaseContext* group);

                    virtual void removeResponse();
            };

            template <class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types >
            class ObbTreeGPUFrictionContact : public sofa::component::collision::FrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>
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
                typedef core::collision::TDetectionOutputVector<CollisionModel1,CollisionModel2> TOutputVector;

            protected:
                ObbTreeGPUFrictionContact() {}

                ObbTreeGPUFrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
                virtual ~ObbTreeGPUFrictionContact();
            public:
                void cleanup();

                void setDetectionOutputs(OutputVector* outputs);

                void createResponse(core::objectmodel::BaseContext* group);

                void removeResponse();
            };
        } // namespace collision
    } // namespace component
} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_OBBTREEGPUBARYCENTRICPENALITYCONTACT_H
