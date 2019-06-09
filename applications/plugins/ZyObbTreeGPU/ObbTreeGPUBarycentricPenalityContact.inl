#ifndef OBBTREE_GPU_BARYCENTRICPENALTYCONTACT_INL
#define OBBTREE_GPU_BARYCENTRICPENALTYCONTACT_INL

#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/RigidContactMapper.inl>

#include "ObbTreeGPUBarycentricPenalityContact.h"
#include <sofa/component/collision/BarycentricPenalityContact.inl>
#include <sofa/component/collision/FrictionContact.inl>

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace sofa::defaulttype;
            using namespace core::collision;

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
            ObbTreeGPUBarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::ObbTreeGPUBarycentricPenalityContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
                : BarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>(model1, model2, intersectionMethod)
            {

            }

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
            ObbTreeGPUBarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::~ObbTreeGPUBarycentricPenalityContact()
            {
            }


            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
            void ObbTreeGPUBarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::cleanup()
            {
                BarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::cleanup();
            }

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
            void ObbTreeGPUBarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setDetectionOutputs(OutputVector* o)
            {
                BarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setDetectionOutputs(o);
            }

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
            void ObbTreeGPUBarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::createResponse(core::objectmodel::BaseContext* group)
            {
                BarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::createResponse(group);
            }

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
            void ObbTreeGPUBarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::removeResponse()
            {
                BarycentricPenalityContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::removeResponse();
            }

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::ObbTreeGPUFrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod):
                FrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>(model1, model2, intersectionMethod)
            {

            }

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::~ObbTreeGPUFrictionContact()
            {
            }

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::cleanup()
            {
                FrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::cleanup();
            }


            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setDetectionOutputs(OutputVector* o)
            {
                FrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setDetectionOutputs(o);
            }

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::createResponse(core::objectmodel::BaseContext* group)
            {
                FrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::createResponse(group);
            }

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::removeResponse()
            {
                FrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::removeResponse();
            }

        } // namespace collision
    } // namespace component
} // namespace sofa

#endif // LGCBARYCENTRICPENALTYCONTACT_INL

