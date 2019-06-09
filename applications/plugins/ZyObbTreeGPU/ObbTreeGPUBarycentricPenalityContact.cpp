#include "ObbTreeGPUBarycentricPenalityContact.h"

#include "ObbTreeGPUBarycentricPenalityContact.inl"
#include "ObbTreeGPUBarycentricContactMapper.h"

#include "ObbTreeGPUCollisionModel.h"

#include <sofa/component/collision/FrictionContact.h>

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace defaulttype;
            using simulation::Node;

            SOFA_DECL_CLASS(ObbTreeGPUBarycentricPenalityContact)
            SOFA_DECL_CLASS(ObbTreeGPUBarycentricFrictionContact)

            //Creator<Contact::Factory, ObbTreeGPUBarycentricPenalityContact<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types>, Vec3Types > > ObbTreeGPUModelPenaltyContactClass("default",true);
            //Creator<Contact::Factory, ObbTreeGPUFrictionContact<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types>, Vec3Types > > ObbTreeGPUModelFrictionContactClass("FrictionContact",true);

            Creator<Contact::Factory, FrictionContact<ObbTreeGPUCollisionModel<>, ObbTreeGPUCollisionModel<> > > ObbTreeGPUFrictionContactClass("FrictionContact",true);
        } // namespace collision
    } // namespace component
} // namespace sofa
