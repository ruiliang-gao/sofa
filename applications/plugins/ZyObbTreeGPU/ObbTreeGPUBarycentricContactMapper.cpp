#define SOFA_COMPONENT_COLLISION_OBBTREEGPUBARYCENTRICCONTACTMAPPER_CPP
#include "ObbTreeGPUBarycentricContactMapper.inl"
#include <sofa/helper/Factory.inl>

#include "initObbTreeGpuPlugin.h"

#include "ObbTreeGPUCollisionModel.h"

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace defaulttype;

            SOFA_DECL_CLASS(ObbTreeGPUBarycentricContactMapper)

            //ContactMapperCreator< ObbTreeGPUBarycentricContactMapper<ObbTreeGPUCollisionModel<> > > ObbTreeGPUContactMapperClassBaryCentric("default",true);
            //ContactMapperCreator< ContactMapper<ObbTreeGPUCollisionModel<Vec3Types>,Vec3Types> > ObbTreeGPUFrictionContactMapperClass("default", true);
            //template class SOFA_OBBTREEGPUPLUGIN_API ContactMapper<ObbTreeGPUCollisionModel<Rigid3Types>, Rigid3Types >;

        } // namespace collision
    } // namespace component
} // namespace sofa
