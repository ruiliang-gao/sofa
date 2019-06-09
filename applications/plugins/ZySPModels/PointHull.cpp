#include "LGCPointCluster.inl"

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            SOFA_DECL_CLASS(LGCPointCluster)

            template class LGCPointCluster<Vec3dTypes>;
            template class LGCPointCluster<Vec3fTypes>;

            template class pcl::SACSegmentation<pcl::LGCPointType>;

            int LGCPointClusterClass = core::RegisterObject("Point cluster for LGC")
            #ifndef SOFA_FLOAT
            .add< LGCPointCluster<sofa::defaulttype::Vec3dTypes> >()
            #endif
            #ifndef SOFA_DOUBLE
            .add< LGCPointCluster<sofa::defaulttype::Vec3fTypes> >()
            #endif
            ;
        }
    }
}
