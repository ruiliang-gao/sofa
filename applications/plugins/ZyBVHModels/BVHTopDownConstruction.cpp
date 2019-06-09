#include "BVHTopDownConstruction.inl"

#include <sofa/core/ObjectFactory.h>

using namespace sofa::component::collision;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(BVHTopDownConstruction)

int BVHTopDownConstructionClass = sofa::core::RegisterObject("Bounding volume hierarchy - top-down construction")
#ifndef SOFA_FLOAT
        .add< BVHTopDownConstruction<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< BVHTopDownConstruction<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_BVHMODELSPLUGIN_API BVHTopDownConstruction<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_BVHMODELSPLUGIN_API BVHTopDownConstruction<Vec3fTypes>;
#endif //SOFA_DOUBLE
