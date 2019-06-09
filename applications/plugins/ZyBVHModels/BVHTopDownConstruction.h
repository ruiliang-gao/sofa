#ifndef BVHTOPDOWNCONSTRUCTION_H
#define BVHTOPDOWNCONSTRUCTION_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/visual/VisualParams.h>

#include "initBVHModelsPlugin.h"

namespace sofa
{
    namespace component
    {
        namespace collision
        {
			using namespace sofa;
			using namespace sofa::core;

            template <class DataTypes>
            class SOFA_BVHMODELSPLUGIN_API BVHTopDownConstruction: public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SOFA_TEMPLATE(BVHTopDownConstruction, DataTypes), sofa::core::objectmodel::BaseObject);
                    BVHTopDownConstruction();

					void init();
					void cleanup();

					void draw(const core::visual::VisualParams*);
            };
        }
    }
}

#endif // BVHTOPDOWNCONSTRUCTION_H
