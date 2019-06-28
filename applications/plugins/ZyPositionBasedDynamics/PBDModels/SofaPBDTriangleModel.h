#ifndef SOFAPDBTRIANGLEMODEL_H
#define SOFAPDBTRIANGLEMODEL_H

#include "PBDTriangleModel.h"

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTriangleModel: public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SofaPBDTriangleModel,sofa::core::objectmodel::BaseObject);
                    SofaPBDTriangleModel();
            };
        }
    }
}

#endif // SOFAPDBTRIANGLEMODEL_H
