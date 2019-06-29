#ifndef SOFAPDBLINEMODEL_H
#define SOFAPDBLINEMODEL_H

#include "PBDLineModel.h"

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDLineModelPrivate;
            class SofaPBDLineModel: public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SofaPBDLineModel,sofa::core::objectmodel::BaseObject);
                    SofaPBDLineModel();

                private:
                    std::shared_ptr<SofaPBDLineModelPrivate> m_d;
            };
        }
    }
}

#endif // SOFAPDBLINEMODEL_H
