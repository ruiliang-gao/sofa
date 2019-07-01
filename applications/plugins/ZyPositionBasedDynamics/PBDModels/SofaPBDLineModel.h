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

                    void init();
                    void bwdInit();

                    void parse(sofa::core::objectmodel::BaseObjectDescription *arg);

                    void draw(const core::visual::VisualParams*) override;

                private:
                    std::shared_ptr<SofaPBDLineModelPrivate> m_d;
            };
        }
    }
}

#endif // SOFAPDBLINEMODEL_H
