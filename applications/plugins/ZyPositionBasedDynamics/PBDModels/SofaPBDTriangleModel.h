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
            class SofaPBDTriangleModelPrivate;
            class SofaPBDTriangleModel: public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(SofaPBDTriangleModel,sofa::core::objectmodel::BaseObject);
                    SofaPBDTriangleModel();

                    void init();
                    void bwdInit();

                    void parse(sofa::core::objectmodel::BaseObjectDescription *arg);

                    void draw(const core::visual::VisualParams*) override;

                private:
                    std::shared_ptr<SofaPBDTriangleModelPrivate> m_d;
            };
        }
    }
}

#endif // SOFAPDBTRIANGLEMODEL_H
