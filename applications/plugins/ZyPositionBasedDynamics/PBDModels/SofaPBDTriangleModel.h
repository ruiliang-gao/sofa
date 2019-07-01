#ifndef SOFAPDBTRIANGLEMODEL_H
#define SOFAPDBTRIANGLEMODEL_H

#include "PBDTriangleModel.h"
#include "SofaPBDModelBase.h"

using namespace sofa::defaulttype;

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTriangleModelPrivate;
            class SofaPBDTriangleModel: public SofaPBDModelBase
            {
                public:
                    SOFA_CLASS(SofaPBDTriangleModel, SofaPBDModelBase);
                    SofaPBDTriangleModel();

                    void init();
                    void bwdInit();

                    void parse(sofa::core::objectmodel::BaseObjectDescription *arg);

                    void draw(const core::visual::VisualParams*) override;

                protected:
                    void buildModel();
                    void applyInitialTransform();

                private:
                    std::shared_ptr<SofaPBDTriangleModelPrivate> m_d;
            };
        }
    }
}

#endif // SOFAPDBTRIANGLEMODEL_H
