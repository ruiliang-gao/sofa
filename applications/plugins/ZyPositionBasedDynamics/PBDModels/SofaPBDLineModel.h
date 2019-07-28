#ifndef SOFAPDBLINEMODEL_H
#define SOFAPDBLINEMODEL_H

#include "PBDLineModel.h"
#include "SofaPBDModelBase.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDLineModelPrivate;
            class SofaPBDLineModel: public SofaPBDModelBase
            {
                public:
                    SOFA_CLASS(SofaPBDLineModel, SofaPBDModelBase);
                    SofaPBDLineModel();

                    void init();
                    void bwdInit();
                    void cleanup();

                    void parse(sofa::core::objectmodel::BaseObjectDescription *arg);

                    virtual void draw(const core::visual::VisualParams*) override;

                    std::shared_ptr<PBDLineModel> getPBDLineModel() const;

                protected:
                    void buildModel();
                    void initializeModel();

                private:
                    std::shared_ptr<SofaPBDLineModelPrivate> m_d;
            };
        }
    }
}

#endif // SOFAPDBLINEMODEL_H
