#ifndef SOFAPDBTRIANGLEMODEL_H
#define SOFAPDBTRIANGLEMODEL_H

#include "RigidBody.h"
#include "SofaPBDModelBase.h"

using namespace sofa::defaulttype;

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDRigidBodyModelPrivate;
            class SofaPBDRigidBodyModel: public SofaPBDModelBase
            {
                public:
                    SOFA_CLASS(SofaPBDRigidBodyModel, SofaPBDModelBase);
                    SofaPBDRigidBodyModel(SimulationModel *model = nullptr);

                    void init();
                    void bwdInit();

                    void parse(sofa::core::objectmodel::BaseObjectDescription *arg);

                    void draw(const core::visual::VisualParams*) override;

                    const int getPBDRigidBodyIndex() const;

                protected:
                    void buildModel();
                    void initializeModel();

                    Data<SReal> mass;
                    Data<SReal> density;
                    Data<SReal> frictionCoefficient;
                    Data<SReal> restitutionCoefficient;
                    Data<sofa::defaulttype::Vec3d> inertiaTensor;

                private:
                    std::shared_ptr<SofaPBDRigidBodyModelPrivate> m_d;
            };
        }
    }
}

#endif // SOFAPDBTRIANGLEMODEL_H
