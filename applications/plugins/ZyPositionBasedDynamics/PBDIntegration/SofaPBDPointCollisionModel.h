#ifndef SOFAPBDPOINTCOLLISIONMODEL_H
#define SOFAPBDPOINTCOLLISIONMODEL_H

#include "initZyPositionBasedDynamicsPlugin.h"
#include <SofaMeshCollision/PointModel.h>

#include "PBDRigidBody.h"
#include "PBDModels/PBDLineModel.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            using namespace sofa::core::objectmodel;

            class SofaPBDPointCollisionModelPrivate;
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDPointCollisionModel: /*public sofa::core::CollisionModel,*/ public sofa::component::collision::PointModel
            {
                public:
                    SOFA_CLASS(SofaPBDPointCollisionModel, sofa::component::collision::PointModel);
                    SofaPBDPointCollisionModel();
                    ~SofaPBDPointCollisionModel();

                    void init() override;
                    void bwdInit() override;

                    void draw(const core::visual::VisualParams*) override;

                    core::CollisionModel *toCollisionModel() override;
                    const core::CollisionModel *toCollisionModel() const override;

                    bool insertInNode(sofa::core::objectmodel::BaseNode *node) override;
                    bool removeInNode(sofa::core::objectmodel::BaseNode *node) override;

                    void computeBoundingTree(int maxDepth = 0) override;

                    void parse(BaseObjectDescription* arg);

                    /// Pre-construction check method called by ObjectFactory.
                    /// Check that DataTypes matches the MechanicalState.
                    template<class T>
                    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
                    {
                        /*if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
                        {
                            msg_info("SofaPBDPointCollisionModel") << "Context has no valid mechanical state, can not instantiate SofaPBDPointCollisionModel!";
                            return false;
                        }*/

                        bool boCanCreate = BaseObject::canCreate(obj, context, arg);
                        msg_info("SofaPBDPointCollisionModel") << "BaseObject::canCreate(): " << boCanCreate;

                        return boCanCreate;
                    }

                private:
                    SofaPBDPointCollisionModelPrivate* m_d;
            };

        }
    }
}

#endif // SOFAPBDPOINTCOLLISIONMODEL_H
