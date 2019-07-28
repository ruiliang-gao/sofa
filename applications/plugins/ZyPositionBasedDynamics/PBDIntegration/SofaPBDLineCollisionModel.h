#ifndef SOFAPBDLINECOLLISIONMODEL_H
#define SOFAPBDLINECOLLISIONMODEL_H

#include "initZyPositionBasedDynamicsPlugin.h"
#include <SofaMeshCollision/LineModel.h>

#include "PBDRigidBody.h"
#include "PBDModels/PBDLineModel.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            using namespace sofa::core::objectmodel;

            class SofaPBDLineCollisionModelPrivate;
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDLineCollisionModel: /*public sofa::core::CollisionModel,*/ public sofa::component::collision::LineModel
            {
                public:
                    SOFA_CLASS(SofaPBDLineCollisionModel, sofa::component::collision::LineModel);
                    SofaPBDLineCollisionModel();
                    ~SofaPBDLineCollisionModel();

                    void init() override;
                    void bwdInit() override;

                    void parse(BaseObjectDescription* arg);

                    void draw(const core::visual::VisualParams*) override;

                    const sofa::core::CollisionModel* toCollisionModel() const override;
                    sofa::core::CollisionModel* toCollisionModel() override;

                    bool insertInNode(sofa::core::objectmodel::BaseNode *node) override;
                    bool removeInNode(sofa::core::objectmodel::BaseNode *node) override;

                    void computeBoundingTree(int maxDepth = 0) override;

                    /// Pre-construction check method called by ObjectFactory.
                    /// Check that DataTypes matches the MechanicalState.
                    template<class T>
                    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
                    {
                        /*if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
                        {
                            msg_info("SofaPBDLineCollisionModel") << "Context has no valid mechanical state, can not instantiate SofaPBDLineCollisionModel!";
                            return false;
                        }*/

                        bool boCanCreate = BaseObject::canCreate(obj, context, arg);
                        msg_info("SofaPBDLineCollisionModel") << "BaseObject::canCreate(): " << boCanCreate;

                        return boCanCreate;
                    }

                private:
                    SofaPBDLineCollisionModelPrivate* m_d;
            };
        }
    }
}

#endif // SOFAPBDLINECOLLISIONMODEL_H
