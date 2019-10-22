#ifndef SOFAPBDPOINTCOLLISIONMODEL_H
#define SOFAPBDPOINTCOLLISIONMODEL_H

#include "initZyPositionBasedDynamicsPlugin.h"
#include <SofaMeshCollision/PointModel.h>

#include "RigidBody.h"
#include "LineModel.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            using namespace sofa::core::objectmodel;

            class SofaPBDPointCollisionModel;

            template<class TDataTypes>
            class TPBDPoint : public core::TCollisionElementIterator<SofaPBDPointCollisionModel>
            {
            public:
                typedef TDataTypes DataTypes;
                typedef typename DataTypes::Coord Coord;
                typedef typename DataTypes::Deriv Deriv;
                typedef SofaPBDPointCollisionModel ParentModel;

                TPBDPoint(ParentModel* model, int index);
                TPBDPoint() {}

                explicit TPBDPoint(const core::CollisionElementIterator& i);

                const Coord p() const;
                const Coord pFree() const;
                const Deriv v() const;
                Deriv n() const;

                /// Return true if the element stores a free position vector
                bool hasFreePosition() const;

                bool testLMD(const sofa::defaulttype::Vector3 &, double &, double &);

                bool activated(core::CollisionModel *cm = nullptr) const;
            };

            class SofaPBDPointCollisionModelPrivate;
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDPointCollisionModel: /*public sofa::core::CollisionModel,*/ public sofa::component::collision::TPointModel<sofa::defaulttype::Vec3Types>
            {
                friend class TPBDPoint<sofa::defaulttype::Vec3Types>;
                public:
                    SOFA_CLASS(SofaPBDPointCollisionModel, sofa::component::collision::PointModel);

                    typedef TPBDPoint<sofa::defaulttype::Vec3Types> Element;

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

                    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

                    void parse(BaseObjectDescription* arg);

                    const int getPBDRigidBodyIndex() const;

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

                    const sofa::defaulttype::Vec3 getCoord(unsigned int) const;
                    const sofa::defaulttype::Vec3 getDeriv(unsigned int) const;

                private:
                    bool m_initCalled;
                    unsigned int m_initCallCount;
                    SofaPBDPointCollisionModelPrivate* m_d;
            };
        }
    }
}

#endif // SOFAPBDPOINTCOLLISIONMODEL_H
