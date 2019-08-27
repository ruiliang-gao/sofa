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

            class SofaPBDLineCollisionModel;

            template<class TDataTypes>
            class TPBDLine : public core::TCollisionElementIterator<SofaPBDLineCollisionModel>
            {
                public:
                    typedef TDataTypes DataTypes;
                    typedef typename DataTypes::Coord Coord;
                    typedef typename DataTypes::Deriv Deriv;
                    typedef SofaPBDLineCollisionModel ParentModel;

                    TPBDLine(ParentModel* model, int index);
                    TPBDLine() {}

                    explicit TPBDLine(const core::CollisionElementIterator& i);

                    int i1() const;
                    int i2() const;
                    int flags() const;

                    const Coord p1() const;
                    const Coord p2() const;

                    const Coord p1Free() const;
                    const Coord p2Free() const;

                    const Deriv v1() const;
                    const Deriv v2() const;

                    /// Return true if the element stores a free position vector
                    bool hasFreePosition() const;

                    bool activated(core::CollisionModel *cm = nullptr) const;
            };

            class SofaPBDLineCollisionModelPrivate;
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDLineCollisionModel: /*public sofa::core::CollisionModel,*/ public sofa::component::collision::LineModel
            {
                friend class TPBDLine<sofa::defaulttype::Vec3Types>;
                public:
                    SOFA_CLASS(SofaPBDLineCollisionModel, sofa::component::collision::LineModel);

                    typedef TPBDLine<sofa::defaulttype::Vec3Types> Element;

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

                    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

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

                    bool usesPBDRigidBody() const;
                    bool usesPBDLineModel() const;

                    const PBDRigidBody* getPBDRigidBody() const;
                    PBDRigidBody* getPBDRigidBody();

                    const PBDLineModel* getPBDLineModel() const;
                    PBDLineModel* getPBDLineModel();

                    const sofa::defaulttype::Vec3 getCoord(unsigned int) const;
                    const sofa::defaulttype::Vec3 getDeriv(unsigned int) const;

                private:
                    SofaPBDLineCollisionModelPrivate* m_d;
            };
        }
    }
}

#endif // SOFAPBDLINECOLLISIONMODEL_H
