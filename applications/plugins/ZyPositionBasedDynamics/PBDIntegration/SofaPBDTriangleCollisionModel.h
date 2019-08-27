#ifndef SOFAPBDTRIANGLECOLLISIONMODEL_H
#define SOFAPBDTRIANGLECOLLISIONMODEL_H

#include "initZyPositionBasedDynamicsPlugin.h"
#include <SofaMeshCollision/TriangleModel.h>

#include "PBDRigidBody.h"
#include "PBDModels/PBDLineModel.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            using namespace sofa::core::objectmodel;
            using namespace sofa::core;

            class SofaPBDTriangleCollisionModel;

            template<class TDataTypes>
            class TPBDTriangle : public core::TCollisionElementIterator<SofaPBDTriangleCollisionModel>
            {
                public:
                    typedef TDataTypes DataTypes;
                    typedef typename DataTypes::Coord Coord;
                    typedef typename DataTypes::Deriv Deriv;
                    typedef SofaPBDTriangleCollisionModel ParentModel;
                    typedef typename DataTypes::Real Real;

                    TPBDTriangle(ParentModel* model, int index);
                    TPBDTriangle() {}
                    explicit TPBDTriangle(const core::CollisionElementIterator& i);
                    TPBDTriangle(ParentModel* model, int index, helper::ReadAccessor<typename DataTypes::VecCoord>& /*x*/);

                    const Coord p1() const;
                    const Coord p2() const;
                    const Coord p3() const;

                    int p1Index() const;
                    int p2Index() const;
                    int p3Index() const;

                    const Coord p1Free() const;
                    const Coord p2Free() const;
                    const Coord p3Free() const;

                    const Deriv v1() const;
                    const Deriv v2() const;
                    const Deriv v3() const;

                    const Deriv n() const;
                    Deriv n();

                    /// Return true if the element stores a free position vector
                    bool hasFreePosition() const;

                    int flags() const;

                    TPBDTriangle& shape() { return *this; }
                    const TPBDTriangle& shape() const { return *this; }

                    Coord interpX(defaulttype::Vec<2,Real> bary) const
                    {
                        return (p1() * (1 - bary[0] - bary[1])) + (p2() * bary[0]) + (p3() * bary[1]);
                    }
            };

            class SofaPBDTriangleCollisionModelPrivate;
            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDTriangleCollisionModel: /*public sofa::core::CollisionModel,*/ public sofa::component::collision::TriangleModel
            {
                friend class TPBDTriangle<sofa::defaulttype::Vec3Types>;
                public:
                    SOFA_CLASS(SofaPBDTriangleCollisionModel, sofa::component::collision::TriangleModel);

                    typedef TPBDTriangle<sofa::defaulttype::Vec3Types> Element;
                    typedef topology::BaseMeshTopology Topology;

                    SofaPBDTriangleCollisionModel();
                    ~SofaPBDTriangleCollisionModel();

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
                        msg_info("SofaPBDTriangleCollisionModel") << "BaseObject::canCreate(): " << boCanCreate;

                        return boCanCreate;
                    }

                    const PBDRigidBody* getPBDRigidBody() const;
                    PBDRigidBody* getPBDRigidBody();

                    const sofa::defaulttype::Vec3 getCoord(unsigned int) const;

                    const sofa::defaulttype::Vec3& getVertex1(const unsigned int) const;
                    const sofa::defaulttype::Vec3& getVertex2(const unsigned int) const;
                    const sofa::defaulttype::Vec3& getVertex3(const unsigned int) const;

                    const int getVertex1Idx(const unsigned int) const;
                    const int getVertex2Idx(const unsigned int) const;
                    const int getVertex3Idx(const unsigned int) const;

                    const int getEdge1Idx(const unsigned int) const;
                    const int getEdge2Idx(const unsigned int) const;
                    const int getEdge3Idx(const unsigned int) const;

                    const sofa::helper::fixed_array<unsigned int,3> getEdgesInTriangle(unsigned int) const;
                    int getTriangleFlags(Topology::TriangleID i);

                    Data<bool>  showIndices; ///< Show indices. (default=false)
                    Data<float> showIndicesScale; ///< Scale for indices display. (default=0.02)

                private:
                    SofaPBDTriangleCollisionModelPrivate* m_d;
            };
        }
    }
}

#endif // SOFAPBDTRIANGLECOLLISIONMODEL_H
