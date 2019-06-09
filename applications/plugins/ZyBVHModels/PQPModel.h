#ifndef PQPMODEL_H
#define PQPMODEL_H

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaBaseCollision/CubeModel.h>

#include <sofa/core/loader/MeshLoader.h>

#include <PQP.h>

#include "initBVHModelsPlugin.h"

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace sofa::defaulttype;
            using namespace sofa::core::topology;
            using namespace sofa::helper;

            class PQPTreeNode
            {
                public:
                    PQPTreeNode(BV* bv = NULL) { m_bv = bv; }

                private:
                    BV* m_bv;
            };

            template <class DataTypes> class PQPCollisionModel;
            class PQPCollisionModelNode : public core::TCollisionElementIterator<PQPCollisionModel<Vec3Types> >
            {
                public:
                    typedef PQPTreeNode Element;
                    PQPCollisionModelNode(PQPCollisionModel<Vec3Types>* model, int index);

                    explicit PQPCollisionModelNode(core::CollisionElementIterator& i);

                    const PQPTreeNode& obb() const { return m_obb;}

                private:
                    PQPTreeNode m_obb;
            };

            template <class DataTypes = Vec3Types>
            class SOFA_BVHMODELSPLUGIN_API PQPCollisionModel: public core::CollisionModel
            {
                public:
                    SOFA_CLASS(SOFA_TEMPLATE(PQPCollisionModel, DataTypes), core::CollisionModel);

                    typedef PQPCollisionModelNode Element;

                    PQPCollisionModel();

                    virtual ~PQPCollisionModel();

                    void computeBoundingTree(int maxDepth = 0);

                    void init();
                    void cleanup();

                    void draw(const core::visual::VisualParams*);

                    sofa::core::behavior::BaseMechanicalState* getObjectMState() { return _objectMState; }
                    sofa::core::behavior::MechanicalState<DataTypes>* getMState() { return _mState; }
                    sofa::core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return _mState; }

                    CubeModel* getCubeModel() { return m_cubeModel; }

                    void getCachedPositionAndOrientation();

                    int getNumBVs() const
                    {
                        if (m_pqp_tree)
                            return m_pqp_tree->num_bvs;

                        return 0;
                    }

                    BV* getChild(int idx)
                    {
                        if (m_pqp_tree && idx < m_pqp_tree->num_bvs)
                            return m_pqp_tree->child(idx);

                        return NULL;
                    }

                    PQP_Model* getPQPModel()
                    {
                        return m_pqp_tree;
                    }

                    bool getPosition(Vector3&) const;
                    bool getOrientation(Matrix3&) const;
                    bool getOrientation(Quaternion&) const;

                    void setCachedPosition(const Vec3d &position);
                    void setCachedOrientation(const Quaternion &orientation);

                private:
                    void computeOBBHierarchy(bool fromFile = true);

                    void computeBoundingTreeRec(PQP_Model* tree, BV* obb, Matrix3 treeOrientation, Vector3& accumulatedChildOffsets, int boxIndex, int currentDepth, int maxDepth);

                    sofa::core::behavior::MechanicalState<DataTypes>* _mState;
                    sofa::component::container::MechanicalObject<DataTypes>* _mObject;
                    sofa::core::behavior::BaseMechanicalState* _objectMState;

                    sofa::component::collision::CubeModel* m_cubeModel;

                    sofa::core::loader::MeshLoader* m_meshLoader;

                    std::string    m_modelFile;
                    bool           m_modelLoaded;
                    PQP_Model*     m_pqp_tree;

                    Vector3        m_initialPosition;
                    Quaternion     m_initialOrientation;
                    Real           m_scale;
                    Matrix3        m_modelOrientation;
                    Vector3        m_modelTranslation;

                    Real           m_lastTimestep;

                    Data<int>      m_drawOBBHierarchy;
            };
        }
    }
}

#endif // PQPMODEL_H
