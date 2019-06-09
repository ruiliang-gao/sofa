#ifndef OBBTREECPUCOLLISIONMODEL_H
#define OBBTREECPUCOLLISIONMODEL_H

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaBaseCollision/CubeModel.h>

//class ModelInstance;
//#include <PQP.h>

#include "ObbTree.h"

namespace sofa
{
    namespace component
    {
        namespace collision
        {

            using namespace sofa::defaulttype;
            using namespace sofa::core::topology;
            using namespace sofa::helper;

            class ObbTreeCPUNode
            {
                public:
                    ObbTreeCPUNode() { }
                    ObbTreeCPUNode(const ObbVolume& bv) { m_bv = bv; }

                private:
                    ObbVolume m_bv;
            };

            template <class DataTypes> class ObbTreeCPUCollisionModel;
            class ObbTreeCPUCollisionModelNode : public core::TCollisionElementIterator<ObbTreeCPUCollisionModel<Vec3Types> >
            {
                public:
                    typedef ObbTreeCPUNode Element;
                    ObbTreeCPUCollisionModelNode(ObbTreeCPUCollisionModel<Vec3Types>* model, int index);

                    explicit ObbTreeCPUCollisionModelNode(core::CollisionElementIterator& i);

                    const ObbTreeCPUNode& obb() const { return m_obb;}

                private:
                    ObbTreeCPUNode m_obb;
            };

            template <class DataTypes = Vec3Types>
            class ObbTreeCPUCollisionModel: public core::CollisionModel
            {
                public:
                    SOFA_CLASS(SOFA_TEMPLATE(ObbTreeCPUCollisionModel, DataTypes), core::CollisionModel);

                    typedef ObbTreeCPUCollisionModelNode Element;

                    ObbTreeCPUCollisionModel();
                    virtual ~ObbTreeCPUCollisionModel();

                    void computeBoundingTree(int maxDepth = 0);

                    void init();
                    void cleanup();

                    void draw(const core::visual::VisualParams*);

                    //PQP_Model* getPQPModel() { return m_pqp_tree; }
                    ObbTree& getObbTree() { return m_obbTree; }

                    unsigned int numVertices();
                    unsigned int numTriangles();
                    unsigned int numOBBs();

                    bool getVertex(unsigned int, Vector3 &outVertex);
                    bool getTriangle(unsigned int, sofa::core::topology::Triangle&);

                    bool getPosition(Vector3&) const;
                    bool getOrientation(Matrix3&) const;
                    bool getOrientation(Quaternion&) const;

                    const sofa::core::behavior::BaseMechanicalState* getObjectMState() { return _objectMState; }

                    void setEmphasizedIndices(const std::vector<int>& indices) { m_emphasizedIndices = indices; }

                private:
                    void computeOBBHierarchy();

                    Data<bool> m_drawOBBHierarchy;
                    Data<unsigned int> m_maxDrawDepth;
                    Data<unsigned int> m_minDrawDepth;

                    std::string m_modelFile;
                    bool        m_modelLoaded;
                    //ModelInstance* m_gpModel;
                    //PQP_Model*     m_pqp_tree;

                    Vector3        m_initialPosition;
                    Quaternion     m_initialOrientation;
                    float          m_scale;

                    ObbTree m_obbTree;

                    std::vector<int> m_emphasizedIndices;

                    sofa::core::behavior::MechanicalState<DataTypes>* _mState;
                    sofa::core::behavior::BaseMechanicalState* _objectMState;

                    sofa::component::container::MechanicalObject<DataTypes>* _mObject;

                    sofa::component::collision::CubeModel* m_cubeModel;
            };

        }
    }
}

#endif // OBBTREECPUCOLLISIONMODEL_H
