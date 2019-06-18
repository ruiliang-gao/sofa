#ifndef OBBTREEGPU_COLLISIONMODEL_H
#define OBBTREEGPU_COLLISIONMODEL_H

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaRigid/RigidMapping.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaBaseCollision/CubeModel.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <gProximity/cuda_collision.h>
#include <PQP.h>
#include "SofaMeshCollision/Triangle.h"

class ModelInstance;

// Fabian Aichele, 09.12.2015: Ersatz für veraltete CUDA-GL-API vorbereitet
#define USE_DEPRECATED_CUDA_API

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace sofa::defaulttype;
            using namespace sofa::core::topology;
            using namespace sofa::helper;

            class ObbTreeGPUNode
            {
                public:
                    ObbTreeGPUNode(BV* bv = NULL) { m_bv = bv; }

                private:
                    BV* m_bv;
            };

            template <class DataTypes> class ObbTreeGPUCollisionModel;
            class ObbTreeGPUCollisionModelNode : public core::TCollisionElementIterator<ObbTreeGPUCollisionModel<Vec3Types> >
            {
                public:
                    typedef ObbTreeGPUNode Element;
                    ObbTreeGPUCollisionModelNode(ObbTreeGPUCollisionModel<Vec3Types>* model, int index);

                    explicit ObbTreeGPUCollisionModelNode(core::CollisionElementIterator& i);

                    const ObbTreeGPUNode& obb() const { return m_obb;}

                private:
                    ObbTreeGPUNode m_obb;
            };

            class ObbTreeGPUCollisionModelPrivate;
            template <class ObbTreeGPUDataTypes = Vec3Types>
            class ObbTreeGPUCollisionModel: public core::CollisionModel
            {
                public:
                    SOFA_CLASS(SOFA_TEMPLATE(ObbTreeGPUCollisionModel, ObbTreeGPUDataTypes), core::CollisionModel);

                    typedef ObbTreeGPUDataTypes DataTypes;
                    typedef typename ObbTreeGPUDataTypes::VecCoord VecCoord;
                    typedef typename ObbTreeGPUDataTypes::Real Real;

                    typedef ObbTreeGPUDataTypes InDataTypes;

                    typedef ObbTreeGPUCollisionModelNode Element;

                    ObbTreeGPUCollisionModel();
                    virtual ~ObbTreeGPUCollisionModel();

                    void computeBoundingTree(int maxDepth = 0);

                    void init();
                    void cleanup();

                    void draw(const core::visual::VisualParams*);

                    void* obbTree_device();
                    void* vertexPointer_device();
                    void* vertexTfPointer_device();
                    void* triIndexPointer_device();

                    unsigned int numVertices() const;
                    unsigned int numTriangles() const;
                    unsigned int numOBBs() const;

                    bool getVertex(const int &, Vector3 &outVertex);
                    bool getTriangle(const int&, sofa::core::topology::Triangle&); // added header after upstream merge

                    bool getPosition(Vector3&) const;
                    bool getOrientation(Matrix3&) const;
                    bool getOrientation(Quaternion&) const;

                    sofa::core::behavior::BaseMechanicalState* getObjectMState() { return m_objectBaseMState; }
                    sofa::core::behavior::MechanicalState<DataTypes>* getMState() { return m_objectMechanicalState; }
                    sofa::core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return m_objectMechanicalState; }

                    PQP_Model* getPqpModel() { return m_pqp_tree; }

                    void setDrawObbHierarchy(bool on)
                    {
						m_drawOBBHierarchy.setValue(on);
					}

                    ModelInstance* getModelInstance() { return m_gpModel; }

                    void updateInternalGeometry();

                    void setGPUModelTransform(void **tr_ptr);
                    void* getGPUModelTransform();

                    void *getTransformedVerticesPtr();

                    /*
                    const Vector3& getCachedModelPosition();
                    const Matrix3& getCachedModelOrientation();
                    */

                    CubeModel* getCubeModel() { return m_cubeModel; }

					defaulttype::Vec3d getCachedPosition();
					defaulttype::Quaternion getCachedOrientation();
                    defaulttype::Matrix3 getCachedOrientationMatrix();

                    void setCachedPosition(const defaulttype::Vec3d& position);

                    void setCachedOrientation(const defaulttype::Quaternion& orientation);

                    bool hasModelPositionChanged() const;

                    const bool& getUseContactManifolds () const { return useContactManifolds.getValue(); }
                    const unsigned int& getMaxNumberOfLineLineManifolds () const { return maxNumberOfLineLineManifolds.getValue(); }
                    const unsigned int& getMaxNumberOfFaceVertexManifolds () const { return maxNumberOfFaceVertexManifolds.getValue(); }

                    void setUseContactManifolds (bool on) { useContactManifolds.setValue(on); }
                    void setMaxNumberOfLineLineManifolds(unsigned int num) { maxNumberOfLineLineManifolds.setValue(num); }
                    void setMaxNumberOfFaceVertexManifolds(unsigned int num) { maxNumberOfFaceVertexManifolds.setValue(num); }

                    const boost::uuids::uuid& getUuid() { return m_uuid; }

                    gProximityContactType getContactType(const boost::uuids::uuid&, const int);
                    void setContactType(const boost::uuids::uuid&, const int, const gProximityContactType);

                    std::pair<int,int> getContactFeatures(const boost::uuids::uuid&, const int);
                    void setContactFeatures(const boost::uuids::uuid&, const int, const int, const int);

                private:
                    void computeOBBHierarchy(bool fromFile = true);
                    void initHierarchy();
                    void disposeHierarchy();

                    void computeBoundingTreeRec(PQP_Model* tree, BV* obb, Matrix3 treeOrientation, Vector3& accumulatedChildOffsets, int boxIndex, int currentDepth, int maxDepth);

                    CubeModel* m_cubeModel;

                    Data<int> m_drawOBBHierarchy;
                    Data<bool> m_useVertexBuffer;

                    Data<double> m_edgeLabelScaleFactor;

                    Data<bool> m_updateVerticesInModel;

                    std::string m_modelFile;
                    bool        m_modelLoaded;
                    ModelInstance* m_gpModel;
                    PQP_Model*     m_pqp_tree;
                    Vector3        m_initialPosition;
                    Quaternion     m_initialOrientation;
                    float          m_scale;
                    double         m_alarmDistance;

                    void**         m_modelTransform;

                    double m_lastTimestep;

                    ObbTreeGPUCollisionModelPrivate* m_d;

                    sofa::core::behavior::MechanicalState<DataTypes>* m_objectMechanicalState;
                    sofa::core::behavior::BaseMechanicalState* m_objectBaseMState;

                    sofa::component::container::MechanicalObject<DataTypes>* m_mechanicalObject;

					defaulttype::Vec3d m_oldCachedPosition;
					defaulttype::Quaternion m_oldCachedOrientation;
                    bool m_cachedPositionChange;
                    bool m_cachedOrientationChange;

                    Data<bool> useContactManifolds;
                    Data<unsigned int> maxNumberOfLineLineManifolds;
                    Data<unsigned int> maxNumberOfFaceVertexManifolds;

					// Fake gripping convenience member variables and accessors
					sofa::component::container::MechanicalObject<Rigid3Types>* m_fromObject;
					sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* m_innerMapping;

                    Data<sofa::defaulttype::Vector3> m_appliedTranslation;
                    Data<sofa::defaulttype::Quaternion> m_appliedRotation;

                    boost::uuids::uuid m_uuid;

				public:
					sofa::component::container::MechanicalObject<Rigid3Types>* getAttachFromObject() { return m_fromObject; }
					sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* getAttachFromInnerMapping() { return m_innerMapping; }

					void setAttachFromObject(sofa::component::container::MechanicalObject<Rigid3Types>* obj) { m_fromObject = obj; }
                    void setAttachFromInnerMapping(sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* mapping) { m_innerMapping = mapping; }

            protected:
                    void getCachedPositionAndOrientation();
            };
        }
    }
}

#endif // OBBTREEGPU_COLLISIONMODEL_H
