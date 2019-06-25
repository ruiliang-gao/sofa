#ifndef DISTANCEFIELDCOLLISIONDETECTION_H
#define DISTANCEFIELDCOLLISIONDETECTION_H

#include "PBDCommon/PBDCommon.h"
#include "CollisionDetection.h"
#include "Aabb.h"
#include "BoundingSphereHierarchy.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDDistanceBasedCD
        {
            /** Distance field collision detection. */
            class DistanceFieldCollisionDetection : public CollisionDetection
            {
            public:
                struct DistanceFieldCollisionObject : public CollisionObject
                {
                    bool m_testMesh;
                    Real m_invertSDF;
                    PointCloudBSH m_bvh;
                    TetMeshBSH m_bvhTets;
                    TetMeshBSH m_bvhTets0;

                    DistanceFieldCollisionObject() { m_testMesh = true; m_invertSDF = 1.0; }
                    virtual ~DistanceFieldCollisionObject() {}
                    virtual bool collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist = 0.0);
                    virtual void approximateNormal(const Eigen::Vector3d &x, const Real tolerance, Vector3r &n);

                    virtual double distance(const Eigen::Vector3d &x, const Real tolerance) = 0;
                    void initTetBVH(const Vector3r *vertices, const unsigned int numVertices, const unsigned int *indices, const unsigned int numTets, const Real tolerance);
                };

                struct DistanceFieldCollisionObjectWithoutGeometry : public DistanceFieldCollisionObject
                {
                    static int TYPE_ID;

                    virtual ~DistanceFieldCollisionObjectWithoutGeometry() {}
                    virtual int &getTypeId() const { return TYPE_ID; }
                    virtual bool collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist = 0.0) { return false; }
                    virtual double distance(const Eigen::Vector3d &x, const Real tolerance) { return 0.0; }
                };

                struct DistanceFieldCollisionBox : public DistanceFieldCollisionObject
                {
                    Vector3r m_box;
                    static int TYPE_ID;

                    virtual ~DistanceFieldCollisionBox() {}
                    virtual int &getTypeId() const { return TYPE_ID; }
                    virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
                };

                struct DistanceFieldCollisionSphere : public DistanceFieldCollisionObject
                {
                    Real m_radius;
                    static int TYPE_ID;

                    virtual ~DistanceFieldCollisionSphere() {}
                    virtual int &getTypeId() const { return TYPE_ID; }
                    virtual bool collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist = 0.0);
                    virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
                };

                struct DistanceFieldCollisionTorus : public DistanceFieldCollisionObject
                {
                    Vector2r m_radii;
                    static int TYPE_ID;

                    virtual ~DistanceFieldCollisionTorus() {}
                    virtual int &getTypeId() const { return TYPE_ID; }
                    virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
                };

                struct DistanceFieldCollisionCylinder : public DistanceFieldCollisionObject
                {
                    Vector2r m_dim;
                    static int TYPE_ID;

                    virtual ~DistanceFieldCollisionCylinder() {}
                    virtual int &getTypeId() const { return TYPE_ID; }
                    virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
                };

                struct DistanceFieldCollisionHollowSphere : public DistanceFieldCollisionObject
                {
                    Real m_radius;
                    Real m_thickness;
                    static int TYPE_ID;

                    virtual ~DistanceFieldCollisionHollowSphere() {}
                    virtual int &getTypeId() const { return TYPE_ID; }
                    virtual bool collisionTest(const Vector3r &x, const Real tolerance, Vector3r &cp, Vector3r &n, Real &dist, const Real maxDist = 0.0);
                    virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
                };

                struct DistanceFieldCollisionHollowBox : public DistanceFieldCollisionObject
                {
                    Vector3r m_box;
                    Real m_thickness;
                    static int TYPE_ID;

                    virtual ~DistanceFieldCollisionHollowBox() {}
                    virtual int &getTypeId() const { return TYPE_ID; }
                    virtual double distance(const Eigen::Vector3d &x, const Real tolerance);
                };

                struct DistanceFieldCollisionLine: public DistanceFieldCollisionObject
                {
                    Vector3r m_startPoint;
                    Vector3r m_endPoint;

                    int m_startPointParticleIndex;
                    int m_endPointParticleIndex;

                    Real m_thickness;
                    static int TYPE_ID;

                    DistanceFieldCollisionLine(): m_startPointParticleIndex(-1), m_endPointParticleIndex(-1) {}
                    virtual ~DistanceFieldCollisionLine() {}
                    virtual int &getTypeId() const { return TYPE_ID; }
                    virtual double distance(const Eigen::Vector3d &x, const Real tolerance);

                    void updateTransformation(const Vector3r &x, const Matrix3r &R, bool endPoint);
                };

                struct ContactData
                {
                    char m_type;
                    unsigned int m_index1;
                    unsigned int m_index2;
                    Vector3r m_cp1;
                    Vector3r m_cp2;
                    Vector3r m_normal;
                    Real m_dist;
                    Real m_restitution;
                    Real m_friction;

                    // Test
                    unsigned int m_elementIndex1;
                    unsigned int m_elementIndex2;
                    Vector3r m_bary1;
                    Vector3r m_bary2;
                };

            protected:
                void collisionDetectionRigidBodies(PBDRigidBody *rb1, DistanceFieldCollisionObject *co1, PBDRigidBody *rb2, DistanceFieldCollisionObject *co2,
                    const Real restitutionCoeff, const Real frictionCoeff
                    , std::vector<std::vector<ContactData> > &contacts_mt
                    );
                void collisionDetectionRBSolid(const ParticleData &pd, const unsigned int offset, const unsigned int numVert,
                    DistanceFieldCollisionObject *co1, PBDRigidBody *rb2, DistanceFieldCollisionObject *co2,
                    const Real restitutionCoeff, const Real frictionCoeff
                    , std::vector<std::vector<ContactData> > &contacts_mt
                    );

                void collisionDetectionSolidSolid(const ParticleData &pd, const unsigned int offset, const unsigned int numVert,
                    DistanceFieldCollisionObject *co1, PBDTetrahedronModel *tm2, DistanceFieldCollisionObject *co2,
                    const Real restitutionCoeff, const Real frictionCoeff
                    , std::vector<std::vector<ContactData> > &contacts_mt
                );

                bool findRefTetAt(const ParticleData &pd, PBDTetrahedronModel *tm, const DistanceFieldCollisionDetection::DistanceFieldCollisionObject *co, const Vector3r &X,
                    unsigned int &tetIndex, Vector3r &barycentricCoordinates);


            public:
                DistanceFieldCollisionDetection();
                virtual ~DistanceFieldCollisionDetection();

                virtual void collisionDetection(PBDSimulationModel &model);

                virtual bool isDistanceFieldCollisionObject(CollisionObject *co) const;

                void addCollisionBox(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector3r &box, const bool testMesh = true, const bool invertSDF = false);
                void addCollisionSphere(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Real radius, const bool testMesh = true, const bool invertSDF = false);
                void addCollisionTorus(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector2r &radii, const bool testMesh = true, const bool invertSDF = false);
                void addCollisionObjectWithoutGeometry(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const bool testMesh);
                void addCollisionHollowSphere(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Real radius, const Real thickness, const bool testMesh = true, const bool invertSDF = false);
                void addCollisionHollowBox(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector3r &box, const Real thickness, const bool testMesh = true, const bool invertSDF = false);

                // Line collision models
                void addCollisionLine(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r& v1, const Vector3r& v2, const int& startPtPartixleIdx, const int& endPtPartixleIdx, const Real thickness, const bool testMesh = true, const bool invertSDF = false);

                /** Add collision cylinder
                 *
                 * @param  bodyIndex index of corresponding body
                 * @param  bodyType type of corresponding body
                 * @param  dim (radius, height) of cylinder
                 */
                void addCollisionCylinder(const unsigned int bodyIndex, const unsigned int bodyType, const Vector3r *vertices, const unsigned int numVertices, const Vector2r &dim, const bool testMesh = true, const bool invertSDF = false);

                std::vector<ContactData> m_tempContacts;
            };
        }
    }
}

#endif // DISTANCEFIELDCOLLISIONDETECTION_H
