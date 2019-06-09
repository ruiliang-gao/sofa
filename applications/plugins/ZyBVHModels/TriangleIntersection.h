#ifndef TRIANGLEINTERSECTION_H
#define TRIANGLEINTERSECTION_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>

using namespace sofa;
using namespace sofa::defaulttype;

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            class TriangleIntersection
            {
                public:
                    TriangleIntersection();
                    virtual bool intersect_tri_tri(const Vector3& p0, const Vector3& p1, const Vector3& p2,
                                                   const Vector3& q0, const Vector3& q1, const Vector3& q2) = 0;
            };

            class TriangleContact
            {
                public:
                    TriangleContact();
                    virtual bool contacts_tri_tri(const Vector3& p0, const Vector3& p1, const Vector3& p2,
                                                  const Vector3& q0, const Vector3& q1, const Vector3& q2) = 0;
            };

            class TriIntersectMoeller: public TriangleIntersection
            {
                public:
                    bool intersect_tri_tri(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &q0, const Vector3 &q1, const Vector3 &q2);

                private:
                    void TRANSFORM_TRIANGLE(const Matrix4& t, Vector3& vert0, Vector3& vert1, Vector3& vert2);
                    void SORT(double& a, double& b);
                    bool EDGE_EDGE_TEST(const Vector3& V0, const Vector3& U0, const Vector3& U1, const unsigned short& i0, const unsigned short& i1, double& Ax, double& Ay, double& Bx, double& By, double& Cx, double& Cy, double& d, double& f);
                    bool EDGE_AGAINST_TRI_EDGES(const Vector3& V0, const Vector3& V1, const Vector3& U0, const Vector3& U1, const Vector3& U2, const unsigned short& i0, const unsigned short& i1);
                    bool POINT_IN_TRI(const Vector3& V0, const Vector3& U0, const Vector3& U1, const Vector3& U2, const unsigned short& i0, const unsigned short& i1);
                    bool CoplanarTriTri(const Vector3& n,
                                   const Vector3& v0, const Vector3& v1, const Vector3& v2,
                                   const Vector3& u0, const Vector3& u1, const Vector3& u2, unsigned short& i0, unsigned short& i1);
                    bool NEWCOMPUTE_INTERVALS(double& VV0, double& VV1, double& VV2, double& D0, double& D1, double& D2, double& D0D1, double& D0D2, double& A, double& B, double& C, double& X0, double& X1);

                    bool tri_tri_overlap_3d(const Vector3 t1[], const Vector3 t2[]);
            };

            class TriIntersectGuigue: public TriangleIntersection
            {
                public:
                    bool intersect_tri_tri(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &q0, const Vector3 &q1, const Vector3 &q2);

                private:
                    bool CHECK_MIN_MAX(const Vector3& p1, const Vector3& q1, const Vector3& r1, const Vector3& p2, const Vector3& q2, const Vector3& r2);
                    bool TRI_TRI_3D(const Vector3& p1, const Vector3& q1, const Vector3& r1, const Vector3& p2, const Vector3& q2, const Vector3& r2, const double& dp2, const double& dq2, const double& dr2, const Vector3& N1);
                    double ORIENT_2D(const Vector2 &a, const Vector2 &b, const Vector2 &c);
                    bool INTERSECTION_TEST_VERTEX(const Vector2& P1, const Vector2& Q1, const Vector2& R1, const Vector2& P2, const Vector2& Q2, const Vector2& R2);
                    bool INTERSECTION_TEST_EDGE(const Vector2& P1, const Vector2& Q1, const Vector2& R1, const Vector2& P2, const Vector2& Q2, const Vector2& R2);

                    bool ccw_tri_tri_intersection_2d(const Vector2& p1, const Vector2& q1, const Vector2& r1,
                                                     const Vector2& p2, const Vector2& q2, const Vector2& r2);
                    bool tri_tri_overlap_test_2d(const Vector2& p1, const Vector2& q1, const Vector2& r1,
                                                 const Vector2& p2, const Vector2& q2, const Vector2& r2);

                    bool coplanar_tri_tri3d(const Vector3& p1, const Vector3& q1, const Vector3& r1,
                                            const Vector3& p2, const Vector3& q2, const Vector3& r2,
                                            const Vector3& normal);
                    bool tri_tri_overlap_3d(const Vector3 t1[], const Vector3 t2[]);

            };

            class TriIntersectPQP: public TriangleIntersection
            {
                public:
                    bool intersect_tri_tri(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &q0, const Vector3 &q1, const Vector3 &q2);

                private:
                    double max3(double a, double b, double c);
                    double min3(double a, double b, double c);
                    bool project6(const Vector3& ax,
                                  const Vector3& p1, const Vector3& p2, const Vector3& p3,
                                  const Vector3& q1, const Vector3& q2, const Vector3& q3);
                    bool tri_tri_overlap_3d(const Vector3 t1[], const Vector3 t2[]);

            };
        }
    }
}

#endif // TRIANGLEINTERSECTION_H
