#ifndef PLANE3_H
#define PLANE3_H

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

#include "Intersectable.h"
#include "Line3.h"

namespace BVHModels
{
    template <typename Real>
    class Plane3Plane3IntersectionResult: public IntersectionResult<Real>
    {
        public:
            Plane3Plane3IntersectionResult(): IntersectionResult<Real>()
            {
                this->primitiveType1 = PT_PLANE;
                this->primitiveType2 = PT_PLANE;
                this->intersectionType = IT_EMPTY;
            }

            // Information about the intersection set.
            Line3<Real> mIntrLine;
    };

    template <typename Real>
    class Plane3: public Intersectable<Real, Vec<3,Real> >
    {
    public:
        // The plane is represented as Dot(N,X) = c where N is a unit-length
        // normal vector, c is the plane constant, and X is any point on the
        // plane.  The user must ensure that the normal vector is unit length.

        // Construction and destruction.
        Plane3 ();  // uninitialized
        ~Plane3 ();

        // Specify N and c directly.
        Plane3 (const Vec<3,Real>& normal, Real constant);

        // N is specified, c = Dot(N,P) where P is a point on the plane.
        Plane3 (const Vec<3,Real>& normal, const Vec<3,Real>& p);

        // N = Cross(P1-P0,P2-P0)/Length(Cross(P1-P0,P2-P0)), c = Dot(N,P0) where
        // P0, P1, P2 are points on the plane.
        Plane3 (const Vec<3,Real>& p0, const Vec<3,Real>& p1,
            const Vec<3,Real>& p2);

		PrimitiveType GetIntersectableType() const { return PT_PLANE; }

        bool IsIntersectionQuerySupported(const PrimitiveType &other);

        // Compute d = Dot(N,P)-c where N is the plane normal and c is the plane
        // constant.  This is a signed distance.  The sign of the return value is
        // positive if the point is on the positive side of the plane, negative if
        // the point is on the negative side, and zero if the point is on the
        // plane.
        Real DistanceTo (const Vec<3,Real>& p) const;

        // The "positive side" of the plane is the half space to which the plane
        // normal points.  The "negative side" is the other half space.  The
        // function returns +1 when P is on the positive side, -1 when P is on the
        // the negative side, or 0 when P is on the plane.
        int WhichSide (const Vec<3,Real>& p) const;

        bool Test(const Intersectable<Real, Vec<3, Real> > &);
        bool Find(const Intersectable<Real, Vec<3, Real> > &, IntersectionResult<Real>&);

        Vec<3,Real> Normal;
        Real Constant;
    };

    typedef Plane3<float> Plane3f;
    typedef Plane3<double> Plane3d;
}

#endif // PLANE3_H
