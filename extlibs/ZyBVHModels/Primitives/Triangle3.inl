#include "Triangle3.h"

#include "Math/MathUtils.h"
#include "Query/Query2.h"

#include "Segment2.h"
#include "Triangle2.h"

#include "Segment3.h"

#include <GL/gl.h>

using namespace BVHModels;

//----------------------------------------------------------------------------
template <typename Real>
Triangle3<Real>::Triangle3 ()
{
}
//----------------------------------------------------------------------------
template <typename Real>
Triangle3<Real>::~Triangle3 ()
{
}
//----------------------------------------------------------------------------
template <typename Real>
Triangle3<Real>::Triangle3 (const Vec<3,Real>& v0,
    const Vec<3,Real>& v1, const Vec<3,Real>& v2)
{
    V[0] = v0;
    V[1] = v1;
    V[2] = v2;
}
//----------------------------------------------------------------------------
template <typename Real>
Triangle3<Real>::Triangle3(const Vec<3,Real> vertex[])
{
    for (int i = 0; i < 3; ++i)
    {
        V[i] = vertex[i];
    }
}

//----------------------------------------------------------------------------
template <typename Real>
bool Triangle3<Real>::ContainsPoint(const Vec<3, Real>& q)
{
	Plane3<Real> trianglePlane(V[0], V[1], V[2]);
	Triangle3IntersectionResult<Real> trianglePointResult;
	return this->ContainsPoint(*this, trianglePlane, q, trianglePointResult);
}

template <typename Real>
void Triangle3<Real>::draw()
{
	glPointSize(12.0f);
	glBegin(GL_POINTS);
	glVertex3d(V[0].x(), V[0].y(), V[0].z());
	glVertex3d(V[1].x(), V[1].y(), V[1].z());
	glVertex3d(V[2].x(), V[2].y(), V[2].z());
	glEnd();
	glPointSize(1.0f);
}

//----------------------------------------------------------------------------
template <typename Real>
Real Triangle3<Real>::DistanceTo(const Vec<3,Real>& q) const
{
    Vec<3,Real> diff = V[0] - q;
    Vec<3,Real> edge0 = V[1] - V[0];
    Vec<3,Real> edge1 = V[2] - V[0];
    Real a00 = edge0.norm2();
    Real a01 = edge0 * edge1;
    Real a11 = edge1.norm2();
    Real b0 = diff * edge0;
    Real b1 = diff * edge1;
    Real c = diff.norm2();
    Real det = fabs(a00*a11 - a01*a01);
    Real s = a01*b1 - a11*b0;
    Real t = a01*b0 - a00*b1;
    Real sqrDistance;

    if (s + t <= det)
    {
        if (s < (Real)0)
        {
            if (t < (Real)0)  // region 4
            {
                if (b0 < (Real)0)
                {
                    if (-b0 >= a00)
                    {
                        sqrDistance = a00 + ((Real)2)*b0 + c;
                    }
                    else
                    {
                        sqrDistance = c - b0*b0/a00;
                    }
                }
                else
                {
                    if (b1 >= (Real)0)
                    {
                        sqrDistance = c;
                    }
                    else if (-b1 >= a11)
                    {
                        sqrDistance = a11 + ((Real)2)*b1 + c;
                    }
                    else
                    {
                        sqrDistance = c - b1*b1/a11;
                    }
                }
            }
            else  // region 3
            {
                if (b1 >= (Real)0)
                {
                    sqrDistance = c;
                }
                else if (-b1 >= a11)
                {
                    sqrDistance = a11 + ((Real)2)*b1 + c;
                }
                else
                {
                    sqrDistance = c - b1*b1/a11;
                }
            }
        }
        else if (t < (Real)0)  // region 5
        {
            if (b0 >= (Real)0)
            {
                sqrDistance = c;
            }
            else if (-b0 >= a00)
            {
                sqrDistance = a00 + ((Real)2)*b0 + c;
            }
            else
            {
                sqrDistance = b0*s + c - b0*b0/a00;
            }
        }
        else  // region 0
        {
            // The minimum is at an interior point of the triangle.
            Real invDet = ((Real)1)/det;
            s *= invDet;
            t *= invDet;
            sqrDistance = s*(a00*s + a01*t + ((Real)2)*b0) +
                t*(a01*s + a11*t + ((Real)2)*b1) + c;
        }
    }
    else
    {
        Real tmp0, tmp1, numer, denom;

        if (s < (Real)0)  // region 2
        {
            tmp0 = a01 + b0;
            tmp1 = a11 + b1;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - ((Real)2)*a01 + a11;
                if (numer >= denom)
                {
                    sqrDistance = a00 + ((Real)2)*b0 + c;
                }
                else
                {
                    s = numer/denom;
                    t = (Real)1 - s;
                    sqrDistance = s*(a00*s + a01*t + ((Real)2)*b0) +
                        t*(a01*s + a11*t + ((Real)2)*b1) + c;
                }
            }
            else
            {
                if (tmp1 <= (Real)0)
                {
                    sqrDistance = a11 + ((Real)2)*b1 + c;
                }
                else if (b1 >= (Real)0)
                {
                    sqrDistance = c;
                }
                else
                {
                    sqrDistance = c - b1*b1/a11;
                }
            }
        }
        else if (t < (Real)0)  // region 6
        {
            tmp0 = a01 + b1;
            tmp1 = a00 + b0;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - ((Real)2)*a01 + a11;
                if (numer >= denom)
                {
                    t = (Real)1;
                    s = (Real)0;
                    sqrDistance = a11 + ((Real)2)*b1 + c;
                }
                else
                {
                    t = numer/denom;
                    s = (Real)1 - t;
                    sqrDistance = s*(a00*s + a01*t + ((Real)2)*b0) +
                        t*(a01*s + a11*t + ((Real)2)*b1) + c;
                }
            }
            else
            {
                if (tmp1 <= (Real)0)
                {
                    sqrDistance = a00 + ((Real)2)*b0 + c;
                }
                else if (b0 >= (Real)0)
                {
                    sqrDistance = c;
                }
                else
                {
                    sqrDistance = c - b0*b0/a00;
                }
            }
        }
        else  // region 1
        {
            numer = a11 + b1 - a01 - b0;
            if (numer <= (Real)0)
            {
                sqrDistance = a11 + ((Real)2)*b1 + c;
            }
            else
            {
                denom = a00 - ((Real)2)*a01 + a11;
                if (numer >= denom)
                {
                    sqrDistance = a00 + ((Real)2)*b0 + c;
                }
                else
                {
                    s = numer/denom;
                    t = (Real)1 - s;
                    sqrDistance = s*(a00*s + a01*t + ((Real)2)*b0) +
                        t*(a01*s + a11*t + ((Real)2)*b1) + c;
                }
            }
        }
    }

    return sqrt(fabs(sqrDistance));
}

//----------------------------------------------------------------------------
template <typename Real>
bool Triangle3<Real>::Test(const Intersectable<Real, Vec<3,Real> >& intersectable)
{
    //TRU Triangle3<Real>* tri2 = dynamic_cast<Triangle3<Real>*>(intersectable);
    const Triangle3<Real>* tri2 = &(dynamic_cast<const Triangle3<Real>&>(intersectable));

    if (!tri2)
        return false;

    // Get edge vectors for triangle0.
    Vec<3,Real> E0[3] =
    {
        this->V[1] - this->V[0],
        this->V[2] - this->V[1],
        this->V[0] - this->V[2]
    };

    // Get normal vector of triangle0.
    //TRU Vec<3,Real> N0 = E0[0].UnitCross(E0[1]);
    Vec<3,Real> N0 = E0[0].cross(E0[1]);
    N0.normalize();

    // Project triangle1 onto normal line of triangle0, test for separation.
    Real N0dT0V0 = N0 * tri2->V[0];
    Real min1, max1;
    ProjectOntoAxis(*tri2, N0, min1, max1);
    if (N0dT0V0 < min1 || N0dT0V0 > max1)
    {
        return false;
    }

    // Get edge vectors for triangle1.
    Vec<3,Real> E1[3] =
    {
        tri2->V[1] - tri2->V[0],
        tri2->V[2] - tri2->V[1],
        tri2->V[0] - tri2->V[2]
    };

    // Get normal vector of triangle1.
    //TRU Vec<3,Real> N1 = E1[0].UnitCross(E1[1]);
    Vec<3,Real> N1 = E1[0].cross(E1[1]);
    N1.normalize();

    Vec<3,Real> dir;
    Real min0, max0;
    int i0, i1;

    //TRU Vec<3,Real> N0xN1 = N0.UnitCross(N1);
    Vec<3,Real> N0xN1 = N0.cross(N1);
    N0xN1.normalize();

    //TRU if (N0xN1.Dot(N0xN1) >= MathUtils<Real>::ZERO_TOLERANCE)
    if ((N0xN1 * N0xN1) >= MathUtils<Real>::ZERO_TOLERANCE)
    {
        // Triangles are not parallel.

        // Project triangle0 onto normal line of triangle1, test for
        // separation.
        Real N1dT1V0 = N1 * tri2->V[0];
        ProjectOntoAxis(*this, N1, min0, max0);
        if (N1dT1V0 < min0 || N1dT1V0 > max0)
        {
            return false;
        }

        // Directions E0[i0]xE1[i1].
        for (i1 = 0; i1 < 3; ++i1)
        {
            for (i0 = 0; i0 < 3; ++i0)
            {
                //TRU dir = E0[i0].UnitCross(E1[i1]);
                dir = E0[i0].cross(E1[i1]);
                dir.normalize();
                ProjectOntoAxis(*this, dir, min0, max0);
                ProjectOntoAxis(*tri2, dir, min1, max1);
                if (max0 < min1 || max1 < min0)
                {
                    return false;
                }
            }
        }

        // The test query does not know the intersection set.
        // mIntersectionType = IT_OTHER;
    }
    else  // Triangles are parallel (and, in fact, coplanar).
    {
        // Directions N0xE0[i0].
        for (i0 = 0; i0 < 3; ++i0)
        {
            //TRU dir = N0.UnitCross(E0[i0]);
            dir = N0.cross(E0[i0]);
            dir.normalize();
            ProjectOntoAxis(*this, dir, min0, max0);
            ProjectOntoAxis(*tri2, dir, min1, max1);
            if (max0 < min1 || max1 < min0)
            {
                return false;
            }
        }

        // Directions N1xE1[i1].
        for (i1 = 0; i1 < 3; ++i1)
        {
            //TRU dir = N1.UnitCross(E1[i1]);
            dir = N1.cross(E1[i1]);
            dir.normalize();
            ProjectOntoAxis(*this, dir, min0, max0);
            ProjectOntoAxis(*tri2, dir, min1, max1);
            if (max0 < min1 || max1 < min0)
            {
                return false;
            }
        }

        // The test query does not know the intersection set.
        // mIntersectionType = IT_PLANE;
    }

    return true;
}
//----------------------------------------------------------------------------
template <typename Real>
bool Triangle3<Real>::Find(const Intersectable<Real, Vec<3,Real> >& intersectable, IntersectionResult<Real>& result)
{
    const Triangle3<Real>* tri2 = &(dynamic_cast<const Triangle3<Real>&>(intersectable));
    Triangle3IntersectionResult<Real>* triRes = static_cast<Triangle3IntersectionResult<Real>*>(&result);

    if (!tri2 || !triRes)
        return false;

    int i, iM, iP;

    // Get the plane of triangle0.
    Plane3<Real> plane0(this->V[0], this->V[1],
        this->V[2]);

    // Compute the signed distances of triangle1 vertices to plane0.  Use
    // an epsilon-thick plane test.
    int pos1, neg1, zero1, sign1[3];
    Real dist1[3];
    TrianglePlaneRelations(*tri2, plane0, dist1, sign1, pos1, neg1,
        zero1);

    if (pos1 == 3 || neg1 == 3)
    {
        // Triangle1 is fully on one side of plane0.
        return false;
    }

    if (zero1 == 3)
    {
        // Triangle1 is contained by plane0.
        if (triRes->mReportCoplanarIntersections)
        {
            return GetCoplanarIntersection(plane0, *this,
                *tri2, *triRes);
        }
        return false;
    }

    // Check for grazing contact between triangle1 and plane0.
    if (pos1 == 0 || neg1 == 0)
    {
        if (zero1 == 2)
        {
            // An edge of triangle1 is in plane0.
            for (i = 0; i < 3; ++i)
            {
                if (sign1[i] != 0)
                {
                    iM = (i + 2) % 3;
                    iP = (i + 1) % 3;
                    return IntersectsSegment(plane0, *this, tri2->V[iM], tri2->V[iP], *triRes);
                }
            }
        }
        else // zero1 == 1
        {
            // A vertex of triangle1 is in plane0.
            for (i = 0; i < 3; ++i)
            {
                if (sign1[i] == 0)
                {
                    return ContainsPoint(*this, plane0, tri2->V[i], *triRes);
                }
            }
        }
    }

    // At this point, triangle1 tranversely intersects plane 0.  Compute the
    // line segment of intersection.  Then test for intersection between this
    // segment and triangle 0.
    Real t;
    Vec<3,Real> intr0, intr1;
    if (zero1 == 0)
    {
        int iSign = (pos1 == 1 ? +1 : -1);
        for (i = 0; i < 3; ++i)
        {
            if (sign1[i] == iSign)
            {
                iM = (i + 2) % 3;
                iP = (i + 1) % 3;
                t = dist1[i]/(dist1[i] - dist1[iM]);
                intr0 = tri2->V[i] + t*(tri2->V[iM] -
                    tri2->V[i]);
                t = dist1[i]/(dist1[i] - dist1[iP]);
                intr1 = tri2->V[i] + t*(tri2->V[iP] -
                    tri2->V[i]);
                return IntersectsSegment(plane0, *this, intr0, intr1, *triRes);
            }
        }
    }

    // zero1 == 1
    for (i = 0; i < 3; ++i)
    {
        if (sign1[i] == 0)
        {
            iM = (i + 2) % 3;
            iP = (i + 1) % 3;
            t = dist1[iM]/(dist1[iM] - dist1[iP]);
            intr0 = tri2->V[iM] + t * (tri2->V[iP] -
                tri2->V[iM]);
            return IntersectsSegment(plane0, *this,
                tri2->V[i],intr0, *triRes);
        }
    }

    // assertion(false, "Should not get here\n");
    return false;
}

//----------------------------------------------------------------------------
template <typename Real>
void Triangle3<Real>::ProjectOntoAxis (
    const Triangle3<Real>& triangle, const Vec<3,Real>& axis, Real& fmin,
    Real& fmax)
{
    Real dot0 = axis * triangle.V[0];
    Real dot1 = axis * triangle.V[1];
    Real dot2 = axis * triangle.V[2];

    fmin = dot0;
    fmax = fmin;

    if (dot1 < fmin)
    {
        fmin = dot1;
    }
    else if (dot1 > fmax)
    {
        fmax = dot1;
    }

    if (dot2 < fmin)
    {
        fmin = dot2;
    }
    else if (dot2 > fmax)
    {
        fmax = dot2;
    }
}

//----------------------------------------------------------------------------
template <typename Real>
void Triangle3<Real>::ProjectOntoAxis (
    const Triangle3<Real>& triangle, const Vec<3,Real>& axis,
    Configuration& cfg)
{
    // Find projections of vertices onto potential separating axis.
    Real d0 = axis * triangle.V[0];
    Real d1 = axis * triangle.V[1];
    Real d2 = axis * triangle.V[2];

    // Explicit sort of vertices to construct a Configuration object.
    if (d0 <= d1)
    {
        if (d1 <= d2) // D0 <= D1 <= D2
        {
            if (d0 != d1)
            {
                if (d1 != d2)
                {
                    cfg.mMap = M111;
                }
                else
                {
                    cfg.mMap = M12;
                }
            }
            else // ( D0 == D1 )
            {
                if (d1 != d2)
                {
                    cfg.mMap = M21;
                }
                else
                {
                    cfg.mMap = M3;
                }
            }
            cfg.mIndex[0] = 0;
            cfg.mIndex[1] = 1;
            cfg.mIndex[2] = 2;
            cfg.mMin = d0;
            cfg.mMax = d2;
        }
        else if (d0 <= d2) // D0 <= D2 < D1
        {
            if (d0 != d2)
            {
                cfg.mMap = M111;
                cfg.mIndex[0] = 0;
                cfg.mIndex[1] = 2;
                cfg.mIndex[2] = 1;
            }
            else
            {
                cfg.mMap = M21;
                cfg.mIndex[0] = 2;
                cfg.mIndex[1] = 0;
                cfg.mIndex[2] = 1;
            }
            cfg.mMin = d0;
            cfg.mMax = d1;
        }
        else // D2 < D0 <= D1
        {
            if (d0 != d1)
            {
                cfg.mMap = M111;
            }
            else
            {
                cfg.mMap = M12;
            }

            cfg.mIndex[0] = 2;
            cfg.mIndex[1] = 0;
            cfg.mIndex[2] = 1;
            cfg.mMin = d2;
            cfg.mMax = d1;
        }
    }
    else if (d2 <= d1) // D2 <= D1 < D0
    {
        if (d2 != d1)
        {
            cfg.mMap = M111;
            cfg.mIndex[0] = 2;
            cfg.mIndex[1] = 1;
            cfg.mIndex[2] = 0;
        }
        else
        {
            cfg.mMap = M21;
            cfg.mIndex[0] = 1;
            cfg.mIndex[1] = 2;
            cfg.mIndex[2] = 0;

        }
        cfg.mMin = d2;
        cfg.mMax = d0;
    }
    else if (d2 <= d0) // D1 < D2 <= D0
    {
        if (d2 != d0)
        {
            cfg.mMap = M111;
        }
        else
        {
            cfg.mMap = M12;
        }

        cfg.mIndex[0] = 1;
        cfg.mIndex[1] = 2;
        cfg.mIndex[2] = 0;
        cfg.mMin = d1;
        cfg.mMax = d0;
    }
    else // D1 < D0 < D2
    {
        cfg.mMap = M111;
        cfg.mIndex[0] = 1;
        cfg.mIndex[1] = 0;
        cfg.mIndex[2] = 2;
        cfg.mMin = d1;
        cfg.mMax = d2;
    }
}

//----------------------------------------------------------------------------
template <typename Real>
bool Triangle3<Real>::FindOverlap (Real tmax, Real speed,
    const Configuration& UC, const Configuration& VC, ContactSide& side,
    Configuration& TUC, Configuration& TVC, Real& tfirst, Real& tlast)
{
    // Constant velocity separating axis test.  UC and VC are the new
    // potential configurations, and TUC and TVC are the best known
    // configurations.

    Real t;

    if (VC.mMax < UC.mMin) // V on left of U
    {
        if (speed <= (Real)0) // V moving away from U
        {
            return false;
        }

        // Find first time of contact on this axis.
        t = (UC.mMin - VC.mMax)/speed;

        // If this is the new maximum first time of contact, set side and
        // configuration.
        if (t > tfirst)
        {
            tfirst = t;
            side = CS_LEFT;
            TUC = UC;
            TVC = VC;
        }

        // Quick out: intersection after desired interval.
        if (tfirst > tmax)
        {
            return false;
        }

        // Find last time of contact on this axis.
        t = (UC.mMax - VC.mMin)/speed;
        if (t < tlast)
        {
            tlast = t;
        }

        // Quick out: intersection before desired interval.
        if (tfirst > tlast)
        {
            return false;
        }
    }
    else if (UC.mMax < VC.mMin)   // V on right of U
    {
        if (speed >= (Real)0) // V moving away from U
        {
            return false;
        }

        // Find first time of contact on this axis.
        t = (UC.mMax - VC.mMin)/speed;

        // If this is the new maximum first time of contact, set side and
        // configuration.
        if (t > tfirst)
        {
            tfirst = t;
            side = CS_RIGHT;
            TUC = UC;
            TVC = VC;
        }

        // Quick out: intersection after desired interval.
        if (tfirst > tmax)
        {
            return false;
        }

        // Find last time of contact on this axis.
        t = (UC.mMin - VC.mMax)/speed;
        if (t < tlast)
        {
            tlast = t;
        }

        // Quick out: intersection before desired interval.
        if (tfirst > tlast)
        {
            return false;
        }
    }
    else // V and U on overlapping interval
    {
        if (speed > (Real)0)
        {
            // Find last time of contact on this axis.
            t = (UC.mMax - VC.mMin)/speed;
            if (t < tlast)
            {
                tlast = t;
            }

            // Quick out: intersection before desired interval.
            if (tfirst > tlast)
            {
                return false;
            }
        }
        else if (speed < (Real)0)
        {
            // Find last time of contact on this axis.
            t = (UC.mMin - VC.mMax)/speed;
            if (t < tlast)
            {
                tlast = t;
            }

            // Quick out: intersection before desired interval.
            if (tfirst > tlast)
            {
                return false;
            }
        }
    }
    return true;
}

//----------------------------------------------------------------------------
template <typename Real>
bool Triangle3<Real>::FindOverlap (const Triangle3<Real>& tri2, const Vec<3,Real>& axis,
    Real tmax, const Vec<3,Real>& velocity, ContactSide& side,
    Configuration& tcfg0, Configuration& tcfg1, Real& tfirst,
    Real& tlast)
{
    Configuration cfg0, cfg1;
    ProjectOntoAxis(*this, axis, cfg0);
    ProjectOntoAxis(tri2, axis, cfg1);
    //TRU Real speed = velocity.Dot(axis);
    Real speed = velocity * axis;
    return FindOverlap(tmax, speed, cfg0, cfg1, side, tcfg0, tcfg1,
        tfirst, tlast);
}

//----------------------------------------------------------------------------
template <typename Real>
void Triangle3<Real>::TrianglePlaneRelations (
    const Triangle3<Real>& triangle, const Plane3<Real>& plane,
    Real distance[3], int sign[3], int& positive, int& negative, int& zero)
{
    // Compute the signed distances of triangle vertices to the plane.  Use
    // an epsilon-thick plane test.
    positive = 0;
    negative = 0;
    zero = 0;
    for (int i = 0; i < 3; ++i)
    {
        distance[i] = plane.DistanceTo(triangle.V[i]);
        if (distance[i] > MathUtils<Real>::ZERO_TOLERANCE)
        {
            sign[i] = 1;
            positive++;
        }
        else if (distance[i] < -MathUtils<Real>::ZERO_TOLERANCE)
        {
            sign[i] = -1;
            negative++;
        }
        else
        {
            distance[i] = (Real)0;
            sign[i] = 0;
            zero++;
        }
    }
}

//----------------------------------------------------------------------------
template <typename Real>
void Triangle3<Real>::GetInterval (
    const Triangle3<Real>& triangle, const Line3<Real>& line,
    const Real distance[3], const int sign[3], Real param[2])
{
    // Project triangle onto line.
    Real proj[3];
    int i;
    for (i = 0; i < 3; i++)
    {
        Vec<3,Real> diff = triangle.V[i] - line.Origin;
        proj[i] = line.Direction * diff;
    }

    // Compute transverse intersections of triangle edges with line.
    Real numer, denom;
    int i0, i1, i2;
    int quantity = 0;
    for (i0 = 2, i1 = 0; i1 < 3; i0 = i1++)
    {
        if (sign[i0]*sign[i1] < 0)
        {
            // assertion(quantity < 2, "Unexpected condition\n");
            numer = distance[i0]*proj[i1] - distance[i1]*proj[i0];
            denom = distance[i0] - distance[i1];
            param[quantity++] = numer/denom;
        }
    }

    // Check for grazing contact.
    if (quantity < 2)
    {
        for (i0 = 1, i1 = 2, i2 = 0; i2 < 3; i0 = i1, i1 = i2++)
        {
            if (sign[i2] == 0)
            {
                // assertion(quantity < 2, "Unexpected condition\n");
                param[quantity++] = proj[i2];
            }
        }
    }

    // Sort.
    // assertion(quantity == 1 || quantity == 2, "Unexpected condition\n");
    if (quantity == 2)
    {
        if (param[0] > param[1])
        {
            Real save = param[0];
            param[0] = param[1];
            param[1] = save;
        }
    }
    else
    {
        param[1] = param[0];
    }
}

//----------------------------------------------------------------------------
template <typename Real>
bool Triangle3<Real>::ContainsPoint(
    const Triangle3<Real>& triangle, const Plane3<Real>& plane,
    const Vec<3,Real>& point, Triangle3IntersectionResult<Real>& result)
{
    // Generate a coordinate system for the plane.  The incoming triangle has
    // vertices <V0,V1,V2>.  The incoming plane has unit-length normal N.
    // The incoming point is P.  V0 is chosen as the origin for the plane. The
    // coordinate axis directions are two unit-length vectors, U0 and U1,
    // constructed so that {U0,U1,N} is an orthonormal set.  Any point Q
    // in the plane may be written as Q = V0 + x0*U0 + x1*U1.  The coordinates
    // are computed as x0 = Dot(U0,Q-V0) and x1 = Dot(U1,Q-V0).
    Vec<3,Real> U0, U1;
    //TRU Vec<3,Real>::GenerateComplementBasis(U0, U1, plane.Normal);
    MathUtils<Real>::GenerateComplementBasis(U0, U1, plane.Normal);

    // Compute the planar coordinates for the points P, V1, and V2.  To
    // simplify matters, the origin is subtracted from the points, in which
    // case the planar coordinates are for P-V0, V1-V0, and V2-V0.
    Vec<3,Real> PmV0 = point - triangle.V[0];
    Vec<3,Real> V1mV0 = triangle.V[1] - triangle.V[0];
    Vec<3,Real> V2mV0 = triangle.V[2] - triangle.V[0];

    // The planar representation of P-V0.
    Vec<2,Real> ProjP((U0 * PmV0), (U1 * PmV0));

    // The planar representation of the triangle <V0-V0,V1-V0,V2-V0>.
    Vec<2,Real> ProjV[3] =
    {
        //TRU Vec<2,Real>(0,0,0),
        Vec<2,Real>((Real)0.0,(Real)0.0),
        Vec<2,Real>((U0 * V1mV0), (U1 * V1mV0)),
        Vec<2,Real>((U0 * V2mV0), (U1 * V2mV0))
    };

    // Test whether P-V0 is in the triangle <0,V1-V0,V2-V0>.
    if (Query2<Real>(3,ProjV).ToTriangle(ProjP,0,1,2) <= 0)
    {
        // Report the point of intersection to the caller.
        result.SetIntersectionType(IT_POINT);
        result.mQuantity = 1;
        result.mPoint[0] = point;
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
template <typename Real>
bool Triangle3<Real>::IntersectsSegment(
    const Plane3<Real>& plane, const Triangle3<Real>& triangle,
	const Vec<3, Real>& end0, const Vec<3, Real>& end1, Triangle3IntersectionResult<Real>& result)
{
    // Compute the 2D representations of the triangle vertices and the
    // segment endpoints relative to the plane of the triangle.  Then
    // compute the intersection in the 2D space.

    // Project the triangle and segment onto the coordinate plane most
    // aligned with the plane normal.
    int maxNormal = 0;
    Real fmax = MathUtils<Real>::FAbs(plane.Normal.x());
    Real absMax = MathUtils<Real>::FAbs(plane.Normal.y());
    if (absMax > fmax)
    {
        maxNormal = 1;
        fmax = absMax;
    }
    absMax = MathUtils<Real>::FAbs(plane.Normal.z());
    if (absMax > fmax)
    {
        maxNormal = 2;
    }

    Triangle2<Real> projTri;
    Vec<2,Real> projEnd0, projEnd1;
    int i;

    if (maxNormal == 0)
    {
        // Project onto yz-plane.
        for (i = 0; i < 3; ++i)
        {
            projTri.V[i].x() = triangle.V[i].y();
            projTri.V[i].y() = triangle.V[i].z();
            projEnd0.x() = end0.y();
            projEnd0.y() = end0.z();
            projEnd1.x() = end1.y();
            projEnd1.y() = end1.z();
        }
    }
    else if (maxNormal == 1)
    {
        // Project onto xz-plane.
        for (i = 0; i < 3; ++i)
        {
            projTri.V[i].x() = triangle.V[i].x();
            projTri.V[i].y() = triangle.V[i].z();
            projEnd0.x() = end0.x();
            projEnd0.y() = end0.z();
            projEnd1.x() = end1.x();
            projEnd1.y() = end1.z();
        }
    }
    else
    {
        // Project onto xy-plane.
        for (i = 0; i < 3; ++i)
        {
            projTri.V[i].x() = triangle.V[i].x();
            projTri.V[i].y() = triangle.V[i].y();
            projEnd0.x() = end0.x();
            projEnd0.y() = end0.y();
            projEnd1.x() = end1.x();
            projEnd1.y() = end1.y();
        }
    }

    Segment2<Real> projSeg(projEnd0, projEnd1);

    Triangle2IntersectionResult<Real> tri2Res;
    if (!projTri.Find(projSeg, tri2Res))
    {
        result.SetIntersectionType(IT_EMPTY);
        result.mQuantity = 0;
        return false;
    }

    Vec<2,Real> intr[2];
    if (tri2Res.GetIntersectionType() == IT_SEGMENT)
    {
        result.SetIntersectionType(IT_SEGMENT);
        result.mQuantity = 2;
        intr[0] = tri2Res.mPoint[0];
        intr[1] = tri2Res.mPoint[1];
    }
    else
    {
        //assertion(calc.GetIntersectionType() == IT_POINT,
        //    "Intersection must be a point\n");
        result.SetIntersectionType(IT_POINT);
        result.mQuantity = 1;
        intr[0] = tri2Res.mPoint[0];
    }

    // Unproject the segment of intersection.
    if (maxNormal == 0)
    {
        Real invNX = ((Real)1) / plane.Normal.x();
        for (i = 0; i < result.mQuantity; ++i)
        {
            result.mPoint[i].y() = intr[i].x();
            result.mPoint[i].z() = intr[i].y();
            result.mPoint[i].x() = invNX*(plane.Constant -
                plane.Normal.y() * result.mPoint[i].y() -
                plane.Normal.z() * result.mPoint[i].z());
        }
    }
    else if (maxNormal == 1)
    {
        Real invNY = ((Real)1)/plane.Normal.y();
        for (i = 0; i < result.mQuantity; ++i)
        {
            result.mPoint[i].x() = intr[i].x();
            result.mPoint[i].z() = intr[i].y();
            result.mPoint[i].y() = invNY*(plane.Constant -
                plane.Normal.x() * result.mPoint[i].x() -
                plane.Normal.z() * result.mPoint[i].z());
        }
    }
    else
    {
        Real invNZ = ((Real)1)/plane.Normal.z();
        for (i = 0; i < result.mQuantity; ++i)
        {
            result.mPoint[i].x() = intr[i].x();
            result.mPoint[i].y() = intr[i].y();
            result.mPoint[i].z() = invNZ*(plane.Constant -
                plane.Normal.x() * result.mPoint[i].x() -
                plane.Normal.y() * result.mPoint[i].y());
        }
    }

    return true;
}

//----------------------------------------------------------------------------
template <typename Real>
bool Triangle3<Real>::GetCoplanarIntersection (
    const Plane3<Real>& plane, const Triangle3<Real>& tri0,
	const Triangle3<Real>& tri1, Triangle3IntersectionResult<Real>& result)
{
    // Project triangles onto coordinate plane most aligned with plane
    // normal.
    int maxNormal = 0;
    Real fmax = MathUtils<Real>::FAbs(plane.Normal.x());
    Real absMax = MathUtils<Real>::FAbs(plane.Normal.y());
    if (absMax > fmax)
    {
        maxNormal = 1;
        fmax = absMax;
    }
    absMax = MathUtils<Real>::FAbs(plane.Normal.z());
    if (absMax > fmax)
    {
        maxNormal = 2;
    }

    Triangle2<Real> projTri0, projTri1;
    int i;

    if (maxNormal == 0)
    {
        // Project onto yz-plane.
        for (i = 0; i < 3; ++i)
        {
            projTri0.V[i].x() = tri0.V[i].y();
            projTri0.V[i].y() = tri0.V[i].z();
            projTri1.V[i].x() = tri1.V[i].y();
            projTri1.V[i].y() = tri1.V[i].z();
        }
    }
    else if (maxNormal == 1)
    {
        // Project onto xz-plane.
        for (i = 0; i < 3; ++i)
        {
            projTri0.V[i].x() = tri0.V[i].x();
            projTri0.V[i].y() = tri0.V[i].z();
            projTri1.V[i].x() = tri1.V[i].x();
            projTri1.V[i].y() = tri1.V[i].z();
        }
    }
    else
    {
        // Project onto xy-plane.
        for (i = 0; i < 3; ++i)
        {
            projTri0.V[i].x() = tri0.V[i].x();
            projTri0.V[i].y() = tri0.V[i].y();
            projTri1.V[i].x() = tri1.V[i].x();
            projTri1.V[i].y() = tri1.V[i].y();
        }
    }

    // 2D triangle intersection routines require counterclockwise ordering.
    Vec<2,Real> save;
    Vec<2,Real> edge0 = projTri0.V[1] - projTri0.V[0];
    Vec<2,Real> edge1 = projTri0.V[2] - projTri0.V[0];
    if ((edge0[0]*edge1[1] - edge0[1]*edge1[0]) < (Real)0)
    {
        // Triangle is clockwise, reorder it.
        save = projTri0.V[1];
        projTri0.V[1] = projTri0.V[2];
        projTri0.V[2] = save;
    }

    edge0 = projTri1.V[1] - projTri1.V[0];
    edge1 = projTri1.V[2] - projTri1.V[0];
    //TRU if (edge0.DotPerp(edge1) < (Real)0)
    if ((edge0[0]*edge1[1] - edge0[1]*edge1[0]) < (Real)0)
    {
        // Triangle is clockwise, reorder it.
        save = projTri1.V[1];
        projTri1.V[1] = projTri1.V[2];
        projTri1.V[2] = save;
    }

    Triangle2IntersectionResult<Real> tri2Res;
    if (!projTri1.Find(projTri1, tri2Res))
    {
        return false;
    }

    // Map 2D intersections back to the 3D triangle space.
    int mQuantity = tri2Res.mQuantity;
    result.mQuantity = mQuantity;

    if (maxNormal == 0)
    {
        Real invNX = ((Real)1)/plane.Normal.x();
        for (i = 0; i < mQuantity; i++)
        {
            result.mPoint[i].y() = tri2Res.mPoint[i].x();
            result.mPoint[i].z() = tri2Res.mPoint[i].y();
            result.mPoint[i].x() = invNX * (plane.Constant -
                plane.Normal.y() * result.mPoint[i].y() -
                plane.Normal.z() * result.mPoint[i].z());
        }
    }
    else if (maxNormal == 1)
    {
        Real invNY = ((Real)1)/plane.Normal.y();
        for (i = 0; i < mQuantity; i++)
        {
            result.mPoint[i].x() = tri2Res.mPoint[i].x();
            result.mPoint[i].z() = tri2Res.mPoint[i].y();
            result.mPoint[i].y() = invNY*(plane.Constant -
                plane.Normal.x() * result.mPoint[i].x() -
                plane.Normal.z() * result.mPoint[i].z());
        }
    }
    else
    {
        Real invNZ = ((Real)1)/plane.Normal.z();
        for (i = 0; i < mQuantity; i++)
        {
            result.mPoint[i].x() = tri2Res.mPoint[i].x();
            result.mPoint[i].y() = tri2Res.mPoint[i].y();
            result.mPoint[i].z() = invNZ*(plane.Constant -
                plane.Normal.x() * result.mPoint[i].x() -
                plane.Normal.y() * result.mPoint[i].y());
        }
    }

    result.SetIntersectionType(IT_PLANE);
    return true;
}

template <typename Real>
bool Triangle3<Real>::TestOverlap (Real tmax, Real speed,
    Real umin, Real umax, Real vmin, Real vmax, Real& tfirst, Real& tlast)
{
    // Constant velocity separating axis test.
    Real t;

    if (vmax < umin) // V on left of U
    {
        if (speed <= (Real)0) // V moving away from U
        {
            return false;
        }

        // Find first time of contact on this axis.
        t = (umin - vmax)/speed;
        if (t > tfirst)
        {
            tfirst = t;
        }

        // Quick out: intersection after desired time interval.
        if (tfirst > tmax)
        {
            return false;
        }

        // Find last time of contact on this axis.
        t = (umax - vmin)/speed;
        if (t < tlast)
        {
            tlast = t;
        }

        // Quick out: intersection before desired time interval.
        if (tfirst > tlast)
        {
            return false;
        }
    }
    else if (umax < vmin)   // V on right of U
    {
        if (speed >= (Real)0) // V moving away from U
        {
            return false;
        }

        // Find first time of contact on this axis.
        t = (umax - vmin)/speed;
        if (t > tfirst)
        {
            tfirst = t;
        }

        // Quick out: intersection after desired time interval.
        if (tfirst > tmax)
        {
            return false;
        }

        // Find last time of contact on this axis.
        t = (umin - vmax)/speed;
        if (t < tlast)
        {
            tlast = t;
        }

        // Quick out: intersection before desired time interval.
        if (tfirst > tlast)
        {
            return false;
        }

    }
    else // V and U on overlapping interval
    {
        if (speed > (Real)0)
        {
            // Find last time of contact on this axis.
            t = (umax - vmin)/speed;
            if (t < tlast)
            {
                tlast = t;
            }

            // Quick out: intersection before desired interval.
            if (tfirst > tlast)
            {
                return false;
            }
        }
        else if (speed < (Real)0)
        {
            // Find last time of contact on this axis.
            t = (umin - vmax)/speed;
            if (t < tlast)
            {
                tlast = t;
            }

            // Quick out: intersection before desired interval.
            if (tfirst > tlast)
            {
                return false;
            }
        }
    }
    return true;
}

//----------------------------------------------------------------------------
template <typename Real>
bool Triangle3<Real>::TestOverlap (const Intersectable<Real, Vec<3,Real> >& intersectable, const Vec<3,Real>& axis,
    Real tmax, const Vec<3,Real>& velocity, Real& tfirst, Real& tlast)
{
    const Triangle3<Real>* mTriangle1 = dynamic_cast<const Triangle3<Real>*>(&intersectable);

    if (!mTriangle1)
        return false;

    Real min0, max0, min1, max1;
    ProjectOntoAxis(*this, axis, min0, max0);
    ProjectOntoAxis(*mTriangle1, axis, min1, max1);
    Real speed = velocity * axis;
    return TestOverlap(tmax, speed, min0, max0, min1, max1, tfirst, tlast);
}

//----------------------------------------------------------------------------
template <typename Real>
void Triangle3<Real>::FindContactSet (
    const Triangle3<Real>& tri0, const Triangle3<Real>& tri1,
	ContactSide& side, Configuration& cfg0, Configuration& cfg1, Triangle3IntersectionResult<Real>& result)
{
    if (side == CS_RIGHT) // tri1 to the right of tri0
    {
        if (cfg0.mMap == M21 || cfg0.mMap == M111)
        {
            // tri0 touching tri1 at a single point
            result.SetIntersectionType(IT_POINT);
            result.mQuantity = 1;
            result.mPoint[0] = tri0.V[cfg0.mIndex[2]];
        }
        else if (cfg1.mMap == M12 || cfg1.mMap == M111)
        {
            // tri1 touching tri0 at a single point
            result.SetIntersectionType(IT_POINT);
            result.mQuantity = 1;
            result.mPoint[0] = tri1.V[cfg1.mIndex[0]];
        }
        else if (cfg0.mMap == M12)
        {
            if (cfg1.mMap == M21)
            {
                // edge0-edge1 intersection
                GetEdgeEdgeIntersection(
                    tri0.V[cfg0.mIndex[1]], tri0.V[cfg0.mIndex[2]],
                    tri1.V[cfg1.mIndex[0]], tri1.V[cfg1.mIndex[1]],
                        result);
            }
            else // cfg1.mMap == m3
            {
                // uedge-vface intersection
                GetEdgeFaceIntersection(
                    tri0.V[cfg0.mIndex[1]], tri0.V[cfg0.mIndex[2]],
                    tri1, result);
            }
        }
        else // cfg0.mMap == M3
        {
            if (cfg1.mMap == M21)
            {
                // face0-edge1 intersection
                GetEdgeFaceIntersection(
                    tri1.V[cfg1.mIndex[0]], tri1.V[cfg1.mIndex[1]],
                    tri0, result);
            }
            else // cfg1.mMap == M3
            {
                // face0-face1 intersection
                Plane3<Real> plane0(tri0.V[0], tri0.V[1], tri0.V[2]);
                GetCoplanarIntersection(plane0, tri0, tri1, result);
            }
        }
    }
    else if (side == CS_LEFT) // tri1 to the left of tri0
    {
        if (cfg1.mMap == M21 || cfg1.mMap == M111)
        {
            // tri1 touching tri0 at a single point
            result.SetIntersectionType(IT_POINT);
            result.mQuantity = 1;
            result.mPoint[0] = tri1.V[cfg1.mIndex[2]];
        }
        else if (cfg0.mMap == M12 || cfg0.mMap == M111)
        {
            // tri0 touching tri1 at a single point
            result.SetIntersectionType(IT_POINT);
            result.mQuantity = 1;
            result.mPoint[0] = tri0.V[cfg0.mIndex[0]];
        }
        else if (cfg1.mMap == M12)
        {
            if (cfg0.mMap == M21)
            {
                // edge0-edge1 intersection
                GetEdgeEdgeIntersection(
                    tri0.V[cfg0.mIndex[0]], tri0.V[cfg0.mIndex[1]],
                    tri1.V[cfg1.mIndex[1]], tri1.V[cfg1.mIndex[2]], result);
            }
            else // cfg0.mMap == M3
            {
                // edge1-face0 intersection
                GetEdgeFaceIntersection(
                    tri1.V[cfg1.mIndex[1]], tri1.V[cfg1.mIndex[2]],
                    tri0, result);
            }
        }
        else // cfg1.mMap == M3
        {
            if (cfg0.mMap == M21)
            {
                // edge0-face1 intersection
                GetEdgeFaceIntersection(
                    tri0.V[cfg0.mIndex[0]], tri0.V[cfg0.mIndex[1]],
                    tri1, result);
            }
            else // cfg0.mMap == M
            {
                // face0-face1 intersection
                Plane3<Real> plane0(tri0.V[0], tri0.V[1], tri0.V[2]);
                //TRU GetCoplanarIntersection(plane0, tri0, tri1);
                GetCoplanarIntersection(plane0, tri0, tri1, result);
            }
        }
    }
    else // side == CS_NONE
    {
        // Triangles are already intersecting tranversely.
        //IntrTriangle3Triangle3<Real> calc(tri0, tri1);
        Triangle3IntersectionResult<Real> tri3Res;
        bool int_result = const_cast<Triangle3<Real>&>(tri0).Find(const_cast<Triangle3<Real>&>(tri1), tri3Res);

        // assertion(result, "Intersection must exist\n");
        // WM5_UNUSED(result);
        result.mQuantity = tri3Res.mQuantity;
        result.SetIntersectionType(tri3Res.GetIntersectionType());
        for (int i = 0; i < result.mQuantity; ++i)
        {
            result.mPoint[i] = tri3Res.mPoint[i];
        }
    }
}

//----------------------------------------------------------------------------
template <typename Real>
void Triangle3<Real>::GetEdgeEdgeIntersection (
    const Vec<3,Real>& U0, const Vec<3,Real>& U1,
	const Vec<3, Real>& V0, const Vec<3, Real>& V1, Triangle3IntersectionResult<Real>& result)
{
    // Compute a normal to the plane of the two edges.
    Vec<3,Real> edge0 = U1 - U0;
    Vec<3,Real> edge1 = V1 - V0;
    Vec<3,Real> normal = edge0.cross(edge1);

    // Solve U0 + s*(U1 - U0) = V0 + t*(V1 - V0).  We know the edges
    // intersect, so s in [0,1] and t in [0,1].  Thus, just solve for s.
    // Note that s*E0 = D + t*E1, where D = V0 - U0. So s*N = s*E0xE1 = DxE1
    // and s = N*DxE1/N*N.
    Vec<3,Real> delta = V0 - U0;
    Real s = normal * (delta.cross(edge1)) / normal.norm2();
    if (s < (Real)0)
    {
        //assertion(s >= -MathUtils<Real>::ZERO_TOLERANCE,
        //    "Unexpected s value.\n");
        s = (Real)0;
    }
    else if (s > (Real)1)
    {
        //assertion(s <= (Real)1 + MathUtils<Real>::ZERO_TOLERANCE,
        //    "Unexpected s value.\n");
        s = (Real)1;
    }

    //TRU result.intersectionType = IT_POINT;
    result.SetIntersectionType(IT_POINT);
    result.mQuantity = 1;
    result.mPoint[0] = U0 + edge0 * s;

    // TODO:  What if the edges are parallel?
}

//----------------------------------------------------------------------------
template <typename Real>
void Triangle3<Real>::GetEdgeFaceIntersection (
    const Vec<3,Real>& U0, const Vec<3,Real>& U1,
	const Triangle3<Real>& tri, Triangle3IntersectionResult<Real>& result)
{
    // Compute a plane of the triangle.
    Vec<3,Real> point = tri.V[0];
    Vec<3,Real> edge0 = tri.V[1] - point;
    Vec<3,Real> edge1 = tri.V[2] - point;
    Vec<3,Real> normal = edge0.cross(edge1);
    normal.normalize();
    Vec<3,Real> dir0, dir1;
    MathUtils<Real>::GenerateComplementBasis(dir0, dir1, normal);

    // Project the edge endpoints onto the plane.
    Vec<2,Real> projU0, projU1;
    Vec<3,Real> diff;
    diff = U0 - point;
    projU0[0] = dir0 * diff;
    projU0[1] = dir1 * diff;
    diff = U1 - point;
    projU1[0] = dir0 * diff;
    projU1[1] = dir1 * diff;
    Segment2<Real> projSeg(projU0, projU1);

    // Compute the plane coordinates of the triangle.
    Triangle2<Real> projTri;
    projTri.V[0] = Vec<2,Real>((Real)0.0,(Real)0.0);
    projTri.V[1] = Vec<2,Real>((dir0 * edge0), (dir1 * edge0));
    projTri.V[2] = Vec<2,Real>((dir0 * edge1), (dir1 * edge1));

    // Compute the intersection.
    Segment2IntersectionResult<Real> seg2Res;
    if (projSeg.Find(projTri, seg2Res))
    {
        result.mQuantity = seg2Res.mQuantity;
        for (int i = 0; i < result.mQuantity; ++i)
        {
            Vec<2,Real> proj = seg2Res.mPoint[i];
            result.mPoint[i] = point + proj[0] * dir0 + proj[1] * dir1;
        }
    }
    else
    {
        // There must be an intersection.  Most likely numerical
        // round-off errors have led to a failure to find it.  Use a slower
        // 3D distance calculator for robustness.
        Segment3<Real> seg(U0, U1);

        Segment3Triangle3DistanceResult seg3Res;
        // We do not need the distance, but we do need the side effect
        // of locating the closest points.
        const DistanceComputable<Real, Vec<3,Real> >& tmp = dynamic_cast<const DistanceComputable<Real, Vec<3,Real> >&>(tri);
        if (&tmp)
        {
            Real distance = seg.GetDistance(tmp, seg3Res);
        }

        Real parameter = seg3Res.mSegmentParameter;
        result.mQuantity = 1;
        result.mPoint[0] = seg.Center + parameter * seg.Direction;
    }

    //TRU result.intersectionType = (result.mQuantity == 2 ? IT_SEGMENT : IT_POINT);
    result.SetIntersectionType((result.mQuantity == 2 ? IT_SEGMENT : IT_POINT));
}

template <typename Real>
Real Triangle3<Real>::GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result)
{
    std::cout << "WARNING! Method 'Real Triangle3<Real>::GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result)' was called, even though it's not properly implemented!" << std::endl;
    return (Real)0;
}

template <typename Real>
Real Triangle3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result)
{
    std::cout << "WARNING! Method 'Real Triangle3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result)' was called, even though it's not properly implemented!" << std::endl;
    return (Real)0;
}

template <typename Real>
Real Triangle3<Real>::GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3, Real>& velocity0, const Vec<3, Real>& velocity1, DistanceResult& result)
{
    std::cout << "WARNING! Method 'Real Triangle3<Real>::GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3, Real>& velocity0, const Vec<3, Real>& velocity1, DistanceResult& result)' was called, even though it's not properly implemented!" << std::endl;
    return (Real)0;
}

template <typename Real>
Real Triangle3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3, Real>& velocity0, const Vec<3, Real>& velocity1, DistanceResult& result)
{
    std::cout << "WARNING! Method 'Real Triangle3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3, Real>& velocity0, const Vec<3, Real>& velocity1, DistanceResult& result)' was called, even though it's not properly implemented!" << std::endl;
    return (Real)0;
}
