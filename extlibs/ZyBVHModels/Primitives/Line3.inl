#include "Line3.h"

#include "Segment3.h"
#include "Triangle3.h"
#include "Rectangle3.h"

#include "Math/MathUtils.h"

using namespace BVHModels;

template <typename Real>
Line3<Real>::Line3 ()
{
}

//----------------------------------------------------------------------------
template <typename Real>
Line3<Real>::~Line3 ()
{
}

//----------------------------------------------------------------------------
template <typename Real>
Line3<Real>::Line3 (const Vec<3,Real>& origin,
    const Vec<3,Real>& direction)
    :
    Origin(origin),
    Direction(direction)
{
}

//----------------------------------------------------------------------------
template <typename Real>
Real Line3<Real>::GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, DistanceResult &result)
{
    return MathUtils<Real>::Sqrt(GetSquaredDistance(other, result));
}

//----------------------------------------------------------------------------
template <typename Real>
Real Line3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, DistanceResult &result)
{
    const Triangle3<Real>* mTriangle = dynamic_cast<const Triangle3<Real>*>(&other);
    Line3Triangle3DistanceResult* lineRes = dynamic_cast<Line3Triangle3DistanceResult*>(&result);
    if (mTriangle && lineRes)
    {
        // Test if line intersects triangle.  If so, the squared distance is zero.
        Vec<3,Real> edge0 = mTriangle->V[1] - mTriangle->V[0];
        Vec<3,Real> edge1 = mTriangle->V[2] - mTriangle->V[0];
        Vec<3,Real> normal = edge0.cross(edge1);
        Real NdD = normal * this->Direction;
        if (MathUtils<Real>::FAbs(NdD) > MathUtils<Real>::ZERO_TOLERANCE)
        {
            // The line and triangle are not parallel, so the line intersects
            // the plane of the triangle.
            Vec<3,Real> diff = this->Origin - mTriangle->V[0];
            Vec<3,Real> U, V;
            MathUtils<Real>::GenerateComplementBasis(U, V, this->Direction);
            Real UdE0 = U * edge0;
            Real UdE1 = U * edge1;
            Real UdDiff = U * diff;
            Real VdE0 = V * edge0;
            Real VdE1 = V * edge1;
            Real VdDiff = V * diff;
            Real invDet = ((Real)1) / (UdE0 * VdE1 - UdE1 * VdE0);

            // Barycentric coordinates for the point of intersection.
            Real b1 = (VdE1*UdDiff - UdE1*VdDiff)*invDet;
            Real b2 = (UdE0*VdDiff - VdE0*UdDiff)*invDet;
            Real b0 = (Real)1 - b1 - b2;

            if (b0 >= (Real)0 && b1 >= (Real)0 && b2 >= (Real)0)
            {
                // Line parameter for the point of intersection.
                Real DdE0 = this->Direction * edge0;
                Real DdE1 = this->Direction * edge1;
                Real DdDiff = this->Direction * diff;
                lineRes->mLineParameter = b1 * DdE0 + b2 * DdE1 - DdDiff;

                // Barycentric coordinates for the point of intersection.
                lineRes->mTriangleBary[0] = b0;
                lineRes->mTriangleBary[1] = b1;
                lineRes->mTriangleBary[2] = b2;

                // The intersection point is inside or on the triangle.
                lineRes->mClosestPoint0 = this->Origin + lineRes->mLineParameter * this->Direction;

                lineRes->mClosestPoint1 = mTriangle->V[0] + b1 * edge0 + b2 * edge1;

                return (Real)0;
            }
        }

        // Either (1) the line is not parallel to the triangle and the point of
        // intersection of the line and the plane of the triangle is outside the
        // triangle or (2) the line and triangle are parallel.  Regardless, the
        // closest point on the triangle is on an edge of the triangle.  Compare
        // the line to all three edges of the triangle.
        Real sqrDist = MathUtils<Real>::MAX_REAL;
        for (int i0 = 2, i1 = 0; i1 < 3; i0 = i1++)
        {
            //TRU /// TODO: IS THAT CORRECT?
            //TRU Vec<3,Real> normalizedDirection = segment.Direction;
            //TRU normalizedDirection.normalize();
            //TRU segment.Extent = normalizedDirection.norm() * ((Real)0.5);
            //TRU segment.ComputeEndPoints();

            Segment3<Real> segment;
            segment.Center = ((Real)0.5) * (mTriangle->V[i0] +
                mTriangle->V[i1]);
            segment.Direction = mTriangle->V[i1] - mTriangle->V[i0];
            segment.Extent = ((Real)0.5)*segment.Direction.norm();
            segment.Direction.normalize();
            segment.ComputeEndPoints();

            //DistLine3Segment3<Real> queryLS(*this, segment);
            Line3Segment3DistanceResult seg3Res;
            Real sqrDistTmp = this->GetSquaredDistance(segment, seg3Res);
            if (sqrDistTmp < sqrDist)
            {
                lineRes->mClosestPoint0 = seg3Res.GetClosestPoint0();
                lineRes->mClosestPoint1 = seg3Res.GetClosestPoint1();
                sqrDist = sqrDistTmp;

                lineRes->mLineParameter = seg3Res.mLineParameter;
                Real ratio = seg3Res.mSegmentParameter / segment.Extent;
                lineRes->mTriangleBary[i0] = ((Real)0.5)*((Real)1 - ratio);
                lineRes->mTriangleBary[i1] = (Real)1 - lineRes->mTriangleBary[i0];
                lineRes->mTriangleBary[3-i0-i1] = (Real)0;
            }
        }

        return sqrDist;
    }

    const Segment3<Real>* mSegment = dynamic_cast<const Segment3<Real>*>(&other);
    Line3Segment3DistanceResult* segRes = dynamic_cast<Line3Segment3DistanceResult*>(&result);
    if (mSegment && segRes)
    {
        Vec<3,Real> diff = this->Origin - mSegment->Center;
        Real a01 = -(this->Direction * mSegment->Direction);
        Real b0 = diff * this->Direction;
        Real c = diff.norm2();
        Real det = MathUtils<Real>::FAbs((Real)1 - a01*a01);
        Real b1, s0, s1, sqrDist, extDet;

        if (det >= MathUtils<Real>::ZERO_TOLERANCE)
        {
            // The line and segment are not parallel.
            b1 = -diff * mSegment->Direction;
            s1 = a01 * b0 - b1;
            extDet = mSegment->Extent*det;

            if (s1 >= -extDet)
            {
                if (s1 <= extDet)
                {
                    // Two interior points are closest, one on the line and one
                    // on the segment.
                    Real invDet = ((Real)1) / det;
                    s0 = (a01 * b1 - b0) * invDet;
                    s1 *= invDet;
                    sqrDist = s0 * (s0 + a01 * s1 + ((Real)2) * b0) +
                        s1 * (a01 * s0 + s1 + ((Real)2) * b1) + c;
                }
                else
                {
                    // The endpoint e1 of the segment and an interior point of
                    // the line are closest.
                    s1 = mSegment->Extent;
                    s0 = -(a01*s1 + b0);
                    sqrDist = -s0*s0 + s1*(s1 + ((Real)2)*b1) + c;
                }
            }
            else
            {
                // The end point e0 of the segment and an interior point of the
                // line are closest.
                s1 = -mSegment->Extent;
                s0 = -(a01*s1 + b0);
                sqrDist = -s0*s0 + s1*(s1 + ((Real)2)*b1) + c;
            }
        }
        else
        {
            // The line and segment are parallel.  Choose the closest pair so that
            // one point is at segment center.
            s1 = (Real)0;
            s0 = -b0;
            sqrDist = b0*s0 + c;
        }

        segRes->mClosestPoint0 = this->Origin + s0 * this->Direction;
        segRes->mClosestPoint1 = mSegment->Center + s1 * mSegment->Direction;
        segRes->mLineParameter = s0;
        segRes->mSegmentParameter = s1;

        // Account for numerical round-off errors.
        if (sqrDist < (Real)0)
        {
            sqrDist = (Real)0;
        }
        return sqrDist;
    }

    return MathUtils<Real>::MAX_REAL;
}

//----------------------------------------------------------------------------
template <typename Real>
Real Line3<Real>::GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result)
{
    const Triangle3<Real>* mTriangle = dynamic_cast<const Triangle3<Real>*>(&other);
    Line3Triangle3DistanceResult* lineRes = dynamic_cast<Line3Triangle3DistanceResult*>(&result);
    if (mTriangle && lineRes)
    {
        Vec<3, Real> movedOrigin = this->Origin + t*velocity0;
        Vec<3, Real> movedV0 = mTriangle->V[0] + t*velocity1;
        Vec<3, Real> movedV1 = mTriangle->V[1] + t*velocity1;
        Vec<3, Real> movedV2 = mTriangle->V[2] + t*velocity1;
        Line3<Real> movedLine(movedOrigin, this->Direction);
        Triangle3<Real> movedTriangle(movedV0, movedV1, movedV2);

        return movedLine.GetDistance(movedTriangle,*lineRes);
        //return DistLine3Triangle3<Real>(movedLine, movedTriangle).Get();
    }

    const Segment3<Real>* mSegment = dynamic_cast<const Segment3<Real>*>(&other);
    Line3Segment3DistanceResult* segRes = dynamic_cast<Line3Segment3DistanceResult*>(&result);
    if (mSegment && segRes)
    {
        Vec<3, Real> movedOrigin = this->Origin + t*velocity0;
        Vec<3, Real> movedCenter = mSegment->Center + t*velocity1;
        Line3<Real> movedLine(movedOrigin, this->Direction);
        Segment3<Real> movedSegment(movedCenter, mSegment->Direction,
            mSegment->Extent);

        return movedLine.GetDistance(movedSegment,*lineRes);
        //return DistLine3Segment3<Real>(movedLine, movedSegment).Get();
    }

    const Rectangle3<Real>* mRectangle = dynamic_cast<const Rectangle3<Real>*>(&other);
    Line3Rectangle3DistanceResult* rectRes = dynamic_cast<Line3Rectangle3DistanceResult*>(&result);
    if (mRectangle && rectRes)
    {
        // Test if line intersects rectangle.  If so, the squared distance is
        // zero.
        Vec<3,Real> N = mRectangle->Axis[0].cross(mRectangle->Axis[1]);
        Real NdD = N * this->Direction;
        if (MathUtils<Real>::FAbs(NdD) > MathUtils<Real>::ZERO_TOLERANCE)
        {
            // The line and rectangle are not parallel, so the line intersects
            // the plane of the rectangle.
            Vec<3,Real> diff = this->Origin - mRectangle->Center;
            Vec<3,Real> U, V;
            MathUtils<Real>::GenerateComplementBasis(U, V, this->Direction);
            Real UdD0 = U * (mRectangle->Axis[0]);
            Real UdD1 = U * (mRectangle->Axis[1]);
            Real UdPmC = U * diff;
            Real VdD0 = V * (mRectangle->Axis[0]);
            Real VdD1 = V * (mRectangle->Axis[1]);
            Real VdPmC = V * diff;
            Real invDet = ((Real)1)/(UdD0 * VdD1 - UdD1 * VdD0);

            // Rectangle coordinates for the point of intersection.
            Real s0 = (VdD1 * UdPmC - UdD1 * VdPmC) * invDet;
            Real s1 = (UdD0 * VdPmC - VdD0 * UdPmC) * invDet;

            if (MathUtils<Real>::FAbs(s0) <= mRectangle->Extent[0]
                &&  MathUtils<Real>::FAbs(s1) <= mRectangle->Extent[1])
            {
                // Line parameter for the point of intersection.
                Real DdD0 = this->Direction * (mRectangle->Axis[0]);
                Real DdD1 = this->Direction * (mRectangle->Axis[1]);
                Real DdDiff = this->Direction * diff;
                rectRes->mLineParameter = s0 * DdD0 + s1 * DdD1 - DdDiff;

                // Rectangle coordinates for the point of intersection.
                rectRes->mRectCoord[0] = s0;
                rectRes->mRectCoord[1] = s1;

                // The intersection point is inside or on the rectangle.
                rectRes->mClosestPoint0 = this->Origin +
                    rectRes->mLineParameter * this->Direction;

                rectRes->mClosestPoint1 = mRectangle->Center +
                    s0 * mRectangle->Axis[0] + s1 * mRectangle->Axis[1];

                return (Real)0;
            }
        }

        // Either (1) the line is not parallel to the rectangle and the point of
        // intersection of the line and the plane of the rectangle is outside the
        // rectangle or (2) the line and rectangle are parallel.  Regardless, the
        // closest point on the rectangle is on an edge of the rectangle.  Compare
        // the line to all four edges of the rectangle.
        Real sqrDist = MathUtils<Real>::MAX_REAL;
        Vec<3,Real> scaledDir[2] =
        {
            mRectangle->Extent[0] * mRectangle->Axis[0],
            mRectangle->Extent[1] * mRectangle->Axis[1]
        };
        for (int i1 = 0; i1 < 2; ++i1)
        {
            for (int i0 = 0; i0 < 2; ++i0)
            {
                Segment3<Real> segment;
                segment.Center = mRectangle->Center +
                    ((Real)(2*i0-1)) * scaledDir[i1];
                segment.Direction = mRectangle->Axis[1-i1];
                segment.Extent = mRectangle->Extent[1-i1];
                segment.ComputeEndPoints();

                Line3Segment3DistanceResult segLineRes;
                Real sqrDistTmp = segment.GetSquaredDistance(*this, segLineRes);
                if (sqrDistTmp < sqrDist)
                {
                    rectRes->mClosestPoint0 = segLineRes.GetClosestPoint0();
                    rectRes->mClosestPoint1 = segLineRes.GetClosestPoint1();
                    sqrDist = sqrDistTmp;

                    rectRes->mLineParameter = segLineRes.mLineParameter;
                    Real ratio = segLineRes.mSegmentParameter / segment.Extent;
                    rectRes->mRectCoord[0] = mRectangle->Extent[0] * ((1 - i1) * (2 * i0 - 1) +
                        i1 * ratio);
                    rectRes->mRectCoord[1] = mRectangle->Extent[1] * ((1 - i0)*(2 * i1 - 1) +
                        i0 * ratio);
                }
            }
        }

        return sqrDist;
    }

    return MathUtils<Real>::MAX_REAL;
}

//----------------------------------------------------------------------------
template <typename Real>
Real Line3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result)
{
    const Triangle3<Real>* mTriangle = dynamic_cast<const Triangle3<Real>*>(&other);
    Line3Triangle3DistanceResult* lineRes = dynamic_cast<Line3Triangle3DistanceResult*>(&result);
    if (mTriangle && lineRes)
    {
        Vec<3, Real> movedOrigin = this->Origin + t*velocity0;
        Vec<3, Real> movedV0 = mTriangle->V[0] + t*velocity1;
        Vec<3, Real> movedV1 = mTriangle->V[1] + t*velocity1;
        Vec<3, Real> movedV2 = mTriangle->V[2] + t*velocity1;
        Line3<Real> movedLine(movedOrigin, this->Direction);
        Triangle3<Real> movedTriangle(movedV0, movedV1, movedV2);

        return movedLine.GetSquaredDistance(movedTriangle,*lineRes);
        //return DistLine3Triangle3<Real>(movedLine, movedTriangle).GetSquared();
    }

    const Segment3<Real>* mSegment = dynamic_cast<const Segment3<Real>*>(&other);
    Line3Segment3DistanceResult* segRes = dynamic_cast<Line3Segment3DistanceResult*>(&result);
    if (mSegment && segRes)
    {
        Vec<3, Real> movedOrigin = this->Origin + t*velocity0;
        Vec<3, Real> movedCenter = mSegment->Center + t*velocity1;
        Line3<Real> movedLine(movedOrigin, this->Direction);
        Segment3<Real> movedSegment(movedCenter, mSegment->Direction,
            mSegment->Extent);

        return movedLine.GetSquaredDistance(movedSegment,*lineRes);
        //return DistLine3Segment3<Real>(movedLine, movedSegment).GetSquared();
    }

    return MathUtils<Real>::MAX_REAL;
}

template <typename Real>
bool Line3<Real>::IsIntersectionQuerySupported(const PrimitiveType &other)
{
    if (other == PT_CAPSULE3)
        return true;

    return false;
}

//----------------------------------------------------------------------------
template <typename Real>
bool Line3<Real>::Test(const Intersectable<Real, Vec<3,Real> >& other)
{
    if (!IsIntersectionQuerySupported(other.GetIntersectableType()))
        return false;

    if (other.GetIntersectableType() == PT_CAPSULE3)
    {
        Line3Segment3DistanceResult line_seg_dist_res;
        const Capsule3<Real>* capsule = dynamic_cast<const Capsule3<Real>*>(&other);
        Real distance = this->GetDistance(capsule->Segment, line_seg_dist_res);
        return distance <= capsule->Radius;
    }

    return false;
}
//----------------------------------------------------------------------------
template <typename Real>
bool Line3<Real>::Find(const Intersectable<Real, Vec<3,Real> >& other, IntersectionResult<Real>& result)
{
    if (!IsIntersectionQuerySupported(other.GetIntersectableType()))
        return false;

    if (other.GetIntersectableType() == PT_CAPSULE3)
    {
        const Capsule3<Real>* mCapsule = static_cast<const Capsule3<Real>*>(&other);

        Line3Capsule3IntersectionResult<Real>* capsuleRes = static_cast<Line3Capsule3IntersectionResult<Real>*>(&result);
        if (!capsuleRes)
            return false;

        Real t[2];
        capsuleRes->mQuantity = this->FindIntersectionWithCapsule3(this->Origin, this->Direction, *mCapsule, t);
        for (int i = 0; i < capsuleRes->mQuantity; ++i)
        {
            capsuleRes->mPoint[i] = this->Origin + t[i] * this->Direction;
        }

        if (capsuleRes->mQuantity == 2)
        {
            capsuleRes->SetIntersectionType(IT_SEGMENT);
        }

        else if (capsuleRes->mQuantity == 1)
        {
            capsuleRes->SetIntersectionType(IT_POINT);
        }
        else
        {
            capsuleRes->SetIntersectionType(IT_EMPTY);
        }

        this->mIntersectionType = capsuleRes->GetIntersectionType();
    }

    return this->mIntersectionType != IT_EMPTY;
}

//----------------------------------------------------------------------------
template <typename Real>
int Line3<Real>::FindIntersectionWithCapsule3(const Vec<3, Real>& origin,
    const Vec<3, Real>& dir, const Capsule3<Real>& capsule, Real t[2])
{
    // Create a coordinate system for the capsule.  In this system, the
    // capsule segment center C is the origin and the capsule axis direction
    // W is the z-axis.  U and V are the other coordinate axis directions.
    // If P = x*U+y*V+z*W, the cylinder containing the capsule wall is
    // x^2 + y^2 = r^2, where r is the capsule radius.  The finite cylinder
    // that makes up the capsule minus its hemispherical end caps has z-values
    // |z| <= e, where e is the extent of the capsule segment.  The top
    // hemisphere cap is x^2+y^2+(z-e)^2 = r^2 for z >= e, and the bottom
    // hemisphere cap is x^2+y^2+(z+e)^2 = r^2 for z <= -e.
    Vec<3, Real> U, V, W = capsule.Segment.Direction;
    MathUtils<Real>::GenerateComplementBasis(U, V, W);
    Real rSqr = capsule.Radius*capsule.Radius;
    Real extent = capsule.Segment.Extent;

    // Convert incoming line origin to capsule coordinates.
    Vec<3, Real> diff = origin - capsule.Segment.Center;
    Vec<3, Real> P(U * diff, V * diff, W * diff);

    // Get the z-value, in capsule coordinates, of the incoming line's
    // unit-length direction.
    Real dz = W * dir;
    if (MathUtils<Real>::FAbs(dz) >= (Real)1 - MathUtils<Real>::ZERO_TOLERANCE)
    {
        // The line is parallel to the capsule axis.  Determine whether the
        // line intersects the capsule hemispheres.
        Real radialSqrDist = rSqr - P.x() * P.x() - P.y() * P.y();
        if (radialSqrDist < (Real)0)
        {
            // Line outside the cylinder of the capsule, no intersection.
            return 0;
        }

        // line intersects the hemispherical caps
        Real zOffset = MathUtils<Real>::Sqrt(radialSqrDist) + extent;
        if (dz > (Real)0)
        {
            t[0] = -P.z() - zOffset;
            t[1] = -P.z() + zOffset;
        }
        else
        {
            t[0] = P.z() - zOffset;
            t[1] = P.z() + zOffset;
        }
        return 2;
    }

    // Convert incoming line unit-length direction to capsule coordinates.
    Vec<3, Real> D(U * dir, V * dir, dz);

    // Test intersection of line P+t*D with infinite cylinder x^2+y^2 = r^2.
    // This reduces to computing the roots of a quadratic equation.  If
    // P = (px,py,pz) and D = (dx,dy,dz), then the quadratic equation is
    //   (dx^2+dy^2)*t^2 + 2*(px*dx+py*dy)*t + (px^2+py^2-r^2) = 0
    Real a0 = P.x() * P.x() + P.y() * P.y() - rSqr;
    Real a1 = P.x() * D.x() + P.y() * D.y();
    Real a2 = D.x() * D.x() + D.y() * D.y();
    Real discr = a1*a1 - a0*a2;
    if (discr < (Real)0)
    {
        // Line does not intersect infinite cylinder.
        return 0;
    }

    Real root, inv, tValue, zValue;
    int quantity = 0;
    if (discr > MathUtils<Real>::ZERO_TOLERANCE)
    {
        // Line intersects infinite cylinder in two places.
        root = MathUtils<Real>::Sqrt(discr);
        inv = ((Real)1)/a2;
        tValue = (-a1 - root)*inv;
        zValue = P.z() + tValue*D.z();
        if (MathUtils<Real>::FAbs(zValue) <= extent)
        {
            t[quantity++] = tValue;
        }

        tValue = (-a1 + root)*inv;
        zValue = P.z() + tValue*D.z();
        if (MathUtils<Real>::FAbs(zValue) <= extent)
        {
            t[quantity++] = tValue;
        }

        if (quantity == 2)
        {
            // Line intersects capsule wall in two places.
            return 2;
        }
    }
    else
    {
        // Line is tangent to infinite cylinder.
        tValue = -a1/a2;
        zValue = P.z() + tValue*D.z();
        if (MathUtils<Real>::FAbs(zValue) <= extent)
        {
            t[0] = tValue;
            return 1;
        }
    }

    // Test intersection with bottom hemisphere.  The quadratic equation is
    //   t^2 + 2*(px*dx+py*dy+(pz+e)*dz)*t + (px^2+py^2+(pz+e)^2-r^2) = 0
    // Use the fact that currently a1 = px*dx+py*dy and a0 = px^2+py^2-r^2.
    // The leading coefficient is a2 = 1, so no need to include in the
    // construction.
    Real PZpE = P.z() + extent;
    a1 += PZpE*D.z();
    a0 += PZpE*PZpE;
    discr = a1*a1 - a0;
    if (discr > MathUtils<Real>::ZERO_TOLERANCE)
    {
        root = MathUtils<Real>::Sqrt(discr);
        tValue = -a1 - root;
        zValue = P.z() + tValue*D.z();
        if (zValue <= -extent)
        {
            t[quantity++] = tValue;
            if (quantity == 2)
            {
                if (t[0] > t[1])
                {
                    Real save = t[0];
                    t[0] = t[1];
                    t[1] = save;
                }
                return 2;
            }
        }

        tValue = -a1 + root;
        zValue = P.z() + tValue * D.z();
        if (zValue <= -extent)
        {
            t[quantity++] = tValue;
            if (quantity == 2)
            {
                if (t[0] > t[1])
                {
                    Real save = t[0];
                    t[0] = t[1];
                    t[1] = save;
                }
                return 2;
            }
        }
    }
    else if (MathUtils<Real>::FAbs(discr) <= MathUtils<Real>::ZERO_TOLERANCE)
    {
        tValue = -a1;
        zValue = P.z() + tValue * D.z();
        if (zValue <= -extent)
        {
            t[quantity++] = tValue;
            if (quantity == 2)
            {
                if (t[0] > t[1])
                {
                    Real save = t[0];
                    t[0] = t[1];
                    t[1] = save;
                }
                return 2;
            }
        }
    }

    // Test intersection with top hemisphere.  The quadratic equation is
    //   t^2 + 2*(px*dx+py*dy+(pz-e)*dz)*t + (px^2+py^2+(pz-e)^2-r^2) = 0
    // Use the fact that currently a1 = px*dx+py*dy+(pz+e)*dz and
    // a0 = px^2+py^2+(pz+e)^2-r^2.  The leading coefficient is a2 = 1, so
    // no need to include in the construction.
    a1 -= ((Real)2)*extent*D.z();
    a0 -= ((Real)4)*extent*P.z();
    discr = a1*a1 - a0;
    if (discr > MathUtils<Real>::ZERO_TOLERANCE)
    {
        root = MathUtils<Real>::Sqrt(discr);
        tValue = -a1 - root;
        zValue = P.z() + tValue*D.z();
        if (zValue >= extent)
        {
            t[quantity++] = tValue;
            if (quantity == 2)
            {
                if (t[0] > t[1])
                {
                    Real save = t[0];
                    t[0] = t[1];
                    t[1] = save;
                }
                return 2;
            }
        }

        tValue = -a1 + root;
        zValue = P.z() + tValue*D.z();
        if (zValue >= extent)
        {
            t[quantity++] = tValue;
            if (quantity == 2)
            {
                if (t[0] > t[1])
                {
                    Real save = t[0];
                    t[0] = t[1];
                    t[1] = save;
                }
                return 2;
            }
        }
    }
    else if (MathUtils<Real>::FAbs(discr) <= MathUtils<Real>::ZERO_TOLERANCE)
    {
        tValue = -a1;
        zValue = P.z() + tValue*D.z();
        if (zValue >= extent)
        {
            t[quantity++] = tValue;
            if (quantity == 2)
            {
                if (t[0] > t[1])
                {
                    Real save = t[0];
                    t[0] = t[1];
                    t[1] = save;
                }
                return 2;
            }
        }
    }

    return quantity;
}
