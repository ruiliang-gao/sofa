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
