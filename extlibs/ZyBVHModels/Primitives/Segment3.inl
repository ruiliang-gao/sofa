#include "Segment3.h"
#include "Line3.h"

#include "Point3.h"
#include "Triangle3.h"
#include "Rectangle3.h"

#include "Math/MathUtils.h"

using namespace BVHModels;

//----------------------------------------------------------------------------
template <typename Real>
Segment3<Real>::Segment3()
{
}

//----------------------------------------------------------------------------
template <typename Real>
Segment3<Real>::~Segment3()
{
}

//----------------------------------------------------------------------------
template <typename Real>
Segment3<Real>::Segment3(const Vec<3,Real>& p0, const Vec<3,Real>& p1)
    :
    P0(p0),
    P1(p1)
{
    ComputeCenterDirectionExtent();
}

//----------------------------------------------------------------------------
template <typename Real>
Segment3<Real>::Segment3 (const Vec<3,Real>& center,
    const Vec<3,Real>& direction, Real extent)
    :
    Center(center),
    Direction(direction),
    Extent(extent)
{
    ComputeEndPoints();
}

//----------------------------------------------------------------------------
template <typename Real>
void Segment3<Real>::ComputeCenterDirectionExtent()
{
    Center = ((Real)0.5)*(P0 + P1);
    Direction = P1 - P0;
    Extent = ((Real)0.5)*Direction.norm();
    Direction.normalize();
}

//----------------------------------------------------------------------------
template <typename Real>
void Segment3<Real>::ComputeEndPoints()
{
    P0 = Center - Extent * Direction;
    P1 = Center + Extent * Direction;
}

//----------------------------------------------------------------------------
template <typename Real>
Real Segment3<Real>::GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result)
{
    return MathUtils<Real>::Sqrt(GetSquaredDistance(other, result));
}

//----------------------------------------------------------------------------
template <typename Real>
Real Segment3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result)
{
    const Segment3<Real>* mSegment1 = dynamic_cast<const Segment3<Real>*>(&other);
    Segment3Segment3DistanceResult* seg3Seg3Res = dynamic_cast<Segment3Segment3DistanceResult*>(&result);

    if (mSegment1 && seg3Seg3Res)
    {
        const Segment3<Real>* mSegment0 = this;
        Vec<3, Real> diff = mSegment0->Center - mSegment1->Center;
        Real a01 = -mSegment0->Direction * mSegment1->Direction;
        Real b0 = diff * mSegment0->Direction;
        Real b1 = -diff * mSegment1->Direction;
        Real c = diff.norm2();
        Real det = MathUtils<Real>::FAbs((Real)1 - a01 * a01);
        Real s0, s1, sqrDist, extDet0, extDet1, tmpS0, tmpS1;

        if (det >= MathUtils<Real>::ZERO_TOLERANCE)
        {
            // Segments are not parallel.
            s0 = a01 * b1 - b0;
            s1 = a01 * b0 - b1;
            extDet0 = mSegment0->Extent * det;
            extDet1 = mSegment1->Extent * det;

            if (s0 >= -extDet0)
            {
                if (s0 <= extDet0)
                {
                    if (s1 >= -extDet1)
                    {
                        if (s1 <= extDet1)  // region 0 (interior)
                        {
                            // Minimum at interior points of segments.
                            Real invDet = ((Real)1)/det;
                            s0 *= invDet;
                            s1 *= invDet;
                            sqrDist = s0*(s0 + a01*s1 + ((Real)2)*b0) +
                                s1*(a01*s0 + s1 + ((Real)2)*b1) + c;
                        }
                        else  // region 3 (side)
                        {
                            s1 = mSegment1->Extent;
                            tmpS0 = -(a01*s1 + b0);
                            if (tmpS0 < -mSegment0->Extent)
                            {
                                s0 = -mSegment0->Extent;
                                sqrDist = s0*(s0 - ((Real)2)*tmpS0) +
                                    s1*(s1 + ((Real)2)*b1) + c;
                            }
                            else if (tmpS0 <= mSegment0->Extent)
                            {
                                s0 = tmpS0;
                                sqrDist = -s0*s0 + s1*(s1 + ((Real)2)*b1) + c;
                            }
                            else
                            {
                                s0 = mSegment0->Extent;
                                sqrDist = s0*(s0 - ((Real)2)*tmpS0) +
                                    s1*(s1 + ((Real)2)*b1) + c;
                            }
                        }
                    }
                    else  // region 7 (side)
                    {
                        s1 = -mSegment1->Extent;
                        tmpS0 = -(a01*s1 + b0);
                        if (tmpS0 < -mSegment0->Extent)
                        {
                            s0 = -mSegment0->Extent;
                            sqrDist = s0*(s0 - ((Real)2)*tmpS0) +
                                s1*(s1 + ((Real)2)*b1) + c;
                        }
                        else if (tmpS0 <= mSegment0->Extent)
                        {
                            s0 = tmpS0;
                            sqrDist = -s0*s0 + s1*(s1 + ((Real)2)*b1) + c;
                        }
                        else
                        {
                            s0 = mSegment0->Extent;
                            sqrDist = s0*(s0 - ((Real)2)*tmpS0) +
                                s1*(s1 + ((Real)2)*b1) + c;
                        }
                    }
                }
                else
                {
                    if (s1 >= -extDet1)
                    {
                        if (s1 <= extDet1)  // region 1 (side)
                        {
                            s0 = mSegment0->Extent;
                            tmpS1 = -(a01*s0 + b1);
                            if (tmpS1 < -mSegment1->Extent)
                            {
                                s1 = -mSegment1->Extent;
                                sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                    s0*(s0 + ((Real)2)*b0) + c;
                            }
                            else if (tmpS1 <= mSegment1->Extent)
                            {
                                s1 = tmpS1;
                                sqrDist = -s1*s1 + s0*(s0 + ((Real)2)*b0) + c;
                            }
                            else
                            {
                                s1 = mSegment1->Extent;
                                sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                    s0*(s0 + ((Real)2)*b0) + c;
                            }
                        }
                        else  // region 2 (corner)
                        {
                            s1 = mSegment1->Extent;
                            tmpS0 = -(a01*s1 + b0);
                            if (tmpS0 < -mSegment0->Extent)
                            {
                                s0 = -mSegment0->Extent;
                                sqrDist = s0*(s0 - ((Real)2)*tmpS0) +
                                    s1*(s1 + ((Real)2)*b1) + c;
                            }
                            else if (tmpS0 <= mSegment0->Extent)
                            {
                                s0 = tmpS0;
                                sqrDist = -s0*s0 + s1*(s1 + ((Real)2)*b1) + c;
                            }
                            else
                            {
                                s0 = mSegment0->Extent;
                                tmpS1 = -(a01*s0 + b1);
                                if (tmpS1 < -mSegment1->Extent)
                                {
                                    s1 = -mSegment1->Extent;
                                    sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                        s0*(s0 + ((Real)2)*b0) + c;
                                }
                                else if (tmpS1 <= mSegment1->Extent)
                                {
                                    s1 = tmpS1;
                                    sqrDist = -s1*s1 + s0*(s0 + ((Real)2)*b0) + c;
                                }
                                else
                                {
                                    s1 = mSegment1->Extent;
                                    sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                        s0*(s0 + ((Real)2)*b0) + c;
                                }
                            }
                        }
                    }
                    else  // region 8 (corner)
                    {
                        s1 = -mSegment1->Extent;
                        tmpS0 = -(a01*s1 + b0);
                        if (tmpS0 < -mSegment0->Extent)
                        {
                            s0 = -mSegment0->Extent;
                            sqrDist = s0*(s0 - ((Real)2)*tmpS0) +
                                s1*(s1 + ((Real)2)*b1) + c;
                        }
                        else if (tmpS0 <= mSegment0->Extent)
                        {
                            s0 = tmpS0;
                            sqrDist = -s0*s0 + s1*(s1 + ((Real)2)*b1) + c;
                        }
                        else
                        {
                            s0 = mSegment0->Extent;
                            tmpS1 = -(a01*s0 + b1);
                            if (tmpS1 > mSegment1->Extent)
                            {
                                s1 = mSegment1->Extent;
                                sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                    s0*(s0 + ((Real)2)*b0) + c;
                            }
                            else if (tmpS1 >= -mSegment1->Extent)
                            {
                                s1 = tmpS1;
                                sqrDist = -s1*s1 + s0*(s0 + ((Real)2)*b0) + c;
                            }
                            else
                            {
                                s1 = -mSegment1->Extent;
                                sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                    s0*(s0 + ((Real)2)*b0) + c;
                            }
                        }
                    }
                }
            }
            else
            {
                if (s1 >= -extDet1)
                {
                    if (s1 <= extDet1)  // region 5 (side)
                    {
                        s0 = -mSegment0->Extent;
                        tmpS1 = -(a01*s0 + b1);
                        if (tmpS1 < -mSegment1->Extent)
                        {
                            s1 = -mSegment1->Extent;
                            sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                s0*(s0 + ((Real)2)*b0) + c;
                        }
                        else if (tmpS1 <= mSegment1->Extent)
                        {
                            s1 = tmpS1;
                            sqrDist = -s1*s1 + s0*(s0 + ((Real)2)*b0) + c;
                        }
                        else
                        {
                            s1 = mSegment1->Extent;
                            sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                s0*(s0 + ((Real)2)*b0) + c;
                        }
                    }
                    else  // region 4 (corner)
                    {
                        s1 = mSegment1->Extent;
                        tmpS0 = -(a01*s1 + b0);
                        if (tmpS0 > mSegment0->Extent)
                        {
                            s0 = mSegment0->Extent;
                            sqrDist = s0*(s0 - ((Real)2)*tmpS0) +
                                s1*(s1 + ((Real)2)*b1) + c;
                        }
                        else if (tmpS0 >= -mSegment0->Extent)
                        {
                            s0 = tmpS0;
                            sqrDist = -s0*s0 + s1*(s1 + ((Real)2)*b1) + c;
                        }
                        else
                        {
                            s0 = -mSegment0->Extent;
                            tmpS1 = -(a01*s0 + b1);
                            if (tmpS1 < -mSegment1->Extent)
                            {
                                s1 = -mSegment1->Extent;
                                sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                    s0*(s0 + ((Real)2)*b0) + c;
                            }
                            else if (tmpS1 <= mSegment1->Extent)
                            {
                                s1 = tmpS1;
                                sqrDist = -s1*s1 + s0*(s0 + ((Real)2)*b0) + c;
                            }
                            else
                            {
                                s1 = mSegment1->Extent;
                                sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                    s0*(s0 + ((Real)2)*b0) + c;
                            }
                        }
                    }
                }
                else   // region 6 (corner)
                {
                    s1 = -mSegment1->Extent;
                    tmpS0 = -(a01*s1 + b0);
                    if (tmpS0 > mSegment0->Extent)
                    {
                        s0 = mSegment0->Extent;
                        sqrDist = s0*(s0 - ((Real)2)*tmpS0) +
                            s1*(s1 + ((Real)2)*b1) + c;
                    }
                    else if (tmpS0 >= -mSegment0->Extent)
                    {
                        s0 = tmpS0;
                        sqrDist = -s0*s0 + s1*(s1 + ((Real)2)*b1) + c;
                    }
                    else
                    {
                        s0 = -mSegment0->Extent;
                        tmpS1 = -(a01*s0 + b1);
                        if (tmpS1 < -mSegment1->Extent)
                        {
                            s1 = -mSegment1->Extent;
                            sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                s0*(s0 + ((Real)2)*b0) + c;
                        }
                        else if (tmpS1 <= mSegment1->Extent)
                        {
                            s1 = tmpS1;
                            sqrDist = -s1*s1 + s0*(s0 + ((Real)2)*b0) + c;
                        }
                        else
                        {
                            s1 = mSegment1->Extent;
                            sqrDist = s1*(s1 - ((Real)2)*tmpS1) +
                                s0*(s0 + ((Real)2)*b0) + c;
                        }
                    }
                }
            }
        }
        else
        {
            // The segments are parallel.  The average b0 term is designed to
            // ensure symmetry of the function.  That is, dist(seg0,seg1) and
            // dist(seg1,seg0) should produce the same number.
            Real e0pe1 = mSegment0->Extent + mSegment1->Extent;
            Real sign = (a01 > (Real)0 ? (Real)-1 : (Real)1);
            Real b0Avr = ((Real)0.5)*(b0 - sign*b1);
            Real lambda = -b0Avr;
            if (lambda < -e0pe1)
            {
                lambda = -e0pe1;
            }
            else if (lambda > e0pe1)
            {
                lambda = e0pe1;
            }

            s1 = -sign*lambda*mSegment1->Extent/e0pe1;
            s0 = lambda + sign*s1;
            sqrDist = lambda*(lambda + ((Real)2)*b0Avr) + c;
        }

        seg3Seg3Res->mClosestPoint0 = mSegment0->Center + s0 * mSegment0->Direction;
        seg3Seg3Res->mClosestPoint1 = mSegment1->Center + s1 * mSegment1->Direction;
        seg3Seg3Res->mSegment0Parameter = s0;
        seg3Seg3Res->mSegment1Parameter = s1;

        // Account for numerical round-off errors.
        if (sqrDist < (Real)0)
        {
            sqrDist = (Real)0;
        }
        return sqrDist;
    }

    const Triangle3<Real>* mTriangle = dynamic_cast<const Triangle3<Real>*>(&other);
    Segment3Triangle3DistanceResult* seg3Tri3Res = dynamic_cast<Segment3Triangle3DistanceResult*>(&result);
    if (mTriangle && seg3Tri3Res)
    {
        Line3<Real> line(this->Center, this->Direction);
        Line3Triangle3DistanceResult line3Tri3Res;
        Real sqrDist = line.GetDistance(other, line3Tri3Res);

        if (line3Tri3Res.mLineParameter >= -this->Extent)
        {
            if (line3Tri3Res.mLineParameter <= this->Extent)
            {
                seg3Tri3Res->mClosestPoint0 = line3Tri3Res.GetClosestPoint0();
                seg3Tri3Res->mClosestPoint1 = line3Tri3Res.GetClosestPoint1();
                seg3Tri3Res->mTriangleBary[0] = line3Tri3Res.mTriangleBary[0];
                seg3Tri3Res->mTriangleBary[1] = line3Tri3Res.mTriangleBary[1];
                seg3Tri3Res->mTriangleBary[2] = line3Tri3Res.mTriangleBary[2];
            }
            else
            {
                seg3Tri3Res->mClosestPoint0 = this->P1;

                Real ptDist = mTriangle->DistanceTo(seg3Tri3Res->mClosestPoint0);
                sqrDist = ptDist * ptDist;

                Point3<Real> closestPoint0(seg3Tri3Res->mClosestPoint0);
                Point3Triangle3DistanceResult pt3Tri3Res;
                //TRU sqrDist = closestPoint0.GetSquaredDistance(*mTriangle, pt3Tri3Res);
                sqrDist = closestPoint0.GetSquaredDistance(other, pt3Tri3Res);

                //TRU seg3Tri3Res->mClosestPoint1 = pt3Tri3Res.GetClosestPoint0();
                seg3Tri3Res->mClosestPoint1 = pt3Tri3Res.GetClosestPoint1();
                seg3Tri3Res->mSegmentParameter = this->Extent;
                seg3Tri3Res->mTriangleBary[0] = pt3Tri3Res.mTriangleBary[0];
                seg3Tri3Res->mTriangleBary[1] = pt3Tri3Res.mTriangleBary[1];
                seg3Tri3Res->mTriangleBary[2] = pt3Tri3Res.mTriangleBary[2];
            }
        }
        else
        {
            seg3Tri3Res->mClosestPoint0 = this->P0;

            Point3<Real> closestPoint0(seg3Tri3Res->mClosestPoint0);
            Point3Triangle3DistanceResult pt3Tri3Res;
            sqrDist = closestPoint0.GetSquaredDistance(other, pt3Tri3Res);

            seg3Tri3Res->mClosestPoint1 = pt3Tri3Res.GetClosestPoint1();
            seg3Tri3Res->mSegmentParameter = -this->Extent;
            seg3Tri3Res->mTriangleBary[0] = pt3Tri3Res.mTriangleBary[0];
            seg3Tri3Res->mTriangleBary[1] = pt3Tri3Res.mTriangleBary[1];
            seg3Tri3Res->mTriangleBary[2] = pt3Tri3Res.mTriangleBary[2];
        }

        return sqrDist;
    }

    const Rectangle3<Real>* mRectangle = dynamic_cast<const Rectangle3<Real>*>(&other);
    Segment3Rectangle3DistanceResult* seg3Rect3Res = dynamic_cast<Segment3Rectangle3DistanceResult*>(&result);
    if (mRectangle && seg3Rect3Res)
    {
        Line3<Real> line(this->Center, this->Direction);

        Line3Rectangle3DistanceResult lineRectRes;
        Real sqrDist = line.GetSquaredDistance(*mRectangle, lineRectRes);
        seg3Rect3Res->mSegmentParameter = lineRectRes.mLineParameter;

        if (seg3Rect3Res->mSegmentParameter >= -this->Extent)
        {
            if (seg3Rect3Res->mSegmentParameter <= this->Extent)
            {
                seg3Rect3Res->mClosestPoint0 = lineRectRes.GetClosestPoint0();
                seg3Rect3Res->mClosestPoint1 = lineRectRes.GetClosestPoint1();
                seg3Rect3Res->mRectCoord[0] = lineRectRes.mRectCoord[0];
                seg3Rect3Res->mRectCoord[1] = lineRectRes.mRectCoord[1];
            }
            else
            {
                seg3Rect3Res->mClosestPoint0 = this->P1;
                Point3<Real> pt(seg3Rect3Res->mClosestPoint0);

                Point3Rectangle3DistanceResult pt3Rect3Res;
                sqrDist = pt.GetSquaredDistance(*mRectangle, pt3Rect3Res);

                seg3Rect3Res->mClosestPoint1 = pt3Rect3Res.GetClosestPoint1();
                seg3Rect3Res->mSegmentParameter = this->Extent;

                seg3Rect3Res->mRectCoord[0] = pt3Rect3Res.mRectCoord[0];
                seg3Rect3Res->mRectCoord[1] = pt3Rect3Res.mRectCoord[1];
            }
        }
        else
        {
            seg3Rect3Res->mClosestPoint0 = this->P0;
            Point3<Real> pt(seg3Rect3Res->mClosestPoint0);

            Point3Rectangle3DistanceResult pt3Rect3Res;
            sqrDist = pt.GetSquaredDistance(*mRectangle, pt3Rect3Res);

            seg3Rect3Res->mClosestPoint1 = pt3Rect3Res.GetClosestPoint1();
            seg3Rect3Res->mSegmentParameter = this->Extent;

            seg3Rect3Res->mRectCoord[0] = pt3Rect3Res.mRectCoord[0];
            seg3Rect3Res->mRectCoord[1] = pt3Rect3Res.mRectCoord[1];
        }

        return sqrDist;
    }

    return MathUtils<Real>::MAX_REAL;
}

template <typename Real>
Real Segment3<Real>::GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result)
{
    const Triangle3<Real>* mTriangle = dynamic_cast<const Triangle3<Real>*>(&other);
    Segment3Triangle3DistanceResult* seg3Tri3Res = dynamic_cast<Segment3Triangle3DistanceResult*>(&result);
    if (mTriangle && seg3Tri3Res)
    {
        Vec<3, Real> movedCenter = this->Center + t*velocity0;
        Vec<3, Real> movedV0 = mTriangle->V[0] + t*velocity1;
        Vec<3, Real> movedV1 = mTriangle->V[1] + t*velocity1;
        Vec<3, Real> movedV2 = mTriangle->V[2] + t*velocity1;
        Segment3<Real> movedSeg(movedCenter, this->Direction,
            this->Extent);
        Triangle3<Real> movedTriangle(movedV0, movedV1, movedV2);

        return movedSeg.GetDistance(movedTriangle,*seg3Tri3Res);
        //return DistSegment3Triangle3<Real>(movedSeg, movedTriangle).Get();
    }

    return MathUtils<Real>::MAX_REAL;
}

template <typename Real>
Real Segment3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result)
{
    const Triangle3<Real>* mTriangle = dynamic_cast<const Triangle3<Real>*>(&other);
    Segment3Triangle3DistanceResult* seg3Tri3Res = dynamic_cast<Segment3Triangle3DistanceResult*>(&result);
    if (mTriangle && seg3Tri3Res)
    {
        Vec<3, Real> movedCenter = this->Center + t*velocity0;
        Vec<3, Real> movedV0 = mTriangle->V[0] + t*velocity1;
        Vec<3, Real> movedV1 = mTriangle->V[1] + t*velocity1;
        Vec<3, Real> movedV2 = mTriangle->V[2] + t*velocity1;
        Segment3<Real> movedSeg(movedCenter, this->Direction,
            this->Extent);
        Triangle3<Real> movedTriangle(movedV0, movedV1, movedV2);

        return movedSeg.GetSquaredDistance(movedTriangle,*seg3Tri3Res);
        //return DistSegment3Triangle3<Real>(movedSeg, movedTriangle).GetSquared();
    }

    return MathUtils<Real>::MAX_REAL;
}

template <typename Real>
Vec<3, Real> Segment3<Real>::ProjectOnSegment(const Vec<3, Real>& point)
{
	Vec<3, Real> w = point - this->P0;
	Real vsq = this->Direction * this->Direction;

	if (vsq == 0.0f)
		return Vec<3,Real>(0,0,0);

	Real proj = w * this->Direction;
	return (this->P0 + this->Direction * (proj / vsq));
}
