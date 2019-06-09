#include "Segment3.h"
#include "Line3.h"

#include "Point3.h"
#include "Triangle3.h"
#include "Rectangle3.h"

#include "Math/MathUtils.h"

using namespace BVHModels;

//----------------------------------------------------------------------------
template <typename Real>
Segment3<Real>::Segment3 ()
{
}

//----------------------------------------------------------------------------
template <typename Real>
Segment3<Real>::~Segment3 ()
{
}

//----------------------------------------------------------------------------
template <typename Real>
Segment3<Real>::Segment3 (const Vec<3,Real>& p0, const Vec<3,Real>& p1)
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
void Segment3<Real>::ComputeCenterDirectionExtent ()
{
    Center = ((Real)0.5)*(P0 + P1);
    Direction = P1 - P0;
    //TRU Extent = ((Real)0.5)*Direction.Normalize();
    Extent = ((Real)0.5)*Direction.norm();
    Direction.normalize();
}

//----------------------------------------------------------------------------
template <typename Real>
void Segment3<Real>::ComputeEndPoints ()
{
    P0 = Center - Extent*Direction;
    P1 = Center + Extent*Direction;
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
    const Triangle3<Real>* mTriangle = dynamic_cast<const Triangle3<Real>*>(&other);
    Segment3Triangle3DistanceResult* seg3Tri3Res = dynamic_cast<Segment3Triangle3DistanceResult*>(&result);
    if (mTriangle && seg3Tri3Res)
    {
        // DistLine3Triangle3<Real> queryLT(line, *mTriangle);
        Line3<Real> line(this->Center, this->Direction);
        Line3Triangle3DistanceResult line3Tri3Res;
        //TRU Real sqrDist = line.GetDistance(*mTriangle, line3Tri3Res);
        Real sqrDist = line.GetDistance(other, line3Tri3Res);
        //Real sqrDist = line.GetDistance(dynamic_cast<const DistanceComputable<Real, Vec<3,Real> >&>(*mTriangle), line3Tri3Res);

        //    Real sqrDist = queryLT.GetSquaredDistance();
        //    mSegmentParameter = queryLT.GetLineParameter();

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

                /*DistPoint3Triangle3<Real> queryPT(mClosestPoint0, *mTriangle);
                sqrDist = queryPT.GetSquared();
                mClosestPoint1 = queryPT.GetClosestPoint1();
                mSegmentParameter = mSegment->Extent;
                mTriangleBary[0] = queryPT.GetTriangleBary(0);
                mTriangleBary[1] = queryPT.GetTriangleBary(1);
                mTriangleBary[2] = queryPT.GetTriangleBary(2);*/
            }
        }
        else
        {
            seg3Tri3Res->mClosestPoint0 = this->P0;

            Point3<Real> closestPoint0(seg3Tri3Res->mClosestPoint0);
            Point3Triangle3DistanceResult pt3Tri3Res;
            //TRU sqrDist = closestPoint0.GetSquaredDistance(*mTriangle, pt3Tri3Res);
            sqrDist = closestPoint0.GetSquaredDistance(other, pt3Tri3Res);

            seg3Tri3Res->mClosestPoint1 = pt3Tri3Res.GetClosestPoint1();
            seg3Tri3Res->mSegmentParameter = -this->Extent;
            seg3Tri3Res->mTriangleBary[0] = pt3Tri3Res.mTriangleBary[0];
            seg3Tri3Res->mTriangleBary[1] = pt3Tri3Res.mTriangleBary[1];
            seg3Tri3Res->mTriangleBary[2] = pt3Tri3Res.mTriangleBary[2];

            /*DistPoint3Triangle3<Real> queryPT(mClosestPoint0, *mTriangle);
            sqrDist = queryPT.GetSquared();
            mClosestPoint1 = queryPT.GetClosestPoint1();
            mSegmentParameter = -mSegment->Extent;
            mTriangleBary[0] = queryPT.GetTriangleBary(0);
            mTriangleBary[1] = queryPT.GetTriangleBary(1);
            mTriangleBary[2] = queryPT.GetTriangleBary(2);*/
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
