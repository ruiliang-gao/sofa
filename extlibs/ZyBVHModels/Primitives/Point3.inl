#include "Point3.h"

#include "Math/MathUtils.h"

#include "Triangle3.h"
#include "Rectangle3.h"

using namespace BVHModels;

template <typename Real>
Point3<Real>::Point3()
{
    Point.clear();
}

template <typename Real>
Point3<Real>::Point3(const Vec<3,Real>& point)
{
    Point = point;
}

template <typename Real>
Point3<Real>::~Point3()
{

}

template <typename Real>
Real Point3<Real>::GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result)
{
    return MathUtils<Real>::Sqrt(GetSquaredDistance(other, result));
}

template <typename Real>
Real Point3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result)
{
    const Triangle3<Real>* mTriangle = dynamic_cast<const Triangle3<Real>*>(&other);
    Point3Triangle3DistanceResult* pt3Tri3Res = dynamic_cast<Point3Triangle3DistanceResult*>(&result);
    if (mTriangle && pt3Tri3Res)
    {
        Vec<3,Real> diff = mTriangle->V[0] - this->Point;
        Vec<3,Real> edge0 = mTriangle->V[1] - mTriangle->V[0];
        Vec<3,Real> edge1 = mTriangle->V[2] - mTriangle->V[0];

        Real a00 = edge0.norm2();
        Real a01 = edge0 * edge1;
        Real a11 = edge1.norm2();
        Real b0 = diff * edge0;
        Real b1 = diff * edge1;
        Real c = diff.norm2();
        Real det = MathUtils<Real>::FAbs(a00 * a11 - a01 * a01);
        Real s = a01 * b1 - a11 * b0;
        Real t = a01 * b0 - a00 * b1;
        Real sqrDistance;

        if (s + t <= det)
        {
            if (s < (Real)0)
            {
                if (t < (Real)0)  // region 4
                {
                    if (b0 < (Real)0)
                    {
                        t = (Real)0;
                        if (-b0 >= a00)
                        {
                            s = (Real)1;
                            sqrDistance = a00 + ((Real)2)*b0 + c;
                        }
                        else
                        {
                            s = -b0/a00;
                            sqrDistance = b0*s + c;
                        }
                    }
                    else
                    {
                        s = (Real)0;
                        if (b1 >= (Real)0)
                        {
                            t = (Real)0;
                            sqrDistance = c;
                        }
                        else if (-b1 >= a11)
                        {
                            t = (Real)1;
                            sqrDistance = a11 + ((Real)2)*b1 + c;
                        }
                        else
                        {
                            t = -b1/a11;
                            sqrDistance = b1*t + c;
                        }
                    }
                }
                else  // region 3
                {
                    s = (Real)0;
                    if (b1 >= (Real)0)
                    {
                        t = (Real)0;
                        sqrDistance = c;
                    }
                    else if (-b1 >= a11)
                    {
                        t = (Real)1;
                        sqrDistance = a11 + ((Real)2)*b1 + c;
                    }
                    else
                    {
                        t = -b1/a11;
                        sqrDistance = b1*t + c;
                    }
                }
            }
            else if (t < (Real)0)  // region 5
            {
                t = (Real)0;
                if (b0 >= (Real)0)
                {
                    s = (Real)0;
                    sqrDistance = c;
                }
                else if (-b0 >= a00)
                {
                    s = (Real)1;
                    sqrDistance = a00 + ((Real)2)*b0 + c;
                }
                else
                {
                    s = -b0/a00;
                    sqrDistance = b0*s + c;
                }
            }
            else  // region 0
            {
                // minimum at interior point
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
                        s = (Real)1;
                        t = (Real)0;
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
                    s = (Real)0;
                    if (tmp1 <= (Real)0)
                    {
                        t = (Real)1;
                        sqrDistance = a11 + ((Real)2)*b1 + c;
                    }
                    else if (b1 >= (Real)0)
                    {
                        t = (Real)0;
                        sqrDistance = c;
                    }
                    else
                    {
                        t = -b1/a11;
                        sqrDistance = b1*t + c;
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
                    t = (Real)0;
                    if (tmp1 <= (Real)0)
                    {
                        s = (Real)1;
                        sqrDistance = a00 + ((Real)2)*b0 + c;
                    }
                    else if (b0 >= (Real)0)
                    {
                        s = (Real)0;
                        sqrDistance = c;
                    }
                    else
                    {
                        s = -b0/a00;
                        sqrDistance = b0*s + c;
                    }
                }
            }
            else  // region 1
            {
                numer = a11 + b1 - a01 - b0;
                if (numer <= (Real)0)
                {
                    s = (Real)0;
                    t = (Real)1;
                    sqrDistance = a11 + ((Real)2)*b1 + c;
                }
                else
                {
                    denom = a00 - ((Real)2)*a01 + a11;
                    if (numer >= denom)
                    {
                        s = (Real)1;
                        t = (Real)0;
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

        // Account for numerical round-off error.
        if (sqrDistance < (Real)0)
        {
            sqrDistance = (Real)0;
        }

        pt3Tri3Res->mClosestPoint0 = this->Point;
        pt3Tri3Res->mClosestPoint1 = mTriangle->V[0] + s * edge0 + t * edge1;
        pt3Tri3Res->mTriangleBary[1] = s;
        pt3Tri3Res->mTriangleBary[2] = t;
        pt3Tri3Res->mTriangleBary[0] = (Real) 1 - s - t;

        return sqrDistance;
    }

    const Rectangle3<Real>* mRectangle = dynamic_cast<const Rectangle3<Real>*>(&other);
    Point3Rectangle3DistanceResult* pt3Rect3Res = dynamic_cast<Point3Rectangle3DistanceResult*>(&result);
    if (mRectangle && pt3Rect3Res)
    {
        Vec<3,Real> diff = mRectangle->Center - this->Point;
        Real b0 = diff * (mRectangle->Axis[0]);
        Real b1 = diff * (mRectangle->Axis[1]);
        Real s0 = -b0, s1 = -b1;
        Real sqrDistance = diff.norm2();

        if (s0 < -mRectangle->Extent[0])
        {
            s0 = -mRectangle->Extent[0];
        }
        else if (s0 > mRectangle->Extent[0])
        {
            s0 = mRectangle->Extent[0];
        }

        sqrDistance += s0 * (s0 + ((Real)2) * b0);

        if (s1 < -mRectangle->Extent[1])
        {
            s1 = -mRectangle->Extent[1];
        }
        else if (s1 > mRectangle->Extent[1])
        {
            s1 = mRectangle->Extent[1];
        }
        sqrDistance += s1 * (s1 + ((Real)2)*b1);

        // Account for numerical round-off error.
        if (sqrDistance < (Real)0)
        {
            sqrDistance = (Real)0;
        }

        pt3Rect3Res->mClosestPoint0 = this->Point;
        pt3Rect3Res->mClosestPoint1 = mRectangle->Center + s0 * mRectangle->Axis[0] +
            s1 * mRectangle->Axis[1];
        pt3Rect3Res->mRectCoord[0] = s0;
        pt3Rect3Res->mRectCoord[1] = s1;

        return sqrDistance;
    }

	return MathUtils<Real>::MAX_REAL;
}

template <typename Real>
Real Point3<Real>::GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result)
{
    const Triangle3<Real>* mTriangle = dynamic_cast<const Triangle3<Real>*>(&other);
    Point3Triangle3DistanceResult* pt3Tri3Res = dynamic_cast<Point3Triangle3DistanceResult*>(&result);
    if (mTriangle && pt3Tri3Res)
    {
        Point3<Real> movedPoint = this->Point + t * velocity0;
        Vec<3, Real> movedV0 = mTriangle->V[0] + t * velocity1;
        Vec<3, Real> movedV1 = mTriangle->V[1] + t * velocity1;
        Vec<3, Real> movedV2 = mTriangle->V[2] + t * velocity1;
        Triangle3<Real> movedTriangle(movedV0, movedV1, movedV2);

        return movedPoint.GetDistance(movedTriangle,*pt3Tri3Res);
    }

    const Rectangle3<Real>* mRectangle = dynamic_cast<const Rectangle3<Real>*>(&other);
    Point3Rectangle3DistanceResult* pt3Rect3Res = dynamic_cast<Point3Rectangle3DistanceResult*>(&result);
    if (mRectangle && pt3Rect3Res)
    {
        Point3<Real> movedPoint = this->Point + t * velocity0;
        Vec<3,Real> movedCenter = mRectangle->Center + t * velocity1;
        Rectangle3<Real> movedRectangle(movedCenter, mRectangle->Axis[0],
                                        mRectangle->Axis[1], mRectangle->Extent[0], mRectangle->Extent[1]);

        return movedPoint.GetDistance(movedRectangle, *pt3Rect3Res);
    }

    return MathUtils<Real>::MAX_REAL;
}

template <typename Real>
Real Point3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result)
{
    const Triangle3<Real>* mTriangle = dynamic_cast<const Triangle3<Real>*>(&other);
    Point3Triangle3DistanceResult* pt3Tri3Res = dynamic_cast<Point3Triangle3DistanceResult*>(&result);
    if (mTriangle && pt3Tri3Res)
    {
        Point3<Real> movedPoint = this->Point + t*velocity0;
        Vec<3, Real> movedV0 = mTriangle->V[0] + t*velocity1;
        Vec<3, Real> movedV1 = mTriangle->V[1] + t*velocity1;
        Vec<3, Real> movedV2 = mTriangle->V[2] + t*velocity1;
        Triangle3<Real> movedTriangle(movedV0, movedV1, movedV2);

        return movedPoint.GetSquaredDistance(movedTriangle,*pt3Tri3Res);
    }

    const Rectangle3<Real>* mRectangle = dynamic_cast<const Rectangle3<Real>*>(&other);
    Point3Rectangle3DistanceResult* pt3Rect3Res = dynamic_cast<Point3Rectangle3DistanceResult*>(&result);
    if (mRectangle && pt3Rect3Res)
    {
        Point3<Real> movedPoint = this->Point + t*velocity0;
        Vec<3,Real> movedCenter = mRectangle->Center + t*velocity1;
        Rectangle3<Real> movedRectangle(movedCenter, mRectangle->Axis[0],
            mRectangle->Axis[1], mRectangle->Extent[0], mRectangle->Extent[1]);
        return movedPoint.GetSquaredDistance(movedRectangle, *pt3Rect3Res);
    }

    return MathUtils<Real>::MAX_REAL;
}
