#include "Segment2.h"

#include "Triangle2.h"
#include "Math/IntervalUtils.h"

#include "Utils_2D.h"

using namespace BVHModels;

//----------------------------------------------------------------------------
template <typename Real>
Segment2<Real>::Segment2 ()
{
}
//----------------------------------------------------------------------------
template <typename Real>
Segment2<Real>::~Segment2 ()
{
}
//----------------------------------------------------------------------------
template <typename Real>
Segment2<Real>::Segment2 (const Vec<2,Real>& p0, const Vec<2,Real>& p1)
    :
    P0(p0),
    P1(p1)
{
    ComputeCenterDirectionExtent();
}
//----------------------------------------------------------------------------
template <typename Real>
Segment2<Real>::Segment2 (const Vec<2,Real>& center,
    const Vec<2,Real>& direction, Real extent)
    :
    Center(center),
    Direction(direction),
    Extent(extent)
{
    ComputeEndPoints();
}
//----------------------------------------------------------------------------
template <typename Real>
void Segment2<Real>::ComputeCenterDirectionExtent ()
{
    Center = ((Real)0.5)*(P0 + P1);
    Direction = P1 - P0;
    //TRU Extent = ((Real)0.5)*Direction.Normalize();
    Extent = ((Real)0.5)*Direction.norm();
    Direction.normalize();
}
//----------------------------------------------------------------------------
template <typename Real>
void Segment2<Real>::ComputeEndPoints ()
{
    P0 = Center - Extent*Direction;
    P1 = Center + Extent*Direction;
}

//----------------------------------------------------------------------------
template <typename Real>
bool Segment2<Real>::Test(const Intersectable<Real, Vec<2, Real> >& intersectable)
{
    const Triangle2<Real>* mTriangle = dynamic_cast<const Triangle2<Real>*>(&intersectable);

    if (mTriangle)
    {
        Real dist[3];
        int sign[3], positive, negative, zero;
        Utils_2D<Real>::TriangleLineRelations(this->Center,
            this->Direction, *mTriangle, dist, sign, positive, negative,
            zero);

        if (positive == 3 || negative == 3)
        {
            Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_EMPTY;
        }
        else
        {
            Real param[2];
            Utils_2D<Real>::GetInterval(this->Center,
                this->Direction, *mTriangle, dist, sign, param);

            Interval1Intersector<Real> intr(param[0], param[1],
                -this->Extent, +this->Extent);

            intr.Find();

            int mQuantity = intr.GetNumIntersections();
            if (mQuantity == 2)
            {
                Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_SEGMENT;
            }
            else if (mQuantity == 1)
            {
                Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_POINT;
            }
            else
            {
                Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_EMPTY;
            }
        }
    }
    return Intersectable<Real, Vec<2,Real> >::mIntersectionType != IT_EMPTY;
}

//----------------------------------------------------------------------------
template <typename Real>
bool Segment2<Real>::Find(const Intersectable<Real, Vec<2, Real> >& intersectable, IntersectionResult<Real>& result)
{
    //TRU Segment2IntersectionResult<Real>* segResult = dynamic_cast<Segment2IntersectionResult<Real>*>(&result);
    Segment2IntersectionResult<Real>* segResult = static_cast<Segment2IntersectionResult<Real>*>(&result);
    if (!segResult)
        return false;

    const Triangle2<Real>* mTriangle = dynamic_cast<const Triangle2<Real>*>(&intersectable);

    if (mTriangle)
    {
        Real dist[3];
        int sign[3], positive, negative, zero;
        Utils_2D<Real>::TriangleLineRelations(this->Center,
            this->Direction, *mTriangle, dist, sign, positive, negative,
            zero);

        if (positive == 3 || negative == 3)
        {
            // No intersections.
            segResult->mQuantity = 0;
            segResult->SetIntersectionType(IT_EMPTY);
            Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_EMPTY;
        }
        else
        {
            Real param[2];
            Utils_2D<Real>::GetInterval(this->Center,
                this->Direction, *mTriangle, dist, sign, param);

            Interval1Intersector<Real> intr(param[0], param[1],
                -this->Extent, +this->Extent);

            intr.Find();

            segResult->mQuantity = intr.GetNumIntersections();
            if (segResult->mQuantity == 2)
            {
                // Segment intersection.
                Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_SEGMENT;
                //TRU segResult->intersectionType = IT_SEGMENT;
                segResult->SetIntersectionType(IT_SEGMENT);

                segResult->mPoint[0] = this->Center +
                    intr.GetIntersection(0) * this->Direction;
                segResult->mPoint[1] = this->Center +
                    intr.GetIntersection(1) * this->Direction;
            }
            else if (segResult->mQuantity == 1)
            {
                // Point intersection.
                Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_POINT;
                //TRU segResult->intersectionType = IT_POINT;
                segResult->SetIntersectionType(IT_POINT);
                segResult->mPoint[0] = this->Center +
                    intr.GetIntersection(0) * this->Direction;
            }
            else
            {
                // No intersections.
                Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_EMPTY;
                //TRU segResult->intersectionType = IT_EMPTY;
                segResult->SetIntersectionType(IT_EMPTY);
            }
        }
    }

    return Intersectable<Real, Vec<2,Real> >::mIntersectionType != IT_EMPTY;
}
