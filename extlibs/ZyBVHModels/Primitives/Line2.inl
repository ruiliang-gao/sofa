#include "Line2.h"

#include "Math/MathUtils.h"

using namespace BVHModels;

//----------------------------------------------------------------------------
template <typename Real>
Line2<Real>::Line2 ()
{
}
//----------------------------------------------------------------------------
template <typename Real>
Line2<Real>::~Line2 ()
{
}
//----------------------------------------------------------------------------
template <typename Real>
Line2<Real>::Line2 (const Vec<2,Real>& origin,
    const Vec<2,Real>& direction)
    :
    Origin(origin),
    Direction(direction)
{
}

template <typename Real>
bool Line2<Real>::IsIntersectionQuerySupported(const PrimitiveType &other)
{
    if (other == PT_LINE2)
        return true;

    return false;
}

template <typename Real>
bool Line2<Real>::Test(const Intersectable<Real, Vec<2,Real> >& other)
{
    if (!IsIntersectionQuerySupported(other.GetIntersectableType()))
        return false;

    if (other.GetIntersectableType() == PT_LINE2)
    {
        const Line2<Real>* mLine1 = static_cast<const Line2<Real>*>(&other);
        int mIntersectionType = Classify(this->Origin, this->Direction,
            mLine1->Origin, mLine1->Direction, MathUtils<Real>::ZERO_TOLERANCE, NULL);

        return mIntersectionType != IT_EMPTY;
    }
    return false;
}
//----------------------------------------------------------------------------
template <typename Real>
bool Line2<Real>::Find(const Intersectable<Real, Vec<2,Real> >& other, IntersectionResult<Real>& result)
{
    if (!IsIntersectionQuerySupported(other.GetIntersectableType()))
        return false;

    if (other.GetIntersectableType() == PT_LINE2)
    {
        const Line2<Real>* mLine1 = static_cast<const Line2<Real>*>(&other);

        Line2Line2IntersectionResult<Real>* line2Res = static_cast<Line2Line2IntersectionResult<Real>*>(&result);
        if (!line2Res)
            return false;

        Real s[2];
        line2Res->SetMIntersectionType(Classify(this->Origin, this->Direction,
                                                mLine1->Origin, mLine1->Direction, line2Res->mDotThreshold, s));

        if (line2Res->GetMIntersectionType() == IT_POINT)
        {
            line2Res->mQuantity = 1;
            line2Res->mPoint = this->Origin + s[0] * this->Direction;
        }
        else if (line2Res->GetMIntersectionType() == IT_LINE)
        {
            line2Res->mQuantity = MathUtils<Real>::MAX_INT;
        }
        else
        {
            line2Res->mQuantity = 0;
        }

        return (line2Res->GetMIntersectionType() != IT_EMPTY);
    }
    return false;
}
//----------------------------------------------------------------------------
template <typename Real>
//TRU int Line2<Real>::Classify (const Vec<2,Real>& P0,    const Vec<2,Real>& D0, const Vec<2,Real>& P1, const Vec<2,Real>& D1,    Real dotThreshold, Real* s)
IntersectionType Line2<Real>::Classify (const Vec<2,Real>& P0,
    const Vec<2,Real>& D0, const Vec<2,Real>& P1, const Vec<2,Real>& D1,
    Real dotThreshold, Real* s)
{
    // Ensure dotThreshold is nonnegative.
    dotThreshold = std::max(dotThreshold, (Real)0);

    // The intersection of two lines is a solution to P0+s0*D0 = P1+s1*D1.
    // Rewrite this as s0*D0 - s1*D1 = P1 - P0 = Q.  If D0.Dot(Perp(D1)) = 0,
    // the lines are parallel.  Additionally, if Q.Dot(Perp(D1)) = 0, the
    // lines are the same.  If D0.Dot(Perp(D1)) is not zero, then
    //   s0 = Q.Dot(Perp(D1))/D0.Dot(Perp(D1))
    // produces the point of intersection.  Also,
    //   s1 = Q.Dot(Perp(D0))/D0.Dot(Perp(D1))

    Vec<2,Real> diff = P1 - P0;
    //TRU Real D0DotPerpD1 = D0.DotPerp(D1);
    Real D0DotPerpD1 = D0[0]*D1[1] - D0[1]*D1[0];
    if (MathUtils<Real>::FAbs(D0DotPerpD1) > dotThreshold)
    {
        // Lines intersect in a single point.
        if (s)
        {
            Real invD0DotPerpD1 = ((Real)1) / D0DotPerpD1;
            //TRU Real diffDotPerpD0 = diff.DotPerp(D0);
            //TRU Real diffDotPerpD1 = diff.DotPerp(D1);
            Real diffDotPerpD0 = diff[0] * D0[1] - diff[1] * D0[0];
            Real diffDotPerpD1 = diff[0] * D1[1] - diff[1] * D1[0];
            s[0] = diffDotPerpD1 * invD0DotPerpD1;
            s[1] = diffDotPerpD0 * invD0DotPerpD1;
        }
        return IT_POINT;
    }

    // Lines are parallel.
    diff.normalize();
    //TRU Real diffNDotPerpD1 = diff.DotPerp(D1);
    Real diffNDotPerpD1 = diff[0] * D1[1] - diff[1] * D1[0];
    if (MathUtils<Real>::FAbs(diffNDotPerpD1) <= dotThreshold)
    {
        // Lines are colinear.
        return IT_LINE;
    }

    // Lines are parallel, but distinct.
    return IT_EMPTY;
}
