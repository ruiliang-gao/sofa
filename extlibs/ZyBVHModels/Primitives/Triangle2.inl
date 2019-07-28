#include "Triangle2.h"

#include "Math/MathUtils.h"
#include "Query/Query2.h"
#include "Utils_2D.h"

#include "Math/IntervalUtils.h"

#include "Segment2.h"

using namespace BVHModels;

//----------------------------------------------------------------------------
template <typename Real>
Triangle2<Real>::Triangle2 ()
{
}
//----------------------------------------------------------------------------
template <typename Real>
Triangle2<Real>::~Triangle2 ()
{
}
//----------------------------------------------------------------------------
template <typename Real>
Triangle2<Real>::Triangle2 (const Vec<2,Real>& v0,
    const Vec<2,Real>& v1, const Vec<2,Real>& v2)
{
    V[0] = v0;
    V[1] = v1;
    V[2] = v2;
}
//----------------------------------------------------------------------------
template <typename Real>
Triangle2<Real>::Triangle2(const Vec<2,Real> vertex[3])
{
    for (int i = 0; i < 3; ++i)
    {
        V[i] = vertex[i];
    }
}
//----------------------------------------------------------------------------
template <typename Real>
Real Triangle2<Real>::DistanceTo(const Vec<2,Real>& q) const
{
    Vec<2,Real> diff = V[0] - q;
    Vec<2,Real> edge0 = V[1] - V[0];
    Vec<2,Real> edge1 = V[2] - V[0];
    Real a00 = edge0.norm2();
    Real a01 = edge0 * edge1;
    Real a11 = edge1.norm2();
    Real b0 = diff * edge0;
    Real b1 = diff * edge1;
    Real c = diff.norm2();
    Real det = MathUtils<Real>::FAbs(a00*a11 - a01*a01);
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

    return MathUtils<Real>::Sqrt(MathUtils<Real>::FAbs(sqrDistance));
}

template <typename Real>
bool Triangle2<Real>::IsIntersectionQuerySupported(const PrimitiveType& other)
{
    if ((other == PT_SEGMENT2) ||
        (other == PT_TRIANGLE2))
        return true;

    return false;
}

template <typename Real>
bool Triangle2<Real>::Test(const Intersectable<Real, Vec<2,Real> >& intersectable)
{
    const Segment2<Real>* mSegment = dynamic_cast<const Segment2<Real>*> (&intersectable);

    if (mSegment)
    {
        Real dist[3];
        int sign[3], positive, negative, zero;
        Utils_2D<Real>::TriangleLineRelations(mSegment->Center,
            mSegment->Direction, *this, dist, sign, positive, negative,
            zero);

        if (positive == 3 || negative == 3)
        {
            Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_EMPTY;
        }
        else
        {
            Real param[2];
            Utils_2D<Real>::GetInterval(mSegment->Center,
                mSegment->Direction, *this, dist, sign, param);

            Interval1Intersector<Real> intr(param[0], param[1],
                -mSegment->Extent, +mSegment->Extent);

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

    const Triangle2<Real>* mTriangle1 = dynamic_cast<const Triangle2<Real>*>(&intersectable);
    if (mTriangle1)
    {
        int i0, i1;
        Vec<2,Real> dir;

        // Test edges of triangle0 for separation.
        for (i0 = 0, i1 = 2; i0 < 3; i1 = i0++)
        {
            // Test axis V0[i1] + t*perp(V0[i0]-V0[i1]), perp(x,y) = (y,-x).
            dir.x() = this->V[i0].y() - this->V[i1].y();
            dir.y() = this->V[i1].x() - this->V[i0].x();
            if (WhichSide(mTriangle1->V, this->V[i1], dir) > 0)
            {
                // Triangle1 is entirely on positive side of triangle0 edge.
                return false;
            }
        }

        // Test edges of triangle1 for separation.
        for (i0 = 0, i1 = 2; i0 < 3; i1 = i0++)
        {
            // Test axis V1[i1] + t*perp(V1[i0]-V1[i1]), perp(x,y) = (y,-x).
            dir.x() = mTriangle1->V[i0].y() - mTriangle1->V[i1].y();
            dir.y() = mTriangle1->V[i1].x() - mTriangle1->V[i0].x();
            if (WhichSide(this->V, mTriangle1->V[i1], dir) > 0)
            {
                // Triangle0 is entirely on positive side of triangle1 edge.
                return false;
            }
        }

        return true;
    }

    return Intersectable<Real, Vec<2,Real> >::mIntersectionType != IT_EMPTY;
}

//----------------------------------------------------------------------------
template <typename Real>
bool Triangle2<Real>::Find(const Intersectable<Real, Vec<2, Real> > & intersectable, IntersectionResult<Real>& result)
{
    const Segment2<Real>* mSegment = dynamic_cast<const Segment2<Real>*> (&intersectable);
    Triangle2IntersectionResult<Real>* triResult = static_cast<Triangle2IntersectionResult<Real>*>(&result);

    if (!triResult)
        return false;

    if (mSegment)
    {
        Real dist[3];
        int sign[3], positive, negative, zero;
        Utils_2D<Real>::TriangleLineRelations(mSegment->Center,
            mSegment->Direction, *this, dist, sign, positive, negative,
            zero);

        if (positive == 3 || negative == 3)
        {
            // No intersections.
            triResult->mQuantity = 0;
            triResult->SetIntersectionType(IT_EMPTY);
            Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_EMPTY;
        }
        else
        {
            Real param[2];
            Utils_2D<Real>::GetInterval(mSegment->Center,
                mSegment->Direction, *this, dist, sign, param);

            Interval1Intersector<Real> intr(param[0], param[1],
                -mSegment->Extent, +mSegment->Extent);

            intr.Find();

            triResult->mQuantity = intr.GetNumIntersections();
            if (triResult->mQuantity == 2)
            {
                // Segment intersection.
                Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_SEGMENT;
                triResult->SetIntersectionType(IT_SEGMENT);
                triResult->mPoint[0] = mSegment->Center +
                    intr.GetIntersection(0)*mSegment->Direction;
                triResult->mPoint[1] = mSegment->Center +
                    intr.GetIntersection(1)*mSegment->Direction;
            }
            else if (triResult->mQuantity == 1)
            {
                // Point intersection.
                Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_POINT;
                triResult->SetIntersectionType(IT_POINT);
                triResult->mPoint[0] = mSegment->Center +
                    intr.GetIntersection(0)*mSegment->Direction;
            }
            else
            {
                // No intersections.
                Intersectable<Real, Vec<2,Real> >::mIntersectionType = IT_EMPTY;
                triResult->SetIntersectionType(IT_EMPTY);
            }
        }
    }

    const Triangle2<Real>* mTriangle = dynamic_cast<const Triangle2<Real>*>(&intersectable);

    // TODO: Triangle2/Triangle2 intersection
    if (mTriangle)
    {

    }

    return Intersectable<Real, Vec<2,Real> >::mIntersectionType != IT_EMPTY;
}

//----------------------------------------------------------------------------
template <typename Real>
int Triangle2<Real>::WhichSide (const Vec<2,Real> V[3],
    const Vec<2,Real>& P, const Vec<2,Real>& D)
{
    // Vertices are projected to the form P+t*D.  Return value is +1 if all
    // t > 0, -1 if all t < 0, 0 otherwise, in which case the line splits the
    // triangle.

    int positive = 0, negative = 0, zero = 0;
    for (int i = 0; i < 3; ++i)
    {
        //TRU Real t = D.Dot(V[i] - P);
        Real t = D * (V[i] - P);
        if (t > (Real)0)
        {
            ++positive;
        }
        else if (t < (Real)0)
        {
            ++negative;
        }
        else
        {
            ++zero;
        }

        if (positive > 0 && negative > 0)
        {
            return 0;
        }
    }
    return (zero == 0 ? (positive > 0 ? 1 : -1) : 0);
}
//----------------------------------------------------------------------------
template <typename Real>
void Triangle2<Real>::ClipConvexPolygonAgainstLine (
    const Vec<2,Real>& N, Real c, int& quantity, Vec<2,Real> V[6])
{
    // The input vertices are assumed to be in counterclockwise order.  The
    // ordering is an invariant of this function.

    // Test on which side of line the vertices are.
    int positive = 0, negative = 0, pIndex = -1;
    Real test[6];
    int i;
    for (i = 0; i < quantity; ++i)
    {
        test[i] = (N * V[i]) - c;
        if (test[i] > (Real)0)
        {
            positive++;
            if (pIndex < 0)
            {
                pIndex = i;
            }
        }
        else if (test[i] < (Real)0)
        {
            negative++;
        }
    }

    if (positive > 0)
    {
        if (negative > 0)
        {
            // Line transversely intersects polygon.
            Vec<2,Real> CV[6];
            int cQuantity = 0, cur, prv;
            Real t;

            if (pIndex > 0)
            {
                // First clip vertex on line.
                cur = pIndex;
                prv = cur - 1;
                t = test[cur]/(test[cur] - test[prv]);
                CV[cQuantity++] = V[cur] + t*(V[prv] - V[cur]);

                // Vertices on positive side of line.
                while (cur < quantity && test[cur] > (Real)0)
                {
                    CV[cQuantity++] = V[cur++];
                }

                // Last clip vertex on line.
                if (cur < quantity)
                {
                    prv = cur - 1;
                }
                else
                {
                    cur = 0;
                    prv = quantity - 1;
                }
                t = test[cur]/(test[cur] - test[prv]);
                CV[cQuantity++] = V[cur] + t*(V[prv]-V[cur]);
            }
            else  // pIndex is 0
            {
                // Vertices on positive side of line.
                cur = 0;
                while (cur < quantity && test[cur] > (Real)0)
                {
                    CV[cQuantity++] = V[cur++];
                }

                // Last clip vertex on line.
                prv = cur - 1;
                t = test[cur]/(test[cur] - test[prv]);
                CV[cQuantity++] = V[cur] + t*(V[prv] - V[cur]);

                // Skip vertices on negative side.
                while (cur < quantity && test[cur] <= (Real)0)
                {
                    ++cur;
                }

                // First clip vertex on line.
                if (cur < quantity)
                {
                    prv = cur - 1;
                    t = test[cur]/(test[cur] - test[prv]);
                    CV[cQuantity++] = V[cur] + t*(V[prv] - V[cur]);

                    // Vertices on positive side of line.
                    while (cur < quantity && test[cur] > (Real)0)
                    {
                        CV[cQuantity++] = V[cur++];
                    }
                }
                else
                {
                    // cur = 0
                    prv = quantity - 1;
                    t = test[0]/(test[0] - test[prv]);
                    CV[cQuantity++] = V[0] + t*(V[prv] - V[0]);
                }
            }

            quantity = cQuantity;
            memcpy(V, CV, cQuantity*sizeof(Vec<2,Real>));
        }
        // else polygon fully on positive side of line, nothing to do.
    }
    else
    {
        // Polygon does not intersect positive side of line, clip all.
        quantity = 0;
    }
}

//----------------------------------------------------------------------------
template <typename Real>
void Triangle2<Real>::ComputeTwo (Configuration& cfg,
    const Vec<2,Real> V[3], const Vec<2,Real>& D, int i0, int i1,
    int i2)
{
    cfg.Map = M12;
    cfg.Index[0] = i0;
    cfg.Index[1] = i1;
    cfg.Index[2] = i2;
    cfg.Min = D * (V[i0] - V[i1]);
    cfg.Max = (Real)0;
}

//----------------------------------------------------------------------------
template <typename Real>
void Triangle2<Real>::ComputeThree (Configuration& cfg,
    const Vec<2,Real> V[3], const Vec<2,Real>& D,
    const Vec<2,Real>& P)
{
    Real d0 = D * (V[0] - P);
    Real d1 = D * (V[1] - P);
    Real d2 = D * (V[2] - P);

    // Make sure that m_aiIndex[...] is an even permutation of (0,1,2)
    // whenever the map value is M12 or M21.  This is needed to guarantee
    // the intersection of overlapping edges is properly computed.

    if (d0 <= d1)
    {
        if (d1 <= d2)  // d0 <= d1 <= d2
        {
            if (d0 != d1)
            {
                cfg.Map = (d1 != d2 ? M11 : M12);
            }
            else
            {
                cfg.Map = M21;
            }

            cfg.Index[0] = 0;
            cfg.Index[1] = 1;
            cfg.Index[2] = 2;
            cfg.Min = d0;
            cfg.Max = d2;
        }
        else if (d0 <= d2)  // d0 <= d2 < d1
        {
            if (d0 != d2)
            {
                cfg.Map = M11;
                cfg.Index[0] = 0;
                cfg.Index[1] = 2;
                cfg.Index[2] = 1;
            }
            else
            {
                cfg.Map = M21;
                cfg.Index[0] = 2;
                cfg.Index[1] = 0;
                cfg.Index[2] = 1;
            }

            cfg.Min = d0;
            cfg.Max = d1;
        }
        else  // d2 < d0 <= d1
        {
            cfg.Map = (d0 != d1 ? M12 : M11);
            cfg.Index[0] = 2;
            cfg.Index[1] = 0;
            cfg.Index[2] = 1;
            cfg.Min = d2;
            cfg.Max = d1;
        }
    }
    else
    {
        if (d2 <= d1)  // d2 <= d1 < d0
        {
            if (d2 != d1)
            {
                cfg.Map = M11;
                cfg.Index[0] = 2;
                cfg.Index[1] = 1;
                cfg.Index[2] = 0;
            }
            else
            {
                cfg.Map = M21;
                cfg.Index[0] = 1;
                cfg.Index[1] = 2;
                cfg.Index[2] = 0;
            }

            cfg.Min = d2;
            cfg.Max = d0;
        }
        else if (d2 <= d0)  // d1 < d2 <= d0
        {
            cfg.Map = (d2 != d0 ? M11 : M12);
            cfg.Index[0] = 1;
            cfg.Index[1] = 2;
            cfg.Index[2] = 0;
            cfg.Min = d1;
            cfg.Max = d0;
        }
        else  // d1 < d0 < d2
        {
            cfg.Map = M11;
            cfg.Index[0] = 1;
            cfg.Index[1] = 0;
            cfg.Index[2] = 2;
            cfg.Min = d1;
            cfg.Max = d2;
        }
    }
}

//----------------------------------------------------------------------------
template <typename Real>
bool Triangle2<Real>::NoIntersect(
    const Configuration& cfg0, const Configuration& cfg1, Real tmax,
    Real speed, int& side, Configuration& tcfg0, Configuration& tcfg1,
    Real& tfirst, Real& tlast)
{
    Real invSpeed, t;

    if (cfg1.Max < cfg0.Min)
    {
        // V1-interval initially on left of V0-interval.
        if (speed <= (Real)0)
        {
            // Intervals moving apart.
            return true;
        }

        // Update first time.
        invSpeed = ((Real)1)/speed;
        t = (cfg0.Min - cfg1.Max)*invSpeed;
        if (t > tfirst)
        {
            tfirst = t;
            side = -1;
            tcfg0 = cfg0;
            tcfg1 = cfg1;
        }

        // Test for exceedance of time interval.
        if (tfirst > tmax)
        {
            return true;
        }

        // Update last time.
        t = (cfg0.Max - cfg1.Min)*invSpeed;
        if (t < tlast)
        {
            tlast = t;
        }

        // Test for separation.
        if (tfirst > tlast)
        {
            return true;
        }
    }
    else if (cfg0.Max < cfg1.Min)
    {
        // V1-interval initially on right of V0-interval.
        if (speed >= (Real)0)
        {
            // Intervals moving apart.
            return true;
        }

        // Update first time.
        invSpeed = ((Real)1)/speed;
        t = (cfg0.Max - cfg1.Min)*invSpeed;
        if (t > tfirst)
        {
            tfirst = t;
            side = 1;
            tcfg0 = cfg0;
            tcfg1 = cfg1;
        }

        // Test for exceedance of time interval.
        if (tfirst > tmax)
        {
            return true;
        }

        // Update last time.
        t = (cfg0.Min - cfg1.Max)*invSpeed;
        if (t < tlast)
        {
            tlast = t;
        }

        // Test for separation.
        if (tfirst > tlast)
        {
            return true;
        }
    }
    else
    {
        // V0-interval and V1-interval initially overlap.
        if (speed > (Real)0)
        {
            // Update last time.
            invSpeed = ((Real)1)/speed;
            t = (cfg0.Max - cfg1.Min)*invSpeed;
            if (t < tlast)
            {
                tlast = t;
            }

            // Test for separation.
            if (tfirst > tlast)
            {
                return true;
            }
        }
        else if (speed < (Real)0)
        {
            // Update last time.
            invSpeed = ((Real)1)/speed;
            t = (cfg0.Min - cfg1.Max)*invSpeed;
            if (t < tlast)
            {
                tlast = t;
            }

            // Test for separation.
            if (tfirst > tlast)
            {
                return true;
            }
        }
    }

    return false;
}

//----------------------------------------------------------------------------
template <typename Real>
void Triangle2<Real>::GetIntersection(
    const Configuration& cfg0, const Configuration& cfg1, int side,
    const Vec<2,Real> V0[3], const Vec<2,Real> V1[3], int& quantity,
    Vec<2,Real> vertex[6])
{
    Vec<2,Real> edge, diff;
    const Vec<2,Real>* origin;
    Real invEdE, emin, emax;
    int i;

    if (side == 1)  // V1-interval contacts V0-interval on right.
    {
        if (cfg0.Map == M21 || cfg0.Map == M11)
        {
            quantity = 1;
            vertex[0] = V0[cfg0.Index[2]];
        }
        else if (cfg1.Map == M12 || cfg1.Map == M11)
        {
            quantity = 1;
            vertex[0] = V1[cfg1.Index[0]];
        }
        else  // cfg0.Map == M12 && cfg1.Map == M21 (edge overlap).
        {
            origin = &V0[cfg0.Index[1]];
            edge = V0[cfg0.Index[2]] - *origin;
            invEdE = ((Real)1) / (edge * edge);
            diff = V1[cfg1.Index[1]] - *origin;
            emin = (edge * diff) * invEdE;
            diff = V1[cfg1.Index[0]] - *origin;
            emax = (edge * diff)*invEdE;
            //assertion(emin <= emax, "Unexpected condition\n");
            Interval1Intersector<Real> intr((Real)0, (Real)1, emin, emax);
            quantity = intr.GetNumIntersections();
            //assertion(quantity > 0, "Unexpected condition\n");
            for (i = 0; i < quantity; ++i)
            {
                vertex[i] = *origin + intr.GetIntersection(i)*edge;
            }
        }
    }
    else if (side == -1)  // V1-interval contacts V0-interval on left.
    {
        if (cfg1.Map == M21 || cfg1.Map == M11)
        {
            quantity = 1;
            vertex[0] = V1[cfg1.Index[2]];
        }
        else if (cfg0.Map == M12 || cfg0.Map == M11)
        {
            quantity = 1;
            vertex[0] = V0[cfg0.Index[0]];
        }
        else  // cfg1.Map == M12 && cfg0.Map == M21 (edge overlap).
        {
            origin = &V1[cfg1.Index[1]];
            edge = V1[cfg1.Index[2]] - *origin;
            invEdE = ((Real)1) / (edge * edge);
            diff = V0[cfg0.Index[1]] - *origin;
            emin = (edge * diff) * invEdE;
            diff = V0[cfg0.Index[0]] - *origin;
            emax = (edge * diff)*invEdE;
            //assertion(emin <= emax, "Unexpected condition\n");
            Interval1Intersector<Real> intr((Real)0, (Real)1, emin, emax);
            quantity = intr.GetNumIntersections();
            //assertion(quantity > 0, "Unexpected condition\n");
            for (i = 0; i < quantity; ++i)
            {
                vertex[i] = *origin + intr.GetIntersection(i) * edge;
            }
        }
    }
    else  // Triangles were initially intersecting.
    {
        Triangle2<Real> tri0(V0), tri1(V1);
        Triangle2IntersectionResult<Real> tri2Res;
        //IntrTriangle2Triangle2 intr(tri0, tri1);
        tri0.Find(tri1, tri2Res);
        quantity = tri2Res.mQuantity;
        for (i = 0; i < quantity; ++i)
        {
            vertex[i] = tri2Res.mPoint[i];
        }
    }
}
