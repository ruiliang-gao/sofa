#include "Utils_2D.h"

#include <Math/MathUtils.h>

using namespace BVHModels;

//----------------------------------------------------------------------------
template <typename Real>
void Utils_2D<Real>::TriangleLineRelations (
    const Vec<2,Real>& origin, const Vec<2,Real>& direction,
    const Triangle2<Real>& triangle, Real dist[3], int sign[3],
    int& positive, int& negative, int& zero)
{
    positive = 0;
    negative = 0;
    zero = 0;
    for (int i = 0; i < 3; ++i)
    {
        Vec<2,Real> diff = triangle.V[i] - origin;
        //TRU dist[i] = diff.DotPerp(direction);
        dist[i] = (diff[0]*direction[1] - diff[1]*direction[0]);
        if (dist[i] > MathUtils<Real>::ZERO_TOLERANCE)
        {
            sign[i] = 1;
            ++positive;
        }
        else if (dist[i] < -MathUtils<Real>::ZERO_TOLERANCE)
        {
            sign[i] = -1;
            ++negative;
        }
        else
        {
            dist[i] = (Real)0;
            sign[i] = 0;
            ++zero;
        }
    }
}
//----------------------------------------------------------------------------
template <typename Real>
void Utils_2D<Real>::GetInterval (const Vec<2,Real>& origin,
    const Vec<2,Real>& direction, const Triangle2<Real>& triangle,
    const Real dist[3], const int sign[3], Real param[2])
{
    // Project triangle onto line.
    Real proj[3];
    int i;
    for (i = 0; i < 3; ++i)
    {
        Vec<2,Real> diff = triangle.V[i] - origin;
        proj[i] = direction * diff;
    }

    // Compute transverse intersections of triangle edges with line.
    Real numer, denom;
    int i0, i1, i2;
    int quantity = 0;
    for (i0 = 2, i1 = 0; i1 < 3; i0 = i1++)
    {
        if (sign[i0]*sign[i1] < 0)
        {
            // assertion(quantity < 2, "Too many intersections\n");
            numer = dist[i0]*proj[i1] - dist[i1]*proj[i0];
            denom = dist[i0] - dist[i1];
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
                // assertion(quantity < 2, "Too many intersections\n");
                param[quantity++] = proj[i2];
            }
        }
    }

    // Sort.
    // assertion(quantity >= 1, "Need at least one intersection\n");
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
