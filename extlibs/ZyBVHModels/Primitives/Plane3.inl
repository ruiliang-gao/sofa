#include "Plane3.h"

#include "Math/MathUtils.h"

using namespace BVHModels;

//----------------------------------------------------------------------------
template <typename Real>
Plane3<Real>::Plane3 ()
{
}

//----------------------------------------------------------------------------
template <typename Real>
Plane3<Real>::~Plane3 ()
{
}

//----------------------------------------------------------------------------
template <typename Real>
Plane3<Real>::Plane3 (const Vec<3,Real>& normal, Real constant)
    :
    Normal(normal),
    Constant(constant)
{
}

template <typename Real>
bool Plane3<Real>::IsIntersectionQuerySupported(const PrimitiveType &other)
{
    if (other == PT_PLANE ||
        other == PT_TRIANGLE3 ||
        other == PT_LINE3)
        return true;

    return false;
}

//----------------------------------------------------------------------------
template <typename Real>
Plane3<Real>::Plane3 (const Vec<3,Real>& normal, const Vec<3,Real>& p)
    :
    Normal(normal)
{
    //TRU Constant = normal.Dot(p);
    Constant = normal* p;
}

//----------------------------------------------------------------------------
template <typename Real>
Plane3<Real>::Plane3 (const Vec<3,Real>& p0, const Vec<3,Real>& p1,
    const Vec<3,Real>& p2)
{
    Vec<3,Real> edge1 = p1 - p0;
    Vec<3,Real> edge2 = p2 - p0;
    //TRU Normal = edge1.UnitCross(edge2);
    Normal = edge1.cross(edge2);
    Normal.normalize();
    //TRU Constant = Normal.Dot(p0);
    Constant = Normal * p0;
}

//----------------------------------------------------------------------------
template <typename Real>
Real Plane3<Real>::DistanceTo (const Vec<3,Real>& p) const
{
    //TRU return Normal.Dot(p) - Constant;
    return (Normal * p) - Constant;
}

//----------------------------------------------------------------------------
template <typename Real>
int Plane3<Real>::WhichSide (const Vec<3,Real>& p) const
{
    Real distance = DistanceTo(p);

    if (distance < (Real)0)
    {
        return -1;
    }
    else if (distance > (Real)0)
    {
        return +1;
    }
    else
    {
        return 0;
    }
}

//----------------------------------------------------------------------------
template <typename Real>
bool Plane3<Real>::Test(const Intersectable<Real, Vec<3, Real> > & other)
{
    if (!IsIntersectionQuerySupported(other.GetIntersectableType()))
        return false;

    if (other.GetIntersectableType() == PT_PLANE)
    {
        const Plane3<Real>* mPlane1 = dynamic_cast<const Plane3<Real>*>(&other);
        // If Cross(N0,N1) is zero, then either planes are parallel and separated
        // or the same plane.  In both cases, 'false' is returned.  Otherwise, the
        // planes intersect.  To avoid subtle differences in reporting between
        // Test() and Find(), the same parallel test is used.  Mathematically,
        //   |Cross(N0,N1)|^2 = Dot(N0,N0)*Dot(N1,N1)-Dot(N0,N1)^2
        //                    = 1 - Dot(N0,N1)^2
        // The last equality is true since planes are required to have unit-length
        // normal vectors.  The test |Cross(N0,N1)| = 0 is the same as
        // |Dot(N0,N1)| = 1.  I test the latter condition in Test() and Find().

        Real dot = this->Normal * mPlane1->Normal;
        return MathUtils<Real>::FAbs(dot) < (Real)1 - MathUtils<Real>::ZERO_TOLERANCE;
    }
    else if (other.GetIntersectableType() == PT_TRIANGLE3)
    {

    }

    return false;
}

//----------------------------------------------------------------------------
template <typename Real>
bool Plane3<Real>::Find(const Intersectable<Real, Vec<3, Real> > & other, IntersectionResult<Real>& result)
{
    if (!IsIntersectionQuerySupported(other.GetIntersectableType()))
        return false;

    if (other.GetIntersectableType() == PT_PLANE)
    {
        const Plane3<Real>* mPlane1 = dynamic_cast<const Plane3<Real>*>(&other);
        //TRU Plane3Plane3IntersectionResult<Real>* plane3Res = dynamic_cast<Plane3Plane3IntersectionResult<Real>*>(&result);
        Plane3Plane3IntersectionResult<Real>* plane3Res = static_cast<Plane3Plane3IntersectionResult<Real>*>(&result);

        if (!plane3Res)
            return false;

        // If N0 and N1 are parallel, either the planes are parallel and separated
        // or the same plane.  In both cases, 'false' is returned.  Otherwise,
        // the intersection line is
        //   L(t) = t*Cross(N0,N1)/|Cross(N0,N1)| + c0*N0 + c1*N1
        // for some coefficients c0 and c1 and for t any real number (the line
        // parameter).  Taking dot products with the normals,
        //   d0 = Dot(N0,L) = c0*Dot(N0,N0) + c1*Dot(N0,N1) = c0 + c1*d
        //   d1 = Dot(N1,L) = c0*Dot(N0,N1) + c1*Dot(N1,N1) = c0*d + c1
        // where d = Dot(N0,N1).  These are two equations in two unknowns.  The
        // solution is
        //   c0 = (d0 - d*d1)/det
        //   c1 = (d1 - d*d0)/det
        // where det = 1 - d^2.

        Real dot = this->Normal * mPlane1->Normal;
        if (MathUtils<Real>::FAbs(dot) >= (Real)1 - MathUtils<Real>::ZERO_TOLERANCE)
        {
            // The planes are parallel.  Check if they are coplanar.
            Real cDiff;
            if (dot >= (Real)0)
            {
                // Normals are in same direction, need to look at c0-c1.
                cDiff = this->Constant - mPlane1->Constant;
            }
            else
            {
                // Normals are in opposite directions, need to look at c0+c1.
                cDiff = this->Constant + mPlane1->Constant;
            }

            if (MathUtils<Real>::FAbs(cDiff) < MathUtils<Real>::ZERO_TOLERANCE)
            {
                // Planes are coplanar.
                //TRU plane3Res->mIntersectionType = IT_PLANE;
                plane3Res->SetMIntersectionType(IT_PLANE);
                // plane3Res->mIntrPlane = *this;
                return true;
            }

            // Planes are parallel, but distinct.
            //TRU plane3Res->mIntersectionType = IT_EMPTY;
            plane3Res->SetMIntersectionType(IT_EMPTY);
            return false;
        }

        Real invDet = ((Real)1)/((Real)1 - dot * dot);
        Real c0 = (this->Constant - dot * mPlane1->Constant) * invDet;
        Real c1 = (mPlane1->Constant - dot * this->Constant)*invDet;
        //TRU plane3Res->mIntersectionType = IT_LINE;
        plane3Res->SetMIntersectionType(IT_LINE);
        plane3Res->mIntrLine.Origin = c0 * this->Normal + c1 * mPlane1->Normal;
        plane3Res->mIntrLine.Direction = this->Normal.cross(mPlane1->Normal);
        plane3Res->mIntrLine.Direction.normalize();

        return true;
    }

    return false;
}
