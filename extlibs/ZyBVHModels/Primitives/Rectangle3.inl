#ifndef RECTANGLE3_INL
#define RECTANGLE3_INL

#include "Math/MathUtils.h"
#include "Rectangle3.h"

#include "Segment3.h"

using namespace BVHModels;

//----------------------------------------------------------------------------
template <typename Real>
Rectangle3<Real>::Rectangle3 ()
{

}
//----------------------------------------------------------------------------
template <typename Real>
Rectangle3<Real>::~Rectangle3 ()
{

}
//----------------------------------------------------------------------------
template <typename Real>
Rectangle3<Real>::Rectangle3 (const Vec<3,Real>& center,
    const Vec<3, Real> axis[2], const Real extent[2])
    :
    Center(center)
{
    Axis[0] = axis[0];
    Axis[1] = axis[1];
    Extent[0] = extent[0];
    Extent[1] = extent[1];
}
//----------------------------------------------------------------------------
template <typename Real>
Rectangle3<Real>::Rectangle3 (const Vec<3,Real>& center,
    const Vec<3,Real>& axis0, const Vec<3,Real>& axis1, Real extent0,
    Real extent1)
    :
    Center(center)
{
    Axis[0] = axis0;
    Axis[1] = axis1;
    Extent[0] = extent0;
    Extent[1] = extent1;
}
//----------------------------------------------------------------------------
template <typename Real>
void Rectangle3<Real>::ComputeVertices (Vec<3,Real> vertex[4]) const
{
    Vec<3,Real> extAxis0 = Axis[0]*Extent[0];
    Vec<3,Real> extAxis1 = Axis[1]*Extent[1];

    vertex[0] = Center - extAxis0 - extAxis1;
    vertex[1] = Center + extAxis0 - extAxis1;
    vertex[2] = Center + extAxis0 + extAxis1;
    vertex[3] = Center - extAxis0 + extAxis1;
}
//----------------------------------------------------------------------------
template <typename Real>
Vec<3,Real> Rectangle3<Real>::GetPPCorner () const
{
    return Center + Extent[0] * Axis[0] + Extent[1] * Axis[1];
}
//----------------------------------------------------------------------------
template <typename Real>
Vec<3,Real> Rectangle3<Real>::GetPMCorner () const
{
    return Center + Extent[0] * Axis[0] - Extent[1] * Axis[1];
}
//----------------------------------------------------------------------------
template <typename Real>
Vec<3,Real> Rectangle3<Real>::GetMPCorner () const
{
    return Center - Extent[0] * Axis[0] + Extent[1] * Axis[1];
}
//----------------------------------------------------------------------------
template <typename Real>
Vec<3,Real> Rectangle3<Real>::GetMMCorner () const
{
    return Center - Extent[0]*Axis[0] - Extent[1]*Axis[1];
}
//----------------------------------------------------------------------------

template <typename Real>
Real Rectangle3<Real>::GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result)
{
    return MathUtils<Real>::Sqrt(GetSquaredDistance(other, result));
}
//----------------------------------------------------------------------------
template <typename Real>
Real Rectangle3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result)
{
    const Rectangle3<Real>* mRectangle = dynamic_cast<const Rectangle3<Real>*>(&other);
    if (mRectangle)
    {
        // Compare edges of rectangle0 to the interior of rectangle1.
        Real sqrDist = MathUtils<Real>::MAX_REAL, sqrDistTmp;
        Segment3<Real> edge;
        int i0, i1;
        for (i1 = 0; i1 < 2; ++i1)
        {
            for (i0 = -1; i0 <= 1; i0 += 2)
            {
                edge.Center = this->Center +
                    (i0 * this->Extent[1-i1]) *
                    this->Axis[1-i1];
                edge.Direction = this->Axis[i1];
                edge.Extent = this->Extent[i1];
                edge.ComputeEndPoints();

                Segment3Rectangle3DistanceResult segRectRes;
                sqrDistTmp = edge.GetSquaredDistance(*mRectangle, segRectRes);
                if (sqrDistTmp < sqrDist)
                {
                    result.mClosestPoint0 = segRectRes.GetClosestPoint0();
                    result.mClosestPoint1 = segRectRes.GetClosestPoint1();
                    sqrDist = sqrDistTmp;
                }
            }
        }

        // Compare edges of rectangle1 to the interior of rectangle0.
        for (i1 = 0; i1 < 2; ++i1)
        {
            for (i0 = -1; i0 <= 1; i0 += 2)
            {
                edge.Center = mRectangle->Center +
                    (i0 * mRectangle->Extent[1-i1]) *
                    mRectangle->Axis[1-i1];
                edge.Direction = mRectangle->Axis[i1];
                edge.Extent = mRectangle->Extent[i1];
                edge.ComputeEndPoints();

                Segment3Rectangle3DistanceResult segRectRes;
                sqrDistTmp = edge.GetSquaredDistance(*this, segRectRes);

                if (sqrDistTmp < sqrDist)
                {
                    result.mClosestPoint0 = segRectRes.GetClosestPoint0();
                    result.mClosestPoint1 = segRectRes.GetClosestPoint1();
                    sqrDist = sqrDistTmp;
                }
            }
        }

        return sqrDist;
    }
    return MathUtils<Real>::MAX_REAL;
}
//----------------------------------------------------------------------------
template <typename Real>
Real Rectangle3<Real>::GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3, Real>& velocity0, const Vec<3, Real>& velocity1, DistanceResult& result)
{
	/*Vector3<Real> movedCenter0 = mRectangle0->Center + t*velocity0;
    Vector3<Real> movedCenter1 = mRectangle1->Center + t*velocity1;
    Rectangle3<Real> movedRect0(movedCenter0, mRectangle0->Axis,
        mRectangle0->Extent);
    Rectangle3<Real> movedRect1(movedCenter1, mRectangle1->Axis,
        mRectangle1->Extent);
    return DistRectangle3Rectangle3<Real>(movedRect0,movedRect1).Get();*/
	return MathUtils<Real>::MAX_REAL;
}
//----------------------------------------------------------------------------
template <typename Real>
Real Rectangle3<Real>::GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3, Real>& velocity0, const Vec<3, Real>& velocity1, DistanceResult& result)
{
    /*Vec<3,Real> movedCenter0 = mRectangle0->Center + t*velocity0;
    Vec<3,Real> movedCenter1 = mRectangle1->Center + t*velocity1;
    Rectangle3<Real> movedRect0(movedCenter0, mRectangle0->Axis,
        mRectangle0->Extent);
    Rectangle3<Real> movedRect1(movedCenter1, mRectangle1->Axis,
        mRectangle1->Extent);
    return DistRectangle3Rectangle3<Real>(movedRect0,movedRect1).GetSquared();*/
	return MathUtils<Real>::MAX_REAL;
}
//----------------------------------------------------------------------------

#endif // RECTANGLE3_INL
