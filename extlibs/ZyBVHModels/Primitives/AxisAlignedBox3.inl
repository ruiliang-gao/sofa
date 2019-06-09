// Geometric Tools, LLC
// Copyright (c) 1998-2014
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.0 (2010/01/01)

#include "AxisAlignedBox3.h"

using namespace BVHModels;

//----------------------------------------------------------------------------
template <typename Real>
AxisAlignedBox3<Real>::AxisAlignedBox3 ()
{
}

//----------------------------------------------------------------------------
template <typename Real>
AxisAlignedBox3<Real>::~AxisAlignedBox3 ()
{
}

//----------------------------------------------------------------------------
template <typename Real>
AxisAlignedBox3<Real>::AxisAlignedBox3 (Real xmin, Real xmax, Real ymin,
    Real ymax, Real zmin, Real zmax)
{
    Min[0] = xmin;
    Max[0] = xmax;
    Min[1] = ymin;
    Max[1] = ymax;
    Min[2] = zmin;
    Max[2] = zmax;
}

//----------------------------------------------------------------------------
template <typename Real>
bool AxisAlignedBox3<Real>::IsIntersectionQuerySupported(const PrimitiveType &other) const
{
    if (other == PT_AABB)
        return true;

    return false;
}

//----------------------------------------------------------------------------
template <typename Real>
//TRU void AxisAlignedBox3<Real>::GetCenterExtents (Vec<3,Real>& center,    Vec<3, Real> extent[3])
void AxisAlignedBox3<Real>::GetCenterExtents (Vec<3,Real>& center,
    Real extent[3])
{
    center[0] = ((Real)0.5)*(Max[0] + Min[0]);
    center[1] = ((Real)0.5)*(Max[1] + Min[1]);
    center[2] = ((Real)0.5)*(Max[2] + Min[2]);
    extent[0] = ((Real)0.5)*(Max[0] - Min[0]);
    extent[1] = ((Real)0.5)*(Max[1] - Min[1]);
    extent[2] = ((Real)0.5)*(Max[2] - Min[2]);
}

//----------------------------------------------------------------------------
template <typename Real>
bool AxisAlignedBox3<Real>::HasXOverlap (const AxisAlignedBox3& box) const
{
    return (Max[0] >= box.Min[0] && Min[0] <= box.Max[0]);
}

//----------------------------------------------------------------------------
template <typename Real>
bool AxisAlignedBox3<Real>::HasYOverlap (const AxisAlignedBox3& box) const
{
    return (Max[1] >= box.Min[1] && Min[1] <= box.Max[1]);
}

//----------------------------------------------------------------------------
template <typename Real>
bool AxisAlignedBox3<Real>::HasZOverlap (const AxisAlignedBox3& box) const
{
    return (Max[2] >= box.Min[2] && Min[2] <= box.Max[2]);
}

//----------------------------------------------------------------------------
template <typename Real>
bool AxisAlignedBox3<Real>::Test (const Intersectable<Real, Vec<3,Real> >& box) const
{
    if (!IsIntersectionQuerySupported(box.GetIntersectableType()))
        return false;

    if (box.GetIntersectableType() == PT_AABB)
    {
        //TRU AxisAlignedBox3<Real>* aabb = static_cast<AxisAlignedBox3<Real>*>(&box);
        const AxisAlignedBox3<Real>* aabb = dynamic_cast<const AxisAlignedBox3<Real>*>(&box);
        if (aabb)
        {
            for (int i = 0; i < 3; i++)
            {
                if (Max[i] < aabb->Min[i] || Min[i] > aabb->Max[i])
                {
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}
//----------------------------------------------------------------------------
template <typename Real>
bool AxisAlignedBox3<Real>::Find (const Intersectable<Real, Vec<3,Real> >&  box, IntersectionResult<Real>& result) const
{
    if (!IsIntersectionQuerySupported(box.GetIntersectableType()))
        return false;

    if (box.GetIntersectableType() == PT_AABB)
    {
        //TRU AxisAlignedBox3<Real>* aabb = static_cast<AxisAlignedBox3<Real>*>(&box);
        const AxisAlignedBox3<Real>* aabb = dynamic_cast<const AxisAlignedBox3<Real>*>(&box);
        //TRU AABBIntersectionResult<Real>* intersection = dynamic_cast<AABBIntersectionResult<Real>*>(&result);
        AABBIntersectionResult<Real>* intersection = static_cast<AABBIntersectionResult<Real>*>(&result);

        if (!intersection)
            return false;

        int i;
        for (i = 0; i < 3; i++)
        {
            if (Max[i] < aabb->Min[i] || Min[i] > aabb->Max[i])
            {
                //TRU intersection->intersectionType = IT_EMPTY;
                intersection->SetIntersectionType(IT_EMPTY);
                return false;
            }
        }

        for (i = 0; i < 3; i++)
        {
            if (Max[i] <= aabb->Max[i])
            {
                intersection->Max[i] = Max[i];
            }
            else
            {
                intersection->Max[i] = aabb->Max[i];
            }

            if (Min[i] <= aabb->Min[i])
            {
                intersection->Min[i] = aabb->Min[i];
            }
            else
            {
                intersection->Min[i] = Min[i];
            }
        }
        //TRU intersection->intersectionType = IT_AABB;
        intersection->SetIntersectionType(IT_AABB);
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------
