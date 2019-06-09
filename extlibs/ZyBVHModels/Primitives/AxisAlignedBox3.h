// Geometric Tools, LLC
// Copyright (c) 1998-2014
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.0 (2010/01/01)

#ifndef AXISALIGNEDBOX3_H
#define AXISALIGNEDBOX3_H

#include <sofa/defaulttype/Vec.h>

#include "Intersectable.h"

using namespace sofa::defaulttype;

namespace BVHModels
{
	template <typename Real>
	class AABBIntersectionResult : public IntersectionResult<Real>
    {
        public:
            AABBIntersectionResult() //: IntersectionResult()
            {
                this->intersectionType = IT_EMPTY;
                this->primitiveType1 = PT_AABB;
                this->primitiveType2 = PT_AABB;
            }

            Vec<3,Real> Min, Max;
    };

	template <typename Real>
    //TRU class AxisAlignedBox3
    class AxisAlignedBox3 : public Intersectable<Real, Vec<3, Real> >
    {
    public:
        // Construction and destruction.
        AxisAlignedBox3 ();  // uninitialized
        ~AxisAlignedBox3 ();

        // The caller must ensure that xmin <= xmax, ymin <= ymax, and
        // zmin <= zmax.
        AxisAlignedBox3 (Real xmin, Real xmax, Real ymin, Real ymax,
            Real zmin, Real zmax);

        bool IsIntersectionQuerySupported(const PrimitiveType &other) const;

        // Compute the center of the box and the extents (half-lengths)
        // of the box edges.
        //TRU void GetCenterExtents (Vec<3,Real>& center, Vec<3,Real> extent[3]);
        void GetCenterExtents (Vec<3,Real>& center, Real extent[3]);

        // Overlap testing is in the strict sense.  If the two boxes are just
        // touching along a common edge or a common face, the boxes are reported
        // as overlapping.
        bool HasXOverlap (const AxisAlignedBox3& box) const;
        bool HasYOverlap (const AxisAlignedBox3& box) const;
        bool HasZOverlap (const AxisAlignedBox3& box) const;
        bool Test(const Intersectable<Real, Vec<3,Real> >& box) const;

        // The return value is 'true' if there is overlap.  In this case the
        // intersection is stored in 'intersection'.  If the return value is
        // 'false', there is no overlap.  In this case 'intersection' is
        // undefined.
        bool Find(const Intersectable<Real, Vec<3,Real> >& box, IntersectionResult<Real>& intersection) const;

        //TRU Vec<3,Real> Min[3], Max[3];
        Real Min[3], Max[3];

        // AABBIntersectionResult getIntersectionResult();
    };

    typedef AxisAlignedBox3<float> AxisAlignedBox3f;
    typedef AxisAlignedBox3<double> AxisAlignedBox3d;

}

#endif
