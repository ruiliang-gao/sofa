// Geometric Tools, LLC
// Copyright (c) 1998-2014
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.0 (2010/01/01)

#ifndef ORIENTEDBOX3_H
#define ORIENTEDBOX3_H

#include <sofa/defaulttype/Vec.h>

#include "Intersectable.h"

using namespace sofa::defaulttype;

namespace BVHModels
{
	template <typename Real>
	class OBBIntersectionResult : public IntersectionResult<Real>
    {
        public:
            OBBIntersectionResult(): IntersectionResult<Real>()
            {
                this->intersectionType = IT_EMPTY;
                this->primitiveType1 = PT_OBB;
                this->primitiveType2 = PT_OBB;
            }

            int mQuantity;
			Real mContactTime;
            Vec<3,Real> mPoint[8];
    };

	template <typename Real> 
	class Box3 : public Intersectable<Real, Vec<3, Real> >
	{
		public:
			// A box has center C, axis directions U[0], U[1], and U[2] (mutually
			// perpendicular unit-length vectors), and extents e[0], e[1], and e[2]
			// (all nonnegative numbers).  A point X = C+y[0]*U[0]+y[1]*U[1]+y[2]*U[2]
			// is inside or on the box whenever |y[i]| <= e[i] for all i.

			// Construction and destruction.
			Box3() 
			{
		
			} // uninitialized
		
			~Box3()
			{

			}

            //TRU Box3 (const Vec<3,Real>& center, const Vec<3,Real> axis[3],                const Vec<3, Real> extent[3]);
            Box3 (const Vec<3,Real>& center, const Vec<3,Real> axis[3],
                const Real extent[3]);

            Box3 (const Vec<3,Real>& center, const Vec<3,Real>& axis0,
                const Vec<3,Real>& axis1, const Vec<3,Real>& axis2,
                const Real extent0, const Real extent1, const Real extent2);

            void computeVertices(Vec<3,Real> vertex[8]) const;

            bool Test(const Intersectable<Real, Vec<3,Real> >& box);

            bool Test(const Intersectable<Real, Vec<3,Real> >&, Real tmax, const Vec<3,Real>& velocity0,
                      const Vec<3,Real>& velocity1);
            bool Find(const Intersectable<Real, Vec<3,Real> >& box, Real tmax, const Vec<3,Real>& velocity0,
                const Vec<3,Real>& velocity1, IntersectionResult<Real>& result);

            bool Test (const Intersectable<Real, Vec<3,Real> >& box, Real tmax, int numSteps,
                const Vec<3,Real>& velocity0, const Vec<3,Real>& rotCenter0,
                const Vec<3,Real>& rotAxis0, const Vec<3,Real>& velocity1,
                const Vec<3,Real>& rotCenter1, const Vec<3,Real>& rotAxis1);

            PrimitiveType GetIntersectableType() const { return PT_OBB; }

            bool IsIntersectionQuerySupported(const PrimitiveType &other);

            Vec<3,Real> Center;
            Vec<3,Real> Axis[3];
            Real Extent[3];

            // Real mContactTime;

        private:
			bool IsSeparated(Real min0, Real max0, Real min1,
				Real max1, Real speed, Real tmax, Real& tlast);

            bool IsSeparated (Real min0, Real max0, Real min1,
                Real max1, Real speed, Real tmax, Real& tlast, OBBIntersectionResult<Real>& result);
    };

    typedef Box3<float> Box3f;
    typedef Box3<double> Box3d;
}

#ifdef _WIN32
#include "Box3.inl"
#endif

#endif
