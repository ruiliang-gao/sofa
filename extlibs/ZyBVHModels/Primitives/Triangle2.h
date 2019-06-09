// Geometric Tools, LLC
// Copyright (c) 1998-2014
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.0 (2010/01/01)

#ifndef TRIANGLE2_H
#define TRIANGLE2_H

#include "Intersectable.h"

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

namespace BVHModels
{
	template <typename Real>
    class Triangle2IntersectionResult: public IntersectionResult<Real>
    {
        public:
            Triangle2IntersectionResult(): IntersectionResult<Real>()
            {
                this->intersectionType = IT_EMPTY;
                this->primitiveType1 = PT_TRIANGLE2;
                this->primitiveType2 = PT_TRIANGLE2;
            }

            int mQuantity;
            //hm Vec<3,Real> mPoint[2];
            Vec<2,Real> mPoint[2];
    };

    template <typename Real>
    class Triangle2: public Intersectable<Real, Vec<2,Real> >
    {
        public:
            // The triangle is represented as an array of three vertices:
            // V0, V1, and V2.

            // Construction and destruction.
            Triangle2 ();  // uninitialized
            ~Triangle2 ();

            Triangle2 (const Vec<2,Real>& v0, const Vec<2,Real>& v1,
                const Vec<2,Real>& v2);

            Triangle2 (const Vec<2,Real> vertex[3]);

			PrimitiveType GetIntersectableType() const { return PT_TRIANGLE2; }

            // Distance from the point Q to the triangle.  TODO:  Move this
            // to the physics library distance code.
            Real DistanceTo (const Vec<2,Real>& q) const;

            bool Test(const Intersectable<Real, Vec<2, Real> > &);
            bool Find(const Intersectable<Real, Vec<2, Real> > &, IntersectionResult<Real> &);

            Vec<2,Real> V[3];

        private:
            static int WhichSide (const Vec<2,Real> V[3], const Vec<2,Real>& P,
                    const Vec<2,Real>& D);

            static void ClipConvexPolygonAgainstLine (const Vec<2,Real>& N,
                Real c, int& quantity, Vec<2,Real> V[6]);

            enum ProjectionMap
            {
                M21,  // 2 vertices map to min, 1 vertex maps to max
                M12,  // 1 vertex maps to min, 2 vertices map to max
                M11   // 1 vertex maps to min, 1 vertex maps to max
            };

            class Configuration
            {
            public:
                ProjectionMap Map;  // how vertices map to the projection interval
                int Index[3];       // the sorted indices of the vertices
                Real Min, Max;      // the interval is [min,max]
            };

            void ComputeTwo (Configuration& cfg, const Vec<2,Real> V[3],
                const Vec<2,Real>& D, int i0, int i1, int i2);

            void ComputeThree (Configuration& cfg, const Vec<2,Real> V[3],
                const Vec<2,Real>& D, const Vec<2,Real>& P);

            static bool NoIntersect (const Configuration& cfg0,
                const Configuration& cfg1, Real tmax, Real speed, int& side,
                Configuration& tcfg0, Configuration& tcfg1, Real& tfirst,
                Real& tlast);

            static void GetIntersection (const Configuration& cfg0,
                const Configuration& cfg1, int side, const Vec<2,Real> V0[3],
                const Vec<2,Real> V1[3], int& quantity, Vec<2,Real> vertex[6]);
    };

    typedef Triangle2<float> Triangle2f;
    typedef Triangle2<double> Triangle2d;
}

#endif
