// Geometric Tools, LLC
// Copyright (c) 1998-2014
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.0 (2010/01/01)

#ifndef TRIANGLE3_H
#define TRIANGLE3_H

#include "Intersectable.h"

#include "Primitives/Plane3.h"
#include "Primitives/Line3.h"

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

namespace BVHModels
{
	template <typename Real>
    class Triangle3IntersectionResult: public IntersectionResult<Real>
    {
        public:
            Triangle3IntersectionResult(): IntersectionResult<Real>()
            {
                this->intersectionType = IT_EMPTY;
                this->primitiveType1 = PT_TRIANGLE3;
                this->primitiveType2 = PT_TRIANGLE3;
                this->mReportCoplanarIntersections = true;
            }

            bool mReportCoplanarIntersections;
            int mQuantity;
            Vec<3,Real> mPoint[6];
    };


    template <typename Real>
    //hm class Triangle3: public Intersectable<Real, Vec<3,Real> >
    class Triangle3: public Intersectable<Real, Vec<3,Real> >, public DistanceComputable<Real, Vec<3,Real> >
    {
        public:
            // The triangle is represented as an array of three vertices:
            // V0, V1, and V2.

            // Construction and destruction.
            Triangle3 ();  // uninitialized
            ~Triangle3 ();

            Triangle3 (const Vec<3,Real>& v0, const Vec<3,Real>& v1,
                const Vec<3,Real>& v2);

            Triangle3 (const Vec<3,Real> vertex[3]);

			Triangle3(const Triangle3& other)
			{
				if (this != &other)
				{
					V[0] = other.V[0];
					V[1] = other.V[1];
					V[2] = other.V[2];
				}
			}

			Triangle3& operator=(const Triangle3& other)
			{
				if (this != &other)
				{
					V[0] = other.V[0];
					V[1] = other.V[1];
					V[2] = other.V[2];
				}
				return *this;
			}

			PrimitiveType GetIntersectableType() const { return PT_TRIANGLE3; }

			void draw();

			bool ContainsPoint(const Vec<3, Real>& point);

            // Distance from the point Q to the triangle.  TODO:  Move this
            // to the physics library distance code.
            Real DistanceTo (const Vec<3,Real>& q) const;

            bool Test(const Intersectable<Real, Vec<3, Real> > &);
            bool Find(const Intersectable<Real, Vec<3, Real> > &, IntersectionResult<Real>&);

            Vec<3,Real> V[3];

            //-----------------------
            //-----------------------

            virtual Real GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result);     // distance
            virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result);  // squared distance

            virtual Real GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3, Real>& velocity0, const Vec<3, Real>& velocity1, DistanceResult& result);
            virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3, Real>& velocity0, const Vec<3, Real>& velocity1, DistanceResult& result);

            //-----------------------
            //-----------------------

        private:

            enum ProjectionMap
                {
                    M2, M11,                // lines
                    M3, M21, M12, M111,     // triangles
                    M44, M2_2, M1_1         // boxes
                };

                enum ContactSide
                {
                    CS_LEFT,
                    CS_RIGHT,
                    CS_NONE
                };

                class Configuration
                {
                public:
                    ProjectionMap mMap;  // how vertices map to the projection interval
                    int mIndex[8];       // the sorted indices of the vertices
                    Real mMin, mMax;      // the interval is [min,max]
                };

            static void ProjectOntoAxis (const Triangle3<Real>& triangle,
                    const Vec<3,Real>& axis, Real& fmin, Real& fmax);

            static void ProjectOntoAxis (const Triangle3<Real>& triangle,
                    const Vec<3,Real>& axis, Configuration& cfg);

            bool FindOverlap (Real tmax, Real speed, const Configuration& UC,
                    const Configuration& VC, ContactSide& side, Configuration& TUC,
                    Configuration& TVC, Real& tfirst, Real& tlast);

            bool FindOverlap (const Triangle3<Real>& tri2, const Vec<3,Real>& axis, Real tmax,
                const Vec<3,Real>& velocity, ContactSide& side,
                Configuration& tcfg0, Configuration& tcfg1, Real& tfirst,
                Real& tlast);

            static void TrianglePlaneRelations (const Triangle3<Real>& triangle,
                const Plane3<Real>& plane, Real distance[3], int sign[3],
                int& positive, int& negative, int& zero);

            static void GetInterval (const Triangle3<Real>& triangle,
                const Line3<Real>& line, const Real distance[3],
                const int sign[3], Real param[2]);

            bool ContainsPoint (const Triangle3<Real>& triangle, const Plane3<Real>& plane, const Vec<3, Real>& point, Triangle3IntersectionResult<Real>& result);

            bool IntersectsSegment (const Plane3<Real>& plane,
                const Triangle3<Real>& triangle, const Vec<3,Real>& end0,
				const Vec<3, Real>& end1, Triangle3IntersectionResult<Real>& result);

            bool GetCoplanarIntersection (const Plane3<Real>& plane,
                const Triangle3<Real>& tri0, const Triangle3<Real>& tri1, Triangle3IntersectionResult<Real>& result);

            static bool TestOverlap (Real tmax, Real speed, Real umin,
                Real umax, Real vmin, Real vmax, Real& tfirst, Real& tlast);

            bool TestOverlap (const Intersectable<Real, Vec<3,Real> >& intersectable, const Vec<3,Real>& axis, Real tmax,
                const Vec<3,Real>& velocity, Real& tfirst, Real& tlast);

            void FindContactSet (const Triangle3<Real>& tri0,
                    const Triangle3<Real>& tri1, ContactSide& side,
					Configuration& cfg0, Configuration& cfg1, Triangle3IntersectionResult<Real>& result);

            void GetEdgeEdgeIntersection (const Vec<3,Real>& U0,
                const Vec<3,Real>& U1, const Vec<3,Real>& V0,
				const Vec<3, Real>& V1, Triangle3IntersectionResult<Real>& result);

            void GetEdgeFaceIntersection (const Vec<3,Real>& U0,
                const Vec<3, Real>& U1, const Triangle3<Real>& tri, Triangle3IntersectionResult<Real>& result);
    };

    typedef Triangle3IntersectionResult<float> Triangle3IntersectionResultf;

    typedef Triangle3<float> Triangle3f;
    typedef Triangle3<double> Triangle3d;
}

#endif
