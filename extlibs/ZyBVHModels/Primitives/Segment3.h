#ifndef SEGMENT3_H
#define SEGMENT3_H

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

#include "DistanceComputable.h"

namespace BVHModels
{
    struct Segment3Triangle3DistanceResult: public DistanceResult
    {
        public:

            Segment3Triangle3DistanceResult(): DistanceResult()
            {
                distanceType = DS_EMPTY;
                primitiveType1 = PT_SEGMENT3;
                primitiveType2 = PT_TRIANGLE3;
            }

            double mSegmentParameter;
            double mTriangleBary[3];
    };

    struct Segment3Rectangle3DistanceResult: public DistanceResult
    {
        public:

            Segment3Rectangle3DistanceResult(): DistanceResult()
            {
                distanceType = DS_EMPTY;
                primitiveType1 = PT_SEGMENT3;
                primitiveType2 = PT_RECTANGLE3;
            }

            // Information about the closest points.
            // closest0 = seg.origin + param * seg.direction
            double mSegmentParameter;

            // closest1 = rect.center + param0 * rect.dir0 + param1 * rect.dir1
            double mRectCoord[2];
    };

    struct Segment3Segment3DistanceResult: public DistanceResult
    {
        public:

            Segment3Segment3DistanceResult(): DistanceResult()
            {
                distanceType = DS_EMPTY;
                primitiveType1 = PT_SEGMENT3;
                primitiveType2 = PT_SEGMENT3;
            }

            using DistanceResult::mClosestPoint0;
            using DistanceResult::mClosestPoint1;

            // Information about the closest points.
            double mSegment0Parameter;  // closest0 = seg0.origin+param*seg0.direction
            double mSegment1Parameter;  // closest1 = seg1.origin+param*seg1.direction
    };

    template <typename Real>
    class Segment3: public DistanceComputable<Real, Vec<3,Real> >
    {
    public:
        // The segment is represented as (1-s)*P0+s*P1, where P0 and P1 are the
        // endpoints of the segment and 0 <= s <= 1.
        //
        // Some algorithms involving segments might prefer a centered
        // representation similar to how oriented bounding boxes are defined.
        // This representation is C+t*D, where C = (P0+P1)/2 is the center of
        // the segment, D = (P1-P0)/Length(P1-P0) is a unit-length direction
        // vector for the segment, and |t| <= e.  The value e = Length(P1-P0)/2
        // is the 'extent' (or radius or half-length) of the segment.

        // Construction and destruction.
        Segment3 ();  // uninitialized
        ~Segment3 ();

        // The constructor computes C, D, and E from P0 and P1.
        Segment3 (const Vec<3,Real>& p0, const Vec<3,Real>& p1);

        // The constructor computes P0 and P1 from C, D, and E.
        Segment3 (const Vec<3,Real>& center, const Vec<3,Real>& direction, Real extent);

        int GetIntersectableType() const { return PT_SEGMENT3; }

        // Call this function when you change P0 or P1.
        void ComputeCenterDirectionExtent();

        // Call this function when you change C, D, or e.
        void ComputeEndPoints();

        Real Length() const { return (P1 - P0).norm(); }
        Real SquaredLength() const { return (P1 - P0).norm2(); }

		Vec<3, Real> ProjectOnSegment(const Vec<3, Real>& point);

        virtual Real GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result);
        virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result);

        virtual Real GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result);
        virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result);

        // End-point representation.
        Vec<3,Real> P0, P1;

        // Center-direction-extent representation.
        Vec<3,Real> Center;
        Vec<3,Real> Direction;
        Real Extent;
    };

    typedef Segment3<float> Segment3f;
    typedef Segment3<double> Segment3d;
}

#endif // SEGMENT_H
