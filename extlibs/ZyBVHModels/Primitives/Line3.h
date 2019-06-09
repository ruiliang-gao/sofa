#ifndef LINE3_H
#define LINE3_H

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

#include "DistanceComputable.h"

namespace BVHModels
{
    struct Line3Triangle3DistanceResult: public DistanceResult
    {
        public:
            Line3Triangle3DistanceResult(): DistanceResult()
            {
                distanceType = DS_EMPTY;
                primitiveType1 = PT_LINE3;
                primitiveType2 = PT_TRIANGLE3;
            }

            // closest0 = line.origin + param*line.direction
            double mLineParameter;

            // closest1 = sum_{i=0}^2 bary[i]*tri.vertex[i]
            double mTriangleBary[3];
    };

    struct Line3Segment3DistanceResult: public DistanceResult
    {
        public:
            Line3Segment3DistanceResult(): DistanceResult()
            {
                distanceType = DS_EMPTY;
                primitiveType1 = PT_LINE3;
                primitiveType2 = PT_SEGMENT3;
            }

            double mLineParameter;  // closest0 = line.origin+param*line.direction
            double mSegmentParameter;  // closest1 = seg.origin+param*seg.direction
    };

    struct Line3Rectangle3DistanceResult: public DistanceResult
    {
        public:
            Line3Rectangle3DistanceResult(): DistanceResult()
            {
                distanceType = DS_EMPTY;
                primitiveType1 = PT_LINE3;
                primitiveType2 = PT_RECTANGLE3;
            }

            // Information about the closest points.
            // closest0 = line.origin + param*line.direction
            double mLineParameter;
            // closest1 = rect.center + param0*rect.dir0 + param1*rect.dir1
            double mRectCoord[2];
    };

    template <typename Real>
    class Line3: public Intersectable<Real, Vec<3,Real> >, public DistanceComputable<Real, Vec<3,Real> >
    {
    public:
        // The line is represented as P+t*D where P is the line origin, D is a
        // unit-length direction vector, and t is any real number.  The user must
        // ensure that D is indeed unit length.

        // Construction and destruction.
        Line3 ();  // uninitialized
        ~Line3 ();

        Line3 (const Vec<3,Real>& origin, const Vec<3,Real>& direction);

        PrimitiveType GetIntersectableType() const { return PT_LINE3; }

        virtual Real GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result);
        virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result);

        virtual Real GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result);
        virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result);

        Vec<3,Real> Origin, Direction;
    };

    typedef Line3<float> Line3f;
    typedef Line3<double> Line3d;
}

#endif // LINE3_H
