#ifndef POINT3_H
#define POINT3_H

#include "DistanceComputable.h"

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

namespace BVHModels
{
    struct Point3Triangle3DistanceResult: public DistanceResult
    {
        public:
            Point3Triangle3DistanceResult(): DistanceResult()
            {
                distanceType = DS_EMPTY;
                primitiveType1 = PT_POINT3;
                primitiveType2 = PT_TRIANGLE3;
            }

            double mTriangleBary[3];
    };

    struct Point3Rectangle3DistanceResult: public DistanceResult
    {
        public:
            Point3Rectangle3DistanceResult(): DistanceResult()
            {
                distanceType = DS_EMPTY;
                primitiveType1 = PT_POINT3;
                primitiveType2 = PT_RECTANGLE3;
            }

            // closest0 = line.origin + param*line.direction
            double mLineParameter;

            // closest1 = rect.center + param0*rect.dir0 + param1*rect.dir1
            double mRectCoord[2];
    };

    template <typename Real>
    class Point3: public DistanceComputable<Real, Vec<3,Real> >
    {
        public:
            Point3();
            Point3(const Vec<3,Real>&);
            ~Point3();

            virtual Real GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result);
            virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result);

            virtual Real GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result);
            virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3,Real>& velocity0, const Vec<3,Real>& velocity1, DistanceResult& result);

            Vec<3, Real> Point;
    };

    typedef Point3<float> Point3f;
    typedef Point3<double> Point3d;
}
#endif // POINT3_H
