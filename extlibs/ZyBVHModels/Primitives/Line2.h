#ifndef LINE2_H
#define LINE2_H

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

#include "Intersectable.h"

namespace BVHModels
{
    template <typename Real>
    class Line2Line2IntersectionResult: public IntersectionResult<Real>
    {
        public:
            Line2Line2IntersectionResult(): IntersectionResult<Real>()
            {
                this->intersectionType = IT_EMPTY;
                this->primitiveType1 = PT_LINE2;
                this->primitiveType2 = PT_LINE2;
            }

            Real mDotThreshold;

            int mQuantity;
            Vec<2,Real> mPoint;
    };

    template <typename Real>
    class Line2: public Intersectable<Real, Vec<2,Real> >
    {
        public:
            // The line is represented as P+t*D where P is the line origin, D is a
            // unit-length direction vector, and t is any real number.  The user must
            // ensure that D is indeed unit length.

            // Construction and destruction.
            Line2 ();  // uninitialized
            ~Line2 ();

            Line2 (const Vec<2,Real>& origin, const Vec<2,Real>& direction);

            PrimitiveType GetIntersectableType() const { return PT_LINE2; }
            bool IsIntersectionQuerySupported(const PrimitiveType &other);

            bool Test(const Intersectable<Real, Vec<2,Real> >& other);
            bool Find(const Intersectable<Real, Vec<2,Real> >& other, IntersectionResult<Real>& result);

            Vec<2,Real> Origin, Direction;

        private:
            IntersectionType Classify (const Vec<2,Real>& P0,
                const Vec<2,Real>& D0, const Vec<2,Real>& P1, const Vec<2,Real>& D1,
                Real dotThreshold, Real* s);
    };

    typedef Line2<float> Line2f;
    typedef Line2<double> Line2d;
}

#endif // LINE2_H
