#ifndef SEGMENT2_H
#define SEGMENT2_H

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

#include "Intersectable.h"

namespace BVHModels
{
	template <typename Real>
    class Segment2IntersectionResult: public IntersectionResult<Real>
    {
        public:
            Segment2IntersectionResult(): IntersectionResult<Real>()
            {
                this->primitiveType1 = PT_SEGMENT2;
                this->primitiveType2 = PT_NONE;
            }

            int mQuantity;
            Vec<2,Real> mPoint[2];
    };

    template <typename Real>
    class Segment2: public Intersectable<Real, Vec<2,Real> >
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
            Segment2 ();  // uninitialized
            ~Segment2 ();

            // The constructor computes C, D, and E from P0 and P1.
            Segment2 (const Vec<2,Real>& p0, const Vec<2,Real>& p1);

            // The constructor computes P0 and P1 from C, D, and E.
            Segment2 (const Vec<2,Real>& center, const Vec<2,Real>& direction,
                Real extent);

			PrimitiveType GetIntersectableType() const { return PT_SEGMENT2; }

            // Call this function when you change P0 or P1.
            void ComputeCenterDirectionExtent ();

            // Call this function when you change C, D, or e.
            void ComputeEndPoints ();

            bool Test(const Intersectable<Real, Vec<2, Real> > &);
			bool Find(const Intersectable<Real, Vec<2, Real> > &, IntersectionResult<Real> &);


            // End-point representation.
            Vec<2,Real> P0, P1;

            // Center-direction-extent representation.
            Vec<2,Real> Center;
            Vec<2,Real> Direction;
            Real Extent;
    };

    typedef Segment2<float> Segment2f;
    typedef Segment2<double> Segment2d;
}

#endif // SEGMENT_H
