#ifndef RECTANGLE3_H
#define RECTANGLE3_H

#include "DistanceComputable.h"

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

namespace BVHModels
{
    template <typename Real>
    class Rectangle3: public DistanceComputable<Real, Vec<3,Real> >
    {
        public:
            Rectangle3 ();  // uninitialized
           ~Rectangle3 ();

           Rectangle3 (const Vec<3,Real>& center, const Vec<3,Real> axis[2],
               const Real extent[2]);

           Rectangle3 (const Vec<3,Real>& center, const Vec<3,Real>& axis0,
               const Vec<3,Real>& axis1, Real extent0, Real extent1);

           void ComputeVertices (Vec<3,Real> vertex[4]) const;

           // Get the rectangle corners.
           Vec<3,Real> GetPPCorner () const;  // C + e0*A0 + e1*A1
           Vec<3,Real> GetPMCorner () const;  // C + e0*A0 - e1*A1
           Vec<3,Real> GetMPCorner () const;  // C - e0*A0 + e1*A1
           Vec<3,Real> GetMMCorner () const;  // C - e0*A0 - e1*A1

           //-----------------------
           //-----------------------

           virtual Real GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result);     // distance
           virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result);  // squared distance

           virtual Real GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3, Real>& velocity0, const Vec<3, Real>& velocity1, DistanceResult& result);
           virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const Vec<3, Real>& velocity0, const Vec<3, Real>& velocity1, DistanceResult& result);

           Vec<3,Real> Center;
           Vec<3,Real> Axis[2];
           Real Extent[2];
    };

	typedef Rectangle3<float> Rectangle3f;
	typedef Rectangle3<double> Rectangle3d;
}

#endif // RECTANGLE3_H
