#ifndef SPHERE3_H
#define SPHERE3_H

#include <sofa/defaulttype/Vec.h>

#include "Intersectable.h"

using namespace sofa::defaulttype;

namespace BVHModels
{
    template <typename Real>
    class Sphere3
    {
        public:
            // The sphere is represented as |X-C| = R where C is the center and R is
            // the radius.

            // Construction and destruction.
            Sphere3();  // uninitialized
            ~Sphere3();

            Sphere3(const Vector3<Real>& center, Real radius);

            Vector3<Real> Center;
            Real Radius;
    };

    typedef Sphere3<float> Sphere3f;
    typedef Sphere3<double> Sphere3d;
}
#endif // SPHERE3_H
