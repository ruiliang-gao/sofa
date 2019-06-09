#ifndef UTILS_2D_H
#define UTILS_2D_H

#include <sofa/defaulttype/Vec.h>

#include "Primitives/Line2.h"
#include "Primitives/Triangle2.h"

using namespace sofa::defaulttype;

namespace BVHModels
{
    template <typename Real>
    class Utils_2D
    {
        public:
        static void TriangleLineRelations (const Vec<2,Real>& origin,
                const Vec<2,Real>& direction, const Triangle2<Real>& triangle,
                Real dist[3], int sign[3], int& positive, int& negative,
                int& zero);

            // Compute the parameter interval for the segment of intersection when
            // the triangle transversely intersects the line.
            static void GetInterval (const Vec<2,Real>& origin,
                const Vec<2,Real>& direction, const Triangle2<Real>& triangle,
                const Real dist[3], const int sign[3], Real param[2]);
    };
}
#endif // UTILS_2D_H
