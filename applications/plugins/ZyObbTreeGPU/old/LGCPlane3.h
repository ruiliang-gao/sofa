#ifndef LGCPLANE3_H
#define LGCPLANE3_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/VecTypes.h>

#include "LGCLine3.h"

namespace sofa
{
    namespace defaulttype
    {
        template <typename Real> class Plane3D
        {
            public:
                enum IntersectionType
                {
                    NONE,
                    POINT,
                    LINE,
                    PARALLEL,
                    COPLANAR
                };

                enum SideOfPlane
                {
                    TOWARDS_NORMAL,
                    AWAY_FROM_NORMAL
                };

                Plane3D(const Vec<3,Real>&, const Vec<3,Real>&, const Vec<3,Real>&);
                Plane3D(const Vec<3,Real>& /* normal */, const Vec<3,Real>& /* point */);

                Plane3D(const Plane3D&);
                Plane3D& operator=(const Plane3D&);

                inline const Vec<3,Real>& normal() const { return _normal; }
                inline Real distance() const { return _d; }
                inline const Vec<3,Real>& point() const { return _p; }

                Real distanceToPoint(const Vec<3,Real>&);
                Vec<3,Real> projectionOnPlane(const Vec<3,Real>&);
                Vec<3,Real> mirrorOnPlane(const Vec<3,Real>&);

                SideOfPlane pointOnWhichSide(const Vec<3,Real>&);

                IntersectionType intersect(const Line3D<Real>&, Vec<3,Real>&);
                IntersectionType intersect(const Plane3D<Real>&, Line3D<Real>&);
                IntersectionType intersect(const Plane3D<Real>&, const Plane3D<Real>&, Vec<3,Real>&);

                bool pointInTriangle(const Vec<3,Real>&, const Vec<3,Real>&, const Vec<3,Real>&, const Vec<3,Real>&);

                void transform(const Mat<4, 4, Real>&);
                void transform(const Vec<3, Real>&, const Quater<Real>&);
                void translate(const Vec<3, Real> &);
                void rotate(const Mat<3, 3, Real> &);
                void rotate(const Quater<Real> &);

            private:
                void copyPlaneData();

                Vec<3,Real> _normal;
                Real _d;
                Vec<3,Real> _p;

        };

        template <typename Real>
        std::ostream &
        operator<<(std::ostream &os, const Plane3D<Real> &plane)
        {
            os << "Plane3D(n: " << plane.normal() << ", d: " << plane.distance() << ", p: " << plane.point() << ", parent: " << plane.parent() << ", referenceFrame: " << plane.referenceFrame() << ")";
            return os;
        }

        typedef Plane3D<float> Plane3Df;
        typedef Plane3D<double> Plane3Dd;

#ifdef SOFA_FLOAT
        typedef Plane3Df Plane;
#else
        typedef Plane3Dd Plane;
#endif
    }
}

#endif // LGCPLANE3_H
