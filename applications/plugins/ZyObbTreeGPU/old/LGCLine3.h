#ifndef LGCLINE3_H
#define LGCLINE3_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Quat.h>

namespace sofa
{
    namespace defaulttype
    {
        using namespace sofa::helper;
        template <typename Real> class Line3D
        {
            public:
                enum LineMode
                {
                    LINE,
                    RAY,
                    SEGMENT,
                    DEFAULT = LINE
                };

                Line3D();
                Line3D(const Line3D&);
                Line3D& operator=(const Line3D&);

                Line3D(const Vec<3,Real>&, const Vec<3,Real>&, const LineMode& = DEFAULT);

                Vec<3,Real> point(const int& = 0) const;
                inline Vec<3,Real> direction() const { return _direction; }

                inline LineMode lineMode() const { return _mode; }

                bool distanceOfPointToLine(const Vec<3,Real>&, Real&);
                Vec<3,Real> closestPointOnLine(const Line3D<Real>&, Vec<3,Real>&);

                Real distanceOfLineToLine(const Line3D<Real>&);

                bool projectionOnLine(const Vec<3,Real>&, Vec<3,Real>&);

                void transform(const Mat<4, 4, Real>&);
                void transform(const Vec<3, Real>&, const Quater<Real>&);
                void translate(const Vec<3, Real> &);
                void rotate(const Mat<3, 3, Real> &);
                void rotate(const Quater<Real> &);

            private:
                Vec<3,Real> _pt;
                Vec<3,Real> _direction;
                LineMode _mode;
        };

        template <typename Real>
        std::ostream &
        operator<<(std::ostream &os, const Line3D<Real> &line)
        {
            return os << "Line3D(p: " << line.point() << ", l: " << line.direction() << ")";
        }

        typedef Line3D<float> Line3Df;
        typedef Line3D<double> Line3Dd;

        #ifdef SOFA_FLOAT
        typedef Line3Df Line;
        #else
        typedef Line3Dd Line;
        #endif
    }
}

#endif // LGCLINE3_H
