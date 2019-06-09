#ifndef LGCPLANE3_INL
#define LGCPLANE3_INL

#include "LGCLine3.h"
#include "LGCPlane3.h"
#include "LGCUtil.h"

using namespace sofa::defaulttype;
using namespace sofa::helper;

template <typename Real>
Plane3D<Real>::Plane3D(const Vec<3,Real>& pt1, const Vec<3,Real>& pt2, const Vec<3,Real>& pt3)
{
    /*std::cout << "=== Plane3D<Real>::Plane3D(const Vec<3,Real>& pt1, const Vec<3,Real>& pt2, const Vec<3,Real>& pt3, LGCTransformable<Real>* parent) ===" << std::endl;
    std::cout << "  points: " << pt1 << ", " << pt2 << ", " << pt3 << std::endl;
    std::cout << "  parent: " << parent << std::endl;
    if (parent != NULL)
        std::cout << "   " << *parent << std::endl;*/

    _normal = (pt2 - pt1).cross(pt3 - pt1);
    _normal.normalize();
    _d = _normal * pt1;
    _p = (pt3 - pt1) * 0.5f;

    copyPlaneData();
}

template <typename Real>
Plane3D<Real>::Plane3D(const Vec<3,Real>& n, const Vec<3,Real>& p)
{
    /*std::cout << "=== Plane3D<Real>::Plane3D(const Vec<3,Real>& n, const Vec<3,Real>& p, LGCTransformable<Real>* parent) ===" << std::endl;
    std::cout << "  normal: " << n << ", point: " << p << std::endl;
    std::cout << "  parent: " << parent << std::endl;
    if (parent != NULL)
        std::cout << "   " << *parent << std::endl;*/

    _normal = n;
    _normal.normalize();
    _p = p;
    _d = _normal * _p;

    copyPlaneData();
}

template <typename Real>
Plane3D<Real>::Plane3D(const Plane3D& other)
{
    /*std::cout << "=== Plane3D<Real>::Plane3D(const Plane3D& other) ===" << std::endl;
    std::cout << "  normal: " << other._normal << ", point: " << other._p << std::endl;
    std::cout << "  parent: " << other._parent << std::endl;
    if (other._parent != NULL)
        std::cout << "   " << *(other._parent) << std::endl;*/

    if (this != &other)
    {
        _normal = other._normal;
        _d = other._d;
        _p = other._p;

        copyPlaneData();
    }
}

template <typename Real>
Plane3D<Real>& Plane3D<Real>::operator=(const Plane3D& other)
{
    /*std::cout << "=== Plane3D<Real>::operator=(const Plane3D& other) ===" << std::endl;
    std::cout << "  normal: " << other._normal << ", point: " << other._p << std::endl;
    if (other._parent != NULL)
        std::cout << "   " << *(other._parent) << std::endl;*/

    if (this != &other)
    {
        _normal = other._normal;
        _d = other._d;
        _p = other._p;
        copyPlaneData();
    }
    return *this;
}

template <typename Real>
void Plane3D<Real>::copyPlaneData()
{
    Vec<3,Real> v1(-_normal.y(), _normal.x(), 0.0f);
    Vec<3,Real> v2(0.0f, -_normal.z(), _normal.y());

    Mat<3,3,Real> rotMatrix;
    rotMatrix[0] = _normal;
    rotMatrix[1] = v1;
    rotMatrix[2] = v2;

    /*Mat<4,4,Real> trMatrix;
    trMatrix.setsub(0,0, rotMatrix);
    trMatrix[3][0] = _p.x(); trMatrix[3][1] = _p.y(); trMatrix[3][2] = _p.z();*/
}

template <typename Real>
Real Plane3D<Real>::distanceToPoint(const Vec<3,Real>& q)
{
    return ((_normal * q) - _d) / (_normal * _normal);
}

template <typename Real>
Vec<3, Real> Plane3D<Real>::projectionOnPlane(const Vec<3,Real>& q)
{
    Real t = ((_normal * q) - _d) / (_normal * _normal);
    return q - (_normal * t);
}

template <typename Real>
typename Plane3D<Real>::SideOfPlane Plane3D<Real>::pointOnWhichSide(const Vec<3,Real>& pt)
{
    if (distanceToPoint(pt) > 0.0f)
        return TOWARDS_NORMAL;
    else
        return AWAY_FROM_NORMAL;
}

template <typename Real>
Vec<3, Real> Plane3D<Real>::mirrorOnPlane(const Vec<3,Real>& p)
{
    Vec<3,Real> v = p - _p;
    Real distance = v * _normal;
    return Vec<3,Real>(p - _normal * (2.0f * distance));
}

template <typename Real>
typename Plane3D<Real>::IntersectionType Plane3D<Real>::intersect(const Line3D<Real>& line, Vec<3,Real>& ipt)
{
    Vec<3,Real> ab = line.point(1) - line.point(0);
    Real denom = _normal * ab;
    if (!IsZero(denom))
    {
        Real t = (_d - _normal * line.point(0)) / denom;

        if (line.lineMode() == Line::SEGMENT)
        {
            if (t >= 0.0f && t <= 1.0f)
            {
                ipt = line.point(0) + ab * t;
                return POINT;
            }
        }
        else if (line.lineMode() == Line::RAY)
        {
            if (t >= 0.0f)
            {
                ipt = line.point(0) + ab * t;
                return POINT;
            }
        }
        else if (line.lineMode() == Line::LINE)
        {
            Real numerator = (_p - line.point(0)) * _normal;
            Real denominator = line.point(1) * _normal;

            if (IsZero(denominator))
            {
                if (IsZero(numerator))
                {
                    return COPLANAR;
                }
                return PARALLEL;
            }
            else
            {
                Real d = numerator / denominator;
                ipt = line.point(0) + line.direction() * d;
                return POINT;
            }
        }
    }
    return NONE;
}

template <typename Real>
typename Plane3D<Real>::IntersectionType Plane3D<Real>::intersect(const Plane3D<Real>& plane, Line3D<Real>& iLine)
{
    Vec<3,Real> d = _normal.cross(plane.normal());

    Real denom = d * d;
    if (IsZero(denom))
    {
        return NONE;
    }

    Vec<3, Real> p = (plane.normal() * _d - _normal * plane.distance()).cross(d);
    iLine = Line3D<Real>(p,d);

    return LINE;
}

//template <typename Real>
//typename Plane3D<Real>::IntersectionType Plane3D<Real>::intersect(const Plane3D<Real>& plane2, const Plane3D<Real>& plane3, Vec<3,Real>& iPoint)
//{
//    /*
//    Mat<3,3,Real> nVecs;
//    nVecs(0,0) = _normal.x(); nVecs(0,1) = _normal.y(); nVecs(0,2) = _normal.z();
//    nVecs(1,0) = plane2.normal().x(); nVecs(1,1) = plane2.normal().y(); nVecs(1,2) = plane2.normal().z();
//    nVecs(2,0) = plane3.normal().x(); nVecs(2,1) = plane3.normal().y(); nVecs(2,2) = plane3.normal().z();

//    std::cout << "normals matrix: " << nVecs << std::endl;

//    Real det = determinant(nVecs);
//    std::cout << "its determinant: " << det << std::endl;

//    if (sofa::helper::IsZero(det))
//    {
//        std::cout << " no intersection: det == " << det << std::endl;
//        return NONE;
//    }

//    iPoint = (plane2.normal().cross(plane3.normal()) * (_p * _normal));
//    iPoint = iPoint + plane3.normal().cross(_normal) * (plane2.point() * plane2.normal());
//    iPoint = iPoint + _normal.cross(plane2.normal()) * (plane3.point() * plane3.normal());
//    iPoint = iPoint * (1.0f / det);

//    std::cout << " intersection at: " << iPoint << std::endl;

//    return POINT;*/

//    Real *x = NumericalRecipes::vector<Real>( 1, 3 );
//    int *indx = NumericalRecipes::ivector( 1, 3 );
//    Real **mat = NumericalRecipes::matrix<Real>( 1, 3, 1, 3 );

//    if ( sofa::helper::collinear(_normal,plane2.normal()) || sofa::helper::collinear(_normal,plane3.normal()) || sofa::helper::collinear(plane2.normal(),plane3.normal()) )
//        return COPLANAR;

//    for ( unsigned int i = 0; i < 3; i ++ )
//        mat[1][1+i] = _normal[i];
//    for ( unsigned int i = 0; i < 3; i ++ )
//        mat[2][1+i] = plane2.normal()[i];
//    for ( unsigned int i = 0; i < 3; i ++ )
//        mat[3][1+i] = plane3.normal()[i];

//    x[1] = _d;  x[2] = plane2.distance(); x[3] = plane3.distance();

//    int ludc;
//    Real ludd;
//    NumericalRecipes::ludcmp( mat, 3, indx, &ludd, &ludc );
//    if ( ! ludc )
//    {
//        iPoint = Vector3(0,0,0);
//        return NONE;
//    }

//    NumericalRecipes::lubksb( mat, 3, indx, x );

//    iPoint = Vector3( *(x+1+0), *(x+1+1), *(x+1+2) );

//    NumericalRecipes::free_vector(x,  1, 3 );
//    NumericalRecipes::free_ivector(indx,  1, 3 );
//    NumericalRecipes::free_matrix(mat, 1, 3, 1, 3 );

//    return POINT;
//}

//template <typename Real>
//bool Plane3D<Real>::pointInTriangle(const Vec<3, Real>& p, const Vec<3, Real>& t1, const Vec<3, Real>& t2, const Vec<3, Real>& t3)
//{
//    Vec<3, Real> v0 = t2 - t1;
//    Vec<3, Real> v1 = t3 - t1;
//    Vec<3, Real> v2 = p - t1;

//    Real d00 = v0 * v0;
//    Real d01 = v0 * v1;
//    Real d11 = v1 * v1;
//    Real d20 = v2 * v0;
//    Real d21 = v2 * v1;
//    Real denom = d00 * d11 - d01 * d01;

//    Real v = (d11 * d20 - d01 * d21) / denom;
//    Real w = (d00 * d21 - d01 * d20) / denom;

//    if (v >= sofa::helper::kEpsilon && w >= sofa::helper::kEpsilon && (v+w) <= 1.0f)
//        return true;

//    return false;
//}
#endif // LGCPLANE3_INL
