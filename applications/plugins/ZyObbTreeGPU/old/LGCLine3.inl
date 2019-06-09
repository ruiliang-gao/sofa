#ifndef LGCLINE3_INL
#define LGCLINE3_INL

#include "LGCLine3.h"
#include "LGCUtil.h"

using namespace sofa::defaulttype;
using namespace sofa::helper;

template <typename Real>
Line3D<Real>::Line3D(): _pt(Vec<3,Real>(0,0,0)), _direction(Vec<3,Real>(1,0,0)), _mode(DEFAULT)
{

}

template <typename Real>
Line3D<Real>::Line3D(const Line3D& other)
{
    if (this != &other)
    {
        _pt = other._pt;
        _direction = other._direction;
        _mode = other._mode;
    }
}

template <typename Real>
Line3D<Real>& Line3D<Real>::operator=(const Line3D& other)
{
    if (this != &other)
    {
        _pt = other._pt;
        _direction = other._direction;
        _mode = other._mode;
    }
    return *this;
}

template <typename Real>
Line3D<Real>::Line3D(const Vec<3,Real>& point0, const Vec<3,Real>& dir, const LineMode& mode)
{
    _pt = point0;
    _direction = dir;
    _mode = mode;
}


template <typename Real>
Vec<3, Real> Line3D<Real>::point(const int& idx) const
{
    int ptIdx = idx;
    if (_mode == RAY && ptIdx < 0)
        ptIdx *= -1;

    if (_mode == LINE)
    {
        if (ptIdx < 0)
            ptIdx = 0;
        if (ptIdx > 1)
            ptIdx = 1;
    }

    return _pt + (_direction * (1.0f * ptIdx));
}

template <typename Real>
bool Line3D<Real>::distanceOfPointToLine(const Vec<3,Real>& pt, Real& distance)
{
    Real vsq = _direction * _direction;
    if (IsZero(vsq))
    {
        distance = 0.0f;
        return false;
    }


    Vec<3,Real> w = pt - _pt;
    Real wd = (w * _direction);
    distance = std::sqrt((w * w) - ((wd * wd) / vsq));

    return true;
}

template <typename Real>
Vec<3, Real> Line3D<Real>::closestPointOnLine(const Line3D<Real>& line, Vec<3, Real>& lpt)
{
    Vec<3, Real> wO = _pt - line.point();
    Real a = _direction * _direction;
    Real b = _direction * line.direction();
    Real c = line.direction() * line.direction();
    Real d = _direction * wO;
    Real e = line.direction() * wO;

    Vec<3,Real> cpt;
    Real denom = a * c - b * b;
    if (IsZero(denom))
    {
        cpt = _pt;
        lpt = line.point(0) + line.direction() * (e / c);
    }
    else
    {
        cpt = _pt + _direction * ((b * e - c * d) / denom);
        lpt = line.point(0) + line.direction() * ((a * e - b * d) / denom);
    }

    return cpt;
}

template <typename Real>
bool Line3D<Real>::projectionOnLine(const Vec<3,Real>& pt, Vec<3,Real>& lpt)
{
    Vec<3,Real> w = pt - _pt;
    Real vsq = _direction * _direction;

    if (vsq == 0.0f)
        return false;

    Real proj = w * _direction;
    lpt = _pt + _direction * (proj / vsq);

    return true;
}

template <typename Real>
Real Line3D<Real>::distanceOfLineToLine(const Line3D<Real>& line)
{
    Vec<3,Real> wO = _pt - line.point(0);
    Real a = _direction * _direction;
    Real b = _direction * line.direction();
    Real c = line.direction() * line.direction();
    Real d = _direction * wO;
    Real e = line.direction() * wO;

    Real denom = (a * c - b * b);

    if (IsZero(denom))
    {
        Vec<3,Real> wc = wO - line.direction() * (e / c);
        return wc * wc;
    }
    else
    {
        Vec<3,Real> wc = wO + _direction * ((b * e - c * d) / denom) - line.direction() * ((a * e - b * d) /denom);
        return wc * wc;
    }
}

#endif // LGCLINE3_INL
