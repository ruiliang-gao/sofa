//#ifndef _WIN32
//#include "MathUtils.inl"
//#else
#include "MathUtils.h"
//#endif

#include <climits>

namespace BVHModels
{
    template<> const float MathUtils<float>::EPSILON = FLT_EPSILON;
    template<> const float MathUtils<float>::ZERO_TOLERANCE = 1e-06f;
    template<> const float MathUtils<float>::MAX_REAL = FLT_MAX;
    template<> const float MathUtils<float>::PI = (float)(4.0*atan(1.0));
    template<> const float MathUtils<float>::TWO_PI = 2.0f*MathUtils<float>::PI;
    template<> const float MathUtils<float>::HALF_PI = 0.5f*MathUtils<float>::PI;
    template<> const float MathUtils<float>::INV_PI = 1.0f/MathUtils<float>::PI;
    template<> const float MathUtils<float>::INV_TWO_PI = 1.0f/MathUtils<float>::TWO_PI;
    template<> const float MathUtils<float>::DEG_TO_RAD = MathUtils<float>::PI/180.0f;
    template<> const float MathUtils<float>::RAD_TO_DEG = 180.0f/MathUtils<float>::PI;
    template<> const float MathUtils<float>::LN_2 = MathUtils<float>::Log(2.0f);
    template<> const float MathUtils<float>::LN_10 = MathUtils<float>::Log(10.0f);
    template<> const float MathUtils<float>::INV_LN_2 = 1.0f/MathUtils<float>::LN_2;
    template<> const float MathUtils<float>::INV_LN_10 = 1.0f/MathUtils<float>::LN_10;
    template<> const float MathUtils<float>::SQRT_2 = (float)(sqrt(2.0));
    template<> const float MathUtils<float>::INV_SQRT_2 = 1.0f/MathUtils<float>::SQRT_2;
    template<> const float MathUtils<float>::SQRT_3 = (float)(sqrt(3.0));
    template<> const float MathUtils<float>::INV_SQRT_3 = 1.0f/MathUtils<float>::SQRT_3;

    template<> const int MathUtils<float>::MAX_INT = INT_MAX;
    template<> const int MathUtils<float>::MIN_INT = INT_MIN;

    template<> const double MathUtils<double>::EPSILON = DBL_EPSILON;
    template<> const double MathUtils<double>::ZERO_TOLERANCE = 1e-08;
    template<> const double MathUtils<double>::MAX_REAL = DBL_MAX;
    template<> const double MathUtils<double>::PI = 4.0*atan(1.0);
    template<> const double MathUtils<double>::TWO_PI = 2.0*MathUtils<double>::PI;
    template<> const double MathUtils<double>::HALF_PI = 0.5*MathUtils<double>::PI;
    template<> const double MathUtils<double>::INV_PI = 1.0/MathUtils<double>::PI;
    template<> const double MathUtils<double>::INV_TWO_PI = 1.0/MathUtils<double>::TWO_PI;
    template<> const double MathUtils<double>::DEG_TO_RAD = MathUtils<double>::PI/180.0;
    template<> const double MathUtils<double>::RAD_TO_DEG = 180.0/MathUtils<double>::PI;
    template<> const double MathUtils<double>::LN_2 = MathUtils<double>::Log(2.0);
    template<> const double MathUtils<double>::LN_10 = MathUtils<double>::Log(10.0);
    template<> const double MathUtils<double>::INV_LN_2 = 1.0/MathUtils<double>::LN_2;
    template<> const double MathUtils<double>::INV_LN_10 = 1.0/MathUtils<double>::LN_10;
    template<> const double MathUtils<double>::SQRT_2 = sqrt(2.0);
    template<> const double MathUtils<double>::INV_SQRT_2 = 1.0f/MathUtils<float>::SQRT_2;
    template<> const double MathUtils<double>::SQRT_3 = sqrt(3.0);
    template<> const double MathUtils<double>::INV_SQRT_3 = 1.0f/MathUtils<float>::SQRT_3;

    template<> const int MathUtils<double>::MAX_INT = INT_MAX;
    template<> const int MathUtils<double>::MIN_INT = INT_MIN;
}



