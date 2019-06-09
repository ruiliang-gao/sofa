#ifndef MATHUTILS_H
#define MATHUTILS_H

#include <cfloat>
#include <cmath>
#include <cstdlib>

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

namespace BVHModels
{
    template <typename Real>
    class MathUtils
    {
        public:
            // Wrappers to hide implementations of functions.  The ACos and ASin
            // functions clamp the input argument to [-1,1] to avoid NaN issues
            // when the input is slightly larger than 1 or slightly smaller than -1.
            // Other functions have the potential for using a fast and approximate
            // algorithm rather than calling the standard math library functions.
            static Real ACos (Real value);
            static Real ASin (Real value);
            static Real ATan (Real value);
            static Real ATan2 (Real y, Real x);
            static Real Ceil (Real value);
            static Real Cos (Real value);
            static Real Exp (Real value);
            static Real FAbs (Real value);
            static Real Floor (Real value);
            static Real FMod (Real x, Real y);
            static Real InvSqrt (Real value);
            static Real Log (Real value);
            static Real Log2 (Real value);
            static Real Log10 (Real value);
            static Real Pow (Real base, Real exponent);
            static Real Sin (Real value);
            static Real Sqr (Real value);
            static Real Sqrt (Real value);
            static Real Tan (Real value);

            // Return -1 if the input is negative, 0 if the input is zero, and +1
            // if the input is positive.
            static int Sign (int value);
            static Real Sign (Real value);

            // Generate a random number in [0,1].  The random number generator may
            // be seeded by a first call to UnitRandom with a positive seed.
            static Real UnitRandom (unsigned int seed = 0);

            // Generate a random number in [-1,1].  The random number generator may
            // be seeded by a first call to SymmetricRandom with a positive seed.
            static Real SymmetricRandom (unsigned int seed = 0);

            // Generate a random number in [min,max].  The random number generator may
            // be seeded by a first call to IntervalRandom with a positive seed.
            static Real IntervalRandom (Real min, Real max, unsigned int seed = 0);

            // Clamp the input to the specified interval [min,max].
            static Real Clamp (Real value, Real minValue, Real maxValue);

            // Clamp the input to [0,1].
            static Real Saturate (Real value);

            // Fast evaluation of trigonometric and inverse trigonometric functions
            // using polynomial approximations.  The speed ups vary with processor.

            // The input must be in [0,pi/2].
            static Real FastSin0 (Real angle);
            static Real FastSin1 (Real angle);

            // The input must be in [0,pi/2]
            static Real FastCos0 (Real angle);
            static Real FastCos1 (Real angle);

            // The input must be in [0,pi/4].
            static Real FastTan0 (Real angle);
            static Real FastTan1 (Real angle);

            // The input must be in [0,1].
            static Real FastInvSin0 (Real value);
            static Real FastInvSin1 (Real value);

            // The input must be in [0,1].
            static Real FastInvCos0 (Real value);
            static Real FastInvCos1 (Real value);

            // The input must be in [-1,1].
            static Real FastInvTan0 (Real value);
            static Real FastInvTan1 (Real value);

            // Fast approximations to exp(-x).  The input x must be in [0,infinity).
            static Real FastNegExp0 (Real value);
            static Real FastNegExp1 (Real value);
            static Real FastNegExp2 (Real value);
            static Real FastNegExp3 (Real value);

            // Common constants.
            static const Real EPSILON;
            static const Real ZERO_TOLERANCE;
            static const Real MAX_REAL;

            static const int MAX_INT;
            static const int MIN_INT;

            static const Real PI;
            static const Real TWO_PI;
            static const Real HALF_PI;
            static const Real INV_PI;
            static const Real INV_TWO_PI;
            static const Real DEG_TO_RAD;
            static const Real RAD_TO_DEG;
            static const Real LN_2;
            static const Real LN_10;
            static const Real INV_LN_2;
            static const Real INV_LN_10;
            static const Real SQRT_2;
            static const Real INV_SQRT_2;
            static const Real SQRT_3;
            static const Real INV_SQRT_3;

            static void Orthonormalize(Vec<3,Real>& u, Vec<3,Real>& v, Vec<3,Real>& w);
            static void Orthonormalize(Vec<3,Real>* vectors);

            static void GenerateComplementBasis (Vec<3,Real>& u, Vec<3,Real>& v,
                                                 const Vec<3,Real>& w);
    };

    typedef MathUtils<float> Mathf;
    typedef MathUtils<double> Mathd;
}

//#ifdef _WIN32
#include "MathUtils.inl"
//#endif

#endif // MATHUTILS_H
