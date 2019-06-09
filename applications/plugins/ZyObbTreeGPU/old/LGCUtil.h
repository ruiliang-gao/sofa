#ifndef LGCUTIL_H
#define LGCUTIL_H

#include <cmath>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

namespace sofa
{
    namespace helper
    {
        using namespace sofa::defaulttype;

#ifndef SOFA_DOUBLE
        static const double kEpsilon = 1.0e-6f;
#else
        static const float kEpsilon = 1.0e-6f;
#endif

        template <typename Real>
        inline bool IsZero(const Real& a)
        {
            return (std::fabs(a) < kEpsilon);
        }

        template <typename Real>
        inline bool AreEqual(const Real& a, const Real& b)
        {
            return (IsZero(a-b));
        }

        template <typename Real>
        bool collinear(const Vec<3,Real> &a, const Vec<3,Real> &b)
        {
            Real l;
            int x;

            // is b == 0?
            x = 0;
            for ( unsigned int i = 1; i < 3; i ++ )
                if ( std::fabs(b[x]) < std::fabs(b[i]) )
                            x = i;
            if ( std::fabs(b[x]) < kEpsilon )
                     return false;

            l = a[x] / b[x];
            if ( l < kEpsilon && l > -kEpsilon )
                    return false;

            return (std::fabs(a[0] - l*b[0]) < kEpsilon &&
                    std::fabs(a[1] - l*b[1]) < kEpsilon &&
                    std::fabs(a[2] - l*b[2]) < kEpsilon);
        }

        /**  Pseudo random number generator
         *
         * @return
         *   A number in the range [0,2147483646].
         *
         * Creates the same sequence of pseudo random numbers on every platform.
         * Use this instead of any random(), drand48(), etc., in regression test
         * programs.
         *
         * @todo
         * - Check that the seed is not the unique "bad" number as explained
         *   in @a http://home.t-online.de/home/mok-kong.shen/ .
         * - @a a should be a primitive root of @a c (see URL above).
         * - Groesseres @a c suchen.
         *
         * @bug
         *   Not multithread-safe.
         *
         * @see
         *   pseudo_randomf
         *
         * @implementation
         *   Based on a linear congruential relation x(new) = a * x(old) + b (mod c).
         *   If you change @a c, you @e must change pseudo_randomf!
         *   Source: @a http://home.t-online.de/home/mok-kong.shen/ .
         *
         **/

        template <typename Num>
        Num pseudo_randomi( void )
        {
                static unsigned long long int x_old = 13;
                static unsigned long long int c = 2147483647;
                static unsigned long long int b = 3131747;
                static unsigned long long int a = 79;

                x_old = a * x_old + b;
                x_old = x_old % c;

                return static_cast<Num>( x_old );
        }

        /**  Pseudo random number generator
         *
         * @return
         *   A number in the range [0,1).
         *
         * Creates the same sequence of pseudo random numbers on every platform.
         * Use this instead of any random(), drand48(), etc., in regression test
         * programs.
         *
         * @see
         *   pseudo_random
         *
         **/

        template <typename Real>
        Real pseudo_randomf( void )
        {
            return static_cast<Real>( pseudo_randomi<unsigned int>() ) /
                       static_cast<Real>( 2147483647 );
        }

        template<typename Real>
        Vec<4,Real> randomVec4()
        {
            double r = drand48();
            double g = drand48();
            double b = drand48();
            double a = drand48();

            if (r < 0.5f)
                r = 0.5f + r;

            if (g < 0.5f)
                g = 0.5f + g;

            if (b < 0.5f)
                b = 0.5f + b;

            if (a < 0.5f)
                a = 0.5f + a;

            return Vec<4,Real>(r,g,b,a);
        }

        template <typename Real>
        bool equal(const Vec<3,Real> &v1, const Vec<3,Real> &v2, Real diff)
        {
            Vec<3,Real> v = v1 - v2;
            for ( unsigned int i=0; i<3; ++ i )
            {
                if ( std::fabs( v[i] ) > diff )
                    return false;
            }
            return true;
        }

        template <typename Real>
        bool barycenter(std::vector<Vec<3,Real> > points, Vector3& bc)
        {
            if (points.size() == 0)
                return false;

            for ( unsigned int i = 0; i < points.size(); i ++ )
            {
                for ( unsigned int j = 0; j < 3; j ++ )
                    bc[j] += points[i][j];

                for ( unsigned int j = 0; j < 3; j ++ )
                    bc[j] /= points.size();
            }

            return true;
        }

        template <typename Real>
        static void ComputeCovarianceMatrix(Mat<3,3,Real>& C, Vec<3,Real>& mean,
                                     const std::vector<Vec<3,Real> >& points)
        {
            mean = Vec<3,Real>(0,0,0);
            unsigned int i;
            for (i = 0; i < points.size(); i++)
               mean += points[i];

            mean /= (Real) points.size();

           // compute the (co)variances
           Real varX = 0.0f;
           Real varY = 0.0f;
           Real varZ = 0.0f;
           Real covXY = 0.0f;
           Real covXZ = 0.0f;
           Real covYZ = 0.0f;

           for (i = 0; i < points.size(); i++)
           {
               Vec<3,Real> diff = points[i] - mean;

               varX += diff.x() * diff.x();
               varY += diff.y() * diff.y();
               varZ += diff.z() * diff.z();
               covXY += diff.x() * diff.z();
               covXZ += diff.x() * diff.z();
               covYZ += diff.y() * diff.z();
           }

           // divide all of the (co)variances by n - 1
           // (skipping the division if n = 2, since that would be division by 1.0
           if (points.size() > 2)
           {
               const Real normalize = (Real)(points.size() - 1);
               varX /= normalize;
               varY /= normalize;
               varZ /= normalize;
               covXY /= normalize;
               covXZ /= normalize;
               covYZ /= normalize;
           }

           // pack values into the covariance matrix, which is symmetric
           C(0,0) = varX;
           C(1,1) = varY;
           C(2,2) = varZ;
           C(1,0) = C(0,1) = covXY;
           C(2,0) = C(0,2) = covXZ;
           C(1,2) = C(2,1) = covYZ;

        }

        template <typename Real>
        static void SymmetricHouseholder3x3 (const Mat<3,3,Real>& M, Mat<3,3,Real>& Q,
                                            Vec<3,Real>& diag, Vec<3,Real>& subd)
        {
            // Computes the Householder reduction of M, computing:
            //
            // T = Q^t M Q
            //
            //   Input:
            //     symmetric 3x3 matrix M
            //   Output:
            //     orthogonal matrix Q
            //     diag, diagonal entries of T
            //     subd, lower-triangle entries of T (T is symmetric)

            // T will be stored as follows (because it is symmetric and tridiagonal):
            //
            // | diag[0]  subd[0]  0       |
            // | subd[0]  diag[1]  subd[1] |
            // | 0        subd[1]  diag[2] |

            diag[0] = M(0,0); // in both cases below, T(0,0) = M(0,0)
            subd[2] = 0; // T is to be a tri-diagonal matrix - the (2,0) and (0,2)
                         // entries must be zero, so we don't need subd[2] for this step

            // so we know that T will actually be:
            //
            // | M(0,0)   subd[0]  0       |
            // | subd[0]  diag[1]  subd[1] |
            // | 0        subd[1]  diag[2] |

            // so the only question remaining is the lower-right block and subd[0]

            if ( std::fabs(M(2,0)) < kEpsilon )
            {
                // if M(2,0) (and thus M(0,2)) is zero, then the matrix is already in
                // tridiagonal form.  As such, the Q matix is the identity, and we
                // just extract the diagonal and subdiagonal components of T as-is
                Q.identity();

                // so we see that T will actually be:
                //
                // | M(0,0)  M(1,0)  0      |
                // | M(1,0)  M(1,1)  M(2,1) |
                // | 0       M(2,1)  M(2,2) |
                diag[1] = M(1,1);
                diag[2] = M(2,2);

                subd[0] = M(1,0);
                subd[1] = M(2,1);
            }
            else
            {
                // grab the lower triangle of the matrix, minus a, which we don't need
                // (see above)
                // |       |
                // | b d   |
                // | c e f |
                const Real b = M(1,0);
                const Real c = M(2,0);
                const Real d = M(1,1);
                const Real e = M(2,1);
                const Real f = M(2,2);

                // normalize b and c to unit length and store in u and v
                // we want the lower-right block we create to be orthonormal
                const Real L = std::sqrt(b * b + c * c);
                const Real u = b / L;
                const Real v = c / L;
                Q(0,0) = 1.0f; Q(0,1) = 0.0f; Q(0,2) = 0.0f;
                Q(1,0) = 0.0f; Q(1,1) = u; Q(1,2) = v;
                Q(2,0) = 0.0f; Q(2,1) = v; Q(2,2) = -u;

                Real q = 2 * u * e + v * (f - d);
                diag[1] = d + v * q;
                diag[2] = f - v * q;

                subd[0] = L;
                subd[1] = e - u * q;

                // so we see that T will actually be:
                //
                // | M(0,0)  L       0     |
                // | L       d+c*q   e-b*q |
                // | 0       e-b*q   f-c*q |
            }
        }

        template <typename Real>
        static int QLAlgorithm (Mat<3,3,Real>& M, Vec<3,Real>& diag, Vec<3,Real>& subd)
        {
            // QL iteration with implicit shifting to reduce matrix from tridiagonal
            // to diagonal

            int L;
            for (L = 0; L < 3; L++)
            {
                // As this is an iterative process, we need to keep a maximum number of
                // iterations, just in case something is wrong - we cannot afford to
                // loop forever
                const int maxIterations = 32;

                int iter;
                for (iter = 0; iter < maxIterations; iter++)
                {
                    int m;
                    for (m = L; m <= 1; m++)
                    {
                        Real dd = std::fabs(diag[m]) + std::fabs(diag[m+1]);
                        if ( std::fabs(subd[m]) + dd == dd )
                            break;
                    }
                    if ( m == L )
                        break;

                    Real g = (diag[L+1]-diag[L])/(2*subd[L]);
                    Real r = std::sqrt(g*g+1);
                    if ( g < 0 )
                        g = diag[m]-diag[L]+subd[L]/(g-r);
                    else
                        g = diag[m]-diag[L]+subd[L]/(g+r);
                    Real s = 1, c = 1, p = 0;
                    for (int i = m-1; i >= L; i--)
                    {
                        Real f = s*subd[i], b = c*subd[i];
                        if ( std::fabs(f) >= std::fabs(g) )
                        {
                            c = g/f;
                            r = std::sqrt(c*c+1);
                            subd[i+1] = f*r;
                            c *= (s = 1/r);
                        }
                        else
                        {
                            s = f/g;
                            r = std::sqrt(s*s+1);
                            subd[i+1] = g*r;
                            s *= (c = 1/r);
                        }
                        g = diag[i+1]-p;
                        r = (diag[i]-g)*s+2*b*c;
                        p = s*r;
                        diag[i+1] = g+p;
                        g = c*r-b;

                        for (int k = 0; k < 3; k++)
                        {
                            f = M(k,i+1);
                            M(k,i+1) = s*M(k,i)+c*f;
                            M(k,i) = c*M(k,i)-s*f;
                        }
                    }
                    diag[L] -= p;
                    subd[L] = g;
                    subd[m] = 0;
                }

                // exceptional case - matrix took more iterations than should be
                // possible to move to diagonal form
                if ( iter == maxIterations )
                    return 0;
            }
            return 1;
        }

        template <typename Real>
        static void GetRealSymmetricEigenvectors(Vec<3,Real>& v1, Vec<3,Real>& v2, Vec<3,Real>& v3,
                                           const Mat<3,3,Real>& A )
        {
            Vec<3,Real> eigenvalues;
            Vec<3,Real> lowerTriangle;
            Mat<3,3,Real> Q;

            SymmetricHouseholder3x3 (A, Q, eigenvalues, lowerTriangle);
            QLAlgorithm(Q, eigenvalues, lowerTriangle);

            // Sort the eigenvalues from greatest to smallest, and use these indices
            // to sort the eigenvectors
            int v1Index, v2Index, v3Index;
            if (eigenvalues[0] > eigenvalues[1])
            {
                if (eigenvalues[1] > eigenvalues[2])
                {
                    v1Index = 0;
                    v2Index = 1;
                    v3Index = 2;
                }
                else if (eigenvalues[2] > eigenvalues[0])
                {
                    v1Index = 2;
                    v2Index = 0;
                    v3Index = 1;
                }
                else
                {
                    v1Index = 0;
                    v2Index = 2;
                    v3Index = 1;
                }
            }
            else // eigenvalues[1] >= eigenvalues[0]
            {
                if (eigenvalues[0] > eigenvalues[2])
                {
                    v1Index = 1;
                    v2Index = 0;
                    v3Index = 2;
                }
                else if (eigenvalues[2] > eigenvalues[1])
                {
                    v1Index = 2;
                    v2Index = 1;
                    v3Index = 0;
                }
                else
                {
                    v1Index = 1;
                    v2Index = 2;
                    v3Index = 0;
                }
            }

            // Sort the eigenvectors into the output vectors
            v1 = Q.col(v1Index);
            v2 = Q.col(v2Index);
            v3 = Q.col(v3Index);

            // If the resulting eigenvectors are left-handed, negate the
            // min-eigenvalue eigenvector to make it right-handed
            if ( v1 * (v2.cross(v3)) < 0.0f )
                v3 = -v3;
        }
    }
}

#endif // LGCUTIL_H
