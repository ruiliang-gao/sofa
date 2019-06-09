#ifndef DISTANCECOMPUTABLE_H
#define DISTANCECOMPUTABLE_H

#include "Intersectable.h"

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

namespace BVHModels
{

    enum DistanceType
    {
        DS_EMPTY
    };

    struct DistanceResult
    {
        public:
            DistanceResult()
            {
                distanceType = DS_EMPTY;
                primitiveType1 = PT_NONE;
                primitiveType2 = PT_NONE;
            }

            int GetDistanceType() const { return distanceType; }
            int GetPrimitiveType1() const { return primitiveType1; }
            int GetPrimitiveType2() const { return primitiveType2; }

            // The time at which minimum distance occurs for the dynamic queries.
            double GetContactTimeMinDistance() const;

            // Closest points on the two objects.  These are valid for static or
            // dynamic queries.  The set of closest points on a single object need
            // not be a single point.  In this case, the Boolean member functions
            // return 'true'.  A derived class may support querying for the full
            // contact set.
            const Vector3& GetClosestPoint0 () const;
            const Vector3& GetClosestPoint1 () const;
            bool HasMultipleClosestPoints0 () const;
            bool HasMultipleClosestPoints1 () const;

            virtual void reset()
            {
                distanceType = DS_EMPTY;
                primitiveType1 = PT_NONE;
                primitiveType2 = PT_NONE;
                mClosestPoint0.clear();
                mClosestPoint1.clear();
                mContactTime = 0.0f;
                mHasMultipleClosestPoints0 = false;
                mHasMultipleClosestPoints1 = false;
            }

            DistanceType distanceType;
            PrimitiveType primitiveType1;
            PrimitiveType primitiveType2;

            Vector3 mClosestPoint0;
            Vector3 mClosestPoint1;

            double mContactTime;

            bool mHasMultipleClosestPoints0;
            bool mHasMultipleClosestPoints1;

    };

	//template <typename Real, typename TVector>	class DistanceComputable
    template <typename Real, typename TVector>
	class DistanceComputable
    {
        public:
            // Abstract base class.
            virtual ~DistanceComputable ();

            // Static distance queries.
            virtual Real GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result) = 0;     // distance
            virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, DistanceResult& result) = 0;  // squared distance

            // Function calculations for dynamic distance queries.
            virtual Real GetDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const TVector& velocity0, const TVector& velocity1, DistanceResult& result) = 0;
            virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3, Real> > &other, Real t, const TVector& velocity0, const TVector& velocity1, DistanceResult& result) = 0;
            //virtual Real GetDistance(Real t, const TVector& velocity0, const TVector& velocity1, DistanceResult& result)  /* = 0*/;
            //virtual Real GetSquaredDistance(Real fT, const TVector& velocity0, const TVector& velocity1, DistanceResult& result) /* = 0*/;
			
            // Derivative calculations for dynamic distance queries.  The defaults
            // use finite difference estimates
            //   f'(t) = (f(t+h)-f(t-h))/(2*h)
            // where h = DifferenceStep.  A derived class may override these and
            // provide implementations of exact formulas that do not require h.
            virtual Real GetDerivativeDistance(const DistanceComputable<Real, Vec<3,Real> >& other, Real t, const TVector& velocity0,
                const TVector& velocity1, DistanceResult& result);
            virtual Real GetDerivativeSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, Real t, const TVector& velocity0,
                const TVector& velocity1, DistanceResult& result);

            // Dynamic distance queries.  The function computes the smallest distance
            // between the two objects over the time interval [tmin,tmax].
            /*virtual Real GetDistance(Real tmin, Real tmax, const TVector& velocity0,
                const TVector& relocity1, DistanceResult& result);*/
            virtual Real GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, Real tmin, Real tmax, const TVector& velocity0,
				const TVector& velocity1, DistanceResult& result);
            virtual Real GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, Real fTMin, Real fTMax,
                const TVector& velocity0, const TVector& velocity1, DistanceResult& result);

            // For Newton's method and inverse parabolic interpolation.
            int MaximumIterations;  // default = 8
            Real ZeroThreshold;     // default = Math<Real>::ZERO_TOLERANCE

            // For derivative approximations.
            void SetDifferenceStep (Real differenceStep);  // default = 1e-03
            Real GetDifferenceStep () const;



        protected:
            DistanceComputable();

			Real mDifferenceStep, mInvTwoDifferenceStep;
	};

	typedef DistanceComputable<float, Vec<2, float> > Distance2f;
	typedef DistanceComputable<float, Vec<3, float> > Distance3f;
	typedef DistanceComputable<double, Vec<2, double> > Distance2d;
	typedef DistanceComputable<double, Vec<3, double> > Distance3d;

}

#endif // DISTANCECOMPUTABLE_H
