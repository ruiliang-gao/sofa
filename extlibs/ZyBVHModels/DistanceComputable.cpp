#include "DistanceComputable.h"
#include "Math/MathUtils.h"

using namespace BVHModels;

//----------------------------------------------------------------------------
double DistanceResult::GetContactTimeMinDistance () const
{
    return mContactTime;
}

//----------------------------------------------------------------------------
const Vector3& DistanceResult::GetClosestPoint0 () const
{
    return mClosestPoint0;
}

//----------------------------------------------------------------------------
const Vector3& DistanceResult::GetClosestPoint1 () const
{
    return mClosestPoint1;
}

//----------------------------------------------------------------------------
bool DistanceResult::HasMultipleClosestPoints0 () const
{
    return mHasMultipleClosestPoints0;
}

//----------------------------------------------------------------------------
bool DistanceResult::HasMultipleClosestPoints1 () const
{
    return mHasMultipleClosestPoints1;
}


template <typename Real, typename TVector>
DistanceComputable<Real,TVector>::DistanceComputable()
    :
    MaximumIterations(8),
    ZeroThreshold(MathUtils<Real>::ZERO_TOLERANCE)
    //mContactTime(MathUtils<Real>::MAX_REAL),
    //mHasMultipleClosestPoints0(false),
    //mHasMultipleClosestPoints1(false)
{
    SetDifferenceStep((Real)1e-03);
}
//----------------------------------------------------------------------------
template <typename Real, typename TVector>
DistanceComputable<Real,TVector>::~DistanceComputable()
{

}

//----------------------------------------------------------------------------
template <typename Real, typename TVector>
Real DistanceComputable<Real,TVector>::GetDifferenceStep() const
{
    return mDifferenceStep;
}


//----------------------------------------------------------------------------
template <typename Real, typename TVector>
void DistanceComputable<Real,TVector>::SetDifferenceStep (Real differenceStep)
{
    if (differenceStep > (Real)0)
    {
        mDifferenceStep = differenceStep;
    }
    else
    {
        //assertion(differenceStep > (Real)0, "Invalid difference step\n");
        mDifferenceStep = (Real)1e-03;
    }

    mInvTwoDifferenceStep = ((Real)0.5)/mDifferenceStep;
}


//----------------------------------------------------------------------------
template <typename Real, typename TVector>
Real DistanceComputable<Real,TVector>::GetDerivativeDistance (const DistanceComputable<Real, Vec<3,Real> >& other, Real t,
    const TVector& velocity0, const TVector& velocity1, DistanceResult& result)
{
    DistanceResult dRes1, dRes2;
    // Use a finite difference approximation:  f'(t) = (f(t+h)-f(t-h))/(2*h)
    Real funcp = GetDistance(other, t + mDifferenceStep, velocity0, velocity1, dRes1);
    Real funcm = GetDistance(other, t - mDifferenceStep, velocity0, velocity1, dRes2);
    Real derApprox = mInvTwoDifferenceStep*(funcp - funcm);
    return derApprox;
}

//----------------------------------------------------------------------------
template <typename Real, typename TVector>
Real DistanceComputable<Real,TVector>::GetDerivativeSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, Real t,
    const TVector& velocity0, const TVector& velocity1, DistanceResult& result)
{
    // A derived class should override this only if there is a faster method
    // to compute the derivative of the squared DistanceComputable for the specific
    // class.
    DistanceResult dRes1;
    Real DistanceComputable = GetDistance(other, t, velocity0, velocity1, dRes1);
    dRes1.reset();
    Real derivative = GetDerivativeDistance(other, t, velocity0, velocity1, dRes1);
    return ((Real)2) * DistanceComputable * derivative;
}
//----------------------------------------------------------------------------
template <typename Real, typename TVector>
Real DistanceComputable<Real, TVector>::GetDistance(const DistanceComputable<Real, Vec<3,Real> >& other, Real tmin, Real tmax,
	const TVector& velocity0, const TVector& velocity1, DistanceResult& result)
{
	// The assumption is that DistanceComputable f(t) is a convex function.  If
	// f'(tmin) >= 0, then the minimum occurs at tmin.  If f'(tmax) <= 0,
	// then the minimum occurs at tmax.  Otherwise, f'(0) < 0 and
	// f'(tmax) > 0 and the minimum occurs at some t in (tmin,tmax).

	DistanceResult dRes1;
	Real t0 = tmin;

    Real f0 = GetDistance(other, t0, velocity0, velocity1, dRes1);
	if (f0 <= ZeroThreshold)
	{
		// The DistanceComputable is effectively zero.  The objects are initially in
		// contact.
		result.mContactTime = t0;
		return (Real)0;
	}

	dRes1.reset();
    Real df0 = GetDerivativeDistance(other, t0, velocity0, velocity1, dRes1);
	if (df0 >= (Real)0)
	{
		// The DistanceComputable is increasing on [0,tmax].
		result.mContactTime = t0;
		return f0;
	}

	dRes1.reset();
	Real t1 = tmax;
    Real f1 = GetDistance(other, t1, velocity0, velocity1, dRes1);
	if (f1 <= ZeroThreshold)
	{
		// The DistanceComputable is effectively zero.
		result.mContactTime = t1;
		return (Real)0;
	}

	dRes1.reset();
    Real df1 = GetDerivativeDistance(other, t1, velocity0, velocity1, dRes1);
	if (df1 <= (Real)0)
	{
		// The DistanceComputable is decreasing on [0,tmax].
		result.mContactTime = t1;
		return f1;
	}

	// Start the process with Newton's method for computing a time when the
	// DistanceComputable is zero.  During this process we will switch to a numerical
	// minimizer if we decide that the DistanceComputable cannot be zero.
	int i;
	for (i = 0; i < MaximumIterations; ++i)
	{
		// Compute the next Newton's iterate.
		Real t = t0 - f0 / df0;
		if (t >= tmax)
		{
			// The convexity of the graph guarantees that when this condition
			// happens, the DistanceComputable is always positive.  Switch to a
			// numerical minimizer.
			break;
		}

		dRes1.reset();
        Real f = GetDistance(other, t, velocity0, velocity1, dRes1);
		if (f <= ZeroThreshold)
		{
			// The DistanceComputable is effectively zero.
			result.mContactTime = t;
			return (Real)0;
		}

		dRes1.reset();
        Real df = GetDerivativeDistance(other, t, velocity0, velocity1, dRes1);
		if (df >= (Real)0)
		{
			// The convexity of the graph guarantees that when this condition
			// happens, the DistanceComputable is always positive.  Switch to a
			// numerical minimizer.
			break;
		}

		t0 = t;
		f0 = f;
		df0 = df;
	}

	if (i == MaximumIterations)
	{
		// Failed to converge within desired number of iterations.  To
		// reach here, the derivative values were always negative, so report
		// the DistanceComputable at the last time.
		result.mContactTime = t0;
		return f0;
	}

	// The DistanceComputable is always positive.  Use bisection to find the root of
	// the derivative function.
	Real tm = t0;
	for (i = 0; i < MaximumIterations; ++i)
	{
		tm = ((Real)0.5)*(t0 + t1);
		dRes1.reset();
        Real dfm = GetDerivativeDistance(other, tm, velocity0, velocity1, dRes1);
		Real product = dfm * df0;
		if (product < -ZeroThreshold)
		{
			t1 = tm;
			df1 = dfm;
		}
		else if (product > ZeroThreshold)
		{
			t0 = tm;
			df0 = dfm;
		}
		else
		{
			break;
		}
	}

	// This is the time at which the minimum occurs and is not the contact
	// time.  Store it anyway for debugging purposes.
	result.mContactTime = tm;
	dRes1.reset();
    Real fm = GetDistance(other, tm, velocity0, velocity1, dRes1);
	return fm;
}
//----------------------------------------------------------------------------

template <typename Real, typename TVector>
Real DistanceComputable<Real,TVector>::GetSquaredDistance(const DistanceComputable<Real, Vec<3,Real> >& other, Real tmin, Real tmax,
    const TVector& velocity0, const TVector& velocity1, DistanceResult& result)
{
    // The assumption is that DistanceComputable f(t) is a convex function.  If
    // f'(tmin) >= 0, then the minimum occurs at tmin.  If f'(tmax) <= 0,
    // then the minimum occurs at tmax.  Otherwise, f'(0) < 0 and
    // f'(tmax) > 0 and the minimum occurs at some t in (tmin,tmax).

    DistanceResult dRes1;
    Real t0 = tmin;
    Real f0 = GetSquaredDistance(other, t0, velocity0, velocity1, dRes1);
    if (f0 <= ZeroThreshold)
    {
        // The DistanceComputable is effectively zero.  The objects are initially in
        // contact.
        result.mContactTime = t0;
        return (Real)0;
    }

    dRes1.reset();
    Real df0 = GetDerivativeSquaredDistance(other, t0, velocity0, velocity1, dRes1);
    if (df0 >= (Real)0)
    {
        // The DistanceComputable is increasing on [0,tmax].
        result.mContactTime = t0;
        return f0;
    }

    dRes1.reset();
    Real t1 = tmax;
    Real f1 = GetSquaredDistance(other, t1, velocity0, velocity1, dRes1);
    if (f1 <= ZeroThreshold)
    {
        // The DistanceComputable is effectively zero.
        result.mContactTime = t1;
        return (Real)0;
    }

    dRes1.reset();
    Real df1 = GetDerivativeSquaredDistance(other, t1, velocity0, velocity1, dRes1);
    if (df1 <= (Real)0)
    {
        // The DistanceComputable is decreasing on [0,tmax].
        result.mContactTime = t1;
        return f1;
    }

    // Start the process with Newton's method for computing a time when the
    // DistanceComputable is zero.  During this process we will switch to a numerical
    // minimizer if we decide that the DistanceComputable cannot be zero.
    int i;
    for (i = 0; i < MaximumIterations; ++i)
    {
        // Compute the next Newton's iterate.
        Real t = t0 - f0/df0;
        if (t >= tmax)
        {
            // The convexity of the graph guarantees that when this condition
            // happens, the DistanceComputable is always positive.  Switch to a
            // numerical minimizer.
            break;
        }

        dRes1.reset();
        Real f = GetSquaredDistance(other, t, velocity0, velocity1, dRes1);
        if (f <= ZeroThreshold)
        {
            // The DistanceComputable is effectively zero.
            result.mContactTime = t;
            return (Real)0;
        }

        dRes1.reset();
        Real df = GetDerivativeSquaredDistance(other, t, velocity0, velocity1, dRes1);
        if (df >= (Real)0)
        {
            // The convexity of the graph guarantees that when this condition
            // happens, the DistanceComputable is always positive.  Switch to a
            // numerical minimizer.
            break;
        }

        t0 = t;
        f0 = f;
        df0 = df;
    }

    if (i == MaximumIterations)
    {
        // Failed to converge within desired number of iterations.  To
        // reach here, the derivative values were always negative, so report
        // the DistanceComputable at the last time.
        result.mContactTime = t0;
        return f0;
    }

    // The DistanceComputable is always positive.  Use bisection to find the root of
    // the derivative function.
    Real tm = t0;
    for (i = 0; i < MaximumIterations; ++i)
    {
        tm = ((Real)0.5)*(t0 + t1);

        dRes1.reset();
        Real dfm = GetDerivativeSquaredDistance(other, tm, velocity0, velocity1, dRes1);
        Real product = dfm*df0;
        if (product < -ZeroThreshold)
        {
            t1 = tm;
            df1 = dfm;
        }
        else if (product > ZeroThreshold)
        {
            t0 = tm;
            df0 = dfm;
        }
        else
        {
            break;
        }
    }

    // This is the time at which the minimum occurs and is not the contact
    // time.  Store it anyway for debugging purposes.
    result.mContactTime = tm;

    dRes1.reset();
    Real fm = GetSquaredDistance(other, tm, velocity0, velocity1, dRes1);
	return fm;
}

/*
template <typename Real, typename TVector>
Real DistanceComputable<Real,TVector>::GetDistance(Real t, const TVector& velocity0, const TVector& velocity1, DistanceResult& result)
{
    return (Real)0;
}

template <typename Real, typename TVector>
Real DistanceComputable<Real,TVector>::GetSquaredDistance(Real fT, const TVector& velocity0, const TVector& velocity1, DistanceResult& result)
{
    return (Real)0;
}
*/

	//----------------------------------------------------------------------------
	// Explicit instantiation.
	//----------------------------------------------------------------------------


template 
class DistanceComputable<float, Vec<2, float> >;

template 
class DistanceComputable<float, Vec<3, float> >;

template 
class DistanceComputable<double, Vec<2, double> >;

template 
class DistanceComputable<double, Vec<3, double> >;


//----------------------------------------------------------------------------
