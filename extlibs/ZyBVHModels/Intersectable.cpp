#include "Intersectable.h"

namespace BVHModels
{
    //----------------------------------------------------------------------------
    template <typename Real, typename TVector>
    Intersectable<Real,TVector>::Intersectable ()
    {
        //mContactTime = (Real)0;
        //mIntersectionType = IT_EMPTY;
    }
    //----------------------------------------------------------------------------
    template <typename Real, typename TVector>
    Intersectable<Real,TVector>::~Intersectable ()
    {
    }

    /*//----------------------------------------------------------------------------
    template <typename Real, typename TVector>
    Real Intersectable<Real,TVector>::GetContactTime () const
    {
        return mContactTime;
    }
    //----------------------------------------------------------------------------
    template <typename Real, typename TVector>
    int Intersectable<Real,TVector>::GetIntersectionType () const
    {
        return mIntersectionType;
    }*/

    //----------------------------------------------------------------------------
    template <typename Real, typename TVector>
    bool Intersectable<Real,TVector>::Test(const Intersectable<Real, Vec<3,Real> >&)
    {
        // Stub for derived class.
        // assertion(false, "Function not yet implemented\n");
        return false;
    }
    //----------------------------------------------------------------------------
    template <typename Real, typename TVector>
    bool Intersectable<Real,TVector>::Find(const Intersectable<Real, Vec<3,Real> >&, IntersectionResult<Real>&)
    {
        // Stub for derived class.
        // assertion(false, "Function not yet implemented\n");
        return false;
    }
    //----------------------------------------------------------------------------
    template <typename Real, typename TVector>
    bool Intersectable<Real,TVector>::Test (const Intersectable<Real, Vec<3,Real> >&, Real, const TVector&, const TVector&)
    {
        // Stub for derived class.
        // assertion(false, "Function not yet implemented\n");
        return false;
    }
    //----------------------------------------------------------------------------
    template <typename Real, typename TVector>
    bool Intersectable<Real,TVector>::Find (const Intersectable<Real, Vec<3, Real> > &, Real, const TVector&, const TVector&, IntersectionResult<Real>&)
    {
        // Stub for derived class.
        // assertion(false, "Function not yet implemented\n");
        return false;
    }
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // Explicit instantiation.
    //----------------------------------------------------------------------------

    /*
    template 
    class Intersectable<float,Vec<2, float> >;

    template 
    class Intersectable<float,Vec<3,float> >;

    template 
    class Intersectable<double,Vec<2,double> >;

    template 
    class Intersectable<double,Vec<3,double> >;
    */

    template class IntersectionResult<SReal>;

    template
    class Intersectable<SReal,Vec<2,SReal> >;

    template
    class Intersectable<SReal,Vec<3,SReal> >;
    //----------------------------------------------------------------------------
}
