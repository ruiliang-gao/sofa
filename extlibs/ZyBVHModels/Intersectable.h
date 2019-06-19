#ifndef BVHMODELS_INTERSECTABLE_H
#define BVHMODELS_INTERSECTABLE_H

#include <sofa/defaulttype/Vec.h>

namespace BVHModels
{
    using namespace sofa::defaulttype;

    enum IntersectionType
    {
        IT_EMPTY,
        IT_POINT,
        IT_SEGMENT,
        IT_RAY,
        IT_LINE,
        IT_POLYGON,
        IT_PLANE,
        IT_POLYHEDRON,
        IT_AABB,
        IT_OTHER
    };

    enum PrimitiveType
    {
        PT_NONE,
        PT_POINT3,
        PT_SEGMENT2,
        PT_SEGMENT3,
        PT_LINE2,
        PT_LINE3,
        PT_PLANE,
        PT_AABB,
        PT_OBB,
        PT_TRIANGLE2,
        PT_TRIANGLE3,
        PT_RECTANGLE3,
        PT_CAPSULE3
    };

	template <typename Real>
    class IntersectionResult
    {
        public:
            IntersectionResult()
            {
                intersectionType = IT_EMPTY;
                primitiveType1 = PT_NONE;
                primitiveType2 = PT_NONE;
            }

            // The time at which two objects are in first contact for the dynamic
            // intersection queries.
            double GetContactTime() const { return mContactTime; }

            // Information about the intersection set
            IntersectionType GetIntersectionType() const { return intersectionType; }
            PrimitiveType GetIntersectingPrimitiveType1() const { return primitiveType1; }
            PrimitiveType GetIntersectingPrimitiveType2() const { return primitiveType2; }

            void SetIntersectionType(IntersectionType iType) { intersectionType = iType; }
            void SetIntersectingPrimitiveType1(PrimitiveType pType) { primitiveType1 = pType; }
            void SetIntersectingPrimitiveType2(PrimitiveType pType) { primitiveType2 = pType; }

            int GetMIntersectionType() const { return mIntersectionType; }
            void SetMIntersectionType(int iType) { mIntersectionType = iType; }

        protected:
            IntersectionType intersectionType;
            PrimitiveType primitiveType1;
            PrimitiveType primitiveType2;

            int mIntersectionType;
            double mContactTime;
    };

    template <typename Real, typename TVector>
    class Intersectable
    {
    public:
        // Abstract base class.
        virtual ~Intersectable();

        // Static intersection queries.  The default implementations return
        // 'false'.  The Find query produces a set of intersection.  The derived
        // class is responsible for providing access to that set, since the nature
        // of the set is dependent on the object types.
        virtual bool Test(const Intersectable<Real, Vec<3,Real> >&);
        virtual bool Find(const Intersectable<Real, Vec<3,Real> >&, IntersectionResult<Real>&);

        // Dynamic intersection queries.  The default implementations return
        // 'false'.  The Find query produces a set of first contact.  The derived
        // class is responsible for providing access to that set, since the nature
        // of the set is dependent on the object types.
        virtual bool Test (const Intersectable<Real, Vec<3,Real> >&, Real tmax, const TVector& velocity0,
            const TVector& velocity1);
        virtual bool Find (const Intersectable<Real, Vec<3,Real> >&, Real tmax, const TVector& velocity0,
            const TVector& velocity1, IntersectionResult<Real>&);

		virtual PrimitiveType GetIntersectableType() const = 0;
        virtual bool IsIntersectionQuerySupported(const PrimitiveType& other) { return false; }

		int mIntersectionType;
		Real mContactTime;

    protected:
        Intersectable();

    };

    typedef Intersectable<float, Vec<2,float> > Intersector2f;
    typedef Intersectable<float, Vec<3,float> > Intersector3f;
    typedef Intersectable<double, Vec<2,double> > Intersector2d;
    typedef Intersectable<double, Vec<3,double> > Intersector3d;

}

#endif //BVHMODELS_INTERSECTABLE_H
