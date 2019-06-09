#ifndef INTERVALUTILS_H
#define INTERVALUTILS_H

#include <sofa/defaulttype/Vec.h>

#include "MathUtils.h"

#include "Primitives/Segment3.h"
#include "Primitives/Box3.h"
#include "Primitives/Triangle3.h"

using namespace sofa::defaulttype;

namespace BVHModels
{
    //----------------------------------------------------------------------------
    template <typename Real>
    class IntrConfiguration
    {
    public:
        // ContactSide (order of the intervals of projection).
        enum
        {
            LEFT,
            RIGHT,
            NONE
        };

        // VertexProjectionMap (how the vertices are projected to the minimum
        // and maximum points of the interval).
        enum
        {
            m2, m11,             // segments
            m3, m21, m12, m111,  // triangles
            m44, m2_2, m1_1      // boxes
        };

        // The VertexProjectionMap value for the configuration.
        int mMap;

        // The order of the vertices.
        int mIndex[8];

        // Projection interval.
        Real mMin, mMax;
    };
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    template <typename Real>
    class IntrAxis
    {
    public:
        // Test-query for intersection of projected intervals.  The velocity
        // input is the difference objectVelocity1 - objectVelocity0.  The
        // first and last times of contact are computed.
        static bool Test (const Vec<3,Real>& axis,
            const Vec<3,Real> segment[2], const Triangle3<Real>& triangle,
            const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast);

        static bool Test (const Vec<3,Real>& axis,
            const Vec<3,Real> segment[2], const Box3<Real>& box,
            const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast);

        static bool Test (const Vec<3,Real>& axis,
            const Triangle3<Real>& triangle, const Box3<Real>& box,
            const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast);

        static bool Test (const Vec<3,Real>& axis,
            const Box3<Real>& box0, const Box3<Real>& box1,
            const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast);

        // Find-query for intersection of projected intervals.  The velocity
        // input is the difference objectVelocity1 - objectVelocity0.  The
        // first and last times of contact are computed, as is information about
        // the contact configuration and the ordering of the projections (the
        // contact side).
        static bool Find (const Vec<3,Real>& axis,
            const Vec<3,Real> segment[2], const Triangle3<Real>& triangle,
            const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast,
            int& side, IntrConfiguration<Real>& segCfgFinal,
            IntrConfiguration<Real>& triCfgFinal);

        static bool Find (const Vec<3,Real>& axis,
            const Vec<3,Real> segment[2], const Box3<Real>& box,
            const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast,
            int& side, IntrConfiguration<Real>& segCfgFinal,
            IntrConfiguration<Real>& boxCfgFinal);

        static bool Find (const Vec<3,Real>& axis,
            const Triangle3<Real>& triangle, const Box3<Real>& box,
            const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast,
            int& side, IntrConfiguration<Real>& triCfgFinal,
            IntrConfiguration<Real>& boxCfgFinal);

        static bool Find (const Vec<3,Real>& axis,
            const Box3<Real>& box0, const Box3<Real>& box1,
            const Vec<3,Real>& velocity, Real tmax, Real& tfirst, Real& tlast,
            int& side, IntrConfiguration<Real>& box0CfgFinal,
            IntrConfiguration<Real>& box1CfgFinal);

        // Projections.
        static void GetProjection (const Vec<3,Real>& axis,
            const Vec<3,Real> segment[2], Real& imin, Real& imax);

        static void GetProjection (const Vec<3,Real>& axis,
            const Triangle3<Real>& triangle, Real& imin, Real& imax);

        static void GetProjection (const Vec<3,Real>& axis,
            const Box3<Real>& box, Real& imin, Real& imax);

        // Configurations.
        static void GetConfiguration (const Vec<3,Real>& axis,
            const Vec<3,Real> segment[2], IntrConfiguration<Real>& cfg);

        static void GetConfiguration (const Vec<3,Real>& axis,
            const Triangle3<Real>& triangle, IntrConfiguration<Real>& cfg);

        static void GetConfiguration (const Vec<3,Real>& axis,
            const Box3<Real>& box, IntrConfiguration<Real>& cfg);

        // Low-level test-query for projections.
        static bool Test (const Vec<3,Real>& axis,
            const Vec<3,Real>& velocity, Real min0, Real max0, Real min1,
            Real max1, Real tmax, Real& tfirst, Real& tlast);

        // Low-level find-query for projections.
        static bool Find (const Vec<3,Real>& axis,
            const Vec<3,Real>& velocity,
            const IntrConfiguration<Real>& cfg0Start,
            const IntrConfiguration<Real>& cfg1Start, Real tmax, int& side,
            IntrConfiguration<Real>& cfg0Final,
            IntrConfiguration<Real>& cfg1Final, Real& tfirst, Real& tlast);
    };
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    template <typename Real>
    class FindContactSet
    {
    public:
        static void FindContactSet_Segment3Triangle3 (const Vec<3,Real> segment[2],
            const Triangle3<Real>& triangle, int side,
            const IntrConfiguration<Real>& segCfg,
            const IntrConfiguration<Real>& triCfg,
            const Vec<3,Real>& segVelocity, const Vec<3,Real>& triVelocity,
            Real tfirst, int& quantity, Vec<3,Real>* P);

		static void FindContactSet_Segment3Box3(const Vec<3, Real> segment[2], const Box3<Real>& box,
            int side, const IntrConfiguration<Real>& segCfg,
            const IntrConfiguration<Real>& boxCfg,
            const Vec<3,Real>& segVelocity, const Vec<3,Real>& boxVelocity,
            Real tfirst, int& quantity, Vec<3,Real>* P);

		static void FindContactSet_Triangle3Box3(const Triangle3<Real>& triangle,
            const Box3<Real>& box, int side,
            const IntrConfiguration<Real>& triCfg,
            const IntrConfiguration<Real>& boxCfg,
            const Vec<3,Real>& triVelocity, const Vec<3,Real>& boxVelocity,
            Real tfirst, int& quantity, Vec<3,Real>* P);

		static void FindContactSet_Box3Box3(const Box3<Real>& box0, const Box3<Real>& box1,
            int side, const IntrConfiguration<Real>& box0Cfg,
            const IntrConfiguration<Real>& box1Cfg,
            const Vec<3,Real>& box0Velocity,
            const Vec<3,Real>& box1Velocity, Real tfirst, int& quantity,
            Vec<3,Real>* P);

    private:
        // These functions are called when it is known that the features are
        // intersecting.  Consequently, they are specialized versions of the
        // object-object intersection algorithms.

        static void ColinearSegments (const Vec<3,Real> segment0[2],
            const Vec<3,Real> segment1[2], int& quantity, Vec<3,Real>* P);

        static void SegmentThroughPlane (const Vec<3,Real> segment[2],
            const Vec<3,Real>& planeOrigin, const Vec<3,Real>& planeNormal,
            int& quantity, Vec<3,Real>* P);

        static void SegmentSegment (const Vec<3,Real> segment0[2],
            const Vec<3,Real> segment1[2], int& quantity, Vec<3,Real>* P);

        static void ColinearSegmentTriangle (const Vec<3,Real> segment[2],
            const Vec<3,Real> triangle[3], int& quantity, Vec<3,Real>* P);

        static void CoplanarSegmentRectangle (const Vec<3,Real> segment[2],
            const Vec<3,Real> rectangle[4], int& quantity, Vec<3,Real>* P);

        static void CoplanarTriangleRectangle (const Vec<3,Real> triangle[3],
            const Vec<3,Real> rectangle[4], int& quantity, Vec<3,Real>* P);

        static void CoplanarRectangleRectangle (
            const Vec<3,Real> rectangle0[4],
            const Vec<3,Real> rectangle1[4], int& quantity, Vec<3,Real>* P);
    };
    //----------------------------------------------------------------------------

    //----------------------------------------------------------------------------
    // Miscellaneous support.
    //----------------------------------------------------------------------------
    // The input and output polygons are stored in P.  The size of P is
    // assumed to be large enough to store the clipped convex polygon vertices.
    // For now the maximum array size is 8 to support the current intersection
    // algorithms.
    template <typename Real> 
    void ClipConvexPolygonAgainstPlane (const Vec<3,Real>& normal,
        Real bonstant, int& quantity, Vec<3,Real>* P);

    // Translates an index into the box back into real coordinates.
    template <typename Real> 
    Vec<3,Real> GetPointFromIndex (int index, const Box3<Real>& box);
    //----------------------------------------------------------------------------

    template <typename Real>
    class Interval1Intersector
    {
    public:
        // A class for intersection of intervals [u0,u1] and [v0,v1].  The end
        // points must be ordered:  u0 <= u1 and v0 <= v1.  Values of MAX_REAL
        // and -MAX_REAL are allowed, and degenerate intervals are allowed:
        // u0 = u1 or v0 = v1.
        Interval1Intersector (Real u0, Real u1, Real v0, Real v1);
        Interval1Intersector (Real u[2], Real v[2]);
        ~Interval1Intersector ();

        // Object access.
        Real GetU (int i) const;
        Real GetV (int i) const;

        // Static intersection queries.
        bool Test ();
        bool Find ();

        // Dynamic intersection queries.  The Find query produces a set of first
        // contact.
        bool Test (Real tmax, Real speedU, Real speedV);
        bool Find (Real tmax, Real speedU, Real speedV);

        // The time at which two intervals are in first/last contact for the
        // dynamic intersection queries.
        Real GetFirstTime () const;
        Real GetLastTime () const;

        // Information about the intersection set.  The number of intersections
        // is 0 (intervals do not overlap), 1 (intervals are just touching), or
        // 2 (intervals intersect in an inteval).
        int GetNumIntersections () const;
        Real GetIntersection (int i) const;

    protected:
        // The intervals to intersect.
        Real mU[2], mV[2];

        // Information about the intersection set.
        Real mFirstTime, mLastTime;
        int mNumIntersections;
        Real mIntersections[2];
    };
}

#endif // INTERVALUTILS_H
