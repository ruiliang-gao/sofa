#include "SofaPBDNarrowPhaseCollisionIntersectors.h"

#include <SofaMeshCollision/TriangleModel.h>

using namespace sofa::simulation::PBDSimulation;
using namespace sofa;
using namespace sofa::component::collision;

template< class TFilter1, class TFilter2 >
inline int SofaPBDNarrowPhaseCollisionIntersectors::doIntersectionLinePoint(double dist2, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& q, core::collision::BaseIntersector::OutputVector* contacts, int id, int indexLine1, int indexPoint2, TFilter1 &f1, TFilter2 &f2, bool swapElems)
{
    std::cout<<"doIntersectionLinePoint is called"<<std::endl;
    const defaulttype::Vector3 AB = p2-p1;
    const defaulttype::Vector3 AQ = q -p1;
    double A;
    double b;
    A = AB*AB;
    b = AQ*AB;

    double alpha = 0.5;

    alpha = b/A;
    if (alpha < 0.0) alpha = 0.0;
    else if (alpha > 1.0) alpha = 1.0;

    defaulttype::Vector3 p,pq, qp;
    p = p1 + AB * alpha;
    pq = q-p;
    qp = p-q;
    if (pq.norm2() >= dist2)
        return 0;

    if (!f1.validLine(indexLine1, pq))
        return 0;

    if (!f2.validPoint(indexPoint2, qp))
        return 0;

    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);

    detection->id = id;
    if (swapElems)
    {
        detection->point[0]=q;
        detection->point[1]=p;
        detection->normal = -pq;
    }
    else
    {
        detection->point[0]=p;
        detection->point[1]=q;
        detection->normal = pq;
    }
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    return 1;
}

template< class TFilter1, class TFilter2 >
inline int SofaPBDNarrowPhaseCollisionIntersectors::doIntersectionPointPoint(double dist2, const defaulttype::Vector3& p, const defaulttype::Vector3& q, core::collision::BaseIntersector::OutputVector* contacts, int id, int indexPoint1, int indexPoint2, TFilter1 &f1, TFilter2 &f2)
{
    defaulttype::Vector3 pq;
    pq = q-p;
    if (pq.norm2() >= dist2)
        return 0;

    if (!f1.validPoint(indexPoint1, pq))
        return 0;

    defaulttype::Vector3 qp = p-q;
    if (!f2.validPoint(indexPoint2, qp))
        return 0;

    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    detection->id = id;
    detection->point[0]=p;
    detection->point[1]=q;
    detection->normal=pq;
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    return 1;
}

template< class TFilter1, class TFilter2 >
inline int SofaPBDNarrowPhaseCollisionIntersectors::doIntersectionTrianglePoint(double dist2, int flags, const defaulttype::Vector3& p1, const defaulttype::Vector3& p2, const defaulttype::Vector3& p3, const defaulttype::Vector3& /*n*/, const defaulttype::Vector3& q, core::collision::BaseIntersector::OutputVector* contacts, int id,  component::collision::Triangle &e1, unsigned int *edgesIndices, int indexPoint2, TFilter1 &f1, TFilter2 &f2, bool swapElems)
{
    const defaulttype::Vector3 AB = p2-p1;
    const defaulttype::Vector3 AC = p3-p1;
    const defaulttype::Vector3 AQ = q -p1;
    sofa::defaulttype::Matrix2 A;
    sofa::defaulttype::Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const double det = determinant(A);

    double alpha = 0.5;
    double beta = 0.5;

    if(det==0.0)
    {
        msg_warning("LMDNewProximityIntersection")<<"(doIntersectionTrianglePoint) point is just on the triangle or the triangle do not exists: computation impossible";
        return 0;
    }


    alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
    beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
    defaulttype::Vector3 pq;
    defaulttype::Vector3 p;
    if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
    {
        // nearest point is on an edge or corner
        // barycentric coordinate on AB
        double pAB = b[0] / A[0][0]; // AQ*AB / AB*AB
        // barycentric coordinate on AC
        double pAC = b[1] / A[1][1]; // AQ*AB / AB*AB
        if (pAB < 0.000001 && pAC < 0.0000001)
        {
            ///////////////////////
            // closest point is A
            ///////////////////////
            if (!(flags&TriangleModel::FLAG_P1)) return 0; // this corner is not considered
            alpha = 0.0;
            beta = 0.0;
            //p = p1 + AB * alpha + AC * beta;
            pq = q-p1;
            if (pq.norm2() >= dist2)
                return 0;

            if (!f1.validPoint(e1.p1Index(), pq))
                return 0;


        }
        else if (pAB < 0.999999 && beta < 0.000001)
        {
            ///////////////////////////
            // closest point is on AB : convention edgesIndices 0
            ///////////////////////////
            if (!(flags&TriangleModel::FLAG_E12)) return 0; // this edge is not considered
            alpha = pAB;
            beta = 0.0;
            pq = q-p1 - AB*alpha;// p= p1 + AB * alpha + AC * beta;
            if (pq.norm2() >= dist2)
                return 0;

            if (!f1.validLine(edgesIndices[0], pq))
                return 0;
        }
        else if (pAC < 0.999999 && alpha < 0.000001)
        {
            ///////////////////////////
            // closest point is on AC: convention edgesIndices 2
            ///////////////////////////
            if (!(flags&TriangleModel::FLAG_E31)) return 0; // this edge is not considered
            alpha = 0.0;
            beta = pAC;
            pq = q-p1 - AC*beta;// p= p1 + AB * alpha + AC * beta;
            if (pq.norm2() >= dist2)
                return 0;

            if (!f1.validLine(edgesIndices[2], pq))
                return 0;
        }
        else
        {
            // barycentric coordinate on BC
            // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
            double pBC = (b[1] - b[0] + A[0][0] - A[0][1]) / (A[0][0] + A[1][1] - 2*A[0][1]); // BQ*BC / BC*BC
            if (pBC < 0.000001)
            {
                //////////////////////
                // closest point is B
                //////////////////////
                if (!(flags&TriangleModel::FLAG_P2)) return 0; // this point is not considered
                alpha = 1.0;
                beta = 0.0;
                pq = q-p2;
                if (pq.norm2() >= dist2)
                    return 0;

                if (!f1.validPoint(e1.p2Index(), pq))
                    return 0;
            }
            else if (pBC > 0.999999)
            {
                // closest point is C
                if (!(flags&TriangleModel::FLAG_P3)) return 0; // this point is not considered
                alpha = 0.0;
                beta = 1.0;
                pq = q-p3;
                if (pq.norm2() >= dist2)
                    return 0;

                if (!f1.validPoint(e1.p3Index(), pq))
                    return 0;
            }
            else
            {
                ///////////////////////////
                // closest point is on BC: convention edgesIndices 1
                ///////////////////////////
                if (!(flags&TriangleModel::FLAG_E23)) return 0; // this edge is not considered
                alpha = 1.0-pBC;
                beta = pBC;
                pq = q-p1 - AB * alpha - AC * beta;
                if (pq.norm2() >= dist2)
                    return 0;

                if (!f1.validLine(edgesIndices[1], pq))
                    return 0;
            }
        }
    }
    else
    {
        // nearest point is on the triangle

        p = p1 + AB * alpha + AC * beta;
        pq = q-p;
        if (pq.norm2() >= dist2)
            return 0;

        if (!f1.validTriangle(e1.getIndex(), pq))
            return 0;
    }

    p = p1 + AB * alpha + AC * beta;
    defaulttype::Vector3 qp = p-q;
    if (!f2.validPoint(indexPoint2, qp))
        return 0;

    contacts->resize(contacts->size()+1);
    sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    detection->id = id;
    if (swapElems)
    {
        detection->point[0]=q;
        detection->point[1]=p;
        detection->normal = -pq;
    }
    else
    {
        detection->point[0]=p;
        detection->point[1]=q;
        detection->normal = pq;
    }
    detection->value = detection->normal.norm();
    detection->normal /= detection->value;
    return 1;
}
