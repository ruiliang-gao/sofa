#include "TriangleIntersection.h"

#include <cmath>

using namespace sofa::component::collision;

TriangleIntersection::TriangleIntersection()
{
}

enum Matrix4Indices
{
    R00 = 0,  R10 = 1,  R20 = 2, HOMOGENEOUS_LINE_1 = 3,
    R01 = 4,  R11 = 5,  R21 = 6, HOMOGENEOUS_LINE_2 = 7,
    R02 = 8,  R12 = 9,  R22 = 10,HOMOGENEOUS_LINE_3 = 11,
    TX  = 12, TY  = 13, TZ  = 14,HOMOGENEOUS_LINE_4 = 15,
};

/*[0 1 2 3,
 4 5 6 7,
 8 9 10 11,
 12 13 14 15]*/
typedef std::map<Matrix4Indices, std::pair<int,int> > Matrix4IndicesMap;

static const Matrix4IndicesMap::value_type matrixHelperIndices[16] =
{
    Matrix4IndicesMap::value_type(R00, std::pair<int,int>(0,0)),
    Matrix4IndicesMap::value_type(R10, std::pair<int,int>(0,1)),
    Matrix4IndicesMap::value_type(R20, std::pair<int,int>(0,2)),
    Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_1, std::pair<int,int>(0,3)),
    Matrix4IndicesMap::value_type(R01, std::pair<int,int>(1,0)),
    Matrix4IndicesMap::value_type(R11, std::pair<int,int>(1,1)),
    Matrix4IndicesMap::value_type(R21, std::pair<int,int>(1,2)),
    Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_2, std::pair<int,int>(1,3)),
    Matrix4IndicesMap::value_type(R02, std::pair<int,int>(2,0)),
    Matrix4IndicesMap::value_type(R12, std::pair<int,int>(2,1)),
    Matrix4IndicesMap::value_type(R22, std::pair<int,int>(2,2)),
    Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_3, std::pair<int,int>(2,3)),

    Matrix4IndicesMap::value_type(TX, std::pair<int,int>(0,3)),
    Matrix4IndicesMap::value_type(TY, std::pair<int,int>(1,3)),
    Matrix4IndicesMap::value_type(TZ, std::pair<int,int>(2,3)),
    Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_4, std::pair<int,int>(3,3))
};

void TriIntersectMoeller::TRANSFORM_TRIANGLE(const Matrix4& t, Vector3& vert0, Vector3& vert1, Vector3& vert2)
{
    double tmp_x;
    double tmp_y;
    double tmp_z;
    tmp_x = vert0[0];
    tmp_y = vert0[1];
    tmp_z = vert0[2];
    vert0[0] = t[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] * tmp_x +
            t[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] * tmp_y +
            t[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] * tmp_z +
            t[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second];
    vert0[1] = t[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] * tmp_x +
            t[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] * tmp_y +
            t[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] * tmp_z +
            t[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second];
    vert0[2] = t[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] * tmp_x +
            t[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] * tmp_y +
            t[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] * tmp_z +
            t[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second];
    tmp_x = vert1[0];
    tmp_y = vert1[1];
    tmp_z = vert1[2];
    vert1[0] = t[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] * tmp_x +
            t[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] * tmp_y +
            t[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] * tmp_z +
            t[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second];
    vert1[1] = t[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] * tmp_x +
            t[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] * tmp_y +
            t[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] * tmp_z +
            t[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second];
    vert1[2] = t[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] * tmp_x +
            t[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] * tmp_y +
            t[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] * tmp_z +
            t[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second];
    tmp_x = vert2[0];
    tmp_y = vert2[1];
    tmp_z = vert2[2];
    vert2[0] = t[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] * tmp_x +
            t[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] * tmp_y +
            t[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] * tmp_z +
            t[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second];
    vert2[1] = t[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] * tmp_x +
            t[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] * tmp_y +
            t[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] * tmp_z +
            t[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second];
    vert2[2] = t[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] * tmp_x +
            t[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] * tmp_y +
            t[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] * tmp_z +
            t[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second];
}

//! sort so that a <= b
void TriIntersectMoeller::SORT(double& a, double& b)
{
    if ((a) > (b))
    {
        const double c = (a);
        (a) = (b);
        (b) = c;
    }
}

//! Edge to edge test based on Franlin Antonio's gem: "Faster Line Segment Intersection", in Graphics Gems III, pp. 199-202
bool TriIntersectMoeller::EDGE_EDGE_TEST(const Vector3& V0, const Vector3& U0, const Vector3& U1, const unsigned short& i0, const unsigned short& i1, double& Ax, double& Ay, double& Bx, double& By, double& Cx, double& Cy, double& d, double& f)
{
    Bx = U0[i0] - U1[i0];
    By = U0[i1] - U1[i1];
    Cx = V0[i0] - U0[i0];
    Cy = V0[i1] - U0[i1];
    f  = Ay*Bx - Ax*By;
    d  = By*Cx - Bx*Cy;
    if((f > 0.0f && d >= 0.0f && d <= f) ||
       (f < 0.0f && d <= 0.0f && d >= f))
    {
        const double e = Ax*Cy - Ay*Cx;
        if (f > 0.0f)
        {
            if (e >= 0.0f && e <= f)
                return true;
        }
        else
        {
            if (e <= 0.0f && e >= f)
                return true;
        }
    }

    return false;
}
//! TO BE DOCUMENTED
bool TriIntersectMoeller::EDGE_AGAINST_TRI_EDGES(const Vector3& V0, const Vector3& V1, const Vector3& U0, const Vector3& U1, const Vector3& U2, const unsigned short& i0, const unsigned short& i1)
{
    double Bx,By,Cx,Cy,d,f;
    double Ax = V1[i0] - V0[i0];
    double Ay = V1[i1] - V0[i1];

    bool edgeEdgeResult;
    /* test edge U0,U1 against V0,V1 */
    edgeEdgeResult = EDGE_EDGE_TEST(V0, U0, U1, i0, i1, Ax, Ay, Bx, By, Cx, Cy, d, f);

    if (!edgeEdgeResult)
        return false;

    /* test edge U1,U2 against V0,V1 */
    edgeEdgeResult = EDGE_EDGE_TEST(V0, U1, U2, i0, i1, Ax, Ay, Bx, By, Cx, Cy, d, f);

    if (!edgeEdgeResult)
        return false;

    /* test edge U2,U1 against V0,V1 */
    edgeEdgeResult = EDGE_EDGE_TEST(V0, U2, U0, i0, i1, Ax, Ay, Bx, By, Cx, Cy, d, f);

    if (!edgeEdgeResult)
        return false;

    return true;
}

//! TO BE DOCUMENTED
bool TriIntersectMoeller::POINT_IN_TRI(const Vector3& V0, const Vector3& U0, const Vector3& U1, const Vector3& U2, const unsigned short& i0, const unsigned short& i1)
{
  /* is T1 completly inside T2? */
  /* check if V0 is inside tri(U0,U1,U2) */
    double a  = U1[i1] - U0[i1];
    double b  = -(U1[i0] - U0[i0]);
    double c  = -a*U0[i0] - b*U0[i1];
    const double d0 = a*V0[i0] + b*V0[i1] + c;

    a  = U2[i1] - U1[i1];
    b  = -(U2[i0] - U1[i0]);
    c  = -a*U1[i0] - b*U1[i1];
    const double d1 = a*V0[i0] + b*V0[i1] + c;

    a  = U0[i1] - U2[i1];
    b  = -(U0[i0] - U2[i0]);
    c  = -a*U2[i0] - b*U2[i1];
    const double d2 = a*V0[i0] + b*V0[i1] + c;
    if ((d0*d1 > 0.0f) && (d0*d2 > 0.0f))
        return true;

    return false;
}

//! TO BE DOCUMENTED
bool TriIntersectMoeller::CoplanarTriTri(const Vector3& n,
               const Vector3& v0, const Vector3& v1, const Vector3& v2,
               const Vector3& u0, const Vector3& u1, const Vector3& u2, unsigned short& i0, unsigned short& i1)
{
  Vector3 A;
  // unsigned short i0, i1;
  /* first project onto an axis-aligned plane, that maximizes the area */
  /* of the triangles, compute indices: i0,i1. */
  A[0] = std::fabs(n[0]);
  A[1] = std::fabs(n[1]);
  A[2] = std::fabs(n[2]);
  if (A[0] > A[1])
  {
      if (A[0] > A[2])
      {
          i0 = 1;      /* A[0] is greatest */
          i1 = 2;
      }
      else
      {
          i0 = 0;      /* A[2] is greatest */
          i1 = 1;
      }
  }
  else   /* A[0]<=A[1] */
  {
      if (A[2] > A[1])
      {
          i0 = 0;      /* A[2] is greatest */
          i1 = 1;
      }
      else
      {
          i0 = 0;      /* A[1] is greatest */
          i1 = 2;
      }
  }

  bool edgeTriResult;
  /* test all edges of triangle 1 against the edges of triangle 2 */
  edgeTriResult = EDGE_AGAINST_TRI_EDGES(v0, v1, u0, u1, u2, i0, i1);
  if (edgeTriResult)
      return true;

  edgeTriResult = EDGE_AGAINST_TRI_EDGES(v1, v2, u0, u1, u2, i0, i1);
  if (edgeTriResult)
      return true;

  edgeTriResult = EDGE_AGAINST_TRI_EDGES(v2, v0, u0, u1, u2, i0, i1);
  if (edgeTriResult)
      return true;

  /* finally, test if tri1 is totally contained in tri2 or vice versa */
  bool pointTriResult;
  pointTriResult = POINT_IN_TRI(v0, u0, u1, u2, i0, i1);
  if (pointTriResult)
      return true;

  pointTriResult = POINT_IN_TRI(u0, v0, v1, v2, i0, i1);
  if (pointTriResult)
      return true;

  return false;
}

//! TO BE DOCUMENTED
bool TriIntersectMoeller::NEWCOMPUTE_INTERVALS(double& VV0, double& VV1, double& VV2, double& D0, double& D1, double& D2, double& D0D1, double& D0D2, double& A, double& B, double& C, double& X0, double& X1)
{
    if (D0D1 > 0.0f)
    {
        /* here we know that D0D2<=0.0 */
        /* that is D0, D1 are on the same side, D2 on the other or on the plane */
        A=VV2; B=(VV0 - VV2)*D2; C=(VV1 - VV2)*D2; X0=D2 - D0; X1=D2 - D1;
    }
    else if (D0D2 > 0.0f)
    {
        /* here we know that d0d1<=0.0 */
        A=VV1; B=(VV0 - VV1)*D1; C=(VV2 - VV1)*D1; X0=D1 - D0; X1=D1 - D2;
    }
    else if (D1*D2 > 0.0f || D0 != 0.0f)
    {
        /* here we know that d0d1<=0.0 or that D0!=0.0 */
        A=VV0; B=(VV1 - VV0)*D0; C=(VV2 - VV0)*D0; X0=D0 - D1; X1=D0 - D2;
    }
    else if (D1 != 0.0f)
    {
        A=VV1; B=(VV0 - VV1)*D1; C=(VV2 - VV1)*D1; X0=D1 - D0; X1=D1 - D2;
    }
    else if (D2 != 0.0f)
    {
        A=VV2; B=(VV0 - VV2)*D2; C=(VV1 - VV2)*D2; X0=D2 - D0; X1=D2 - D1;
    }
    else
    {
        /* triangles are coplanar */
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 *  Triangle/triangle intersection test routine,
 *  by Tomas Moller, 1997.
 *  See article "A Fast Triangle-Triangle Intersection Test",
 *  Journal of Graphics Tools, 2(2), 1997
 *
 *  Updated June 1999: removed the divisions -- a little faster now!
 *  Updated October 1999: added {} to CROSS and SUB macros
 *
 *  int NoDivTriTriIsect(Real V0[3],Real V1[3],Real V2[3],
 *                      Real U0[3],Real U1[3],Real U2[3])
 *
 *  \param      V0      [in] triangle 0, vertex 0
 *  \param      V1      [in] triangle 0, vertex 1
 *  \param      V2      [in] triangle 0, vertex 2
 *  \param      U0      [in] triangle 1, vertex 0
 *  \param      U1      [in] triangle 1, vertex 1
 *  \param      U2      [in] triangle 1, vertex 2
 *  \return     true if triangles overlap
 */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define LOCAL_EPSILON ((double) 0.000001)
#define YAOBI_TRITRI_EPSILON_TEST

bool TriIntersectMoeller::tri_tri_overlap_3d(const Vector3 t1[], const Vector3 t2[])
{
    Vector3 E1;
    Vector3 E2;
    Vector3 N1;

    // Compute plane equation of triangle(V0,V1,V2)
    E1 = t1[1] - t1[0];
    E2 = t1[2] - t1[0];
    N1 = E1.cross(E2);

    std::cout << "   tri1 vertices: " << t1[0] << ", " << t1[1] << "," << t1[2] << std::endl;
    std::cout << "   tri2 vertices: " << t2[0] << ", " << t2[1] << "," << t2[2] << std::endl;
    std::cout << "   E1 = " << E1 << ", E2 = " << E2 << ", N1 = " << N1 << std::endl;

    const double d1 = -N1 * t1[0];
    // Plane equation 1: N1.X+d1=0
    std::cout << "   d1 = " << d1 << std::endl;

    // Put U0,U1,U2 into plane equation 1 to compute signed distances to the plane
    double du0 = (N1 * t2[0]) + d1;
    double du1 = (N1 * t2[1]) + d1;
    double du2 = (N1 * t2[2]) + d1;

    std::cout << "   du0 = " << du0 << ", du1 = " << du1 << ", du2 = " << du2  << std::endl;

    // Coplanarity robustness check
#ifdef YAOBI_TRITRI_EPSILON_TEST
    if (std::fabs(du0) < LOCAL_EPSILON) du0 = 0.0f;
    if (std::fabs(du1) < LOCAL_EPSILON) du1 = 0.0f;
    if (std::fabs(du2) < LOCAL_EPSILON) du2 = 0.0f;
#endif

    double du0du1 = du0 * du1;
    double du0du2 = du0 * du2;

    std::cout << "   du0du1 = " << du0du1 << ", du0du2 = " << du0du2 << std::endl;

    if (du0du1 > 0.0f && du0du2 > 0.0f)  // same sign on all of them + not equal 0 ?
    {
        std::cout << "  no overlap" << std::endl;
        return false;                      // no intersection occurs
    }
    // Compute plane of triangle (U0,U1,U2)
    Vector3 N2;
    E1 = t2[1] - t2[0];
    E2 = t2[2] - t2[0];
    N2 = E1.cross(E2);

    std::cout << "   E1 = " << E1 << ", E2 = " << E2 << ", N2 = " << N2 << std::endl;

    const double d2 = -N2 * t2[0];
    // plane equation 2: N2.X+d2=0
    std::cout << "   d2 = " << d2 << std::endl;

    // put V0,V1,V2 into plane equation 2
    double dv0 = (N2 * t1[0]) + d2;
    double dv1 = (N2 * t1[1]) + d2;
    double dv2 = (N2 * t1[2]) + d2;

    std::cout << "   dv0 = " << dv0 << ", dv1 = " << dv1 << ", dv2 = " << dv2 << std::endl;

#ifdef YAOBI_TRITRI_EPSILON_TEST
    if (std::fabs(dv0) < LOCAL_EPSILON) dv0 = 0.0f;
    if (std::fabs(dv1) < LOCAL_EPSILON) dv1 = 0.0f;
    if (std::fabs(dv2) < LOCAL_EPSILON) dv2 = 0.0f;
#endif

    double dv0dv1 = dv0 * dv1;
    double dv0dv2 = dv0 * dv2;

    std::cout << "   dv0dv1 = " << dv0dv1 << ", dv0dv2 = " << dv0dv2 << std::endl;

    if (dv0dv1 > 0.0f && dv0dv2 > 0.0f)  // same sign on all of them + not equal 0 ?
    {
        std::cout << "  no overlap" << std::endl;
        return false;                      // no intersection occurs
    }

    // Compute direction of intersection line
    Vector3 D = N1.cross(N2);

    std::cout << "  D = " << D << std::endl;

    // Compute and index to the largest component of D
    double max           = std::fabs(D[0]);
    unsigned short index = 0;
    double bb            = std::fabs(D[1]);
    double cc            = std::fabs(D[2]);
    if (bb > max) { max = bb; index = 1; }
    if (cc > max) { max = cc; index = 2; }

    std::cout << "   max_D = " << max << ", index = " << index << std::endl;

    // This is the simplified projection onto L
    double vp0 = t1[0][index];
    double vp1 = t1[1][index];
    double vp2 = t1[2][index];

    std::cout << "   vp0 = " << vp0 << ", vp1 = " << vp1 << ", vp2 = " << vp2 << std::endl;

    double up0 = t2[0][index];
    double up1 = t2[1][index];
    double up2 = t2[2][index];

    std::cout << "   up0 = " << up0 << ", up1 = " << up1 << ", up2 = " << up2 << std::endl;

    // Compute interval for triangle 1
    double a, b, c, x0, x1;
    unsigned short i0, i1;

    bool intervalResult = NEWCOMPUTE_INTERVALS(vp0,vp1,vp2,dv0,dv1,dv2,dv0dv1,dv0dv2,a,b,c,x0,x1);

    std::cout << "  intervalResult 1 = " << intervalResult << std::endl;
    std::cout << "   vp0 = " << vp0 << ", vp1 = " << vp1 << ", vp2 = " << vp2 << std::endl;
    std::cout << "   dv0 = " << dv0 << ", dv1 = " << dv1 << ", dv2 = " << dv2 << std::endl;
    std::cout << "   dv0dv1 = " << dv0dv1 << ", dv0dv2 = " << dv0dv2 << std::endl;
    std::cout << "   a = " << a << ", b = " << b  << ", c = " << c << ", x0 = " << x0 << ", x1 = " << x1 << std::endl;

    bool coplanarResult;
    if (!intervalResult)
    {
        coplanarResult = CoplanarTriTri(N1, t1[0], t1[1], t1[2], t2[0], t2[1], t2[2], i0, i1);
        std::cout << "  coplanarResult 1 = " << coplanarResult << std::endl;
        if (coplanarResult)
        {
            std::cout << "  triangles are coplanar" << std::endl;
            return true;
        }
    }

    // Compute interval for triangle 2
    double d, e, f, y0, y1;
    intervalResult = NEWCOMPUTE_INTERVALS(up0,up1,up2,du0,du1,du2,du0du1,du0du2,d,e,f,y0,y1);

    std::cout << "  intervalResult 2 = " << intervalResult << std::endl;
    std::cout << "   up0 = " << up0 << ", up1 = " << up1 << ", up2 = " << up2 << std::endl;
    std::cout << "   du0 = " << du0 << ", du1 = " << du1 << ", du2 = " << du2 << std::endl;
    std::cout << "   du0du1 = " << du0du1 << ", du0du2 = " << du0du2 << std::endl;
    std::cout << "   d = " << d << ", e = " << e  << ", f = " << f << ", x0 = " << y0 << ", y1 = " << y1 << std::endl;

    if (!intervalResult)
    {
        coplanarResult = CoplanarTriTri(N1, t1[0], t1[1], t1[2], t2[0], t2[1], t2[2], i0, i1);
        std::cout << "  coplanarResult 2 = " << coplanarResult << std::endl;
        if (coplanarResult)
        {
            std::cout << "  triangles are coplanar" << std::endl;
            return true;
        }
    }
    double xx   = x0*x1;
    double yy   = y0*y1;
    double xxyy = xx*yy;

    std::cout << "  xx = " << xx << ", yy = " << yy << ", xxyy = " << xxyy << std::endl;

    double isect1[2], isect2[2];

    double tmp =  a * xxyy;
    isect1[0]  =  tmp + b*x1*yy;
    isect1[1]  =  tmp + c*x0*yy;

    std::cout << "  isect1[0] = " << isect1[0] << ", isect1[1] = " << isect1[1] << std::endl;

    tmp       = d * xxyy;
    isect2[0] = tmp + e*xx*y1;
    isect2[1] = tmp + f*xx*y0;

    std::cout << "  isect2[0] = " << isect2[0] << ", isect2[1] = " << isect2[1] << std::endl;

    SORT(isect1[0],isect1[1]);
    SORT(isect2[0],isect2[1]);

    if (isect1[1] < isect2[0] || isect2[1] < isect1[0])
    {
        std::cout << " no overlap" << std::endl;
        return false;
    }

    std::cout << " overlap" << std::endl;
    return true;
}

bool TriIntersectMoeller::intersect_tri_tri(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &q0, const Vector3 &q1, const Vector3 &q2)
{
    Vector3 tri_verts_a[3], tri_verts_b[3];
    tri_verts_a[0] = p0; tri_verts_a[1] = p1; tri_verts_a[2] = p2;
    tri_verts_b[0] = q0; tri_verts_b[1] = q1; tri_verts_b[2] = q2;

    return tri_tri_overlap_3d(tri_verts_a, tri_verts_b);
}


/*
*
*  Triangle-Triangle Overlap Test Routines
*  July, 2002
*  Updated December 2003
*
*  This file contains C implementation of algorithms for
*  performing two and three-dimensional triangle-triangle intersection test
*  The algorithms and underlying theory are described in
*
* "Fast and Robust Triangle-Triangle Overlap Test
*  Using Orientation Predicates"  P. Guigue - O. Devillers
*
*  Journal of Graphics Tools, 8(1), 2003
*
*  Several geometric predicates are defined.  Their parameters are all
*  points.  Each point is an array of two or three Real precision
*  doubleing point numbers. The geometric predicates implemented in
*  this file are:
*
*    int tri_tri_overlap_test_3d(p1,q1,r1,p2,q2,r2)
*    int tri_tri_overlap_test_2d(p1,q1,r1,p2,q2,r2)
*
*    int tri_tri_intersection_test_3d(p1,q1,r1,p2,q2,r2,
*                                     coplanar,source,target)
*
*       is a version that computes the segment of intersection when
*       the triangles overlap (and are not coplanar)
*
*    each function returns 1 if the triangles (including their
*    boundary) intersect, otherwise 0
*
*
*  Other information are available from the Web page
*  http://www.acm.org/jgt/papers/GuigueDevillers03/
*
*/

bool TriIntersectGuigue::CHECK_MIN_MAX(const Vector3& p1, const Vector3& q1, const Vector3& r1, const Vector3& p2, const Vector3& q2, const Vector3& r2)
{
    //VEC_SUB(v1,p2,q1);
    Vector3 v1 = p2 - q1;
    //VEC_SUB(v2,p1,q1);
    Vector3 v2 = p1 - q1;
    //CROSS_PROD(N1,v1,v2);
    Vector3 N1 = v1.cross(v2);
    //VEC_SUB(v1,q2,q1);
    v1 = q2 - q1;
    //if (DOT_PROD(v1,N1) > 0.0f)
    if (v1 * N1 > 0.0f)
        return false;
    //VEC_SUB(v1,p2,p1);
    v1 = p2 - p1;
    //VEC_SUB(v2,r1,p1);
    v2 = r1 - p1;
    //CROSS_PROD(N1,v1,v2);
    N1 = v1.cross(v2);
    //VEC_SUB(v1,r2,p1);
    v1 = r2 - p1;
    //if (DOT_PROD(v1,N1) > 0.0f)
    if (v1 * N1 > 0.0f)
        return false;
    else
        return true;
}

/* Permutation in a canonical form of T2's vertices */

bool TriIntersectGuigue::TRI_TRI_3D(const Vector3& p1, const Vector3& q1, const Vector3& r1, const Vector3& p2, const Vector3& q2, const Vector3& r2, const double& dp2, const double& dq2, const double& dr2, const Vector3& N1)
{
    if (dp2 > 0.0f)
    {
        if (dq2 > 0.0f)
            return CHECK_MIN_MAX(p1,r1,q1,r2,p2,q2);
        else if (dr2 > 0.0f)
            return CHECK_MIN_MAX(p1,r1,q1,q2,r2,p2);
        else
            return CHECK_MIN_MAX(p1,q1,r1,p2,q2,r2);
    }
    else if (dp2 < 0.0f)
    {
        if (dq2 < 0.0f)
            return CHECK_MIN_MAX(p1,q1,r1,r2,p2,q2);
        else if (dr2 < 0.0f)
            return CHECK_MIN_MAX(p1,q1,r1,q2,r2,p2);
        else
            return CHECK_MIN_MAX(p1,r1,q1,p2,q2,r2);
    } else
    {
        if (dq2 < 0.0f)
        {
            if (dr2 >= 0.0f)
                return CHECK_MIN_MAX(p1,r1,q1,q2,r2,p2);
            else
                return CHECK_MIN_MAX(p1,q1,r1,p2,q2,r2);
        }
        else if (dq2 > 0.0f)
        {
            if (dr2 > 0.0f)
                return CHECK_MIN_MAX(p1,r1,q1,p2,q2,r2);
            else
                return CHECK_MIN_MAX(p1,q1,r1,q2,r2,p2);
        }
        else
        {
            if (dr2 > 0.0f)
                return CHECK_MIN_MAX(p1,q1,r1,r2,p2,q2);
            else if (dr2 < 0.0f)
                return CHECK_MIN_MAX(p1,r1,q1,r2,p2,q2);
            else
                return coplanar_tri_tri3d(p1,q1,r1,p2,q2,r2,N1);
        }
    }
}

/*
*
*  Two dimensional Triangle-Triangle Overlap Test
*
*/


/* some 2D macros */

double TriIntersectGuigue::ORIENT_2D(const Vector2& a, const Vector2& b, const Vector2& c)
{
    return ((a[0]-c[0])*(b[1]-c[1])-(a[1]-c[1])*(b[0]-c[0]));
}

bool TriIntersectGuigue::INTERSECTION_TEST_VERTEX(const Vector2& P1, const Vector2& Q1, const Vector2& R1, const Vector2& P2, const Vector2& Q2, const Vector2& R2)
{
    if (ORIENT_2D(R2,P2,Q1) >= 0.0f)
    {
        if (ORIENT_2D(R2,Q2,Q1) <= 0.0f)
        {
            if (ORIENT_2D(P1,P2,Q1) > 0.0f)
            {
                if (ORIENT_2D(P1,Q2,Q1) <= 0.0f)
                    return true;
                else
                    return false;
            }
            else
            {
                if (ORIENT_2D(P1,P2,R1) >= 0.0f)
                {
                    if (ORIENT_2D(Q1,R1,P2) >= 0.0f)
                        return true;
                    else
                        return false;
                }
                else
                {
                    return false;
                }
            }
        }
        else
        {
            if (ORIENT_2D(P1,Q2,Q1) <= 0.0f)
            {
                if (ORIENT_2D(R2,Q2,R1) <= 0.0f)
                {
                    if (ORIENT_2D(Q1,R1,Q2) >= 0.0f)
                        return true;
                    else
                        return false;
                }
                else
                    return false;
            }
            else
            {
                return false;
            }
        }
    }
    else
    {
        if (ORIENT_2D(R2,P2,R1) >= 0.0f)
        {
            if (ORIENT_2D(Q1,R1,R2) >= 0.0f)
            {
                if (ORIENT_2D(P1,P2,R1) >= 0.0f)
                    return true;
                else
                    return false;
            }
            else
            {
                if (ORIENT_2D(Q1,R1,Q2) >= 0.0f)
                {
                    if (ORIENT_2D(R2,R1,Q2) >= 0.0f)
                        return true;
                    else
                        return false;
                }
                else
                    return false;
            }
        }
        else
        {
            return false;
        }
    }
}

bool TriIntersectGuigue::INTERSECTION_TEST_EDGE(const Vector2& P1, const Vector2& Q1, const Vector2& R1, const Vector2& P2, const Vector2& Q2, const Vector2& R2)
{
    if (ORIENT_2D(R2,P2,Q1) >= 0.0f)
    {
        if (ORIENT_2D(P1,P2,Q1) >= 0.0f)
        {
            if (ORIENT_2D(P1,Q1,R2) >= 0.0f)
                return true;
            else
                return false;
        }
        else
        {
            if (ORIENT_2D(Q1,R1,P2) >= 0.0f)
            {
                if (ORIENT_2D(R1,P1,P2) >= 0.0f)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
    {
        if (ORIENT_2D(R2,P2,R1) >= 0.0f)
        {
            if (ORIENT_2D(P1,P2,R1) >= 0.0f)
            {
                if (ORIENT_2D(P1,R1,R2) >= 0.0f)
                    return true;
                else
                {
                    if (ORIENT_2D(Q1,R1,R2) >= 0.0f)
                        return true;
                    else
                        return false;
                }
            }
            else
                return false;
        }
        else
        {
            return false;
        }
    }
}

bool TriIntersectGuigue::ccw_tri_tri_intersection_2d(const Vector2& p1, const Vector2& q1, const Vector2& r1,
                                 const Vector2& p2, const Vector2& q2, const Vector2& r2)
{
    if (ORIENT_2D(p2,q2,p1) >= 0.0f)
    {
        if (ORIENT_2D(q2,r2,p1) >= 0.0f)
        {
            if (ORIENT_2D(r2,p2,p1) >= 0.0f)
            {
                return true;
            }
            else
            {
                return INTERSECTION_TEST_EDGE(p1,q1,r1,p2,q2,r2);
            }
        }
        else
        {
            if (ORIENT_2D(r2,p2,p1) >= 0.0f)
            {
                return INTERSECTION_TEST_EDGE(p1,q1,r1,r2,p2,q2);
            }
            else
            {
                return INTERSECTION_TEST_VERTEX(p1,q1,r1,p2,q2,r2);
            }
        }
    }
    else
    {
        if (ORIENT_2D(q2,r2,p1) >= 0.0f)
        {
            if (ORIENT_2D(r2,p2,p1) >= 0.0f)
            {
                return INTERSECTION_TEST_EDGE(p1,q1,r1,q2,r2,p2);
            }
            else
            {
                return INTERSECTION_TEST_VERTEX(p1,q1,r1,q2,r2,p2);
            }
        }
        else
        {
            return INTERSECTION_TEST_VERTEX(p1,q1,r1,r2,p2,q2);
        }
    }
}

bool TriIntersectGuigue::tri_tri_overlap_test_2d(const Vector2& p1, const Vector2& q1, const Vector2& r1,
                                                 const Vector2& p2, const Vector2& q2, const Vector2& r2)
{
    if (ORIENT_2D(p1,q1,r1) < 0.0f)
    {
        if (ORIENT_2D(p2,q2,r2) < 0.0f)
            return ccw_tri_tri_intersection_2d(p1,r1,q1,p2,r2,q2);
        else
            return ccw_tri_tri_intersection_2d(p1,r1,q1,p2,q2,r2);
    }
    else
    {
        if (ORIENT_2D(p2,q2,r2) < 0.0f)
            return ccw_tri_tri_intersection_2d(p1,q1,r1,p2,r2,q2);
        else
            return ccw_tri_tri_intersection_2d(p1,q1,r1,p2,q2,r2);
    }
}

/*
*
*  Three-dimensional Triangle-Triangle Overlap Test
*
*/
bool TriIntersectGuigue::tri_tri_overlap_3d(const Vector3 t1[], const Vector3 t2[])
{
    Vector3 v1, v2;
    Vector3 N2;

    const Vector3 p1 = t1[0];
    const Vector3 q1 = t1[1];
    const Vector3 r1 = t1[2];

    const Vector3 p2 = t2[0];
    const Vector3 q2 = t2[1];
    const Vector3 r2 = t2[2];

    /* Compute distance signs  of p1, q1 and r1 to the plane of
     triangle(p2,q2,r2) */

    v1 = q2 - p2;
    v2 = r2 - p2;
    N2 = v1.cross(v2);

    const double d2 = N2 * p2;
    double dp1      = (N2 * p1) - d2;
    double dq1      = (N2 * q1) - d2;
    double dr1      = (N2 * r1) - d2;

    // Coplanarity robustness check
    if (std::fabs(dp1) < LOCAL_EPSILON) dp1 = 0.0f;
    if (std::fabs(dq1) < LOCAL_EPSILON) dq1 = 0.0f;
    if (std::fabs(dr1) < LOCAL_EPSILON) dr1 = 0.0f;

    if ((dp1 * dq1 > 0.0f) && (dp1 * dr1 > 0.0f))
        return false;

    /* Compute distance signs  of p2, q2 and r2 to the plane of
     triangle(p1,q1,r1) */

    Vector3 N1;
    v1 = q1 - p1;
    v2 = r1 - p1;
    N1 = v1.cross(v2);

    const double d1 = N1 * p1;
    double dp2      = (N1 * p2) - d1;
    double dq2      = (N1 * q2) - d1;
    double dr2      = (N1 * r2) - d1;

    // Coplanarity robustness check
    if (std::fabs(dp2) < LOCAL_EPSILON) dp2 = 0.0f;
    if (std::fabs(dq2) < LOCAL_EPSILON) dq2 = 0.0f;
    if (std::fabs(dr2) < LOCAL_EPSILON) dr2 = 0.0f;


    if ((dp2 * dq2 > 0.0f) && (dp2 * dr2 > 0.0f))
        return false;

    /* Permutation in a canonical form of T1's vertices */

    if (dp1 > 0.0f)
    {
        if (dq1 > 0.0f)
        {
            return TRI_TRI_3D(r1,p1,q1,p2,r2,q2,dp2,dr2,dq2, N1);
        }
        else if (dr1 > 0.0f)
        {
            return TRI_TRI_3D(q1,r1,p1,p2,r2,q2,dp2,dr2,dq2, N1);
        }
        else
        {
            return TRI_TRI_3D(p1,q1,r1,p2,q2,r2,dp2,dq2,dr2, N1);
        }
    }
    else if (dp1 < 0.0f)
    {
        if (dq1 < 0.0f)
        {
            return TRI_TRI_3D(r1,p1,q1,p2,q2,r2,dp2,dq2,dr2, N1);
        }
        else if (dr1 < 0.0f)
        {
            return TRI_TRI_3D(q1,r1,p1,p2,q2,r2,dp2,dq2,dr2, N1);
        }
        else
        {
            return TRI_TRI_3D(p1,q1,r1,p2,r2,q2,dp2,dr2,dq2, N1);
        }
    }
    else
    {
        if (dq1 < 0.0f)
        {
            if (dr1 >= 0.0f)
            {
                return TRI_TRI_3D(q1,r1,p1,p2,r2,q2,dp2,dr2,dq2, N1);
            }
            else
            {
                return TRI_TRI_3D(p1,q1,r1,p2,q2,r2,dp2,dq2,dr2, N1);
            }
        }
        else if (dq1 > 0.0f)
        {
            if (dr1 > 0.0f)
            {
                return TRI_TRI_3D(p1,q1,r1,p2,r2,q2,dp2,dr2,dq2, N1);
            }
            else
            {
                return TRI_TRI_3D(q1,r1,p1,p2,q2,r2,dp2,dq2,dr2, N1);
            }
        }
        else
        {
            if (dr1 > 0.0f)
            {
                return TRI_TRI_3D(r1,p1,q1,p2,q2,r2,dp2,dq2,dr2, N1);
            }
            else if (dr1 < 0.0f)
            {
                return TRI_TRI_3D(r1,p1,q1,p2,r2,q2,dp2,dr2,dq2, N1);
            }
            else
            {
                return coplanar_tri_tri3d(p1,q1,r1,p2,q2,r2,N1);
            }
        }
    }
}

//============================================================================
bool TriIntersectGuigue::coplanar_tri_tri3d(const Vector3& p1, const Vector3& q1, const Vector3& r1,
                                            const Vector3& p2, const Vector3& q2, const Vector3& r2,
                                            const Vector3& normal)
{
    Vector2 P1,Q1,R1;
    Vector2 P2,Q2,R2;

    double n_x, n_y, n_z;

    n_x = (normal[0] < 0.0f)? -normal[0] : normal[0];
    n_y = (normal[1] < 0.0f)? -normal[1] : normal[1];
    n_z = (normal[2] < 0.0f)? -normal[2] : normal[2];


    /* Projection of the triangles in 3D onto 2D such that the area of
     the projection is maximized. */

    if (( n_x > n_z ) && ( n_x >= n_y ))
    {
        // Project onto plane YZ

        P1[0] = q1[2]; P1[1] = q1[1];
        Q1[0] = p1[2]; Q1[1] = p1[1];
        R1[0] = r1[2]; R1[1] = r1[1];

        P2[0] = q2[2]; P2[1] = q2[1];
        Q2[0] = p2[2]; Q2[1] = p2[1];
        R2[0] = r2[2]; R2[1] = r2[1];
    }
    else if (( n_y > n_z ) && ( n_y >= n_x ))
    {
        // Project onto plane XZ

        P1[0] = q1[0]; P1[1] = q1[2];
        Q1[0] = p1[0]; Q1[1] = p1[2];
        R1[0] = r1[0]; R1[1] = r1[2];

        P2[0] = q2[0]; P2[1] = q2[2];
        Q2[0] = p2[0]; Q2[1] = p2[2];
        R2[0] = r2[0]; R2[1] = r2[2];
    }
    else
    {
        // Project onto plane XY

        P1[0] = p1[0]; P1[1] = p1[1];
        Q1[0] = q1[0]; Q1[1] = q1[1];
        R1[0] = r1[0]; R1[1] = r1[1];

        P2[0] = p2[0]; P2[1] = p2[1];
        Q2[0] = q2[0]; Q2[1] = q2[1];
        R2[0] = r2[0]; R2[1] = r2[1];
    }

    return tri_tri_overlap_test_2d(P1, Q1, R1, P2, Q2, R2);
}

bool TriIntersectGuigue::intersect_tri_tri(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &q0, const Vector3 &q1, const Vector3 &q2)
{
    Vector3 tri_verts_a[3], tri_verts_b[3];
    tri_verts_a[0] = p0; tri_verts_a[1] = p1; tri_verts_a[2] = p2;
    tri_verts_b[0] = q0; tri_verts_b[1] = q1; tri_verts_b[2] = q2;

    return tri_tri_overlap_3d(tri_verts_a, tri_verts_b);
}

double TriIntersectPQP::max3(double a, double b, double c)
{
    const double t = (a > b)? a : b;
    return (t > c) ? t : c;
}

double TriIntersectPQP::min3(double a, double b, double c)
{
    const double t = (a < b)? a : b;
    return (t < c)? t : c;
}

bool TriIntersectPQP::project6(const Vector3& ax,
                               const Vector3& p1, const Vector3& p2, const Vector3& p3,
                               const Vector3& q1, const Vector3& q2, const Vector3& q3)
{
    const double P1 = (ax * p1);
    const double P2 = (ax * p2);
    const double P3 = (ax * p3);
    const double Q1 = (ax * q1);
    const double Q2 = (ax * q2);
    const double Q3 = (ax * q3);

    if (min3(P1, P2, P3) > max3(Q1, Q2, Q3))
        return false;

    if (min3(Q1, Q2, Q3) > max3(P1, P2, P3))
        return false;

    return true;
}

//============================================================================


// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles
bool TriIntersectPQP::tri_tri_overlap_3d(const Vector3 t1[], const Vector3 t2[])
{
    // One triangle is (p1,p2,p3).  Other is (q1,q2,q3).
    // Edges are (e1,e2,e3) and (f1,f2,f3).
    // Normals are n1 and m1
    // Outwards are (g1,g2,g3) and (h1,h2,h3).
    //
    // We assume that the triangle vertices are in the same coordinate system.
    //
    // First thing we do is establish a new c.s. so that p1 is at (0,0,0).

    Vector3 p1, p2, p3;
    Vector3 q1, q2, q3;
    Vector3 e1, e2, e3;
    Vector3 f1, f2, f3;
    Vector3 g1, g2, g3;
    Vector3 h1, h2, h3;
    Vector3 n1, m1;

    Vector3 ef11, ef12, ef13;
    Vector3 ef21, ef22, ef23;
    Vector3 ef31, ef32, ef33;

    p1[0] = 0.0f;                 p1[1] = 0.0f;                 p1[2] = 0.0f;
    p2[0] = t1[1][0] - t1[0][0];  p2[1] = t1[1][1] - t1[0][1];  p2[2] = t1[1][2] - t1[0][2];
    p3[0] = t1[2][0] - t1[0][0];  p3[1] = t1[2][1] - t1[0][1];  p3[2] = t1[2][2] - t1[0][2];

    q1[0] = t2[0][0] - t1[0][0];  q1[1] = t2[0][1] - t1[0][1];  q1[2] = t2[0][2] - t1[0][2];
    q2[0] = t2[1][0] - t1[0][0];  q2[1] = t2[1][1] - t1[0][1];  q2[2] = t2[1][2] - t1[0][2];
    q3[0] = t2[2][0] - t1[0][0];  q3[1] = t2[2][1] - t1[0][1];  q3[2] = t2[2][2] - t1[0][2];

    e1[0] = p2[0];          e1[1] = p2[1];          e1[2] = p2[2];
    e2[0] = p3[0] - p2[0];  e2[1] = p3[1] - p2[1];  e2[2] = p3[2] - p2[2];
    e3[0] = p1[0] - p3[0];  e3[1] = p1[1] - p3[1];  e3[2] = p1[2] - p3[2];

    f1[0] = q2[0] - q1[0];  f1[1] = q2[1] - q1[1];  f1[2] = q2[2] - q1[2];
    f2[0] = q3[0] - q2[0];  f2[1] = q3[1] - q2[1];  f2[2] = q3[2] - q2[2];
    f3[0] = q1[0] - q3[0];  f3[1] = q1[1] - q3[1];  f3[2] = q1[2] - q3[2];

    n1 = e1.cross(e2);
    m1 = f1.cross(f2);

    ef11 = e1.cross(f1);
    ef12 = e1.cross(f2);
    ef13 = e1.cross(f3);
    ef21 = e2.cross(f1);
    ef22 = e2.cross(f2);
    ef23 = e2.cross(f3);
    ef31 = e3.cross(f1);
    ef32 = e3.cross(f2);
    ef33 = e3.cross(f3);

    // now begin the series of tests

    if (!project6(n1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(m1, p1, p2, p3, q1, q2, q3))
        return false;

    if (!project6(ef11, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef12, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef13, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef21, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef22, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef23, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef31, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef32, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef33, p1, p2, p3, q1, q2, q3))
        return false;

    g1 = e1.cross(n1);
    g2 = e2.cross(n1);
    g3 = e3.cross(n1);
    h1 = f1.cross(m1);
    h2 = f2.cross(m1);
    h3 = f3.cross(m1);

    if (!project6(g1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g2, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g3, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h2, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h3, p1, p2, p3, q1, q2, q3))
        return false;

    return true;
}

bool TriIntersectPQP::intersect_tri_tri(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &q0, const Vector3 &q1, const Vector3 &q2)
{
    Vector3 tri_verts_a[3], tri_verts_b[3];
    tri_verts_a[0] = p0; tri_verts_a[1] = p1; tri_verts_a[2] = p2;
    tri_verts_b[0] = q0; tri_verts_b[1] = q1; tri_verts_b[2] = q2;

    return tri_tri_overlap_3d(tri_verts_a, tri_verts_b);
}
