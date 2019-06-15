#ifndef OBBTREEGPU_LINEAR_ALGEBRA_H
#define OBBTREEGPU_LINEAR_ALGEBRA_H

#include <vector_types.h>
#include <vector_functions.h>

/*****************************************/
/* Vector                                */
/*****************************************/

__device__
inline float fastDiv(float numerator, float denominator)
{
    //return __fdividef(numerator, denominator);
    return numerator / denominator;
}

__device__
inline float getSqrtf(float f2)
{
    return sqrtf(f2);
    // return sqrt(f2);
}

__device__
inline float getReverseSqrt(float f2)
{
    //return rsqrtf(f2);

    return 1.0f / sqrtf(f2);
}

__device__
inline float3 getCrossProduct(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__
inline float4 getCrossProduct(float4 a, float4 b)
{
    float3 v1 = make_float3(a.x, a.y, a.z);
    float3 v2 = make_float3(b.x, b.y, b.z);
    float3 v3 = make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);

    return make_float4(v3.x, v3.y, v3.z, 0.0f);
}

__device__
inline float getDotProduct(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
inline float getDotProduct(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float3 getNormalizedVec(const float3 v)
{
    float invLen = 1.0f / sqrtf(getDotProduct(v, v));
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ __forceinline__ float4 getNormalizedVec(const float4 v)
{
    float invLen = 1.0f / sqrtf(getDotProduct(v, v));
    return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, v.w * invLen);
}

__device__
inline float dot3F4(float4 a, float4 b)
{
    float4 a1 = make_float4(a.x, a.y, a.z,0.f);
    float4 b1 = make_float4(b.x, b.y, b.z,0.f);
    return getDotProduct(a1, b1);
}

__device__
inline float getLength(float3 a)
{
    return sqrtf(getDotProduct(a, a));
}

__device__
inline float getLength(float4 a)
{
    return sqrtf(getDotProduct(a, a));
}

/*****************************************
Matrix2x2
*****************************************/

typedef struct
{
    float2 m_row[2];
} Matrix2x2;

__device__
inline void setZero(Matrix2x2& m)
{
    m.m_row[0] = make_float2(0.0f, 0.0f);
    m.m_row[1] = make_float2(0.0f, 0.0f);
}

__device__
inline Matrix2x2 getZeroMatrix2x2()
{
    Matrix2x2 m;
    m.m_row[0] = make_float2(0.0f, 0.0f);
    m.m_row[1] = make_float2(0.0f, 0.0f);
    return m;
}

__device__
inline void setIdentity(Matrix2x2& m)
{
    m.m_row[0] = make_float2(1,0);
    m.m_row[1] = make_float2(0,1);
}

__device__
inline Matrix2x2 getIdentityMatrix2x2()
{
    Matrix2x2 m;
    m.m_row[0] = make_float2(1,0);
    m.m_row[1] = make_float2(0,1);
    return m;
}

__device__  inline double determinant(const Matrix2x2& m)
{
    return m.m_row[0].x * m.m_row[1].y - m.m_row[1].x * m.m_row[0].y;
}


/*****************************************
Matrix3x3
*****************************************/
typedef struct
{
    float4 m_row[3];
}Matrix3x3_d;

__device__
inline void setZero(Matrix3x3_d& m)
{
    m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__
inline Matrix3x3_d getZeroMatrix3x3()
{
    Matrix3x3_d m;
    m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return m;
}

__device__  inline Matrix3x3_d copyMatrix3x3(const Matrix3x3_d& in)
{
    Matrix3x3_d m;
    m.m_row[0] = make_float4(in.m_row[0].x, in.m_row[0].y, in.m_row[0].z, in.m_row[0].w);
    m.m_row[1] = make_float4(in.m_row[1].x, in.m_row[1].y, in.m_row[1].z, in.m_row[1].w);
    m.m_row[2] = make_float4(in.m_row[2].x, in.m_row[2].y, in.m_row[2].z, in.m_row[2].w);
    return m;
}

__device__
inline void setIdentity(Matrix3x3_d& m)
{
    m.m_row[0] = make_float4(1,0,0,0);
    m.m_row[1] = make_float4(0,1,0,0);
    m.m_row[2] = make_float4(0,0,1,0);
}

__device__
inline Matrix3x3_d getIdentityMatrix3x3()
{
    Matrix3x3_d m;
    m.m_row[0] = make_float4(1,0,0,0);
    m.m_row[1] = make_float4(0,1,0,0);
    m.m_row[2] = make_float4(0,0,1,0);
    return m;
}

__device__
inline Matrix3x3_d getTranspose(const Matrix3x3_d m)
{
    Matrix3x3_d out;
    out.m_row[0].x = m.m_row[0].x; out.m_row[0].y = m.m_row[1].x; out.m_row[0].z = m.m_row[2].x;
    out.m_row[1].x = m.m_row[0].y; out.m_row[1].y = m.m_row[1].y; out.m_row[1].z = m.m_row[2].y;
    out.m_row[2].x = m.m_row[0].z; out.m_row[2].y = m.m_row[0].y; out.m_row[2].z = m.m_row[2].z;

    out.m_row[0].w = m.m_row[0].w;
    out.m_row[1].w = m.m_row[1].w;
    out.m_row[2].w = m.m_row[2].w;

    return out;
}

__device__
inline Matrix3x3_d MatrixMul(Matrix3x3_d& a, Matrix3x3_d& b)
{
    Matrix3x3_d transB = getTranspose( b );
    Matrix3x3_d ans;
    // why this doesn't run when 0ing in the for{}
    a.m_row[0].w = 0.f;
    a.m_row[1].w = 0.f;
    a.m_row[2].w = 0.f;
    for(int i=0; i<3; i++)
    {
        // a.m_row[i].w = 0.f;
        ans.m_row[i].x = dot3F4(a.m_row[i],transB.m_row[0]);
        ans.m_row[i].y = dot3F4(a.m_row[i],transB.m_row[1]);
        ans.m_row[i].z = dot3F4(a.m_row[i],transB.m_row[2]);
        ans.m_row[i].w = 0.f;
    }
    return ans;
}

__device__  inline
float3 mtMul1(Matrix3x3_d a, float3 b)
{
    float3 ans;
    float4 b_4 = make_float4(b.x, b.y, b.z, 0);
    ans.x = dot3F4( a.m_row[0], b_4 );
    ans.y = dot3F4( a.m_row[1], b_4 );
    ans.z = dot3F4( a.m_row[2], b_4 );
    return ans;
}



/*****************************************
Quaternion
******************************************/

typedef float4 gpQuaternion;

__device__
inline gpQuaternion quaternionMul(gpQuaternion a, gpQuaternion b);

__device__
inline gpQuaternion qtNormalize(gpQuaternion in);

__device__
inline float4 qtRotate(gpQuaternion q, float4 vec);

__device__
inline gpQuaternion qtInvert(gpQuaternion q);

__device__
inline Matrix3x3_d qtGetRotationMatrix(gpQuaternion q);

__device__
inline gpQuaternion quaternionMul(gpQuaternion a, gpQuaternion b)
{
    gpQuaternion ans;
    ans = getCrossProduct(a, b);
    ans = make_float4(ans.x + a.w*b.x + b.w*a.x + b.w*a.y, ans.y + a.w*b.y + b.w*a.z, ans.z + a.w*b.z, ans.w + a.w*b.w + b.w*a.w);
    // ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);
    ans.w = a.w*b.w - dot3F4(a, b);
    return ans;
}

__device__
inline gpQuaternion qtNormalize(gpQuaternion in)
{
    return getNormalizedVec(in);
    // in /= length( in );
    // return in;
}

__device__
inline gpQuaternion qtInvert(const gpQuaternion q)
{
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

__device__
inline float4 qtRotate(const gpQuaternion q, const float4 vec)
{
    gpQuaternion qInv = qtInvert( q );
    float4 vcpy = vec;
    vcpy.w = 0.f;
    float4 out = quaternionMul(quaternionMul(q,vcpy),qInv);
    return out;
}

__device__
inline float4 qtInvRotate(const gpQuaternion q, const float4 vec)
{
    return qtRotate( qtInvert( q ), vec );
}

__device__
inline Matrix3x3_d qtGetRotationMatrix(gpQuaternion quat)
{
    float4 quat2 = make_float4(quat.x*quat.x, quat.y*quat.y, quat.z*quat.z, 0.f);
    Matrix3x3_d out;

    out.m_row[0].x=1-2*quat2.y-2*quat2.z;
    out.m_row[0].y=2*quat.x*quat.y-2*quat.w*quat.z;
    out.m_row[0].z=2*quat.x*quat.z+2*quat.w*quat.y;
    out.m_row[0].w = 0.f;

    out.m_row[1].x=2*quat.x*quat.y+2*quat.w*quat.z;
    out.m_row[1].y=1-2*quat2.x-2*quat2.z;
    out.m_row[1].z=2*quat.y*quat.z-2*quat.w*quat.x;
    out.m_row[1].w = 0.f;

    out.m_row[2].x=2*quat.x*quat.z-2*quat.w*quat.y;
    out.m_row[2].y=2*quat.y*quat.z+2*quat.w*quat.x;
    out.m_row[2].z=1-2*quat2.x-2*quat2.y;
    out.m_row[2].w = 0.f;

    return out;
}

__device__  inline void TINV_MUL_T(Matrix3x3_d& dest, const Matrix3x3_d& a, const Matrix3x3_d& b)
{
    dest.m_row[0].x =
        a.m_row[0].x * b.m_row[0].x + a.m_row[0].y * b.m_row[0].y + a.m_row[0].z * b.m_row[0].z;
    dest.m_row[1].x =
        a.m_row[1].x * b.m_row[0].x + a.m_row[1].y * b.m_row[0].y + a.m_row[1].z * b.m_row[0].z;
    dest.m_row[2].x =
        a.m_row[2].x * b.m_row[0].x + a.m_row[2].y * b.m_row[0].y + a.m_row[2].z * b.m_row[0].z;
    dest.m_row[0].y =
        a.m_row[0].x * b.m_row[1].x + a.m_row[0].y * b.m_row[1].y + a.m_row[0].z * b.m_row[1].z;
    dest.m_row[1].y =
        a.m_row[1].x * b.m_row[1].x + a.m_row[1].y * b.m_row[1].y + a.m_row[1].z * b.m_row[1].z;
    dest.m_row[2].y =
        a.m_row[2].x * b.m_row[1].x + a.m_row[2].y * b.m_row[1].y + a.m_row[2].z * b.m_row[1].z;

    dest.m_row[0].z = (dest.m_row[1].x * dest.m_row[2].y) - (dest.m_row[2].x * dest.m_row[1].y);
    dest.m_row[1].z = (dest.m_row[2].x * dest.m_row[0].y) - (dest.m_row[0].x * dest.m_row[2].y);
    dest.m_row[2].z = (dest.m_row[0].x * dest.m_row[1].y) - (dest.m_row[1].x * dest.m_row[0].y);

    dest.m_row[0].w = a.m_row[0].x * (b.m_row[0].w - a.m_row[0].w) + a.m_row[1].x * (b.m_row[1].w - a.m_row[1].w) + a.m_row[2].x * (b.m_row[2].w - a.m_row[2].w);
    dest.m_row[1].w = a.m_row[0].y * (b.m_row[0].w - a.m_row[0].w) + a.m_row[1].y * (b.m_row[1].w - a.m_row[1].w) + a.m_row[2].y * (b.m_row[2].w - a.m_row[2].w);
    dest.m_row[2].w = a.m_row[0].z * (b.m_row[0].w - a.m_row[0].w) + a.m_row[1].z * (b.m_row[1].w - a.m_row[1].w) + a.m_row[2].z * (b.m_row[2].w - a.m_row[2].w);
}

/*#define TRANSFORM_INV(dest,src)                                                     \
  (dest)[R00] = (src)[R00]; (dest)[R01] = (src)[R10]; (dest)[R02] = (src)[R20];     \
  (dest)[R10] = (src)[R01]; (dest)[R11] = (src)[R11]; (dest)[R12] = (src)[R21];     \
  (dest)[R20] = (src)[R02]; (dest)[R21] = (src)[R12]; (dest)[R22] = (src)[R22];     \
  (dest)[TX] = -(src)[R00]*(src)[TX] - (src)[R10]*(src)[TY] - (src)[R20]*(src)[TZ]; \
  (dest)[TY] = -(src)[R01]*(src)[TX] - (src)[R11]*(src)[TY] - (src)[R21]*(src)[TZ]; \
  (dest)[TZ] = -(src)[R02]*(src)[TX] - (src)[R12]*(src)[TY] - (src)[R22]*(src)[TZ]
*/

__device__  inline void TRANSFORM_INV(Matrix3x3_d& dest, const Matrix3x3_d& src)
{
    (dest).m_row[0].x = (src).m_row[0].x;
    (dest).m_row[0].y = (src).m_row[1].x;
    (dest).m_row[0].z = (src).m_row[2].x;
    (dest).m_row[1].x = (src).m_row[0].y;
    (dest).m_row[1].y = (src).m_row[1].y;
    (dest).m_row[1].z = (src).m_row[2].y;
    (dest).m_row[2].x = (src).m_row[0].z;
    (dest).m_row[2].y = (src).m_row[1].z;
    (dest).m_row[2].z = (src).m_row[2].z;

    (dest).m_row[0].w = -(src).m_row[0].x * (src).m_row[0].w - (src).m_row[1].y * (src).m_row[1].w - (src).m_row[2].x * (src).m_row[2].w;
    (dest).m_row[1].w = -(src).m_row[0].y * (src).m_row[0].w - (src).m_row[1].y * (src).m_row[1].w - (src).m_row[2].y * (src).m_row[2].w;
    (dest).m_row[2].w = -(src).m_row[0].z * (src).m_row[0].w - (src).m_row[1].z * (src).m_row[1].w - (src).m_row[2].z * (src).m_row[2].w;
}

//#define TR_MULT(ab,a,b)                                                               \
//  ROT_MTRX_MULT(ab,a,b);                                                              \
//  (ab)[TX] = ((a)[R00]*(b)[TX])  + ((a)[R01]*(b)[TY]) + ((a)[R02]*(b)[TZ]) + (a)[TX]; \
//  (ab)[TY] = ((a)[R10]*(b)[TX])  + ((a)[R11]*(b)[TY]) + ((a)[R12]*(b)[TZ]) + (a)[TY]; \
//  (ab)[TZ] = ((a)[R20]*(b)[TX])  + ((a)[R21]*(b)[TY]) + ((a)[R22]*(b)[TZ]) + (a)[TZ]

__device__  inline void TR_MULT(Matrix3x3_d& ab, Matrix3x3_d& a, Matrix3x3_d& b)
{
    Matrix3x3_d ab_tmp = MatrixMul(a, b);

    ab.m_row[0].x = ab_tmp.m_row[0].x; ab.m_row[0].y = ab_tmp.m_row[0].y; ab.m_row[0].z = ab_tmp.m_row[0].z;
    ab.m_row[1].x = ab_tmp.m_row[1].x; ab.m_row[1].y = ab_tmp.m_row[1].y; ab.m_row[1].z = ab_tmp.m_row[1].z;
    ab.m_row[2].x = ab_tmp.m_row[2].x; ab.m_row[2].y = ab_tmp.m_row[2].y; ab.m_row[2].z = ab_tmp.m_row[2].z;

    (ab).m_row[0].w = ((a).m_row[0].x * (b).m_row[0].w) + ((a).m_row[0].y * (b).m_row[1].w) + ((a).m_row[0].z * (b).m_row[2].w) + (a).m_row[0].w;
    (ab).m_row[1].w = ((a).m_row[1].x * (b).m_row[0].w) + ((a).m_row[1].y * (b).m_row[1].w) + ((a).m_row[1].z * (b).m_row[2].w) + (a).m_row[1].w;
    (ab).m_row[2].w = ((a).m_row[2].x * (b).m_row[0].w) + ((a).m_row[2].y * (b).m_row[1].w) + ((a).m_row[2].z * (b).m_row[2].w) + (a).m_row[2].w;
}

__device__  inline void TR_INV_MULT(Matrix3x3_d& ab, const Matrix3x3_d& a, const Matrix3x3_d& b)
{
    (ab).m_row[0].x = (a).m_row[0].x * (b).m_row[0].x + (a).m_row[1].x * (b).m_row[1].x + (a).m_row[2].x * (b).m_row[2].x;
    (ab).m_row[1].x = (a).m_row[0].y * (b).m_row[0].x + (a).m_row[1].y * (b).m_row[1].x + (a).m_row[2].y * (b).m_row[2].x;
    (ab).m_row[2].x = (a).m_row[0].z * (b).m_row[0].x + (a).m_row[1].z * (b).m_row[1].x + (a).m_row[2].z * (b).m_row[2].x;
    (ab).m_row[0].y = (a).m_row[0].x * (b).m_row[0].y + (a).m_row[1].x * (b).m_row[1].y + (a).m_row[2].x * (b).m_row[2].y;
    (ab).m_row[1].y = (a).m_row[0].y * (b).m_row[0].y + (a).m_row[1].y * (b).m_row[1].y + (a).m_row[2].y * (b).m_row[2].y;
    (ab).m_row[2].y = (a).m_row[0].z * (b).m_row[0].y + (a).m_row[1].z * (b).m_row[1].y + (a).m_row[2].z * (b).m_row[2].y;

    (ab).m_row[0].z = ((ab).m_row[1].x * (ab).m_row[2].y) - ((ab).m_row[2].x * (ab).m_row[1].y);
    (ab).m_row[1].z = ((ab).m_row[2].x * (ab).m_row[0].y) - ((ab).m_row[0].x * (ab).m_row[2].y);
    (ab).m_row[2].z = ((ab).m_row[0].x * (ab).m_row[1].y) - ((ab).m_row[1].x * (ab).m_row[0].y);

    (ab).m_row[0].w = (a).m_row[0].x * ((b).m_row[0].w - (a).m_row[0].w) + (a).m_row[1].x * ((b).m_row[1].w - (a).m_row[1].w) + (a).m_row[2].x * ((b).m_row[2].w - (a).m_row[2].w);
    (ab).m_row[1].w = (a).m_row[0].y * ((b).m_row[0].w - (a).m_row[1].w) + (a).m_row[1].y * ((b).m_row[1].w - (a).m_row[1].w) + (a).m_row[2].y * ((b).m_row[2].w - (a).m_row[2].w);
    (ab).m_row[2].w = (a).m_row[0].z * ((b).m_row[0].w - (a).m_row[0].w) + (a).m_row[1].z * ((b).m_row[1].w - (a).m_row[1].w) + (a).m_row[2].z * ((b).m_row[2].w - (a).m_row[2].w);
}


#endif //OBBTREEGPU_LINEAR_ALGEBRA_H
