/*************************************************************************\

  Copyright 1999 The University of North Carolina at Chapel Hill.
  All Rights Reserved.

  Permission to use, copy, modify and distribute this software and its
  documentation for educational, research and non-profit purposes, without
  fee, and without a written agreement is hereby granted, provided that the
  above copyright notice and the following three paragraphs appear in all
  copies.

  IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL BE
  LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
  CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
  USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY
  OF NORTH CAROLINA HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH
  DAMAGES.

  THE UNIVERSITY OF NORTH CAROLINA SPECIFICALLY DISCLAIM ANY
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
  PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
  NORTH CAROLINA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

  The authors may be contacted via:

  US Mail:             S. Gottschalk, E. Larsen
                       Department of Computer Science
                       Sitterson Hall, CB #3175
                       University of N. Carolina
                       Chapel Hill, NC 27599-3175

  Phone:               (919)962-1749

  EMail:               geom@cs.unc.edu


\**************************************************************************/

#include <stdio.h>
#include <string.h>
#include "PQP.h"
#include "BVTQ.h"
#include "Build.h"
#include "MatVec.h"
#include "GetTime.h"
#include "TriDist.h"

#include <iostream>

#if defined(WIN32)
#include <algorithm>
#if _MSC_VER <= 1700
# define max(x,y) ((std::max)(x,y))
# define min(x,y) ((std::min)(x,y))
#else
# define max(x,y) (std::max(x,y))
# define min(x,y) (std::min(x,y))
#endif
#else
//# include <cmath>
//# define max(x,y) (std::max(x,y))
//# define min(x,y) (std::min(x,y))
# define isnan(x) (std::isnan(x))
#endif

enum BUILD_STATE
{ 
  PQP_BUILD_STATE_EMPTY,     // empty state, immediately after constructor
  PQP_BUILD_STATE_BEGUN,     // after BeginModel(), state for adding triangles
  PQP_BUILD_STATE_PROCESSED  // after tree has been built, ready to use
};

PQP_Model::PQP_Model()
{
  // no bounding volume tree yet

  b = 0;  
  num_bvs_alloced = 0;
  num_bvs = 0;

  // no tri list yet

  tris = 0;
  num_tris = 0;
  num_tris_alloced = 0;

  last_tri = 0;

  build_state = PQP_BUILD_STATE_EMPTY;
}

PQP_Model::~PQP_Model()
{
  if (b != NULL)
    delete [] b;
  if (tris != NULL)
    delete [] tris;
}

int
PQP_Model::BeginModel(int n)
{
  // reset to initial state if necessary

  if (build_state != PQP_BUILD_STATE_EMPTY) 
  {
    delete [] b;
    delete [] tris;
  
    num_tris = num_bvs = num_tris_alloced = num_bvs_alloced = 0;
  }

  // prepare model for addition of triangles

  if (n <= 0) n = 8;
  num_tris_alloced = n;
  tris = new Tri[n];
  if (!tris) 
  {
    fprintf(stderr, "PQP Error!  Out of memory for tri array on "
                    "BeginModel() call!\n");
    return PQP_ERR_MODEL_OUT_OF_MEMORY;  
  }

  // give a warning if called out of sequence

  if (build_state != PQP_BUILD_STATE_EMPTY)
  {
    fprintf(stderr,
            "PQP Warning! Called BeginModel() on a PQP_Model that \n"
            "was not empty. This model was cleared and previous\n"
            "triangle additions were lost.\n");
    build_state = PQP_BUILD_STATE_BEGUN;
    return PQP_ERR_BUILD_OUT_OF_SEQUENCE;
  }

  build_state = PQP_BUILD_STATE_BEGUN;
  return PQP_OK;
}

int
PQP_Model::AddTri(const PQP_REAL *p1, 
                  const PQP_REAL *p2, 
                  const PQP_REAL *p3, 
                  int id)
{
  if (build_state == PQP_BUILD_STATE_EMPTY)
  {
    BeginModel();
  }
  else if (build_state == PQP_BUILD_STATE_PROCESSED)
  {
    fprintf(stderr,"PQP Warning! Called AddTri() on PQP_Model \n"
                   "object that was already ended. AddTri() was\n"
                   "ignored.  Must do a BeginModel() to clear the\n"
                   "model for addition of new triangles\n");
    return PQP_ERR_BUILD_OUT_OF_SEQUENCE;
  }
        
  // allocate for new triangles

  if (num_tris >= num_tris_alloced)
  {
    Tri *temp;
    temp = new Tri[num_tris_alloced*2];
    if (!temp)
    {
      fprintf(stderr, "PQP Error!  Out of memory for tri array on"
	              " AddTri() call!\n");
      return PQP_ERR_MODEL_OUT_OF_MEMORY;  
    }
    memcpy(temp, tris, sizeof(Tri)*num_tris);
    delete [] tris;
    tris = temp;
    num_tris_alloced = num_tris_alloced*2;
  }
  
  // initialize the new triangle

  tris[num_tris].p1[0] = p1[0];
  tris[num_tris].p1[1] = p1[1];
  tris[num_tris].p1[2] = p1[2];

  tris[num_tris].p2[0] = p2[0];
  tris[num_tris].p2[1] = p2[1];
  tris[num_tris].p2[2] = p2[2];

  tris[num_tris].p3[0] = p3[0];
  tris[num_tris].p3[1] = p3[1];
  tris[num_tris].p3[2] = p3[2];

  tris[num_tris].id = id;

  num_tris += 1;

  return PQP_OK;
}

//#define PQP_DEBUG_FIX_DEGENERATE_OBB
void fixDegenerateObb(BV* bv, const double& dimensionLimit)
{
#ifdef PQP_DEBUG_FIX_DEGENERATE_OBB
    std::cout << "fixDegenerateObb()" << std::endl;
    std::cout << " center      : " << bv->To[0] << "," << bv->To[1] << "," << bv->To[2] << std::endl;
    std::cout << " half-extents: " << bv->d[0] << "," << bv->d[1] << "," << bv->d[2] << std::endl;
    std::cout << " dimension limit: " << dimensionLimit << std::endl;
#endif
    int minDimension = -1;
    for (int i = 0; i < 3; i++)
    {
#ifdef PQP_DEBUG_FIX_DEGENERATE_OBB
        std::cout << " _halfExtents[" << i << "] = " << bv->d[i] << std::endl;
#endif
        if (fabs(bv->d[i]) <= dimensionLimit)
        {
#ifdef PQP_DEBUG_FIX_DEGENERATE_OBB
            std::cout << "Detected 'too small' obb extent at dim. " << i << ": " << bv->d[i] << std::endl;
#endif
            minDimension = i;

            bv->d[i] = dimensionLimit;
#ifdef PQP_DEBUG_FIX_DEGENERATE_OBB
            std::cout << " reassigned dim " << i << " = " << bv->d[i] << std::endl;
#endif
            int attempts = 0;
            while (attempts < 10 && bv->d[i] < dimensionLimit)
            {
                bv->d[i] *= 10.0;
                attempts++;
            }

            PQP_REAL minDimensionVal = bv->d[i] / 2.0f;

            bv->min_dimension = minDimension;
            bv->min_dimension_val = minDimensionVal;

#ifdef PQP_DEBUG_FIX_DEGENERATE_OBB
            std::cout << " final dim " << i << " = " << bv->d[i] << ", stored min_dimension_val = " << bv->min_dimension_val << ", min_dimension = " << bv->min_dimension << std::endl;
#endif
        }
    }
}


int
PQP_Model::EndModel(double alarmDistance, bool parent_relative, bool fixDegenerate)
{
  if (build_state == PQP_BUILD_STATE_PROCESSED)
  {
    fprintf(stderr,"PQP Warning! Called EndModel() on PQP_Model \n"
                   "object that was already ended. EndModel() was\n"
                   "ignored.  Must do a BeginModel() to clear the\n"
                   "model for addition of new triangles\n");
    return PQP_ERR_BUILD_OUT_OF_SEQUENCE;
  }

  // report error is no tris

  if (num_tris == 0)
  {
    fprintf(stderr,"PQP Error! EndModel() called on model with"
                   " no triangles\n");
    return PQP_ERR_BUILD_EMPTY_MODEL;
  }

  // shrink fit tris array 

  if (num_tris_alloced > num_tris)
  {
    Tri *new_tris = new Tri[num_tris];
    if (!new_tris) 
    {
      fprintf(stderr, "PQP Error!  Out of memory for tri array "
                      "in EndModel() call!\n");
      return PQP_ERR_MODEL_OUT_OF_MEMORY;  
    }
    memcpy(new_tris, tris, sizeof(Tri)*num_tris);
    delete [] tris;
    tris = new_tris;
    num_tris_alloced = num_tris;
  }

  // create an array of BVs for the model

  b = new BV[2*num_tris - 1];
  if (!b)
  {
    fprintf(stderr,"PQP Error! out of memory for BV array "
                   "in EndModel()\n");
    return PQP_ERR_MODEL_OUT_OF_MEMORY;
  }
  num_bvs_alloced = 2*num_tris - 1;
  num_bvs = 0;

  // we should build the model now.

  build_model(this, parent_relative);
  build_state = PQP_BUILD_STATE_PROCESSED;

  if (fixDegenerate)
  {
	  std::cout << "=== fixDegenerateOBBs: " << this->num_bvs << " BV nodes. ===" << std::endl;
	  for (int k = 0; k < this->num_bvs; k++)
	  {
		  if (this->child(k)->Leaf() != 0)
			  fixDegenerateObb(this->child(k), 0.15);
	  }
  }

  // Blow up OBBTree, so the alarmDistance can trigger contacts (BE)
  std::cout << "=== blow up OBBTree according to alarmDistance = " << alarmDistance << std::endl;
  alarmDistance *= 0.5; // if boths trees go half the way, the tree is blown up to exactly the right size.
  for (int k = 0; k < this->num_bvs; k++)
  {
      this->child(k)->d[0] += alarmDistance;
      this->child(k)->d[1] += alarmDistance;
      this->child(k)->d[2] += alarmDistance;
  }

  last_tri = tris;

  return PQP_OK;
}

int
PQP_Model::MemUsage(int msg)
{
  int mem_bv_list = sizeof(BV)*num_bvs;
  int mem_tri_list = sizeof(Tri)*num_tris;

  int total_mem = mem_bv_list + mem_tri_list + sizeof(PQP_Model);

  if (msg) 
  {
    fprintf(stderr,"Total for model %x: %d bytes\n", this, total_mem);
    fprintf(stderr,"BVs: %d alloced, take %d bytes each\n", 
            num_bvs, sizeof(BV));
    fprintf(stderr,"Tris: %d alloced, take %d bytes each\n", 
            num_tris, sizeof(Tri));
  }
  
  return total_mem;
}

//  COLLIDE STUFF
//
//--------------------------------------------------------------------------

PQP_CollideResult::PQP_CollideResult()
{
  pairs = 0;
  num_pairs = num_pairs_alloced = 0;

  num_obb_pairs = num_obb_pairs_alloced = 0;
  obb_pairs = 0;

  num_bv_tests = 0;
  num_tri_tests = 0;
}

PQP_CollideResult::~PQP_CollideResult()
{
  delete [] pairs;
}

void
PQP_CollideResult::FreePairsList()
{
  num_pairs = num_pairs_alloced = 0;
  delete [] pairs;
  pairs = 0;

  num_obb_pairs = num_obb_pairs_alloced = 0;
  delete[] obb_pairs;
  obb_pairs = 0;
}

// may increase OR reduce mem usage
void
PQP_CollideResult::SizeTo(int n)
{
  CollisionPair *temp;

  if (n < num_pairs) 
  {
    fprintf(stderr, "PQP Error: Internal error in "
                    "'PQP_CollideResult::SizeTo(int n)'\n");
    fprintf(stderr, "       n = %d, but num_pairs = %d\n", n, num_pairs);
    return;
  }
  
  temp = new CollisionPair[n];
  memcpy(temp, pairs, num_pairs*sizeof(CollisionPair));
  delete [] pairs;
  pairs = temp;
  num_pairs_alloced = n;
  return;
}

void
PQP_CollideResult::Add(int a, int b)
{
  if (num_pairs >= num_pairs_alloced) 
  {
    // allocate more

    SizeTo(num_pairs_alloced*2+8);
  }

  // now proceed as usual

  pairs[num_pairs].id1 = a;
  pairs[num_pairs].id2 = b;
  num_pairs++;
}

void
PQP_CollideResult::SizeOBBTo(int n)
{
  CollisionPair *temp;

  if (n < num_obb_pairs)
  {
    fprintf(stderr, "PQP Error: Internal error in "
                    "'PQP_CollideResult::SizeOBBTo(int n)'\n");
    fprintf(stderr, "       n = %d, but num_pairs = %d\n", n, num_pairs);
    return;
  }

  temp = new CollisionPair[n];
  memcpy(temp, obb_pairs, num_obb_pairs*sizeof(CollisionPair));
  delete [] obb_pairs;
  obb_pairs = temp;
  num_obb_pairs_alloced = n;
  return;
}

void
PQP_CollideResult::AddOBB(int a, int b)
{
  if (num_obb_pairs >= num_obb_pairs_alloced)
  {
    // allocate more

    SizeOBBTo(num_obb_pairs_alloced*2+8);
  }

  // now proceed as usual

  obb_pairs[num_obb_pairs].id1 = a;
  obb_pairs[num_obb_pairs].id2 = b;
  num_obb_pairs++;
}

// TRIANGLE OVERLAP TEST
       
/*inline
PQP_REAL
max(PQP_REAL a, PQP_REAL b, PQP_REAL c)
{
  PQP_REAL t = a;
  if (b > t) t = b;
  if (c > t) t = c;
  return t;
}

inline
PQP_REAL
min(PQP_REAL a, PQP_REAL b, PQP_REAL c)
{
  PQP_REAL t = a;
  if (b < t) t = b;
  if (c < t) t = c;
  return t;
}*/

int
project6(PQP_REAL *ax, 
         PQP_REAL *p1, PQP_REAL *p2, PQP_REAL *p3, 
         PQP_REAL *q1, PQP_REAL *q2, PQP_REAL *q3)
{
  PQP_REAL P1 = VdotV(ax, p1);
  PQP_REAL P2 = VdotV(ax, p2);
  PQP_REAL P3 = VdotV(ax, p3);
  PQP_REAL Q1 = VdotV(ax, q1);
  PQP_REAL Q2 = VdotV(ax, q2);
  PQP_REAL Q3 = VdotV(ax, q3);
  
#ifdef _WIN32
  PQP_REAL mx1 = max(max(P1, P2), P3);
  PQP_REAL mn1 = min(min(P1, P2), P3);
  PQP_REAL mx2 = max(max(Q1, Q2), Q3);
  PQP_REAL mn2 = min(min(Q1, Q2), Q3);
#else
  PQP_REAL mx1 = std::max(std::max(P1, P2), P3);
  PQP_REAL mn1 = std::min(std::min(P1, P2), P3);
  PQP_REAL mx2 = std::max(std::max(Q1, Q2), Q3);
  PQP_REAL mn2 = std::min(std::min(Q1, Q2), Q3);
#endif
  if (mn1 > mx2) return 0;
  if (mn2 > mx1) return 0;
  return 1;
}

int coplanar_tri_tri(PQP_REAL N[3], PQP_REAL V0[3], PQP_REAL V1[3], PQP_REAL V2[3],
                     PQP_REAL U0[3], PQP_REAL U1[3], PQP_REAL U2[3]);

// some vector macros
#define CROSS(dest,v1,v2)                       \
               dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
               dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
               dest[2]=v1[0]*v2[1]-v1[1]*v2[0];



#define   sVpsV_2( Vr, s1,  V1,s2, V2);\
    {\
  Vr[0] = s1*V1[0] + s2*V2[0];\
  Vr[1] = s1*V1[1] + s2*V2[1];\
}\

#define myVpV(g,v2,v1);\
{\
    g[0] = v2[0]+v1[0];\
    g[1] = v2[1]+v1[1];\
    g[2] = v2[2]+v1[2];\
    }\

  #define myVmV(g,v2,v1);\
{\
    g[0] = v2[0]-v1[0];\
    g[1] = v2[1]-v1[1];\
    g[2] = v2[2]-v1[2];\
    }\

// 2D intersection of segment and triangle.
#define seg_collide3( q, r)\
{\
    p1[0]=SF*P1[0];\
    p1[1]=SF*P1[1];\
    p2[0]=SF*P2[0];\
    p2[1]=SF*P2[1];\
    det1 = p1[0]*q[1]-q[0]*p1[1];\
    gama1 = (p1[0]*r[1]-r[0]*p1[1])*det1;\
    alpha1 = (r[0]*q[1] - q[0]*r[1])*det1;\
    alpha1_legal = (alpha1>=0) && (alpha1<=(det1*det1)  && (det1!=0));\
    det2 = p2[0]*q[1] - q[0]*p2[1];\
    alpha2 = (r[0]*q[1] - q[0]*r[1]) *det2;\
    gama2 = (p2[0]*r[1] - r[0]*p2[1]) * det2;\
    alpha2_legal = (alpha2>=0) && (alpha2<=(det2*det2) && (det2 !=0));\
    det3=det2-det1;\
    gama3=((p2[0]-p1[0])*(r[1]-p1[1]) - (r[0]-p1[0])*(p2[1]-p1[1]))*det3;\
    if (alpha1_legal)\
    {\
        if (alpha2_legal)\
        {\
            if ( ((gama1<=0) && (gama1>=-(det1*det1))) || ((gama2<=0) && (gama2>=-(det2*det2))) || (gama1*gama2<0)) return 12;\
        }\
        else\
        {\
            if ( ((gama1<=0) && (gama1>=-(det1*det1))) || ((gama3<=0) && (gama3>=-(det3*det3))) || (gama1*gama3<0)) return 13;\
            }\
    }\
    else\
    if (alpha2_legal)\
    {\
        if ( ((gama2<=0) && (gama2>=-(det2*det2))) || ((gama3<=0) && (gama3>=-(det3*det3))) || (gama2*gama3<0)) return 23;\
        }\
    return 0;\
    }




//main procedure

int tr_tri_intersect3D (PQP_REAL *C1, PQP_REAL *P1, PQP_REAL *P2,
         PQP_REAL *D1, PQP_REAL *Q1, PQP_REAL *Q2)
{
    double  t[3],p1[3], p2[3],r[3],r4[3];
    double beta1, beta2, beta3;
    double gama1, gama2, gama3;
    double det1, det2, det3;
    double dp0, dp1, dp2;
    double dq1,dq2,dq3,dr, dr3;
    double alpha1, alpha2;
    bool alpha1_legal, alpha2_legal;
    double  SF;
    bool beta1_legal, beta2_legal;

    myVmV(r,D1,C1);
    // determinant computation
    dp0 = P1[1]*P2[2]-P2[1]*P1[2];
    dp1 = P1[0]*P2[2]-P2[0]*P1[2];
    dp2 = P1[0]*P2[1]-P2[0]*P1[1];
    dq1 = Q1[0]*dp0 - Q1[1]*dp1 + Q1[2]*dp2;
    dq2 = Q2[0]*dp0 - Q2[1]*dp1 + Q2[2]*dp2;
    dr  = -r[0]*dp0  + r[1]*dp1  - r[2]*dp2;

    beta1 = dr*dq2;  // beta1, beta2 are scaled so that beta_i=beta_i*dq1*dq2
    beta2 = dr*dq1;
    beta1_legal = (beta2>=0) && (beta2 <=dq1*dq1) && (dq1 != 0);
    beta2_legal = (beta1>=0) && (beta1 <=dq2*dq2) && (dq2 != 0);

    dq3=dq2-dq1;
    dr3=+dr-dq1;   // actually this is -dr3


    if ((dq1 == 0) && (dq2 == 0))
    {
        if (dr!=0) return 0;  // triangles are on parallel planes
        else
        {						// triangles are on the same plane
            PQP_REAL C2[3],C3[3],D2[3],D3[3], N1[3];
            // We use the coplanar test of Moller which takes the 6 vertices and 2 normals
            //as input.
            myVpV(C2,C1,P1);
            myVpV(C3,C1,P2);
            myVpV(D2,D1,Q1);
            myVpV(D3,D1,Q2);
            CROSS(N1,P1,P2);
            return coplanar_tri_tri(N1,C1, C2,C3,D1,D2,D3);
        }
    }

    else if (!beta2_legal && !beta1_legal) return 0;// fast reject-all vertices are on
                                                    // the same side of the triangle plane

    else if (beta2_legal && beta1_legal)    //beta1, beta2
    {
        SF = dq1*dq2;
        sVpsV_2(t,beta2,Q2, (-beta1),Q1);
    }

    else if (beta1_legal && !beta2_legal)   //beta1, beta3
    {
        SF = dq1*dq3;
        beta1 =beta1-beta2;   // all betas are multiplied by a positive SF
        beta3 =dr3*dq1;
        sVpsV_2(t,(SF-beta3-beta1),Q1,beta3,Q2);
    }

    else if (beta2_legal && !beta1_legal) //beta2, beta3
    {
        SF = dq2*dq3;
        beta2 =beta1-beta2;   // all betas are multiplied by a positive SF
        beta3 =dr3*dq2;
        sVpsV_2(t,(SF-beta3),Q1,(beta3-beta2),Q2);
        Q1=Q2;
        beta1=beta2;
    }
    sVpsV_2(r4,SF,r,beta1,Q1);
    seg_collide3(t,r4);  // calculates the 2D intersection
    return 0;
}






/* this edge to edge test is based on Franlin Antonio's gem:
   "Faster Line Segment Intersection", in Graphics Gems III,
   pp. 199-202 */
#define FABS(x) (x>=0?x:-x)        /* implement as is fastest on your machine */

#define EDGE_EDGE_TEST(V0,U0,U1)                      \
  Bx=U0[i0]-U1[i0];                                   \
  By=U0[i1]-U1[i1];                                   \
  Cx=V0[i0]-U0[i0];                                   \
  Cy=V0[i1]-U0[i1];                                   \
  f=Ay*Bx-Ax*By;                                      \
  d=By*Cx-Bx*Cy;                                      \
  if((f>0 && d>=0 && d<=f) || (f<0 && d<=0 && d>=f))  \
  {                                                   \
    e=Ax*Cy-Ay*Cx;                                    \
    if(f>0)                                           \
    {                                                 \
      if(e>=0 && e<=f) return 1;                      \
    }                                                 \
    else                                              \
    {                                                 \
      if(e<=0 && e>=f) return 1;                      \
    }                                                 \
  }

#define EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2) \
{                                              \
  double Ax,Ay,Bx,By,Cx,Cy,e,d,f;               \
  Ax=V1[i0]-V0[i0];                            \
  Ay=V1[i1]-V0[i1];                            \
  /* test edge U0,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U0,U1);                    \
  /* test edge U1,U2 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U1,U2);                    \
  /* test edge U2,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U2,U0);                    \
}

#define POINT_IN_TRI(V0,U0,U1,U2)           \
{                                           \
  double a,b,c,d0,d1,d2;                     \
  /* is T1 completly inside T2? */          \
  /* check if V0 is inside tri(U0,U1,U2) */ \
  a=U1[i1]-U0[i1];                          \
  b=-(U1[i0]-U0[i0]);                       \
  c=-a*U0[i0]-b*U0[i1];                     \
  d0=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U2[i1]-U1[i1];                          \
  b=-(U2[i0]-U1[i0]);                       \
  c=-a*U1[i0]-b*U1[i1];                     \
  d1=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U0[i1]-U2[i1];                          \
  b=-(U0[i0]-U2[i0]);                       \
  c=-a*U2[i0]-b*U2[i1];                     \
  d2=a*V0[i0]+b*V0[i1]+c;                   \
  if(d0*d1>0.0)                             \
  {                                         \
    if(d0*d2>0.0) return 1;                 \
  }                                         \
}

//This procedure testing for intersection between coplanar triangles is taken
// from Tomas Moller's
//"A Fast Triangle-Triangle Intersection Test",Journal of Graphics Tools, 2(2), 1997
int coplanar_tri_tri(PQP_REAL N[], PQP_REAL V0[], PQP_REAL V1[], PQP_REAL V2[],
                     PQP_REAL U0[], PQP_REAL U1[], PQP_REAL U2[])
{
   double A[3];
   short i0,i1;
   /* first project onto an axis-aligned plane, that maximizes the area */
   /* of the triangles, compute indices: i0,i1. */
   A[0]=FABS(N[0]);
   A[1]=FABS(N[1]);
   A[2]=FABS(N[2]);
   if(A[0]>A[1])
   {
      if(A[0]>A[2])
      {
          i0=1;      /* A[0] is greatest */
          i1=2;
      }
      else
      {
          i0=0;      /* A[2] is greatest */
          i1=1;
      }
   }
   else   /* A[0]<=A[1] */
   {
      if(A[2]>A[1])
      {
          i0=0;      /* A[2] is greatest */
          i1=1;
      }
      else
      {
          i0=0;      /* A[1] is greatest */
          i1=2;
      }
    }

    /* test all edges of triangle 1 against the edges of triangle 2 */
    EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2);
    EDGE_AGAINST_TRI_EDGES(V1,V2,U0,U1,U2);
    EDGE_AGAINST_TRI_EDGES(V2,V0,U0,U1,U2);

    /* finally, test if tri1 is totally contained in tri2 or vice versa */
    POINT_IN_TRI(V0,U0,U1,U2);
    POINT_IN_TRI(U0,V0,V1,V2);

    return 0;
}

/*
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
using namespace sofa::defaulttype;

inline int doIntersectionTrianglePoint(float dist2, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& n, const Vector3& q, bool swapElems)
{
    const Vector3 AB = p2-p1;
    const Vector3 AC = p3-p1;
    const Vector3 AQ = q -p1;
    Matrix2 A;
    Vector2 b;
    A[0][0] = AB*AB;
    A[1][1] = AC*AC;
    A[0][1] = A[1][0] = AB*AC;
    b[0] = AQ*AB;
    b[1] = AQ*AC;
    const float det = determinant(A);

    float alpha = 0.5;
    float beta = 0.5;

    //if (det < -0.000000000001 || det > 0.000000000001)
    {
        alpha = (b[0]*A[1][1] - b[1]*A[0][1])/det;
        beta  = (b[1]*A[0][0] - b[0]*A[1][0])/det;
        //if (alpha < 0.000001 ||
        //    beta  < 0.000001 ||
        //    alpha + beta  > 0.999999)
        //        return 0;
        if (alpha < 0.000001 || beta < 0.000001 || alpha + beta > 0.999999)
        {
            // nearest point is on an edge or corner
            // barycentric coordinate on AB
            float pAB = b[0] / A[0][0]; // AQ*AB / AB*AB
            // barycentric coordinate on AC
            float pAC = b[1] / A[1][1]; // AQ*AB / AB*AB
            if (pAB < 0.000001 && pAC < 0.0000001)
            {
                // closest point is A
                alpha = 0.0;
                beta = 0.0;
            }
            else if (pAB < 0.999999 && beta < 0.000001)
            {
                // closest point is on AB
                alpha = pAB;
                beta = 0.0;
            }
            else if (pAC < 0.999999 && alpha < 0.000001)
            {
                // closest point is on AC
                alpha = 0.0;
                beta = pAC;
            }
            else
            {
                // barycentric coordinate on BC
                // BQ*BC / BC*BC = (AQ-AB)*(AC-AB) / (AC-AB)*(AC-AB) = (AQ*AC-AQ*AB + AB*AB-AB*AC) / (AB*AB+AC*AC-2AB*AC)
                float pBC = (b[1] - b[0] + A[0][0] - A[0][1]) / (A[0][0] + A[1][1] - 2*A[0][1]); // BQ*BC / BC*BC
                if (pBC < 0.000001)
                {
                    // closest point is B
                    alpha = 1.0;
                    beta = 0.0;
                }
                else if (pBC > 0.999999)
                {
                    // closest point is C
                    alpha = 0.0;
                    beta = 1.0;
                }
                else
                {
                    // closest point is on BC
                    alpha = 1.0-pBC;
                    beta = pBC;
                }
            }
        }
    }

    Vector3 p, pq;
    p = p1 + AB * alpha + AC * beta;
    pq = q-p;
    SReal norm2 = pq.norm2();
    if (norm2 >= dist2)
        return 0;

    return 1;
}


void segNearestPoints(const Vector3 & p0,const Vector3 & p1, const Vector3 & q0,const Vector3 & q1,Vector3 & P,Vector3 & Q,
                                      float & alpha,float & beta){
    const Vector3 AB = p1-p0;
    const Vector3 CD = q1-q0;
    const Vector3 AC = q0-p0;

    Matrix2 Amat;//matrix helping us to find the two nearest points lying on the segments of the two segments
    Vector2 b;

    Amat[0][0] = AB*AB;
    Amat[1][1] = CD*CD;
    Amat[0][1] = Amat[1][0] = -CD*AB;
    b[0] = AB*AC;
    b[1] = -CD*AC;
    const float det = determinant(Amat);

    float AB_norm2 = AB.norm2();
    float CD_norm2 = CD.norm2();
    alpha = 0.5;
    beta = 0.5;
    //Check that the determinant is not null which would mean that the segment segments are lying on a same plane.
    //in this case we can solve the little system which gives us
    //the two coefficients alpha and beta. We obtain the two nearest points P and Q lying on the segments of the two segments.
    //P = A + AB * alpha;
    //Q = C + CD * beta;
    if (det < -1e-6 || det > 1e-6)
    {
        alpha = (b[0]*Amat[1][1] - b[1]*Amat[0][1])/det;
        beta  = (b[1]*Amat[0][0] - b[0]*Amat[1][0])/det;
    }
    else{//segment segments on a same plane. Here the idea to find the nearest points
        //is to project segment apexes on the other segment.
        //Visual example with semgents AB and CD :
        //            A----------------B
        //                     C----------------D
        //After projection :
        //            A--------c-------B
        //                     C-------b--------D
        //So the nearest points are p and q which are respecively in the middle of cB and Cb:
        //            A--------c---p---B
        //                     C---q---b--------D
        Vector3 AD = q1 - p0;
        Vector3 CB = p1 - q0;

        float c_proj= b[0]/AB_norm2;//alpha = (AB * AC)/AB_norm2
        float d_proj = (AB * AD)/AB_norm2;
        float a_proj = b[1]/CD_norm2;//beta = (-CD*AC)/CD_norm2
        float b_proj= (CD*CB)/CD_norm2;

        if(c_proj >= 0 && c_proj <= 1){//projection of C on AB is lying on AB
            if(d_proj > 1){//case :
                           //             A----------------B
                           //                      C---------------D
                alpha = (1.0 + c_proj)/2.0;
                beta = b_proj/2.0;
            }
            else if(d_proj < 0){//case :
                                //             A----------------B
                                //     D----------------C
                alpha = c_proj/2.0;
                beta = (1 + a_proj)/2.0;
            }
            else{//case :
                //             A----------------B
                //                 C------D
                alpha = (c_proj + d_proj)/2.0;
                beta  = 0.5;
            }
        }
        else if(d_proj >= 0 && d_proj <= 1){
            if(c_proj < 0){//case :
                           //             A----------------B
                           //     C----------------D
                alpha = d_proj /2.0;
                beta = (1 + a_proj)/2.0;
            }
            else{//case :
                 //          A---------------B
                 //                 D-------------C
                alpha = (1 + d_proj)/2.0;
                beta = b_proj/2.0;
            }
        }
        else{
            if(c_proj * d_proj < 0){//case :
                                    //           A--------B
                                    //       D-----------------C
                alpha = 0.5;
                beta = (a_proj + b_proj)/2.0;
            }
            else{
                if(c_proj < 0){//case :
                               //                    A---------------B
                               // C-------------D
                    alpha = 0;
                }
                else{
                    alpha = 1;
                }

                if(a_proj < 0){//case :
                               // A---------------B
                               //                     C-------------D
                    beta = 0;
                }
                else{//case :
                     //                     A---------------B
                     //   C-------------D
                    beta = 1;
                }
            }
        }

        P = p0 + AB * alpha;
        Q = q0 + CD * beta;

        return;
    }

    if(alpha < 0){
        alpha = 0;
        beta = (CD * (p0 - q0))/CD_norm2;
    }
    else if(alpha > 1){
        alpha = 1;
        beta = (CD * (p1 - q0))/CD_norm2;
    }

    if(beta < 0){
        beta = 0;
        alpha = (AB * (q0 - p0))/AB_norm2;
    }
    else if(beta > 1){
        beta = 1;
        alpha = (AB * (q1 - p0))/AB_norm2;
    }

    if(alpha < 0)
        alpha = 0;
    else if (alpha > 1)
        alpha = 1;

    assert(alpha >= 0);
    assert(alpha <= 1);
    assert(beta >= 0);
    assert(beta <= 1);

    P = p0 + AB * alpha;
    Q = q0 + CD * beta;
}

inline int doIntersectionLineLine(SReal dist2, const Vector3& p1, const Vector3& p2, const Vector3& q1, const Vector3& q2)
{
    Vector3 p,q;
    float alpha, beta;
    segNearestPoints(p1,p2,q1,q2,p,q,alpha,beta);

    Vector3 pq = p-q;
    float norm2 = pq.norm2();

    if (norm2 >= dist2)
        return 0;

    return 1;
}
*/

// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles
int 
TriContact(PQP_REAL *P1, PQP_REAL *P2, PQP_REAL *P3,
           PQP_REAL *Q1, PQP_REAL *Q2, PQP_REAL *Q3) 
{
    /*const Vector3 p1(P1[0],P1[1],P1[2]);
    const Vector3 p2(P2[0],P2[1],P2[2]);
    const Vector3 p3(P3[0],P3[1],P3[2]);
    const Vector3 pn = (p2 - p1).cross(p3 - p1);
    const Vector3 q1(Q1[0],Q1[1],Q1[2]);
    const Vector3 q2(Q2[0],Q2[1],Q2[2]);
    const Vector3 q3(Q3[0],Q3[1],Q3[2]);
    const Vector3 qn = (q2 - q1).cross(q3 - q1);

    int n = 0;

    float dist2 = 0.01;
    n += doIntersectionTrianglePoint(dist2, q1, q2, q3, qn, p1, true);
    n += doIntersectionTrianglePoint(dist2, q1, q2, q3, qn, p2, true);
    n += doIntersectionTrianglePoint(dist2, q1, q2, q3, qn, p3, true);

    n += doIntersectionTrianglePoint(dist2, p1, p2, p3, pn, q1, false);
    n += doIntersectionTrianglePoint(dist2, p1, p2, p3, pn, q2, false);
    n += doIntersectionTrianglePoint(dist2, p1, p2, p3, pn, q3, false);

    n += doIntersectionLineLine(dist2, p1, p2, q1, q2);
    n += doIntersectionLineLine(dist2, p1, p2, q2, q3);
    n += doIntersectionLineLine(dist2, p1, p2, q3, q1);
    n += doIntersectionLineLine(dist2, p2, p3, q1, q2);
    n += doIntersectionLineLine(dist2, p2, p3, q2, q3);
    n += doIntersectionLineLine(dist2, p2, p3, q3, q1);
    n += doIntersectionLineLine(dist2, p3, p1, q1, q2);
    n += doIntersectionLineLine(dist2, p3, p1, q2, q3);
    n += doIntersectionLineLine(dist2, p3, p1, q3, q1);

    return n > 0;*/

   /*PQP_REAL v1[3], v2[3], w1[3], w2[3];
   v1[0] = P2[0] - P1[0]; v1[1] = P2[1] - P1[1]; v1[2] = P2[2] - P1[2];
   v2[0] = P3[0] - P1[0]; v2[1] = P3[1] - P1[1]; v2[2] = P3[2] - P1[2];

   w1[0] = Q2[0] - Q1[0]; w1[1] = Q2[1] - Q1[1]; w1[2] = Q2[2] - Q1[2];
   w2[0] = Q3[0] - Q1[0]; w2[1] = Q3[1] - Q1[1]; w2[2] = Q3[2] - Q1[2];

   return tr_tri_intersect3D(P1, v1, v2, Q1, w1, w2);*/


  // One triangle is (p1,p2,p3).  Other is (q1,q2,q3).
  // Edges are (e1,e2,e3) and (f1,f2,f3).
  // Normals are n1 and m1
  // Outwards are (g1,g2,g3) and (h1,h2,h3).
  //  
  // We assume that the triangle vertices are in the same coordinate system.
  //
  // First thing we do is establish a new c.s. so that p1 is at (0,0,0).

//    PQP_REAL p1[3], p2[3], p3[3];
//    PQP_REAL q1[3], q2[3], q3[3];
//    PQP_REAL e1[3], e2[3], e3[3];
//    PQP_REAL f1[3], f2[3], f3[3];

//    p1[0] = 0;             p1[1] = 0;             p1[2] = 0;
//    p2[0] = P2[0] - P1[0]; p2[1] = P2[1] - P1[1]; p2[2] = P2[2] - P1[2];
//    p3[0] = P3[0] - P1[0]; p3[1] = P3[1] - P1[1]; p3[2] = P3[2] - P1[2];

//    q1[0] = Q1[0] - P1[0]; q1[1] = Q1[1] - P1[1]; q1[2] = Q1[2] - P1[2];
//    q2[0] = Q2[0] - P1[0]; q2[1] = Q2[1] - P1[1]; q2[2] = Q2[2] - P1[2];
//    q3[0] = Q3[0] - P1[0]; q3[1] = Q3[1] - P1[1]; q3[2] = Q3[2] - P1[2];

//    e1[0] = p2[0] - p1[0]; e1[1] = p2[1] - p1[1]; e1[2] = p2[2] - p1[2];
//    e2[0] = p3[0] - p2[0]; e2[1] = p3[1] - p2[1]; e2[2] = p3[2] - p2[2];
//    e3[0] = p1[0] - p3[0]; e3[1] = p1[1] - p3[1]; e3[2] = p1[2] - p3[2];

//    f1[0] = q2[0] - q1[0]; e1[1] = q2[1] - q1[1]; e1[2] = q2[2] - q1[2];
//    f2[0] = q3[0] - q2[0]; e2[1] = q3[1] - q2[1]; e2[2] = q3[2] - q2[2];
//    f3[0] = q1[0] - q3[0]; e3[1] = q1[1] - q3[1]; e3[2] = q1[2] - q3[2];

//    PQP_REAL t[3];
//    PQP_REAL m1[3], n1[3];

//    VcrossV(n1, e1, e2);
//    if(!project6(n1, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(m1, f1, f2);
//    if(!project6(m1, p1, p2, p3, q1, q2, q3)) return 0;


//    VcrossV(t, e1, f1);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, e1, f2);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, e1, f3);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, e2, f1);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, e2, f2);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, e2, f3);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, e3, f1);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, e3, f2);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, e3, f3);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;


//    VcrossV(t, e1, n1);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, e2, n1);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, e3, n1);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, f1, m1);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, f2, m1);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;
//    VcrossV(t, f3, m1);
//    if(!project6(t, p1, p2, p3, q1, q2, q3)) return 0;

//    return 1;

  PQP_REAL p1[3], p2[3], p3[3];
  PQP_REAL q1[3], q2[3], q3[3];
  PQP_REAL e1[3], e2[3], e3[3];
  PQP_REAL f1[3], f2[3], f3[3];
  PQP_REAL g1[3], g2[3], g3[3];
  PQP_REAL h1[3], h2[3], h3[3];
  PQP_REAL n1[3], m1[3];

  PQP_REAL ef11[3], ef12[3], ef13[3];
  PQP_REAL ef21[3], ef22[3], ef23[3];
  PQP_REAL ef31[3], ef32[3], ef33[3];
  
  p1[0] = P1[0] - P1[0];  p1[1] = P1[1] - P1[1];  p1[2] = P1[2] - P1[2];
  p2[0] = P2[0] - P1[0];  p2[1] = P2[1] - P1[1];  p2[2] = P2[2] - P1[2];
  p3[0] = P3[0] - P1[0];  p3[1] = P3[1] - P1[1];  p3[2] = P3[2] - P1[2];
  
  q1[0] = Q1[0] - P1[0];  q1[1] = Q1[1] - P1[1];  q1[2] = Q1[2] - P1[2];
  q2[0] = Q2[0] - P1[0];  q2[1] = Q2[1] - P1[1];  q2[2] = Q2[2] - P1[2];
  q3[0] = Q3[0] - P1[0];  q3[1] = Q3[1] - P1[1];  q3[2] = Q3[2] - P1[2];
  
  e1[0] = p2[0] - p1[0];  e1[1] = p2[1] - p1[1];  e1[2] = p2[2] - p1[2];
  e2[0] = p3[0] - p2[0];  e2[1] = p3[1] - p2[1];  e2[2] = p3[2] - p2[2];
  e3[0] = p1[0] - p3[0];  e3[1] = p1[1] - p3[1];  e3[2] = p1[2] - p3[2];

  f1[0] = q2[0] - q1[0];  f1[1] = q2[1] - q1[1];  f1[2] = q2[2] - q1[2];
  f2[0] = q3[0] - q2[0];  f2[1] = q3[1] - q2[1];  f2[2] = q3[2] - q2[2];
  f3[0] = q1[0] - q3[0];  f3[1] = q1[1] - q3[1];  f3[2] = q1[2] - q3[2];
  
  VcrossV(n1, e1, e2);
  VcrossV(m1, f1, f2);

  VcrossV(g1, e1, n1);
  VcrossV(g2, e2, n1);
  VcrossV(g3, e3, n1);
  VcrossV(h1, f1, m1);
  VcrossV(h2, f2, m1);
  VcrossV(h3, f3, m1);

  VcrossV(ef11, e1, f1);
  VcrossV(ef12, e1, f2);
  VcrossV(ef13, e1, f3);
  VcrossV(ef21, e2, f1);
  VcrossV(ef22, e2, f2);
  VcrossV(ef23, e2, f3);
  VcrossV(ef31, e3, f1);
  VcrossV(ef32, e3, f2);
  VcrossV(ef33, e3, f3);
  
  // now begin the series of tests

  if (!project6(n1, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(m1, p1, p2, p3, q1, q2, q3)) return 0;
  
  if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return 0;

  if (!project6(g1, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(g2, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(g3, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(h1, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(h2, p1, p2, p3, q1, q2, q3)) return 0;
  if (!project6(h3, p1, p2, p3, q1, q2, q3)) return 0;

  return 1;
}

inline
PQP_REAL
TriDistance(PQP_REAL R[3][3], PQP_REAL T[3], Tri *t1, Tri *t2,
            PQP_REAL p[3], PQP_REAL q[3])
{
  // transform tri 2 into same space as tri 1

  PQP_REAL tri1[3][3], tri2[3][3];

  VcV(tri1[0], t1->p1);
  VcV(tri1[1], t1->p2);
  VcV(tri1[2], t1->p3);
  MxVpV(tri2[0], R, t2->p1, T);
  MxVpV(tri2[1], R, t2->p2, T);
  MxVpV(tri2[2], R, t2->p3, T);
                                
  return TriDist(p,q,tri1,tri2);
}


void
CollideRecurse(PQP_CollideResult *res,
               PQP_REAL R[3][3], PQP_REAL T[3], // b2 relative to b1
               PQP_Model *o1, int b1, 
               PQP_Model *o2, int b2, int flag)
{
    std::cout << " CollideRecurse(): " << b1 << " - " << b2;
  // first thing, see if we're overlapping

  res->num_bv_tests++;

  if (!BV_Overlap(R, T, o1->child(b1), o2->child(b2)))
  {
      return;
  }

  res->AddOBB(b1, b2);
  // if we are, see if we test triangles next

  int l1 = o1->child(b1)->Leaf();
  int l2 = o2->child(b2)->Leaf();

  if (l1 && l2)
  {
    res->num_tri_tests++;

#if 1

    Tri *t1 = &o1->tris[-o1->child(b1)->first_child - 1];
    Tri *t2 = &o2->tris[-o2->child(b2)->first_child - 1];

    res->Add(t1->id, t2->id);

    // transform the points in b2 into space of b1, then compare

    PQP_REAL q1[3], q2[3], q3[3];
    PQP_REAL *p1 = t1->p1;
    PQP_REAL *p2 = t1->p2;
    PQP_REAL *p3 = t1->p3;

    MxVpV(q1, res->R, t2->p1, res->T);
    MxVpV(q2, res->R, t2->p2, res->T);
    MxVpV(q3, res->R, t2->p3, res->T);

    PQP_REAL *r1 = t2->p1;
    PQP_REAL *r2 = t2->p2;
    PQP_REAL *r3 = t2->p3;

    if (TriContact(p1, p2, p3, q1, q2, q3))
    {
      // add this to result
      res->Add(t1->id, t2->id);
    }


#else
    PQP_REAL p[3], q[3];

    Tri *t1 = &o1->tris[-o1->child(b1)->first_child - 1];
    Tri *t2 = &o2->tris[-o2->child(b2)->first_child - 1];

    if (TriDistance(res->R,res->T,t1,t2,p,q) == 0.0)
    {
      // add this to result

      res->Add(t1->id, t2->id);
    }
#endif
    return;
  }

  // we dont, so decide whose children to visit next

  PQP_REAL sz1 = o1->child(b1)->GetSize();
  PQP_REAL sz2 = o2->child(b2)->GetSize();

  PQP_REAL Rc[3][3],Tc[3],Ttemp[3];
    
  if (l2 || (!l1 && (sz1 > sz2)))
  {
    int c1 = o1->child(b1)->first_child;
    int c2 = c1 + 1;

    MTxM(Rc,o1->child(c1)->R,R);
#if PQP_BV_TYPE & OBB_TYPE
    VmV(Ttemp,T,o1->child(c1)->To);
#else
    VmV(Ttemp,T,o1->child(c1)->Tr);
#endif
    MTxV(Tc,o1->child(c1)->R,Ttemp);

    CollideRecurse(res,Rc,Tc,o1,c1,o2,b2,flag);

    if ((flag == PQP_FIRST_CONTACT) && (res->num_pairs > 0)) return;

    MTxM(Rc,o1->child(c2)->R,R);
#if PQP_BV_TYPE & OBB_TYPE
    VmV(Ttemp,T,o1->child(c2)->To);
#else
    VmV(Ttemp,T,o1->child(c2)->Tr);
#endif
    MTxV(Tc,o1->child(c2)->R,Ttemp);

    CollideRecurse(res,Rc,Tc,o1,c2,o2,b2,flag);
  }
  else 
  {
    int c1 = o2->child(b2)->first_child;
    int c2 = c1 + 1;

    MxM(Rc,R,o2->child(c1)->R);
#if PQP_BV_TYPE & OBB_TYPE
    MxVpV(Tc,R,o2->child(c1)->To,T);
#else
    MxVpV(Tc,R,o2->child(c1)->Tr,T);
#endif

    CollideRecurse(res,Rc,Tc,o1,b1,o2,c1,flag);

    if ((flag == PQP_FIRST_CONTACT) && (res->num_pairs > 0)) return;

    MxM(Rc,R,o2->child(c2)->R);
#if PQP_BV_TYPE & OBB_TYPE
    MxVpV(Tc,R,o2->child(c2)->To,T);
#else
    MxVpV(Tc,R,o2->child(c2)->Tr,T);
#endif
    CollideRecurse(res,Rc,Tc,o1,b1,o2,c2,flag);
  }
}

int 
PQP_Collide(PQP_CollideResult *res,
            PQP_REAL R1[3][3], PQP_REAL T1[3], PQP_Model *o1,
            PQP_REAL R2[3][3], PQP_REAL T2[3], PQP_Model *o2,
            int flag)
{
  double t1 = GetTime();

  // make sure that the models are built

  if (o1->build_state != PQP_BUILD_STATE_PROCESSED) 
    return PQP_ERR_UNPROCESSED_MODEL;
  if (o2->build_state != PQP_BUILD_STATE_PROCESSED) 
    return PQP_ERR_UNPROCESSED_MODEL;

  // clear the stats

  res->num_bv_tests = 0;
  res->num_tri_tests = 0;
  
  // don't release the memory, but reset the num_pairs counter

  res->num_pairs = 0;
  
  // Okay, compute what transform [R,T] that takes us from cs1 to cs2.
  // [R,T] = [R1,T1]'[R2,T2] = [R1',-R1'T1][R2,T2] = [R1'R2, R1'(T2-T1)]
  // First compute the rotation part, then translation part

  MTxM(res->R,R1,R2);

  std::cout << "=== Transform from CS1 to CS2:" << std::endl << "Rotation:" << std::endl;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cout << res->R[i][j] << ";";
        }
        std::cout << std::endl;
    }

  PQP_REAL Ttemp[3];
  VmV(Ttemp, T2, T1);

  std::cout << "Translation without rotation: " << Ttemp[0] << "," << Ttemp[1] << "," << Ttemp[2] << std::endl;

  MTxV(res->T, R1, Ttemp);

  std::cout << "Translation with rotation   : " << res->T[0] << "," << res->T[1] << "," << res->T[2] << std::endl;
  
  // compute the transform from o1->child(0) to o2->child(0)

  PQP_REAL Rtemp[3][3], R[3][3], T[3];

  MxM(Rtemp,res->R,o2->child(0)->R);
  MTxM(R,o1->child(0)->R,Rtemp);

    std::cout << "Rotation between top OBBs, step 1:" << std::endl;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cout << Rtemp[i][j] << ";";
        }
        std::cout << std::endl;
    }

    std::cout << "Rotation between top OBBs, step 2:" << std::endl;
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
          std::cout << R[i][j] << ";";
      }
      std::cout << std::endl;
    }

#if PQP_BV_TYPE & OBB_TYPE
  MxVpV(Ttemp,res->R,o2->child(0)->To,res->T);
  std::cout << "Translation between top OBBs, step 1: " << Ttemp[0] << "," << Ttemp[1] << "," << Ttemp[2] << std::endl;

  VmV(Ttemp,Ttemp,o1->child(0)->To);
  std::cout << "Translation between top OBBs, step 2: " << Ttemp[0] << "," << Ttemp[1] << "," << Ttemp[2] << std::endl;
#else
  MxVpV(Ttemp,res->R,o2->child(0)->Tr,res->T);
  VmV(Ttemp,Ttemp,o1->child(0)->Tr);
#endif

  MTxV(T,o1->child(0)->R,Ttemp);
  std::cout << "Translation between top OBBs, step 3: " << Ttemp[0] << "," << Ttemp[1] << "," << Ttemp[2] << std::endl;

  // now start with both top level BVs  

  CollideRecurse(res,R,T,o1,0,o2,0,flag);
  
  double t2 = GetTime();
  res->query_time_secs = t2 - t1;
  
  return PQP_OK; 
}

#if PQP_BV_TYPE & RSS_TYPE // distance/tolerance only available with RSS
                           // unless an OBB distance test is supplied in 
                           // BV.cpp

// DISTANCE STUFF
//
//--------------------------------------------------------------------------

void
DistanceRecurse(PQP_DistanceResult *res,
                PQP_REAL R[3][3], PQP_REAL T[3], // b2 relative to b1
                PQP_Model *o1, int b1,
                PQP_Model *o2, int b2)
{
  PQP_REAL sz1 = o1->child(b1)->GetSize();
  PQP_REAL sz2 = o2->child(b2)->GetSize();
  int l1 = o1->child(b1)->Leaf();
  int l2 = o2->child(b2)->Leaf();

  if (l1 && l2)
  {
    // both leaves.  Test the triangles beneath them.

    res->num_tri_tests++;

    PQP_REAL p[3], q[3];

    Tri *t1 = &o1->tris[-o1->child(b1)->first_child - 1];
    Tri *t2 = &o2->tris[-o2->child(b2)->first_child - 1];

    PQP_REAL d = TriDistance(res->R,res->T,t1,t2,p,q);
  
    if (d < res->distance) 
    {
      res->distance = d;

      VcV(res->p1, p);         // p already in c.s. 1
      VcV(res->p2, q);         // q must be transformed 
                               // into c.s. 2 later
      o1->last_tri = t1;
      o2->last_tri = t2;
    }

    return;
  }

  // First, perform distance tests on the children. Then traverse 
  // them recursively, but test the closer pair first, the further 
  // pair second.

  int a1,a2,c1,c2;  // new bv tests 'a' and 'c'
  PQP_REAL R1[3][3], T1[3], R2[3][3], T2[3], Ttemp[3];

  if (l2 || (!l1 && (sz1 > sz2)))
  {
    // visit the children of b1

    a1 = o1->child(b1)->first_child;
    a2 = b2;
    c1 = o1->child(b1)->first_child+1;
    c2 = b2;
    
    MTxM(R1,o1->child(a1)->R,R);
#if PQP_BV_TYPE & RSS_TYPE
    VmV(Ttemp,T,o1->child(a1)->Tr);
#else
    VmV(Ttemp,T,o1->child(a1)->To);
#endif
    MTxV(T1,o1->child(a1)->R,Ttemp);

    MTxM(R2,o1->child(c1)->R,R);
#if PQP_BV_TYPE & RSS_TYPE
    VmV(Ttemp,T,o1->child(c1)->Tr);
#else
    VmV(Ttemp,T,o1->child(c1)->To);
#endif
    MTxV(T2,o1->child(c1)->R,Ttemp);
  }
  else 
  {
    // visit the children of b2

    a1 = b1;
    a2 = o2->child(b2)->first_child;
    c1 = b1;
    c2 = o2->child(b2)->first_child+1;

    MxM(R1,R,o2->child(a2)->R);
#if PQP_BV_TYPE & RSS_TYPE
    MxVpV(T1,R,o2->child(a2)->Tr,T);
#else
    MxVpV(T1,R,o2->child(a2)->To,T);
#endif

    MxM(R2,R,o2->child(c2)->R);
#if PQP_BV_TYPE & RSS_TYPE
    MxVpV(T2,R,o2->child(c2)->Tr,T);
#else
    MxVpV(T2,R,o2->child(c2)->To,T);
#endif
  }

  res->num_bv_tests += 2;

  PQP_REAL d1 = BV_Distance(R1, T1, o1->child(a1), o2->child(a2));
  PQP_REAL d2 = BV_Distance(R2, T2, o1->child(c1), o2->child(c2));

  if (d2 < d1)
  {
    if ((d2 < (res->distance - res->abs_err)) || 
        (d2*(1 + res->rel_err) < res->distance)) 
    {      
      DistanceRecurse(res, R2, T2, o1, c1, o2, c2);      
    }

    if ((d1 < (res->distance - res->abs_err)) || 
        (d1*(1 + res->rel_err) < res->distance)) 
    {      
      DistanceRecurse(res, R1, T1, o1, a1, o2, a2);
    }
  }
  else 
  {
    if ((d1 < (res->distance - res->abs_err)) || 
        (d1*(1 + res->rel_err) < res->distance)) 
    {      
      DistanceRecurse(res, R1, T1, o1, a1, o2, a2);
    }

    if ((d2 < (res->distance - res->abs_err)) || 
        (d2*(1 + res->rel_err) < res->distance)) 
    {      
      DistanceRecurse(res, R2, T2, o1, c1, o2, c2);      
    }
  }
}

void
DistanceQueueRecurse(PQP_DistanceResult *res, 
                     PQP_REAL R[3][3], PQP_REAL T[3],
                     PQP_Model *o1, int b1,
                     PQP_Model *o2, int b2)
{
  BVTQ bvtq(res->qsize);

  BVT min_test;
  min_test.b1 = b1;
  min_test.b2 = b2;
  McM(min_test.R,R);
  VcV(min_test.T,T);

  while(1) 
  {  
    int l1 = o1->child(min_test.b1)->Leaf();
    int l2 = o2->child(min_test.b2)->Leaf();
    
    if (l1 && l2) 
    {  
      // both leaves.  Test the triangles beneath them.

      res->num_tri_tests++;

      PQP_REAL p[3], q[3];

      Tri *t1 = &o1->tris[-o1->child(min_test.b1)->first_child - 1];
      Tri *t2 = &o2->tris[-o2->child(min_test.b2)->first_child - 1];

      PQP_REAL d = TriDistance(res->R,res->T,t1,t2,p,q);
  
      if (d < res->distance)
      {
        res->distance = d;

        VcV(res->p1, p);         // p already in c.s. 1
        VcV(res->p2, q);         // q must be transformed 
                                 // into c.s. 2 later
        o1->last_tri = t1;
        o2->last_tri = t2;
      }
    }		 
    else if (bvtq.GetNumTests() == bvtq.GetSize() - 1) 
    {  
      // queue can't get two more tests, recur
      
      DistanceQueueRecurse(res,min_test.R,min_test.T,
                           o1,min_test.b1,o2,min_test.b2);
    }
    else 
    {  
      // decide how to descend to children
      
      PQP_REAL sz1 = o1->child(min_test.b1)->GetSize();
      PQP_REAL sz2 = o2->child(min_test.b2)->GetSize();

      res->num_bv_tests += 2;
 
      BVT bvt1,bvt2;
      PQP_REAL Ttemp[3];

      if (l2 || (!l1 && (sz1 > sz2)))	
      {  
        // put new tests on queue consisting of min_test.b2 
        // with children of min_test.b1 
      
        int c1 = o1->child(min_test.b1)->first_child;
        int c2 = c1 + 1;

        // init bv test 1

        bvt1.b1 = c1;
        bvt1.b2 = min_test.b2;
        MTxM(bvt1.R,o1->child(c1)->R,min_test.R);
#if PQP_BV_TYPE & RSS_TYPE
        VmV(Ttemp,min_test.T,o1->child(c1)->Tr);
#else
        VmV(Ttemp,min_test.T,o1->child(c1)->To);
#endif
        MTxV(bvt1.T,o1->child(c1)->R,Ttemp);
        bvt1.d = BV_Distance(bvt1.R,bvt1.T,
                            o1->child(bvt1.b1),o2->child(bvt1.b2));

        // init bv test 2

        bvt2.b1 = c2;
        bvt2.b2 = min_test.b2;
        MTxM(bvt2.R,o1->child(c2)->R,min_test.R);
#if PQP_BV_TYPE & RSS_TYPE
        VmV(Ttemp,min_test.T,o1->child(c2)->Tr);
#else
        VmV(Ttemp,min_test.T,o1->child(c2)->To);
#endif
        MTxV(bvt2.T,o1->child(c2)->R,Ttemp);
        bvt2.d = BV_Distance(bvt2.R,bvt2.T,
                            o1->child(bvt2.b1),o2->child(bvt2.b2));
      }
      else 
      {
        // put new tests on queue consisting of min_test.b1 
        // with children of min_test.b2
      
        int c1 = o2->child(min_test.b2)->first_child;
        int c2 = c1 + 1;

        // init bv test 1

        bvt1.b1 = min_test.b1;
        bvt1.b2 = c1;
        MxM(bvt1.R,min_test.R,o2->child(c1)->R);
#if PQP_BV_TYPE & RSS_TYPE
        MxVpV(bvt1.T,min_test.R,o2->child(c1)->Tr,min_test.T);
#else
        MxVpV(bvt1.T,min_test.R,o2->child(c1)->To,min_test.T);
#endif
        bvt1.d = BV_Distance(bvt1.R,bvt1.T,
                            o1->child(bvt1.b1),o2->child(bvt1.b2));

        // init bv test 2

        bvt2.b1 = min_test.b1;
        bvt2.b2 = c2;
        MxM(bvt2.R,min_test.R,o2->child(c2)->R);
#if PQP_BV_TYPE & RSS_TYPE
        MxVpV(bvt2.T,min_test.R,o2->child(c2)->Tr,min_test.T);
#else
        MxVpV(bvt2.T,min_test.R,o2->child(c2)->To,min_test.T);
#endif
        bvt2.d = BV_Distance(bvt2.R,bvt2.T,
                            o1->child(bvt2.b1),o2->child(bvt2.b2));
      }

      bvtq.AddTest(bvt1);	
      bvtq.AddTest(bvt2);
    }

    if (bvtq.Empty())
    {
      break;
    }
    else
    {
      min_test = bvtq.ExtractMinTest();

      if ((min_test.d + res->abs_err >= res->distance) && 
         ((min_test.d * (1 + res->rel_err)) >= res->distance)) 
      {
        break;
      }
    }
  }  
}	

int 
PQP_Distance(PQP_DistanceResult *res,
             PQP_REAL R1[3][3], PQP_REAL T1[3], PQP_Model *o1,
             PQP_REAL R2[3][3], PQP_REAL T2[3], PQP_Model *o2,
             PQP_REAL rel_err, PQP_REAL abs_err,
             int qsize)
{
  
  double time1 = GetTime();
  
  // make sure that the models are built

  if (o1->build_state != PQP_BUILD_STATE_PROCESSED) 
    return PQP_ERR_UNPROCESSED_MODEL;
  if (o2->build_state != PQP_BUILD_STATE_PROCESSED) 
    return PQP_ERR_UNPROCESSED_MODEL;

  // Okay, compute what transform [R,T] that takes us from cs2 to cs1.
  // [R,T] = [R1,T1]'[R2,T2] = [R1',-R1'T][R2,T2] = [R1'R2, R1'(T2-T1)]
  // First compute the rotation part, then translation part

  MTxM(res->R,R1,R2);
  PQP_REAL Ttemp[3];
  VmV(Ttemp, T2, T1);  
  MTxV(res->T, R1, Ttemp);
  
  // establish initial upper bound using last triangles which 
  // provided the minimum distance

  PQP_REAL p[3],q[3];
  res->distance = TriDistance(res->R,res->T,o1->last_tri,o2->last_tri,p,q);
  VcV(res->p1,p);
  VcV(res->p2,q);

  // initialize error bounds

  res->abs_err = abs_err;
  res->rel_err = rel_err;
  
  // clear the stats

  res->num_bv_tests = 0;
  res->num_tri_tests = 0;
  
  // compute the transform from o1->child(0) to o2->child(0)

  PQP_REAL Rtemp[3][3], R[3][3], T[3];

  MxM(Rtemp,res->R,o2->child(0)->R);
  MTxM(R,o1->child(0)->R,Rtemp);
  
#if PQP_BV_TYPE & RSS_TYPE
  MxVpV(Ttemp,res->R,o2->child(0)->Tr,res->T);
  VmV(Ttemp,Ttemp,o1->child(0)->Tr);
#else
  MxVpV(Ttemp,res->R,o2->child(0)->To,res->T);
  VmV(Ttemp,Ttemp,o1->child(0)->To);
#endif
  MTxV(T,o1->child(0)->R,Ttemp);

  // choose routine according to queue size
  
  if (qsize <= 2)
  {
    DistanceRecurse(res,R,T,o1,0,o2,0);    
  }
  else 
  { 
    res->qsize = qsize;

    DistanceQueueRecurse(res,R,T,o1,0,o2,0);
  }

  // res->p2 is in cs 1 ; transform it to cs 2

  PQP_REAL u[3];
  VmV(u, res->p2, res->T);
  MTxV(res->p2, res->R, u);

  double time2 = GetTime();
  res->query_time_secs = time2 - time1;  

  return PQP_OK;
}

// Tolerance Stuff
//
//---------------------------------------------------------------------------
void 
ToleranceRecurse(PQP_ToleranceResult *res, 
                 PQP_REAL R[3][3], PQP_REAL T[3],
                 PQP_Model *o1, int b1, PQP_Model *o2, int b2)
{
  PQP_REAL sz1 = o1->child(b1)->GetSize();
  PQP_REAL sz2 = o2->child(b2)->GetSize();
  int l1 = o1->child(b1)->Leaf();
  int l2 = o2->child(b2)->Leaf();

  if (l1 && l2) 
  {
    // both leaves - find if tri pair within tolerance
    
    res->num_tri_tests++;

    PQP_REAL p[3], q[3];

    Tri *t1 = &o1->tris[-o1->child(b1)->first_child - 1];
    Tri *t2 = &o2->tris[-o2->child(b2)->first_child - 1];

    PQP_REAL d = TriDistance(res->R,res->T,t1,t2,p,q);
    
    if (d <= res->tolerance)  
    {  
      // triangle pair distance less than tolerance

      res->closer_than_tolerance = 1;
      res->distance = d;
      VcV(res->p1, p);         // p already in c.s. 1
      VcV(res->p2, q);         // q must be transformed 
                               // into c.s. 2 later
    }

    return;
  }

  int a1,a2,c1,c2;  // new bv tests 'a' and 'c'
  PQP_REAL R1[3][3], T1[3], R2[3][3], T2[3], Ttemp[3];

  if (l2 || (!l1 && (sz1 > sz2)))
  {
    // visit the children of b1

    a1 = o1->child(b1)->first_child;
    a2 = b2;
    c1 = o1->child(b1)->first_child+1;
    c2 = b2;
    
    MTxM(R1,o1->child(a1)->R,R);
#if PQP_BV_TYPE & RSS_TYPE
    VmV(Ttemp,T,o1->child(a1)->Tr);
#else
    VmV(Ttemp,T,o1->child(a1)->To);
#endif
    MTxV(T1,o1->child(a1)->R,Ttemp);

    MTxM(R2,o1->child(c1)->R,R);
#if PQP_BV_TYPE & RSS_TYPE
    VmV(Ttemp,T,o1->child(c1)->Tr);
#else
    VmV(Ttemp,T,o1->child(c1)->To);
#endif
    MTxV(T2,o1->child(c1)->R,Ttemp);
  }
  else 
  {
    // visit the children of b2

    a1 = b1;
    a2 = o2->child(b2)->first_child;
    c1 = b1;
    c2 = o2->child(b2)->first_child+1;

    MxM(R1,R,o2->child(a2)->R);
#if PQP_BV_TYPE & RSS_TYPE
    MxVpV(T1,R,o2->child(a2)->Tr,T);
#else
    MxVpV(T1,R,o2->child(a2)->To,T);
#endif
    MxM(R2,R,o2->child(c2)->R);
#if PQP_BV_TYPE & RSS_TYPE
    MxVpV(T2,R,o2->child(c2)->Tr,T);
#else
    MxVpV(T2,R,o2->child(c2)->To,T);
#endif
  }

  res->num_bv_tests += 2;

  PQP_REAL d1 = BV_Distance(R1, T1, o1->child(a1), o2->child(a2));
  PQP_REAL d2 = BV_Distance(R2, T2, o1->child(c1), o2->child(c2));

  if (d2 < d1) 
  {
    if (d2 <= res->tolerance) ToleranceRecurse(res, R2, T2, o1, c1, o2, c2);
    if (res->closer_than_tolerance) return;
    if (d1 <= res->tolerance) ToleranceRecurse(res, R1, T1, o1, a1, o2, a2);
  }
  else 
  {
    if (d1 <= res->tolerance) ToleranceRecurse(res, R1, T1, o1, a1, o2, a2);
    if (res->closer_than_tolerance) return;
    if (d2 <= res->tolerance) ToleranceRecurse(res, R2, T2, o1, c1, o2, c2);
  }
}

void
ToleranceQueueRecurse(PQP_ToleranceResult *res,
                      PQP_REAL R[3][3], PQP_REAL T[3],
                      PQP_Model *o1, int b1,
                      PQP_Model *o2, int b2)
{
  BVTQ bvtq(res->qsize);
  BVT min_test;
  min_test.b1 = b1;
  min_test.b2 = b2;
  McM(min_test.R,R);
  VcV(min_test.T,T);

  while(1)
  {  
    int l1 = o1->child(min_test.b1)->Leaf();
    int l2 = o2->child(min_test.b2)->Leaf();
    
    if (l1 && l2) 
    {  
      // both leaves - find if tri pair within tolerance
    
      res->num_tri_tests++;

      PQP_REAL p[3], q[3];

      Tri *t1 = &o1->tris[-o1->child(min_test.b1)->first_child - 1];
      Tri *t2 = &o2->tris[-o2->child(min_test.b2)->first_child - 1];

      PQP_REAL d = TriDistance(res->R,res->T,t1,t2,p,q);
    
      if (d <= res->tolerance)  
      {  
        // triangle pair distance less than tolerance

        res->closer_than_tolerance = 1;
        res->distance = d;
        VcV(res->p1, p);         // p already in c.s. 1
        VcV(res->p2, q);         // q must be transformed 
                                 // into c.s. 2 later
        return;
      }
    }
    else if (bvtq.GetNumTests() == bvtq.GetSize() - 1)
    {  
      // queue can't get two more tests, recur
      
      ToleranceQueueRecurse(res,min_test.R,min_test.T,
                            o1,min_test.b1,o2,min_test.b2);
      if (res->closer_than_tolerance == 1) return;
    }
    else 
    {  
      // decide how to descend to children
      
      PQP_REAL sz1 = o1->child(min_test.b1)->GetSize();
      PQP_REAL sz2 = o2->child(min_test.b2)->GetSize();

      res->num_bv_tests += 2;
      
      BVT bvt1,bvt2;
      PQP_REAL Ttemp[3];

      if (l2 || (!l1 && (sz1 > sz2)))	
      {
	      // add two new tests to queue, consisting of min_test.b2
        // with the children of min_test.b1

        int c1 = o1->child(min_test.b1)->first_child;
        int c2 = c1 + 1;

        // init bv test 1

        bvt1.b1 = c1;
        bvt1.b2 = min_test.b2;
        MTxM(bvt1.R,o1->child(c1)->R,min_test.R);
#if PQP_BV_TYPE & RSS_TYPE
        VmV(Ttemp,min_test.T,o1->child(c1)->Tr);
#else
        VmV(Ttemp,min_test.T,o1->child(c1)->To);
#endif
        MTxV(bvt1.T,o1->child(c1)->R,Ttemp);
        bvt1.d = BV_Distance(bvt1.R,bvt1.T,
                            o1->child(bvt1.b1),o2->child(bvt1.b2));

	      // init bv test 2

	      bvt2.b1 = c2;
	      bvt2.b2 = min_test.b2;
	      MTxM(bvt2.R,o1->child(c2)->R,min_test.R);
#if PQP_BV_TYPE & RSS_TYPE
	      VmV(Ttemp,min_test.T,o1->child(c2)->Tr);
#else
	      VmV(Ttemp,min_test.T,o1->child(c2)->To);
#endif
	      MTxV(bvt2.T,o1->child(c2)->R,Ttemp);
        bvt2.d = BV_Distance(bvt2.R,bvt2.T,
                            o1->child(bvt2.b1),o2->child(bvt2.b2));
      }
      else 
      {
        // add two new tests to queue, consisting of min_test.b1
        // with the children of min_test.b2

        int c1 = o2->child(min_test.b2)->first_child;
        int c2 = c1 + 1;

        // init bv test 1

        bvt1.b1 = min_test.b1;
        bvt1.b2 = c1;
        MxM(bvt1.R,min_test.R,o2->child(c1)->R);
#if PQP_BV_TYPE & RSS_TYPE
        MxVpV(bvt1.T,min_test.R,o2->child(c1)->Tr,min_test.T);
#else
        MxVpV(bvt1.T,min_test.R,o2->child(c1)->To,min_test.T);
#endif
        bvt1.d = BV_Distance(bvt1.R,bvt1.T,
                            o1->child(bvt1.b1),o2->child(bvt1.b2));

        // init bv test 2

        bvt2.b1 = min_test.b1;
        bvt2.b2 = c2;
        MxM(bvt2.R,min_test.R,o2->child(c2)->R);
#if PQP_BV_TYPE & RSS_TYPE
        MxVpV(bvt2.T,min_test.R,o2->child(c2)->Tr,min_test.T);
#else
        MxVpV(bvt2.T,min_test.R,o2->child(c2)->To,min_test.T);
#endif
        bvt2.d = BV_Distance(bvt2.R,bvt2.T,
                            o1->child(bvt2.b1),o2->child(bvt2.b2));
      }

      // put children tests in queue

      if (bvt1.d <= res->tolerance) bvtq.AddTest(bvt1);
      if (bvt2.d <= res->tolerance) bvtq.AddTest(bvt2);
    }

    if (bvtq.Empty() || (bvtq.MinTest() > res->tolerance)) 
    {
      res->closer_than_tolerance = 0;
      return;
    }
    else 
    {
      min_test = bvtq.ExtractMinTest();
    }
  }  
}	

int
PQP_Tolerance(PQP_ToleranceResult *res,
              PQP_REAL R1[3][3], PQP_REAL T1[3], PQP_Model *o1,
              PQP_REAL R2[3][3], PQP_REAL T2[3], PQP_Model *o2,
              PQP_REAL tolerance,
              int qsize)
{
  double time1 = GetTime();

  // make sure that the models are built

  if (o1->build_state != PQP_BUILD_STATE_PROCESSED) 
    return PQP_ERR_UNPROCESSED_MODEL;
  if (o2->build_state != PQP_BUILD_STATE_PROCESSED) 
    return PQP_ERR_UNPROCESSED_MODEL;
  
  // Compute the transform [R,T] that takes us from cs2 to cs1.
  // [R,T] = [R1,T1]'[R2,T2] = [R1',-R1'T][R2,T2] = [R1'R2, R1'(T2-T1)]

  MTxM(res->R,R1,R2);
  PQP_REAL Ttemp[3];
  VmV(Ttemp, T2, T1);
  MTxV(res->T, R1, Ttemp);

  // set tolerance, used to prune the search

  if (tolerance < 0.0) tolerance = 0.0;
  res->tolerance = tolerance;
  
  // clear the stats

  res->num_bv_tests = 0;
  res->num_tri_tests = 0;

  // initially assume not closer than tolerance

  res->closer_than_tolerance = 0;
  
  // compute the transform from o1->child(0) to o2->child(0)

  PQP_REAL Rtemp[3][3], R[3][3], T[3];

  MxM(Rtemp,res->R,o2->child(0)->R);
  MTxM(R,o1->child(0)->R,Rtemp);
#if PQP_BV_TYPE & RSS_TYPE
  MxVpV(Ttemp,res->R,o2->child(0)->Tr,res->T);
  VmV(Ttemp,Ttemp,o1->child(0)->Tr);
#else
  MxVpV(Ttemp,res->R,o2->child(0)->To,res->T);
  VmV(Ttemp,Ttemp,o1->child(0)->To);
#endif
  MTxV(T,o1->child(0)->R,Ttemp);

  // find a distance lower bound for trivial reject

  PQP_REAL d = BV_Distance(R, T, o1->child(0), o2->child(0));
  
  if (d <= res->tolerance)
  {
    // more work needed - choose routine according to queue size

    if (qsize <= 2) 
    {
      ToleranceRecurse(res, R, T, o1, 0, o2, 0);
    }
    else 
    {
      res->qsize = qsize;
      ToleranceQueueRecurse(res, R, T, o1, 0, o2, 0);
    }
  }

  // res->p2 is in cs 1 ; transform it to cs 2

  PQP_REAL u[3];
  VmV(u, res->p2, res->T);
  MTxV(res->p2, res->R, u);

  double time2 = GetTime();
  res->query_time_secs = time2 - time1;

  return PQP_OK;
}

#endif
