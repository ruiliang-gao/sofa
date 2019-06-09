#ifndef OBBTREE_COLLISIONMODEL_INL
#define OBBTREE_COLLISIONMODEL_INL

#include "ObbTree.h"

using namespace sofa::component::collision;
using namespace sofa;

NodeStack::NodeStack(unsigned int n): nodes(0), num_alloced(0), num_nodes(0)
{
    Reserve(n);
}

NodeStack::~NodeStack()
{
    delete[] nodes;
}

void NodeStack::Push(const NodePair& pair)
{
    if (num_nodes >= num_alloced)
    {
        Reserve(2 * num_nodes + 1);
    }
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << " Push(" << pair.node1 << " - " << pair.node2 << ", type " << pair.type << ")" << std::endl;
#endif
    nodes[num_nodes++] = pair;

    return;
}

const NodePair NodeStack::Pop()
{
    assert(num_nodes > 0);
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << " Pop(" << nodes[num_nodes-1].node1 << "," << nodes[num_nodes-1].node2 << ", type = " << nodes[num_nodes-1].type << ")" << std::endl;
#endif
    return nodes[--num_nodes];
}

void NodeStack::Pop(unsigned int n)
{
    assert(n <= num_nodes);
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << " Pop(" << n << ")" << std::endl;
#endif
    num_nodes -= n;

    return;
}

int NodeStack::IsEmpty() const
{
  return num_nodes == 0;
}

inline void NodeStack::Clear()
{
  num_nodes = 0;
  return;
}

unsigned int NodeStack::Size() const
{
  return num_nodes;
}

void NodeStack::Reserve(unsigned int n)
{
  if (n > num_alloced)
  {
    NodePair* const tmp = new NodePair[n];

    if (nodes != 0) {
      memcpy(tmp, nodes, sizeof(NodePair) * num_nodes);

      delete[] nodes;
    }

    nodes       = tmp;
    num_alloced = n;
  }

  return;
}

CollideResult::CollideResult(unsigned int stack_size):
num_bv_tests(0),
num_tri_tests(0),
num_tri_box_tests(0),
num_pairs_alloced(0),
num_pairs(0),
pairs(0),
num_obb_pairs(0),
num_obb_pairs_alloced(0),
obb_pairs(0),
stack(0)
{
  stack = new NodeStack(stack_size);
  Reserve(10);
  ReserveOBB(10);
}

CollideResult::~CollideResult()
{
  delete   stack;
  delete[] pairs;
  delete[] obb_pairs;
}

void CollideResult::Reserve(unsigned int n)
{
    if (n > num_pairs_alloced)
    {
        CollisionPair* const tmp = new CollisionPair[n];

        if (pairs != 0)
        {
            memcpy(tmp, pairs, sizeof(CollisionPair) * num_pairs);

            delete[] pairs;
        }

        pairs = tmp;
        num_pairs_alloced = n;
    }

    return;
}

void CollideResult::Add(int i1, int i2)
{
    if (num_pairs >= num_pairs_alloced)
    {
        Reserve(2 * num_pairs + 8);
    }

    pairs[num_pairs].id1 = i1;
    pairs[num_pairs].id2 = i2;

    ++num_pairs;

    return;
}

void CollideResult::ReserveOBB(unsigned int n)
{
    if (n > num_obb_pairs_alloced)
    {
        CollisionPair* const tmp = new CollisionPair[n];

        if (obb_pairs != 0)
        {
            memcpy(tmp, obb_pairs, sizeof(CollisionPair) * num_obb_pairs);

            delete[] obb_pairs;
        }

        obb_pairs = tmp;
        num_obb_pairs_alloced = n;
    }

    return;
}

void CollideResult::AddOBB(int i1, int i2)
{
    if (num_obb_pairs >= num_obb_pairs_alloced)
    {
        ReserveOBB(2 * num_obb_pairs + 8);
    }

    obb_pairs[num_obb_pairs].id1 = i1;
    obb_pairs[num_obb_pairs].id2 = i2;

    ++num_obb_pairs;

    return;
}

void
CollideResult::Clear()
{
    num_bv_tests      = 0;
    num_tri_tests     = 0;
    num_tri_box_tests = 0;
    num_pairs         = 0;

    num_obb_pairs = 0;

    return;
}

ObbVolume::ObbVolume(const NodeType& nodeType): m_nodeType(nodeType), m_first_child(0), m_second_child(0), m_child_range_min(0), m_child_range_max(0)
{
    m_identifier = "ObbVolume";
    t_rel_top.identity();
}

ObbVolume::ObbVolume(const ObbVolume& other)
{
    this->m_position = other.m_position;
    this->m_halfExtents = other.m_halfExtents;
    this->m_localAxes = other.m_localAxes;
    this->m_first_child = other.m_first_child;
    this->m_second_child = other.m_second_child;
    this->m_child_range_min = other.m_child_range_min;
    this->m_child_range_max = other.m_child_range_max;
    this->m_childrenSet[0] = other.m_childrenSet[0];
    this->m_childrenSet[1] = other.m_childrenSet[1];
    this->m_childIsLeaf[0] = other.m_childIsLeaf[0];
    this->m_childIsLeaf[1] = other.m_childIsLeaf[1];

    this->t_rel_top = other.t_rel_top;

    this->m_identifier = std::string(other.m_identifier);

    // std::cout << "ObbVolume::ObbVolume(const ObbVolume&): from " << other.m_identifier << " to " <<  m_identifier << "; child0 = " << m_first_child << ", child1 = " << m_second_child << std::endl;
}

ObbVolume& ObbVolume::operator=(const ObbVolume& other)
{
    if (this != &other)
    {
        this->m_position = other.m_position;
        this->m_halfExtents = other.m_halfExtents;
        this->m_localAxes = other.m_localAxes;
        this->m_first_child = other.m_first_child;
        this->m_second_child = other.m_second_child;
        this->m_child_range_min = other.m_child_range_min;
        this->m_child_range_max = other.m_child_range_max;
        this->m_childrenSet[0] = other.m_childrenSet[0];
        this->m_childrenSet[1] = other.m_childrenSet[1];
        this->m_childIsLeaf[0] = other.m_childIsLeaf[0];
        this->m_childIsLeaf[1] = other.m_childIsLeaf[1];

        this->t_rel_top = other.t_rel_top;

        this->m_identifier = std::string(other.m_identifier);
        // std::cout << "ObbVolume::operator=(const ObbVolume&): from " << other.m_identifier << " to " <<  m_identifier << "; child0 = " << m_first_child << ", child1 = " << m_second_child << std::endl;
    }
    return *this;
}

void INV_ROTATE_VEC(Vector3& vr, const Matrix4& r, const Vector3& v)
{
  (vr)[0] = ((r)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second]*(v)[0]) +
            ((r)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second]*(v)[1]) +
            ((r)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second]*(v)[2]);
  (vr)[1] = ((r)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second]*(v)[0]) +
            ((r)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second]*(v)[1]) +
            ((r)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second]*(v)[2]);
  (vr)[2] = ((r)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second]*(v)[0]) +
            ((r)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second]*(v)[1]) +
            ((r)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second]*(v)[2]);
}

void TEST_VERTEX(Vector3 point, unsigned int extremal_verts[], unsigned int vrtx, double& minx, double& miny, double& minz, double& maxx, double& maxy, double& maxz)
{
    #define MIN_X 0
    #define MIN_Y 1
    #define MIN_Z 2
    #define MAX_X 3
    #define MAX_Y 4
    #define MAX_Z 5

    if (point[0] < minx)
    {
        minx                  = point[0];
        extremal_verts[MIN_X] = vrtx;
    }
    if (point[0] > maxx)
    {
        maxx                  = point[0];
        extremal_verts[MAX_X] = vrtx;
    }
    if (point[1] < miny)
    {
        miny                  = point[1];
        extremal_verts[MIN_Y] = vrtx;
    }
    if (point[1] > maxy)
    {
        maxy                  = point[1];
        extremal_verts[MAX_Y] = vrtx;
    }
    if (point[2] < minz)
    {
        minz                  = point[2];
        extremal_verts[MIN_Z] = vrtx;
    }
    if (point[2] > maxz)
    {
        maxz                  = point[2];
        extremal_verts[MAX_Z] = vrtx;
    }
}

void
most_dist_points_on_aabb(const sofa::core::topology::BaseMeshTopology *trim,
                         const sofa::core::behavior::MechanicalState<Vec3Types> * mState,
                         const unsigned int      verts[6],
                         unsigned int&           v1,
                         unsigned int&           v2)
{
    unsigned int minx = 0, maxx = 0, miny = 0, maxy = 0, minz = 0, maxz = 0;

    Vector3 v[6];

    typename core::behavior::MechanicalState<Vec3Types>::ReadVecCoord pos = mState->readPositions();

    for (unsigned int i = 0; i < 6; ++i)
    {
        v[i] = pos[verts[i]];
    }

    for (unsigned int i = 1; i < 6; ++i)
    {
        if (v[i][0] < v[minx][0]) { minx = i; }
        if (v[i][0] > v[maxx][0]) { maxx = i; }

        if (v[i][1] < v[miny][1]) { miny = i; }
        if (v[i][1] > v[maxy][1]) { maxy = i; }

        if (v[i][2] < v[minz][2]) { minz = i; }
        if (v[i][2] > v[maxz][2]) { maxz = i; }
    }

    Vector3 diff;

    diff = v[maxx] - v[minx];
    const double dist2x = diff * diff;

    diff = v[maxy] - v[miny];
    const double dist2y = diff * diff;

    diff = v[maxz] - v[minz];
    const double dist2z = diff * diff;

    v1 = minx;
    v2 = maxx;

    if (dist2y > dist2x && dist2y > dist2z)
    {
        v1 = miny;
        v2 = maxy;
    }

    if (dist2z > dist2x && dist2z > dist2y)
    {
        v1 = minz;
        v2 = maxz;
    }

    v1 = verts[v1];
    v2 = verts[v2];

    return;
}

void ObbVolume::FitToTriangles(core::topology::BaseMeshTopology *trim, const sofa::core::behavior::MechanicalState<Vec3Types> * mState,
                               const TriIndxVec &indx_vec, Vector3 &diam_vec)
{
    unsigned int extremal_verts[6] = {0, };

    typename core::behavior::MechanicalState<Vec3Types>::ReadVecCoord pos = mState->readPositions();

    unsigned int num_tris = indx_vec.size();
    unsigned int num_verts = pos.size();
    std::vector<bool> used_verts(num_verts, false);
    sofa::core::topology::BaseMeshTopology::Triangle tri;

    // project points of tris to local coordinates and find the extreme
    // values
    double minx, miny, minz;
    double maxx, maxy, maxz;
    Vector3 point; // transformed
    Vector3 v;     // original

    minx = miny = minz =  DBL_MAX;
    maxx = maxy = maxz = -DBL_MAX;

    for (unsigned int i = 0; i < num_tris; ++i)
    {
        //tri = trim.GetIndexedTriangle(indx_vec[i]);
        tri = trim->getTriangle(indx_vec[i]);

        if (!used_verts[tri[0]])
        {
            used_verts[tri[0]] = true;
            v = pos[tri[0]];
            //trim.GetVertex(tri.p1, v);

            INV_ROTATE_VEC(point, t_rel_top, v);

            TEST_VERTEX(point, extremal_verts, tri[0], minx, miny, minz, maxx, maxy, maxz);
        }

        if (!used_verts[tri[1]])
        {
            used_verts[tri[1]] = true;
            v = pos[tri[1]];

            INV_ROTATE_VEC(point, t_rel_top, v);

            TEST_VERTEX(point, extremal_verts, tri[1], minx, miny, minz, maxx, maxy, maxz);
        }

        if (!used_verts[tri[2]])
        {
            used_verts[tri[2]] = true;
            v = pos[tri[2]];

            INV_ROTATE_VEC(point, t_rel_top, v);

            TEST_VERTEX(point, extremal_verts, tri[2], minx, miny, minz, maxx, maxy, maxz);
        }
    }

    const Vector3 center(0.5f * (minx + maxx),
                        0.5f * (miny + maxy),
                        0.5f * (minz + maxz));

    Vector3 transl;
    /*(t_rel_top[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second],
    t_rel_top[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second],
    t_rel_top[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second]);*/

    //ROTATE_VEC(transl, t_rel_top, center);
    transl = t_rel_top.transform(center);
    m_position = transl;

    std::cout << " position vec = " << m_position << " (from center = " << center << ")" << std::endl;

    m_halfExtents[0] = 0.5f * (maxx - minx);
    m_halfExtents[1] = 0.5f * (maxy - miny);
    m_halfExtents[2] = 0.5f * (maxz - minz);

    //if (diam_vec != 0)
    {
        if (num_verts < 2)
        {
            diam_vec = Vector3(0.0f, 0.0f, 0.0f);
        }
        else
        {
            unsigned int v1 = 0;
            unsigned int v2 = 0;

            most_dist_points_on_aabb(trim, mState, extremal_verts, v1, v2);

            /*trim.GetVertex(v1, p1);
            trim.GetVertex(v2, p2);*/
            Vector3 p1 = pos[v1];
            Vector3 p2 = pos[v2];

            diam_vec = p2 - p1;
        }
    }

    return;
}


void SWAP(double& a, double& b)
{
    const double tmp = (a);
    (a)            = (b);
    (b)            = tmp;
}

void SWAP_COLUMNS(Matrix4& t, int c1, int c2)
{
    SWAP(t[0][c1], t[0][c2]);
    SWAP(t[1][c1], t[1][c2]);
    SWAP(t[2][c1], t[2][c2]);
}
void ObbVolume::SortDimensions()
{
    // make sure the smallest dimension is in the z-direction
    if (m_halfExtents[2] > m_halfExtents[0])
    {
        SWAP(m_halfExtents[2], m_halfExtents[0]);
        SWAP_COLUMNS(t_rel_top, 2, 0);

        t_rel_top[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *= -1.0f;
        t_rel_top[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *= -1.0f;
        t_rel_top[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *= -1.0f;
    }

    if (m_halfExtents[2] > m_halfExtents[1])
    {
        SWAP(m_halfExtents[2], m_halfExtents[1]);
        SWAP_COLUMNS(t_rel_top, 2, 1);

        t_rel_top[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *= -1.0f;
        t_rel_top[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *= -1.0f;
        t_rel_top[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *= -1.0f;
    }

    // make sure x-dimension is largest
    if (m_halfExtents[0] < m_halfExtents[1])
    {
        SWAP(m_halfExtents[0], m_halfExtents[1]);
        SWAP_COLUMNS(t_rel_top, 0, 1);

        t_rel_top[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] *= -1.0f;
        t_rel_top[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *= -1.0f;
        t_rel_top[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] *= -1.0f;
    }

    return;
}

ObbTree::ObbTree(const ObbTree& other): ObbVolume(other)
{
    this->m_name = other.m_name;
    this->m_worldPos = other.m_worldPos;
    this->m_worldRot = other.m_worldRot;

    m_minDrawDepth = other.m_minDrawDepth;
    m_maxDrawDepth = other.m_maxDrawDepth;
    m_topology = other.m_topology;
    m_state = other.m_state;

    m_drawOverlappingOnly = other.m_drawOverlappingOnly;
    m_drawTriangleTestsOnly = other.m_drawTriangleTestsOnly;

    m_obbNodes.clear();

    for (std::vector<ObbVolume>::const_iterator it = other.m_obbNodes.begin(); it != other.m_obbNodes.end(); it++)
    {
        m_obbNodes.push_back(*it);
    }
}

ObbTree& ObbTree::operator=(const ObbTree& other)
{
    if (this != &other)
    {
        m_obbNodes.clear();

        this->m_name = other.m_name;
        this->m_worldPos = other.m_worldPos;
        this->m_worldRot = other.m_worldRot;

        m_minDrawDepth = other.m_minDrawDepth;
        m_maxDrawDepth = other.m_maxDrawDepth;
        m_topology = other.m_topology;
        m_state = other.m_state;

        m_drawOverlappingOnly = other.m_drawOverlappingOnly;
        m_drawTriangleTestsOnly = other.m_drawTriangleTestsOnly;

        for (std::vector<ObbVolume>::const_iterator it = other.m_obbNodes.begin(); it != other.m_obbNodes.end(); it++)
        {
            m_obbNodes.push_back(*it);
        }
    }
    return *this;
}

#include "BVHDrawHelpers.h"

void ObbTree::assignOBBNodeColors()
{
    m_obbNodeColors.clear();
    unsigned int obbIdx = 0;
    for (std::vector<ObbVolume>::const_iterator it = m_obbNodes.begin(); it != m_obbNodes.end(); it++)
    {
        m_obbNodeColors.insert(std::make_pair(obbIdx, BVHDrawHelpers::randomVec4()));
        obbIdx++;
    }
}

void Moments::clear_moments(Moment& m)
{
    m.area = 0.0;

    m.mean[0] = 0.0;
    m.mean[1] = 0.0;
    m.mean[2] = 0.0;

    m.cov[0][0] = 0.0; m.cov[0][1] = 0.0; m.cov[0][2] = 0.0;
    m.cov[1][0] = 0.0; m.cov[1][1] = 0.0; m.cov[1][2] = 0.0;
    m.cov[2][0] = 0.0; m.cov[2][1] = 0.0; m.cov[2][2] = 0.0;
}

//============================================================================

void Moments::accum_moments(Moment& a, const Moment& b)
{
    std::cout << " accum_moments: a.mean 1 = " << a.mean << ", b.mean = " << b.mean << ", b.area = " << b.area << std::endl;

    a.mean[0] += b.mean[0] * b.area;
    a.mean[1] += b.mean[1] * b.area;
    a.mean[2] += b.mean[2] * b.area;

    std::cout << " accum_moments: a.mean 2 = " << a.mean << std::endl;

    std::cout << " a.cov 1 = " << a.cov << ", b.cov = " << b.cov << std::endl;
    a.cov[0][0] += b.cov[0][0];
    a.cov[1][0] += b.cov[1][0];
    a.cov[2][0] += b.cov[2][0];
    a.cov[0][1] += b.cov[0][1];
    a.cov[1][1] += b.cov[1][1];
    a.cov[2][1] += b.cov[2][1];
    a.cov[0][2] += b.cov[0][2];
    a.cov[1][2] += b.cov[1][2];
    a.cov[2][2] += b.cov[2][2];

//    a.cov[0][0] += b.cov[0][0];
//    a.cov[0][1] += b.cov[0][1];
//    a.cov[0][2] += b.cov[0][2];
//    a.cov[1][0] += b.cov[1][0];
//    a.cov[1][1] += b.cov[1][1];
//    a.cov[1][2] += b.cov[1][2];
//    a.cov[2][0] += b.cov[2][0];
//    a.cov[2][1] += b.cov[2][1];
//    a.cov[2][2] += b.cov[2][2];

    std::cout << " a.cov 2 = " << a.cov << std::endl;

    a.area += b.area;
}

void Moments::compute_moments(const Vector3& p, const Vector3& q, const Vector3& r, Moment& m)
{
    Vector3 u, v, nrml;

    // compute the area of the triangle
    u = q - p;
    v = r - p;

    nrml = u.cross(v);

    m.area = 0.5 * sqrt(nrml * nrml);

    // compute the mean
    m.mean[0] = (p[0] + q[0] + r[0]) / 3.0;
    m.mean[1] = (p[1] + q[1] + r[1]) / 3.0;
    m.mean[2] = (p[2] + q[2] + r[2]) / 3.0;

    if (m.area == 0.0)
    {
        // second-order components in case of zero area
        m.cov[0][0] = p[0]*p[0] + q[0]*q[0] + r[0]*r[0];
        m.cov[1][0] = p[0]*p[1] + q[0]*q[1] + r[0]*r[1];
        m.cov[2][0] = p[0]*p[2] + q[0]*q[2] + r[0]*r[2];
        m.cov[1][1] = p[1]*p[1] + q[1]*q[1] + r[1]*r[1];
        m.cov[2][1] = p[1]*p[2] + q[1]*q[2] + r[1]*r[2];
        m.cov[2][2] = p[2]*p[2] + q[2]*q[2] + r[2]*r[2];
    }
    else
    {
        // get the second-order components -- note the weighting by the area
        m.cov[0][0] = m.area * ((9.0*m.mean[0] * m.mean[0]) + p[0]*p[0] + q[0]*q[0] + r[0]*r[0]) / 12.0;
        m.cov[1][0] = m.area * ((9.0*m.mean[0] * m.mean[1]) + p[0]*p[1] + q[0]*q[1] + r[0]*r[1]) / 12.0;
        m.cov[1][1] = m.area * ((9.0*m.mean[1] * m.mean[1]) + p[1]*p[1] + q[1]*q[1] + r[1]*r[1]) / 12.0;
        m.cov[2][0] = m.area * ((9.0*m.mean[0] * m.mean[2]) + p[0]*p[2] + q[0]*q[2] + r[0]*r[2]) / 12.0;
        m.cov[2][1] = m.area * ((9.0*m.mean[1] * m.mean[2]) + p[1]*p[2] + q[1]*q[2] + r[1]*r[2]) / 12.0;
        m.cov[2][2] = m.area * ((9.0*m.mean[2] * m.mean[2]) + p[2]*p[2] + q[2]*q[2] + r[2]*r[2]) / 12.0;
    }

  // make sure the covariance matrix is symmetric
  m.cov[1][2] = m.cov[2][1];
  m.cov[0][1] = m.cov[1][0];
  m.cov[0][2] = m.cov[2][0];

  return;
}

//============================================================================


#define ROTATE(a,i,j,k,l)             \
  g       = a[j][i];                  \
  h       = a[l][k];                  \
  a[j][i] = g - (s * (h + g * tau));  \
  a[l][k] = h + (s * (g - h * tau))
#define MAX_NUM_ROT 50

void Moments::eigen_3x3(Matrix3& vout, Vector3& dout, Matrix3& a)
{
    int n = 3;
    int j, iq, ip, i;
    double tresh, theta, tau, t, sm, s, h, g, c;
    Vector3 b(a[0][0], a[1][1], a[2][2]);
    Vector3 d(a[0][0], a[1][1], a[2][2]);
    Vector3 z;

    Matrix3 v; v.identity();

    for (i = 0; i < MAX_NUM_ROT; i++)
    {
        // sum the off-diagonal components
        sm = 0.0f;
        for (ip = 0; ip < n; ip++)
        {
            for (iq = ip + 1; iq < n; iq++)
            {
                sm += fabs(a[iq][ip]);
            }
        }

        // std::cout << " - " << i << ": sm = " << sm << ", matrix a = " << a << ", matrix v = " << v << ", vector d = " << d << ", vector z = " << z << std::endl;

        if (sm == 0.0f)
        {
            vout[0][0] = v[0][0]; vout[1][0] = v[1][0]; vout[2][0] = v[2][0];
            vout[0][1] = v[0][1]; vout[1][1] = v[1][1]; vout[2][1] = v[2][1];
            vout[0][2] = v[0][2]; vout[1][2] = v[1][2]; vout[2][2] = v[2][2];

            dout = d;
            return;
        }

        // special treshold the first three sweeps
        tresh = (i < 3)? 0.2f * sm / (n*n) : 0.0f;

        //std::cout << " tresh = " << tresh << std::endl;
        for (ip = 0; ip < n; ip++)
        {
            for (iq = ip + 1; iq < n; iq++)
            {
                g = 100.0f * fabs(a[iq][ip]);

                //std::cout << "   - g = " << g << ", fabs(d[" << ip << "]) = " << fabs(d[ip]) << " + " << g << " = " << fabs(d[ip]) + g << std::endl;
                //std::cout << "                     fabs(d[" << iq << "]) = " << fabs(d[iq]) << " + " << g << " = " << fabs(d[iq]) + g << std::endl;
                if (i > 3                          &&
                        fabs(d[ip]) + g == fabs(d[ip]) &&
                        fabs(d[iq]) + g == fabs(d[iq]))
                {
                    a[iq][ip] = 0.0f;
                }
                else if (fabs(a[iq][ip]) > tresh)
                {
                    h = d[iq] - d[ip];

                    if (fabs(h) + g == fabs(h))
                    {
                        t = a[iq][ip] / h; // t = 1 / (2 * theta)
                    }
                    else
                    {
                        theta = 0.5f * h / a[iq][ip];
                        t     = 1.0f / (fabs(theta) + sqrt(1.0f + theta*theta));
                        if (theta < 0.0f) { t = -t; }
                    }

                    c         = 1.0f / sqrt(1.0f + t * t);
                    s         = t * c;
                    tau       = s / (1.0f + c);
                    h         = t * a[iq][ip];
                    z[ip]    -= h;
                    z[iq]    += h;
                    d[ip]    -= h;
                    d[iq]    += h;

                    // std::cout <<  "   setze a[" << ip << "][" << iq << "] = 0" << std::endl;
                    a[ip][iq] = 0.0f;

                    // cyclic jacobi method
                    for (j = 0; j < ip; j++)
                    {
                        //ROTATE(a,j,ip,j,iq);
                        g       = a[j][ip];
                        h       = a[j][iq];
                        a[j][ip] = g - (s * (h + g * tau));
                        a[j][iq] = h + (s * (g - h * tau));
                    } // rotations j in [0, ip)
                    for (j = ip + 1; j < iq; j++)
                    {
                        //ROTATE(a,ip,j,j,iq);
                        g       = a[ip][j];
                        h       = a[j][iq];
                        a[ip][j] = g - (s * (h + g * tau));
                        a[j][iq] = h + (s * (g - h * tau));
                    } // rotations j in (ip, iq)
                    for (j = iq + 1; j < n; j++)
                    {
                        //ROTATE(a,ip,j,iq,j);
                        g       = a[ip][j];
                        h       = a[iq][j];
                        a[ip][j] = g - (s * (h + g * tau));
                        a[iq][j] = h + (s * (g - h * tau));
                    } // rotations j in (q, n)
                    for (j = 0; j < n; j++)
                    {
                        //ROTATE(v,j,ip,j,iq);
                        g       = v[j][ip];
                        h       = v[j][iq];
                        v[j][ip] = g - (s * (h + g * tau));
                        v[j][iq] = h + (s * (g - h * tau));
                    }

                    //std::cout << "   SCHEISS FOTZE a 2 = " << a << std::endl;
                    //std::cout << "   SCHEISS FOTZE v 2 = " << v << std::endl;
                }
            }


            b = b + z;
            d = b;
            z = Vector3(0,0,0);
        }
    }

    std::cout << "eigen: too many iterations in Jacobi transform: " << MAX_NUM_ROT << std::endl;

    vout[0][0] = v[0][0]; vout[1][0] = v[1][0]; vout[2][0] = v[2][0];
    vout[0][1] = v[0][1]; vout[1][1] = v[1][1]; vout[2][1] = v[2][1];
    vout[0][2] = v[0][2]; vout[1][2] = v[1][2]; vout[2][2] = v[2][2];

    dout = d;

    return;
}

void ObbTreeBuilder::compute_moments(sofa::core::topology::BaseMeshTopology* topology, sofa::core::behavior::MechanicalState<Vec3Types>* mState, const TriIndxVec& indx_vec, Moments::Moment& m_tot)
{
    typename core::behavior::MechanicalState<Vec3Types>::ReadVecCoord pos = mState->readPositions();
    // first collect all the moments, and obtain the area of the
    // smallest nonzero area triangle.

    if (indx_vec.size() == 0)
        return;

    double a_min          = -1.0;
    bool degenerate_found = false;
    std::vector<Moments::Moment> m_tris;
    m_tris.resize(indx_vec.size());
    Vector3 tri_verts[3];

    for (unsigned int i = 0; i < indx_vec.size(); ++i)
    {
        sofa::core::topology::BaseMeshTopology::Triangle tri = topology->getTriangle(indx_vec[i]);
        tri_verts[0] = pos[tri[0]];
        tri_verts[1] = pos[tri[1]];
        tri_verts[2] = pos[tri[2]];

        Moments::compute_moments(tri_verts[0], tri_verts[1], tri_verts[2], m_tris[i]);

        if (m_tris[i].area <= 0.0)
        {
            degenerate_found = true;
        }
        else
        {
            if (a_min <= 0.0)
            {
                a_min = m_tris[i].area;
            }
            else if (m_tris[i].area < a_min)
            {
                a_min = m_tris[i].area;
            }
        }
    }

    if (degenerate_found)
    {
        fprintf(stderr, "----\n");
        fprintf(stderr, "Warning! Some triangle have zero area!\n");
        fprintf(stderr, "----\n");

        // if there are any zero-area triangles, go back and set their area

        // if ALL the triangles have zero area, then set the area to 1.0
        if (a_min <= 0.0) { a_min = 1.0; }

        for (unsigned int i = 0; i < indx_vec.size(); ++i)
        {
            if (m_tris[i].area <= 0.0f)
            {
                m_tris[i].area = a_min;
            }
        }
    }

    clear_moments(m_tot);

    // now compute the moments for all triangles together
    for (unsigned int i = 0; i < indx_vec.size(); ++i)
    {
        accum_moments(m_tot, m_tris[i]);
    }

    // get correct mean by dividing with total area
    const double a_inv = 1.0 / m_tot.area;
    m_tot.mean[0] *= a_inv;
    m_tot.mean[1] *= a_inv;
    m_tot.mean[2] *= a_inv;

    // compute (scaled) covariance matrix
    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            m_tot.cov[i][j] = m_tot.cov[i][j] - (m_tot.mean[i] * m_tot.mean[j]) * m_tot.area;
        }
    }

    printf("accum mean: %f %f %f\n", m_tot.mean[0], m_tot.mean[1], m_tot.mean[2]);
    return;
}


bool ObbTreeBuilder::buildTree()
{
    std::cout << "ObbTreeBuilder::buildTree(" << mObbTree->getName() << ")" << std::endl;
    if (!mTopology || !mState)
        return false;

    typename core::behavior::MechanicalState<Vec3Types>::ReadVecCoord pos = mState->readPositions();

    std::cout << " triangles: " << mTopology->getNbTriangles() << ", vertices: " << pos.size() << std::endl;

    if (mTopology->getNbTriangles() == 0 || pos.size() == 0)
        return false;

    ObbVolume left_obb;
    ObbVolume right_obb;

    BuildStack build_stack;

    BuildNode parent;
    BuildNode left_child;
    BuildNode right_child;

    int num_tris = mTopology->getNbTriangles();
    //m_triNodes.resize(num_tris);

    parent.obb_indx = 0;

    for (unsigned int i = 0; i < num_tris; ++i)
    {
        parent.tris.push_back(i);
    }

    mObbTree->setIdentifier(mObbTree->getName() + " Root node");
    m_obbNodes.push_back(*mObbTree);

    compute_obb(m_obbNodes.at(0), parent, mTopology, mState, 0);

    build_stack.push(parent);

    std::map<std::string, std::pair<int,int> > obbNodeChildIndices;

    int numIterations = 0;
    std::stringstream obbIdStream;

    while (!build_stack.empty())
    {
        parent = build_stack.top();
        build_stack.pop();

        ObbVolume& parent_obb = m_obbNodes[parent.obb_indx];

        std::cout << "=== iteration " << numIterations << ": OBB index " << parent.obb_indx << ": " << parent_obb.identifier() << " ===" << std::endl;

        if (parent.tris.size() < 3)
        {
            std::cout << " parent " << parent.obb_indx << " has " << parent.tris.size() << " indices; creating TriNode" << std::endl;
            TriNode trinode;
            m_triNodes.push_back(trinode);
            const int trinode_indx = m_triNodes.size() - 1;
            parent_obb.setFirstChild(-(trinode_indx + 1), TRIANGLE_PAIR_CHILD_NODE);

            if (parent.tris.size() == 0)
            {
                m_triNodes.at(trinode_indx).setFirstChild(-1);
                m_triNodes.at(trinode_indx).setSecondChild(-1);
                std::cout << "TriNode << " << trinode_indx << " got zero triangles, should never happen!" << std::endl;
            }
            else if (parent.tris.size() == 1)
            {
                m_triNodes.at(trinode_indx).setFirstChild(parent.tris[0]);
                m_triNodes.at(trinode_indx).setSecondChild(-1);
                std::cout << " TriNode with 1 triangle: " << parent.tris[0] << std::endl;

                std::cout << " triNode " << trinode_indx << ": child0 = " << m_triNodes.at(trinode_indx).getFirstChild() <<
                             ", child1 = " << m_triNodes.at(trinode_indx).getSecondChild() << std::endl;
            }
            else
            {
                m_triNodes.at(trinode_indx).setFirstChild(parent.tris[0]);
                m_triNodes.at(trinode_indx).setSecondChild(parent.tris[1]);

                std::cout << " TriNode with 2 triangles: " << parent.tris[0] << " and " << parent.tris[1] << std::endl;

                std::cout << " triNode " << trinode_indx << ": child0 = " << m_triNodes.at(trinode_indx).getFirstChild() <<
                             ", child1 = " << m_triNodes.at(trinode_indx).getSecondChild() << std::endl;

                // swap triangles?
                /*if (swap_tris) {
                    if (tri_area_sqrd(*m.tri_mesh, parent.tris[0]) < tri_area_sqrd(*m.tri_mesh, parent.tris[1])) {
                        if (!smallest_tri_first) {
                            std::swap(m.tri_nodes[trinode_indx].tri1, m.tri_nodes[trinode_indx].tri2);
                        }
                    } else if (smallest_tri_first) {
                        std::swap(m.tri_nodes[trinode_indx].tri1, m.tri_nodes[trinode_indx].tri2);
                    }
                }*/
            }
        }
        else if (parent.tris.size() > 1)
        {
            std::cout << "  split OBB " << parent.obb_indx << " with " << parent.tris.size() << " triangles." << std::endl;
            partition_triangles(parent, mTopology, mState, left_child.tris, right_child.tris);

            compute_obb(left_obb, left_child, mTopology, mState, 0);
            compute_obb(right_obb, right_child, mTopology, mState, 0);
            //left_obb.fitObb(mTopology, mState, left_child);
            //right_obb.fitObb(mTopology, mState, right_child);


            //std::cout << " left child obb : " << left_obb.getPosition() << ", half-extents: " << left_obb.getHalfExtents() << ", local axes: " << left_obb.getOrientation() << ", children: " << left_obb.getFirstChild() << " + " << left_obb.getSecondChild() << std::endl;
            //std::cout << " right child obb: " << right_obb.getPosition() << ", half-extents: " << right_obb.getHalfExtents() << ", local axes: " << right_obb.getOrientation() << ", children: " << right_obb.getFirstChild() << " + " << right_obb.getSecondChild() << std::endl;

            obbIdStream << mObbTree->getName() << " OBB inner node " << m_obbNodes.size() /* + 1*/;
            left_obb.setIdentifier(obbIdStream.str());
            obbIdStream.str("");
            obbIdStream << mObbTree->getName() << " OBB inner node " << m_obbNodes.size() + 1 /*+ 2*/;
            right_obb.setIdentifier(obbIdStream.str());
            obbIdStream.str("");

            m_obbNodes.push_back(left_obb);
            m_obbNodes.push_back(right_obb);

            parent_obb.setFirstChild(m_obbNodes.size() - 2, INNER_CHILD_NODE);
            parent_obb.setSecondChild(m_obbNodes.size() - 1, INNER_CHILD_NODE);

            if (obbNodeChildIndices.find(parent_obb.identifier()) == obbNodeChildIndices.end())
            {
                obbNodeChildIndices.insert(std::make_pair(parent_obb.identifier(), std::make_pair(m_obbNodes.size() - 2, m_obbNodes.size() - 1)));
            }

            left_child.obb_indx    = m_obbNodes.size() - 2;
            right_child.obb_indx   = m_obbNodes.size() - 1;

            std::cout << " OBB node count now: " << m_obbNodes.size() << "; " << parent_obb.identifier() << " set firstChild = " << parent_obb.getFirstChild() << ", secondChild = " << parent_obb.getSecondChild() << std::endl;

            build_stack.push(right_child);
            build_stack.push(left_child);

            // ObbTree root Zuweisung
            if (parent.obb_indx == 0)
            {
                m_obbNodes[parent.obb_indx] = parent_obb;
            }
        }
        else
        {
            std::cout << "  reached OBB leaf at index " << parent.obb_indx << ": " << parent.tris.size() << " triangles; " << parent_obb.identifier() << " set child0 = " << -(parent.tris[0] + 1) << std::endl;
            parent_obb.setFirstChild(-(parent.tris[0] + 1), TRIANGLE_LEAF_CHILD_NODE);
            obbIdStream.str("");
            obbIdStream << mObbTree->getName() << " OBB leaf node " << (parent.tris[0] + 1);
            parent_obb.setIdentifier(obbIdStream.str());

            //parent_obb.setSecondChild(-(parent.tris[0] + 2), ObbVolume::LEAF_NODE);
        }

        /*std::cout << " parent obb: " << parent_obb.getPosition() << ", half-extents: " << parent_obb.getHalfExtents() << ", local axes: " << parent_obb.getOrientation() << ", children: " << parent_obb.getFirstChild() << " + " << parent_obb.getSecondChild() << std::endl;

        int t = 0;
        std::cout << " ObbVolume count now: " << m_obbNodes.size() << std::endl;
        for (std::vector<ObbVolume>::iterator it = m_obbNodes.begin(); it != m_obbNodes.end(); it++)
        {
            ObbVolume& obb = *it;
            std::cout << "   * ObbVolume " << t++ << " -- " << obb.identifier() << " position: " << obb.getPosition() << ", half-extents: " << obb.getHalfExtents() << ", local axes: " << obb.getOrientation() << ", children: " << obb.getFirstChild() << " + " << obb.getSecondChild() << std::endl;
        }*/
        numIterations++;
   }

    std::cout << "OBB tree: " << m_obbNodes.size() << " OBBs" << std::endl;
    std::cout << " at: " << mObbTree->getWorldPosition() << ", orientation = " << mObbTree->getWorldOrientation() << std::endl;
    for (std::vector<ObbVolume>::iterator it = m_obbNodes.begin(); it != m_obbNodes.end(); it++)
    {
        ObbVolume& obb = *it;

        mObbTree->m_obbNodes.push_back(ObbVolume(obb));

        if (it == m_obbNodes.begin())
        {
            if (m_obbNodes.size() < 2)
            {
                mObbTree->setFirstChild(obb.getFirstChild(), TRIANGLE_LEAF_CHILD_NODE);
                mObbTree->setSecondChild(obb.getSecondChild(), TRIANGLE_LEAF_CHILD_NODE);
            }
            else
            {
                mObbTree->setFirstChild(obb.getFirstChild(), INNER_CHILD_NODE);
                mObbTree->setSecondChild(obb.getSecondChild(), INNER_CHILD_NODE);
            }
            mObbTree->setIdentifier(obb.identifier());
        }
        else
        {
            if (obbNodeChildIndices.find(obb.identifier()) != obbNodeChildIndices.end())
            {
                mObbTree->m_obbNodes.back().setFirstChild(obbNodeChildIndices[obb.identifier()].first, INNER_CHILD_NODE);
                mObbTree->m_obbNodes.back().setSecondChild(obbNodeChildIndices[obb.identifier()].second, INNER_CHILD_NODE);
            }
        }
        std::cout << " * " << obb.identifier() << " t_rel_top = " << obb.t_rel_top <<  ", position: " << obb.getPosition() << ", half-extents: " << obb.getHalfExtents() << ", local axes: " << obb.getOrientation() << ", children: " << obb.getFirstChild() << " + " << obb.getSecondChild() << std::endl;
    }

    std::cout << " TriNode count: " << m_triNodes.size() << std::endl;
    for (std::vector<TriNode>::iterator it = m_triNodes.begin(); it != m_triNodes.end(); it++)
    {
        mObbTree->m_triNodes.push_back(*it);
        std::cout << " * " << (*it).getFirstChild() << " - " << (*it).getSecondChild() << std::endl;
    }

    std::cout << "==== SCHEISS WICHS FOTZE BLÖDE NUTTE DRECKSFOTZE: obb root position = " << mObbTree->m_position << std::endl;

    return true;
}


void ObbTreeBuilder::partition_triangles(const BuildNode&        parent,
                                         sofa::core::topology::BaseMeshTopology *topology,
                                         sofa::core::behavior::MechanicalState<Vec3Types> *m_state,
                                         TriIndxVec&             left_child,
                                         TriIndxVec&             right_child)
{
    Vector3 tri_verts[3];
    const TriIndxVec& tris      = parent.tris;
    const unsigned int num_tris = tris.size();

    // std::cout << "partition_triangles: " << num_tris << " triangles" << std::endl;

    // make sure each child contains zero triangles
    left_child.clear();
    right_child.clear();

    typename core::behavior::MechanicalState<Vec3Types>::ReadVecCoord pos = mState->readPositions();

    for (unsigned int i = 0; i < num_tris; ++i)
    {
        const int indx = tris[i];

        sofa::core::topology::Topology::Triangle tri = topology->getTriangle(indx);

        // std::cout << " Triangle " << indx << ": " << tri << std::endl;

        tri_verts[0] = pos[tri[0]];
        tri_verts[1] = pos[tri[1]];
        tri_verts[2] = pos[tri[2]];

        // std::cout << "  vertices: " << tri_verts[0] << ", " << tri_verts[1] << ", " << tri_verts[2] << std::endl;

        tri_verts[0] += tri_verts[1];
        tri_verts[0] += tri_verts[2];

        // project onto axis

        // std::cout << " parent split_axis = " << parent.split_axis << std::endl;

        const double x = tri_verts[0] * parent.split_axis / 3.0;

        if (x < parent.split_coord)
        {
            // std::cout << "  triangle " << indx << ": LEFT" << std::endl;
            left_child.push_back(indx);
        }
        else
        {
            // std::cout << "  triangle " << indx << ": RIGHT" << std::endl;
            right_child.push_back(indx);
        }
    }

    if ((left_child.empty() || right_child.empty()) && num_tris > 1)
    {
        // do an arbitrary partitioning
        left_child.clear();
        right_child.clear();

        const unsigned int mid = num_tris / 2;
        left_child.insert(left_child.end(),   tris.begin(), tris.begin() + mid);
        right_child.insert(right_child.end(), tris.begin() + mid, tris.end());
    }

    return;
}

void ObbTreeBuilder::compute_obb(ObbVolume&              obb,
                                 BuildNode&              bnode,
                                 sofa::core::topology::BaseMeshTopology *topology,
                                 sofa::core::behavior::MechanicalState<Vec3Types> *m_state,
                                 unsigned int            flags)
{
    std::cout << " compute_obb from " << bnode.tris.size() << " triangles." << std::endl;
    Moments::Moment tris_mom;
    Matrix3 e; // eigen vectors
    Vector3 s; // eigen values

    // compute moments for all triangles
    compute_moments(topology, m_state, bnode.tris, tris_mom);

    std::cout << " moments: area = " << tris_mom.area << ", means = " << tris_mom.mean << ", covariance = " << tris_mom.cov << std::endl;

    // compute eigen values for the covariance matrix
    Moments::eigen_3x3(e, s, tris_mom.cov);

    printf("eigen values: %f  %f  %f\n", s[0], s[1], s[2]);
    printf("eigen vectors:\n");
    printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n",
      e[0][0], e[0][1], e[0][2],
      e[1][0], e[1][1], e[1][2],
      e[2][0], e[2][1], e[2][2]);

    // sort the eigen vectors
    unsigned int min, mid, max;
    if (s[0] > s[1]) { max = 0; min = 1; }
    else             { min = 0; max = 1; }
    if (s[2] < s[min])      { mid = min; min = 2; }
    else if (s[2] > s[max]) { mid = max; max = 2; }
    else                    { mid = 2; }

    obb.t_rel_top[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] = (e[0][max]);
    obb.t_rel_top[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] = (e[1][max]);
    obb.t_rel_top[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] = (e[2][max]);

    obb.t_rel_top[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] = (e[0][mid]);
    obb.t_rel_top[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] = (e[1][mid]);
    obb.t_rel_top[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] = (e[2][mid]);

    // compute the third column as the cross-product of the first two
    obb.t_rel_top[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] = (e[1][max]*e[2][mid] - e[2][max]*e[1][mid]);
    obb.t_rel_top[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] = (e[2][max]*e[0][mid] - e[0][max]*e[2][mid]);
    obb.t_rel_top[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] = (e[0][max]*e[1][mid] - e[1][max]*e[0][mid]);

    // fit the OBB to the triangles
    // this will set its dimensions and center
    Vector3 diam_vec;
    obb.FitToTriangles(topology, m_state, bnode.tris, diam_vec);
    obb.SortDimensions();

    // m_position PASSIERT in FitToTriangles!!!
    obb.t_rel_top[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] = obb.m_position.x();
    obb.t_rel_top[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] = obb.m_position.y();
    obb.t_rel_top[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] = obb.m_position.z();

    obb.t_rel_top.getsub(0,0,obb.m_localAxes);
    // compute split-axis and split coordinate

    // Options:
    //  1) longest box dimension
    //  2) direction formed by the two most distant points
    /*if (flags & YAOBI_SPLIT_AXIS_FROM_DIAMETER) {
        const double len = sqrt(DOT_PROD(diam_vec, diam_vec));

        bnode.split_axis[0] = diam_vec[0] / len;
        bnode.split_axis[1] = diam_vec[1] / len;
        bnode.split_axis[2] = diam_vec[2] / len;
    } else */
    {
        bnode.split_axis = Vector3(obb.t_rel_top[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second],
                                   obb.t_rel_top[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second],
                                   obb.t_rel_top[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second]);
    }

    // Options:
    //  1) mean of triangle centroids
    //  2) median of triangle centroids
    //  3) box center
    /*if (flags & YAOBI_TRIS_MEDIAN_SPLIT)
    {
        printf("median split!\n");
        bnode.split_coord = compute_median(bnode.split_axis, trim, bnode.tris);
    }
    else*/
    {
        bnode.split_coord = bnode.split_axis * tris_mom.mean;
    }

    return;
}

#endif //OBBTREE_COLLISIONMODEL_INL
