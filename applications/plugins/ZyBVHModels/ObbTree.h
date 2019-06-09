#ifndef OWN_OBBTREE_H
#define OWN_OBBTREE_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include "initBVHModelsPlugin.h"

#include <stack>

using namespace sofa::defaulttype;
namespace sofa
{
    namespace component
    {
        namespace collision
        {
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
                /*Matrix4IndicesMap::value_type(R00, std::pair<int,int>(0,0)),
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
                Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_4, std::pair<int,int>(3,3))*/

                Matrix4IndicesMap::value_type(R00, std::pair<int,int>(0,0)),
                Matrix4IndicesMap::value_type(R10, std::pair<int,int>(1,0)),
                Matrix4IndicesMap::value_type(R20, std::pair<int,int>(2,0)),
                Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_1, std::pair<int,int>(3,0)),
                Matrix4IndicesMap::value_type(R01, std::pair<int,int>(0,1)),
                Matrix4IndicesMap::value_type(R11, std::pair<int,int>(1,1)),
                Matrix4IndicesMap::value_type(R21, std::pair<int,int>(2,1)),
                Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_2, std::pair<int,int>(3,1)),
                Matrix4IndicesMap::value_type(R02, std::pair<int,int>(0,2)),
                Matrix4IndicesMap::value_type(R12, std::pair<int,int>(1,2)),
                Matrix4IndicesMap::value_type(R22, std::pair<int,int>(2,2)),
                Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_3, std::pair<int,int>(3,2)),

                Matrix4IndicesMap::value_type(TX, std::pair<int,int>(3,0)),
                Matrix4IndicesMap::value_type(TY, std::pair<int,int>(3,1)),
                Matrix4IndicesMap::value_type(TZ, std::pair<int,int>(3,2)),
                Matrix4IndicesMap::value_type(HOMOGENEOUS_LINE_4, std::pair<int,int>(3,3))
            };

            //typedef std::vector<std::pair<int, int> > OBBOverlapResult;

            typedef std::vector<int> TriIndxVec;

            struct BuildNode
            {
                TriIndxVec   tris;           // the OBB contains these triangles
                Vector3      split_axis;  // the OBB will be split across this axis
                double       split_coord;    // the coordinate on the split axis
                unsigned int obb_indx;       // the index of the obb containing the triangles
            };

            enum BVHNodeType
            {
                PAIR_OBB_OBB,
                PAIR_TRINODE_OBB,
                PAIR_OBB_TRINODE,
                PAIR_TRI_OBB,
                PAIR_OBB_TRI,
                PAIR_TRI_TRINODE,
                PAIR_TRINODE_TRI,
                PAIR_TRINODE_TRINODE,
                PAIR_TRI_TRI
            };

            static std::string BVHNodeTypeName(const BVHNodeType& type)
            {
                switch (type)
                {
                    case PAIR_OBB_OBB:
                        return "OBB <-> OBB";
                        break;
                    case PAIR_TRINODE_OBB:
                        return "TriNode <-> OBB";
                        break;
                    case PAIR_OBB_TRINODE:
                        return "OBB <-> TriNode";
                        break;
                    case PAIR_TRI_OBB:
                        return "Triangle <-> OBB";
                        break;
                    case PAIR_OBB_TRI:
                        return "OBB <-> Triangle";
                        break;
                    case PAIR_TRI_TRINODE:
                        return "Triangle <-> TriNode";
                        break;
                    case PAIR_TRINODE_TRI:
                        return "TriNode <-> Triangle";
                        break;
                    case PAIR_TRINODE_TRINODE:
                        return "TriNode <-> TriNode";
                        break;
                    case PAIR_TRI_TRI:
                        return "Triangle <-> Triangle";
                        break;
                    default:
                        return "Unknown Type";
                        break;
                }
            }

            struct NodePair
            {
                BVHNodeType type;
                int node1;
                int node2;
            };

            class NodeStack
            {
                public:
                    NodeStack(unsigned int n = 10);
                    ~NodeStack();

                    void Push(const NodePair& pair);

                    const NodePair Pop();

                    void Pop(unsigned int n);

                    int IsEmpty() const;

                    void Clear();

                    unsigned int Size() const;

                    // does not delete any memory if n is less than the
                    // number of items on the stack
                    void Reserve(unsigned int n);

                private:
                    NodeStack(const NodeStack& src);            // not defined
                    NodeStack& operator=(const NodeStack& rhs); // not defined

                    NodePair*    nodes;
                    unsigned int num_alloced;
                    unsigned int num_nodes;
            };

            typedef std::stack<BuildNode> BuildStack;

            //! \brief Represents a pair of colliding triangles
            struct CollisionPair
            {
                int id1; //!< Index to triangle in first object
                int id2; //!< Index to triangle in second object
            };


            //! \brief Contains the result from a collision query.
            struct CollideResult
            {
                // statistics variables
                unsigned int num_bv_tests;      //!< The number of bounding-volume tests
                unsigned int num_tri_tests;     //!< The number of triangle overlap tests
                unsigned int num_tri_box_tests; //!< The number of triangle-box overlap tests

                //! Transform from model 1 to model 2
                //double obj2_rel_obj1[12];


                unsigned int num_pairs_alloced; //!< The number of allocated triangle pairs
                unsigned int num_pairs;         //!< The number of colliding triangles
                CollisionPair* pairs;           //!< Pointer to allocated triangle pairs

                unsigned int num_obb_pairs_alloced; //!< The number of allocated OBB pairs
                unsigned int num_obb_pairs;         //!< The number of overlapping OBBs
                CollisionPair* obb_pairs;           //!< Pointer to allocated OBB pairs

                //! Used internally to avoid recursion.
                NodeStack* stack;

                //! The ctor argument controls the intial stack size. The stack will
                //! allocate more memory if needed.
                CollideResult(unsigned int stack_size = 10);
                ~CollideResult();

                //! Reserve space for more colliding triangles
                void Reserve(unsigned int n);

                //! Add a pair of colliding triangles.
                void Add(int i1, int i2);

                //! Reserve space for more colliding triangles
                void ReserveOBB(unsigned int n);

                //! Add a pair of colliding triangles.
                void AddOBB(int i1, int i2);

                //! Clears both the statistics variables and the list of colliding triangles.
                //!
                //! \note Clients do not have to call this as it is called inside the
                //! function \a Collide().
                void Clear();

                //! \brief The number of bounding-volume tests
                unsigned int NumBVTests()     const { return num_bv_tests;      }

                //! \brief The number of triangle overlap tests
                unsigned int NumTriTests()    const { return num_tri_tests;     }

                //! \brief The number of triangle-box overlap tests
                unsigned int NumTriBoxTests() const { return num_tri_box_tests; }

                //! Free the list of contact pairs; ordinarily this list is reused
                //! for each query, and only deleted in the destructor.
                void FreePairsList();


                //! \return TRUE if two models are colliding
                bool IsColliding()      const { return (num_pairs != 0); }

                //! \brief The number of colliding triangle pairs
                unsigned int NumPairs() const { return num_pairs;        }

                //! \brief The number of colliding OBB pairs
                unsigned int NumOBBPairs() const { return num_obb_pairs;        }

                //! \return id of the first triangle in collision pair \a k
                //! \pre \a k must be less than NumPairs. This is an unchecked condition.
                int Id1(unsigned int k) const { return pairs[k].id1;     }

                //! \return id of the second triangle in collision pair \a k
                //! \pre \a k must be less than NumPairs. This is an unchecked condition.
                int Id2(unsigned int k) const { return pairs[k].id2;     }

                private:
                    CollideResult(const CollideResult& src);            //!< not defined
                    CollideResult& operator=(const CollideResult& rhs); //!< not defined
            };

            enum NodeType
            {
                INNER_NODE,
                TRIANGLE_PAIR_NODE,
                LEAF_NODE
            };

            enum ChildType
            {
                INNER_CHILD_NODE,
                TRIANGLE_PAIR_CHILD_NODE,
                TRIANGLE_LEAF_CHILD_NODE
            };

            class ObbVolume
            {
                public:

                    ObbVolume(const NodeType& nodeType = INNER_NODE);
                    ObbVolume(const ObbVolume&);
                    virtual ObbVolume& operator=(const ObbVolume&);
                    ObbVolume(const Vector3& position, const Vector3& halfExtents, const Matrix3& localAxes):
                        m_position(position), m_halfExtents(halfExtents), m_localAxes(localAxes)
                    {}

                    virtual Vector3 getPosition() const { return m_position; }
                    Vector3 getHalfExtents() const { return m_halfExtents; }
                    virtual Vector3 getLocalAxis(int i) const { return m_localAxes.col(i); }
                    virtual Matrix3 getOrientation() const { return m_localAxes; }

                    const int getFirstChild() const { return m_first_child; }
                    const int getSecondChild() const { return m_second_child; }

                    void setFirstChild(const int& idx, const ChildType& childType)
                    {
                        m_first_child = idx;
                        m_childrenSet[0] = true;
                        if (childType == INNER_CHILD_NODE)
                            m_childIsLeaf[0] = false;
                        else
                            m_childIsLeaf[0] = true;

                    }

                    void setSecondChild(const int& idx, const ChildType& childType)
                    {
                        m_second_child = idx;
                        m_childrenSet[1] = true;
                        if (childType == INNER_CHILD_NODE)
                            m_childIsLeaf[1] = false;
                        else
                            m_childIsLeaf[1] = true;

                    }

                    bool testOverlap(ObbVolume &obb2);

                    inline double getSize() const
                    {
                        return (m_halfExtents[0] * m_halfExtents[0] + m_halfExtents[1] * m_halfExtents[1] + m_halfExtents[2] * m_halfExtents[2]);
                    }

                    ChildType getChildType(const unsigned short& idx)
                    {
                        if (m_childIsLeaf[idx])
                            return TRIANGLE_LEAF_CHILD_NODE;
                        else if (m_childrenSet[0] && m_first_child < 0 &&
                                 m_childrenSet[1] && m_second_child < 0)
                            return TRIANGLE_PAIR_CHILD_NODE;
                        else
                            return INNER_CHILD_NODE;
                    }

                    bool getIsChildSet(const unsigned short& idx)
                    {
                        return m_childrenSet[idx];
                    }

                    void draw(sofa::core::visual::VisualParams* vparams);

                    void fitObb(const std::vector<Vector3>&);
                    void fitObb(sofa::core::topology::BaseMeshTopology*, sofa::core::behavior::MechanicalState<Vec3Types>*, BuildNode&);

                    const std::string& identifier() const { return m_identifier; }
                    void setIdentifier(const std::string& id) { m_identifier = id; }

                    const NodeType& getNodeType() const { return m_nodeType; }

                    Matrix4 t_rel_top;

                    void FitToTriangles(core::topology::BaseMeshTopology *trim, const sofa::core::behavior::MechanicalState<Vec3Types>*, const TriIndxVec& indx_vec, Vector3& diam_vec);
                    void SortDimensions();

                    Vector3 m_position;
                    Vector3 m_halfExtents;
                    Matrix3 m_localAxes;

                protected:

                    bool m_childrenSet[2];
                    bool m_childIsLeaf[2];

                    int m_first_child, m_second_child;

                    int m_child_range_min, m_child_range_max;

                    std::string m_identifier;
                    NodeType m_nodeType;

                    void fitToVertices(const Matrix3&, const std::vector<Vector3>&);
                    bool OBBDisJoint(const Matrix4& a_rel_w, const Matrix4& b_rel_w, const Vector3& a, const Vector3& b);

            };

            class TriNode
            {
                public:
                    TriNode(const int& firstChild = 0, const int& secondChild = 0): m_first_child(firstChild), m_second_child(secondChild)
                    {}

                    TriNode(const TriNode& other)
                    {
                        m_first_child = other.m_first_child;
                        m_second_child = other.m_second_child;
                    }

                    TriNode& operator=(const TriNode& other)
                    {
                        if (this != &other)
                        {
                            m_first_child = other.m_first_child;
                            m_second_child = other.m_second_child;
                        }
                        return *this;
                    }

                    void setFirstChild(const int& idx) { m_first_child = idx; }
                    void setSecondChild(const int& idx) { m_second_child = idx; }

                    const int getFirstChild() const { return m_first_child; }
                    const int getSecondChild() const { return m_second_child; }

                    bool singleTriangle() const { return m_second_child < 0; }

            private:
                    int m_first_child, m_second_child;
            };

            class ObbTree: public ObbVolume
            {
                friend class ObbTreeBuilder;
                public:
                    ObbTree(const std::string& name = "", sofa::core::topology::BaseMeshTopology* topology = NULL, sofa::core::behavior::MechanicalState<Vec3Types>* state = NULL):
                        ObbVolume(), m_name(name), m_state(state), m_topology(topology), m_drawOverlappingOnly(false), m_drawTriangleTestsOnly(false)
                    {
                        m_worldRot.identity();
                        m_minDrawDepth = 0;
                        m_maxDrawDepth = 100;
                    }

                    ObbTree(const ObbTree&);
                    virtual ObbTree& operator=(const ObbTree&);
                    ObbTree(sofa::core::topology::BaseMeshTopology* topology, sofa::core::behavior::MechanicalState<Vec3Types>* state, const Vector3& position, const Vector3& halfExtents, const Matrix3& localAxes):
                        m_worldPos(position), m_worldRot(localAxes), m_state(state), m_topology(topology), m_drawOverlappingOnly(false), m_drawTriangleTestsOnly(false)
                    {
                        this->m_halfExtents = halfExtents;
                        m_minDrawDepth = 0;
                        m_maxDrawDepth = 100;
                    }

                    Vector3 getPosition() const { return m_obbNodes[0].m_position; }
                    Vector3 getLocalAxis(int i) const { return m_obbNodes[0].m_localAxes[i]; }
                    Matrix3 getOrientation() const { return m_obbNodes[0].m_localAxes; }

                    bool testOverlap(ObbTree &tree2, CollideResult &res);
                    bool computeOverlap(const ObbTree& tree2, CollideResult& res);

                    void draw(const core::visual::VisualParams *vparams);

                    void translate(const Vector3&);
                    void rotate(const Matrix3&);

                    Vector3 getWorldPosition() { return m_worldPos; }
                    Matrix3 getWorldOrientation() { return m_worldRot; }

                    std::vector<ObbVolume>& getObbNodes() { return m_obbNodes; }

                    std::vector<TriNode>& getTriNodes() { return m_triNodes; }

                    sofa::core::topology::BaseMeshTopology* getTopology() { return m_topology; }
                    sofa::core::behavior::MechanicalState<Vec3Types>* getMState() { return m_state; }

                    const std::string& getName() const { return m_name; }

                    unsigned int getMaxDrawDepth() const { return m_maxDrawDepth; }
                    void setMaxDrawDepth(const unsigned int& depth) { m_maxDrawDepth = depth; }

                    unsigned int getMinDrawDepth() const { return m_minDrawDepth; }
                    void setMinDrawDepth(const unsigned int& depth) { m_minDrawDepth = depth; }

                    void assignOBBNodeColors();

                    std::map<std::string, std::map<int, std::vector<int> > >& getTestedTriangles() { return m_testedTriangles; }
                    std::map<std::string, std::map<int, std::vector<int> > >& getIntersectingTriangles() { return m_intersectingTriangles; }

                protected:
                    Vector3 m_worldPos;
                    Matrix3 m_worldRot;

                    sofa::core::topology::BaseMeshTopology* m_topology;
                    sofa::core::behavior::MechanicalState<Vec3Types>* m_state;

                    unsigned int m_minDrawDepth, m_maxDrawDepth;

                    std::vector<ObbVolume> m_obbNodes;
                    std::vector<TriNode> m_triNodes;

                    std::string m_name;

                    bool m_drawOverlappingOnly;
                    bool m_drawTriangleTestsOnly;
                    std::map<int, Vec4f> m_obbNodeColors;

                    std::map<std::string, std::map<int, std::vector<std::string> > > m_intersectingOBBs;
                    //std::map<std::string, std::map<int, std::vector<int> > > m_intersectingOBBIndices;

                    std::vector<int> m_intersectingOBBIndices;

                    std::map<std::string, std::map<int, std::vector<int> > > m_testedTriangles;
                    std::map<std::string, std::map<int, std::vector<int> > >m_intersectingTriangles;

                    void drawRec(ObbVolume& parent, unsigned int depth, bool overlappingOnly = false);
            };

            namespace LGCOBBUtils
            {
                using namespace sofa::defaulttype;

                template <typename Real>
                void getCovarianceOfVertices(Mat<3,3,Real>& M, const std::vector<Vec<3,Real> >& vertices)
                {
                    int i;
                    Real S1[3];
                    Real S2[3][3];

                    S1[0] = S1[1] = S1[2] = 0.0;
                    S2[0][0] = S2[1][0] = S2[2][0] = 0.0;
                    S2[0][1] = S2[1][1] = S2[2][1] = 0.0;
                    S2[0][2] = S2[1][2] = S2[2][2] = 0.0;

                    // get center of mass
                    for(i=0; i < vertices.size(); i++)
                    {
                        S1[0] += vertices[i].x();
                        S1[1] += vertices[i].y();
                        S1[2] += vertices[i].z();

                        S2[0][0] += vertices[i].x() * vertices[i].x();
                        S2[1][1] += vertices[i].y() * vertices[i].y();
                        S2[2][2] += vertices[i].z() * vertices[i].z();

                        S2[0][1] += vertices[i].x() * vertices[i].y();
                        S2[0][2] += vertices[i].x() * vertices[i].z();
                        S2[1][2] += vertices[i].y() * vertices[i].z();
                    }

                    Real n = vertices.size();

                    // now get covariances

                    M[0][0] = S2[0][0] - S1[0]*S1[0] / n;
                    M[1][1] = S2[1][1] - S1[1]*S1[1] / n;
                    M[2][2] = S2[2][2] - S1[2]*S1[2] / n;
                    M[0][1] = S2[0][1] - S1[0]*S1[1] / n;
                    M[1][2] = S2[1][2] - S1[1]*S1[2] / n;
                    M[0][2] = S2[0][2] - S1[0]*S1[2] / n;
                    M[1][0] = M[0][1];
                    M[2][0] = M[0][2];
                    M[2][1] = M[1][2];
                }

                #define EVV_ROTATE(a,i,j,k,l) g=a[i][j]; h=a[k][l]; a[i][j]=g-s*(h+g*tau); a[k][l]=h+s*(g-h*tau);

                template <typename Real>
                void eigenValuesAndVectors(Mat<3,3,Real>& vout, Vec<3,Real>& dout, Mat<3,3,Real>& a)
                {
                    int n = 3;
                    int j,iq,ip,i;
                    Real tresh,theta,tau,t,sm,s,h,g,c;
                    int nrot;
                    Vec<3,Real> b;
                    Vec<3,Real> z;
                    Mat<3,3,Real> v;
                    Vec<3,Real> d;

                    v.identity();

                    for(ip = 0; ip < n; ip++)
                    {
                        b[ip] = a[ip][ip];
                        d[ip] = a[ip][ip];
                        z[ip] = 0.0;
                    }

                    nrot = 0;

                    for(i=0; i<50; i++)
                    {
                        sm=0.0;
                        for(ip=0;ip<n;ip++) for(iq=ip+1;iq<n;iq++)
                            sm+=fabs(a[ip][iq]);

                        if (sm == 0.0)
                        {
                            vout = v;
                            dout = d;
                            return;
                        }

                        if (i < 3)
                            tresh = (Real)0.2 * sm / (n*n);
                        else
                            tresh = 0.0;

                        for(ip=0; ip<n; ip++) for(iq=ip+1; iq<n; iq++)
                        {
                            g = (Real)100.0 * fabs(a[ip][iq]);
                            if (i>3 &&
                                fabs(d[ip]) + g == fabs(d[ip]) &&
                                fabs(d[iq]) + g == fabs(d[iq]))
                                    a[ip][iq]=0.0;
                            else if (fabs(a[ip][iq]) > tresh)
                            {
                                h = d[iq] - d[ip];
                                if (fabs(h) + g == fabs(h))
                                    t = (a[ip][iq]) / h;
                                else
                                {
                                    theta = (Real)0.5 * h / (a[ip][iq]);
                                    t=(Real)(1.0 / (fabs(theta) + sqrt(1.0 + theta*theta)));
                                    if (theta < 0.0)
                                        t = -t;
                                }
                                c = (Real)1.0 / sqrt(1+t*t);
                                s = t*c;
                                tau = s / ((Real)1.0 + c);
                                h = t * a[ip][iq];
                                z[ip] -= h;
                                z[iq] += h;
                                d[ip] -= h;
                                d[iq] += h;
                                a[ip][iq] = 0.0;

                                for(j = 0; j < ip; j++) { EVV_ROTATE(a,j,ip,j,iq); }
                                for(j = ip + 1; j < iq; j++) { EVV_ROTATE(a,ip,j,j,iq); }
                                for(j = iq + 1;j < n;j++) { EVV_ROTATE(a,ip,j,iq,j); }
                                for(j = 0;j < n; j++) { EVV_ROTATE(v,j,ip,j,iq); }
                                nrot++;
                            }
                        }
                        for(ip = 0; ip < n; ip++)
                        {
                            b[ip] += z[ip];
                            d[ip] = b[ip];
                            z[ip] = 0.0;
                        }
                    }

                    std::cerr << "eigenValuesAndVectors: too many iterations in Jacobi transform." << std::endl;
                    return;
                }

            }

            namespace Moments
            {
                struct Moment
                {
                  double area;      //!< The area
                  Vector3 mean;   //!< The centroid
                  Matrix3 cov; //!< Second order statistics
                };

                //! Computes first and second order statistics for the given triangle
                void compute_moments(const Vector3& p, const Vector3& q, const Vector3& r, Moment& m);

                //! Sets all members to zero.
                void clear_moments(Moment& m);

                //! Accumulates the second order moments, the area, and the area-weighted mean.
                void accum_moments(Moment& a, const Moment& b);

                //! Computes the eigen values and the eigen vectors of the 3x3 matrix \a a.
                //! The eigen vectors end up in \a vout, and the eigen values in \a dout.
                void eigen_3x3(Matrix3& vout, Vector3& dout, Matrix3 &a);
            }

            class SOFA_BVHMODELSPLUGIN_API ObbTreeBuilder: public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(ObbTreeBuilder, core::objectmodel::BaseObject);

                    ObbTreeBuilder() {}
                    ObbTreeBuilder(ObbTree* obbTree, sofa::core::topology::BaseMeshTopology *topology, sofa::core::behavior::MechanicalState<Vec3Types> *m_state):
                        mTopology(topology), mState(m_state), mObbTree(obbTree)
                    {
                        std::cout << "ObbTreeBuilder constructor: obbTree instance = " << &mObbTree << std::endl;
                    }

                    bool buildTree();

                private:
                    sofa::core::topology::BaseMeshTopology* mTopology;
                    sofa::core::behavior::MechanicalState<Vec3Types>* mState;

                    std::vector<ObbVolume> m_obbNodes;
                    std::vector<TriNode> m_triNodes;
                    ObbTree* mObbTree;

                    void
                    compute_obb(ObbVolume&               obb,
                                BuildNode&              bnode,
                                sofa::core::topology::BaseMeshTopology *topology,
                                sofa::core::behavior::MechanicalState<Vec3Types> *m_state,
                                unsigned int            flags);

                    void partition_triangles(const BuildNode& parent,
                                             sofa::core::topology::BaseMeshTopology *topology,
                                             sofa::core::behavior::MechanicalState<Vec3Types> *m_state,
                                             TriIndxVec&             left_child,
                                             TriIndxVec&             right_child);

                    void compute_moments(sofa::core::topology::BaseMeshTopology* topology,
                                         sofa::core::behavior::MechanicalState<Vec3Types>* mState,
                                         const TriIndxVec& indx_vec,
                                         Moments::Moment& m_tot);
            };
        }
    }
}

#endif // OWN_OBBTREE_H
