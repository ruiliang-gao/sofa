#include "ObbTree.h"

#include "BVHDrawHelpers.h"

using namespace sofa::component::collision;

#include <climits>
#include <float.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

#include "ObbTree.inl"

#include "TriangleIntersection.h"

#ifdef _WIN32
#include <gl/glut.h>
#else
#include <GL/glut.h>
#endif

/// See Christer Ericson's book Real-Time Collision Detection, page 83.
void ExtremePointsAlongDirection(const Vector3 &dir, const std::vector<Vector3> pointArray, int &idxSmallest, int &idxLargest)
{
    idxSmallest = idxLargest = 0;

    if (pointArray.size() == 0)
        return;

    float smallestD = FLT_MAX;
    float largestD = -FLT_MAX;
    for(int i = 0; i < pointArray.size(); ++i)
    {
        float d = pointArray[i] * dir;
        if (d < smallestD)
        {
            smallestD = d;
            idxSmallest = i;
        }
        if (d > largestD)
        {
            largestD = d;
            idxLargest = i;
        }
    }
}


int LexFloat3Cmp(const Vector3 &a, const Vector3 &b)
{
    if (a.x() < b.x())
        return 1;

    if (a.y() < b.y())
        return 1;

    if (a.z() < b.z())
        return 1;

    return 0;
}

Vector3 Perpendicular(Vector3 in, const Vector3 &hint = Vector3(0,1,0), const Vector3 &hint2 = Vector3(0,0,1))
{
    /*assume(!this->IsZero());
    assume(hint.IsNormalized());
    assume(hint2.IsNormalized());*/

    Vector3 v = in.cross(hint);
    v.normalize();
    float len = v.norm();
    if (len == 0)
        return hint2;
    else
        return v;
}

Vector3 AnotherPerpendicular(Vector3 in, const Vector3 &hint = Vector3(0,1,0), const Vector3 &hint2 = Vector3(0,0,1))
{
    /*assume(!this->IsZero());
    assume(hint.IsNormalized());
    assume(hint2.IsNormalized());*/

    Vector3 firstPerpendicular = Perpendicular(hint, hint2);
    Vector3 v = in.cross(firstPerpendicular);
    v.normalize();
    return v;
}

Vector2 Rotated90CCW(Vector2 in)
{
    return Vector2(-in.y(), in.x());
}

float MinAreaRect(std::vector<Vector2>& pts, Vector2 &center, Vector2 &uDir, Vector2 &vDir, float &minU, float &maxU, float &minV, float &maxV)
{
    //assume(pts || numPoints == 0);
    if (pts.size() == 0)
        return 0.f;

    float minArea = FLT_MAX;

    // Loop through all edges formed by pairs of points.
    for(int i = 0, j = pts.size() - 1; i < pts.size(); j = i, ++i)
    {
        // The edge formed by these two points.
        Vector2 e0 = pts[i] - pts[j];
        e0.normalize();
        float len = e0.norm();

        if (len == 0)
            continue; // the points are duplicate, skip this axis.

        Vector2 e1 = Rotated90CCW(e0);

        // Find the most extreme points along the coordinate frame { e0, e1 }.

        ///@todo Examine. A bug in the book? All the following are initialized to 0!.
        double min0 = DBL_MAX;
        double min1 = DBL_MAX;
        double max0 = -DBL_MAX;
        double max1 = -DBL_MAX;
        for(int k = 0; k < pts.size(); ++k)
        {
            Vector2 d = pts[k] - pts[j];
            double dot =  d * e0;
            if (dot < min0) min0 = dot;
            if (dot > max0) max0 = dot;
            dot = d * e1;
            if (dot < min1) min1 = dot;
            if (dot > max1) max1 = dot;
        }
        double area = (max0 - min0) * (max1 - min1);

        if (area < minArea)
        {
            minArea = area;
            Vector2 ct1 = (min0 + max0) * e0 + (min1 + max1) * e1;
            Vector2 ct2 = ct1 * 0.5f;
            center = pts[j] + ct2;
            uDir = e0;
            vDir = e1;
            minU = min0;
            maxU = max0;
            minV = min1;
            maxV = max1;
        }
    }
    return minArea;
}

ObbVolume OptimalEnclosingOBB(const std::vector<Vector3> pointArray)
{
    std::cout << "OptimalEnclosingOBB(): " << pointArray.size() << " vertices" << std::endl;
    for (int k = 0; k < pointArray.size(); k++)
        std::cout << pointArray[k] << ";";

    std::cout << std::endl;
    ObbVolume minOBB;
    float minVolume = FLT_MAX;

    std::vector<Vector2> pts;
    pts.resize(pointArray.size());

    std::vector<Vector3> dirs;
    dirs.reserve((pointArray.size() * pointArray.size() - 1) / 2);
    for(int i = 0; i < pointArray.size(); ++i)
    {
        for(int j = i+1; j < pointArray.size(); ++j)
        {
            Vector3 edge = pointArray[i] - pointArray[j];
            edge.normalize();
            float oldLength = edge.norm();

            if (edge.z() < 0.f)
                edge = -edge;

            if (oldLength > 0.f)
            {
                std::cout << "  * direction vector: " << edge << ", length = " << oldLength << std::endl;
                dirs.push_back(edge);
            }
        }
    }

    std::cout << "Got " << dirs.size() << " directions." << std::endl;

    std::cout << "dirs vector (before sorting): ";
    for (int k = 0; k < pointArray.size(); k++)
        std::cout << pointArray[k] << ";";

    std::cout << std::endl;

    std::sort(dirs.begin(), dirs.end(), LexFloat3Cmp);

    std::cout << "dirs vector (after sorting): ";
    for (int k = 0; k < pointArray.size(); k++)
        std::cout << pointArray[k] << ";";

    std::cout << std::endl;

    //sort::QuickSort(&dirs[0], (int)dirs.size(), LexFloat3Cmp);
    for(int i = dirs.size()-1; i >= 0; --i)
    {
        for(int j = i-1; j >= 0; --j)
        {
            float distX = dirs[i].x() - dirs[j].x();
            if (distX > 1e-1f)
                break;

            Vector3 dirs_i = dirs[i];
            Vector3 dirs_j = dirs[j];

            if ((dirs_j - dirs_i).norm2() < 1e-3f)
            {
                std::cout << "  * erase: " << (dirs[j]) << std::endl;
                dirs.erase(dirs.begin() + j);
                --i;
            }
        }
    }
    std::cout << "Pruned to " <<  (int) dirs.size() << " directions." << std::endl;

    for(size_t i = 0; i < dirs.size(); ++i)
    {
        Vector3 edge = dirs[i];

        int e1, e2;
        ExtremePointsAlongDirection(edge, pointArray, e1, e2);
        std::cout << " * " << i << " = " << edge << ": ExtremePoints = " << e1 << " = " << pointArray[e1] << "," << e2 << " = " << pointArray[e2] << std::endl;


        float edgeLength = fabs((pointArray[e1] * edge) - (pointArray[e2] * edge));

        Vector3 u = Perpendicular(edge);
        Vector3 v = AnotherPerpendicular(edge);
        for(int k = 0; k < pts.size(); ++k)
            pts[k] = Vector2(pointArray[k] * u, pointArray[k] * v);

        Vector2 rectCenter;
        Vector2 rectU;
        Vector2 rectV;
        float minU, maxU, minV, maxV;
        float rectArea = MinAreaRect(pts, rectCenter, rectU, rectV, minU, maxU, minV, maxV);
        Vector3 rectCenterPos = u * rectCenter.x() + v * rectCenter.y();

        float volume = rectArea * edgeLength;
        if (volume < minVolume)
        {
            Matrix3 localAxes;
            localAxes.col(0) = edge;
            localAxes.col(1) = rectU.x() * u + rectU.y() * v;
            localAxes.col(2) = rectV.x() * u + rectV.y() * v;

            Vector3 halfExtents(edgeLength * 0.5f, (maxU - minU) * 0.5f, (maxV - minV) * 0.5f);
            Vector3 obbPos = ((pointArray[e1] * edge) + (pointArray[e2] * edge)) * 0.5f * edge + rectCenterPos;

            minOBB = ObbVolume(obbPos, halfExtents, localAxes);

            minVolume = volume;

            std::cout << " - " << i << ": new minOBB at " << obbPos << ", extents = " << halfExtents << ", localAxes = " << localAxes << std::endl;

            /*minOBB.axis[0] = edge;
            minOBB.axis[1] = rectU.x * u + rectU.y * v;
            minOBB.axis[2] = rectV.x * u + rectV.y * v;
            minOBB.pos = (Dot(pointArray[e1], edge) + Dot(pointArray[e2], edge)) * 0.5f * edge + rectCenterPos;
            minOBB.r[0] = edgeLength * 0.5f;
            minOBB.r[1] = (maxU - minU) * 0.5f;
            minOBB.r[2] = (maxV - minV) * 0.5f;*/

        }
    }
    return minOBB;
}

//OBB::OBB(const std::vector<Vector3>& vertices)
//{
//    fitObb(vertices);
//}

void ObbVolume::fitObb(const std::vector<Vector3>& vertices)
{
    /*std::cout << "fitOBB(): " << vertices.size() << " vertices." << std::endl;

    for (int k = 0; k < vertices.size(); k++)
        std::cout << vertices[k] << ";";

    std::cout << std::endl;*/

    Matrix3 C, E, R;
    Vector3 s, axis, mean;
    double coord;

    LGCOBBUtils::getCovarianceOfVertices(C, vertices);

    // std::cout << " Covariance matrix: " << C << std::endl;

    LGCOBBUtils::eigenValuesAndVectors(E, s, C);
    // std::cout << " Eigenvalues = " << s << ", eigenvectors = " << E << std::endl;

    // place axes of E in order of increasing s

    int min, mid, max;
    if (s[0] > s[1]) { max = 0; min = 1; }
    else { min = 0; max = 1; }
    if (s[2] < s[min]) { mid = min; min = 2; }
    else if (s[2] > s[max]) { mid = max; max = 2; }
    else { mid = 2; }

    // std::cout << " min = " << min << ", mid = " << mid << ", max = " << max << std::endl;

    R[0][0] = E[0][max];
    R[1][0] = E[1][max];
    R[2][0] = E[2][max];

    R[0][1] = E[0][mid];
    R[1][1] = E[1][mid];
    R[2][1] = E[2][mid];

    R[0][2] = E[1][max]*E[2][mid] - E[1][mid]*E[2][max];
    R[1][2] = E[0][mid]*E[2][max] - E[0][max]*E[2][mid];
    R[2][2] = E[0][max]*E[1][mid] - E[0][mid]*E[1][max];

    // std::cout << " rotation matrix = " << R << std::endl;

    fitToVertices(R, vertices);
}

void ObbVolume::fitToVertices(const Matrix3& R, const std::vector<Vector3>& vertices)
{
    // store orientation

    //McM(R,O);
    m_localAxes = R;

    // project points of tris to R coordinates

    std::vector<Vector3> projectedVertices;

    Matrix3 R_transposed = R.transposed();
    // std::cout << " projecting vertices: R_transposed = " << R_transposed << std::endl;
    for (int i = 0; i < vertices.size(); i++)
    {
        projectedVertices.push_back(R_transposed * vertices[i]);
        // std::cout << "  - " << i << ": " << projectedVertices[i] << " = " << R_transposed << " * " << vertices[i] << std::endl;
    }

    // std::cout << " projectedVertices size = " << projectedVertices.size() << std::endl;
    double minx, maxx, miny, maxy, minz, maxz;
    Vector3 c;

    minx = maxx = projectedVertices[0][0];
    miny = maxy = projectedVertices[0][1];
    minz = maxz = projectedVertices[0][2];
    for (int i = 1; i < projectedVertices.size(); i++)
    {
        /*std::cout << "  * " << i << ": " << projectedVertices[i][0] << " < " << minx << ": " << (projectedVertices[i][0] < minx) << ";";
        std::cout << projectedVertices[i][0] << " > " << maxx << ": " << (projectedVertices[i][0] > maxx) << ";";

        std::cout << projectedVertices[i][1] << " < " << miny << ": " << (projectedVertices[i][1] < miny) << ";";
        std::cout << projectedVertices[i][1] << " > " << maxy << ": " << (projectedVertices[i][1] > maxy) << ";";

        std::cout << projectedVertices[i][2] << " < " << minz << ": " << (projectedVertices[i][2] < minz) << ";";
        std::cout << projectedVertices[i][2] << " > " << maxz << ": " << (projectedVertices[i][2] > maxz) << ";" << std::endl;*/

        if (projectedVertices[i][0] < minx)
            minx = projectedVertices[i][0];
        else if (projectedVertices[i][0] > maxx)
            maxx = projectedVertices[i][0];

        if (projectedVertices[i][1] < miny)
            miny = projectedVertices[i][1];
        else if (projectedVertices[i][1] > maxy)
            maxy = projectedVertices[i][1];

        if (projectedVertices[i][2] < minz)
            minz = projectedVertices[i][2];
        else if (projectedVertices[i][2] > maxz)
            maxz = projectedVertices[i][2];
    }

    /*std::cout << " maxx = " << maxx << ", minx = " << minx << std::endl;
    std::cout << " maxy = " << maxy << ", miny = " << miny << std::endl;
    std::cout << " maxz = " << maxz << ", minz = " << minz << std::endl;*/

    c[0] = 0.5f * (maxx + minx);
    c[1] = 0.5f * (maxy + miny);
    c[2] = 0.5f * (maxz + minz);

    // std::cout << " center = " << c << std::endl;

    m_position = R * c;

    // std::cout << " position = " << m_position << " = " << R << " * " << c << std::endl;

    m_halfExtents[0] = 0.5f * (maxx - minx);
    m_halfExtents[1] = 0.5f * (maxy - miny);
    m_halfExtents[2] = 0.5f * (maxz - minz);

    // std::cout << " halfExtents = " << m_halfExtents << std::endl;
}

void ObbVolume::fitObb(sofa::core::topology::BaseMeshTopology* mTopology, sofa::core::behavior::MechanicalState<Vec3Types>* mState, BuildNode& buildNode)
{
    // std::cout << "ObbVolume::fitObb(): " << buildNode.tris.size() << " triangles to fit." << std::endl;
    // std::cout << " topology triangle count: " << mTopology->getNbTriangles() << std::endl;
    std::vector<Vector3> vertices;
    const Vec3Types::VecCoord& x = mState->read(core::ConstVecCoordId::position())->getValue();
    for (int i = 0; i < buildNode.tris.size(); i++)
    {
        // std::cout << "  triangle " << i << " = " << buildNode.tris.at(i) << std::endl;
        const sofa::core::topology::Topology::Triangle tri = mTopology->getTriangle(buildNode.tris.at(i));
        Vector3 pt1 = x[tri[0]];
        Vector3 pt2 = x[tri[1]];
        Vector3 pt3 = x[tri[2]];

        vertices.push_back(pt1);
        vertices.push_back(pt2);
        vertices.push_back(pt3);
    }

    if (vertices.size() > 0)
        fitObb(vertices);
}


//#define TR_INV_MULT_ROW0(ab,a,b)                                                                          \
//  (ab)[R00] = ((ab)[R11]*(ab)[R22]) - ((ab)[R12]*(ab)[R21]);                                              \
//  (ab)[R01] = ((ab)[R12]*(ab)[R20]) - ((ab)[R10]*(ab)[R22]);                                              \
//  (ab)[R02] = ((ab)[R10]*(ab)[R21]) - ((ab)[R11]*(ab)[R20]);                                              \
//  (ab)[TX]  = (a)[R00]*((b)[TX] - (a)[TX]) + (a)[R10]*((b)[TY] - (a)[TY]) + (a)[R20]*((b)[TZ] - (a)[TZ])

void TR_INV_MULT_ROW0(Matrix4& ab, const Matrix4& a, const Matrix4& b)
{
  (ab)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] =
  ((ab)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
   (ab)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second]) -
  ((ab)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
   (ab)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second]);

  (ab)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] =
  ((ab)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
   (ab)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second]) -
  ((ab)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
   (ab)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second]);

  (ab)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] =
  ((ab)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
   (ab)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second]) -
  ((ab)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
   (ab)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second]);

  (ab)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] =
  (a)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] *
  ((b)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] -
   (a)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]) +
   (a)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
   ((b)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] -
   (a)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second]) +
   (a)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] *
   ((b)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] -
   (a)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second]);
}

//#define TR_INV_MULT_ROW1(ab,a,b)                                                                          \
//  (ab)[R10] = (a)[R01]*(b)[R00] + (a)[R11]*(b)[R10] + (a)[R21]*(b)[R20];                                  \
//  (ab)[R11] = (a)[R01]*(b)[R01] + (a)[R11]*(b)[R11] + (a)[R21]*(b)[R21];                                  \
//  (ab)[R12] = (a)[R01]*(b)[R02] + (a)[R11]*(b)[R12] + (a)[R21]*(b)[R22];                                  \
//  (ab)[TY]  = (a)[R01]*((b)[TX] - (a)[TX]) + (a)[R11]*((b)[TY] - (a)[TY]) + (a)[R21]*((b)[TZ] - (a)[TZ])

void TR_INV_MULT_ROW1(Matrix4& ab, const Matrix4& a, const Matrix4& b)
{
    (ab)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] =
    ((a)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] *
    (b)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second]) +
    ((a)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
    (b)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second]) +
    ((a)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] *
    (b)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second]);

    (ab)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] =
    ((a)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] *
    (b)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second]) +
    ((a)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
    (b)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second]) +
    ((a)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] *
    (b)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second]);

    (ab)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] =
    ((a)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] *
    (b)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second]) +
    ((a)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
    (b)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second]) +
    ((a)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] *
    (b)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second]);

    (ab)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] =
    (a)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] *
    ((b)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] -
     (a)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]) +
     (a)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
     ((b)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] -
     (a)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second]) +
     (a)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] *
     ((b)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] -
     (a)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second]);
}

//#define TR_INV_MULT_ROW2(ab,a,b)                                                                          \
//  (ab)[R20] = (a)[R02]*(b)[R00] + (a)[R12]*(b)[R10] + (a)[R22]*(b)[R20];                                  \
//  (ab)[R21] = (a)[R02]*(b)[R01] + (a)[R12]*(b)[R11] + (a)[R22]*(b)[R21];                                  \
//  (ab)[R22] = (a)[R02]*(b)[R02] + (a)[R12]*(b)[R12] + (a)[R22]*(b)[R22];                                  \
//  (ab)[TZ]  = (a)[R02]*((b)[TX] - (a)[TX]) + (a)[R12]*((b)[TY] - (a)[TY]) + (a)[R22]*((b)[TZ] - (a)[TZ])

void TR_INV_MULT_ROW2(Matrix4& ab, const Matrix4& a, const Matrix4& b)
{
    (ab)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] =
    ((a)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *
    (b)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second]) +
    ((a)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
    (b)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second]) +
    ((a)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *
    (b)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second]);

    (ab)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] =
    ((a)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *
    (b)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second]) +
    ((a)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
    (b)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second]) +
    ((a)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *
    (b)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second]);

    (ab)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] =
    ((a)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *
    (b)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second]) +
    ((a)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
    (b)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second]) +
    ((a)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *
    (b)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second]);

    (ab)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] =
    (a)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *
    ((b)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] -
     (a)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]) +
     (a)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
     ((b)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] -
     (a)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second]) +
     (a)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *
     ((b)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] -
     (a)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second]);
}

#define OBB_EPS 1e-6
bool ObbVolume::OBBDisJoint(const Matrix4& a_rel_w, const Matrix4& b_rel_w, const Vector3& a, const Vector3& b)
{
  Matrix4 b_rel_a;
  Matrix4 bf;        // bf = fabs(b_rel_a) + eps
  double t, t2;

  // Class I tests
  TR_INV_MULT_ROW2(b_rel_a, a_rel_w, b_rel_w);

  bf[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] = fabs(b_rel_a[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second]) + OBB_EPS;
  bf[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] = fabs(b_rel_a[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second]) + OBB_EPS;
  bf[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] = fabs(b_rel_a[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second]) + OBB_EPS;

  // A0 x A1 = A2
  /*t  = b_rel_a[TZ];
  t2 = a[2] + b[0] * bf[R20] + b[1] * bf[R21] + b[2] * bf[R22];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second];
  t2 = a[2] + b[0] * bf[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] +
       b[1] * bf[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] +
       b[2] * bf[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second];

  if (t > t2) { return true; }

  TR_INV_MULT_ROW1(b_rel_a, a_rel_w, b_rel_w);

  bf[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] = fabs(b_rel_a[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second]) + OBB_EPS;
  bf[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] = fabs(b_rel_a[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second]) + OBB_EPS;
  bf[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] = fabs(b_rel_a[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second]) + OBB_EPS;

  // A2 x A0 = A1
  /*t  = b_rel_a[TY];
  t2 = a[1] + b[0] * bf[R10] + b[1] * bf[R11] + b[2] * bf[R12];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second];
  t2 = a[1] + b[0] * bf[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] +
       b[1] * bf[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] +
       b[2] * bf[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second];
  if (t > t2) { return true; }


  TR_INV_MULT_ROW0(b_rel_a, a_rel_w, b_rel_w);

  bf[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] = fabs(b_rel_a[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second]) + OBB_EPS;
  bf[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] = fabs(b_rel_a[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second]) + OBB_EPS;
  bf[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] = fabs(b_rel_a[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second]) + OBB_EPS;

  // A1 x A2 = A0
  /*t  = b_rel_a[TX];
  t2 = a[0] + b[0] * bf[R00] + b[1] * bf[R01] + b[2] * bf[R02];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second];
  t2 = a[0] + b[0] * bf[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] +
       b[1] * bf[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] +
       b[2] * bf[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second];
  if (t > t2) { return true; }

  //assert(is_rot_matrix(b_rel_a));

  // Class II tests

  // B0 x B1 = B2
  /*t  = b_rel_a[TX]*b_rel_a[R02] + b_rel_a[TY]*b_rel_a[R12] + b_rel_a[TZ]*b_rel_a[R22];
    t2 = b[2] + a[0] * bf[R02] + a[1] * bf[R12] + a[2] * bf[R22];
    if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] *
       b_rel_a[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] +
       b_rel_a[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] *
       b_rel_a[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] +
       b_rel_a[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] *
       b_rel_a[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second];

  t2 = b[2] + a[0] * bf[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] +
       a[1] * bf[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] +
       a[2] * bf[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second];

  if (t > t2) { return true; }

  // B2 x B0 = B1
  t  = b_rel_a[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] *
       b_rel_a[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] +
       b_rel_a[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] *
       b_rel_a[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] +
       b_rel_a[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] *
       b_rel_a[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second];

  t2 = b[1] + a[0] * bf[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] +
       a[1] * bf[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] +
       a[2] * bf[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second];

  if (t > t2) { return true; }

  // B1 x B2 = B0
  t  = b_rel_a[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] *
       b_rel_a[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] +
       b_rel_a[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] *
       b_rel_a[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] +
       b_rel_a[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] *
       b_rel_a[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second];

  t2 = b[0] + a[0] * bf[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] +
       a[1] * bf[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] +
       a[2] * bf[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second];

  if (t > t2) { return true; }

  // Class III tests

  // A0 x B0
  t  = b_rel_a[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] *
       b_rel_a[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] -
       b_rel_a[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] *
       b_rel_a[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second];

  t2 = a[1] * bf[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] +
       a[2] * bf[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] +
       b[1] * bf[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] +
       b[2] * bf[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second];

  if (t > t2) { return true; }

  // A0 x B1
  /*t  = b_rel_a[TZ] * b_rel_a[R11] - b_rel_a[TY] * b_rel_a[R21];
  t2 = a[1] * bf[R21] + a[2] * bf[R11] + b[0] * bf[R02] + b[2] * bf[R00];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] *
       b_rel_a[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] -
       b_rel_a[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] *
       b_rel_a[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second];

  t2 = a[1] * bf[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] +
       a[2] * bf[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] +
       b[0] * bf[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] +
       b[2] * bf[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second];

  if (t > t2) { return true; }

  // A0 x B2
  /*t  = b_rel_a[TZ] * b_rel_a[R12] - b_rel_a[TY] * b_rel_a[R22];
  t2 = a[1] * bf[R22] + a[2] * bf[R12] + b[0] * bf[R01] + b[1] * bf[R00];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] *
       b_rel_a[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] -
       b_rel_a[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] *
       b_rel_a[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second];

  t2 = a[1] * bf[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] +
       a[2] * bf[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] +
       b[0] * bf[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] +
       b[1] * bf[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second];

  if (t > t2) { return true; }


  // A1 x B0
  /*t  = b_rel_a[TX] * b_rel_a[R20] - b_rel_a[TZ] * b_rel_a[R00];
  t2 = a[0] * bf[R20] + a[2] * bf[R00] + b[1] * bf[R12] + b[2] * bf[R11];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] *
       b_rel_a[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] -
       b_rel_a[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] *
       b_rel_a[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second];

  t2 = a[0] * bf[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] +
       a[2] * bf[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] +
       b[1] * bf[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] +
       b[2] * bf[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second];

  if (t > t2) { return true; }

  // A1 x B1
  /*t  = b_rel_a[TX] * b_rel_a[R21] - b_rel_a[TZ] * b_rel_a[R01];
  t2 = a[0] * bf[R21] + a[2] * bf[R01] + b[0] * bf[R12] + b[2] * bf[R10];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] *
       b_rel_a[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] -
       b_rel_a[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] *
       b_rel_a[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second];

  t2 = a[0] * bf[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] +
       a[2] * bf[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] +
       b[0] * bf[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] +
       b[2] * bf[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second];

  if (t > t2) { return true; }

  // A1 x B2
  /*t  = b_rel_a[TX] * b_rel_a[R22] - b_rel_a[TZ] * b_rel_a[R02];
  t2 = a[0] * bf[R22] + a[2] * bf[R02] + b[0] * bf[R11] + b[1] * bf[R10];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] *
       b_rel_a[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] -
       b_rel_a[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] *
       b_rel_a[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second];

  t2 = a[0] * bf[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] +
       a[2] * bf[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] +
       b[0] * bf[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] +
       b[1] * bf[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second];

  if (t > t2) { return true; }

  // A2 x B0
  /*t  = b_rel_a[TY] * b_rel_a[R00] - b_rel_a[TX] * b_rel_a[R10];
  t2 = a[0] * bf[R10] + a[1] * bf[R00] + b[1] * bf[R22] + b[2] * bf[R21];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] *
       b_rel_a[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] -
       b_rel_a[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] *
       b_rel_a[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second];

  t2 = a[0] * bf[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] +
       a[1] * bf[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] +
       b[1] * bf[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] +
       b[2] * bf[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second];

  if (t > t2) { return true; }

  // A2 x B1
  /*t  = b_rel_a[TY] * b_rel_a[R01] - b_rel_a[TX] * b_rel_a[R11];
  t2 = a[0] * bf[R11] + a[1] * bf[R01] + b[0] * bf[R22] + b[2] * bf[R20];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] *
       b_rel_a[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] -
       b_rel_a[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] *
       b_rel_a[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second];

  t2 = a[0] * bf[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] +
       a[1] * bf[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] +
       b[0] * bf[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] +
       b[2] * bf[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second];

  if (t > t2) { return true; }

  // A2 x B2
  /*t  = b_rel_a[TY] * b_rel_a[R02] - b_rel_a[TX] * b_rel_a[R12];
  t2 = a[0] * bf[R12] + a[1] * bf[R02] + b[0] * bf[R21] + b[1] * bf[R20];
  if (GREATER(t, t2)) { return TRUE; }*/

  t  = b_rel_a[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] *
       b_rel_a[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] -
       b_rel_a[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] *
       b_rel_a[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second];

  t2 = a[0] * bf[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] +
       a[1] * bf[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] +
       b[0] * bf[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] +
       b[1] * bf[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second];

  if (t > t2) { return true; }

  return false;
}

bool ObbVolume::testOverlap(ObbVolume &obb2)
{
    double ra, rb;
    Matrix3 R, AbsR;

    for (unsigned int i = 0; i < 3; i++)
        for (unsigned int j = 0; j < 3; j++)
            R[i][j] = m_localAxes.col(i) * obb2.getLocalAxis(j);

    const Vector3 obb1Center = m_position;
    const Vector3 obb2Center = obb2.getPosition();

    std::cout << " OBB1 at " << obb1Center <<", OBB2 at " << obb2Center << std::endl;
    Vector3 tt = obb2Center - obb1Center;
    std::cout << " OBB2 - OBB1 = " << tt << std::endl;
    std::cout << " OBB1 localAxes = " << getLocalAxis(0) << "/" << getLocalAxis(1) << "/" << getLocalAxis(2) << std::endl;
    std::cout << " OBB2 localAxes = " << obb2.getLocalAxis(0) << "/" << obb2.getLocalAxis(1) << "/" << obb2.getLocalAxis(2) << std::endl;

    double sp1 = tt * getLocalAxis(0);
    double sp2 = tt * getLocalAxis(1);
    double sp3 = tt * getLocalAxis(2);
    std::cout << " scalar products: " << sp1 << "," << sp2 << "," << sp3 << std::endl;
    Vector3 t(sp1,sp2,sp3);
    std::cout << " OBB2 - OBB1 center in local coordinates of OBB1 = " << t << std::endl;

    for (unsigned int i = 0; i < 3; i++)
        for (unsigned int j = 0; j < 3; j++)
            AbsR.col(i)[j] = fabs(R.col(i)[j]) + 1e-6;

    for (unsigned int i = 0; i < 3; i++)
    {
        ra = getHalfExtents()[i];
        rb = obb2.getHalfExtents()[0] * AbsR.col(i)[0] + obb2.getHalfExtents()[1] * AbsR.col(i)[1] + obb2.getHalfExtents()[2] * AbsR.col(i)[2];

        std::cout << " axis test A" << i << ": " << ra << " + " << rb << " = " << ra + rb;
        if (fabs(t[i]) > (ra + rb))
        {
            std::cout << " state: no intersection" << std::endl;
            return false;
        }
        std::cout << "state: intersection" << std::endl;
    }

    for (unsigned int i = 0; i < 3; i++)
    {
        ra = getHalfExtents()[0] * AbsR.col(0)[i] + getHalfExtents()[1] * AbsR.col(1)[i] + getHalfExtents()[2] * AbsR.col(2)[i];
        rb = obb2.getHalfExtents()[i];

        std::cout << " axis test B" << i << ": " << ra << " + " << rb << " = " << ra + rb << std::endl;
        if (fabs(t[0] * R.col(0)[i] + t[1] * R.col(1)[i] + t[2] * R.col(2)[i]) > ra + rb)
        {
            std::cout << " state: no intersection" << std::endl;
            return false;
        }
        std::cout << "state: intersection" << std::endl;
    }

    ra = getHalfExtents()[1] * AbsR.col(2)[0] + getHalfExtents()[2] * AbsR.col(1)[0];
    rb = obb2.getHalfExtents()[1] * AbsR.col(0)[2] + obb2.getHalfExtents()[2] * AbsR.col(0)[1];
    std::cout << " axis test A0xB0: " << ra << " + " << rb << " = " << ra + rb <<std::endl;
    if (fabs(t[2] * R.col(1)[0] - t[1] * R.col(2)[0]) > ra + rb)
    {
        std::cout << " state: no intersection" << std::endl;
        return false;
    }
    std::cout << " state: intersection" << std::endl;

    ra = getHalfExtents()[1] * AbsR.col(2)[1] + getHalfExtents()[2] * AbsR.col(1)[1];
    rb = obb2.getHalfExtents()[0] * AbsR.col(0)[2] + obb2.getHalfExtents()[2] * AbsR.col(0)[0];
    std::cout << " axis test A0xB1: " << ra << " + " << rb << " = " << ra + rb <<std::endl;
    if (fabs(t[2] * R.col(1)[1] - t[1] * R.col(2)[1]) > ra + rb)
    {
        std::cout << " state: no intersection" <<std::endl;
        return false;
    }
    std::cout << " state: intersection" << std::endl;

    ra = getHalfExtents()[1] * AbsR.col(2)[2] + getHalfExtents()[2] * AbsR.col(1)[2];
    rb = obb2.getHalfExtents()[0] * AbsR.col(0)[1] + obb2.getHalfExtents()[1] * AbsR.col(0)[0];
    std::cout << " axis test A0xB2: " << ra << " + " << rb << " = " << ra + rb <<std::endl;
    if (fabs(t[2] * R.col(1)[2] - t[1] * R.col(2)[2]) > ra + rb)
    {
        std::cout << " state: no intersection" <<std::endl;
        return false;
    }
    std::cout << " state: intersection" << std::endl;

    ra = getHalfExtents()[0] * AbsR.col(2)[0] + getHalfExtents()[2] * AbsR.col(0)[0];
    rb = obb2.getHalfExtents()[1] * AbsR.col(1)[2] + obb2.getHalfExtents()[2] * AbsR.col(1)[1];
    std::cout << " axis test A1xB0: " << ra << " + " << rb << " = " << ra + rb <<std::endl;
    if (fabs(t[0] * R.col(2)[0] - t[2] * R.col(0)[0]) > ra + rb)
    {
        std::cout << " state: no intersection" <<std::endl;
        return false;
    }
    std::cout << " state: intersection" << std::endl;

    ra = getHalfExtents()[0] * AbsR.col(2)[1] + getHalfExtents()[2] * AbsR.col(0)[1];
    rb = obb2.getHalfExtents()[0] * AbsR.col(1)[2] + obb2.getHalfExtents()[2] * AbsR.col(1)[0];
    std::cout << " axis test A1xB1: " << ra << " + " << rb << " = " << ra + rb <<std::endl;
    if (fabs(t[0] * R.col(2)[1] - t[2] * R.col(0)[1]) > ra + rb)
    {
        std::cout << " state: no intersection" <<std::endl;
        return false;
    }
    std::cout << " state: intersection" << std::endl;

    ra = getHalfExtents()[0] * AbsR.col(2)[2] + getHalfExtents()[2] * AbsR.col(0)[2];
    rb = obb2.getHalfExtents()[0] * AbsR.col(1)[1] + obb2.getHalfExtents()[1] * AbsR.col(1)[0];
    std::cout << " axis test A1xB2: " << ra << " + " << rb << " = " << ra + rb <<std::endl;
    if (fabs(t[0] * R.col(2)[2] - t[2] * R.col(0)[2]) > ra + rb)
    {
        std::cout << " state: no intersection" <<std::endl;
        return false;
    }
    std::cout << " state: intersection" << std::endl;

    ra = getHalfExtents()[0] * AbsR.col(1)[0] + getHalfExtents()[1] * AbsR.col(0)[0];
    rb = obb2.getHalfExtents()[1] * AbsR.col(2)[2] + obb2.getHalfExtents()[2] * AbsR.col(2)[1];
    std::cout << " axis test A2xB0: " << ra << " + " << rb << " = " << ra + rb <<std::endl;
    if (fabs(t[1] * R.col(0)[0] - t[0] * R.col(1)[0]) > ra + rb)
    {
        std::cout << " state: no intersection" <<std::endl;
        return false;
    }
    std::cout << " state: intersection" << std::endl;

    ra = getHalfExtents()[0] * AbsR.col(1)[1] + getHalfExtents()[1] * AbsR.col(0)[1];
    rb = obb2.getHalfExtents()[0] * AbsR.col(2)[2] + obb2.getHalfExtents()[2] * AbsR.col(2)[0];
    std::cout << " axis test A2xB1: " << ra << " + " << rb << " = " << ra + rb <<std::endl;
    if (fabs(t[1] * R.col(0)[1] - t[0] * R.col(1)[1]) > ra + rb)
    {
        std::cout << " state: no intersection" <<std::endl;
        return false;
    }
    std::cout << " state: intersection" << std::endl;

    ra = getHalfExtents()[0] * AbsR.col(1)[2] + getHalfExtents()[1] * AbsR.col(0)[2];
    rb = obb2.getHalfExtents()[0] * AbsR.col(2)[1] + obb2.getHalfExtents()[1] * AbsR.col(2)[0];
    std::cout << " axis test A2xB2: " << ra << " + " << rb << " = " << ra + rb <<std::endl;
    if (fabs(t[1] * R.col(0)[2] - t[0] * R.col(1)[2]) > ra + rb)
    {
        std::cout << " state: no intersection" <<std::endl;
        return false;
    }
    std::cout << " state: intersection" << std::endl;

    std::cout << " state: all tests passed, OBBs intersect" << std::endl;
    return true;
}

void ObbVolume::draw(sofa::core::visual::VisualParams *vparams)
{

}

//ObbTree::ObbTree(const std::vector<Vector3>& vertices): OBB(vertices)
//{
//    m_worldRot.identity();
//}

void TINV_MUL_T(Matrix4& dest, const Matrix4& a, const Matrix4& b)
{
    dest[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] =
        a[0][0]*b[0][0] + a[0][1]*b[0][1] + a[0][2]*b[0][2];
    dest[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] =
        a[1][0]*b[0][0] + a[1][1]*b[0][1] + a[1][2]*b[0][2];
    dest[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] =
        a[2][0]*b[0][0] + a[2][1]*b[0][1] + a[2][2]*b[0][2];
    dest[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] =
        a[0][0]*b[1][0] + a[0][1]*b[1][1] + a[0][2]*b[1][2];
    dest[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] =
        a[1][0]*b[1][0] + a[1][1]*b[1][1] + a[1][2]*b[1][2];
    dest[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] =
        a[2][0]*b[1][0] + a[2][1]*b[1][1] + a[2][2]*b[1][2];

    dest[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] =
        (dest[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
         dest[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second]) -
        (dest[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] *
         dest[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second]);

    dest[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] =
        (dest[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] *
         dest[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second]) -
        (dest[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] *
         dest[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second]);

    dest[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] =
        (dest[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] *
         dest[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second]) -
        (dest[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
         dest[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second]);

    dest[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] =
         a[0][0]*(b[0][3] - a[0][3]) + a[1][0]*(b[1][3] - a[1][3]) + a[2][0]*(b[2][3] - a[2][3]);

    //std::cout << " dest[" << matrixHelperIndices[TY].second.first << "][" << matrixHelperIndices[TY].second.second << "]" << std::endl;
    //std::cout << a[0][1] << "*" << (b[0][3] - a[0][3]) << " + " << a[1][1] << " * " << (b[1][3] - a[1][3]) << " + " << a[2][1] << " * " << (b[2][3] - a[2][3]) << std::endl;
    dest[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] =
         a[0][1]*(b[0][3] - a[0][3]) + a[1][1]*(b[1][3] - a[1][3]) + a[2][1]*(b[2][3] - a[2][3]);
    dest[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] =
         a[0][2]*(b[0][3] - a[0][3]) + a[1][2]*(b[1][3] - a[1][3]) + a[2][2]*(b[2][3] - a[2][3]);

    dest[matrixHelperIndices[HOMOGENEOUS_LINE_4].second.first][matrixHelperIndices[HOMOGENEOUS_LINE_4].second.second] = 1.0f;
}

//#define TRANSFORM_INV(dest,src)                                                     \
//  (dest)[R00] = (src)[R00]; (dest)[R01] = (src)[R10]; (dest)[R02] = (src)[R20];     \
//  (dest)[R10] = (src)[R01]; (dest)[R11] = (src)[R11]; (dest)[R12] = (src)[R21];     \
//  (dest)[R20] = (src)[R02]; (dest)[R21] = (src)[R12]; (dest)[R22] = (src)[R22];     \
//  (dest)[TX] = -(src)[R00]*(src)[TX] - (src)[R10]*(src)[TY] - (src)[R20]*(src)[TZ]; \
//  (dest)[TY] = -(src)[R01]*(src)[TX] - (src)[R11]*(src)[TY] - (src)[R21]*(src)[TZ]; \
//  (dest)[TZ] = -(src)[R02]*(src)[TX] - (src)[R12]*(src)[TY] - (src)[R22]*(src)[TZ]


void TRANSFORM_INV(Matrix4& dest, const Matrix4& src)
{
    (dest)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] = (src)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second];
    (dest)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] = (src)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second];
    (dest)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] = (src)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second];
    (dest)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] = (src)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second];
    (dest)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] = (src)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second];
    (dest)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] = (src)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second];
    (dest)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] = (src)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second];
    (dest)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] = (src)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second];
    (dest)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] = (src)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second];
    (dest)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] =
          -(src)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] *
           (src)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] -
           (src)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
           (src)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] -
           (src)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] *
           (src)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second];

    (dest)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] =
           -(src)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] *
            (src)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] -
            (src)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
            (src)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] -
            (src)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] *
            (src)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second];

    (dest)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] =
            -(src)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *
             (src)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] -
             (src)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
             (src)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] -
             (src)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *
             (src)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second];

            dest[matrixHelperIndices[HOMOGENEOUS_LINE_4].second.first][matrixHelperIndices[HOMOGENEOUS_LINE_4].second.second] = 1.0f;
}

//#define TR_MULT(ab,a,b)                                                               \
//  ROT_MTRX_MULT(ab,a,b);                                                              \
//  (ab)[TX] = ((a)[R00]*(b)[TX])  + ((a)[R01]*(b)[TY]) + ((a)[R02]*(b)[TZ]) + (a)[TX]; \
//  (ab)[TY] = ((a)[R10]*(b)[TX])  + ((a)[R11]*(b)[TY]) + ((a)[R12]*(b)[TZ]) + (a)[TY]; \
//  (ab)[TZ] = ((a)[R20]*(b)[TX])  + ((a)[R21]*(b)[TY]) + ((a)[R22]*(b)[TZ]) + (a)[TZ]

void TR_MULT(Matrix4& ab, const Matrix4& a, const Matrix4& b)
{
    ab = a * b;
    (ab)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] =
    ((a)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] *
     (b)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]) +
    ((a)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] *
     (b)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second]) +
    ((a)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *
     (b)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second]) +
     (a)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second];

    (ab)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] =
    ((a)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
     (b)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]) +
    ((a)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
     (b)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second]) +
    ((a)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
     (b)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second]) +
     (a)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second];

    (ab)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] =
    ((a)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] *
     (b)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]) +
    ((a)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] *
     (b)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second]) +
    ((a)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *
     (b)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second]) +
     (a)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second];

    ab[matrixHelperIndices[HOMOGENEOUS_LINE_4].second.first][matrixHelperIndices[HOMOGENEOUS_LINE_4].second.second] = 1.0f;
}

void TR_INV_MULT(Matrix4& ab, const Matrix4& a, const Matrix4& b)
{
    (ab)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] =
     (a)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] *
     (b)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] +
     (a)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
     (b)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] +
     (a)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] *
     (b)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second];

    (ab)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] =
     (a)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] *
     (b)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] +
     (a)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
     (b)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] +
     (a)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] *
     (b)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second];

    (ab)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] =
     (a)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *
     (b)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] +
     (a)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
     (b)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] +
     (a)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *
     (b)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second];

    (ab)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] =
     (a)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] *
     (b)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] +
     (a)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
     (b)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] +
     (a)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] *
     (b)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second];

    (ab)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] =
     (a)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] *
     (b)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] +
     (a)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
     (b)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] +
     (a)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] *
     (b)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second];

    (ab)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] =
     (a)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *
     (b)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] +
     (a)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
     (b)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] +
     (a)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *
     (b)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second];

    (ab)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] =
    ((ab)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
    (ab)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second]) -
    ((ab)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] *
    (ab)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second]);

    (ab)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] =
    ((ab)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] *
    (ab)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second]) -
    ((ab)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] *
    (ab)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second]);

    (ab)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] =
    ((ab)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] *
    (ab)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second]) -
    ((ab)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
    (ab)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second]);

    (ab)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]  =
     (a)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] *
    ((b)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] -
     (a)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]) +
     (a)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] *
    ((b)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] -
     (a)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second]) +
     (a)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] *
    ((b)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] -
     (a)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second]);

    (ab)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] =
     (a)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] *
    ((b)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] -
     (a)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]) +
     (a)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] *
    ((b)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] -
     (a)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second]) +
     (a)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] *
    ((b)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] -
     (a)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second]);

    (ab)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] =
     (a)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] *
    ((b)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second] -
     (a)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]) +
     (a)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] *
    ((b)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second] -
     (a)[matrixHelperIndices[TY].second.first][matrixHelperIndices[TY].second.second]) +
     (a)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] *
    ((b)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second] -
     (a)[matrixHelperIndices[TZ].second.first][matrixHelperIndices[TZ].second.second]);

    ab[matrixHelperIndices[HOMOGENEOUS_LINE_4].second.first][matrixHelperIndices[HOMOGENEOUS_LINE_4].second.second] = 1.0f;
}

// Acknowledgement:
//   The following macros are from Pierre Terdiman's
//   Opcode library, http://www.codercorner.com/Opcode.htm

// macro to find the min and max among three variables
void FINDMINMAX(double x0, double x1, double x2, double& min, double& max)
{
    min = max = x0 ;
    if (x1 < min)      min = x1;
    else if (x1 > max) max = x1;
    if (x2 < min)      min = x2;
    else if (x2 > max) max = x2;
}

//============================================================================


bool AXISTEST_X01(const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& dim, double a, double b, double fa, double fb, double& min, double& max, double& rad)
{
    min = a * v0[1] - b * v0[2];
    max = a * v2[1] - b * v2[2];
    if (min > max) { std::swap(min, max); }
    rad = fa * dim[1] + fb * dim[2];
    if (min > rad || max < -rad)
        return false;

    return true;
}
//============================================================================


bool AXISTEST_X2(const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& dim, double a, double b, double fa, double fb, double& min, double& max, double& rad)
{
    min = a * v0[1] - b * v0[2];
    max = a * v1[1] - b * v1[2];
    if (min > max) { std::swap(min, max); }
    rad = fa * dim[1] + fb * dim[2];
    if (min > rad || max < -rad)
        return false;

    return true;
}
//============================================================================


bool AXISTEST_Y02(const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& dim, double a, double b, double fa, double fb, double& min, double& max, double& rad)
{
    min = b * v0[2] - a * v0[0];
    max = b * v2[2] - a * v2[0];
    if (min > max) { std::swap(min, max); }
    rad = fa * dim[0] + fb * dim[2];
    if (min > rad || max < -rad)
        return false;

    return true;
}
//============================================================================


bool AXISTEST_Y1(const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& dim, double a, double b, double fa, double fb, double& min, double& max, double& rad)
{
    min = b * v0[2] - a * v0[0];
    max = b * v1[2] - a * v1[0];
    if (min > max) { std::swap(min, max); }
    rad = fa * dim[0] + fb * dim[2];
    if (min > rad || max < -rad)
        return false;

    return true;
}
//============================================================================


bool AXISTEST_Z12(const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& dim, double a, double b, double fa, double fb, double& min, double& max, double& rad)
{
    min = a * v1[0] - b * v1[1];
    max = a * v2[0] - b * v2[1];
    if (min > max) { std::swap(min, max); }
    rad = fa * dim[0] + fb * dim[1];
    if (min > rad || max < -rad)
        return false;

    return true;
}
//============================================================================


bool AXISTEST_Z0(const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& dim, double a, double b, double fa, double fb, double& min, double& max, double& rad)
{
    min = a * v0[0] - b * v0[1];
    max = a * v1[0] - b * v1[1];
    if (min > max) { std::swap(min, max); }
    rad = fa * dim[0] + fb * dim[1];
    if (min > rad || max < -rad)
        return false;

    return true;
}
//============================================================================


// compute triangle edges
// - edges lazy evaluated to take advantage of early exits
// - fabs precomputed (half less work, possible since extents are always >0)
// - customized macros to take advantage of the null component
// - axis vector discarded, possibly saves useless movs
#define IMPLEMENT_CLASS3_TESTS                  \
    Real rad;                                   \
                                                \
    const Real fey0 = FABS(e0[1]);              \
    const Real fez0 = FABS(e0[2]);              \
    AXISTEST_X01(e0[2], e0[1], fez0, fey0);     \
    const Real fex0 = FABS(e0[0]);              \
    AXISTEST_Y02(e0[2], e0[0], fez0, fex0);     \
    AXISTEST_Z12(e0[1], e0[0], fey0, fex0);     \
                                                \
    const Real fey1 = FABS(e1[1]);              \
    const Real fez1 = FABS(e1[2]);              \
    AXISTEST_X01(e1[2], e1[1], fez1, fey1);     \
    const Real fex1 = FABS(e1[0]);              \
    AXISTEST_Y02(e1[2], e1[0], fez1, fex1);     \
    AXISTEST_Z0(e1[1], e1[0], fey1, fex1);      \
                                                \
    Vector3 e2;                                 \
    VEC_SUB(e2, v0, v2);                        \
    const Real fey2 = FABS(e2[1]);              \
    const Real fez2 = FABS(e2[2]);              \
    AXISTEST_X2(e2[2], e2[1], fez2, fey2);      \
    const Real fex2 = FABS(e2[0]);              \
    AXISTEST_Y1(e2[2], e2[0], fez2, fex2);      \
    AXISTEST_Z12(e2[1], e2[0], fey2, fex2)


//! Computes only the x-coordinate when applying the transformation matrix \a t to
//! the vector \a v. The resulting coordinate is put in \a vt.
//! \param[out] vt Vector containing the transformed x-coordinate
//! \param[in]  v  A vector
//! \param[in]  t  A transformation matrix
void TRANSFORM_VEC_X(Vector3& vt, const Matrix4& t, const Vector3& v)
{
  ((vt)[0] = ((t)[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] * (v)[0]) +
             ((t)[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] * (v)[1]) +
             ((t)[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] * (v)[2]) +
             (t)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TX].second.second]);
}
//! Computes only the y-coordinate when applying the transformation matrix \a t to
//! the vector \a v. The resulting coordinate is put in \a vt.
//! \param[out] vt Vector containing the transformed y-coordinate
//! \param[in]  v  A vector
//! \param[in]  t  A transformation matrix
void TRANSFORM_VEC_Y(Vector3& vt, const Matrix4& t, const Vector3& v)
{
    ((vt)[1] = ((t)[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] * (v)[0]) +
               ((t)[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] * (v)[1]) +
               ((t)[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] * (v)[2]) +
               (t)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TY].second.second]);
}
//! Computes only the z-coordinate when applying the transformation matrix \a t to
//! the vector \a v. The resulting coordinate is put in \a vt.
//! \param[out] vt Vector containing the transformed z-coordinate
//! \param[in]  v  A vector
//! \param[in]  t  A transformation matrix
void TRANSFORM_VEC_Z(Vector3& vt, const Matrix4& t, const Vector3& v)
{
    ((vt)[2] = ((t)[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] * (v)[0]) +
               ((t)[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] * (v)[1]) +
               ((t)[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] * (v)[2]) +
               (t)[matrixHelperIndices[TX].second.first][matrixHelperIndices[TZ].second.second]);
}
//============================================================================

bool PlaneBoxOverlap(const Vector3& normal, const double& d, const Vector3& maxbox)
{
  Vector3 vmin(-maxbox[0], -maxbox[1], -maxbox[2]);
  Vector3 vmax(maxbox[0],  maxbox[1],  maxbox[2]);

  if (normal[0] <= 0.0f) { std::swap(vmin[0], vmax[0]); }
  if (normal[1] <= 0.0f) { std::swap(vmin[1], vmax[1]); }
  if (normal[2] <= 0.0f) { std::swap(vmin[2], vmax[2]); }

  return ((normal * vmin) + d <= 0.0f) && ((normal * vmax) + d >= 0.0f);
}


#define TRIBOXCOLLIDES_USE_CLASS_3_TESTS
// Returns true if the triangle defined by tri_verts intersects with the axis-aligned
// box with the dimensions dim.
// Note: The function transforms the coordinates of the triangle lazily, therefore
// the parameter tri_rel_box is needed.
bool TriBoxCollides(const Vector3 tri_verts[],
                    const Matrix4& tri_rel_box,
                    const Vector3& dim)
{
    // Use separating axis theorem to test overlap between triangle and box .
    // We need to test for overlap in these directions:
    // 1) the {x,y,z}-directions
    // 2) normal of the triangle
    // 3) crossproduct(edge from tri, {x,y,z}-directin)
    //    this gives 3x3 = 9 more tests

    // transformed triangle vertices (computed lazily)
    Vector3 v0;
    Vector3 v1;
    Vector3 v2;

    // First, test overlap in the {x,y,z}-directions
    double min, max;

    // Test Z-direction first (we know the boxes are thinnest in that direction)
    /*TRANSFORM_VEC_Z(v0, tri_rel_box, tri_verts[0]);
    TRANSFORM_VEC_Z(v1, tri_rel_box, tri_verts[1]);
    TRANSFORM_VEC_Z(v2, tri_rel_box, tri_verts[2]);*/

    FINDMINMAX(v0[2], v1[2], v2[2], min, max);
    if (min > dim[2] || max < -dim[2])
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   z-direction test: no overlap" << std::endl;
#endif
        return false;
    }
    // Test Y-direction
    /*TRANSFORM_VEC_Y(v0, tri_rel_box, tri_verts[0]);
    TRANSFORM_VEC_Y(v1, tri_rel_box, tri_verts[1]);
    TRANSFORM_VEC_Y(v2, tri_rel_box, tri_verts[2]);*/

    FINDMINMAX(v0[1], v1[1], v2[1], min, max);
    if (min > dim[1] || max < -dim[1])
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   y-direction test: no overlap" << std::endl;
#endif
        return false;
    }

    // Test X-direction
    /*TRANSFORM_VEC_X(v0, tri_rel_box, tri_verts[0]);
    TRANSFORM_VEC_X(v1, tri_rel_box, tri_verts[1]);
    TRANSFORM_VEC_X(v2, tri_rel_box, tri_verts[2]);*/

    FINDMINMAX(v0[0], v1[0], v2[0], min, max);
    if (min > dim[0] || max < -dim[0])
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   x-direction test: no overlap" << std::endl;
#endif
        return false;
    }

    Vector3 e0 = v1 - v0;
    Vector3 e1 = v2 - v1;

    // 3) "Class III" tests
#ifdef TRIBOXCOLLIDES_USE_CLASS_3_TESTS
    double rad;

    bool axisTestResult;
    const double fey0 = std::fabs(e0[1]);
    const double fez0 = std::fabs(e0[2]);
    axisTestResult = AXISTEST_X01(v0, v1, v2, dim, e0[2], e0[1], fez0, fey0, min, max, rad);

    if (!axisTestResult)
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   AXISTEST_X01 1: no overlap" << std::endl;
#endif
        return false;
    }

    const double fex0 = std::fabs(e0[0]);
    axisTestResult = AXISTEST_Y02(v0, v1, v2, dim, e0[2], e0[0], fez0, fex0, min, max, rad);
    if (!axisTestResult)
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   AXISTEST_Y02: no overlap" << std::endl;
#endif
        return false;
    }

    axisTestResult = AXISTEST_Z12(v0, v1, v2, dim, e0[1], e0[0], fey0, fex0, min, max, rad);
    if (!axisTestResult)
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   AXISTEST_Z12 1: no overlap" << std::endl;
#endif
        return false;
    }

    const double fey1 = std::fabs(e1[1]);
    const double fez1 = std::fabs(e1[2]);
    axisTestResult = AXISTEST_X01(v0, v1, v2, dim, e1[2], e1[1], fez1, fey1, min, max, rad);
    if (!axisTestResult)
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   AXISTEST_X01 2: no overlap" << std::endl;
#endif
        return false;
    }

    const double fex1 = std::fabs(e1[0]);
    axisTestResult = AXISTEST_Y02(v0, v1, v2, dim, e1[2], e1[0], fez1, fex1, min, max, rad);
    if (!axisTestResult)
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   AXISTEST_Y02: no overlap" << std::endl;
#endif
        return false;
    }

    axisTestResult = AXISTEST_Z0(v0, v1, v2, dim, e1[1], e1[0], fey1, fex1, min, max, rad);
    if (!axisTestResult)
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   AXISTEST_Z0: no overlap" << std::endl;
#endif
        return false;
    }

    Vector3 e2 = v0 - v2;
    const double fey2 = std::fabs(e2[1]);
    const double fez2 = std::fabs(e2[2]);
    axisTestResult = AXISTEST_X2(v0, v1, v2, dim, e2[2], e2[1], fez2, fey2, min, max, rad);
    if (!axisTestResult)
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   AXISTEST_X2: no overlap" << std::endl;
#endif
        return false;
    }

    const double fex2 = std::fabs(e2[0]);
    axisTestResult = AXISTEST_Y1(v0, v1, v2, dim, e2[2], e2[0], fez2, fex2, min, max, rad);
    if (!axisTestResult)
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   AXISTEST_Y1: no overlap" << std::endl;
#endif
        return false;
    }

    axisTestResult = AXISTEST_Z12(v0, v1, v2, dim, e2[1], e2[0], fey2, fex2, min, max, rad);
    if (!axisTestResult)
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   AXISTEST_Z12: no overlap" << std::endl;
#endif
        return false;
    }
#endif
    // 2) Test if the box intersects the plane of the triangle
    Vector3 nrml = e0.cross(e1);
    const double d = -nrml * v0;

    bool planeBoxResult = PlaneBoxOverlap(nrml, d, dim);
    if (!planeBoxResult)
    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "   PlaneBoxOverlap: no overlap" << std::endl;
#endif
        return false;
    }
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << "  TriBoxCollide: overlap" << std::endl;
#endif
    return true;
}

/*
bool TriangleAABBIntersection(const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& aabbMin, const Vector3& aabbMax)
{
    double p0, p1, p2, r;

    Vector3 c = (aabbMin + aabbMax) * 0.5f;
    double e0 = (aabbMax.x() - aabbMin.x()) * 0.5f;
    double e1 = (aabbMax.y() - aabbMin.y()) * 0.5f;
    double e2 = (aabbMax.z() - aabbMin.z()) * 0.5f;

    Vector3 v0_orig = v0 - c;
    Vector3 v1_orig = v1 - c;
    Vector3 v2_orig = v2 - c;

    Vector3 f0 = v1 - v0;
    Vector3 f1 = v2 - v1;
    Vector3 f2 = v0 - v2;


}
*/

void TRANSFORM_TRIANGLE(const Matrix4& t, Vector3& vert0, Vector3& vert1, Vector3& vert2)
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
void SORT(double& a, double& b)
{
    if ((a) > (b))
    {
        const double c = (a);
        (a) = (b);
        (b) = c;
    }
}

//! Edge to edge test based on Franlin Antonio's gem: "Faster Line Segment Intersection", in Graphics Gems III, pp. 199-202
bool EDGE_EDGE_TEST(const Vector3& V0, const Vector3& U0, const Vector3& U1, const unsigned short& i0, const unsigned short& i1, double& Ax, double& Ay, double& Bx, double& By, double& Cx, double& Cy, double& d, double& f)
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
bool EDGE_AGAINST_TRI_EDGES(const Vector3& V0, const Vector3& V1, const Vector3& U0, const Vector3& U1, const Vector3& U2, const unsigned short& i0, const unsigned short& i1)
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
bool POINT_IN_TRI(const Vector3& V0, const Vector3& U0, const Vector3& U1, const Vector3& U2, const unsigned short& i0, const unsigned short& i1)
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
bool CoplanarTriTri(const Vector3& n,
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
bool NEWCOMPUTE_INTERVALS(double& VV0, double& VV1, double& VV2, double& D0, double& D1, double& D2, double& D0D1, double& D0D2, double& A, double& B, double& C, double& X0, double& X1)
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

bool tri_tri_overlap_3d(const Vector3 t1[], const Vector3 t2[])
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

void
MxVpV(Vector3& Vr, const Matrix4& M1, const Vector3& V1, const Vector3& V2)
{
  Vr[0] = (M1[matrixHelperIndices[R00].second.first][matrixHelperIndices[R00].second.second] * V1[0] +
           M1[matrixHelperIndices[R01].second.first][matrixHelperIndices[R01].second.second] * V1[1] +
           M1[matrixHelperIndices[R02].second.first][matrixHelperIndices[R02].second.second] * V1[2] +
           V2[0]);
  Vr[1] = (M1[matrixHelperIndices[R10].second.first][matrixHelperIndices[R10].second.second] * V1[0] +
           M1[matrixHelperIndices[R11].second.first][matrixHelperIndices[R11].second.second] * V1[1] +
           M1[matrixHelperIndices[R12].second.first][matrixHelperIndices[R12].second.second] * V1[2] +
           V2[1]);
  Vr[2] = (M1[matrixHelperIndices[R20].second.first][matrixHelperIndices[R20].second.second] * V1[0] +
           M1[matrixHelperIndices[R21].second.first][matrixHelperIndices[R21].second.second] * V1[1] +
           M1[matrixHelperIndices[R22].second.first][matrixHelperIndices[R22].second.second] * V1[2] +
           V2[2]);
}

#define OBBTREE_TEST_OVERLAP_USE_TRIBOX_COLLIDES
//#define OBBTREE_TEST_OVERLAP_VERBOSE
bool ObbTree::testOverlap(ObbTree &tree2, CollideResult& res)
{
    m_intersectingOBBs.clear();
    m_intersectingOBBIndices.clear();
    m_testedTriangles.clear();
    m_intersectingTriangles.clear();

    Matrix4 x_form, b_rel_a, a_rel_b;
    Matrix4 ta(this->getWorldOrientation()), tb(tree2.getWorldOrientation());

    ta[0][3] = getWorldPosition().x(); ta[1][3] = getWorldPosition().y(); ta[2][3] = getWorldPosition().z();
    tb[0][3] = tree2.getWorldPosition().x(); tb[1][3] = tree2.getWorldPosition().y(); tb[2][3] = tree2.getWorldPosition().z();
    ta[3][3] = tb[3][3] = 1;
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << "Matrices ta = " << ta << ", tb = " << tb << std::endl;
#endif
    TINV_MUL_T(b_rel_a, ta, tb);
    //b_rel_a.invert(a_rel_b);
    TRANSFORM_INV(a_rel_b,  b_rel_a);
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << "Matrices b_rel_a = " << b_rel_a << ", a_rel_b = " << a_rel_b << std::endl;
#endif
    Matrix4 obbARelTop(getOrientation());
    obbARelTop[3][3] = 1;
    Matrix4 obbBRelTop(tree2.getOrientation());
    obbBRelTop[3][3] = 1;
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << "Matrices obbARelTop = " << obbARelTop << ", obbBRelTop = " << obbBRelTop << std::endl;
#endif
    obbARelTop[0][3] = getPosition().x(); obbARelTop[1][3] = getPosition().y(); obbARelTop[2][3] = getPosition().z();
    obbBRelTop[0][3] = tree2.getPosition().x(); obbBRelTop[1][3] = tree2.getPosition().y(); obbBRelTop[2][3] = tree2.getPosition().z();
    // x_form = b_rel_a * obbBRelTop;
    TR_MULT(x_form, b_rel_a, obbBRelTop);
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << "Matrix x_form = " << x_form << std::endl;
#endif
    bool disjoint = OBBDisJoint(obbARelTop, x_form, getHalfExtents(), tree2.getHalfExtents());
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << "Root OBB's disjoint: " << disjoint << std::endl;
#endif
    if (disjoint)
        return false;

    //m_intersectingOBBs.insert(std::make_pair(tree2.identifier(), std::make_pair(0, std::vector<std::string>())));
    m_intersectingOBBs[tree2.identifier()][0].push_back(tree2.getObbNodes().at(0).identifier());

    TINV_MUL_T(b_rel_a, ta, tb);
    TRANSFORM_INV(a_rel_b, b_rel_a);

    Vector3 tri_verts_a[3];
    Vector3 tri_verts_b[3];
    NodePair pair;

    int tri_a, tri_b;
    sofa::core::topology::BaseMeshTopology::Triangle triangle_a, triangle_b;

    pair.node1 = 0;
    pair.node2 = 0;
    pair.type = PAIR_OBB_OBB;

    res.Clear();

    BVHNodeType pair_trinode_obb = PAIR_TRINODE_OBB;
    BVHNodeType pair_obb_trinode = PAIR_OBB_TRINODE;
    BVHNodeType pair_trinode_tri = PAIR_TRINODE_TRI;
    BVHNodeType pair_tri_trinode = PAIR_TRI_TRINODE;

    // a collision model cannot have zero OBBs and zero TriNodes at the same time!
    if (m_obbNodes.size() == 0)
    {
        pair_trinode_obb = PAIR_TRI_OBB;
        pair_trinode_tri = PAIR_TRI_TRI;
    }
    if (tree2.getObbNodes().size() == 0)
    {
        pair_obb_trinode = PAIR_OBB_TRI;
        pair_tri_trinode = PAIR_TRI_TRI;
    }

    if (m_obbNodes.size() == 0)
    {
        if (tree2.getObbNodes().size() == 0)
        {
            pair.type = PAIR_TRINODE_TRINODE;
        }
        else
        {
            pair.type = PAIR_TRINODE_OBB;
        }
    }
    else if (tree2.getObbNodes().size() == 0)
    {
        pair.type = PAIR_OBB_TRINODE;
    }

    NodeStack& stack = *res.stack;
    stack.Clear();

    std::map<std::pair<std::string, std::string>, bool> checkedObbs;
    std::vector<std::pair<int, int> > checkedTriPairs;
    std::vector<std::pair<int, int> > intersectingTriPairs;

    int numIterations = 0;
    int maxIterations = 500000;

    TriIntersectPQP intersectPqp;

    const Vec3Types::VecCoord& x = m_state->read(core::ConstVecCoordId::position())->getValue();
    const Vec3Types::VecCoord& x2 = tree2.getMState()->read(core::ConstVecCoordId::position())->getValue();

    do
    {
        const int id1 = pair.node1;
        const int id2 = pair.node2;
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "== Iteration " << numIterations << ": " << id1 << " <-> " << id2 << " of type " <<  BVHNodeTypeName(pair.type) << std::endl;
#endif
        switch (pair.type)
        {
            case PAIR_OBB_OBB: // the most common case
            {
                const ObbVolume& a_obb = m_obbNodes.at(id1);
                const ObbVolume& b_obb = tree2.getObbNodes().at(id2);

#ifdef OBBTREE_TEST_OVERLAP_MANUAL_T_REL_TOP
                Matrix4 obbARelToTop(a_obb.getOrientation()/*.transposed()*/);
                obbARelToTop[3][3] = 1;
                obbARelToTop[0][3] = a_obb.getPosition().x(); obbARelToTop[1][3] = a_obb.getPosition().y(); obbARelToTop[2][3] = a_obb.getPosition().z();

                Matrix4 obbBRelToTop(b_obb.getOrientation()/*.transposed()*/);
                obbBRelToTop[3][3] = 1;
                obbBRelToTop[0][3] = b_obb.getPosition().x(); obbBRelToTop[1][3] = b_obb.getPosition().y(); obbBRelToTop[2][3] = b_obb.getPosition().z();

                TR_MULT(x_form, b_rel_a, obbBRelToTop);
#endif
                TR_MULT(x_form, b_rel_a, b_obb.t_rel_top);

                //assert(is_rot_matrix(x_form));

                ++res.num_bv_tests;
                bool obbDisjoint = OBBDisJoint(a_obb.t_rel_top, x_form, a_obb.getHalfExtents(), b_obb.getHalfExtents());
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                std::cout << " OBB vs. OBB test: " << a_obb.identifier() << " - " << b_obb.identifier() << ": disjoint = " << obbDisjoint << std::endl;
#endif

#ifdef OBBTREE_TEST_OVERLAP_MANUAL_T_REL_TOP
                std::cout << "  obbARelToTop = " << obbARelToTop << ", obbBRelToTop = " << obbBRelToTop << std::endl;
#endif
                checkedObbs.insert(std::make_pair(std::make_pair(a_obb.identifier(), b_obb.identifier()), obbDisjoint));

                if (!obbDisjoint)
                {
                    /*if (m_intersectingOBBs.find(tree2.identifier()) == m_intersectingOBBs.end())
                    {
                        //m_intersectingOBBs.insert(std::make_pair(tree2.identifier(), std::make_pair(id1, std::vector<std::string>())));
                    }*/

                    m_intersectingOBBs[tree2.identifier()][id1].push_back(b_obb.identifier());
                    //m_intersectingOBBIndices[tree2.identifier()][id1].push_back(id2);

                    m_intersectingOBBIndices.push_back(id1);

                    if (a_obb.getSize() > b_obb.getSize())
                    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                        std::cout << "  visit " << a_obb.identifier() << " first: " << a_obb.getSize() << " > " << b_obb.getSize() << std::endl;
#endif
                        // visit the children of 'a' first
                        pair.node1 = a_obb.getFirstChild();

                        stack.Push(pair);
                        ++pair.node1;
                        if (pair.node1 <= 0)
                        {
                            pair.type  =  pair_trinode_obb;
                            pair.node1 = -pair.node1;
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                            std::cout << " node1 is TriNode = " << pair.node1 << std::endl;
#endif
                            stack.Pop(1); // undo
                        }
                    }
                    else
                    {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                        std::cout << "  visit " << b_obb.identifier() << " first: " << a_obb.getSize() << " <= " << b_obb.getSize() << std::endl;
#endif
                        // visit the children of 'b' first
                        pair.node2 = b_obb.getFirstChild();

                        stack.Push(pair);
                        ++pair.node2;
                        if (pair.node2 <= 0)
                        {
                            pair.type  =  pair_obb_trinode;
                            pair.node2 = -pair.node2;
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                            std::cout << " node2 is TriNode = " << pair.node2 << std::endl;
#endif
                            stack.Pop(1); // undo
                        }
                    }

                    numIterations++;
                    if (numIterations >= maxIterations)
                        break;

                    continue;
                }
            }
            break;
#if 1
            case PAIR_TRINODE_OBB:
            {
                if (id2 >= 0 && id2 < tree2.getObbNodes().size() &&
                    id1 >= 0 && id1 < m_triNodes.size())
                {
                    const ObbVolume& b_obb = tree2.getObbNodes().at(id2);

                    // assert(id2 >= 0 && (unsigned int)id2 < b.num_obbs && id1 >= 0 && (unsigned int)id1 < a.num_tri_nodes);

#ifdef OBBTREE_TEST_OVERLAP_MANUAL_T_REL_TOP
                    Matrix4 obbBRelToTop(b_obb.getOrientation()/*.transposed()*/);
                    obbBRelToTop[3][3] = 1;
                    obbBRelToTop[0][3] = b_obb.getPosition().x(); obbBRelToTop[1][3] = b_obb.getPosition().y(); obbBRelToTop[2][3] = b_obb.getPosition().z();
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                    std::cout << " b_obb position = " << b_obb.getPosition() << std::endl;
                    std::cout << " obbBRelToTop = " << obbBRelToTop << std::endl;
#endif
                    TR_INV_MULT(x_form, obbBRelToTop, a_rel_b);
                    // assert(is_rot_matrix(x_form));
#endif
                    TR_INV_MULT(x_form, b_obb.t_rel_top, a_rel_b);

                    tri_a = m_triNodes.at(id1).getFirstChild();
                    if (tri_a >= 0 && tri_a < m_topology->getNbTriangles())
                    {
                        // assert(tri_a >= 0 && (unsigned int)tri_a < a.tri_mesh->NumTriangles());
                        // a.tri_mesh->GetTriangle(tri_a, tri_verts_a);

                        triangle_a = m_topology->getTriangle(tri_a);

                        tri_verts_a[0] = x[triangle_a[0]];
                        tri_verts_a[1] = x[triangle_a[1]];
                        tri_verts_a[2] = x[triangle_a[2]];

                        ++res.num_tri_box_tests;
#ifdef OBBTREE_TEST_OVERLAP_USE_TRIBOX_COLLIDES
                        if (TriBoxCollides(tri_verts_a, x_form, b_obb.getHalfExtents()))
#endif
                        {
                            pair.type  = PAIR_TRI_OBB;
                            pair.node1 = tri_a;
                            pair.node2 = b_obb.getFirstChild();

                            stack.Push(pair);
                            ++pair.node2;
                            stack.Push(pair);

                            if (pair.node2 <= 0)
                            {
                                pair.type  =  pair_tri_trinode;
                                pair.node2 = -pair.node2;
                                stack.Pop(2); // undo
                                stack.Push(pair);
                            }
                        }
                    }

                    //tri_a = a.tri_nodes[id1].tri2;
                    tri_a = m_triNodes.at(id1).getSecondChild();
                    if (tri_a >= 0 && tri_a < m_topology->getNbTriangles())
                    {
                        //assert(tri_a == -1 || (unsigned int)tri_a < a.tri_mesh->NumTriangles());

                        if (tri_a >= 0)
                        {
                            // assert(tri_a >= 0 && (unsigned int)tri_a < a.tri_mesh->NumTriangles());
                            // a.tri_mesh->GetTriangle(tri_a, tri_verts_a);

                            triangle_a = m_topology->getTriangle(tri_a);

                            tri_verts_a[0] = x[triangle_a[0]];
                            tri_verts_a[1] = x[triangle_a[1]];
                            tri_verts_a[2] = x[triangle_a[2]];

                            ++res.num_tri_box_tests;
#ifdef OBBTREE_TEST_OVERLAP_USE_TRIBOX_COLLIDES
                            if (TriBoxCollides(tri_verts_a, x_form, b_obb.getHalfExtents()))
#endif
                            {
                                pair.type  = PAIR_TRI_OBB;
                                pair.node1 = tri_a;
                                pair.node2 = b_obb.getFirstChild();

                                stack.Push(pair);
                                ++pair.node2;
                                stack.Push(pair);

                                if (pair.node2 <= 0)
                                {
                                    pair.type  =  pair_tri_trinode;
                                    pair.node2 = -pair.node2;
                                    stack.Pop(2); // undo
                                    stack.Push(pair);
                                }
                            }
                        }
                    }
                }
            }
            break;
#endif
#if 1
            case PAIR_OBB_TRINODE:
            {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                std::cout << " PAIR_OBB_TRINODE: " << id1 << " - " << id2 << std::endl;
#endif
                if (id1 >= 0 && id1 < m_obbNodes.size() &&
                    id2 >= 0 && id2 < tree2.getTriNodes().size())
                {
                    const ObbVolume& a_obb = m_obbNodes.at(id1);

                    //assert(id1 >= 0 && (unsigned int)id1 < a.num_obbs && id2 >= 0 && (unsigned int)id2 < b.num_tri_nodes);

#ifdef OBBTREE_TEST_OBBTREE_TEST_OVERLAP_MANUAL_T_REL_TOP
                    Matrix4 obbARelToTop(a_obb.getOrientation()/*.transposed()*/);
                    obbARelToTop[3][3] = 1;
                    obbARelToTop[0][3] = a_obb.getPosition().x(); obbARelToTop[1][3] = a_obb.getPosition().y(); obbARelToTop[2][3] = a_obb.getPosition().z();
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                    std::cout << " a_obb position = " << a_obb.getPosition() << std::endl;
                    std::cout << " obbARelToTop = " << obbARelToTop << std::endl;
#endif
                    TR_INV_MULT(x_form, obbARelToTop, b_rel_a);
#endif
                    TR_INV_MULT(x_form, a_obb.t_rel_top, b_rel_a);
                    //assert(is_rot_matrix(x_form));

                    //tri_b = b.tri_nodes[id2].tri1;
                    //assert(tri_b >= 0 && (unsigned int)tri_b < b.tri_mesh->NumTriangles());
                    //b.tri_mesh->GetTriangle(tri_b, tri_verts_b);

                    tri_b = tree2.getTriNodes().at(id2).getFirstChild();
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                    std::cout << " tri_b (1) = " << tri_b << std::endl;
#endif
                    if (tri_b >= 0 && tri_b < tree2.getTopology()->getNbTriangles())
                    {
                        triangle_b = tree2.getTopology()->getTriangle(tri_b);

                        tri_verts_b[0] = x2[triangle_b[0]];
                        tri_verts_b[1] = x2[triangle_b[1]];
                        tri_verts_b[2] = x2[triangle_b[2]];

                        ++res.num_tri_box_tests;
#ifdef OBBTREE_TEST_OVERLAP_USE_TRIBOX_COLLIDES
                        if (TriBoxCollides(tri_verts_b, x_form, a_obb.getHalfExtents()))
#endif
                        {
                            pair.type  = PAIR_OBB_TRI;
                            pair.node1 = a_obb.getFirstChild();
                            pair.node2 = tri_b;

                            stack.Push(pair);
                            ++pair.node1;
                            stack.Push(pair);

                            if (pair.node1 <= 0)
                            {
                                pair.type  =  pair_trinode_tri;
                                pair.node1 = -pair.node1;
                                stack.Pop(2); // undo
                                stack.Push(pair);
                            }
                        }
                    }

                    //tri_b = b.tri_nodes[id2].tri2;
                    //assert(tri_b < 0 || (unsigned int)tri_b < b.tri_mesh->NumTriangles());

                    tri_b = tree2.getTriNodes().at(id2).getSecondChild();
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                    std::cout << " tri_b (2) = " << tri_b << std::endl;
#endif
                    if (tri_b < 0 || tri_b < tree2.getTopology()->getNbTriangles())
                    {
                        if (tri_b >= 0)
                        {
                            //b.tri_mesh->GetTriangle(tri_b, tri_verts_b);
                            triangle_b = tree2.getTopology()->getTriangle(tri_b);

                            tri_verts_b[0] = x2[triangle_b[0]];
                            tri_verts_b[1] = x2[triangle_b[1]];
                            tri_verts_b[2] = x2[triangle_b[2]];

                            ++res.num_tri_box_tests;
#ifdef OBBTREE_TEST_OVERLAP_USE_TRIBOX_COLLIDES
                            if (TriBoxCollides(tri_verts_b, x_form, a_obb.getHalfExtents()))
#endif
                            {
                                pair.type  = PAIR_OBB_TRI;
                                pair.node1 = a_obb.getFirstChild();
                                pair.node2 = tri_b;

                                stack.Push(pair);
                                ++pair.node1;
                                stack.Push(pair);

                                if (pair.node1 <= 0)
                                {
                                    pair.type  =  pair_trinode_tri;
                                    pair.node1 = -pair.node1;
                                    stack.Pop(2); // undo
                                    stack.Push(pair);
                                }
                            }
                        }
                    }
                }
            }
            break;
#endif
#if 1
            case PAIR_TRI_OBB:
            {
                //assert(id1 >= 0 && (unsigned int)id1 < a.tri_mesh->NumTriangles() && id2 >= 0 && (unsigned int)id2 < b.num_obbs);

                if (id1 >= 0 && id1 < m_topology->getNbTriangles() &&
                    id2 >= 0 && id2 < tree2.getObbNodes().size())
                {
#ifdef OBBTREE_TEST_OBBTREE_TEST_OVERLAP_MANUAL_T_REL_TOP
                    Matrix4 obbBRelToTop(tree2.getObbNodes()[id2].getOrientation()/*.transposed()*/);
                    obbBRelToTop[3][3] = 1;
                    obbBRelToTop[0][3] = tree2.getObbNodes()[id2].getPosition().x(); obbBRelToTop[1][3] = tree2.getObbNodes()[id2].getPosition().y(); obbBRelToTop[2][3] = tree2.getObbNodes()[id2].getPosition().z();

                    TR_INV_MULT(x_form, obbBRelToTop, a_rel_b);
#endif

                    TR_INV_MULT(x_form, tree2.getObbNodes()[id2].t_rel_top, a_rel_b);
                    //assert(is_rot_matrix(x_form));

                    //a.tri_mesh->GetTriangle(id1, tri_verts_a);
                    triangle_a = m_topology->getTriangle(id1);
                    tri_verts_a[0] = x[triangle_a[0]];
                    tri_verts_a[1] = x[triangle_a[1]];
                    tri_verts_a[2] = x[triangle_a[2]];

                    ++res.num_tri_box_tests;
#ifdef OBBTREE_TEST_OVERLAP_USE_TRIBOX_COLLIDES
                    if (TriBoxCollides(tri_verts_a, x_form, tree2.getObbNodes()[id2].getHalfExtents()))
#endif
                    {
                        pair.node2 = tree2.getObbNodes()[id2].getFirstChild();

                        stack.Push(pair);
                        ++pair.node2;
                        if (pair.node2 <= 0)
                        {
                            pair.type  =  pair_tri_trinode;
                            pair.node2 = -pair.node2;
                            stack.Pop(1); // undo
                        }
                        continue;
                    }
                }
            }
            break;

            case PAIR_OBB_TRI:
            {
                //assert(id1 >= 0 && (unsigned int)id1 < a.num_obbs && id2 >= 0 && (unsigned int)id2 < b.tri_mesh->NumTriangles());
                if (id1 >= 0 && id1 < m_obbNodes.size() &&
                    id2 < tree2.getTopology()->getNbTriangles())
                {
#ifdef OBBTREE_TEST_OBBTREE_TEST_OVERLAP_MANUAL_T_REL_TOP
                    Matrix4 obbARelToTop(m_obbNodes[id1].getOrientation()/*.transposed()*/);
                    obbARelToTop[3][3] = 1;
                    obbARelToTop[0][3] = m_obbNodes[id1].getPosition().x(); obbARelToTop[1][3] = m_obbNodes[id1].getPosition().y(); obbARelToTop[2][3] = m_obbNodes[id1].getPosition().z();

                    TR_INV_MULT(x_form, obbARelToTop, b_rel_a);
#endif
                    TR_INV_MULT(x_form, m_obbNodes[id1].t_rel_top, b_rel_a);
                    //assert(is_rot_matrix(x_form));

                    //b.tri_mesh->GetTriangle(id2, tri_verts_b);
                    triangle_b = tree2.getTopology()->getTriangle(id2);
                    tri_verts_b[0] = x2[triangle_b[0]];
                    tri_verts_b[1] = x2[triangle_b[1]];
                    tri_verts_b[2] = x2[triangle_b[2]];

                    ++res.num_tri_box_tests;
#ifdef OBBTREE_TEST_OVERLAP_USE_TRIBOX_COLLIDES
                    if (TriBoxCollides(tri_verts_b, x_form, m_obbNodes[id1].getHalfExtents()))
#endif
                    {
                        pair.node1 = m_obbNodes[id1].getFirstChild();

                        stack.Push(pair);
                        ++pair.node1;
                        if (pair.node1 <= 0)
                        {
                            pair.type  =  pair_trinode_tri;
                            pair.node1 = -pair.node1;
                            stack.Pop(1); // undo
                        }
                        continue;
                    }
                }
            }
            break;

            case PAIR_TRI_TRINODE:
            {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                std::cout << " PAIR_TRI_TRINODE: TriNode[" << id2 << "] child0 = " << tree2.getTriNodes().at(id2).getFirstChild() << ", child1 = " << tree2.getTriNodes().at(id2).getSecondChild() << std::endl;
#endif
                //assert((unsigned int)id2 < b.num_tri_nodes);
                if (id2 < tree2.getTriNodes().size())
                {
                    // a.tri_mesh->GetTriangle(id1, tri_verts_a);
                    triangle_a = m_topology->getTriangle(id1);
                    tri_verts_a[0] = x[triangle_a[0]];
                    tri_verts_a[1] = x[triangle_a[1]];
                    tri_verts_a[2] = x[triangle_a[2]];

                    // transform the triangle of object 'a' into the frame of object 'b'
                    //TRANSFORM_TRIANGLE(a_rel_b, tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                    //tri_b = b.tri_nodes[id2].tri1;
                    tri_b = tree2.getTriNodes().at(id2).getFirstChild();
                    //b.tri_mesh->GetTriangle(tri_b, tri_verts_b);
                    triangle_b = tree2.getTopology()->getTriangle(tri_b);
                    tri_verts_b[0] = x2[triangle_b[0]];
                    tri_verts_b[1] = x2[triangle_b[1]];
                    tri_verts_b[2] = x2[triangle_b[2]];

                    //DO_TRI_TRI_TEST(id1, tri_b);
                    ++res.num_tri_tests;
                    bool intersects = intersectPqp.intersect_tri_tri(tri_verts_a[0], tri_verts_a[1], tri_verts_a[2],
                                                   tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                    checkedTriPairs.push_back(std::make_pair(id1, tri_b));
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                    std::cout << " intersect triangles " << id1 << " - " << tri_b << ": " << intersects << std::endl;
#endif
                    if (intersects)
                    {
                        res.Add(id1, tri_b);
                        intersectingTriPairs.push_back(std::make_pair(id1, tri_b));
                        /*if (m_intersectingTriangles.find(tree2.identifier()) == m_intersectingTriangles.end())
                        {
                            m_intersectingTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(id1, std::vector<int>())));
                        }*/

                        m_intersectingTriangles[tree2.identifier()][id1].push_back(tri_b);

                        /*if (tree2.getIntersectingTriangles().find(m_identifier) == tree2.getIntersectingTriangles().end())
                            tree2.getIntersectingTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                        tree2.getIntersectingTriangles()[m_identifier][tri_b].push_back(id1);
                    }
                    else
                    {
                        /*if (m_testedTriangles.find(tree2.identifier()) == m_testedTriangles.end())
                            m_testedTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(id1, std::vector<int>())));*/

                        m_testedTriangles[tree2.identifier()][id1].push_back(tri_b);

                        /*if (tree2.getTestedTriangles().find(m_identifier) == tree2.getTestedTriangles().end())
                            tree2.getTestedTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                        tree2.getTestedTriangles()[m_identifier][tri_b].push_back(id1);
                    }

                    //tri_b = b.tri_nodes[id2].tri2;
                    tri_b = tree2.getTriNodes().at(id2).getSecondChild();
                    if (tri_b >= 0)
                    {
                        //b.tri_mesh->GetTriangle(tri_b, tri_verts_b);
                        triangle_b = tree2.getTopology()->getTriangle(tri_b);
                        tri_verts_b[0] = x2[triangle_b[0]];
                        tri_verts_b[1] = x2[triangle_b[1]];
                        tri_verts_b[2] = x2[triangle_b[2]];

                        //DO_TRI_TRI_TEST(id1, tri_b);
                        ++res.num_tri_tests;
                        bool intersects = intersectPqp.intersect_tri_tri(tri_verts_a[0], tri_verts_a[1], tri_verts_a[2],
                                                       tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                        checkedTriPairs.push_back(std::make_pair(id1, tri_b));
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                        std::cout << " intersect triangles " << id1 << " - " << tri_b << ": " << intersects << std::endl;
#endif
                        if (intersects)
                        {
                            res.Add(id1, tri_b);
                            intersectingTriPairs.push_back(std::make_pair(id1, tri_b));
                            /*if (m_intersectingTriangles.find(tree2.identifier()) == m_intersectingTriangles.end())
                                m_intersectingTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(id1, std::vector<int>())));*/

                            m_intersectingTriangles[tree2.identifier()][id1].push_back(tri_b);

                            /*if (tree2.getIntersectingTriangles().find(m_identifier) == tree2.getIntersectingTriangles().end())
                                tree2.getIntersectingTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                            tree2.getIntersectingTriangles()[m_identifier][tri_b].push_back(id1);
                        }
                        else
                        {
                            /*if (m_testedTriangles.find(tree2.identifier()) == m_testedTriangles.end())
                                m_testedTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(id1, std::vector<int>())));*/

                            m_testedTriangles[tree2.identifier()][id1].push_back(tri_b);

                            /*if (tree2.getTestedTriangles().find(m_identifier) == tree2.getTestedTriangles().end())
                                tree2.getTestedTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                            tree2.getTestedTriangles()[m_identifier][tri_b].push_back(id1);
                        }
                    }
                }
            }
            break;
#endif
#if 1
            case PAIR_TRINODE_TRI:
            {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                std::cout << " PAIR_TRINODE_TRI: TriNode[" << id1 << "] child0 = " << m_triNodes[id1].getFirstChild() << ", child1 = " << m_triNodes[id1].getSecondChild() << std::endl;
#endif
                //assert((unsigned int)id1 < a.num_tri_nodes);
                if (id1 < m_triNodes.size())
                {
                    // transform the triangle of object 'b' into the frame of object 'a'
                    //b.tri_mesh->GetTriangle(id2, tri_verts_b);
                    triangle_b = tree2.getTopology()->getTriangle(id2);
                    tri_verts_b[0] = x2[triangle_b[0]];
                    tri_verts_b[1] = x2[triangle_b[1]];
                    tri_verts_b[2] = x2[triangle_b[2]];
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                    std::cout << "  triangle_b = " << triangle_b << ": " << tri_verts_b[0] << "," << tri_verts_b[1] << "," << tri_verts_b[2] << std::endl;
#endif
                    //TRANSFORM_TRIANGLE(b_rel_a, tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                    //tri_a = a.tri_nodes[id1].tri1;
                    tri_a = m_triNodes[id1].getFirstChild();

                    //a.tri_mesh->GetTriangle(tri_a, tri_verts_a);

                    triangle_a = m_topology->getTriangle(tri_a);
                    tri_verts_a[0] = x[triangle_a[0]];
                    tri_verts_a[1] = x[triangle_a[1]];
                    tri_verts_a[2] = x[triangle_a[2]];

                    //DO_TRI_TRI_TEST(tri_a, id2);
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                    std::cout << "  triangle_a (1) = " << triangle_a << ": " << tri_verts_a[0] << "," << tri_verts_a[1] << "," << tri_verts_a[2] << std::endl;
#endif
                    ++res.num_tri_tests;
                    bool intersects = intersectPqp.intersect_tri_tri(tri_verts_a[0], tri_verts_a[1], tri_verts_a[2],
                                                   tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                    checkedTriPairs.push_back(std::make_pair(tri_a, id2));
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                    std::cout << " intersect triangles " << tri_a << " - " << id2 << ": " << intersects << std::endl;
#endif
                    if (intersects)
                    {
                        intersectingTriPairs.push_back(std::make_pair(tri_a, id2));
                        res.Add(tri_a, id2);
                        /*if (m_intersectingTriangles.find(tree2.identifier()) == m_intersectingTriangles.end())
                            m_intersectingTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                        m_intersectingTriangles[tree2.identifier()][tri_a].push_back(id2);

                        /*if (tree2.getIntersectingTriangles().find(m_identifier) == tree2.getIntersectingTriangles().end())
                            tree2.getIntersectingTriangles().insert(std::make_pair(m_identifier, std::make_pair(id2, std::vector<int>())));*/

                        tree2.getIntersectingTriangles()[m_identifier][id2].push_back(tri_a);
                    }
                    else
                    {
                        /*if (m_testedTriangles.find(tree2.identifier()) == m_testedTriangles.end())
                            m_testedTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                        m_testedTriangles[tree2.identifier()][tri_a].push_back(id2);

                        /*if (tree2.getTestedTriangles().find(m_identifier) == tree2.getTestedTriangles().end())
                            tree2.getTestedTriangles().insert(std::make_pair(m_identifier, std::make_pair(id2, std::vector<int>())));*/

                        tree2.getTestedTriangles()[m_identifier][id2].push_back(tri_a);
                    }

                    // tri_a = a.tri_nodes[id1].tri2;
                    tri_a = m_triNodes[id1].getSecondChild();
                    if (tri_a >= 0)
                    {

                        //a.tri_mesh->GetTriangle(tri_a, tri_verts_a);
                        triangle_a = m_topology->getTriangle(tri_a);
                        tri_verts_a[0] = x[triangle_a[0]];
                        tri_verts_a[1] = x[triangle_a[1]];
                        tri_verts_a[2] = x[triangle_a[2]];
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                        std::cout << "  triangle_a (2) = " << triangle_a << ": " << tri_verts_a[0] << "," << tri_verts_a[1] << "," << tri_verts_a[2] << std::endl;
#endif
                        //DO_TRI_TRI_TEST(tri_a, id2);
                        ++res.num_tri_tests;
                        bool intersects = intersectPqp.intersect_tri_tri(tri_verts_a[0], tri_verts_a[1], tri_verts_a[2],
                                                       tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                        checkedTriPairs.push_back(std::make_pair(tri_a, id2));
                        std::cout << " intersect triangles " << tri_a << " - " << id2 << ": " << intersects << std::endl;
                        if (intersects)
                        {
                            res.Add(tri_a, id2);
                            intersectingTriPairs.push_back(std::make_pair(tri_a, id2));
                            /*if (m_intersectingTriangles.find(tree2.identifier()) == m_intersectingTriangles.end())
                                m_intersectingTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                            m_intersectingTriangles[tree2.identifier()][tri_a].push_back(id2);

                            /*if (tree2.getIntersectingTriangles().find(m_identifier) == tree2.getIntersectingTriangles().end())
                                tree2.getIntersectingTriangles().insert(std::make_pair(m_identifier, std::make_pair(id2, std::vector<int>())));*/

                            tree2.getIntersectingTriangles()[m_identifier][id2].push_back(tri_a);
                        }
                        else
                        {
                            /*if (m_testedTriangles.find(tree2.identifier()) == m_testedTriangles.end())
                                m_testedTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                            m_testedTriangles[tree2.identifier()][tri_a].push_back(id2);

                            /*if (tree2.getTestedTriangles().find(m_identifier) == tree2.getTestedTriangles().end())
                                tree2.getTestedTriangles().insert(std::make_pair(m_identifier, std::make_pair(id2, std::vector<int>())));*/

                            tree2.getTestedTriangles()[m_identifier][id2].push_back(tri_a);
                        }
                    }
                }
            }
            break;
#endif
#if 1
            case PAIR_TRINODE_TRINODE:
            {
                //assert((unsigned int)id1 < a.num_tri_nodes && (unsigned int)id2 < b.num_tri_nodes);

                if (id1 < m_triNodes.size() && id2 < tree2.getTriNodes().size())
                {
                    //tri_a = a.tri_nodes[id1].tri1;
                    tri_a = m_triNodes.at(id1).getFirstChild();
                    //a.tri_mesh->GetTriangle(tri_a, tri_verts_a);

                    //tri_b = b.tri_nodes[id2].tri1;
                    tri_b = tree2.getTriNodes().at(id2).getFirstChild();
                    //b.tri_mesh->GetTriangle(tri_b, tri_verts_b);

                    triangle_a = m_topology->getTriangle(tri_a);
                    tri_verts_a[0] = x[triangle_a[0]];
                    tri_verts_a[1] = x[triangle_a[1]];
                    tri_verts_a[2] = x[triangle_a[2]];

                    triangle_b = tree2.getTopology()->getTriangle(tri_b);
                    tri_verts_b[0] = x2[triangle_b[0]];
                    tri_verts_b[1] = x2[triangle_b[1]];
                    tri_verts_b[2] = x2[triangle_b[2]];

                    // transform the triangle of object 'b' into the frame of object 'a'
                    //TRANSFORM_TRIANGLE(b_rel_a, tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);
                    //DO_TRI_TRI_TEST(tri_a, tri_b); //  <a1, b1>

                    bool intersects = intersectPqp.intersect_tri_tri(tri_verts_a[0], tri_verts_a[1], tri_verts_a[2],
                                                   tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                    checkedTriPairs.push_back(std::make_pair(tri_a, tri_b));
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                    std::cout << " intersect triangles " << tri_a << " - " << tri_b << ": " << intersects << std::endl;
#endif
                    if (intersects)
                    {
                        res.Add(tri_a, tri_b);
                        intersectingTriPairs.push_back(std::make_pair(tri_a, tri_b));
                        /*if (m_intersectingTriangles.find(tree2.identifier()) == m_intersectingTriangles.end())
                            m_intersectingTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                        m_intersectingTriangles[tree2.identifier()][tri_a].push_back(tri_b);

                        /*if (tree2.getIntersectingTriangles().find(m_identifier) == tree2.getIntersectingTriangles().end())
                            tree2.getIntersectingTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                        tree2.getIntersectingTriangles()[m_identifier][tri_b].push_back(tri_a);
                    }
                    else
                    {
                        /*if (m_testedTriangles.find(tree2.identifier()) == m_testedTriangles.end())
                            m_testedTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                        m_testedTriangles[tree2.identifier()][id1].push_back(tri_b);

                        /*if (tree2.getTestedTriangles().find(m_identifier) == tree2.getTestedTriangles().end())
                            tree2.getTestedTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                        tree2.getTestedTriangles()[m_identifier][tri_b].push_back(id1);
                    }

                    //tri_a = a.tri_nodes[id1].tri2;
                    tri_a = m_triNodes.at(id1).getSecondChild();
                    if (tri_a >= 0)
                    {
                        //a.tri_mesh->GetTriangle(tri_a, tri_verts_a);
                        triangle_a = m_topology->getTriangle(tri_a);
                        tri_verts_a[0] = x[triangle_a[0]];
                        tri_verts_a[1] = x[triangle_a[1]];
                        tri_verts_a[2] = x[triangle_a[2]];

                        //DO_TRI_TRI_TEST(tri_a, tri_b); // <a2, b1>
                        bool intersects = intersectPqp.intersect_tri_tri(tri_verts_a[0], tri_verts_a[1], tri_verts_a[2],
                                                       tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                        checkedTriPairs.push_back(std::make_pair(tri_a, tri_b));
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                        std::cout << " intersect triangles " << tri_a << " - " << tri_b << ": " << intersects << std::endl;
#endif
                        if (intersects)
                        {
                            res.Add(tri_a, tri_b);
                            intersectingTriPairs.push_back(std::make_pair(tri_a, tri_b));
                            /*if (m_intersectingTriangles.find(tree2.identifier()) == m_intersectingTriangles.end())
                                m_intersectingTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                            m_intersectingTriangles[tree2.identifier()][tri_a].push_back(tri_b);

                            /*if (tree2.getIntersectingTriangles().find(m_identifier) == tree2.getIntersectingTriangles().end())
                                tree2.getIntersectingTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                            tree2.getIntersectingTriangles()[m_identifier][tri_b].push_back(tri_a);
                        }
                        else
                        {
                            /*if (m_testedTriangles.find(tree2.identifier()) == m_testedTriangles.end())
                                m_testedTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                            m_testedTriangles[tree2.identifier()][tri_a].push_back(tri_b);

                            /*if (tree2.getTestedTriangles().find(m_identifier) == tree2.getTestedTriangles().end())
                                tree2.getTestedTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_a, std::vector<int>())));*/

                            tree2.getTestedTriangles()[m_identifier][tri_a].push_back(tri_b);
                        }
                    }

                    //tri_b = b.tri_nodes[id2].tri2;
                    tri_b = tree2.getTriNodes().at(id2).getSecondChild();
                    if (tri_b >= 0)
                    {
                        //b.tri_mesh->GetTriangle(tri_b, tri_verts_b);
                        triangle_b = tree2.getTopology()->getTriangle(tri_b);
                        tri_verts_b[0] = x2[triangle_b[0]];
                        tri_verts_b[1] = x2[triangle_b[1]];
                        tri_verts_b[2] = x2[triangle_b[2]];
                        // transform the triangle of object 'b' into the frame of object 'a'
                        //TRANSFORM_TRIANGLE(b_rel_a, tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                        // we are already have the second triangle in the node of object 'a'
                        if (tri_a >= 0)
                        {
                            //DO_TRI_TRI_TEST(tri_a, tri_b); // <a2, b2>
                            bool intersects = intersectPqp.intersect_tri_tri(tri_verts_a[0], tri_verts_a[1], tri_verts_a[2],
                                                           tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                            checkedTriPairs.push_back(std::make_pair(tri_a, tri_b));
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                            std::cout << " intersect triangles " << tri_a << " - " << tri_b << ": " << intersects << std::endl;
#endif
                            if (intersects)
                            {
                                res.Add(tri_a, tri_b);
                                intersectingTriPairs.push_back(std::make_pair(tri_a, tri_b));
                                /*if (m_intersectingTriangles.find(tree2.identifier()) == m_intersectingTriangles.end())
                                    m_intersectingTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                                m_intersectingTriangles[tree2.identifier()][tri_a].push_back(tri_b);

                                /*if (tree2.getIntersectingTriangles().find(m_identifier) == tree2.getIntersectingTriangles().end())
                                    tree2.getIntersectingTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                                tree2.getIntersectingTriangles()[m_identifier][tri_b].push_back(tri_a);
                            }
                            else
                            {
                                /*if (m_testedTriangles.find(tree2.identifier()) == m_testedTriangles.end())
                                    m_testedTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                                m_testedTriangles[tree2.identifier()][tri_a].push_back(tri_b);

                                /*if (tree2.getTestedTriangles().find(m_identifier) == tree2.getTestedTriangles().end())
                                    tree2.getTestedTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                                tree2.getTestedTriangles()[m_identifier][tri_b].push_back(tri_a);
                            }
                        }

                        //tri_a = a.tri_nodes[id1].tri1;
                        tri_a = m_triNodes.at(id1).getFirstChild();
                        //a.tri_mesh->GetTriangle(tri_a, tri_verts_a);
                        triangle_a = m_topology->getTriangle(tri_a);
                        tri_verts_a[0] = x[triangle_a[0]];
                        tri_verts_a[1] = x[triangle_a[1]];
                        tri_verts_a[2] = x[triangle_a[2]];

                        bool intersects = intersectPqp.intersect_tri_tri(tri_verts_a[0], tri_verts_a[1], tri_verts_a[2],
                                                       tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                        checkedTriPairs.push_back(std::make_pair(tri_a, tri_b));
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                        std::cout << " intersect triangles " << tri_a << " - " << tri_b << ": " << intersects << std::endl;
#endif
                        if (intersects)
                        {
                            res.Add(tri_a, tri_b);
                            intersectingTriPairs.push_back(std::make_pair(tri_a, tri_b));
                            /*if (m_intersectingTriangles.find(tree2.identifier()) == m_intersectingTriangles.end())
                                m_intersectingTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                            m_intersectingTriangles[tree2.identifier()][tri_a].push_back(tri_b);

                            /*if (tree2.getIntersectingTriangles().find(m_identifier) == tree2.getIntersectingTriangles().end())
                                tree2.getIntersectingTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                            tree2.getIntersectingTriangles()[m_identifier][tri_b].push_back(tri_a);
                        }
                        else
                        {
                            /*if (m_testedTriangles.find(tree2.identifier()) == m_testedTriangles.end())
                                m_testedTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(tri_a, std::vector<int>())));*/

                            m_testedTriangles[tree2.identifier()][tri_a].push_back(tri_b);

                            /*if (tree2.getTestedTriangles().find(m_identifier) == tree2.getTestedTriangles().end())
                                tree2.getTestedTriangles().insert(std::make_pair(m_identifier, std::make_pair(tri_b, std::vector<int>())));*/

                            tree2.getTestedTriangles()[m_identifier][tri_b].push_back(tri_a);
                        }

                        //DO_TRI_TRI_TEST(tri_a, tri_b); // <a1, b2>
                    }
                }
            }
            break;
#endif
#if 1
            case PAIR_TRI_TRI:
            {

                //a.tri_mesh->GetTriangle(id1, tri_verts_a);
                //b.tri_mesh->GetTriangle(id2, tri_verts_b);

                triangle_a = m_topology->getTriangle(id1);
                tri_verts_a[0] = x[triangle_a[0]];
                tri_verts_a[1] = x[triangle_a[1]];
                tri_verts_a[2] = x[triangle_a[2]];

                triangle_b = tree2.getTopology()->getTriangle(id2);
                tri_verts_b[0] = x2[triangle_b[0]];
                tri_verts_b[1] = x2[triangle_b[1]];
                tri_verts_b[2] = x2[triangle_b[2]];

                // transform the triangle of object 'b' into the frame of object 'a'
                //TRANSFORM_TRIANGLE(b_rel_a, tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                //DO_TRI_TRI_TEST(id1, id2);
                bool intersects = intersectPqp.intersect_tri_tri(tri_verts_a[0], tri_verts_a[1], tri_verts_a[2],
                                               tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

                checkedTriPairs.push_back(std::make_pair(tri_a, tri_b));
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
                std::cout << " intersect triangles " << id1 << " - " << id2 << ": " << intersects << std::endl;
#endif
                if (intersects)
                {
                    res.Add(id1, id2);
                    intersectingTriPairs.push_back(std::make_pair(tri_a, tri_b));
                    /*if (m_intersectingTriangles.find(tree2.identifier()) == m_intersectingTriangles.end())
                        m_intersectingTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(id1, std::vector<int>())));*/

                    m_intersectingTriangles[tree2.identifier()][id1].push_back(id2);

                    /*if (tree2.getIntersectingTriangles().find(m_identifier) == tree2.getIntersectingTriangles().end())
                        tree2.getIntersectingTriangles().insert(std::make_pair(m_identifier, std::make_pair(id2, std::vector<int>())));*/

                    tree2.getIntersectingTriangles()[m_identifier][id2].push_back(id1);
                }
                else
                {
                    /*if (m_testedTriangles.find(tree2.identifier()) == m_testedTriangles.end())
                        m_testedTriangles.insert(std::make_pair(tree2.identifier(), std::make_pair(id1, std::vector<int>())));*/

                    m_testedTriangles[tree2.identifier()][id1].push_back(id2);

                    /*if (tree2.getTestedTriangles().find(m_identifier) == tree2.getTestedTriangles().end())
                        tree2.getTestedTriangles().insert(std::make_pair(m_identifier, std::make_pair(id2, std::vector<int>())));*/

                    tree2.getTestedTriangles()[m_identifier][id2].push_back(id1);
                }
            }
            break;
#endif
            default:
            {
                std::cout << "Error in testOverlap: type = " << BVHNodeTypeName(pair.type) << " is unhandled in switch-statement!" << std::endl;
                break;
            }
        }

        if (stack.IsEmpty())
        {
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
            std::cout << "Stack empty at iteration " << numIterations << "; break loop." << std::endl;
#endif
            break;
        }

        pair = stack.Pop();
#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
        std::cout << "Pair on top of stack: " << pair.node1 << " <-> " << pair.node2 << ", type = " << pair.type << std::endl;
#endif
        numIterations++;
#ifdef OBBTREE_TEST_OVERLAP_ITERATION_LIMIT
        if (numIterations >= maxIterations)
        {
            std::cout << "BREAK AFTER " << numIterations << std::endl;
            break;
        }
#endif
    } while(true);

#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << "CHECKED OBB DUMP: " << checkedObbs.size() << std::endl;
    for (std::map<std::pair<std::string, std::string>,bool>::const_iterator it = checkedObbs.begin(); it != checkedObbs.end(); it++)
    {
        std::cout << " - " << it->first.first << " <-> " << it->first.second << ": " << it->second << std::endl;
    }
#endif

#ifdef OBBTREE_TEST_OVERLAP_VERBOSE
    std::cout << "CHECKED TRI-PAIR DUMP: " << checkedTriPairs.size() << std::endl;
    for (std::vector<std::pair<int,int> >::const_iterator it = checkedTriPairs.begin(); it != checkedTriPairs.end(); it++)
    {
        std::cout << " -> " << it->first << " -- " << it->second << std::endl;
    }
#endif
    std::cout << "INTERSECTING TRI-PAIR DUMP: " << intersectingTriPairs.size() << std::endl;
    for (std::vector<std::pair<int,int> >::const_iterator it = intersectingTriPairs.begin(); it != intersectingTriPairs.end(); it++)
    {
        std::cout << " -> " << it->first << " -- " << it->second << std::endl;
    }

#ifdef OBBTREE_TEST_OVERLAP_BRUTE_FORCE_REFERENCE
    std::cout << "=== Brute force tri-tri tests ===" << std::endl;
    int intersectingTris = 0;
    for (int k = 0; k < m_topology->getNbTriangles(); k++)
    {
        triangle_a = m_topology->getTriangle(k);
        tri_verts_a[0] = (*m_state->getX())[triangle_a[0]];
        tri_verts_a[1] = (*m_state->getX())[triangle_a[1]];
        tri_verts_a[2] = (*m_state->getX())[triangle_a[2]];


        for (int l = 0; l < tree2.getTopology()->getNbTriangles(); l++)
        {
            triangle_b = tree2.getTopology()->getTriangle(l);
            tri_verts_b[0] = (*tree2.getMState()->getX())[triangle_b[0]];
            tri_verts_b[1] = (*tree2.getMState()->getX())[triangle_b[1]];
            tri_verts_b[2] = (*tree2.getMState()->getX())[triangle_b[2]];

            bool intersects = intersectPqp.intersect_tri_tri(tri_verts_a[0], tri_verts_a[1], tri_verts_a[2],
                    tri_verts_b[0], tri_verts_b[1], tri_verts_b[2]);

            if (intersects)
            {
                std::cout << " -> intersect triangles " << k << " - " << l << ": " << intersects << std::endl;
//                m_intersectingTriangles[tree2.identifier()][k].push_back(l);
//                tree2.getIntersectingTriangles()[m_identifier][l].push_back(k);
                intersectingTris++;
            }
        }
    }
    std::cout << "intersecting triangle pairs total: " << intersectingTris << std::endl;
#endif
    return true;
}

bool ObbTree::computeOverlap(const ObbTree &tree2, CollideResult &res)
{
    return false;
}

void ObbTree::drawRec(ObbVolume& parent, unsigned int depth, bool overlappingOnly)
{
    //std::cout << " drawRec(" << parent.identifier() << ", " << depth << "); child0 = " << parent.getFirstChild() << ", child1 = " << parent.getSecondChild() << std::endl;

    if (depth >= m_maxDrawDepth)
        return;

    /*if (parent.getFirstChild() <= 0 || parent.getSecondChild() <= 0)
        return;*/

    if (parent.getFirstChild() > 0)
    {
        ObbVolume& child1 = m_obbNodes.at(std::abs(parent.getFirstChild()));
        glTranslated(child1.getPosition().x(),
                     child1.getPosition().y(),
                     child1.getPosition().z());
        Matrix4 c1_ori(child1.getOrientation()); c1_ori[3][3] = 1;
        glMultMatrixd(c1_ori.transposed().ptr());

        if (depth >= m_minDrawDepth)
        {
            bool emph = (std::find(m_intersectingOBBIndices.begin(), m_intersectingOBBIndices.end(), parent.getFirstChild()) != m_intersectingOBBIndices.end());
            if (!overlappingOnly)
            {
                BVHDrawHelpers::drawCoordinateMarkerGL(1.0f, 0.5f);
                if (emph)
                    BVHDrawHelpers::drawObbVolume(child1.getHalfExtents(), Vec4f(0,1,0,1), true);
                else
                    BVHDrawHelpers::drawObbVolume(child1.getHalfExtents(), m_obbNodeColors[parent.getFirstChild()], true);
            }
            else
            {
                if (emph)
                    BVHDrawHelpers::drawObbVolume(child1.getHalfExtents(), m_obbNodeColors[parent.getFirstChild()], true);
            }

            if (true)
            {
                Mat<4,4, GLfloat> modelviewM;
                sofa::defaulttype::Vec4f color;

                if (m_obbNodeColors.find(parent.getFirstChild()) != m_obbNodeColors.end())
                    color = Vec4f(m_obbNodeColors[parent.getFirstChild()].x(), m_obbNodeColors[parent.getFirstChild()].y(), m_obbNodeColors[parent.getFirstChild()].z(), m_obbNodeColors[parent.getFirstChild()].w());
                else
                    color = Vec4f(1,1,1,0.75f);

                glColor4f(color[0], color[1], color[2], color[3]);
                glDisable(GL_LIGHTING);

                const sofa::defaulttype::BoundingBox& bbox = m_state->getContext()->f_bbox.getValue();
                float scale  = (float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.0001);

                std::string tmp = child1.identifier();
                for (std::map<std::string, std::map<int, std::vector<std::string> > >::const_iterator obbIt = m_intersectingOBBs.begin(); obbIt != m_intersectingOBBs.end(); obbIt++)
                {
                    if (m_intersectingOBBs[obbIt->first].find(parent.getFirstChild()) != m_intersectingOBBs[obbIt->first].end())
                    {
                        tmp += " intersects: ";
                        for (std::vector<std::string>::const_iterator it = m_intersectingOBBs[obbIt->first][parent.getFirstChild()].begin(); it != m_intersectingOBBs[obbIt->first][parent.getFirstChild()].end(); it++)
                            tmp += ((*it) + ";");
                    }
                }
                const char* s = tmp.c_str();
                glPushMatrix();

                //glTranslatef(center[0], center[1], center[2]);
                glScalef(scale,scale,scale);

                // Makes text always face the viewer by removing the scene rotation
                // get the current modelview matrix
                glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
                modelviewM.transpose();

                sofa::defaulttype::Vec3f temp = modelviewM.transform(child1.getPosition());

                //glLoadMatrixf(modelview);
                glLoadIdentity();

                glTranslatef(temp[0], temp[1], temp[2]);
                glScalef(scale,scale,scale);

                while(*s)
                {
                    glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                    s++;
                }

                glPopMatrix();
            }
        }
        glMultMatrixd(c1_ori.ptr());

        glTranslated(-child1.getPosition().x(),
                     -child1.getPosition().y(),
                     -child1.getPosition().z());

        drawRec(child1, depth + 1);
    }
    else if (parent.getChildType(0) == TRIANGLE_PAIR_CHILD_NODE||
             parent.getChildType(1) == TRIANGLE_LEAF_CHILD_NODE)
    {
        //std::cout << " draw TRIANGLE_PAIR_CHILD_NODE 1" << std::endl;
        Matrix4 parent_ori(parent.getOrientation()); parent_ori[3][3] = 1;
        glTranslated(parent.getPosition().x(), parent.getPosition().y(), parent.getPosition().z());
        glMultMatrixd(parent_ori.transposed().ptr());
        BVHDrawHelpers::drawObbVolume(parent.getHalfExtents(), Vec4f(1,1,1,0.5f));
        glMultMatrixd(parent_ori.ptr());
        glTranslated(-parent.getPosition().x(), -parent.getPosition().y(), -parent.getPosition().z());
    }

    if (parent.getSecondChild() > 0)
    {
        ObbVolume& child2 = m_obbNodes.at(std::abs(parent.getSecondChild()));

        glTranslated(child2.getPosition().x(),
                     child2.getPosition().y(),
                     child2.getPosition().z());
        Matrix4 c2_ori(child2.getOrientation()); c2_ori[3][3] = 1;
        glMultMatrixd(c2_ori.transposed().ptr());

        if (depth >= m_minDrawDepth)
        {
            bool emph = (std::find(m_intersectingOBBIndices.begin(), m_intersectingOBBIndices.end(), parent.getSecondChild()) != m_intersectingOBBIndices.end());
            if (!overlappingOnly)
            {
                BVHDrawHelpers::drawCoordinateMarkerGL(1.0f, 0.5f);
                if (emph)
                    BVHDrawHelpers::drawObbVolume(child2.getHalfExtents(), Vec4f(0,1,0,1), true);
                else
                    BVHDrawHelpers::drawObbVolume(child2.getHalfExtents(), m_obbNodeColors[parent.getSecondChild()], true);
            }
            else
            {
                if (emph)
                    BVHDrawHelpers::drawObbVolume(child2.getHalfExtents(), m_obbNodeColors[parent.getSecondChild()], true);
            }

            if (true)
            {
                Mat<4,4, GLfloat> modelviewM;

                Vec4f color;
                if (m_obbNodeColors.find(parent.getFirstChild()) != m_obbNodeColors.end())
                    color = Vec4f(m_obbNodeColors[parent.getSecondChild()].x(), m_obbNodeColors[parent.getSecondChild()].y(), m_obbNodeColors[parent.getSecondChild()].z(), m_obbNodeColors[parent.getSecondChild()].w());
                else
                    color = Vec4f(1,1,1,0.75f);

                glColor4f(color[0], color[1], color[2], color[3]);
                glDisable(GL_LIGHTING);

                const sofa::defaulttype::BoundingBox& bbox = m_state->getContext()->f_bbox.getValue();
                float scale  = (float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.0001);

                std::string tmp = child2.identifier();
                for (std::map<std::string, std::map<int, std::vector<std::string> > >::const_iterator obbIt = m_intersectingOBBs.begin(); obbIt != m_intersectingOBBs.end(); obbIt++)
                {
                    if (m_intersectingOBBs[obbIt->first].find(parent.getSecondChild()) != m_intersectingOBBs[obbIt->first].end())
                    {
                        tmp += " intersects: ";
                        for (std::vector<std::string>::const_iterator it = m_intersectingOBBs[obbIt->first][parent.getSecondChild()].begin(); it != m_intersectingOBBs[obbIt->first][parent.getSecondChild()].end(); it++)
                            tmp += ((*it) + ";");
                    }
                }
                const char* s = tmp.c_str();
                glPushMatrix();

                //glTranslatef(center[0], center[1], center[2]);
                glScalef(scale,scale,scale);

                // Makes text always face the viewer by removing the scene rotation
                // get the current modelview matrix
                glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
                modelviewM.transpose();

                sofa::defaulttype::Vec3f temp = modelviewM.transform(child2.getPosition());

                //glLoadMatrixf(modelview);
                glLoadIdentity();

                glTranslatef(temp[0], temp[1], temp[2]);
                glScalef(scale,scale,scale);

                while(*s)
                {
                    glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                    s++;
                }

                glPopMatrix();
            }
        }
        glMultMatrixd(c2_ori.ptr());
        glTranslated(-child2.getPosition().x(),
                  -child2.getPosition().y(),
                  -child2.getPosition().z());

        drawRec(child2, depth + 1);
    }
    else if (parent.getChildType(1) == TRIANGLE_PAIR_CHILD_NODE ||
             parent.getChildType(1) == TRIANGLE_LEAF_CHILD_NODE)
    {
        Matrix4 parent_ori(parent.getOrientation()); parent_ori[3][3] = 1;
        //std::cout << " draw TRIANGLE_PAIR_CHILD_NODE 2" << std::endl;
        glTranslated(parent.getPosition().x(), parent.getPosition().y(), parent.getPosition().z());
        glMultMatrixd(parent_ori.transposed().ptr());
        BVHDrawHelpers::drawObbVolume(parent.getHalfExtents(), Vec4f(1,1,1,0.5f));
        glMultMatrixd(parent_ori.ptr());
        glTranslated(-parent.getPosition().x(), -parent.getPosition().y(), -parent.getPosition().z());
    }
}

void ObbTree::draw(const sofa::core::visual::VisualParams *vparams)
{
    //std::cout << "=== draw(" << identifier() << "), with " << m_obbNodes.size() << " child volumes ===" << std::endl;
    //std::cout << "  ROOT at = " << m_worldPos << ", position = " << m_position << ", extents = " << m_obbNodes.at(0).getPosition() << std::endl;

	if (m_obbNodes.size() == 0)
		return;

    const Vec3Types::VecCoord& x = m_state->read(core::ConstVecCoordId::position())->getValue();

    if (!m_drawTriangleTestsOnly)
    {
        glPushMatrix();

        glPushAttrib(GL_ENABLE_BIT);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);

        if (m_obbNodeColors.empty())
            assignOBBNodeColors();

        glTranslated(m_worldPos.x(), m_worldPos.y(), m_worldPos.z());
        Matrix4 w_ori(m_worldRot); w_ori[3][3] = 1;
        glMultMatrixd(w_ori.transposed().ptr());
        glTranslated(getPosition().x(), getPosition().y(), getPosition().z());
        Matrix4 ori(getOrientation()); ori[3][3] = 1;
        glMultMatrixd(ori.transposed().ptr());

        BVHDrawHelpers::drawCoordinateMarkerGL(3.0f, 2.0f);
        BVHDrawHelpers::drawObbVolume(m_halfExtents, Vec4f(1,1,1,1), true);

        // Draw OBB intersections
        if (false)
        {
            Mat<4,4, GLfloat> modelviewM;
            const sofa::defaulttype::Vec3f& color = Vec3f(1,1,1);
            glColor3f(color[0], color[1], color[2]);
            glDisable(GL_LIGHTING);

            const sofa::defaulttype::BoundingBox& bbox = m_state->getContext()->f_bbox.getValue();
            float scale  = (float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.0001);

            std::string tmp = m_identifier;
            for (std::map<std::string, std::map<int, std::vector<std::string> > >::const_iterator obbIt = m_intersectingOBBs.begin(); obbIt != m_intersectingOBBs.end(); obbIt++)
            {
                if (m_intersectingOBBs[obbIt->first].find(0) != m_intersectingOBBs[obbIt->first].end())
                {
                    tmp += " intersects: ";
                    for (std::vector<std::string>::const_iterator it = m_intersectingOBBs[obbIt->first][0].begin(); it != m_intersectingOBBs[obbIt->first][0].end(); it++)
                        tmp += ((*it) + ";");
                }
            }
            const char* s = tmp.c_str();
            glPushMatrix();

            //glTranslatef(center[0], center[1], center[2]);
            glScalef(scale,scale,scale);

            // Makes text always face the viewer by removing the scene rotation
            // get the current modelview matrix
            glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
            modelviewM.transpose();

            sofa::defaulttype::Vec3f temp = modelviewM.transform(m_worldPos);

            //glLoadMatrixf(modelview);
            glLoadIdentity();

            glTranslatef(temp[0], temp[1], temp[2]);
            glScalef(scale,scale,scale);

            while(*s)
            {
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                s++;
            }

            glPopMatrix();
        }


        glMultMatrixd(ori.ptr());
        glTranslated(-m_position.x(), -m_position.y(), -m_position.z());

#if 0
        if (m_obbNodes.size() > 0)
        {
            if (m_obbNodes.at(0).getFirstChild() > 0)
            {
                ObbVolume& child1 = m_obbNodes.at(getFirstChild());

                //std::cout << " child 1: pos = " << child1.getPosition() << ", extents = " << child1.getHalfExtents() << std::endl;

                glTranslated(child1.getPosition().x(),
                             child1.getPosition().y(),
                             child1.getPosition().z());

                Matrix4 c1_ori(child1.getOrientation()); c1_ori[3][3] = 1;
                glMultMatrixd(c1_ori.transposed().ptr());

                if (1 > m_minDrawDepth)
                {
                    bool emph = (std::find(m_intersectingOBBIndices.begin(), m_intersectingOBBIndices.end(), getFirstChild()) != m_intersectingOBBIndices.end());
                    BVHDrawHelpers::drawCoordinateMarkerGL(2.0f, 1.0f);

                    if (!m_drawOverlappingOnly)
                    {
                        if (emph)
                            BVHDrawHelpers::drawObbVolume(child1.getHalfExtents(), Vec4f(0,1,0,1), true);
                        else
                            BVHDrawHelpers::drawObbVolume(child1.getHalfExtents(), m_obbNodeColors[getFirstChild()], true);
                    }
                    else
                    {
                        if (emph)
                            BVHDrawHelpers::drawObbVolume(child1.getHalfExtents(), Vec4f(0,1,0,1)/*m_obbNodeColors[getFirstChild()]*/, true);
                    }

                    // Draw OBB intersections
                    if (false)
                    {
                        Mat<4,4, GLfloat> modelviewM;
                        const sofa::defaulttype::Vec4f& color = Vec4f(m_obbNodeColors[getFirstChild()].x(), m_obbNodeColors[getFirstChild()].y(), m_obbNodeColors[getFirstChild()].z(), m_obbNodeColors[getFirstChild()].w());
                        glColor4f(color[0], color[1], color[2], color[3]);
                        glDisable(GL_LIGHTING);

                        const sofa::defaulttype::BoundingBox& bbox = m_state->getContext()->f_bbox.getValue();
                        float scale  = (float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.0001);

                        std::string tmp = child1.identifier();
                        for (std::map<std::string, std::map<int, std::vector<std::string> > >::const_iterator obbIt = m_intersectingOBBs.begin(); obbIt != m_intersectingOBBs.end(); obbIt++)
                        {
                            if (m_intersectingOBBs[obbIt->first].find(getFirstChild()) != m_intersectingOBBs[obbIt->first].end())
                            {
                                tmp += " intersects: ";
                                for (std::vector<std::string>::const_iterator it = m_intersectingOBBs[obbIt->first][getFirstChild()].begin(); it != m_intersectingOBBs[obbIt->first][getFirstChild()].end(); it++)
                                    tmp += ((*it) + ";");
                            }
                        }
                        const char* s = tmp.c_str();
                        glPushMatrix();

                        //glTranslatef(center[0], center[1], center[2]);
                        glScalef(scale,scale,scale);

                        // Makes text always face the viewer by removing the scene rotation
                        // get the current modelview matrix
                        glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
                        modelviewM.transpose();

                        sofa::defaulttype::Vec3f temp = modelviewM.transform(child1.getPosition());

                        //glLoadMatrixf(modelview);
                        glLoadIdentity();

                        glTranslatef(temp[0], temp[1], temp[2]);
                        glScalef(scale,scale,scale);

                        while(*s)
                        {
                            glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                            s++;
                        }

                        glPopMatrix();
                    }
                }
                glMultMatrixd(c1_ori.ptr());


                glTranslated(-child1.getPosition().x(),
                             -child1.getPosition().y(),
                             -child1.getPosition().z());

                drawRec(child1, 1);
            }

            if (m_obbNodes.at(0).getSecondChild() > 0)
            {
                ObbVolume& child2 = m_obbNodes.at(getSecondChild());

                //std::cout << " child 2: pos = " << child2.getPosition() << ", extents = " << child2.getHalfExtents() << std::endl;

                glTranslated(child2.getPosition().x(),
                             child2.getPosition().y(),
                             child2.getPosition().z());
                Matrix4 c2_ori(child2.getOrientation()); c2_ori[3][3] = 1;
                glMultMatrixd(c2_ori.transposed().ptr());

                if (1 > m_minDrawDepth)
                {
                    bool emph = (std::find(m_intersectingOBBIndices.begin(), m_intersectingOBBIndices.end(), getSecondChild()) != m_intersectingOBBIndices.end());
                    BVHDrawHelpers::drawCoordinateMarkerGL(2.0f, 1.0f);

                    if (!m_drawOverlappingOnly)
                    {
                        if (emph)
                            BVHDrawHelpers::drawObbVolume(child2.getHalfExtents(), Vec4f(0,1,0,1), true);
                        else
                            BVHDrawHelpers::drawObbVolume(child2.getHalfExtents(), Vec4f(0,1,0,1) /*m_obbNodeColors[getSecondChild()]*/, true);
                    }
                    else
                    {
                        if (emph)
                            BVHDrawHelpers::drawObbVolume(child2.getHalfExtents(), Vec4f(0,1,0,1)/*m_obbNodeColors[getSecondChild()]*/, true);
                    }


                    // Draw OBB intersections
                    if (false)
                    {
                        Mat<4,4, GLfloat> modelviewM;
                        const sofa::defaulttype::Vec4f& color = Vec4f(m_obbNodeColors[getSecondChild()].x(), m_obbNodeColors[getSecondChild()].y(), m_obbNodeColors[getSecondChild()].z(), m_obbNodeColors[getSecondChild()].w());
                        glColor4f(color[0], color[1], color[2], color[3]);
                        glDisable(GL_LIGHTING);

                        const sofa::defaulttype::BoundingBox& bbox = m_state->getContext()->f_bbox.getValue();
                        float scale  = (float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.0001);

                        std::string tmp = child2.identifier();
                        for (std::map<std::string, std::map<int, std::vector<std::string> > >::const_iterator obbIt = m_intersectingOBBs.begin(); obbIt != m_intersectingOBBs.end(); obbIt++)
                        {
                            if (m_intersectingOBBs[obbIt->first].find(getSecondChild()) != m_intersectingOBBs[obbIt->first].end())
                            {
                                tmp += " intersects: ";
                                for (std::vector<std::string>::const_iterator it = m_intersectingOBBs[obbIt->first][getSecondChild()].begin(); it != m_intersectingOBBs[obbIt->first][getSecondChild()].end(); it++)
                                    tmp += ((*it) + ";");
                            }
                        }
                        const char* s = tmp.c_str();
                        glPushMatrix();

                        glScalef(scale,scale,scale);

                        // Makes text always face the viewer by removing the scene rotation
                        // get the current modelview matrix
                        glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
                        modelviewM.transpose();

                        sofa::defaulttype::Vec3f temp = modelviewM.transform(child2.getPosition());

                        //glLoadMatrixf(modelview);
                        glLoadIdentity();

                        glTranslatef(temp[0], temp[1], temp[2]);
                        glScalef(scale,scale,scale);

                        while(*s)
                        {
                            glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                            s++;
                        }

                        glPopMatrix();
                    }
                }
                glMultMatrixd(c2_ori.ptr());

                glTranslated(-child2.getPosition().x(),
                             -child2.getPosition().y(),
                             -child2.getPosition().z());

                drawRec(child2, 1);
            }


        }

#endif

        //    if (m_intersectingOBBIndices.size() > 0)
        //    {
        //        for (std::map<std::string, std::map<int, std::vector<int> > >::const_iterator it = m_intersectingOBBIndices.begin(); it != m_intersectingOBBIndices.end(); it++)
        //        {
        //            for (std::map<int, std::vector<int> >::const_iterator obbIt = m_intersectingOBBIndices[it->first].begin();
        //                 obbIt != m_intersectingOBBIndices[it->first].end(); obbIt++)
        //            {
        //                ObbVolume& child1 = m_obbNodes.at(obbIt->first);

        //                glTranslated(child1.getPosition().x(),
        //                             child1.getPosition().y(),
        //                             child1.getPosition().z());

        //                Matrix4 c1_ori(child1.getOrientation()); c1_ori[3][3] = 1;
        //                glMultMatrixd(c1_ori.transposed().ptr());

        //                BVHDrawHelpers::drawObbVolume(child1.getHalfExtents(), Vec4f(0,1,0,1), true);
        //                glMultMatrixd(c1_ori.ptr());
        //                glTranslated(-child1.getPosition().x(),
        //                             -child1.getPosition().y(),
        //                             -child1.getPosition().z());
        //            }
        //        }
        //    }

        glPopAttrib();
        glPopMatrix();
    }


    std::vector<Vector3> testedTris;
    for (std::map<std::string, std::map<int, std::vector<int> > >::const_iterator triIt = m_testedTriangles.begin(); triIt != m_testedTriangles.end(); triIt++)
    {
        for (std::map<int, std::vector<int> >::const_iterator it = triIt->second.begin(); it != triIt->second.end(); it++)
        {
            sofa::core::topology::BaseMeshTopology::Triangle tri = m_topology->getTriangle(it->first);
            Vector3 pt1 = x[tri[0]];
            Vector3 pt2 = x[tri[1]];
            Vector3 pt3 = x[tri[2]];
            testedTris.push_back(pt1); testedTris.push_back(pt2);
            testedTris.push_back(pt2); testedTris.push_back(pt3);
            testedTris.push_back(pt3); testedTris.push_back(pt1);

            Vec4f color(1,1,0,0.5);
            // Draw Triangles indices
            {
                Mat<4,4, GLfloat> modelviewM;
                glColor4f(color[0], color[1], color[2], color[3]);
                glDisable(GL_LIGHTING);
                float scale = 0.0001f;

                sofa::defaulttype::Vec3f center = (pt1 + pt2 + pt3)/3;

                std::ostringstream oss;
                oss << it->first;
                std::string tmp = oss.str();
                const char* s = tmp.c_str();
                glPushMatrix();

                glTranslatef(center[0], center[1], center[2]);
                glScalef(scale,scale,scale);

                // Makes text always face the viewer by removing the scene rotation
                // get the current modelview matrix
                glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
                modelviewM.transpose();

                sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

                //glLoadMatrixf(modelview);
                glLoadIdentity();

                glTranslatef(temp[0], temp[1], temp[2]);
                glScalef(scale,scale,scale);

                while(*s)
                {
                    glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                    s++;
                }

                glPopMatrix();
            }
        }
    }

    vparams->drawTool()->drawLines(testedTris, 5.0f, Vec4f(1,1,0,0.5));

    std::vector<Vector3> intersectingTris;
    //std::cout << " m_intersectingTriangles size = " << m_intersectingTriangles.size() << ": ";
    for (std::map<std::string, std::map<int, std::vector<int> > >::const_iterator triIt = m_intersectingTriangles.begin(); triIt != m_intersectingTriangles.end(); triIt++)
    {
        for (std::map<int, std::vector<int> >::const_iterator it = triIt->second.begin(); it != triIt->second.end(); it++)
        {
            //std::cout << it->first << ";";
            sofa::core::topology::BaseMeshTopology::Triangle tri = m_topology->getTriangle(it->first);
            Vector3 pt1 = x[tri[0]];
            Vector3 pt2 = x[tri[1]];
            Vector3 pt3 = x[tri[2]];
            intersectingTris.push_back(pt1); intersectingTris.push_back(pt2);
            intersectingTris.push_back(pt2); intersectingTris.push_back(pt3);
            intersectingTris.push_back(pt3); intersectingTris.push_back(pt1);

            Vec4f color(1,0,1,0.5);
            // Draw Triangles indices
            {
                Mat<4,4, GLfloat> modelviewM;
                glColor4f(color[0], color[1], color[2], color[3]);
                glDisable(GL_LIGHTING);
                float scale = 0.0001f;

                sofa::defaulttype::Vec3f center = (pt1 + pt2 + pt3)/3;

                std::ostringstream oss;
                //oss << it->first << ": ";
                for (std::vector<int>::const_iterator t_it = it->second.begin(); t_it != it->second.end(); t_it++)
                    oss << *t_it << ";";

                std::string tmp = oss.str();
                const char* s = tmp.c_str();
                glPushMatrix();

                glTranslatef(center[0], center[1], center[2]);
                glScalef(scale,scale,scale);

                // Makes text always face the viewer by removing the scene rotation
                // get the current modelview matrix
                glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
                modelviewM.transpose();

                sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

                //glLoadMatrixf(modelview);
                glLoadIdentity();

                glTranslatef(temp[0], temp[1], temp[2]);
                glScalef(scale,scale,scale);

                while(*s)
                {
                    glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                    s++;
                }

                glPopMatrix();
            }
        }
    }
    //std::cout << std::endl;
    vparams->drawTool()->drawLines(intersectingTris, 10.0f, Vec4f(1,0,1,0.5));
}

void ObbTree::translate(const Vector3 & offset)
{
    m_worldPos += offset;
}

void ObbTree::rotate(const Matrix3& transform)
{
    m_worldRot = transform * m_worldRot;
    /*m_position =  transform * m_position;
    m_localAxes.col(0) = transform * (m_halfExtents.x() * m_localAxes.col(0));
    m_localAxes.col(1) = transform * (m_halfExtents.y() * m_localAxes.col(1));
    m_localAxes.col(2) = transform * (m_halfExtents.z() * m_localAxes.col(2));

    m_halfExtents[0] = m_localAxes.col(0).norm();
    m_halfExtents[1] = m_localAxes.col(1).norm();
    m_halfExtents[2] = m_localAxes.col(2).norm();*/
}

#include "initBVHModelsPlugin.h"
#include <sofa/core/ObjectFactory.h>

SOFA_DECL_CLASS(ObbTreeBuilder)

int ObbTreeBuilderClass = sofa::core::RegisterObject("OBB tree builder class")
        .add< ObbTreeBuilder >();

