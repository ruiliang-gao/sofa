#ifndef LGCPOINTCLUSTER_H
#define LGCPOINTCLUSTER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/component/topology/MeshTopology.h>

#include "initPlugin.h"

#include "LGCTransformable.h"
#include "LGCIntersectable.h"
#include "LGCObb.h"
#include "LGCPlane3.h"
#include "LGCPlane3Drawable.h"

#include <PQP/include/PQP.h>

#include <pcl/point_types.h>
#include <pcl/octree/octree_base.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/octree_search.h>

namespace pcl
{
    namespace gpu
    {
        class Octree;
    }
}

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace sofa::defaulttype;

            template <class TDataTypes> class LGCPointCluster;
            template <class LGCDataTypes>
            class SOFA_LGC_API LGCPointClusterObb: public LGCObb<LGCDataTypes>,
                                                   public sofa::core::objectmodel::BaseObject
            {
                public:
                    friend class LGCPointCluster<LGCDataTypes>;
                    SOFA_CLASS(LGCPointClusterObb, BaseObject);
                    typedef typename LGCDataTypes::Real Real;

                    LGCPointClusterObb(LGCPointCluster<LGCDataTypes>* cluster, const Vector3& translation = Vector3(0,0,0), const Quaternion& rotation = Quaternion::identity(), const double& scale = 1.0f, const core::CollisionModel* cm = 0);

                    virtual void draw(const core::visual::VisualParams *vparams);
                    void fitToCluster();

                    virtual typename LGCObb<LGCDataTypes>::ObbNodeType nodeType() const { return LGCObb<LGCDataTypes>::OBB_CLUSTER; }

                    LGCPointCluster<LGCDataTypes>* pointCluster() { return _cluster; }

                    unsigned long numChildren() const
                    {
                        return _allChildren.size();
                    }

                    virtual void addChild(LGCObb<LGCDataTypes>* child)
                    {
                        addTreeChild(child);
                    }

                    virtual LGCObb<LGCDataTypes>* child(const unsigned long &i)
                    {
                        return treeChild(i);
                    }

                    void addTreeChild(LGCObb<LGCDataTypes>* child)
                    {
                        if (std::find(_allChildren.begin(), _allChildren.end(), child) == _allChildren.end())
                        {
                            _allChildren.push_back(child);
                        }
                    }

                    LGCObb<LGCDataTypes>* treeChild(const unsigned long& i)
                    {
                        if (i < _allChildren.size())
                        {
                            return _allChildren.at(i);
                        }
                        return NULL;
                    }

                    LGCIntersectableType intersectableType() const { return LGC_OBB_POINTCLUSTER; }

                private:
                    LGCPointCluster<LGCDataTypes>* _cluster;

                    std::deque<LGCObb<LGCDataTypes>*> _allChildren;

                    void drawObbVolume(const Vec<3, Real> &halfExtents, const Vec4f &color);
                    void drawCoordinateMarkerGL(float lineLength = 1.0f, float lineWidth = 1.0f, const Vec<4, Real>& xColor = Vec<4, Real>(1,0,0,1), const Vec<4, Real>& yColor = Vec<4, Real>(0,1,0,1), const Vec<4, Real>& zColor = Vec<4, Real>(0,0,1,1));
            };

            template <class TDataTypes> class LGCPointClusterPrivate;

            typedef pcl::octree::OctreeBase<> OcTree;
            typedef pcl::octree::OctreePointCloudPointVector<pcl::LGCPointTypeMin> OcTreeSinglePoint;
            typedef pcl::octree::OctreePointCloudSearch<pcl::LGCPointTypeMin> OctreeSearch;

            template <class LGCDataTypes>
            class SOFA_LGC_API LGCPointCluster: public LGCTransformable<typename LGCDataTypes::Real>,
                                                public LGCIntersectable<typename LGCDataTypes::Real>,
                                                public sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_CLASS(LGCPointCluster, BaseObject);

                    typedef typename LGCDataTypes::Real Real;

                    LGCPointCluster(const Vector3& position = Vector3(0,0,0), const Quaternion& orientation = Quaternion::identity(), LGCPointCluster<LGCDataTypes>* parent = NULL, core::CollisionModel* model = 0);
                    LGCPointCluster(const std::vector<pcl::LGCPointTypeMin>& clusterSurfacePoints, const std::vector<pcl::LGCPointTypeMin>& clusterEdgePoints, const std::vector<pcl::LGCPointTypeMin>& clusterVertices, const Vector3& position, const Quaternion& orientation = Quaternion::identity(), const ReferenceFrame& referenceFrame = DEFAULT_REFERENCE_FRAME, LGCPointCluster<LGCDataTypes>* parent = NULL, const core::CollisionModel* model = 0);

                    ~LGCPointCluster();

                    virtual void transform(const Vector3& position, const Quaternion& orientation);

                    LGCIntersectableType intersectableType() const { return LGC_POINTCLUSTER; }
                    bool testIntersection(const LGCIntersectable<typename LGCDataTypes::Real> &);

                    LGCPointCluster<LGCDataTypes>* parentCluster() { return _parentCluster; }

                    LGCCollisionModel<LGCDataTypes>* lgcCollisionModel() const { return _lgcCollisionModel; }

                    unsigned long numSurfacePoints() const { return _clusterPoints.size(); }

                    const Vector3& surfacePoint(const unsigned long& idx) const
                    {
                        return _clusterPoints.at(idx);
                    }

                    Vector3& surfacePoint(const unsigned long& idx)
                    {
                        return _clusterPoints.at(idx);
                    }

                    unsigned long surfacePointIndex(const unsigned long& idx) const
                    {
                        return _surfacePointIndices[idx];
                    }

                    unsigned long surfacePointFacetIndex(const unsigned long& idx) const
                    {
                        return _surfacePointFacetIndices[idx];
                    }

                    void setSurfacePointFacetIndex(const unsigned long& idx, const unsigned long& value)
                    {
                        if (_surfacePointFacetIndices.find(idx) == _surfacePointFacetIndices.end())
                            _surfacePointFacetIndices.insert(std::make_pair(idx, value));
                        else
                            _surfacePointFacetIndices[idx] = value;
                    }

                    void addSurfacePoint(const Vector3& point, const unsigned long& pointIdx, const unsigned long& facetIdx)
                    {
                        if (std::find(_clusterPoints.begin(), _clusterPoints.end(), point) == _clusterPoints.end())
                        {
                            _clusterPoints.push_back(point);
                            //std::cout << " * Add surface point " <<  _clusterPoints.size() - 1 << " for facet " << facetIdx << " (count " << pointIdx << "): " << point << std::endl;
                            _surfacePointFacetIndices.insert(std::make_pair(_clusterPoints.size() - 1, facetIdx));
                            _surfacePointIndices.insert(std::make_pair(_clusterPoints.size() - 1, pointIdx));
                        }
                    }

                    unsigned long numEdgePoints() const { return _edgePoints.size(); }

                    const Vector3& edgePoint(const unsigned long& idx) const
                    {
                        return _edgePoints.at(idx);
                    }

                    Vector3& edgePoint(const unsigned long& idx)
                    {
                        return _edgePoints.at(idx);
                    }

                    unsigned long edgePointIndex(const unsigned long& idx) const
                    {
                        return _edgePointIndices[idx];
                    }

                    std::list<unsigned long>& edgePointFacetIndex(const unsigned long& idx) const
                    {
                        return _edgePointFacetIndices[idx];
                    }

                    unsigned long edgePointEdgeIndex(const unsigned long& idx) const
                    {
                        return _edgePointEdgeIndices[idx];
                    }

                    void setEdgePointFacetIndex(const unsigned long& idx, const unsigned long& value)
                    {
                        if (_edgePointFacetIndices.find(idx) == _edgePointFacetIndices.end())
                        {
                            _edgePointFacetIndices.insert(std::make_pair(idx, std::list<unsigned long>()));
                            _edgePointFacetIndices[idx].push_back(value);
                        }
                        else
                        {
                            if (std::find(_edgePointFacetIndices[idx].begin(), _edgePointFacetIndices[idx].end(), value) == _edgePointFacetIndices[idx].end())
                                _edgePointFacetIndices[idx].push_back(value);
                        }
                    }

                    void setEdgePointEdgeIndex(const unsigned long& idx, const unsigned long& value)
                    {
                        // std::cout << "     setEdgePointEdgeIndex(" << idx << "," << value << ")" << std::endl;
                        if (_edgePointEdgeIndices.find(idx) == _edgePointEdgeIndices.end())
                            _edgePointEdgeIndices.insert(std::make_pair(idx, value));
                        else
                            _edgePointEdgeIndices[idx] = value;
                    }

                    bool addEdgePoint(const Vector3& point, const unsigned long& pointIdx, const unsigned long& edgeIdx, const unsigned long& facetIdx)
                    {
                        // std::cout << "    add edge point " << point << ", index " << pointIdx << ", facet index " << facetIdx << ", edge index " << edgeIdx << std::endl;
                        std::deque<Vector3>::iterator pPos = std::find(_edgePoints.begin(), _edgePoints.end(), point);
                        if (pPos == _edgePoints.end())
                        {
                            _edgePoints.push_back(point);
                            setEdgePointEdgeIndex(_edgePoints.size() - 1, edgeIdx);
                            setEdgePointFacetIndex(_edgePoints.size() - 1, facetIdx);
                            _edgePointIndices.insert(std::make_pair(_edgePoints.size() - 1, pointIdx));

                            return true;
                        }
                        else
                        {
                            /*std::cout << "      point " << point << " already registered as edge point!" << std::endl;
                            unsigned long pIdx = 0;
                            for (std::deque<Vector3>::iterator vit = _edgePoints.begin(); vit != _edgePoints.end(); vit++)
                            {
                                if (*vit == *pPos)
                                {
                                    std::cout << "      with edge index: " << edgePointEdgeIndex(pIdx) << ", facet indices: ";
                                    std::list<unsigned long>& fInds = edgePointFacetIndex(pIdx);
                                    for (std::list<unsigned long>::const_iterator it = fInds.begin(); it != fInds.end(); it++)
                                        std::cout << *it << " ";

                                    std::cout << std::endl;
                                    break;
                                }
                                pIdx++;
                            }*/

                            return false;
                        }
                    }

                    unsigned long numVertices() const { return _vertices.size(); }

                    const Vector3& vertex(const unsigned long& idx) const
                    {
                        return _vertices.at(idx);
                    }

                    Vector3& vertex(const unsigned long& idx)
                    {
                        return _vertices.at(idx);
                    }

                    unsigned long vertexIndex(const unsigned long& idx) const
                    {
                        return _vertexIndices[idx];
                    }

                    std::list<unsigned long>& vertexFacetIndex(const unsigned long& idx) const
                    {
                        return _vertexFacetIndices[idx];
                    }

                    void setVertexFacetIndex(const unsigned long& idx, const unsigned long& value)
                    {
                        // std::cout << "  setVertexFacetIndex(" << idx << "," << value << ")" << std::endl;
                        if (_vertexFacetIndices.find(idx) == _vertexFacetIndices.end())
                            _vertexFacetIndices.insert(std::make_pair(idx, std::list<unsigned long>()));

                        /*std::list<unsigned long>& adjFacets = _vertexFacetIndices[idx];
                        if (std::find(adjFacets.begin(), adjFacets.end(), value) == adjFacets.end())
                        {
                            // std::cout << "  INSERT: Vertex " << idx << " adjacent to facet " << value << std::endl;

                            // std::cout << "    " << value << " not registered for " << idx << ", appending to facet list." << std::endl;
                            adjFacets.push_back(value);
                            std::cout << "  adjacent facets for vertex " << this->vertex(this->vertexIndex(idx)) << ":";
                            for (std::list<unsigned long>::const_iterator it = adjFacets.begin(); it != adjFacets.end(); it++)
                                std::cout << " " << *it;

                            std::cout << std::endl;
                        }*/
                    }

                    bool addVertex(const Vector3& point, const unsigned long& pointIdx, const unsigned long& facetIdx)
                    {
                        // std::cout << "addVertex(" << point << "," << pointIdx << "," << facetIdx << ")" << std::endl;
                        if (std::find(_vertices.begin(), _vertices.end(), point) == _vertices.end())
                        {
                            _vertices.push_back(point);
                            _vertexFacetIndices.insert(std::make_pair(_vertices.size() - 1, std::list<unsigned long>()));
                            _vertexFacetIndices[_vertices.size() - 1].push_back(facetIdx);
                            _vertexIndices.insert(std::make_pair(_vertices.size() - 1, pointIdx));
                            _vertexRegistryTimeAdjacency.insert(std::make_pair(point, _vertices.size() - 1));
                            return true;
                        }
                        else
                        {
                            std::list<unsigned long>& adjFacets = _vertexFacetIndices[_vertexRegistryTimeAdjacency[point]];
                            if (std::find(adjFacets.begin(), adjFacets.end(), facetIdx) == adjFacets.end())
                            {
                                // std::cout << " vertex " << point << " already known, but not with facet index " << facetIdx << ", registering." << std::endl;
                                adjFacets.push_back(facetIdx);
                            }
                            return false;
                        }
                    }

                    void clearEdgePointLists(bool allEdges = true, unsigned long facetIdx = 0)
                    {
                        if (allEdges)
                        {
                            _edgePoints.clear();
                            _edgePointFacetIndices.clear();
                            _edgePointEdgeIndices.clear();
                            _edgePointIndices.clear();
                        }
                        else
                        {
                            {
                                std::vector<unsigned long> edgePtIndices;
                                for (std::map<unsigned long, std::list<unsigned long> >::iterator lit = _edgePointFacetIndices.begin(); lit != _edgePointFacetIndices.end(); lit++)
                                {
                                    std::list<unsigned long>& efInds = lit->second;
                                    if (std::find(efInds.begin(), efInds.end(), facetIdx) != efInds.end())
                                    {
                                        if (std::find(edgePtIndices.begin(), edgePtIndices.end(), lit->first) == edgePtIndices.end())
                                            edgePtIndices.push_back(lit->first);
                                    }
                                }

                                // std::cout << " edge points to remove: " << edgePtIndices.size() << std::endl;

                                for (std::vector<unsigned long>::const_iterator it = edgePtIndices.begin(); it != edgePtIndices.end(); it++)
                                {
                                    _edgePointEdgeIndices.erase(*it);
                                    _edgePointFacetIndices.erase(*it);

                                    std::map<unsigned long, unsigned long>::iterator eit = _edgePointIndices.end();
                                    for (std::map<unsigned long, unsigned long>::iterator eIt = _edgePointIndices.begin(); eIt != _edgePointIndices.end(); eIt++)
                                    {
                                        if (eIt->second == *it)
                                        {
                                            eit = eIt;
                                            break;
                                        }
                                    }
                                    if (eit != _edgePointIndices.end())
                                    {
                                        _edgePoints.erase(_edgePoints.begin() + eit->first);
                                        _edgePointIndices.erase(eit);
                                    }
                                }

                                /*for (std::map<unsigned long, std::list<unsigned long> >::iterator pfit = _edgePointFacetIndices.begin(); pfit != _edgePointFacetIndices.end(); pfit++)
                                {
                                    std::list<unsigned long>& facetInds = pfit->second;
                                    facetInds.remove(facetIdx);
                                }*/

                                /*std::vector<unsigned long> edgePointsToRemove;
                                for (std::map<unsigned long, std::deque<unsigned long> >::const_iterator pfit = _edgePointFacetIndices.begin(); pfit != _edgePointFacetIndices.end(); pfit++)
                                {
                                    if (pfit->second == (unsigned long) facetIdx)
                                        edgePointsToRemove.push_back(pfit->first);
                                }

                                std::cout << " removing edge points with facet index " << facetIdx << " from edge point lists: ";
                                for (std::vector<unsigned long>::const_iterator epit = edgePointsToRemove.begin(); epit != edgePointsToRemove.end(); epit++)
                                {
                                    std::cout << *epit << " .. ";
                                }
                                std::cout << std::endl;

                                std::map<unsigned long, unsigned long>::iterator pfit = _edgePointFacetIndices.begin();
                                while (pfit != _edgePointFacetIndices.end())
                                {
                                    if (pfit->first == (unsigned long) facetIdx)
                                    {
                                       _edgePointFacetIndices.erase(pfit++);
                                    }
                                    else
                                    {
                                        ++pfit;
                                    }
                                }

                                pfit = _edgePointEdgeIndices.begin();
                                while (pfit != _edgePointEdgeIndices.end())
                                {
                                    if (std::find(edgePointsToRemove.begin(), edgePointsToRemove.end(), pfit->first) != edgePointsToRemove.end())
                                    {
                                       _edgePointEdgeIndices.erase(pfit++);
                                    }
                                    else
                                    {
                                        ++pfit;
                                    }
                                }

                                pfit = _edgePointIndices.begin();
                                while (pfit != _edgePointIndices.end())
                                {
                                    if (std::find(edgePointsToRemove.begin(), edgePointsToRemove.end(), pfit->first) != edgePointsToRemove.end())
                                    {
                                       _edgePointIndices.erase(pfit++);
                                    }
                                    else
                                    {
                                        ++pfit;
                                    }
                                }

                                for (std::vector<unsigned long>::const_iterator epit = edgePointsToRemove.begin(); epit != edgePointsToRemove.end(); epit++)
                                {
                                    _edgePoints.erase(_edgePoints.begin() + *epit);
                                }*/
                            }
                        }
                    }

                    void clearVertexPointLists(bool allVertices = true, int facetIdx = -1)
                    {
                        if (allVertices)
                        {
                            _vertices.clear();
                            _vertexFacetIndices.clear();
                            _vertexIndices.clear();
                        }
                        else
                        {
                            if (facetIdx >= 0)
                            {
                                for (std::map<unsigned long, std::list<unsigned long> >::iterator pfit = _edgePointFacetIndices.begin(); pfit != _edgePointFacetIndices.end(); pfit++)
                                {
                                    std::list<unsigned long>& facetInds = pfit->second;
                                    facetInds.remove(facetIdx);
                                }
                            }

                            /*if (facetIdx > 0)
                            {
                                std::vector<unsigned long> verticesToRemove;
                                for (std::map<unsigned long, unsigned long>::const_iterator pfit = _vertexFacetIndices.begin(); pfit != _vertexFacetIndices.end(); pfit++)
                                {
                                    if (pfit->second == (unsigned long) facetIdx)
                                        verticesToRemove.push_back(pfit->first);
                                }

                                std::cout << " removing vertices with facet index " << facetIdx << " from vertex lists: ";
                                for (std::vector<unsigned long>::const_iterator epit = verticesToRemove.begin(); epit != verticesToRemove.end(); epit++)
                                {
                                    std::cout << *epit << " .. ";
                                }
                                std::cout << std::endl;

                                std::map<unsigned long, unsigned long>::iterator pfit = _vertexFacetIndices.begin();
                                while (pfit != _vertexFacetIndices.end())
                                {
                                    if (pfit->first == (unsigned long) facetIdx)
                                    {
                                       _vertexFacetIndices.erase(pfit++);
                                    }
                                    else
                                    {
                                        ++pfit;
                                    }
                                }

                                pfit = _vertexIndices.begin();
                                while (pfit != _vertexIndices.end())
                                {
                                    if (std::find(verticesToRemove.begin(), verticesToRemove.end(), pfit->first) != verticesToRemove.end())
                                    {
                                       _vertexIndices.erase(pfit++);
                                    }
                                    else
                                    {
                                        ++pfit;
                                    }
                                }

                                for (std::vector<unsigned long>::const_iterator epit = verticesToRemove.begin(); epit != verticesToRemove.end(); epit++)
                                {
                                    _vertices.erase(_vertices.begin() + *epit);
                                }
                            }*/
                        }

                    }

                    void clearPointLists()
                    {
                        _clusterPoints.clear();
                        clearEdgePointLists();
                        clearVertexPointLists();
                    }

                    unsigned long numChildren() const
                    {
                        return _children.size();
                    }

                    const LGCPointCluster* childCluster(const unsigned long& idx) const
                    {
                        //std::cout << "LGCPointCluster::childCluster(" << idx << ")" << std::endl;
                        if (idx < _children.size())
                        {
                            //std::cout << " OK:" << idx << " < " << _children.size() << std::endl;
                            return _children.at(idx);
                        }
                        //std::cout << " FAIL: " << idx << " >= " << _children.size() << std::endl;
                        return NULL;
                    }

                    LGCPointCluster* childCluster(const unsigned long& idx)
                    {
                        //std::cout << "LGCPointCluster::childCluster(" << idx << ")" << std::endl;
                        if (idx < _children.size())
                        {
                            //std::cout << " OK:" << idx << " < " << _children.size() << std::endl;
                            return _children.at(idx);
                        }
                        //std::cout << " FAIL: " << idx << " >= " << _children.size() << std::endl;
                        return NULL;
                    }

                    void addChild(LGCPointCluster* child)
                    {
                        std::cout << "LGCPointCluster::addChild(" << child << ")" << std::endl;
                        if (std::find(_children.begin(), _children.end(), child) == _children.end())
                        {
                            std::cout << " OK: Not in child list yet." << std::endl;
                            _children.push_back(child);
                        }
                        else
                        {
                            std::cout << " FAIL: Already in child list!" << std::endl;
                        }
                    }

                    virtual void draw(const core::visual::VisualParams *);

                    bool writePCDFile(const std::string& = "");
                    void buildChildHierarchy();
                    void fitObbs();
                    bool readFromPCD();

                    void buildClusterDrawList();

                    LGCObb<LGCDataTypes>* clusterObb() const;
                    const Vec<3, Real>& clusterCentroid() const;

                    inline bool computeOcTree() const { return _computeOctree; }
                    void setComputeOcTree(bool compute);

                    inline bool computeKdTree() const { return _computeKdtree; }
                    void setComputeKdTree(bool compute);

                    inline bool getDoSegmentation() const { return _computeSegmentation; }
                    inline void setDoSegmentation(bool segment) { _computeSegmentation = segment; }

                    inline bool showOcTree() const { return _showOcTree; }
                    inline void setShowOcTree(bool show) { _showOcTree = show; }
                    inline bool showOcTreeVerbose() const { return _showOcTreeVerbose; }
                    inline void setShowOcTreeVerbose(bool show) { _showOcTreeVerbose = show; }

                    inline bool showKdTree() const { return _showKdTree; }
                    inline void setShowKdTree(bool show) { _showKdTree = show; }

                    inline bool drawVerbose() const { return _drawVerbose; }
                    inline void setDrawVerbose(bool verbose) { _drawVerbose = verbose; }

                    inline bool showObbPlanes() const { return _showOBBPlanes; }
                    inline void setShowObbPlanes(bool show) { _showOBBPlanes = show; }

                    void setFacetRange(const unsigned long& minVal, const unsigned long& maxVal)
                    {
                        _facetRangeMin = minVal;
                        _facetRangeMax = maxVal;
                    }

                    inline unsigned long minFacetRange() const { return _facetRangeMin; }
                    inline unsigned long maxFacetRange() const { return _facetRangeMax; }

                    inline void setOriginalModelPtr(Exact_Polyhedron_3* polyhedron) { _originalModel = polyhedron; }

                    inline bool drawOBBTreeStructure() { return _drawObbTreeStructure; }
                    inline void setDrawOBBTreeStructure(const bool& structureDraw) { _drawObbTreeStructure = structureDraw; }

                    inline bool drawOBBVolumes() { return _drawObbVolumes; }
                    inline void setDrawOBBVolumes(const bool& volumesDraw) { _drawObbVolumes = volumesDraw; }

                    inline void setDrawLimits(const int& minDepth, const int& maxDepth)
                    {
                        _minDrawLimit = minDepth;
                        _maxDrawLimit = maxDepth;
                    }

                    void buildPQPModel();
                    PQP_FlatModel* pqpModel()
                    {
                        return _pqp_tree;
                    }

                    pcl::gpu::Octree* clusterOctreeGPU();
                    OcTreeSinglePoint* clusterOctreeCPU();
                    OctreeSearch* clusterOctreeSearch();

                    void adjustFacetIndices();

                    void buildSubCluster();

                    const sofa::helper::vector<std::pair<unsigned int, sofa::component::topology::BaseMeshTopology::Triangle> >& getTriangleIndices()
                    {
                        return _indices;
                    }

                    void setTriangleIndices(const sofa::helper::vector<std::pair<unsigned int, sofa::component::topology::BaseMeshTopology::Triangle> >& indices)
                    {
                        _indices = indices;
                    }

                    void setClusterColor(const Vec4f& color) { _clusterColor = color; }
                    const Vec4f getClusterColor() const { return _clusterColor; }

                protected:
                    friend class LGCPointClusterPrivate<LGCDataTypes>;
                    friend class LGCPointModel;

                    std::deque<Vector3> _clusterPoints;
                    std::deque<Vector3> _edgePoints;
                    std::deque<Vector3> _vertices;
                    std::deque<LGCPointCluster*> _children;

                    std::vector<Vec<3, Real> > _allClusterPts;
#ifdef LGC_POINTCLUSTER_DEBUG_ALL_CLUSTER_POINTS
                    std::vector<Vector3> _allClusterPtsDraw;
#endif
                    Vec4f _clusterColor;

                    mutable std::map<unsigned long, unsigned long> _surfacePointFacetIndices;
                    mutable std::map<unsigned long, std::list<unsigned long> > _edgePointFacetIndices;
                    mutable std::map<unsigned long, std::list<unsigned long> > _vertexFacetIndices;

                    mutable std::map<unsigned long, unsigned long> _edgePointEdgeIndices;

                    mutable std::map<unsigned long, unsigned long> _surfacePointIndices;
                    mutable std::map<unsigned long, unsigned long> _edgePointIndices;
                    mutable std::map<unsigned long, unsigned long> _vertexIndices;
                    mutable std::map<Vector3, unsigned long> _vertexRegistryTimeAdjacency;

                    bool _showOcTree, _showOcTreeVerbose;
                    bool _showKdTree;
                    bool _showOBBPlanes;
                    bool _computeOctree, _computeKdtree;
                    bool _drawVerbose;
                    bool _drawObbTreeStructure;
                    bool _drawObbVolumes;
                    bool _computeSegmentation;

                    int _minDrawLimit, _maxDrawLimit;

                    LGCPointCluster* _parentCluster;
                    Exact_Polyhedron_3* _originalModel;
                    LGCCollisionModel<LGCDataTypes>* _lgcCollisionModel;

                    long _facetRangeMin, _facetRangeMax;
                    std::vector<long> _childFacets;
                    bool _facetRangeAdjusted;
                    long _clusterIndex;

                    sofa::helper::vector<std::pair<unsigned int, sofa::component::topology::BaseMeshTopology::Triangle> > _indices;

                    typedef sofa::helper::vector<Vec<4, long> > VecPointMask;
                    typedef sofa::helper::vector<Vec<3, Real> > VecPointMaskDiagonal;
                    typedef sofa::helper::vector<Vec<3, Real> > VecPointMaskOrigin;

                    VecPointMask _pointMasks;
                    VecPointMaskDiagonal _pointMaskSizeX;
                    VecPointMaskDiagonal _pointMaskSizeY;
                    VecPointMaskOrigin _pointMaskOrigins;

                    std::vector<Plane3D<Vec3Types::Real> > _cvPlanes;
                    std::vector<Plane3Drawable<Vec3Types::Real> > _cvPlaneDrawables;

                private:
                    LGCPointClusterPrivate<LGCDataTypes>* d;
                    PQP_FlatModel* _pqp_tree;
            };

            template <class LGCDataTypes>
            std::ostream &
            operator<<(std::ostream &os, const LGCPointCluster<LGCDataTypes> &cluster)
            {
                os << "PointCluster(" << cluster.getName() << " -- vertices: " << cluster.numVertices() << ", edge points: " << cluster.numEdgePoints() << ", surface points: " << cluster.numSurfacePoints() << ", children: " << cluster.numChildren() << ")" << std::endl;

                //if (cluster.f_printLog.getValue())
                {
                    if (cluster.numChildren() > 0)
                    {
                        for (unsigned int i = 0; i < cluster.numChildren(); i++)
                        {
                            std::cout << "Child clusters: " << cluster.numChildren() << std::endl;
                            if (cluster.childCluster(i) != NULL)
                                os << " * " << i << ": " << *(cluster.childCluster(i)) << std::endl;
                        }
                    }

                    os << "Vertices: ";
                    for (unsigned int i = 0; i < cluster.numVertices(); i++)
                    {
                        os << cluster.vertex(i);
                        if (i < cluster.numVertices())
                            os << " / ";
                    }
                }
                return os;
            }
        }
    }
}

#endif // LGCPOINTCLUSTER_H
