#ifndef LGCPOINTCLUSTER_INL
#define LGCPOINTCLUSTER_INL

#include "LGCTriangleModelPolyhedron.h"
#include "LGCPointCluster.h"

#include "LGCUtil.h"
#include "LGCOBBUtils.h"
#include "LGCPlane3.h"
#include "LGCPlane3Drawable.h"
#include "LGCObbDrawable.h"
#include "LGCCube.h"

#include <sofa/core/visual/VisualParams.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_search.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/impl/octree_pointcloud.hpp>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/octree/internal.hpp>
#include <pcl/gpu/containers/impl/device_array.hpp>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/helper/AdvancedTimer.h>

#include <LGCTriangleModel_TypeDefs.h>
#include <CGAL/linear_least_squares_fitting_3.h>

#include <fstream>

#include <PQP/include/PQP.h>

#ifndef LGC_POINT_MODEL_GRID_SIZE
#define LGC_POINT_MODEL_GRID_SIZE 16
#endif

#ifndef LGC_POINT_MODEL_GRID_TILE_SIZE
#define LGC_POINT_MODEL_GRID_TILE_SIZE 8
#endif

using namespace sofa::defaulttype;
namespace sofa
{
    namespace component
    {
        namespace collision
        {
            template <typename PointT = pcl::LGCPointType, typename Dist = flann::L2_Simple<float> >
            class LGCKdTree: public pcl::KdTreeFLANN<PointT, Dist>
            {
                typedef pcl::KdTreeFLANN<PointT, Dist> super;
                typedef typename super::FLANNIndex FLANNIndex;
                typedef typename flann::KDTreeSingleIndex<Dist> KDTreeIndex;

                bool _kdTreeValid;
                KDTreeIndex* _kdTreeIndex;

                public:
                    typedef typename flann::KDTreeSingleIndex<Dist>::NodePtr NodePtr;
                    typedef boost::shared_ptr <std::vector<int> > IndicesPtr;
                    typedef boost::shared_ptr <const std::vector<int> > IndicesConstPtr;

                    typedef pcl::PointCloud<PointT> PointCloud;
                    typedef boost::shared_ptr<PointCloud> PointCloudPtr;
                    typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

                private:
                    NodePtr _kdTreeRoot;

                public:
                    LGCKdTree(bool sorted = true): pcl::KdTreeFLANN<PointT, Dist>(sorted)
                    {
                        _kdTreeValid = false;
                        _kdTreeIndex = NULL;
                        _kdTreeRoot = NULL;
                    }

                    inline bool kdTreeValid() const { return _kdTreeValid; }

                    flann::Index<Dist>* flannIndex()
                    {
                        return this->flann_index_;
                    }

                    NodePtr kdTreeRoot() const { return _kdTreeRoot; }
                    flann::KDTreeSingleIndex<Dist>* kdTreeIndex() const { return _kdTreeIndex; }

                    virtual void setInputCloud(const PointCloudConstPtr &cloud, const IndicesConstPtr &indices)
                    {
                        std::cout << "LGCKdTree::setInputCloud()" << std::endl;
                        super::cleanup();   // Perform an automatic cleanup of structures

                        super::epsilon_ = 0.0;   // default error bound value
                        super::dim_ = super::point_representation_->getNumberOfDimensions (); // Number of dimensions - default is 3 = xyz

                        super::input_   = cloud;
                        super::indices_ = indices;

                        // Allocate enough data
                        if (!super::input_)
                        {
                            PCL_ERROR ("[pcl::KdTreeFLANN::setInputCloud] Invalid input!\n");
                            return;
                        }
                        if (indices != NULL)
                        {
                            super::total_nr_points_ = (int) super::indices_->size ();
                            super::convertCloudToArray (*super::input_, *super::indices_);
                        }
                        else
                        {
                            super::total_nr_points_ = (int) super::input_->points.size ();
                            super::convertCloudToArray (*super::input_);
                        }

                        std::cout << "Creating FLANN k-d-tree index" << std::endl;
                        super::flann_index_ = boost::shared_ptr<FLANNIndex>(new FLANNIndex (flann::Matrix<float> (super::cloud_, super::index_mapping_.size (), super::dim_),
                        flann::KDTreeSingleIndexParams(15)));
                        super::flann_index_->buildIndex();

                        flann::Index<Dist>* flIdx = dynamic_cast<flann::Index<Dist>*>(super::flann_index_.get());
                        if (flIdx)
                        {
                            std::cout << "Cast part 1 OK: Index type reported as " << flIdx->getType() << std::endl;
                            flann::NNIndex<Dist>* flNNIdx = flIdx->getIndex();
                            flann::KDTreeSingleIndex<Dist>* kdTreeIdx = dynamic_cast<flann::KDTreeSingleIndex<Dist>*> (flNNIdx);
                            if (kdTreeIdx)
                            {
                                std::cout << "Casted to flann::KDTreeSingleIndex<Dist>: Index type reported as " << kdTreeIdx->getType() << std::endl;
                                _kdTreeRoot = kdTreeIdx->kdTree();
                                if (_kdTreeRoot)
                                {
                                    _kdTreeIndex = kdTreeIdx;
                                    _kdTreeValid = true;
                                }
                                else
                                {
                                    std::cout << "ERROR: Received NULL k-d-tree root from FLANN!" << std::endl;
                                }
                            }
                            else
                            {
                                std::cout << "ERROR: Flann index type received is: " << super::flann_index_->getType() << ", expected: " << flann::FLANN_INDEX_KDTREE_SINGLE << std::endl;
                            }
                        }
                    }
            };

            //typedef pcl::octree::OctreeBase<int, pcl::octree::OctreeLeafDataTVector<int>, pcl::octree::OctreeBranch> OcTree;
            //typedef pcl::octree::OctreePointCloudPointVector<pcl::LGCPointType, pcl::octree::OctreeLeafDataTVector<int>, OcTree > OcTreeSinglePoint;

            //typedef pcl::octree::OctreeBase</*int, pcl::octree::OctreeContainerPointIndices*/> OcTree;
            //typedef pcl::octree::OctreePointCloudPointVector<pcl::LGCPointTypeMin/*, pcl::octree::OctreeContainerPointIndices, OcTree*/ > OcTreeSinglePoint;

            template <class LGCDataTypes>
            class LGCPointClusterPrivate
            {
                public:
                    typedef typename LGCDataTypes::Real Real;

                    //typedef pcl::octree::OctreeBranchNode<pcl::octree::OctreeContainerPointIndices> OcTreeBranch;

                    //typedef OcTreeSinglePoint::BranchNode OcTreeBranch;
                    //typedef pcl::octree::OctreeNode OcTreeNode;
                    //typedef pcl::octree::OctreeContainerPointIndices OcTreeLeaf;
                    //typedef pcl::octree::OctreeKey OcTreeKey;

                    typedef OcTree::BranchNode OcTreeBranch;
                    typedef pcl::octree::OctreeNode OcTreeNode;
                    typedef OcTree::LeafNode OcTreeLeaf;
                    typedef pcl::octree::OctreeKey OcTreeKey;

                    typedef LGCKdTree<pcl::LGCPointTypeMin>::NodePtr KdTreeNode;
                    typedef flann::KDTreeSingleIndex<flann::L2_Simple<float> >::BoundingBox KdTreeBoundingBox;

                    LGCPointCluster<LGCDataTypes>* _cluster;
                    std::string _pcdFile;
                    std::string _pcdBaseName;
                    pcl::PointCloud<pcl::LGCPointTypeMin>::Ptr _cloud;
                    pcl::PointCloud<pcl::LGCPointTypeMin>::Ptr _cloud_filtered;

                    // pcl::VoxelGrid<pcl::LGCPointType> _voxelGrid;
                    // pcl::PointCloud<pcl::LGCPointType>::Ptr _voxelCloud;

                    pcl::gpu::Octree::PointCloud* _cloud_device;
                    pcl::gpu::Octree* _octree_device;

                    LGCKdTree<pcl::LGCPointTypeMin>*_kdTree;
                    OcTreeSinglePoint* _spOcTree;
                    OctreeSearch* _searchOcTree;

                    float* _cloudArray;
                    std::vector<int> _indexMapping;
                    bool _identityMapping;

                    Vec<3,Real> _centroid;
                    Vec<3,Real> _eigenVectors[3];
                    Plane3D<Real>* _obbPlanes[6];
                    Plane3Drawable<Real>* _obbPlaneDrawables[6];

                    Plane3Drawable<Real>* _eigenPlanes[3];
                    Vec<3,Real> _mostDistPts[6];

                    Vec<3,Real> _obbCorners[8];
                    LGCObb<LGCDataTypes>* _clusterObb;
                    ObbDrawable<LGCDataTypes>* _clusterObbDrawable;

                    Vec<3, Real> _obbCenter;
                    Vec<3, Real> _obbExtents;
                    Matrix4 _obbAxes;

                    LGCPointClusterObb<LGCDataTypes>* _pcObb;

                    std::map<unsigned long, std::vector<pcl::LGCPointTypeMin> > _segmentedPlanes;
                    std::map<unsigned long, std::vector<pcl::LGCPointTypeMin> > _segmentedEdges;
                    std::map<unsigned long, std::vector<pcl::LGCPointTypeMin> > _segmentedVertices;
                    std::map<unsigned long, pcl::PointCloud<pcl::LGCPointTypeMin>::Ptr> _segmentedClusters;

                    LGCPointClusterPrivate(LGCPointCluster<LGCDataTypes>* cluster): _cluster(cluster), _cloudArray(NULL), _kdTree(NULL), _spOcTree(NULL), _octree_device(NULL), _cloud_device(NULL), _searchOcTree(NULL)
                    {
                        _clusterObb = NULL;
                        _clusterObbDrawable = NULL;

                        _pcObb = NULL;

                        for (int i = 0; i < 3; i++)
                        {
                            _eigenPlanes[i] = NULL;
                        }

                        for (int i = 0; i < 6; i++)
                        {
                            _obbPlanes[i] = NULL;
                            _obbPlaneDrawables[i] = NULL;
                        }
                    }

                    ~LGCPointClusterPrivate()
                    {
                        if (_spOcTree)
                            delete _spOcTree;

                        if (_searchOcTree)
                            delete _searchOcTree;

                        if (_kdTree)
                            delete _kdTree;

                        for (int i = 0; i < 6; i++)
                        {
                            if (_obbPlanes[i])
                                delete _obbPlanes[i];

                            if (_obbPlaneDrawables[i])
                                delete _obbPlaneDrawables[i];
                        }

                        if (_pcObb)
                            delete _pcObb;

                        if (_clusterObb)
                            delete _clusterObb;

                        if (_clusterObbDrawable)
                            delete _clusterObbDrawable;

                        if (_octree_device)
                        {
                            //_octree_device->clear();
                            delete _octree_device;
                        }

                        if (_cloud_device)
                        {
                            //_cloud_device->release();
                            delete _cloud_device;
                        }
                    }

                    /*void downSamplePCL()
                    {
                        if (_cloud.get())
                        {
                              _voxelCloud = pcl::PointCloud<pcl::LGCPointType>::Ptr(new pcl::PointCloud<pcl::LGCPointType>);
                              _voxelGrid.setInputCloud(_cloud);
                              _voxelGrid.setLeafSize (0.5f, 0.5f, 0.5f);
                              _voxelGrid.setSaveLeafLayout(true);
                              _voxelGrid.setFilterLimitsNegative(true);
                              _voxelGrid.filter(*_voxelCloud);
                              _cloud_filtered = _voxelCloud;
                              std::cout << "PointCloud after filtering has: " << _voxelCloud->points.size()  << " data points." << std::endl;
                        }
                    }*/

                    void computeTrees()
                    {
                        // Creating the KdTree object for the search method of the extraction
                        if (_cluster->computeKdTree() && _cluster->parentCluster() != NULL)
                        {
                            _kdTree = new LGCKdTree<pcl::LGCPointTypeMin>();
                            std::cout << "=== GENERATE kd-tree ===" << std::endl;
                            _kdTree->setInputCloud((LGCKdTree<pcl::LGCPointTypeMin>::PointCloudConstPtr)_cloud, LGCKdTree<pcl::LGCPointTypeMin>::IndicesConstPtr());
                        }

                        if (_cluster->computeOcTree() && _cluster->parentCluster() != NULL)
                        {
                            _spOcTree = new OcTreeSinglePoint(0.05f);
                            std::cout << "=== GENERATE single point oc-tree ===" << std::endl;
                            _spOcTree->setInputCloud(_cloud);
                            _spOcTree->addPointsFromInputCloud();

                            _searchOcTree = new OctreeSearch(0.05f);
                            _searchOcTree->setInputCloud(_cloud);
                            _searchOcTree->addPointsFromInputCloud();
                        }
                    }

                    void planarSegmentation()
                    {
                        pcl::PointCloud<pcl::LGCPointTypeMin>::Ptr cloud_f(new pcl::PointCloud<pcl::LGCPointTypeMin>);
                        // Create the segmentation object for the planar model and set all the parameters
                        pcl::SACSegmentation<pcl::LGCPointTypeMin> seg;
                        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
                        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
                        pcl::PointCloud<pcl::LGCPointTypeMin>::Ptr cloud_plane(new pcl::PointCloud<pcl::LGCPointTypeMin> ());
                        seg.setOptimizeCoefficients(true);
                        seg.setModelType(pcl::SACMODEL_PLANE);
                        seg.setMethodType(pcl::SAC_RANSAC);
                        seg.setMaxIterations(100);
                        seg.setDistanceThreshold(0.002);

                        int i = 0, iterations = 0; //, nr_points = (int) _cloud_filtered->points.size ();
                        while (_cloud_filtered->points.size () > 0 && iterations < 100)
                        {
                            // Segment the largest planar component from the remaining cloud
                            seg.setInputCloud (_cloud_filtered);
                            seg.segment (*inliers, *coefficients);
                            if (inliers->indices.size () == 0)
                            {
                                // std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
                                std::cout << "Remaining points in input data set: " << _cloud_filtered->points.size() << std::endl;

                                std::vector<pcl::LGCPointTypeMin> planePts;
                                std::vector<pcl::LGCPointTypeMin> edgePts;
                                std::vector<pcl::LGCPointTypeMin> vertices;

                                for (size_t k = 0; k < _cloud_filtered->points.size(); ++k)
                                {
                                    planePts.push_back(_cloud_filtered->points[k]);

                                    /*if (_cloud_filtered->points[k].pointType == 0)
                                        planePts.push_back(_cloud_filtered->points[k]);
                                    else if (_cloud_filtered->points[k].pointType == 1)
                                        edgePts.push_back(_cloud_filtered->points[k]);
                                    else if (_cloud_filtered->points[k].pointType == 2)
                                        vertices.push_back(_cloud_filtered->points[k]);*/
                                }
                                _segmentedPlanes.insert(std::make_pair((unsigned long)i, planePts));
                                _segmentedEdges.insert(std::make_pair((unsigned long)i, edgePts));
                                _segmentedVertices.insert(std::make_pair((unsigned long)i, vertices));

                                _segmentedClusters.insert(std::make_pair((unsigned long) i, _cloud_filtered));
                                std::cout << " * inserted segmented cluster 1: " << i << ", point count: " << _cloud_filtered->points.size() << "; surface points: " << planePts.size() << ", edge points: " << edgePts.size() << std::endl;
                                i++;

                                break;
                            }

                            // Extract the planar inliers from the input cloud
                            pcl::ExtractIndices<pcl::LGCPointTypeMin> extract;
                            extract.setInputCloud (_cloud_filtered);
                            extract.setIndices (inliers);
                            extract.setNegative(false);

                            extract.filter (*cloud_plane);
                            std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

                            std::vector<pcl::LGCPointTypeMin> planePts;
                            std::vector<pcl::LGCPointTypeMin> edgePts;
                            std::vector<pcl::LGCPointTypeMin> vertices;
                            for (size_t k = 0; k < cloud_plane->points.size(); ++k)
                            {
                                planePts.push_back(cloud_plane->points[k]);
                                /*if (cloud_plane->points[k].pointType == 0)
                                    planePts.push_back(cloud_plane->points[k]);
                                else if (cloud_plane->points[k].pointType == 1)
                                    edgePts.push_back(cloud_plane->points[k]);
                                else if (cloud_plane->points[k].pointType == 2)
                                    vertices.push_back(cloud_plane->points[k]);*/
                            }

                            _segmentedPlanes.insert(std::make_pair((unsigned long)i, planePts));
                            _segmentedEdges.insert(std::make_pair((unsigned long)i, edgePts));
                            _segmentedVertices.insert(std::make_pair((unsigned long)i, vertices));

                            {
                                pcl::PointCloud<pcl::LGCPointTypeMin>::Ptr plane(new pcl::PointCloud<pcl::LGCPointTypeMin>(*(cloud_plane.get())));
                                _segmentedClusters.insert(std::make_pair((unsigned long) i, plane));
                                std::cout << " * inserted segmented cluster 2: " << i << ", point count: " << plane->points.size() << "; surface points: " << planePts.size() << ", edge points: " << edgePts.size() << std::endl;
                                i++;
                            }

                            // Remove the planar inliers, extract the rest
                            extract.setNegative (true);
                            extract.filter (*cloud_f);
                            _cloud_filtered = cloud_f;
                            iterations++;
                        }

                        std::cout << "Filtered cloud points remaining     : " << _cloud_filtered->points.size() << std::endl;
                        /*if (_voxelCloud.get())
                            std::cout << "Original downsampled cloud pt. count: " << _voxelCloud->points.size() << std::endl;*/

                        if (_cloud_filtered->points.size() > 0)
                        {
                            pcl::search::KdTree<pcl::LGCPointTypeMin>::Ptr tree(new pcl::search::KdTree<pcl::LGCPointTypeMin>);
                            tree->setInputCloud(_cloud_filtered);
                            std::cout << "Input for kdtree set." << std::endl;
                            std::vector<pcl::PointIndices> cluster_indices;
                            pcl::EuclideanClusterExtraction<pcl::LGCPointTypeMin> ec;
                            ec.setClusterTolerance(0.2);
                            ec.setMinClusterSize(3);
                            ec.setMaxClusterSize(10000);
                            ec.setSearchMethod(tree);
                            ec.setInputCloud (_cloud_filtered);
                            ec.extract(cluster_indices);
                            std::cout << "Euclidian cluster extracted:" << cluster_indices.size() << std::endl;

                            int j = i + 1;
                            for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
                            {
                                pcl::PointCloud<pcl::LGCPointTypeMin>::Ptr cloud_cluster (new pcl::PointCloud<pcl::LGCPointTypeMin>);
                                for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
                                    cloud_cluster->points.push_back (_cloud_filtered->points[*pit]);

                                cloud_cluster->width = cloud_cluster->points.size ();
                                cloud_cluster->height = 1;
                                cloud_cluster->is_dense = true;

                                std::cout << "PointCloud representing the Cluster " << j << ": " << cloud_cluster->points.size () << " data points." << std::endl;
                                /*std::stringstream ss;
                                ss << sofa::helper::system::DataRepository.getPaths().at(0) << "/";
                                ss << "cloud_cluster_" << _pcdBaseName << "_" << j << ".pcd";
                                writer.write<LGCPointType> (ss.str(), *cloud_cluster, false);*/

                                std::vector<pcl::LGCPointTypeMin> cloudPts;
                                for (size_t k = 0; k < cloud_cluster->points.size (); ++k)
                                {
                                    cloudPts.push_back(cloud_cluster->points[k]);
                                }

                                _segmentedPlanes.insert(std::make_pair((unsigned long)j, cloudPts));
                                _segmentedClusters.insert(std::make_pair((unsigned long) j, cloud_cluster));
                                j++;
                            }
                        }
                    }

                    void buildChildHierarchy();
                    void fitObbToCluster();
                    void fitTopOBB();
                    void computeClusterCentroid();

                    void updateOBBHierarchy();
                    void updateOBBHierarchyRec(LGCObb<LGCDataTypes> *parent);

                    void drawCoordinateMarkerGL(float lineLength = 1.0f, float lineWidth = 1.0f, const Vec<4, Real>& xColor = Vec<4, Real>(1,0,0,1), const Vec<4, Real>& yColor = Vec<4, Real>(0,1,0,1), const Vec<4, Real>& zColor = Vec<4, Real>(0,0,1,1));
                    void drawObbVolume(const Vec<3, Real>& halfExtents, const Vec4f& color);

                    void buildClusterDrawList();

                    std::vector<Vector3> _clusterEdgePoints;
                    std::vector<Vector3> _clusterVertices;

                    void dumpOcTree();
                    void dumpOcTreeRec(OcTreeSinglePoint*, const OcTreeBranch*, const /*OcTreeSinglePoint::OctreeKey*/ pcl::octree::OctreeKey&, unsigned long);
                    void dumpKdTree();
                    void dumpKdTreeRec(KdTreeNode node);

                    void drawOcTree(const core::visual::VisualParams*, bool = false);
                    void drawOcTreeRec(const core::visual::VisualParams*, const OcTreeBranch*, const /*OcTree::OctreeKey*/ pcl::octree::OctreeKey&, unsigned long, Vector3 translation, Matrix4 rotation, bool = false);
                    void drawVoxel(const core::visual::VisualParams*, const Vector3&, const Vector3&, const Vector3&, const Vec4f&, bool);

                    void drawKdTree(const core::visual::VisualParams*);
                    void drawKdTreeRec(const core::visual::VisualParams*, KdTreeNode);

                    Vec4f randomColor()
                    {
                        double r = drand48();
                        double g = drand48();
                        double b = drand48();
                        double a = drand48();

                        if (r < 0.15f)
                            r = 0.15f + r;

                        if (g < 0.15f)
                            g = 0.15f + g;

                        if (b < 0.15f)
                            b = 0.15f + b;

                        if (a < 0.15f)
                            a = 1.0f - a;

                        return Vec4f(r,g,b,a);
                    }

                    std::map<Vector3, Vec4f> _nodeColors;
            };
        }
    }
}

using namespace sofa::component::collision;

template <class LGCDataTypes>
LGCPointClusterObb<LGCDataTypes>::LGCPointClusterObb(LGCPointCluster<LGCDataTypes>* cluster, const Vector3 &translation, const Quaternion &rotation, const double &scale, const core::CollisionModel *cm):
    LGCObb<LGCDataTypes>(translation, rotation), _cluster(cluster)
{
    LGCObb<LGCDataTypes>::_lgcCollisionModel = dynamic_cast<LGCCollisionModel<LGCDataTypes>* >(const_cast<core::CollisionModel*>(cm));
    fitToCluster();
}

template <class LGCDataTypes>
void LGCPointClusterObb<LGCDataTypes>::fitToCluster()
{
    if (!_cluster)
        return;

    std::cout << "LGCPointClusterObb<LGCDataTypes>::fitToCluster(): " << _cluster->getName() << std::endl;
    std::cout << " cluster position = " << _cluster->position() << ", orientation = " << _cluster->orientation() << std::endl;
    const LGCCollisionModel<LGCDataTypes>* cm = (LGCCollisionModel<LGCDataTypes> *)(_cluster->collisionModel());
    if (cm)
        std::cout << " model position = " << cm->position() << ", orientation = " << cm->orientation() << std::endl;

    std::vector<Vec<3, Real> > allOBBPts;
    if (this->numChildren() == 0)
    {
        std::cout << " number of vertices = " << _cluster->numVertices() << std::endl;
        for (unsigned long i = 0; i < _cluster->numVertices(); i++)
        {
            Vec<3, Real> tp = _cluster->vertex(i) - _cluster->position();
            tp = _cluster->orientation().inverseRotate(tp);
            allOBBPts.push_back(tp);
        }
    }
    else
    {
        std::cout << " add children's corner points" << std::endl;
        for (unsigned int p = 0; p < this->numChildren(); p++)
        {
            std::cout << " child " << p << ": childOffset = " << this->child(p)->lcData()._childOffset << ", childTransform = " << this->child(p)->lcData()._childTransform << std::endl;
            for (unsigned int i = 0; i <= 7; i++)
            {
                Vec<3, Real> tp = this->child(p)->obrCorner(i);
                /*Vec<3, Real> tp = this->child(p)->obrCorner(i) - _cluster->position();
                tp = _cluster->orientation().inverseRotate(tp);*/

                allOBBPts.push_back(tp);
            }
        }
    }

    unsigned int i,j;
    // compute covariance matrix
    Vec<3, Real> centroid;
    Mat<3,3,Real> C;
    sofa::helper::ComputeCovarianceMatrix(C, centroid, allOBBPts);

    // get basis vectors
    Vec<3, Real> basis[3];
    sofa::helper::GetRealSymmetricEigenvectors(basis[0], basis[1], basis[2], C);

    Vector3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    Vector3 max(FLT_MIN, FLT_MIN, FLT_MIN);

    // compute min, max projections of box on axes
    for ( i = 0; i < allOBBPts.size(); ++i )
    {
        Vector3 diff = allOBBPts[i] - centroid;
        for (j = 0; j < 3; ++j)
        {
            double length = diff * basis[j];
            if (length > max[j])
            {
                max[j] = length;
            }
            else if (length < min[j])
            {
                min[j] = length;
            }
        }
    }

    // compute center, extents
    Vec<3, Real> obbPosition = centroid;
    Vec<3, Real> obbExtents;
    for ( i = 0; i < 3; ++i )
    {
        obbPosition += basis[i] * 0.5f * (min[i]+max[i]);
        obbExtents[i] = (max[i]-min[i]) * 0.5f;
    }
    this->setHalfExtents(obbExtents);
    this->setCenter(obbPosition);

    this->setLocalAxis(0, basis[0]);
    this->setLocalAxis(1, basis[1]);
    this->setLocalAxis(2, basis[2]);

    this->wcData()._centerOffset = obbPosition;
    this->lcData()._childOffset = obbPosition;

    std::cout << " center =  " << obbPosition << ", halfextents = " << obbExtents << "; basis = " << basis[0] << " / " << basis[1] << " / " << basis[2] << ", centerOffset = " << this->wcData()._centerOffset << std::endl;
}

template <class LGCDataTypes>
void LGCPointClusterObb<LGCDataTypes>::drawCoordinateMarkerGL(float lineLength, float lineWidth, const Vec<4, Real>& xColor, const Vec<4, Real>& yColor, const Vec<4, Real>& zColor)
{
    glLineWidth(lineWidth);
    glBegin(GL_LINES);

    glColor4f(xColor.x(), xColor.y(), xColor.z(), xColor.w());
    glVertex3d(0, 0, 0);
    glVertex3d(lineLength,0,0);

    glColor4f(yColor.x(), yColor.y(), yColor.z(), yColor.w());
    glVertex3d(0, 0, 0);
    glVertex3d(0,lineLength,0);

    glColor4f(zColor.x(), zColor.y(), zColor.z(), zColor.w());
    glVertex3d(0, 0, 0);
    glVertex3d(0,0,lineLength);

    glEnd();
    glLineWidth(1.0f);
}

template <class LGCDataTypes>
void LGCPointClusterObb<LGCDataTypes>::drawObbVolume(const Vec<3, Real> &halfExtents, const Vec4f &color)
{
    glBegin(GL_LINES);
    glColor4d(color.x(), color.y(), color.z(), color.w());
    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());

    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glEnd();
}

template <class LGCDataTypes>
void LGCPointClusterObb<LGCDataTypes>::draw(const core::visual::VisualParams *vparams)
{
    LGCObb<LGCDataTypes>::draw(vparams);
#if 0
    Matrix4 obbOrientation; obbOrientation.identity();
    Matrix3 modelRotation;
    this->lgcCollisionModel()->orientation().toMatrix(modelRotation);
    Matrix4 modelOrientation; modelOrientation.identity();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            modelOrientation(i,j) = modelRotation(i,j);
            obbOrientation(i,j) = this->_wcData._localAxes[i][j];
        }
    }

    glPushMatrix();
    glTranslated(this->lgcCollisionModel()->position().x(), this->lgcCollisionModel()->position().y(), this->lgcCollisionModel()->position().z());
    glMultMatrixd(modelOrientation.transposed().ptr());
    glTranslated(this->wcData()._centerOffset.x(), this->wcData()._centerOffset.y(), this->wcData()._centerOffset.z());

    glMultMatrixd(obbOrientation.ptr());

    glLineWidth(4.0f);
    drawObbVolume(this->halfExtents(), Vec4f(1,0,1,1));
    glLineWidth(1.0f);

    glPopMatrix();

    this->obbDrawable().draw(vparams, Vec4f(1,0,1,1));
#endif
}

template <class LGCDataTypes>
LGCPointCluster<LGCDataTypes>::LGCPointCluster(const Vector3& position, const Quaternion& orientation, LGCPointCluster<LGCDataTypes>* parent, core::CollisionModel* model):
    LGCTransformable<typename LGCDataTypes::Real>(position, orientation),
    LGCIntersectable<typename LGCDataTypes::Real>(model),
    _showOcTree(false), _showOcTreeVerbose(false), _showKdTree(false), _computeSegmentation(false),
    _computeOctree(false), _computeKdtree(false), _parentCluster(parent), _drawVerbose(false), _showOBBPlanes(false),
    _facetRangeMin(-1), _facetRangeMax(-1), _originalModel(NULL), _clusterIndex(-1), _pqp_tree(NULL),
    _facetRangeAdjusted(false)
{
    d = new LGCPointClusterPrivate<LGCDataTypes>(this);
    _clusterColor = sofa::helper::randomVec4<float>();

    _lgcCollisionModel = dynamic_cast<LGCCollisionModel<LGCDataTypes>*>(model);
}

template <class LGCDataTypes>
LGCPointCluster<LGCDataTypes>::LGCPointCluster(const std::vector<pcl::LGCPointTypeMin>& clusterSurfacePoints, const std::vector<pcl::LGCPointTypeMin>& clusterEdgePoints, const std::vector<pcl::LGCPointTypeMin>& clusterVertices, const Vector3& position, const Quaternion& orientation, const ReferenceFrame& referenceFrame, LGCPointCluster<LGCDataTypes>* parent, const core::CollisionModel* model):
    LGCTransformable<typename LGCDataTypes::Real>(position, orientation, referenceFrame),
    LGCIntersectable<typename LGCDataTypes::Real>(model),
    _showOcTree(false), _showOcTreeVerbose(false), _showKdTree(false),
    _computeOctree(false), _computeKdtree(false), _parentCluster(parent), _drawVerbose(false), _showOBBPlanes(false),
    _facetRangeMin(-1), _facetRangeMax(-1), _originalModel(NULL), _clusterIndex(-1), _pqp_tree(NULL),
    _facetRangeAdjusted(false)
{
    d = new LGCPointClusterPrivate<LGCDataTypes>(this);
    if (clusterSurfacePoints.size() > 0)
    {
        unsigned long surfacePtIdx = 0;
        for (unsigned long i = 0; i < clusterSurfacePoints.size(); i++)
        {
            const pcl::LGCPointTypeMin& p = clusterSurfacePoints.at(i);
            addSurfacePoint(Vector3(p.x, p.y, p.z), surfacePtIdx, (int) p.i /*p.facetIdx*/);
        }
    }
#if 0
    if (clusterEdgePoints.size() > 0)
    {
        unsigned long edgePtIdx = 0;
        for (unsigned long i = 0; i < clusterEdgePoints.size(); i++)
        {
            const pcl::LGCPointTypeMin& p = clusterEdgePoints.at(i);
            if (addEdgePoint(Vector3(p.x, p.y, p.z), edgePtIdx, (unsigned long) p.edgeIdx, (unsigned long) p.facetIdx))
                edgePtIdx++;
        }
    }
#endif
    if (clusterVertices.size() > 0)
    {
        unsigned long vertexIdx = 0;
        for (unsigned long i = 0; i < clusterVertices.size(); i++)
        {
            const pcl::LGCPointTypeMin& p = clusterVertices.at(i);
            if (addVertex(Vector3(p.x, p.y, p.z), vertexIdx, p.i))
                vertexIdx++;
        }
    }


    _clusterColor = sofa::helper::randomVec4<float>();
    _lgcCollisionModel = dynamic_cast<LGCCollisionModel<LGCDataTypes>*>(const_cast<core::CollisionModel*>(model));
}

template <class LGCDataTypes>
LGCPointCluster<LGCDataTypes>::~LGCPointCluster()
{
    if (d)
    {
        std::cout << "LGCPointClusterPrivate deleted." << std::endl;
        delete d;
    }
    if (_pqp_tree)
    {
        delete _pqp_tree;
        _pqp_tree = NULL;
    }
}

template <class LGCDataTypes>
void LGCPointCluster<LGCDataTypes>::adjustFacetIndices()
{
    std::cout << "LGCPointCluster<LGCDataTypes>::adjustFacetIndices(" << this->getName() << ")" << std::endl;

    if (_facetRangeAdjusted)
    {
        std::cout << " facet range adjusted externally, aborting!" << std::endl;
        std::cout << " facet indices set: " << this->_childFacets.size() << std::endl << " Indices: ";
        for (int k = 0; k < _childFacets.size(); k++)
            std::cout << _childFacets[k] << " ";

        std::cout << std::endl;
        return;
    }

    long minFacetIdx = -1;
    long maxFacetIdx = -1;

    unsigned long pointIdx = 0;
    std::cout << "=== Vertices ===" << std::endl;
    for (std::deque<Vector3>::const_iterator it = _vertices.begin(); it != _vertices.end(); it++)
    {
        long facetIdx = -1, ptIdx = -1;
        if (_vertexFacetIndices.find(pointIdx) != _vertexFacetIndices.end())
            facetIdx = _vertexFacetIndices[pointIdx].front();

        if (_vertexIndices.find(pointIdx) != _vertexIndices.end())
            ptIdx = _vertexIndices[pointIdx];

        std::cout << " * Vertex " << pointIdx << ": Facet index " << facetIdx << ", point index: " << ptIdx << std::endl;

        std::cout << "  facet range tests: " << minFacetIdx << " < " << facetIdx << ": " << (facetIdx < minFacetIdx) << ", " << facetIdx << " > " << maxFacetIdx << ": " << (facetIdx > maxFacetIdx) << std::endl;
        if (minFacetIdx == -1 && facetIdx >= 0)
        {
            std::cout << "  setting initial min facet index: " << facetIdx << std::endl;
            minFacetIdx = facetIdx;
        }
        if (maxFacetIdx == -1 && facetIdx >= 0)
        {
            std::cout << "  setting initial max facet index: " << facetIdx << std::endl;
            maxFacetIdx = facetIdx;
        }

        if (facetIdx < minFacetIdx)
        {
            std::cout << "  setting new min facet index: " << facetIdx << std::endl;
            minFacetIdx = facetIdx;
        }
        if (facetIdx > maxFacetIdx)
        {
            std::cout << "  setting new max facet index: " << facetIdx << std::endl;
            maxFacetIdx = facetIdx;
        }

        pointIdx++;
    }

    pointIdx = 0;
    // std::cout << "=== Edge points ===" << std::endl;
    for (std::deque<Vector3>::const_iterator it = _edgePoints.begin(); it != _edgePoints.end(); it++)
    {
        long facetIdx = -1, ptIdx = -1, edgeIdx = -1;
        if (_edgePointFacetIndices.find(pointIdx) != _edgePointFacetIndices.end())
            facetIdx = _edgePointFacetIndices[pointIdx].front();

        if (_edgePointIndices.find(pointIdx) != _edgePointIndices.end())
            ptIdx = _edgePointIndices[pointIdx];

        if (_edgePointEdgeIndices.find(pointIdx) != _edgePointEdgeIndices.end())
            edgeIdx = _edgePointEdgeIndices[pointIdx];

        // std::cout << " * Edge Point " << pointIdx << ": Facet index " << facetIdx << ", edge index: " << edgeIdx << ", point index: " << ptIdx << std::endl;

        //std::cout << "  facet range tests: " << minFacetIdx << " < " << facetIdx << ": " << (facetIdx < minFacetIdx) << ", " << facetIdx << " > " << maxFacetIdx << ": " << (facetIdx > maxFacetIdx) << std::endl;
        if (minFacetIdx == -1 && facetIdx >= 0)
        {
            //std::cout << "  setting initial min facet index: " << facetIdx << std::endl;
            minFacetIdx = facetIdx;
        }
        if (maxFacetIdx == -1 && facetIdx >= 0)
        {
            //std::cout << "  setting initial max facet index: " << facetIdx << std::endl;
            maxFacetIdx = facetIdx;
        }

        if (facetIdx < minFacetIdx)
        {
            //std::cout << "  setting new min facet index: " << facetIdx << std::endl;
            minFacetIdx = facetIdx;
        }
        if (facetIdx > maxFacetIdx)
        {
            //std::cout << "  setting new max facet index: " << facetIdx << std::endl;
            maxFacetIdx = facetIdx;
        }

        pointIdx++;
    }

    pointIdx = 0;
    // std::cout << "=== Surface points ===" << std::endl;
    for (std::deque<Vector3>::const_iterator it = _clusterPoints.begin(); it != _clusterPoints.end(); it++)
    {
        unsigned long facetIdx = -1, ptIdx = -1;
        if (_surfacePointFacetIndices.find(pointIdx) != _surfacePointFacetIndices.end())
            facetIdx = _surfacePointFacetIndices[pointIdx];

        if (_surfacePointIndices.find(pointIdx) != _surfacePointIndices.end())
            ptIdx = _surfacePointIndices[pointIdx];

        // std::cout << " * Surface Point " << pointIdx << ": Facet index " << facetIdx << ", point index: " << ptIdx << std::endl;

        //std::cout << "  facet range tests: " << minFacetIdx << " < " << facetIdx << ": " << (facetIdx < minFacetIdx) << ", " << facetIdx << " > " << maxFacetIdx << ": " << (facetIdx > maxFacetIdx) << std::endl;
        if (minFacetIdx == -1 && facetIdx >= 0)
        {
            //std::cout << "  setting initial min facet index: " << facetIdx << std::endl;
            minFacetIdx = facetIdx;
        }
        if (maxFacetIdx == -1 && facetIdx >= 0)
        {
            //std::cout << "  setting initial max facet index: " << facetIdx << std::endl;
            maxFacetIdx = facetIdx;
        }

        if (facetIdx < minFacetIdx)
        {
            //std::cout << "  setting new min facet index: " << facetIdx << std::endl;
            minFacetIdx = facetIdx;
        }
        if (facetIdx > maxFacetIdx)
        {
            //std::cout << "  setting new max facet index: " << facetIdx << std::endl;
            maxFacetIdx = facetIdx;
        }

        pointIdx++;
    }

    if (minFacetIdx >= 0 && maxFacetIdx >= 0)
    {
        std::cout << "Setting facet range for point cluster '" << this->getName() << "': " << minFacetIdx << " -> " << maxFacetIdx << std::endl;
        this->setFacetRange(minFacetIdx, maxFacetIdx);
        if (this->clusterObb())
            this->clusterObb()->setFacetRange(minFacetIdx, maxFacetIdx);
    }
    else
    {
        std::cerr << "WARNING: Failed to determine facet range for cluster " << this->getName() << "; range min = " << minFacetIdx << ", range max = " << maxFacetIdx << std::endl;
    }

    if (this->numChildren() > 0)
    {
        for (unsigned long k = 0; k < this->numChildren(); k++)
        {
            this->childCluster(k)->adjustFacetIndices();
        }
    }
}

template <class LGCDataTypes>
bool LGCPointCluster<LGCDataTypes>::writePCDFile(const std::string& fileName)
{
    std::cout << "LGCPointCluster<LGCDataTypes>::writePCDFile(" << fileName << ")" << std::endl;
    std::cout << " clusterPoints count -- Surface: " << _clusterPoints.size() << ", Edge: " << _edgePoints.size() << ", Vertices: " << _vertices.size() << std::endl;
    if (_clusterPoints.size() > 0 || _vertices.size() > 0 || _edgePoints.size() > 0)
    {
        std::string dataRepPath = sofa::helper::system::DataRepository.getPaths().at(0);
        std::cout << "Base path for PCL file saving: " << dataRepPath << std::endl;
        if (!dataRepPath.empty())
        {
            dataRepPath.append("/");
            if (fileName.empty())
            {
                dataRepPath.append(this->getName() + ".pcd");
            }
            else
            {
                dataRepPath.append(fileName + ".pcd");
                d->_pcdBaseName = fileName;
            }

            std::cout << "Full name for file to save to: " << dataRepPath << std::endl;

            std::ofstream pclStream;
            pclStream.open(dataRepPath.c_str());
            if (pclStream.is_open())
            {
                pclStream << "# .PCD v.7 - Point Cloud Data file format" << std::endl;
                pclStream << "VERSION .7" << std::endl;
                //pclStream << "FIELDS x y z pointIdx facetIdx edgeIdx pointType" << std::endl;
                pclStream << "FIELDS x y z i" << std::endl;
                //pclStream << "SIZE 4 4 4 4 4 4 4" << std::endl;
                pclStream << "SIZE 4 4 4 4" << std::endl;
                //pclStream << "TYPE F F F U U U U" << std::endl;
                //pclStream << "COUNT 1 1 1 1 1 1 1" << std::endl;
                pclStream << "TYPE F F F F" << std::endl;
                pclStream << "COUNT 1 1 1 1" << std::endl;
                pclStream << "WIDTH " << _clusterPoints.size() + _edgePoints.size() + _vertices.size() << std::endl;
                pclStream << "HEIGHT 1" << std::endl;
                pclStream << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
                pclStream << "POINTS " << _clusterPoints.size() + _edgePoints.size() + _vertices.size() << std::endl;
                pclStream << "DATA ascii" << std::endl;

                long minFacetIdx = -1;
                long maxFacetIdx = -1;

                unsigned long pointIdx = 0;
                // std::cout << "=== Vertices ===" << std::endl;
                for (std::deque<Vector3>::const_iterator it = _vertices.begin(); it != _vertices.end(); it++)
                {
                    long facetIdx = -1, ptIdx = -1;
                    if (_vertexFacetIndices.find(pointIdx) != _vertexFacetIndices.end())
                        facetIdx = _vertexFacetIndices[pointIdx].front();

                    if (_vertexIndices.find(pointIdx) != _vertexIndices.end())
                        ptIdx = _vertexIndices[pointIdx];

                    // std::cout << " * Vertex " << pointIdx << ": Facet index " << facetIdx << ", point index: " << ptIdx << std::endl;

                    // std::cout << "  facet range tests: " << minFacetIdx << " < " << facetIdx << ": " << (facetIdx < minFacetIdx) << ", " << facetIdx << " > " << maxFacetIdx << ": " << (facetIdx > maxFacetIdx) << std::endl;
                    if (minFacetIdx == -1 && facetIdx >= 0)
                    {
                        // std::cout << "  setting initial min facet index: " << facetIdx << std::endl;
                        minFacetIdx = facetIdx;
                    }
                    if (maxFacetIdx == -1 && facetIdx >= 0)
                    {
                        // std::cout << "  setting initial max facet index: " << facetIdx << std::endl;
                        maxFacetIdx = facetIdx;
                    }

                    if (facetIdx < minFacetIdx)
                    {
                        // std::cout << "  setting new min facet index: " << facetIdx << std::endl;
                        minFacetIdx = facetIdx;
                    }
                    if (facetIdx > maxFacetIdx)
                    {
                        // std::cout << "  setting new max facet index: " << facetIdx << std::endl;
                        maxFacetIdx = facetIdx;
                    }

                    Vector3 p = (*it);
                    pclStream << p.x() << " " << p.y() << " " << p.z() << " " << facetIdx << std::endl; //<< " " << facetIdx << " 0 2" << std::endl;
                    pointIdx++;
                }

                pointIdx = 0;
                // std::cout << "=== Edge points ===" << std::endl;
                for (std::deque<Vector3>::const_iterator it = _edgePoints.begin(); it != _edgePoints.end(); it++)
                {
                    long facetIdx = -1, ptIdx = -1, edgeIdx = -1;
                    if (_edgePointFacetIndices.find(pointIdx) != _edgePointFacetIndices.end())
                        facetIdx = _edgePointFacetIndices[pointIdx].front();

                    if (_edgePointIndices.find(pointIdx) != _edgePointIndices.end())
                        ptIdx = _edgePointIndices[pointIdx];

                    if (_edgePointEdgeIndices.find(pointIdx) != _edgePointEdgeIndices.end())
                        edgeIdx = _edgePointEdgeIndices[pointIdx];

                    // std::cout << " * Edge Point " << pointIdx << ": Facet index " << facetIdx << ", edge index: " << edgeIdx << ", point index: " << ptIdx << std::endl;

                    // std::cout << "  facet range tests: " << minFacetIdx << " < " << facetIdx << ": " << (facetIdx < minFacetIdx) << ", " << facetIdx << " > " << maxFacetIdx << ": " << (facetIdx > maxFacetIdx) << std::endl;
                    if (minFacetIdx == -1 && facetIdx >= 0)
                    {
                        // std::cout << "  setting initial min facet index: " << facetIdx << std::endl;
                        minFacetIdx = facetIdx;
                    }
                    if (maxFacetIdx == -1 && facetIdx >= 0)
                    {
                        // std::cout << "  setting initial max facet index: " << facetIdx << std::endl;
                        maxFacetIdx = facetIdx;
                    }

                    if (facetIdx < minFacetIdx)
                    {
                        // std::cout << "  setting new min facet index: " << facetIdx << std::endl;
                        minFacetIdx = facetIdx;
                    }
                    if (facetIdx > maxFacetIdx)
                    {
                        // std::cout << "  setting new max facet index: " << facetIdx << std::endl;
                        maxFacetIdx = facetIdx;
                    }

                    Vector3 p = (*it);
                    pclStream << p.x() << " " << p.y() << " " << p.z() << " " << facetIdx << std::endl; //<< ptIdx << " " << facetIdx << " " << edgeIdx << " 1" << std::endl;
                    pointIdx++;
                }

                pointIdx = 0;
                // std::cout << "=== Surface points ===" << std::endl;
                for (std::deque<Vector3>::const_iterator it = _clusterPoints.begin(); it != _clusterPoints.end(); it++)
                {
                    unsigned long facetIdx = -1, ptIdx = -1;
                    if (_surfacePointFacetIndices.find(pointIdx) != _surfacePointFacetIndices.end())
                        facetIdx = _surfacePointFacetIndices[pointIdx];

                    if (_surfacePointIndices.find(pointIdx) != _surfacePointIndices.end())
                        ptIdx = _surfacePointIndices[pointIdx];

                    // std::cout << " * Surface Point " << pointIdx << ": Facet index " << facetIdx << ", point index: " << ptIdx << std::endl;

                    // std::cout << "  facet range tests: " << minFacetIdx << " < " << facetIdx << ": " << (facetIdx < minFacetIdx) << ", " << facetIdx << " > " << maxFacetIdx << ": " << (facetIdx > maxFacetIdx) << std::endl;
                    if (minFacetIdx == -1 && facetIdx >= 0)
                    {
                        // std::cout << "  setting initial min facet index: " << facetIdx << std::endl;
                        minFacetIdx = facetIdx;
                    }
                    if (maxFacetIdx == -1 && facetIdx >= 0)
                    {
                        // std::cout << "  setting initial max facet index: " << facetIdx << std::endl;
                        maxFacetIdx = facetIdx;
                    }

                    if (facetIdx < minFacetIdx)
                    {
                        // std::cout << "  setting new min facet index: " << facetIdx << std::endl;
                        minFacetIdx = facetIdx;
                    }
                    if (facetIdx > maxFacetIdx)
                    {
                        // std::cout << "  setting new max facet index: " << facetIdx << std::endl;
                        maxFacetIdx = facetIdx;
                    }

                    Vector3 p = (*it);
                    pclStream << p.x() << " " << p.y() << " " << p.z() << " " << facetIdx << std::endl; //<< ptIdx << " " << facetIdx << " 0 0" << std::endl;
                    pointIdx++;
                }

                if (minFacetIdx >= 0 && maxFacetIdx >= 0)
                {
                    // std::cout << "Setting facet range for point cluster '" << this->getName() << "': " << minFacetIdx << " -> " << maxFacetIdx << std::endl;
                    this->setFacetRange(minFacetIdx, maxFacetIdx);
                }
                else
                {
                    std::cerr << "WARNING: Failed to determine facet range for cluster " << this->getName() << "; range min = " << minFacetIdx << ", range max = " << maxFacetIdx << std::endl;
                }

                pclStream.close();
                d->_pcdFile = dataRepPath;
                std::cout << " Finished writing PCD file." << std::endl;
                return true;
            }
        }
    }
    std::cerr << " Failed writing PCD file to " << fileName << "!" << std::endl;
    return false;
}

template <class LGCDataTypes>
bool LGCPointCluster<LGCDataTypes>::readFromPCD()
{
    std::cout << "LGCPointCluster<LGCDataTypes>::readFromPCD(" << d->_pcdFile << ")" << std::endl;
    pcl::PCDReader reader;
    d->_cloud = pcl::PointCloud<pcl::LGCPointTypeMin>::Ptr(new pcl::PointCloud<pcl::LGCPointTypeMin>);
    if (reader.read(d->_pcdFile, *(d->_cloud)) == 0)
    {
        std::cout << "PCL PointCloud for " << d->_pcdFile << " element count: "  << d->_cloud->points.size() << std::endl;

        typedef sofa::component::collision::EPEC_Kernel Kernel;
        std::list<CGAL::Point_3<Kernel> > points;

        for (std::deque<Vector3>::const_iterator it = _vertices.begin(); it != _vertices.end(); it++)
        {
            points.push_back(CGAL::Point_3<Kernel>((*it).x(),(*it).y(),(*it).z()));
        }

        for (std::deque<Vector3>::const_iterator it = _edgePoints.begin(); it != _edgePoints.end(); it++)
        {
            points.push_back(CGAL::Point_3<Kernel>((*it).x(),(*it).y(),(*it).z()));
        }

#if 0
        for (std::deque<Vector3>::const_iterator it = _clusterPoints.begin(); it != _clusterPoints.end(); it++)
        {
            points.push_back(CGAL::Point_3<Kernel>((*it).x(),(*it).y(),(*it).z()));
        }
#endif

        CGAL::Point_3<Kernel> centroid = CGAL::centroid(points.begin(), points.end(), CGAL::Dimension_tag<0>());
        d->_centroid = Vector3(CGAL::to_double(centroid.x()),
                               CGAL::to_double(centroid.y()),
                               CGAL::to_double(centroid.z()));

        //d->downSamplePCL();
        d->_cloud_filtered = d->_cloud;

        /// TODO: Partitionierungsstrategien - Alternativen?
        if (this->getDoSegmentation())
            d->planarSegmentation();

        std::cout << "Cluster centroid: " << d->_centroid << std::endl;
        /*Eigen::Vector3i gridDimensions = d->_voxelGrid.getNrDivisions();
        std::cout << "Voxel grid dimensions: " << gridDimensions.x() << "x" << gridDimensions.y() << "x" << gridDimensions.z() << std::endl;*/

        /*pcl::search::Octree<pcl::LGCPointType>* ocTree = d->_ocTree;
          std::cout << "Tree depth of ocTree: " << ocTree->tree_->getTreeDepth() << std::endl;*/

        std::cout << "Point cloud re-alignment to origin coordinates: " << d->_cloud->points.size() << " input points." << std::endl;

        Matrix3 rotationMatrix;
        _lgcCollisionModel->orientation().inverse().toMatrix(rotationMatrix);

        Matrix4 orientationMatrix; orientationMatrix.identity();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                orientationMatrix(i,j) = rotationMatrix(i,j);

        for (int k = 0; k < d->_cloud->points.size(); k++)
        {
            pcl::LGCPointTypeMin pt = d->_cloud->points[k];
            pcl::LGCPointTypeMin origPt = pt;
            Vector3 tp(pt.x,pt.y,pt.z);
            tp -= _lgcCollisionModel->position();
            tp = orientationMatrix.transform(tp);

            pt.x = tp.x(); pt.y = tp.y(); pt.z = tp.z();
            d->_cloud->points[k] = pt;
            std::cout << "  * " << k << ": " << origPt << " -> " << pt << std::endl;
        }

        d->computeTrees();

        if (_computeOctree)
        {
            std::cout << "=== Octree structure dump BEGIN ===" << std::endl;
            double minX, minY, minZ, maxX, maxY, maxZ;
            d->_spOcTree->getBoundingBox(minX,minY,minZ,maxX,maxY,maxZ);
            std::cout << " octree bbox : " << minX << "," << minY << "," << minZ << " -- " << maxX << "," << maxY << "," << maxZ << std::endl;
            std::cout << " octree depth: " << d->_spOcTree->getTreeDepth() << std::endl;
            d->dumpOcTree();
            std::cout << "=== Octree structure dump END   ===" << std::endl;
        }

        if (_computeKdtree)
        {
            std::cout << "=== k-d-tree structure dump BEGIN ===" << std::endl;
            if (d->_kdTree)
            {
                std::cout << " k-d-tree valid: " << d->_kdTree->kdTreeValid() << std::endl;
                if (d->_kdTree->kdTreeValid())
                {
                    std::cout << " epsilon       : " << d->_kdTree->getEpsilon() << std::endl;
                    d->dumpKdTree();
                }
            }
            std::cout << "=== k-d-tree structure dump END   ===" << std::endl;
        }
        return true;
    }
    else
    {
        std::cerr << "Failed to read PCD file " << d->_pcdFile << " for LGCPointCluster " << getName() << std::endl;
    }
    return false;
}

template <class LGCDataTypes>
void LGCPointCluster<LGCDataTypes>::buildChildHierarchy()
{
    std::cout << getName() << ": Building child hierarchy." << std::endl;
    d->buildChildHierarchy();
}

template <class LGCDataTypes>
void LGCPointCluster<LGCDataTypes>::fitObbs()
{
    std::cout << getName() << ": Fitting OBB's of child clusters: " << _children.size() << " children." << std::endl;

    d->_pcObb = new LGCPointClusterObb<LGCDataTypes>(this, this->position(), this->orientation(), 1.0f, this->collisionModel());
    d->_clusterObbDrawable = new ObbDrawable<LGCDataTypes>(d->_pcObb, NULL, ObbDrawable<LGCDataTypes>::ROTATE_THEN_TRANSLATE);

    d->_pcObb->setObbDrawable(*(d->_clusterObbDrawable));

    if (_children.size() > 0)
    {
        std::cout << "=== Fitting OBB's of child clusters: " << _children.size() << " ===" << std::endl;
        for (unsigned long i = 0; i < _children.size(); i++)
        {
            _children[i]->d->computeClusterCentroid();
            _children[i]->d->fitObbToCluster();

            _children[i]->d->_clusterObb->fixDegenerateObb();
            _children[i]->d->_clusterObb->setObbDrawable(*(_children[i]->d->_clusterObbDrawable));
            _children[i]->d->_clusterObbDrawable->updateFromObb();

            std::cout << " * CHILD " << i << " OBB: " << *(_children[i]->d->_clusterObb);

            d->_pcObb->addChild(_children[i]->d->_clusterObb);
            d->_clusterObbDrawable->addChild(_children[i]->d->_clusterObbDrawable);
        }
    }

    if (d->_pcObb)
    {
        std::cout << " refit cluster level OBB " << d->_pcObb->identifier() << std::endl;
        d->_pcObb->fitToCluster();
        d->_pcObb->_allChildren.push_front(d->_pcObb);

        /*d->_pcObb->refitObb();
        d->updateOBBHierarchy();*/
    }

    /*if (d->_clusterObbDrawable)
        d->_clusterObbDrawable->updateFromObb(true);*/

}

//#define LGCPOINTCLUSTER_USE_OBSOLETE_TRANSFORMS
template <class LGCDataTypes>
void LGCPointCluster<LGCDataTypes>::transform(const Vector3 &position, const Quaternion &orientation)
{
#ifndef LGCPOINTCLUSTER_USE_OBSOLETE_TRANSFORMS
    std::cout << "LGCPointCluster<LGCDataTypes>::transform(" << position << "," << orientation << ") -- " << this->getName() << std::endl;
    LGCTransformable<Real>::transform(position, orientation);

#else
    std::cout << "LGCPointCluster<LGCDataTypes>::transform(" << position << "," << orientation << ") -- " << this->getName() << std::endl;
    LGCTransformable<Real>::transform(position, orientation);
    Quaternion rotationOp1 = this->_orientation.inverse();
    //if (this->_parentCluster == NULL)
    {
        for (unsigned long i = 0; i < _clusterPoints.size(); i++)
        {
            _clusterPoints[i] -= this->position();
            _clusterPoints[i] = rotationOp1.rotate(_clusterPoints[i]);
            _clusterPoints[i] = orientation.rotate(_clusterPoints[i]);
            _clusterPoints[i] += position;
        }

        for (unsigned long i = 0; i < _edgePoints.size(); i++)
        {
            _edgePoints[i] -= this->position();
            _edgePoints[i] = rotationOp1.rotate(_edgePoints[i]);
            _edgePoints[i] = orientation.rotate(_edgePoints[i]);
            _edgePoints[i] += position;
        }

        for (unsigned long i = 0; i < _vertices.size(); i++)
        {
            _vertices[i] -= this->position();
            _vertices[i] = rotationOp1.rotate(_vertices[i]);
            _vertices[i] = orientation.rotate(_vertices[i]);
            _vertices[i] += position;
        }

        d->_centroid -= this->position();
        d->_centroid = rotationOp1.rotate(d->_centroid);
        d->_centroid = orientation.rotate(d->_centroid);
        d->_centroid += position;
    }

    for (unsigned long i = 0; i < d->_clusterEdgePoints.size(); i++)
    {
        d->_clusterEdgePoints[i] -= this->position();
        d->_clusterEdgePoints[i] = rotationOp1.rotate(d->_clusterEdgePoints[i]);
        d->_clusterEdgePoints[i] = orientation.rotate(d->_clusterEdgePoints[i]);
        d->_clusterEdgePoints[i] += position;
    }

    for (unsigned long i = 0; i < d->_clusterVertices.size(); i++)
    {
        d->_clusterVertices[i] -= this->position();
        d->_clusterVertices[i] = rotationOp1.rotate(d->_clusterVertices[i]);
        d->_clusterVertices[i] = orientation.rotate(d->_clusterVertices[i]);
        d->_clusterVertices[i] += position;
    }

    // std::cout << getName() << ": transform _obbPlanes" << std::endl;
    //if (this->_parentCluster == NULL)
    {
        for (unsigned int i = 0; i < 6; i++)
        {
            // std::cout << " .. " << i << ": drawable: " << (d->_obbPlaneDrawables[i] != NULL) << ", plane: " << (d->_obbPlanes[i] != NULL) << std::endl;
            if (d->_obbPlaneDrawables[i] != NULL && d->_obbPlanes[i] != NULL)
            {
                d->_obbPlanes[i]->transform(position, orientation);
                d->_obbPlaneDrawables[i]->transform(position, orientation);
            }
        }

        if (d->_clusterObbDrawable != NULL)
            d->_clusterObbDrawable->transform(position, orientation);
    }

    d->_clusterObb->transform(position, orientation);

    for (unsigned long  i = 0; i < _children.size(); i++)
    {
        _children[i]->transform(position, orientation);
    }
    LGCTransformable<Real>::transform(position, orientation);
#endif
}

//#define LGC_POINTCLUSTER_DEBUG_POINTS
//#define LGC_POINTCLUSTER_DEBUG_POINTS_FROM_TOPOLOGY
template <class LGCDataTypes>
void LGCPointCluster<LGCDataTypes>::draw(const core::visual::VisualParams *vparams)
{
    if (vparams->displayFlags().getShowBoundingCollisionModels())
    {   
        /*if (_drawObbVolumes && d->_clusterObbDrawable)
        {
            d->_clusterObbDrawable->setDrawLimits(_minDrawLimit, _maxDrawLimit);
            d->_clusterObbDrawable->setDrawStructure(_drawObbTreeStructure);
            d->_clusterObbDrawable->setDrawVolumes(_drawObbVolumes);
        }*/

        // Draw trees if in parent cluster
        if (_parentCluster == NULL)
        {
            if (_drawObbVolumes)
            {
                if (d->_pcObb)
                {
                    d->_pcObb->draw(vparams);
                }
            }

            if (_showOcTree)
                d->drawOcTree(vparams, _showOcTreeVerbose);

            if (_showKdTree)
                d->drawKdTree(vparams);

            for (unsigned int k = 0; k < _children.size(); k++)
                _children.at(k)->draw(vparams /*, _children.at(k)->getClusterColor()*/);

            if (clusterObb())
                clusterObb()->intersectionHistory().draw(vparams);
        }
        else
        {
            // LGCIntersectable<LGCDataTypes>::draw(vparams);
            if (clusterObb())
            {
                /*if (this->_indices.size() > 0)
                    clusterObb()->draw(vparams, getClusterColor());*/

                clusterObb()->intersectionHistory().draw(vparams);
            }

            //if (_showOcTree)
            d->drawOcTree(vparams, _showOcTreeVerbose);

#ifdef LGC_POINTCLUSTER_DEBUG_POINTS_FROM_TOPOLOGY
#ifdef LGC_POINTCLUSTER_DEBUG_ALL_CLUSTER_POINTS
            if (_allClusterPtsDraw.size() > 0)
            {
                vparams->drawTool()->drawSpheres(_allClusterPtsDraw, 0.03f, Vec4f(1,1,0,0.8f));
            }
#endif

            const LGCCollisionModel<LGCDataTypes>* lgcModel = dynamic_cast<const LGCCollisionModel<LGCDataTypes>*>(_parentCluster->collisionModel());
            if (lgcModel)
            {
                //std::cout << " draw pts " << this->getName() << std::endl;
                sofa::component::topology::MeshTopology* meshTopology = dynamic_cast<sofa::component::topology::MeshTopology*>(lgcModel->topology());
                if (meshTopology)
                {
                    //std::cout << " meshTopology obtained" << std::endl;

                    sofa::component::topology::MeshTopology::SeqTriangles triangles = meshTopology->getTriangles();
                    ReadAccessor<Data<LGCDataTypes::VecCoord> > meshPoints(lgcModel->mechanicalState()->read(core::ConstVecCoordId::position()));

                    glPointSize(5.0f);
                    glBegin(GL_POINTS);
                    for (unsigned int k = this->_facetRangeMin; k <= this->_facetRangeMax; k++)
                    {
                        sofa::component::topology::MeshTopology::Triangle tri = triangles[k];
                        Vector3 corner0 = meshPoints[tri[0]];
                        Vector3 corner1 = meshPoints[tri[1]];
                        Vector3 corner2 = meshPoints[tri[2]];

                        glColor4d(this->_clusterColor.x(),this->_clusterColor.y(),this->_clusterColor.z(), 0.8);
                        glVertex3d(corner0.x(), corner0.y(), corner0.z());
                        glColor4d(this->_clusterColor.x(),this->_clusterColor.y(),this->_clusterColor.z(), 0.8);
                        glVertex3d(corner1.x(), corner1.y(), corner1.z());
                        glColor4d(this->_clusterColor.x(),this->_clusterColor.y(),this->_clusterColor.z(), 0.8);
                        glVertex3d(corner2.x(), corner2.y(), corner2.z());

                        Vector3 vtx, nvtx;
                        for (unsigned int o = 0; o < 3; o++)
                        {
                            if (o == 0)
                            {
                                vtx = corner0;
                                nvtx = corner1;
                            }
                            else if (o == 1)
                            {
                                vtx = corner1;
                                nvtx = corner2;
                            }
                            else if (o == 2)
                            {
                                vtx = corner2;
                                nvtx = corner0;
                            }
                            for (unsigned int p = 0; p < 5; p++)
                            {
                                Vector3 ep = vtx + ((nvtx - vtx) * (1.0f * p / 5));
                                glColor4d(this->_clusterColor.x(),this->_clusterColor.y(),this->_clusterColor.z(), 0.8);
                                glVertex3d(ep.x(), ep.y(), ep.z());
                            }
                        }
                    }
                    glEnd();
                    glPointSize(1.0f);
                }
            }
#endif

#ifdef LGC_POINTCLUSTER_DEBUG_POINTS
            /// TODO: Attention: Not transformed with model!
            glPointSize(10.0f);
            glBegin(GL_POINTS);
            for (unsigned int k = 0; k < _vertices.size(); k++)
            {
                glColor4d(this->_clusterColor.x(),this->_clusterColor.y(),this->_clusterColor.z(), 0.5);
                glVertex3d(_vertices.at(k).x(), _vertices.at(k).y(), _vertices.at(k).z());
            }
            for (unsigned int k = 0; k < _edgePoints.size(); k++)
            {
                glColor4d(this->_clusterColor.x(),this->_clusterColor.y(),this->_clusterColor.z(), 0.5);
                glVertex3d(_edgePoints.at(k).x(), _edgePoints.at(k).y(), _edgePoints.at(k).z());
            }
            for (unsigned int k = 0; k < _clusterPoints.size(); k++)
            {
                glColor4d(this->_clusterColor.x(),this->_clusterColor.y(),this->_clusterColor.z(), 0.5);
                glVertex3d(_clusterPoints.at(k).x(), _clusterPoints.at(k).y(), _clusterPoints.at(k).z());
            }
            glEnd();
            glPointSize(1.0f);
#endif
        }

    }
}

template <class LGCDataTypes>
pcl::gpu::Octree* LGCPointCluster<LGCDataTypes>::clusterOctreeGPU()
{
    return d->_octree_device;
}

template <class LGCDataTypes>
sofa::component::collision::OcTreeSinglePoint *LGCPointCluster<LGCDataTypes>::clusterOctreeCPU()
{
    return d->_spOcTree;
}

template <class LGCDataTypes>
OctreeSearch* LGCPointCluster<LGCDataTypes>::clusterOctreeSearch()
{
    return d->_searchOcTree;
}

template <class LGCDataTypes>
void LGCPointCluster<LGCDataTypes>::setComputeOcTree(bool compute)
{
    _computeOctree = compute;
    if (this->parentCluster() != NULL && _computeOctree)
    {
        if (d->_spOcTree)
            delete d->_spOcTree;

        d->_spOcTree = new OcTreeSinglePoint(0.05f);
        std::cout << "=== GENERATE single point oc-tree: " << this->getName() << " ===" << std::endl;
        d->_spOcTree->setInputCloud(d->_cloud);
        d->_spOcTree->addPointsFromInputCloud();

        if (d->_searchOcTree)
            delete d->_searchOcTree;

        d->_searchOcTree = new OctreeSearch(0.05f);
        std::cout << "=== GENERATE search oc-tree: " << this->getName() << " ===" << std::endl;
        d->_searchOcTree->setInputCloud(d->_cloud);
        d->_searchOcTree->addPointsFromInputCloud();

        std::cout << " points in octree: " << d->_cloud->points.size() << std::endl;

        if (d->_cloud_device)
        {
            // d->_cloud_device->release();
            delete d->_cloud_device;
        }

        std::cout << "=== GENERATE GPU oc-tree: " << this->getName() << " ===" << std::endl;
        d->_cloud_device = new pcl::gpu::Octree::PointCloud();
        d->_cloud_device->upload(d->_cloud->points);

        if (d->_octree_device)
        {
            // d->_octree_device->clear();
            delete d->_octree_device;
        }

        d->_octree_device = new pcl::gpu::Octree();
        d->_octree_device->setCloud(*(d->_cloud_device));

        sofa::helper::AdvancedTimer::stepBegin("LGC: BuildGPUOctree", this->getName());
        d->_octree_device->build();
        sofa::helper::AdvancedTimer::stepEnd("LGC: BuildGPUOctree", this->getName());

        std::cout << " built: " << d->_octree_device->isBuilt() << std::endl;
        if (d->_octree_device->isBuilt())
        {
            pcl::device::OctreeImpl* octreeImpl = (pcl::device::OctreeImpl*) d->_octree_device->getImpl();
            if (!octreeImpl->host_octree.downloaded)
                octreeImpl->internalDownload();

            int number = 0;
            pcl::gpu::DeviceArray<int>(octreeImpl->octreeGlobal.nodes_num, 1).download(&number);
            std::cout << " number of nodes        : " << number << std::endl;
            std::cout << " points in orginal cloud: " << octreeImpl->host_octree.points_sorted.size() / 3 << std::endl;
            std::cout << " morton key count       : " << octreeImpl->host_octree.codes.size() << std::endl;
            std::cout << " begin array size       : " << octreeImpl->host_octree.begs.size() << std::endl;
            std::cout << " end array size         : " << octreeImpl->host_octree.ends.size() << std::endl;

            int nodeNum = 0;
            for (std::vector<int>::const_iterator nit = octreeImpl->host_octree.nodes.begin();
                                                  nit != octreeImpl->host_octree.nodes.end(); nit++)
            {
                std::cout << " * Node " << nodeNum << ": " << *nit << ", bits: ";
                int sizeOfInt = sizeof(int) * 8;
                for (int k = sizeOfInt; k > 0; k--)
                {
                    int bitMask = 1 << k;
                    if ((*nit) & bitMask)
                        std::cout << "1";
                    else
                        std::cout << "0";
                }
                std::cout << std::endl;
                std::cout << "   Morton key: ";
                int mortonKey = octreeImpl->host_octree.codes.at(nodeNum);
                for (int k = sizeOfInt; k > 0; k--)
                {
                    int bitMask = 1 << k;
                    if (mortonKey & bitMask)
                        std::cout << "1";
                    else
                        std::cout << "0";
                }
                std::cout << std::endl;

                std::cout << " points contained: " << octreeImpl->host_octree.begs.at(nodeNum) << " - " << octreeImpl->host_octree.ends.at(nodeNum) << std::endl;

                nodeNum++;
            }
        }
    }
}

template <class LGCDataTypes>
void LGCPointCluster<LGCDataTypes>::setComputeKdTree(bool compute)
{
    _computeKdtree = compute;
    if (_computeKdtree && this->parentCluster() != NULL)
    {
        if (d->_kdTree)
            delete d->_kdTree;

        d->_kdTree = new LGCKdTree<pcl::LGCPointTypeMin>();
        std::cout << "=== GENERATE kd-tree ===" << std::endl;
        d->_kdTree->setInputCloud((LGCKdTree<pcl::LGCPointTypeMin>::PointCloudConstPtr)d->_cloud, LGCKdTree<pcl::LGCPointTypeMin>::IndicesConstPtr());
    }
}

#define LGC_POINT_CLUSTER_ADD_SUBCLUSTER_EDGE_POINTS
#define LGC_POINT_CLUSTER_ADD_SUBCLUSTER_SURFACE_POINTS
template <class LGCDataTypes>
void LGCPointCluster<LGCDataTypes>::buildSubCluster()
{
    std::cout << "=== LGCPointCluster::buildSubCluster(" << this->getName() << ") ===" << std::endl;

    if (this->_indices.size() > 0)
    {
        Matrix3 modelRotation;
        this->_lgcCollisionModel->orientation().toMatrix(modelRotation);

        Vector3 modelPosition = _lgcCollisionModel->position();
        Matrix4 modelOrientation; modelOrientation.identity();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                modelOrientation(i,j) = modelRotation(i,j);

        sofa::component::topology::MeshTopology* meshTopology = dynamic_cast<sofa::component::topology::MeshTopology*>(_lgcCollisionModel->topology());
        ReadAccessor<Data<LGCDataTypes::VecCoord> > meshPoints(_lgcCollisionModel->mechanicalState()->read(core::ConstVecCoordId::position()));

        unsigned int numTiles = (LGC_POINT_MODEL_GRID_SIZE * LGC_POINT_MODEL_GRID_SIZE) / (LGC_POINT_MODEL_GRID_TILE_SIZE * LGC_POINT_MODEL_GRID_TILE_SIZE);
        unsigned int tilesPerSide = std::sqrt(numTiles);
        unsigned int pmSize = tilesPerSide * LGC_POINT_MODEL_GRID_TILE_SIZE;

        unsigned long facetIndex = 0;
        unsigned long edgePtIdx = 0;
        unsigned long vertexIdx = 0;

        for (int i = 0; i < _indices.size(); i++)
        {
            Vector3 corner0 = meshPoints[_indices[i].second[0]];
            Vector3 corner1 = meshPoints[_indices[i].second[1]];
            Vector3 corner2 = meshPoints[_indices[i].second[2]];

            Vec2i gridSize = _lgcCollisionModel->pointGridDimension(_indices[i].first);

            std::cout << "  Facet " << facetIndex << ": corners before transform: " << corner0 << "," << corner1 << "," << corner2 << std::endl;

            corner0 = modelOrientation.transform(corner0);
            corner0 += modelPosition;
            corner1 = modelOrientation.transform(corner1);
            corner1 += modelPosition;
            corner2 = modelOrientation.transform(corner2);
            corner2 += modelPosition;

            std::cout << "  Facet " << facetIndex << ": corners after transform: " << corner0 << "," << corner1 << "," << corner2 << std::endl;

            if (addVertex(corner0, vertexIdx, _indices[i].first))
            {
                std::cout << "  * added corner0: " << corner0 << ", vertex index = " << vertexIdx << ", facet index = " << _indices[i].first << std::endl;
                vertexIdx++;
            }

            if (addVertex(corner1, vertexIdx, _indices[i].first))
            {
                std::cout << "  * added corner1: " << corner1 << ", vertex index = " << vertexIdx << ", facet index = " << _indices[i].first << std::endl;
                vertexIdx++;
            }

            if (addVertex(corner2, vertexIdx, _indices[i].first))
            {
                std::cout << "  * added corner0: " << corner2 << ", vertex index = " << vertexIdx << ", facet index = " << _indices[i].first << std::endl;
                vertexIdx++;
            }
#ifdef LGC_POINT_CLUSTER_ADD_SUBCLUSTER_EDGE_POINTS
            unsigned int edgeCount = 0;
            for (int k = 0; k <= 2; k++)
            {
                int edgeIdx = -1;
                Vector3 vtx, nvtx;
                if (edgeCount == 0)
                {
                    vtx = corner0;
                    nvtx = corner1;
                    edgeIdx = meshTopology->getEdgeIndex(_indices[i].second[0], _indices[i].second[1]);
                }
                else if (edgeCount == 1)
                {
                    vtx = corner1;
                    nvtx = corner2;
                    edgeIdx = meshTopology->getEdgeIndex(_indices[i].second[1], _indices[i].second[2]);
                }
                else if (edgeCount == 2)
                {
                    vtx = corner2;
                    nvtx = corner0;
                    edgeIdx = meshTopology->getEdgeIndex(_indices[i].second[2], _indices[i].second[0]);
                }

                for (unsigned int p = 0; p < 5; p++)
                {
                    Vector3 ep = vtx + ((nvtx - vtx) * (1.0f * p / 5));
                    if (addEdgePoint(ep, edgePtIdx, edgeIdx, (unsigned long) _indices[i].first))
                    {
                        std::cout << "  * added edge point: " << ep << ", edge point index = " << edgePtIdx << ", edge index = " << edgeIdx <<  ", facet index = " << _indices[i].first << std::endl;
                        setEdgePointFacetIndex(edgePtIdx, _indices[i].first);
                        edgePtIdx++;
                    }
                }
                edgeCount++;
            }
#endif

#ifdef LGC_POINT_CLUSTER_ADD_SUBCLUSTER_SURFACE_POINTS
            Vector3 corner0Offset = corner0 - this->_position;
            Vector3 gridOriginOffset = _pointMaskOrigins[i];
            Vector3 gridStepX = _pointMaskSizeX[i];
            Vector3 gridStepY = _pointMaskSizeY[i];

            Vector3 gridPt1 = this->_position;
            Vector3 gridPt15 = gridPt1 + corner0Offset;
            Vector3 gridPt2 = gridPt15 + modelOrientation.transform(gridOriginOffset);

            std::cout << " surface point grid data: corner0 = " << corner0 << ", corner0Offsett = " << corner0Offset
                      << ", gridOriginOffset = " << gridOriginOffset << ", gridStepX = " << gridStepX << ", gridStepY = " << gridStepY << std::endl;

            unsigned long numSurfacePts = 0;

            for (unsigned int p = 0; p < pmSize; p++)
            {
                for (unsigned int q = 0; q < pmSize; q++)
                {
                    unsigned int tileX = (p) / LGC_POINT_MODEL_GRID_TILE_SIZE;
                    unsigned int tileY = (q) / LGC_POINT_MODEL_GRID_TILE_SIZE;

                    unsigned int tileIdxX = 0;
                    if (p < LGC_POINT_MODEL_GRID_TILE_SIZE)
                        tileIdxX = p;
                    else
                        tileIdxX = p % LGC_POINT_MODEL_GRID_TILE_SIZE;

                    unsigned int tileIdxY = 0;
                    if (q < LGC_POINT_MODEL_GRID_TILE_SIZE)
                        tileIdxY = q;
                    else
                        tileIdxY = q % LGC_POINT_MODEL_GRID_TILE_SIZE;

                    unsigned long targetedBit = (tileIdxX * LGC_POINT_MODEL_GRID_TILE_SIZE + tileIdxY);
                    unsigned long shift = (unsigned long) (pow(2, targetedBit));

                    if (_pointMasks[i][tilesPerSide * tileX + tileY] & shift)
                    {
                        Vector3 gp(gridPt2 + modelOrientation.transform(gridStepX * p) - modelOrientation.transform(gridStepY * q));
                        addSurfacePoint(gp, p * gridSize[0] + q, (unsigned long) _indices[i].first);
                        std::cout << "  * added surf. point: " << gp << ", point index = " << p * gridSize[0] + q << ", facet index = " << _indices[i].first << std::endl;
                        numSurfacePts++;
                    }
                }
            }

            if (numSurfacePts == 0)
            {
                std::cerr << " WARNING: Surface point calculation failure (degenerate case, cause unknown); placing centroid as replacement" << std::endl;
                Vector3 triCentroid((corner0.x() + corner1.x() + corner2.x()) / 3.0f,
                                    (corner0.y() + corner1.y() + corner2.y()) / 3.0f,
                                    (corner0.z() + corner1.z() + corner2.z()) / 3.0f);

                addSurfacePoint(triCentroid, 0, (unsigned long) _indices[i].first);
            }

            facetIndex++;
#endif
        }
    }
}

template <class LGCDataTypes>
void LGCPointCluster<LGCDataTypes>::buildPQPModel()
{
    std::cout << "LGCPointCluster<LGCDataTypes>::buildPQPModel(" << this->getName() << ")" << std::endl;
    _pqp_tree = new PQP_FlatModel();

    BV rootBV;
    rootBV.child_range_min = this->minFacetRange();
    rootBV.child_range_min = this->maxFacetRange();

    rootBV.d[0] = this->clusterObb()->halfExtents().x();
    rootBV.d[1] = this->clusterObb()->halfExtents().y();
    rootBV.d[2] = this->clusterObb()->halfExtents().z();

    rootBV.To[0] = this->clusterObb()->center().x();
    rootBV.To[1] = this->clusterObb()->center().y();
    rootBV.To[2] = this->clusterObb()->center().z();

    rootBV.R[0][0] = this->clusterObb()->localAxis(0).x();
    rootBV.R[1][0] = this->clusterObb()->localAxis(0).y();
    rootBV.R[2][0] = this->clusterObb()->localAxis(0).z();

    rootBV.R[0][1] = this->clusterObb()->localAxis(1).x();
    rootBV.R[1][1] = this->clusterObb()->localAxis(1).y();
    rootBV.R[2][1] = this->clusterObb()->localAxis(1).z();

    rootBV.R[0][2] = this->clusterObb()->localAxis(2).x();
    rootBV.R[1][2] = this->clusterObb()->localAxis(2).y();
    rootBV.R[2][2] = this->clusterObb()->localAxis(2).z();

    _pqp_tree->AddBV(rootBV);

    std::cout << " Root BV tri range: " << _pqp_tree->child(0)->child_range_min << " - " << _pqp_tree->child(0)->child_range_max << std::endl;

    for (unsigned long i = 0; i < this->numChildren(); i++)
    {
        LGCObb<LGCDataTypes>* clObb = childCluster(i)->clusterObb();
        BV flatBV;
        flatBV.child_range_min = childCluster(i)->minFacetRange();
        flatBV.child_range_max = childCluster(i)->maxFacetRange();

        flatBV.d[0] = clObb->halfExtents().x();
        flatBV.d[1] = clObb->halfExtents().y();
        flatBV.d[2] = clObb->halfExtents().z();

        flatBV.To[0] = clObb->center().x();
        flatBV.To[1] = clObb->center().y();
        flatBV.To[2] = clObb->center().z();

        flatBV.R[0][0] = clObb->localAxis(0).x();
        flatBV.R[1][0] = clObb->localAxis(0).y();
        flatBV.R[2][0] = clObb->localAxis(0).z();

        flatBV.R[0][1] = clObb->localAxis(1).x();
        flatBV.R[1][1] = clObb->localAxis(1).y();
        flatBV.R[2][1] = clObb->localAxis(1).z();

        flatBV.R[0][2] = clObb->localAxis(2).x();
        flatBV.R[1][2] = clObb->localAxis(2).y();
        flatBV.R[2][2] = clObb->localAxis(2).z();

        _pqp_tree->AddBV(flatBV);

        std::cout << "  BV " << i+1 << " tri range: " << _pqp_tree->child(i+1)->child_range_min << " - " << _pqp_tree->child(i+1)->child_range_max << std::endl;
    }
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::drawVoxel(const sofa::core::visual::VisualParams* vparams, const Vector3& center, const Vector3& vmin, const Vector3& vmax, const Vec4f& color, bool rootNode)
{
#if 0
    glPushMatrix();

    glTranslated(-translation.x(), -translation.y(), -translation.z());
    drawCoordinateMarkerGL();
    glTranslated(center.x(), center.y(), center.z());
    drawCoordinateMarkerGL(1,1,color, color, color);

    glMultMatrixd(rotation.transposed().ptr());
    drawCoordinateMarkerGL();
    glMultMatrixd(rotation.ptr());

    glColor3f(color.x(), color.y(), color.z());
    Vector3 halfExtents = (vmax-vmin)* 0.5f;

    Vector3 corners[8];
    corners[0] = /*center + */Vector3(halfExtents.x(), halfExtents.y(), halfExtents.z());
    corners[1] = /*center + */Vector3(halfExtents.x(), halfExtents.y(), -halfExtents.z());
    corners[2] = /*center + */Vector3(halfExtents.x(), -halfExtents.y(), halfExtents.z());
    corners[3] = /*center + */Vector3(halfExtents.x(), -halfExtents.y(), -halfExtents.z());
    corners[4] = /*center + */Vector3(-halfExtents.x(), halfExtents.y(), halfExtents.z());
    corners[5] = /*center + */Vector3(-halfExtents.x(), halfExtents.y(), -halfExtents.z());
    corners[6] = /*center + */Vector3(-halfExtents.x(), -halfExtents.y(), halfExtents.z());
    corners[7] = /*center + */Vector3(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    for (int k = 0; k <= 7; k++)
    {
        corners[k] = rotation.transform(corners[k]);
    }

    /*std::vector<Vector3> boxPts;

    boxPts.push_back(corners[0]);
    boxPts.push_back(corners[1]);
    boxPts.push_back(corners[0]);
    boxPts.push_back(corners[2]);
    boxPts.push_back(corners[1]);
    boxPts.push_back(corners[3]);
    boxPts.push_back(corners[2]);
    boxPts.push_back(corners[3]);

    boxPts.push_back(corners[4]);
    boxPts.push_back(corners[5]);
    boxPts.push_back(corners[4]);
    boxPts.push_back(corners[6]);
    boxPts.push_back(corners[5]);
    boxPts.push_back(corners[7]);
    boxPts.push_back(corners[6]);
    boxPts.push_back(corners[7]);

    boxPts.push_back(corners[0]);
    boxPts.push_back(corners[4]);
    boxPts.push_back(corners[1]);
    boxPts.push_back(corners[5]);
    boxPts.push_back(corners[2]);
    boxPts.push_back(corners[6]);
    boxPts.push_back(corners[3]);
    boxPts.push_back(corners[7]);*/

    glBegin(GL_LINES);
    glVertex3d(corners[0].x(), corners[0].y(),corners[0].z());
    glVertex3d(corners[7].x(), corners[7].y(),corners[7].z());

    glVertex3d(corners[0].x(), corners[0].y(), corners[0].z());
    glVertex3d(corners[1].x(), corners[1].y(), corners[1].z());
    glVertex3d(corners[0].x(), corners[0].y(), corners[0].z());
    glVertex3d(corners[2].x(), corners[2].y(), corners[2].z());

    glVertex3d(corners[1].x(), corners[1].y(), corners[1].z());
    glVertex3d(corners[3].x(), corners[3].y(), corners[3].z());
    glVertex3d(corners[2].x(), corners[2].y(), corners[2].z());
    glVertex3d(corners[3].x(), corners[3].y(), corners[3].z());

    glVertex3d(corners[4].x(), corners[4].y(), corners[4].z());
    glVertex3d(corners[5].x(), corners[5].y(), corners[5].z());
    glVertex3d(corners[4].x(), corners[4].y(), corners[4].z());
    glVertex3d(corners[6].x(), corners[6].y(), corners[6].z());

    glVertex3d(corners[5].x(), corners[5].y(), corners[5].z());
    glVertex3d(corners[7].x(), corners[7].y(), corners[7].z());
    glVertex3d(corners[6].x(), corners[6].y(), corners[6].z());
    glVertex3d(corners[7].x(), corners[7].y(), corners[7].z());

    glVertex3d(corners[0].x(), corners[0].y(), corners[0].z());
    glVertex3d(corners[4].x(), corners[4].y(), corners[4].z());
    glVertex3d(corners[1].x(), corners[1].y(), corners[1].z());
    glVertex3d(corners[5].x(), corners[5].y(), corners[5].z());

    glVertex3d(corners[2].x(), corners[2].y(), corners[2].z());
    glVertex3d(corners[6].x(), corners[6].y(), corners[6].z());
    glVertex3d(corners[3].x(), corners[3].y(), corners[3].z());
    glVertex3d(corners[7].x(), corners[7].y(), corners[7].z());
    glEnd();

    //vparams->drawTool()->drawLines(boxPts, 0.003f, color);
    glPopMatrix();
#else

    Matrix4 modelOrientation; modelOrientation.identity();
    Matrix3 modelRotation;

    _cluster->lgcCollisionModel()->orientation().toMatrix(modelRotation);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            modelOrientation(i,j) = modelRotation(i,j);

    Vector3 rotatedModelPosition = _cluster->lgcCollisionModel()->position();

    Vector3 halfExtents = (vmax-vmin)* 0.5f;

    if (rootNode) drawCoordinateMarkerGL();

    Vector3 targetPos = modelOrientation.transposed().transform(rotatedModelPosition);
    Vector3 targetCenter = center;

    glPushMatrix();
    glMultMatrixd(modelOrientation.transposed().ptr());
    if (rootNode)
    {
        drawCoordinateMarkerGL(4,4);
        glBegin(GL_LINES);
        glColor4d(color.x(), color.y(), color.z(), color.w());
        glVertex3d(0,0,0);
        glVertex3d(targetPos.x(), targetPos.y(), targetPos.z());
        glEnd();
    }

    glPushMatrix();

    glTranslated(targetPos.x(), targetPos.y(), targetPos.z());
    if (rootNode)
    {
        drawCoordinateMarkerGL();
        glBegin(GL_LINES);
        glColor4d(color.x(), color.y(), color.z(), color.w());
        glVertex3d(0,0,0);
        glVertex3d(targetCenter.x(), targetCenter.y(), targetCenter.z());
        glEnd();
    }

    glPushMatrix();
    glTranslated(targetCenter.x(), targetCenter.y(), targetCenter.z());
    if (rootNode) drawCoordinateMarkerGL(1,4,color,color,color);

    glLineWidth(1);
    glBegin(GL_LINES);
    glColor4d(color.x(), color.y(), color.z(), color.w());
    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());

    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glEnd();
    glPopMatrix();
    glPopMatrix();
    glPopMatrix();
#endif
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::dumpOcTree()
{
    if (!_cluster->computeOcTree())
        return;

    OcTreeSinglePoint* ocTree = _spOcTree;
    if (ocTree != NULL)
    {
        std::cout << " Cast to OcTreeBase OK." << std::endl;
        std::cout << " tree depth: " << ocTree->getTreeDepth() << std::endl;
        /*OcTree::OctreeKey*/ pcl::octree::OctreeKey ocKey;
        ocKey.x = ocKey.y = ocKey.z = 0;
        dumpOcTreeRec(ocTree, (OcTreeBranch*) ocTree->getRootNode(), ocKey, 0);
    }
    else
    {
        std::cout << "ERROR: Cast to OcTreeBase failed!" << std::endl;
    }
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::dumpOcTreeRec(OcTreeSinglePoint* ocTree, const OcTreeBranch* ocNode, const /*OcTreeSinglePoint::OctreeKey*/ pcl::octree::OctreeKey& ocKey, unsigned long depth)
{
    // child iterator
    unsigned int childIdx;
    std::cout << " Cell coordinates: " << ocKey.x << "," << ocKey.y << "," << ocKey.z << ", node type: " << ocNode->getNodeType() << std::endl;

    //char nodeMask = ocTree->getBranchBitPattern(*ocNode);
    //std::cout << " node occupancy mask: " << (nodeMask & 0x1) << (nodeMask & 0x2) << (nodeMask & 0x4) << (nodeMask & 0x8) << (nodeMask & 16) << (nodeMask & 32) << (nodeMask & 64) << (nodeMask & 128) << std::endl;

    std::cout << " Occupied child nodes: ";
    for (childIdx = 0; childIdx < 8; childIdx++)
    {
        // if child exist
        if (ocTree->branchHasChild(*ocNode, (unsigned char) childIdx))
        {
            std::cout << "|" << childIdx;
        }
    }
    std::cout << "|" << std::endl;

    // iterate over all children
    for (childIdx = 0; childIdx < 8; childIdx++)
    {
        // if child exist
        if (ocTree->branchHasChild(*ocNode, (unsigned char) childIdx))
        {
            //std::cout << "  child " << childIdx << " occupied" << std::endl;

            const OcTreeNode * childNode = ocTree->getBranchChildPtr(*ocNode, (unsigned char) childIdx);
            //childNode = ocTree->getBranchChild(*ocNode, childIdx);

            switch (childNode->getNodeType ())
            {
                default:
                    break;

                case pcl::octree::LEAF_NODE:
                {
                    OcTreeLeaf* childLeaf = (OcTreeLeaf*)childNode;
                    std::cout << "  hit an octree leaf" << std::endl; //, data type " << childLeaf->getLeafDataType()

                    std::cout << "  leaf's parent coordinates: " << ocKey.x << "," << ocKey.y << "," << ocKey.z << std::endl;

                    Eigen::Vector3f voxMin, voxMax;
                    _spOcTree->genVoxelBoundsFromOctreeKey(ocKey, depth, voxMin, voxMax);

                    std::cout << "   voxel bounds: " << voxMin.x() << "," << voxMin.y() << "," << voxMin.z() << " -- " << voxMax.x() << "," << voxMax.y() << "," << voxMax.z() << std::endl;

#if 0
                    std::vector<int> leafPts;
                    if (childLeaf->getLeafDataType() == pcl::octree::LEAF_WITH_SCALAR_DATA)
                    {
                        std::cout << "   single-point arg: ";
                        childLeaf->getData(leafPts);
                        if (leafPts.size() > 0)
                            std::cout << leafPts.at(0) << std::endl;
                        else
                            std::cout << "   no point received!" << std::endl;
                    }
                    else if (childLeaf->getLeafDataType() == pcl::octree::LEAF_WITH_VECTOR_DATA)
                    {
                        childLeaf->getData(leafPts);
                        if (leafPts.size() > 0)
                        {
                            std::cout << "   multi-point arg: " << leafPts.size() << std::endl;
                            for (unsigned int p = 0; p < leafPts.size(); p++)
                                std::cout << "    * " << leafPts.at(p) << std::endl;

                            OcTree::OctreeKey leafKey;
                            _spOcTree->genOctreeKeyforPoint(_cloud->points.at(leafPts.at(0)), leafKey);

                            std::cout << "  leaf coordinates: " << leafKey.x << "," << leafKey.y << "," << leafKey.z << std::endl;
                            Eigen::Vector3f voxMin, voxMax;
                            _spOcTree->genVoxelBoundsFromOctreeKey(leafKey, depth, voxMin, voxMax);

                            std::cout << "   voxel bounds: " << voxMin.x() << "," << voxMin.y() << "," << voxMin.z() << " -- " << voxMax.x() << "," << voxMax.y() << "," << voxMax.z() << std::endl;
                        }
                    }
#endif
                    break;
                }
            }
        }
    }

    for (childIdx = 0; childIdx < 8; childIdx++)
    {
        // if child exist
        if (ocTree->branchHasChild(*ocNode, childIdx))
        {
            //std::cout << "  child " << childIdx << " occupied" << std::endl;

            const OcTreeNode * childNode = ocTree->getBranchChildPtr(*ocNode, childIdx);

            switch (childNode->getNodeType ())
            {
                default:
                   break;

                case pcl::octree::BRANCH_NODE:
                {
                    // generate new key for current branch voxel
                    OcTreeKey newKey;
                    newKey.x = (ocKey.x << 1) | (!!(childIdx & (1 << 2)));
                    newKey.y = (ocKey.y << 1) | (!!(childIdx & (1 << 1)));
                    newKey.z = (ocKey.z << 1) | (!!(childIdx & (1 << 0)));

                    // recursively proceed with indexed child branch
                    std::cout << "== Proceeding into deeper branch index " << childIdx << ", depth " << depth + 1 << std::endl;
                    dumpOcTreeRec(ocTree, (OcTreeBranch*)childNode, newKey, depth + 1);
                    break;
                }
            }
        }
    }
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::dumpKdTree()
{
    if (!_cluster->computeKdTree())
        return;

    std::cout << "Root of k-d-tree: " << std::endl;
    std::cout << " index range   : " << _kdTree->kdTreeRoot()->left << ", " << _kdTree->kdTreeRoot()->right << std::endl;
    std::cout << " subdiv. dim.  : " << _kdTree->kdTreeRoot()->divfeat;
    if (_kdTree->kdTreeRoot()->divfeat == 0)
        std::cout << " along X" << std::endl;
    else if (_kdTree->kdTreeRoot()->divfeat == 1)
        std::cout << " along Y" << std::endl;
    else if (_kdTree->kdTreeRoot()->divfeat == 2)
        std::cout << " along Z" << std::endl;
    else
        std::cout << std::endl;

    std::cout << " divlow/divhigh: " << _kdTree->kdTreeRoot()->divlow << "/" << _kdTree->kdTreeRoot()->divhigh << std::endl;
    std::cout << " child 1/2 set : " << (_kdTree->kdTreeRoot()->child1 != NULL) << "/" << (_kdTree->kdTreeRoot()->child2 != NULL) << std::endl;
    std::cout << " node is leaf  : " << ((_kdTree->kdTreeRoot()->child1 == NULL) && (_kdTree->kdTreeRoot()->child2 == NULL)) << std::endl;

    const KdTreeBoundingBox& rbb = _kdTree->kdTreeIndex()->rootBoundingBox();
    std::cout << " Root bounding box: " << rbb[0].low << "," << rbb[1].low << "," << rbb[2].low << "," << rbb[0].high << "," << rbb[1].high << "," << rbb[2].high << std::endl;

    if (_kdTree->kdTreeRoot()->child1)
        dumpKdTreeRec(_kdTree->kdTreeRoot()->child1);

    if (_kdTree->kdTreeRoot()->child2)
        dumpKdTreeRec(_kdTree->kdTreeRoot()->child2);
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::dumpKdTreeRec(KdTreeNode node)
{
    std::cout << " k-d-tree child: " << std::endl;
    std::cout << "  index range   : " << node->left << ", " << node->right << std::endl;
    std::cout << "  subdiv. dim.  : " << node->divfeat;
    if (node->divfeat == 0)
        std::cout << " along X" << std::endl;
    else if (node->divfeat == 1)
        std::cout << " along Y" << std::endl;
    else if (node->divfeat == 2)
        std::cout << " along Z" << std::endl;
    else
        std::cout << std::endl;

    std::cout << "  divlow/divhigh: " << node->divlow << "/" << node->divhigh << std::endl;
    std::cout << "  child 1/2 set : " << (node->child1 != NULL) << "/" << (node->child2 != NULL) << std::endl;
    std::cout << "  node is leaf  : " << ((node->child1 == NULL) && (node->child2 == NULL)) << std::endl;

    std::cout << "  bounding box  : " << node->bBoxMin[0] << "," << node->bBoxMin[1] << "," << node->bBoxMin[2] << "," << node->bBoxMax[0] << "," << node->bBoxMax[1] << "," << node->bBoxMax[2] << std::endl;

    if (node->child1)
        dumpKdTreeRec(node->child1);

    if (node->child2)
        dumpKdTreeRec(node->child2);
}


template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::drawOcTree(const core::visual::VisualParams* vparams, bool showVerbose)
{
    if (!_cluster->computeOcTree())
        return;

    if (_searchOcTree /*_spOcTree*/ == NULL)
        return;

    Matrix4 modelOrientation; modelOrientation.identity();
    Matrix3 modelRotation;

    _cluster->lgcCollisionModel()->orientation().toMatrix(modelRotation);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            modelOrientation(i,j) = modelRotation(i,j);

    Vector3 targetPos = modelOrientation.transposed().transform(_cluster->lgcCollisionModel()->position());

    glPushMatrix();
    glMultMatrixd(modelOrientation.transposed().ptr());

    glTranslated(targetPos.x(), targetPos.y(), targetPos.z());
    const std::vector<pcl::LGCPointTypeMin, Eigen::aligned_allocator<pcl::LGCPointTypeMin> > cloudPoints = _searchOcTree->getInputCloud()->points;
    //std::cout << "   " << _cluster->getName() << " cloud size: " << cloudPoints.size() << std::endl;
    if (cloudPoints.size() > 0)
    {
        glColor4d(_cluster->getClusterColor().x(), _cluster->getClusterColor().y(), _cluster->getClusterColor().z(), _cluster->getClusterColor().w());
        glPointSize(4.0f);
        glBegin(GL_POINTS);
        for (int i = 0; i < cloudPoints.size(); i++)
        {
            glVertex3d(cloudPoints[i].x, cloudPoints[i].y, cloudPoints[i].z);
        }
        glEnd();
        glPointSize(1.0f);
    }
    glPopMatrix();

    /*OcTree::OctreeKey*/ pcl::octree::OctreeKey ocKey;
    ocKey.x = ocKey.y = ocKey.z = 0;

    OcTreeBranch* ocNode = (OcTreeBranch*) /*_spOcTree*/_searchOcTree->getRootNode();

    Vec4f nodeColor(1,1,1,0.75f);
    Vector3 nodeKeyVec(ocKey.x, ocKey.y, ocKey.z);
    if (_nodeColors.find(nodeKeyVec) == _nodeColors.end())
    {
        // std::cout << " key " << nodeKeyVec << " inserts new randomColor: " << nodeColor << std::endl;
        _nodeColors[nodeKeyVec] = nodeColor;
    }

    Eigen::Vector3f voxMin, voxMax;
    /*_spOcTree*/_searchOcTree->genVoxelBoundsFromOctreeKey(ocKey, 0, voxMin, voxMax);
    pcl::LGCPointTypeMin vcp;
    /*_spOcTree*/_searchOcTree->genVoxelCenterFromOctreeKey(ocKey, 0, vcp);

    Vector3 vtcenter(vcp.x, vcp.y, vcp.z);
    Vector3 vtmin(voxMin.x(), voxMin.y(), voxMin.z());
    Vector3 vtmax(voxMax.x(), voxMax.y(), voxMax.z());

    drawVoxel(vparams, vtcenter, vtmin, vtmax, nodeColor, true);

#if 1
    for (unsigned int childIdx = 0; childIdx < 8; childIdx++)
    {
        if (/*_spOcTree*/_searchOcTree->branchHasChild(*ocNode, childIdx))
        {
            OcTreeNode* childNode = /*_spOcTree*/_searchOcTree->getBranchChildPtr(*ocNode, childIdx);
            if (childNode)
            {
                if (childNode->getNodeType() == pcl::octree::BRANCH_NODE)
                {
                    OcTreeKey branchKey;
                    branchKey.x = (ocKey.x << 1) | (!!(childIdx & (1 << 2)));
                    branchKey.y = (ocKey.y << 1) | (!!(childIdx & (1 << 1)));
                    branchKey.z = (ocKey.z << 1) | (!!(childIdx & (1 << 0)));

                    Eigen::Vector3f bvoxMin, bvoxMax;
                    /*_spOcTree*/_searchOcTree->genVoxelBoundsFromOctreeKey(branchKey, 1, bvoxMin, bvoxMax);
                    pcl::LGCPointTypeMin bvcp;
                    /*_spOcTree*/_searchOcTree->genVoxelCenterFromOctreeKey(branchKey, 1, bvcp);

                    Vec4f nodeColor;
                    Vector3 nodeKeyVec(branchKey.x, branchKey.y, branchKey.z);
                    if (_nodeColors.find(nodeKeyVec) == _nodeColors.end())
                    {
                        nodeColor = Vec4f(0.12f * childIdx,0.12f * childIdx,0.12f * childIdx,1.0f);
                        // std::cout << " key " << nodeKeyVec << " inserts new randomColor: " << nodeColor << std::endl;
                        _nodeColors[nodeKeyVec] = nodeColor;
                    }
                    else
                    {
                        nodeColor = _nodeColors[nodeKeyVec];
                        if (nodeColor.x() == 0.0f ||
                            nodeColor.y() == 0.0f ||
                            nodeColor.z() == 0.0f)
                            nodeColor = Vec4f(0.8f, 0, 0, 1);
                    }

                    Vector3 bvcenter(bvcp.x, bvcp.y, bvcp.z);
                    Vector3 bvmin(bvoxMin.x(), bvoxMin.y(), bvoxMin.z());
                    Vector3 bvmax(bvoxMax.x(), bvoxMax.y(), bvoxMax.z());

                    drawVoxel(vparams, bvcenter, bvmin, bvmax, nodeColor, false);
                }
                else if (childNode->getNodeType() == pcl::octree::LEAF_NODE)
                {
#if 0
                    if (showVerbose)
                    {
                        OcTreeLeaf* childLeaf = (OcTreeLeaf*)childNode;
                        std::vector<int>& pointIndices = childLeaf->getContainer().getPointIndicesVector();

                        std::vector<Vector3> cellPts;
                        for (unsigned int p = 0; p < pointIndices.size(); p++)
                        {
                            pcl::LGCPointTypeMin pt = _cloud->points.at(pointIndices.at(p));
                            cellPts.push_back(Vector3(pt.x,pt.y,pt.z));
                        }

                        vparams->drawTool()->drawSpheres(cellPts, 0.001f, color);
                    }

                    std::vector<int> leafPts;
                    if (childLeaf->getLeafDataType() == pcl::octree::LEAF_WITH_VECTOR_DATA)
                    {
                        std::vector<Vector3> cellPts;
                        childLeaf->getData(leafPts);
                        if (leafPts.size() > 0)
                        {
                            if (showVerbose)
                            {
                                for (unsigned int p = 0; p < leafPts.size(); p++)
                                {
                                    pcl::LGCPointType pt = _cloud->points.at(leafPts.at(p));
                                    cellPts.push_back(Vector3(pt.x,pt.y,pt.z));
                                }
                            }
                            vparams->drawTool()->drawSpheres(cellPts, 0.001f, nodeColor);

                            OcTree::OctreeKey leafKey;
                            _spOcTree->genOctreeKeyforPoint(_cloud->points.at(leafPts.at(0)), leafKey);

                            Vec4f nodeColor;
                            Vector3 nodeKeyVec(leafKey.x, leafKey.y, leafKey.z);
                            if (_nodeColors.find(nodeKeyVec) == _nodeColors.end())
                            {
                                nodeColor = randomColor();
                                std::cout << " key " << nodeKeyVec << " inserts new randomColor: " << nodeColor << std::endl;
                                _nodeColors[nodeKeyVec] = nodeColor;
                            }
                            else
                            {
                                nodeColor = _nodeColors[nodeKeyVec];
                                if (nodeColor.x() == 0.0f ||
                                    nodeColor.y() == 0.0f ||
                                    nodeColor.z() == 0.0f)
                                    nodeColor = Vec4f(0.8f, 0, 0, 1);
                            }

                            Eigen::Vector3f voxMin, voxMax;
                            _spOcTree->genVoxelBoundsFromOctreeKey(leafKey, 1, voxMin, voxMax);
                            pcl::LGCPointType vcp;
                            _spOcTree->genVoxelCenterFromOctreeKey(leafKey, 1, vcp);

                            Vector3 vcenter(vcp.x, vcp.y, vcp.z);
                            Vector3 vmin(voxMin.x(), voxMin.y(), voxMin.z());
                            Vector3 vmax(voxMax.x(), voxMax.y(), voxMax.z());
                            drawVoxel(vparams, vcenter, vmin, vmax, nodeColor);
                        }
                    }
#endif
                }
            }
        }
    }
#endif
#if 1
    for (unsigned int childIdx = 0; childIdx < 8; childIdx++)
    {
        if (/*_spOcTree*/_searchOcTree->branchHasChild(*ocNode, childIdx))
        {
            OcTreeNode* childNode = /*_spOcTree*/_searchOcTree->getBranchChildPtr(*ocNode, childIdx);
            if (childNode)
            {
                if (childNode->getNodeType() == pcl::octree::BRANCH_NODE)
                {
                    OcTreeKey newKey;
                    newKey.x = (ocKey.x << 1) | (!!(childIdx & (1 << 2)));
                    newKey.y = (ocKey.y << 1) | (!!(childIdx & (1 << 1)));
                    newKey.z = (ocKey.z << 1) | (!!(childIdx & (1 << 0)));

                    drawOcTreeRec(vparams, (const OcTreeBranch*) childNode, newKey, 2, _cluster->_lgcCollisionModel->position(), modelOrientation, showVerbose);
                }
            }
        }
    }
#endif
    //glPopMatrix();
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::drawOcTreeRec(const core::visual::VisualParams* vparams, const OcTreeBranch* branchNode, const /*OcTree::OctreeKey*/ pcl::octree::OctreeKey& ocKey, unsigned long depth, Vector3 translation, Matrix4 rotation, bool showVerbose)
{
    for (unsigned int childIdx = 0; childIdx < 8; childIdx++)
    {
        if (/*_spOcTree*/_searchOcTree->branchHasChild(*branchNode, childIdx))
        {
            OcTreeNode* childNode = /*_spOcTree*/_searchOcTree->getBranchChildPtr(*branchNode, childIdx);
            if (childNode)
            {
                if (childNode->getNodeType() == pcl::octree::BRANCH_NODE)
                {
                    OcTreeKey branchKey;
                    branchKey.x = (ocKey.x << 1) | (!!(childIdx & (1 << 2)));
                    branchKey.y = (ocKey.y << 1) | (!!(childIdx & (1 << 1)));
                    branchKey.z = (ocKey.z << 1) | (!!(childIdx & (1 << 0)));

                    Eigen::Vector3f bvoxMin, bvoxMax;
                    /*_spOcTree*/_searchOcTree->genVoxelBoundsFromOctreeKey(branchKey, depth, bvoxMin, bvoxMax);
                    pcl::LGCPointTypeMin bvcp;
                    /*_spOcTree*/_searchOcTree->genVoxelCenterFromOctreeKey(branchKey, depth, bvcp);

                    Vec4f nodeColor;
                    Vector3 nodeKeyVec(branchKey.x, branchKey.y, branchKey.z);
                    if (_nodeColors.find(nodeKeyVec) == _nodeColors.end())
                    {
                        nodeColor = randomColor();
                        // std::cout << " key " << nodeKeyVec << " inserts new randomColor: " << nodeColor << std::endl;
                        _nodeColors[nodeKeyVec] = nodeColor;
                    }
                    else
                    {
                        nodeColor = _nodeColors[nodeKeyVec];
                        if (nodeColor.x() == 0.0f ||
                            nodeColor.y() == 0.0f ||
                            nodeColor.z() == 0.0f)
                            nodeColor = Vec4f(0.8f, 0, 0, 1);
                    }

                    Vector3 bvcenter(bvcp.x, bvcp.y, bvcp.z);
                    Vector3 bvmin(bvoxMin.x(), bvoxMin.y(), bvoxMin.z());
                    Vector3 bvmax(bvoxMax.x(), bvoxMax.y(), bvoxMax.z());

                    drawVoxel(vparams, bvcenter, bvmin, bvmax, nodeColor, false);
                }
                else if (childNode->getNodeType() == pcl::octree::LEAF_NODE)
                {
#if 0
                    if (showVerbose)
                    {
                        OcTreeLeaf* childLeaf = (OcTreeLeaf*)childNode;
                        std::vector<int>& pointIndices = childLeaf->getContainer().getPointIndicesVector();

                        std::vector<Vector3> cellPts;
                        for (unsigned int p = 0; p < pointIndices.size(); p++)
                        {
                            pcl::LGCPointTypeMin pt = _cloud->points.at(pointIndices.at(p));
                            cellPts.push_back(Vector3(pt.x,pt.y,pt.z));
                        }

                        vparams->drawTool()->drawSpheres(cellPts, 0.001f, color);
                    }

                    OcTreeLeaf* childLeaf = (OcTreeLeaf*)childNode;
                    std::vector<int> leafPts;
                    if (childLeaf->getLeafDataType() == pcl::octree::LEAF_WITH_VECTOR_DATA)
                    {
                        std::vector<Vector3> cellPts;
                        childLeaf->getData(leafPts);
                        if (leafPts.size() > 0)
                        {
                            OcTree::OctreeKey leafKey;
                            _spOcTree->genOctreeKeyforPoint(_cloud->points.at(leafPts.at(0)), leafKey);

                            Vec4f nodeColor;
                            Vector3 nodeKeyVec(leafKey.x, leafKey.y, leafKey.z);
                            if (_nodeColors.find(nodeKeyVec) == _nodeColors.end())
                            {
                                nodeColor = randomColor();
                                // std::cout << " key " << nodeKeyVec << " inserts new randomColor: " << nodeColor << std::endl;
                                _nodeColors[nodeKeyVec] = nodeColor;
                            }
                            else
                            {
                                nodeColor = _nodeColors[nodeKeyVec];
                                if (nodeColor.x() == 0.0f ||
                                    nodeColor.y() == 0.0f ||
                                    nodeColor.z() == 0.0f)
                                    nodeColor = Vec4f(0.8f, 0, 0, 1);
                            }

                            if (showVerbose)
                            {
                                for (unsigned int p = 0; p < leafPts.size(); p++)
                                {
                                    pcl::LGCPointType pt = _cloud->points.at(leafPts.at(p));
                                    cellPts.push_back(Vector3(pt.x,pt.y,pt.z));
                                }
                                vparams->drawTool()->drawSpheres(cellPts, 0.001f, nodeColor);
                            }

                            Eigen::Vector3f voxMin, voxMax;
                            _spOcTree->genVoxelBoundsFromOctreeKey(leafKey, depth, voxMin, voxMax);
                            pcl::LGCPointType vcp;
                            _spOcTree->genVoxelCenterFromOctreeKey(leafKey, depth, vcp);

                            Vector3 vcenter(vcp.x, vcp.y, vcp.z);
                            Vector3 vmin(voxMin.x(), voxMin.y(), voxMin.z());
                            Vector3 vmax(voxMax.x(), voxMax.y(), voxMax.z());
                            drawVoxel(vparams, vcenter, vmin, vmax, nodeColor);
                        }
                    }
#endif
                }
            }
        }
    }

    for (unsigned int childIdx = 0; childIdx < 8; childIdx++)
    {
        if (/*_spOcTree*/_searchOcTree->branchHasChild(*branchNode, childIdx))
        {
            OcTreeNode* childNode = /*_spOcTree*/_searchOcTree->getBranchChildPtr(*branchNode, childIdx);
            if (childNode)
            {
                if (childNode->getNodeType() == pcl::octree::BRANCH_NODE)
                {
                    OcTreeKey newKey;
                    newKey.x = (ocKey.x << 1) | (!!(childIdx & (1 << 2)));
                    newKey.y = (ocKey.y << 1) | (!!(childIdx & (1 << 1)));
                    newKey.z = (ocKey.z << 1) | (!!(childIdx & (1 << 0)));

                    drawOcTreeRec(vparams, (const OcTreeBranch*) childNode, newKey, depth + 1, translation, rotation, showVerbose);
                }
            }
        }
    }
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::drawKdTree(const core::visual::VisualParams* vparams)
{
    if (!_cluster->computeKdTree())
        return;

    if (_kdTree == NULL)
        return;

    const KdTreeBoundingBox& rbb = _kdTree->kdTreeIndex()->rootBoundingBox();
    Vector3 rbMin(rbb[0].low,rbb[1].low,rbb[2].low);
    Vector3 rbMax(rbb[0].high,rbb[1].high,rbb[2].high);
    Vector3 rbCenter = (rbMax + rbMin) * 0.5f;

    drawVoxel(vparams, rbCenter, rbMin, rbMax, Vec4f(0,0.5f,0,0.5f), false);

    if (_kdTree->kdTreeRoot()->child1 != NULL)
        drawKdTreeRec(vparams, _kdTree->kdTreeRoot()->child1);

    if (_kdTree->kdTreeRoot()->child2 != NULL)
        drawKdTreeRec(vparams, _kdTree->kdTreeRoot()->child2);
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::drawKdTreeRec(const core::visual::VisualParams* vparams, KdTreeNode kdNode)
{
    if (kdNode != NULL)
    {
        Vector3 rbMin(kdNode->bBoxMin[0], kdNode->bBoxMin[1], kdNode->bBoxMin[2]);
        Vector3 rbMax(kdNode->bBoxMax[0], kdNode->bBoxMax[1], kdNode->bBoxMax[2]);
        Vector3 rbCenter = (rbMax + rbMin) * 0.5f;

        drawVoxel(vparams, rbCenter, rbMin, rbMax, Vec4f(0,0.5f,0,0.5f), false);

        if (kdNode->child1 != NULL)
            drawKdTreeRec(vparams, kdNode->child1);

        if (kdNode->child2 != NULL)
            drawKdTreeRec(vparams, kdNode->child2);
    }
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::buildChildHierarchy()
{
    typedef sofa::component::collision::EPEC_Kernel Kernel;

    std::cout << "LGCPointClusterPrivate<LGCDataTypes>::buildChildHierarchy()" << std::endl;
    if (this->_segmentedClusters.size() > 1)
    {
        std::cout << "Adding " << _segmentedClusters.size() << " sub-clusters." << std::endl;
        unsigned int childClusterCount = 1;
        _cluster->_clusterIndex = 0;
        std::stringstream childNameStream;
        for (std::map<unsigned long, pcl::PointCloud<pcl::LGCPointTypeMin>::Ptr>::const_iterator clIt = this->_segmentedClusters.begin();
             clIt != this->_segmentedClusters.end(); clIt++)
        {
            pcl::PointCloud<pcl::LGCPointTypeMin>::Ptr pcpt = clIt->second;
            std::cout << " * Cloud " << clIt->first << ": " << pcpt->size() << " points" << std::endl;
#if 0
            std::list<CGAL::Point_3<Kernel> > points;
            for (std::vector<Vector3>::const_iterator it = _segmentedPlanes[clIt->first].begin(); it != _segmentedPlanes[clIt->first].end(); it++)
            {
                points.push_back(CGAL::Point_3<Kernel>((*it).x(),(*it).y(),(*it).z()));
            }

            CGAL::Point_3<Kernel> centroid = CGAL::centroid(points.begin(), points.end(), CGAL::Dimension_tag<0>());
            Vector3 childCentroid(CGAL::to_double(centroid.x()),
                                   CGAL::to_double(centroid.y()),
                                   CGAL::to_double(centroid.z()));

            Vector3 childOffset = childCentroid; //(_cluster->position() + _centroid) - (_cluster->position() - childCentroid);
            std::cout << "  child cluster offset: " << childOffset << std::endl;
#endif
            if (_segmentedPlanes.find(clIt->first) != _segmentedPlanes.end() &&
                _segmentedEdges.find(clIt->first) != _segmentedEdges.end() &&
                _segmentedVertices.find(clIt->first) != _segmentedVertices.end())
            {
                LGCPointCluster<LGCDataTypes>* childCluster = new LGCPointCluster<LGCDataTypes>(_segmentedPlanes[clIt->first], _segmentedEdges[clIt->first], _segmentedVertices[clIt->first],
                                                                                                _cluster->position() /*Vector3(0,0,0)*/, _cluster->orientation() /*Quaternion::identity()*/, PARENT_RELATIVE_REFERENCE_FRAME, _cluster, _cluster->collisionModel());
                childNameStream << _cluster->getName() << " child cluster " << childClusterCount;
                childCluster->setName(childNameStream.str());

                childCluster->setOriginalModelPtr(_cluster->_originalModel);
                childCluster->_clusterIndex = childClusterCount;

                childClusterCount++;

#if 0
                pcl::PointCloud<pcl::LGCPointType>::Ptr childCloud = _segmentedClusters[clIt->first];
                unsigned long childSurfacePoints = 0, childEdgePoints = 0, childVertices = 0;

                std::cout << "  transfer cloud data from cloud with " << childCloud->points.size() << " points." << std::endl;
                for (unsigned long r = 0; r < childCloud->points.size(); r++)
                {
                    // Surface
                    if (childCloud->points[r].pointType == 0)
                    {
                        childCluster->setSurfacePointFacetIndex(childSurfacePoints++, childCloud->points[r].facetIdx);
                        std::cout << "   surface point " << childCloud->points[r] << " to child index " << childSurfacePoints << std::endl;
                    }
                    // Edge
                    else if (childCloud->points[r].pointType == 1)
                    {
                        childEdgePoints++;
                        childCluster->setEdgePointFacetIndex(childEdgePoints, childCloud->points[r].facetIdx);
                        childCluster->setEdgePointEdgeIndex(childEdgePoints, childCloud->points[r].edgeIdx);
                        std::cout << "   edge point " << childCloud->points[r] << " to child index " << childEdgePoints << std::endl;
                    }
                    //Vertex
                    else if (childCloud->points[r].pointType == 2)
                    {
                        childCluster->setVertexFacetIndex(childVertices++, childCloud->points[r].facetIdx);
                        std::cout << "   vertex " << childCloud->points[r] << " to child index " << childVertices << std::endl;
                    }
                }
#endif
                this->_cluster->addChild(childCluster);
                childNameStream.str("");
            }
        }
        std::cout << "Child cluster count now: " << _cluster->numChildren() << std::endl;
    }
}

//#define LGC_POINTCLUSTER_USE_TOPOLOGY_FOR_OBB_COMPUTATION
template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::fitObbToCluster()
{
    std::cout << "LGCPointClusterPrivate<LGCDataTypes>::fitObbToCluster(): " << _cluster->getName() << std::endl;

    for (int i = 0; i < 6; i++)
    {
        _obbPlanes[i] = NULL;
        _obbPlaneDrawables[i] = NULL;
    }

    for (int i = 0; i < 3; i++)
    {
        _eigenPlanes[i] = NULL;
    }

    std::cout << "Add vertices: " << _cluster->numVertices() << std::endl;
    for (unsigned long i = 0; i < _cluster->numVertices(); i++)
    {
        Vec<3, Real> tp = _cluster->vertex(i) - _cluster->_parentCluster->position();
        tp = _cluster->_parentCluster->orientation().inverseRotate(tp);
        // Vec<3, Real> tp = _cluster->vertex(i);
        std::cout << " * " << i << ": " << _cluster->vertex(i) << " -- " << tp << std::endl;
        _cluster->_allClusterPts.push_back(tp);
#ifdef LGC_POINTCLUSTER_DEBUG_ALL_CLUSTER_POINTS
        _cluster->_allClusterPtsDraw.push_back(tp);
#endif
    }

#ifdef LGC_OBB_ADD_ALL_CLUSTER_POINTS
    if (_cluster->parentCluster() == NULL)
    {
        std::cout << "Add edge pts: " << _cluster->numEdgePoints() << std::endl;
        for (unsigned long i = 0; i < _cluster->numEdgePoints(); i++)
        {
            //<< ", facet index: " << _cluster->edgePointFacetIndex(_cluster->edgePointIndex(i)) << ", edge index: " << _cluster->edgePointEdgeIndex(_cluster->edgePointIndex(i)) << std::endl;
            Vector3 tp = _cluster->edgePoint(i) - _cluster->_parentCluster->position();
            tp = _cluster->_parentCluster->orientation().inverseRotate(tp);
            //Vec<3, Real> tp = _cluster->edgePoint(i);
            std::cout << " * " << i << ": " << _cluster->edgePoint(i) << " -- " << tp << std::endl;
            _cluster->_allClusterPts.push_back(tp);
    #ifdef LGC_POINTCLUSTER_DEBUG_ALL_CLUSTER_POINTS
            _cluster->_allClusterPtsDraw.push_back(tp);
    #endif
        }

        std::cout << "Add surface pts: " << _cluster->numSurfacePoints() << std::endl;
        for (unsigned long i = 0; i < _cluster->numSurfacePoints(); i++)
        {
            Vector3 tp = _cluster->surfacePoint(i) - _cluster->_parentCluster->position();
            tp = _cluster->_parentCluster->orientation().inverseRotate(tp);
            //Vec<3, Real> tp = _cluster->surfacePoint(i);
            _cluster->_allClusterPts.push_back(tp);
    #ifdef LGC_POINTCLUSTER_DEBUG_ALL_CLUSTER_POINTS
            _cluster->_allClusterPtsDraw.push_back(tp);
    #endif
        }
    }
#endif
    unsigned int i,j;
    // compute covariance matrix
    Vec<3, Real> centroid;
    Mat<3,3,Real> C;
    sofa::helper::ComputeCovarianceMatrix(C, centroid, _cluster->_allClusterPts);

    // get basis vectors
    Vec<3, Real> basis[3];
    sofa::helper::GetRealSymmetricEigenvectors(basis[0], basis[1], basis[2], C);

    Vector3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    Vector3 max(FLT_MIN, FLT_MIN, FLT_MIN);

    // compute min, max projections of box on axes
    for ( i = 0; i < _cluster->_allClusterPts.size(); ++i )
    {
        Vector3 diff = _cluster->_allClusterPts[i] - centroid;
        for (j = 0; j < 3; ++j)
        {
            double length = diff * basis[j];
            if (length > max[j])
            {
                max[j] = length;
            }
            else if (length < min[j])
            {
                min[j] = length;
            }
        }
    }

    // compute center, extents
    Vec<3, Real> obbPosition = centroid;
    Vec<3, Real> obbExtents;
    for ( i = 0; i < 3; ++i )
    {
        obbPosition += basis[i] * 0.5f * (min[i]+max[i]);
        obbExtents[i] = (max[i]-min[i]) * 0.5f;
    }

    _clusterObb = new LGCObbLeaf<LGCDataTypes>(obbPosition, basis[0], basis[1], basis[2], obbExtents, 1, _cluster->_parentCluster->clusterObb() /*NULL*/, _cluster->collisionModel());
    _clusterObb->lcData()._childOffset = obbPosition;

    Vec<3, Real> corner0(_clusterObb->wcData()._halfExtents.x(), _clusterObb->wcData()._halfExtents.y(), _clusterObb->wcData()._halfExtents.z());
    Vec<3, Real> corner1(_clusterObb->wcData()._halfExtents.x(), _clusterObb->wcData()._halfExtents.y(), -_clusterObb->wcData()._halfExtents.z());
    Vec<3, Real> corner2(_clusterObb->wcData()._halfExtents.x(), -_clusterObb->wcData()._halfExtents.y(), _clusterObb->wcData()._halfExtents.z());
    Vec<3, Real> corner3(_clusterObb->wcData()._halfExtents.x(), -_clusterObb->wcData()._halfExtents.y(), -_clusterObb->wcData()._halfExtents.z());
    Vec<3, Real> corner4(-_clusterObb->wcData()._halfExtents.x(), _clusterObb->wcData()._halfExtents.y(), _clusterObb->wcData()._halfExtents.z());
    Vec<3, Real> corner5(-_clusterObb->wcData()._halfExtents.x(), _clusterObb->wcData()._halfExtents.y(), -_clusterObb->wcData()._halfExtents.z());
    Vec<3, Real> corner6(-_clusterObb->wcData()._halfExtents.x(), -_clusterObb->wcData()._halfExtents.y(), _clusterObb->wcData()._halfExtents.z());
    Vec<3, Real> corner7(-_clusterObb->wcData()._halfExtents.x(), -_clusterObb->wcData()._halfExtents.y(), -_clusterObb->wcData()._halfExtents.z());

    Mat<3, 3, Real> obbRotation; obbRotation.identity();
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            obbRotation[j][i] = basis[i][j];

    corner0 = obbRotation * corner0;
    corner1 = obbRotation * corner1;
    corner2 = obbRotation * corner2;
    corner3 = obbRotation * corner3;
    corner4 = obbRotation * corner4;
    corner5 = obbRotation * corner5;
    corner6 = obbRotation * corner6;
    corner7 = obbRotation * corner7;

    _clusterObb->setObrCorner(0, _clusterObb->center() + corner0);
    _clusterObb->setObrCorner(1, _clusterObb->center() + corner4);
    _clusterObb->setObrCorner(2, _clusterObb->center() + corner6);
    _clusterObb->setObrCorner(3, _clusterObb->center() + corner2);
    _clusterObb->setObrCorner(4, _clusterObb->center() + corner1);
    _clusterObb->setObrCorner(5, _clusterObb->center() + corner5);
    _clusterObb->setObrCorner(6, _clusterObb->center() + corner7);
    _clusterObb->setObrCorner(7, _clusterObb->center() + corner3);

    std::stringstream idStr;
    idStr << _cluster->getName() << " cluster OBB";
    _clusterObb->setIdentifier(idStr.str());
    _clusterObb->intersectionHistory().setName(idStr.str() + " intersection history");

    Vec4f historyColor = _cluster->_clusterColor;
    historyColor.w() = historyColor.w() * 0.2f;
    _clusterObb->intersectionHistory().setColor(historyColor);

    _clusterObbDrawable = new ObbDrawable<LGCDataTypes>(_clusterObb, _cluster->d->_clusterObbDrawable, ObbDrawable<LGCDataTypes>::ROTATE_THEN_TRANSLATE);
    _clusterObb->setObbDrawable(*_clusterObbDrawable);

    // Clean up
    _cluster->_allClusterPts.clear();
#if 0
    std::cout << "Add surface pts: " << _cluster->numSurfacePoints() << std::endl;
    for (unsigned long i = 0; i < _cluster->numSurfacePoints(); i++)
    {
        std::cout << " * " << i << ": " << _cluster->surfacePoint(i) << std::endl;
        allClusterPts.push_back(_cluster->surfacePoint(i));
    }

    std::cout << "Dump allClusterPts: " << allClusterPts.size() << std::endl;
    for (unsigned long k = 0; k < allClusterPts.size(); k++)
        std::cout << " * " << allClusterPts.at(k) << std::endl;


    LGCOBBUtils::getCovarianceOfVertices(covarianceMatrix, allClusterPts);

    std::cout << "covariance matrix: " << covarianceMatrix << std::endl;

    Vector3 eigenValues;
    Matrix3 eigenVectors;
    LGCOBBUtils::eigenValuesAndVectors(eigenVectors, eigenValues, covarianceMatrix);

    _eigenVectors[0] = eigenVectors.col(0);
    _eigenVectors[1] = eigenVectors.col(1);
    _eigenVectors[2] = eigenVectors.col(2);

    std::cout << "eigenvalues      : " << eigenValues << std::endl;
    std::cout << "eigenVectors     : " << eigenVectors << std::endl;

    for (int i = 0; i < 3; i++)
    {
        Plane3D<Real> pPlane(_eigenVectors[i], _centroid);
        Real maxDist1 = 0.0f, maxDist2 = 0.0f;

        std::cout << "Plane " << i << ": " << pPlane << std::endl;
        _eigenPlanes[i] = new Plane3Drawable<Real>(pPlane, _cluster);

        std::cout << " initial max. distances: " << maxDist1 << " / " << maxDist2 << std::endl;

        Real curDist = 0.0f, absCurDist = 0.0f;
        long maxDistPt1 = -1, maxDistPt2 = -1;

        for (unsigned long j = 0; j < allClusterPts.size(); j++)
        {
            Vector3 curPt = allClusterPts[j];
            curDist = pPlane.distanceToPoint(curPt);
            absCurDist = std::fabs(curDist);
            if (pPlane.pointOnWhichSide(curPt) == Plane::TOWARDS_NORMAL)
            {
                std::cout << "  point " << j << " = " << curPt << " = TOWARDS_NORMAL; dist. = " << curDist << " abs. dist. = " << absCurDist << std::endl;
                if (absCurDist > maxDist1)
                {
                    maxDistPt1 = j;
                    maxDist1 = absCurDist;
                    std::cout << "  new max. dist TOWARDS_NORMAL for point " << j << " = " << maxDist1 << std::endl;
                }
            }
            else
            {
                std::cout << "  point " << j << " = " << curPt <<  " = AWAY_FROM_NORMAL; dist. = " << curDist << " abs. dist. = " << absCurDist << std::endl;
                if (absCurDist > maxDist2)
                {
                    maxDistPt2 = j;
                    maxDist2 = absCurDist;
                    std::cout << "  new max. dist AWAY_FROM_NORMAL for point " << j << " = " << maxDist2 << std::endl;
                }
            }
        }

        if (maxDistPt1 >= 0)
        {
            std::cout << " * Plane " << i * 2 << ": maxDist at " << maxDistPt1 << ", maxDistPt = " << allClusterPts.at(maxDistPt1) << std::endl;
            if (_obbPlanes[i*2] == NULL)
            {
                _obbPlanes[i*2] = new Plane3D<Real>(_eigenVectors[i], allClusterPts.at(maxDistPt1), _cluster);
                _obbPlaneDrawables[i*2] = new Plane3Drawable<Real>(*(_obbPlanes[i*2]), _cluster);
                _obbPlaneDrawables[i*2]->setScale(Vector3(2.0f, 2.0f, 2.0f));
                std::cout << "    Plane        : " << *(_obbPlanes[i*2]) << std::endl;
                _mostDistPts[i * 2] = allClusterPts.at(maxDistPt1);
                // std::cout << "  erasing index " << maxDistPt1 << " from search list." << std::endl;
            }
        }
        else
        {
            std::cout << "No further-away point found for plane " << i * 2 << "; using maxDist1 = " << maxDist1 << ", corresponding to point 0: " << allClusterPts.at(0) << std::endl;
            if (_obbPlanes[i*2] == NULL)
            {
                _obbPlanes[i*2] = new Plane3D<Real>(_eigenVectors[i], allClusterPts.at(0), _cluster);
                _obbPlaneDrawables[i*2] = new Plane3Drawable<Real>(*(_obbPlanes[i*2]), _cluster);
                _obbPlaneDrawables[i*2]->setScale(Vector3(2.0f, 2.0f, 2.0f));
                std::cout << "    Plane        : " << *(_obbPlanes[i*2]) << std::endl;
                _mostDistPts[i * 2] = allClusterPts.at(0);
            }
        }

        if (maxDistPt2 >= 0)
        {
            std::cout << " * Plane " << 2*i+1 << ": maxDist at " << maxDistPt2 << ", maxDistPt = " << allClusterPts.at(maxDistPt2) << std::endl;
            if (_obbPlanes[2*i+1] == NULL)
            {
                _obbPlanes[2*i+1] = new Plane3D<Real>(-_eigenVectors[i], allClusterPts.at(maxDistPt2), _cluster);
                _obbPlaneDrawables[2*i+1] = new Plane3Drawable<Real>(*(_obbPlanes[2*i+1]), _cluster);
                _obbPlaneDrawables[2*i+1]->setScale(Vector3(2.0f, 2.0f, 2.0f));
                std::cout << "    Plane        : " << *(_obbPlanes[2*i+1]) << std::endl;
                _mostDistPts[2 * i + 1] = allClusterPts.at(maxDistPt2);
                // std::cout << "  erasing index " << maxDistPt2 << " from search list." << std::endl;
            }
        }
        else
        {
            std::cout << "No further-away point found for plane " << 2 * i + 1 << "; using maxDist2 = " << maxDist2 << ", corresponding to point 0: " << allClusterPts.at(0) << std::endl;
            if (_obbPlanes[2*i+1] == NULL)
            {
                _obbPlanes[2*i+1] = new Plane3D<Real>(-_eigenVectors[i], allClusterPts.at(0), _cluster);
                _obbPlaneDrawables[2*i+1] = new Plane3Drawable<Real>(*(_obbPlanes[2*i+1]), _cluster);
                _obbPlaneDrawables[2*i+1]->setScale(Vector3(2.0f, 2.0f, 2.0f));
                std::cout << "    Plane        : " << *(_obbPlanes[2*i+1]) << std::endl;
                _mostDistPts[2 * i + 1] = allClusterPts.at(0);
            }
        }
    }

    bool allPlanesFound = true;
    for (unsigned int p = 0; p < 6; p++)
    {
        if (_obbPlanes[p] == NULL)
        {
            std::cout << "WARNING: Could not safely fit cluster plane " << p << "!" << std::endl;
            allPlanesFound = false;
        }
    }

    if (allPlanesFound)
    {
        typename Plane3D<Real>::IntersectionType ist1, ist2;
        ist1 = _obbPlanes[0]->intersect(*(_obbPlanes[2]), *(_obbPlanes[4]), _obbCorners[0]);
        ist1 = _obbPlanes[0]->intersect(*(_obbPlanes[3]), *(_obbPlanes[4]), _obbCorners[1]);
        ist1 = _obbPlanes[0]->intersect(*(_obbPlanes[3]), *(_obbPlanes[5]), _obbCorners[2]);
        ist1 = _obbPlanes[0]->intersect(*(_obbPlanes[2]), *(_obbPlanes[5]), _obbCorners[3]);

        ist2 = _obbPlanes[1]->intersect(*(_obbPlanes[3]), *(_obbPlanes[5]), _obbCorners[4]);
        ist2 = _obbPlanes[1]->intersect(*(_obbPlanes[2]), *(_obbPlanes[5]), _obbCorners[5]);
        ist2 = _obbPlanes[1]->intersect(*(_obbPlanes[2]), *(_obbPlanes[4]), _obbCorners[6]);
        ist2 = _obbPlanes[1]->intersect(*(_obbPlanes[3]), *(_obbPlanes[4]), _obbCorners[7]);

        Vec<3,Real> obbCenter = _obbCorners[0] + ((_obbCorners[4] - _obbCorners[0]) * 0.5f);
        Vec<3,Real> obbExtents;
        obbExtents[0] = std::fabs(_obbCorners[4][0] - obbCenter[0]);
        obbExtents[1] = std::fabs(_obbCorners[4][1] - obbCenter[1]);
        obbExtents[2] = std::fabs(_obbCorners[4][2] - obbCenter[2]);

        std::cout << " cluster " << _cluster->getName() << " OBB center : " << obbCenter << std::endl;
        std::cout << " cluster " << _cluster->getName() << " OBB extents: " << obbExtents << std::endl;

        std::cout << " cluster " << _cluster->getName() << " OBB corners:" << std::endl;
        for (unsigned int k = 0; k < 8; k++)
            std::cout << " * " << k << ": " << _obbCorners[k] << std::endl;

        std::cout << " create _clusterObb;";
        if (_cluster->collisionModel() == NULL)
            std::cout << " _cluster's collisionModel == NULL!" << std::endl;
        else
            std::cout << " _cluster's collisionModel == " << _cluster->collisionModel()->getName() << " of type " << _cluster->collisionModel()->getTypeName() << std::endl;

        if (_cluster->_parentCluster == NULL)
        {
            _clusterObb = new LGCObbNode<LGCDataTypes>(obbCenter, _eigenVectors[0], _eigenVectors[1], _eigenVectors[2], obbExtents, 0, NULL, _cluster->collisionModel());
            _clusterObb->setReferenceFrame(ABSOLUTE_REFERENCE_FRAME);
        }
        else
        {
            if (_cluster->_parentCluster->clusterObb() == NULL)
                std::cout << " create ObbLeaf: parent OBB = NULL!" << std::endl;
            else
                std::cout << " create ObbLeaf: parent OBB = " << _cluster->_parentCluster->clusterObb()->identifier() << std::endl;

            _clusterObb = new LGCObbLeaf<LGCDataTypes>(obbCenter, _eigenVectors[0], _eigenVectors[1], _eigenVectors[2], obbExtents, 1, _cluster->_parentCluster->clusterObb(), _cluster->collisionModel());
        }

        _clusterObb->setObrCorner(0, _obbCorners[0]);
        _clusterObb->setObrCorner(1, _obbCorners[2]);
        _clusterObb->setObrCorner(2, _obbCorners[4]);
        _clusterObb->setObrCorner(3, _obbCorners[6]);
        _clusterObb->setObrCorner(4, _obbCorners[1]);
        _clusterObb->setObrCorner(5, _obbCorners[3]);
        _clusterObb->setObrCorner(6, _obbCorners[5]);
        _clusterObb->setObrCorner(7, _obbCorners[7]);

        std::stringstream idStr;
        idStr << _cluster->getName() << " cluster OBB";
        _clusterObb->setIdentifier(idStr.str());
        _clusterObb->intersectionHistory().setName(idStr.str() + " intersection history");

        Vec4f historyColor = _cluster->_clusterColor;
        historyColor.w() = historyColor.w() * 0.2f;
        _clusterObb->intersectionHistory().setColor(historyColor);

        _clusterObbDrawable = new ObbDrawable<LGCDataTypes>(_clusterObb, _cluster->d->_clusterObbDrawable, ObbDrawable<LGCDataTypes>::ROTATE_THEN_TRANSLATE);
    }
#endif
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::fitTopOBB()
{
    std::cout << "LGCPointClusterPrivate<LGCDataTypes>::fitTopOBB(): " << _cluster->getName() << std::endl;
    std::cout << " cluster position = " << _cluster->position() << ", orientation = " << _cluster->orientation() << std::endl;
    const LGCCollisionModel<LGCDataTypes>* cm = (LGCCollisionModel<LGCDataTypes> *)(_cluster->collisionModel());
    std::cout << " model position = " << cm->position() << ", orientation = " << cm->orientation() << std::endl;

    std::vector<Vec<3, Real> > allOBBPts;
    for (std::deque<Vector3>::const_iterator it = _cluster->_vertices.begin(); it != _cluster->_vertices.end(); it++)
    {
        Vec<3, Real> tp = _cluster->orientation().inverseRotate(*it);
        tp -= _cluster->position();
        allOBBPts.push_back(tp);
    }
#if 0
    for (std::deque<Vector3>::const_iterator it = _cluster->_edgePoints.begin(); it != _cluster->_edgePoints.end(); it++)
    {
        Vec<3, Real> tp = _cluster->orientation().inverseRotate(*it);
        tp -= _cluster->position();
        allOBBPts.push_back(tp);
    }
#endif

#if 1
    unsigned int i,j;
    // compute covariance matrix
    Vec<3, Real> centroid;
    Mat<3,3,Real> C;
    sofa::helper::ComputeCovarianceMatrix(C, centroid, allOBBPts);

    // get basis vectors
    Vec<3, Real> basis[3];
    sofa::helper::GetRealSymmetricEigenvectors(basis[0], basis[1], basis[2], C);

    Vector3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    Vector3 max(FLT_MIN, FLT_MIN, FLT_MIN);

    // compute min, max projections of box on axes
    for ( i = 0; i < allOBBPts.size(); ++i )
    {
        Vector3 diff = allOBBPts[i] - centroid;
        for (j = 0; j < 3; ++j)
        {
            double length = diff * basis[j];
            if (length > max[j])
            {
                max[j] = length;
            }
            else if (length < min[j])
            {
                min[j] = length;
            }
        }
    }

    // compute center, extents
    Vec<3, Real> obbPosition = centroid;
    Vec<3, Real> obbExtents;
    for ( i = 0; i < 3; ++i )
    {
        obbPosition += basis[i] * 0.5f * (min[i]+max[i]);
        obbExtents[i] = (max[i]-min[i]) * 0.5f;
    }

    _obbAxes.identity();
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            _obbAxes(i,j) = basis[i][j];

    _obbCenter = obbPosition;
    _obbExtents = obbExtents;

#else

    for (int i = 0; i < 6; i++)
    {
        if (_obbPlanes[i])
        {
            delete _obbPlanes[i];
            _obbPlanes[i] = NULL;
        }
        if (_obbPlaneDrawables[i])
        {
            delete _obbPlaneDrawables[i];
            _obbPlaneDrawables[i] = NULL;
        }
    }
    for (int i = 0; i < 3; i++)
    {
        Plane3D<Real> pPlane(_eigenVectors[i], _centroid);
        Real maxDist1 = 0.0f, maxDist2 = 0.0f;

        Real curDist = 0.0f, absCurDist = 0.0f;
        long maxDistPt1 = -1, maxDistPt2 = -1;

        for (unsigned long j = 0; j < allOBBPts.size(); j++)
        {
            Vector3 curPt = allOBBPts[j];
            curDist = pPlane.distanceToPoint(curPt);
            absCurDist = std::fabs(curDist);
            if (pPlane.pointOnWhichSide(curPt) == Plane::TOWARDS_NORMAL)
            {
                std::cout << "  point " << j << " = " << curPt << " = TOWARDS_NORMAL; dist. = " << curDist << " abs. dist. = " << absCurDist << std::endl;
                if (absCurDist > maxDist1)
                {
                    maxDistPt1 = j;
                    maxDist1 = absCurDist;
                    std::cout << "  new max. dist TOWARDS_NORMAL for point " << j << " = " << maxDist1 << std::endl;
                }
            }
            else
            {
                std::cout << "  point " << j << " = " << curPt <<  " = AWAY_FROM_NORMAL; dist. = " << curDist << " abs. dist. = " << absCurDist << std::endl;
                if (absCurDist > maxDist2)
                {
                    maxDistPt2 = j;
                    maxDist2 = absCurDist;
                    std::cout << "  new max. dist AWAY_FROM_NORMAL for point " << j << " = " << maxDist2 << std::endl;
                }
            }
        }

        if (maxDistPt1 >= 0)
        {
            std::cout << " * Plane " << i * 2 << ": maxDist at " << maxDistPt1 << ", maxDistPt = " << allOBBPts.at(maxDistPt1) << std::endl;
            if (_obbPlanes[i*2] == NULL)
            {
                _obbPlanes[i*2] = new Plane3D<Real>(_eigenVectors[i], allOBBPts.at(maxDistPt1), _cluster);
                _obbPlaneDrawables[i*2] = new Plane3Drawable<Real>(*(_obbPlanes[i*2]), _cluster);
                _obbPlaneDrawables[i*2]->setScale(Vector3(2.0f, 2.0f, 2.0f));
                std::cout << "    Plane        : " << *(_obbPlanes[i*2]) << std::endl;
                _mostDistPts[i * 2] = allOBBPts.at(maxDistPt1);
                // std::cout << "  erasing index " << maxDistPt1 << " from search list." << std::endl;
            }
        }
        else
        {
            std::cout << "No further-away point found for plane " << i * 2 << "; using maxDist1 = " << maxDist1 << ", corresponding to point 0: " << allOBBPts.at(0) << std::endl;
            if (_obbPlanes[i*2] == NULL)
            {
                _obbPlanes[i*2] = new Plane3D<Real>(_eigenVectors[i], allOBBPts.at(0), _cluster);
                _obbPlaneDrawables[i*2] = new Plane3Drawable<Real>(*(_obbPlanes[i*2]), _cluster);
                _obbPlaneDrawables[i*2]->setScale(Vector3(2.0f, 2.0f, 2.0f));
                std::cout << "    Plane        : " << *(_obbPlanes[i*2]) << std::endl;
                _mostDistPts[i * 2] = allOBBPts.at(0);
            }
        }

        if (maxDistPt2 >= 0)
        {
            std::cout << " * Plane " << 2*i+1 << ": maxDist at " << maxDistPt2 << ", maxDistPt = " << allOBBPts.at(maxDistPt2) << std::endl;
            if (_obbPlanes[2*i+1] == NULL)
            {
                _obbPlanes[2*i+1] = new Plane3D<Real>(-_eigenVectors[i], allOBBPts.at(maxDistPt2), _cluster);
                _obbPlaneDrawables[2*i+1] = new Plane3Drawable<Real>(*(_obbPlanes[2*i+1]), _cluster);
                _obbPlaneDrawables[2*i+1]->setScale(Vector3(2.0f, 2.0f, 2.0f));
                std::cout << "    Plane        : " << *(_obbPlanes[2*i+1]) << std::endl;
                _mostDistPts[2 * i + 1] = allOBBPts.at(maxDistPt2);
                // std::cout << "  erasing index " << maxDistPt2 << " from search list." << std::endl;
            }
        }
        else
        {
            std::cout << "No further-away point found for plane " << 2 * i + 1 << "; using maxDist2 = " << maxDist2 << ", corresponding to point 0: " << allOBBPts.at(0) << std::endl;
            if (_obbPlanes[2*i+1] == NULL)
            {
                _obbPlanes[2*i+1] = new Plane3D<Real>(-_eigenVectors[i], allOBBPts.at(0), _cluster);
                _obbPlaneDrawables[2*i+1] = new Plane3Drawable<Real>(*(_obbPlanes[2*i+1]), _cluster);
                _obbPlaneDrawables[2*i+1]->setScale(Vector3(2.0f, 2.0f, 2.0f));
                std::cout << "    Plane        : " << *(_obbPlanes[2*i+1]) << std::endl;
                _mostDistPts[2 * i + 1] = allOBBPts.at(0);
            }
        }
    }

    typename Plane3D<Real>::IntersectionType ist1, ist2;
    ist1 = _obbPlanes[0]->intersect(*(_obbPlanes[2]), *(_obbPlanes[4]), _obbCorners[0]);
    ist2 = _obbPlanes[1]->intersect(*(_obbPlanes[3]), *(_obbPlanes[5]), _obbCorners[4]);

    ist1 = _obbPlanes[0]->intersect(*(_obbPlanes[3]), *(_obbPlanes[4]), _obbCorners[1]);
    ist1 = _obbPlanes[0]->intersect(*(_obbPlanes[2]), *(_obbPlanes[5]), _obbCorners[3]);
    ist1 = _obbPlanes[0]->intersect(*(_obbPlanes[3]), *(_obbPlanes[5]), _obbCorners[2]);

    ist2 = _obbPlanes[1]->intersect(*(_obbPlanes[2]), *(_obbPlanes[5]), _obbCorners[5]);
    ist2 = _obbPlanes[1]->intersect(*(_obbPlanes[2]), *(_obbPlanes[4]), _obbCorners[6]);
    ist2 = _obbPlanes[1]->intersect(*(_obbPlanes[3]), *(_obbPlanes[4]), _obbCorners[7]);

    Vec<3,Real> obbCenter = _obbCorners[0] + ((_obbCorners[4] - _obbCorners[0]) * 0.5f);
    Vec<3,Real> obbExtents = _obbCorners[4] - obbCenter;

    std::cout << " cluster " << _cluster->getName() << " OBB center : " << obbCenter << std::endl;
    std::cout << " cluster " << _cluster->getName() << " OBB extents: " << obbExtents << std::endl;

    if (_clusterObb)
        delete _clusterObb;

    _clusterObb = new LGCObbNode<LGCDataTypes>(obbCenter, _eigenVectors[0], _eigenVectors[1], _eigenVectors[2], obbExtents, 0, NULL, _cluster->collisionModel());

    std::cout << " cluster " << _cluster->getName() << " OBB corners:" << std::endl;
    for (unsigned int k = 0; k < 8; k++)
    {
        std::cout << " * " << k << ": " << _obbCorners[k] << std::endl;
        _clusterObb->setObrCorner(k, _obbCorners[k]);
    }
    _clusterObbDrawable = new ObbDrawable<LGCDataTypes>(_clusterObb);
#endif
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::computeClusterCentroid()
{
    std::cout << "LGCPointClusterPrivate<LGCDataTypes>::computeClusterCentroid(" << _cluster->getName() << ")";

    typedef sofa::component::collision::EPEC_Kernel Kernel;
    std::list<CGAL::Point_3<Kernel> > points;

    for (std::deque<Vector3>::const_iterator it = _cluster->_edgePoints.begin(); it != _cluster->_edgePoints.end(); it++)
    {
        points.push_back(CGAL::Point_3<Kernel>((*it).x(),(*it).y(),(*it).z()));
    }

    for (std::deque<Vector3>::const_iterator it = _cluster->_clusterPoints.begin(); it != _cluster->_clusterPoints.end(); it++)
    {
        points.push_back(CGAL::Point_3<Kernel>((*it).x(),(*it).y(),(*it).z()));
    }

    for (std::deque<Vector3>::const_iterator it = _cluster->_vertices.begin(); it != _cluster->_vertices.end(); it++)
    {
        points.push_back(CGAL::Point_3<Kernel>((*it).x(),(*it).y(),(*it).z()));
    }

    CGAL::Point_3<Kernel> centroid = CGAL::centroid(points.begin(), points.end(), CGAL::Dimension_tag<0>());
    _centroid = Vector3(CGAL::to_double(centroid.x()),
                           CGAL::to_double(centroid.y()),
                           CGAL::to_double(centroid.z()));

    std::cout << " centroid computed: " << _centroid << std::endl;
}

template <class LGCDataTypes>
LGCObb<LGCDataTypes>* LGCPointCluster<LGCDataTypes>::clusterObb() const
{
    if (_parentCluster == NULL)
        return d->_pcObb;
    else
        return d->_clusterObb;
}

template <class LGCDataTypes>
const Vec<3, typename LGCDataTypes::Real>& LGCPointCluster<LGCDataTypes>::clusterCentroid() const
{
    return d->_centroid;
}

template <class LGCDataTypes>
bool LGCPointCluster<LGCDataTypes>::testIntersection(const LGCIntersectable<typename LGCDataTypes::Real>& is2)
{
#if 0
    LGCObb<LGCDataTypes>* obb1 = d->_clusterObb;
    if (!obb1)
        obb1 = d->_pcObb;

    if (!obb1)
    {
        std::cerr << "No valid OBB pointer for cluster " << this->getName() <<  " set!!!" << std::endl;
        return false;
    }
    std::cout << " LGCPointCluster(" << getName() << ") with OBB " << *obb1 << "::testIntersection(" << is2 << "), is-a ";

    const LGCObb<LGCDataTypes>* obb2 = NULL;
    Vec<3, Real> obb2Center, obb2Extents;
    Vec<3, Real> obb2Axes[3];
    if (is2.intersectableType() == LGC_CUBE)
    {
        const LGCCube<Real>& cube2 = dynamic_cast<const LGCCube<Real>& >(is2);
        std::cout << "LGCCube: " << cube2 << std::endl;

        obb2Center = cube2.cube().minVect() + ((cube2.cube().maxVect() - cube2.cube().minVect()) * 0.5f);
        obb2Extents = cube2.cube().maxVect() - obb2Center;

        obb2Axes[0] = Vec<3,Real>(1,0,0);
        obb2Axes[1] = Vec<3,Real>(0,1,0);
        obb2Axes[2] = Vec<3,Real>(0,0,1);
    }
    else if (is2.intersectableType() == LGC_OBB ||
             is2.intersectableType() == LGC_POINTCLUSTER)
    {
        if (is2.intersectableType() == LGC_OBB)
        {
            obb2 = dynamic_cast<const LGCObb<LGCDataTypes>* >(&is2);
            std::cout << "LGCObb: " << *obb2 << std::endl;
        }
        else if (is2.intersectableType() == LGC_POINTCLUSTER)
        {
            const LGCPointCluster<LGCDataTypes>* cluster = dynamic_cast<const LGCPointCluster<LGCDataTypes>* >(&is2);
            obb2 = cluster->clusterObb();
            std::cout << "LGCPointCluster with OBB: " << *obb2 << std::endl;
        }

        if (obb2 == NULL)
        {
            std::cout << " No valid OBB obtained!" << std::endl;
            return false;
        }

        obb2Center = obb2->lgcCollisionModel()->position() /*obb2->center()*/;
        obb2Extents = obb2->halfExtents();
        for (unsigned int k = 0; k < 3; k++)
            obb2Axes[k] = obb2->localAxis(k);
    }
    else
    {
        std::cout << " unsupported intersectable for LGCPointCluster!" << std::endl;
        return false;
    }


    Real ra, rb;
    Mat<3,3,Real> R, AbsR;

    for (unsigned int i = 0; i < 3; i++)
        for (unsigned int j = 0; j < 3; j++)
            R[i][j] = obb1->localAxis(i) * obb2Axes[j];

    Vec<3, Real> tt = obb2Center - obb1->lgcCollisionModel()->position() /*obb1->center()*/;
    Vec<3, Real> t(tt * obb1->localAxis(0), tt * obb1->localAxis(1), tt * obb1->localAxis(2));

    for (unsigned int i = 0; i < 3; i++)
        for (unsigned int j = 0; j < 3; j++)
            AbsR[i][j] = std::fabs(R[i][j]) + sofa::helper::kEpsilon;

    for (unsigned int i = 0; i < 3; i++)
    {
        ra = obb1->halfExtents()[i];
        rb = obb2Extents[0] * AbsR[i][0] + obb2Extents[1] * AbsR[i][1] + obb2Extents[2] * AbsR[i][2];

        if (std::fabs(t[i]) > ra + rb)
        {
            std::cout << " state: no intersection, axis test A" << i << std::endl;
            std::cout << ra << " + " << rb << " = " << ra + rb << std::endl;

            return false;
        }
    }

    for (unsigned int i = 0; i < 3; i++)
    {
        ra = obb1->halfExtents()[0] * AbsR[0][i] + obb1->halfExtents()[1] * AbsR[1][i] + obb1->halfExtents()[2] * AbsR[2][i];
        rb = obb2Extents[i];

        if (std::fabs(t[0] * R[0][i] + t[1] * R[1][i] + t[2] * R[2][i]) > ra + rb)
        {
            std::cout << " state: no intersection, axis test B" << i << std::endl;

            std::cout << ra << " + " << rb << " = " << ra + rb << std::endl;
            return false;
        }
    }

    std::cout << " half extents -- obb1: " << obb1->halfExtents()[0] << ", " << obb1->halfExtents()[1] << ", " << obb1->halfExtents()[2] << "/ obb2: " <<
                 obb2Extents[0] << ", " << obb2Extents[1] << ", " << obb2Extents[2];

    ra = obb1->halfExtents()[1] * AbsR[2][0] + obb1->halfExtents()[2] * AbsR[1][0];
    rb = obb2Extents[1] * AbsR[0][2] + obb2Extents[2] * AbsR[2][0];
    if (std::fabs(t[2] * R[1][0] - t[1] * R[2][0]) > ra + rb)
    {
        std::cout << " state: no intersection, axis test A0xB0" <<std::endl;
        return false;
    }

    ra = obb1->halfExtents()[1] * AbsR[2][1] + obb1->halfExtents()[2] * AbsR[1][1];
    rb = obb2Extents[0] * AbsR[0][2] + obb2Extents[2] * AbsR[0][0];
    if (std::fabs(t[2] * R[1][1] - t[1] * R[2][1]) > ra + rb)
    {
        std::cout << " state: no intersection, axis test A0xB1" <<std::endl;
        return false;
    }

    ra = obb1->halfExtents()[1] * AbsR[2][2] + obb1->halfExtents()[2] * AbsR[1][2];
    rb = obb2Extents[0] * AbsR[0][1] + obb2Extents[1] * AbsR[0][0];
    if (std::fabs(t[2] * R[1][2] - t[1] * R[2][2]) > ra + rb)
    {
        std::cout << " state: no intersection, axis test A0xB2" <<std::endl;
        return false;
    }

    ra = obb1->halfExtents()[0] * AbsR[2][0] + obb1->halfExtents()[2] * AbsR[0][0];
    rb = obb2Extents[1] * AbsR[1][2] + obb2Extents[2] * AbsR[1][1];
    if (std::fabs(t[0] * R[2][0] - t[2] * R[0][0]) > ra + rb)
    {
        std::cout << " state: no intersection, axis test A1xB0" <<std::endl;
        return false;
    }

    ra = obb1->halfExtents()[0] * AbsR[2][1] + obb1->halfExtents()[2] * AbsR[0][1];
    rb = obb2Extents[0] * AbsR[1][2] + obb2Extents[2] * AbsR[1][0];
    if (std::fabs(t[0] * R[2][1] - t[2] * R[0][1]) > ra + rb)
    {
        std::cout << " state: no intersection, axis test A1xB1" <<std::endl;
        return false;
    }

    ra = obb1->halfExtents()[0] * AbsR[2][2] + obb1->halfExtents()[2] * AbsR[0][2];
    rb = obb2Extents[0] * AbsR[1][1] + obb2Extents[1] * AbsR[1][0];
    if (std::fabs(t[0] * R[2][2] - t[2] * R[0][2]) > ra + rb)
    {
        std::cout << " state: no intersection, axis test A1xB2" <<std::endl;
        return false;
    }

    ra = obb1->halfExtents()[0] * AbsR[1][0] + obb1->halfExtents()[1] * AbsR[0][0];
    rb = obb2Extents[1] * AbsR[2][2] + obb2Extents[2] * AbsR[2][1];
    if (std::fabs(t[1] * R[0][0] - t[0] * R[1][0]) > ra + rb)
    {
        std::cout << " state: no intersection, axis test A2xB0" <<std::endl;
        return false;
    }

    ra = obb1->halfExtents()[0] * AbsR[1][1] + obb1->halfExtents()[1] * AbsR[0][1];
    rb = obb2Extents[0] * AbsR[2][2] + obb2Extents[2] * AbsR[2][0];
    if (std::fabs(t[1] * R[0][1] - t[0] * R[1][1]) > ra + rb)
    {
        std::cout << " state: no intersection, axis test A2xB1" <<std::endl;
        return false;
    }

    ra = obb1->halfExtents()[0] * AbsR[1][2] + obb1->halfExtents()[1] * AbsR[0][2];
    rb = obb2Extents[0] * AbsR[2][1] + obb2Extents[1] * AbsR[2][0];
    if (std::fabs(t[1] * R[0][2] - t[0] * R[1][2]) > ra + rb)
    {
        std::cout << " state: no intersection, axis test A2xB2" <<std::endl;
        return false;
    }


    std::cout << " state: intersects" << std::endl;
#else

    std::cout << "LGCPointCluster::testIntersection(" << this->getName() << " -- " << is2.collisionModel()->getName() << std::endl;
    const LGCPointCluster<LGCDataTypes>* cluster2 = NULL;
    if (is2.intersectableType() == LGC_POINTCLUSTER)
    {
         cluster2 = dynamic_cast<const LGCPointCluster<LGCDataTypes>* >(&is2);
    }

    std::cout << " cluster2 = " << cluster2 << std::endl;
    if (cluster2 != NULL)
    {
        PQP_FlatModel* pqpModel1 = this->pqpModel();
        PQP_FlatModel* pqpModel2 = cluster2->pqpModel();

        std::cout << " pqpModel1 = " << pqpModel1 << ", pqpModel2 = " << pqpModel2 << std::endl;
        if (pqpModel1 && pqpModel2)
        {
            PQP_REAL R1[3][3], R2[3][3], T1[3], T2[3];
            Matrix3 rot1, rot2;

            this->lgcCollisionModel()->orientation().toMatrix(rot1);
            cluster2->lgcCollisionModel()->orientation().toMatrix(rot2);

            std::cout << " model1 pos = " << this->lgcCollisionModel()->position() << ", model2 pos = " << cluster2->lgcCollisionModel()->position() << std::endl;
            std::cout << " model1 ori = " << rot1 << ", model2 ori = " << rot2 << std::endl;

            T1[0] = this->lgcCollisionModel()->position().x(); T1[1] = this->lgcCollisionModel()->position().y(); T1[2] = this->lgcCollisionModel()->position().z();
            T2[0] = cluster2->lgcCollisionModel()->position().x(); T2[1] = cluster2->lgcCollisionModel()->position().y(); T2[2] = cluster2->lgcCollisionModel()->position().z();

            for (unsigned short i = 0; i < 3; i++)
            {
                for (unsigned short j = 0; j < 3; j++)
                {
                    R1[j][i] = rot1(i,j);
                    R2[j][i] = rot2(i,j);
                }
            }

            PQP_CollideResult pqpRes;
            int status = PQP_CollideFlatFlat(&pqpRes, R1, T1, pqpModel1, R2, T2, pqpModel2, PQP_FIRST_CONTACT);

            std::cout << " PQP_CollideFlatFlat status = " << status << "; colliding pairs count =  " << pqpRes.NumPairs() << std::endl;

            if (status == PQP_OK && pqpRes.num_pairs > 0)
            {
                std::cout << " Status: Intersecting." << std::endl;
                return true;
            }
        }
    }
#endif
    std::cout << " Status: NO INTERSECTION." << std::endl;
    return false;
}

template <class LGCDataTypes>
void LGCPointCluster<LGCDataTypes>::buildClusterDrawList()
{
    d->buildClusterDrawList();
    for (unsigned long i = 0; i < _children.size(); i++)
        _children[i]->d->buildClusterDrawList();
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::buildClusterDrawList()
{
    _clusterEdgePoints.resize(_cluster->_edgePoints.size());
    std::copy(_cluster->_edgePoints.begin(), _cluster->_edgePoints.end(), _clusterEdgePoints.begin());

    _clusterVertices.resize(_cluster->_vertices.size());
    std::copy(_cluster->_vertices.begin(), _cluster->_vertices.end(), _clusterVertices.begin());
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::drawCoordinateMarkerGL(float lineLength, float lineWidth, const Vec<4, Real>& xColor, const Vec<4, Real>& yColor, const Vec<4, Real>& zColor)
{
    glLineWidth(lineWidth);
    glBegin(GL_LINES);

    glColor4f(xColor.x(), xColor.y(), xColor.z(), xColor.w());
    glVertex3d(0, 0, 0);
    glVertex3d(lineLength,0,0);

    glColor4f(yColor.x(), yColor.y(), yColor.z(), yColor.w());
    glVertex3d(0, 0, 0);
    glVertex3d(0,lineLength,0);

    glColor4f(zColor.x(), zColor.y(), zColor.z(), zColor.w());
    glVertex3d(0, 0, 0);
    glVertex3d(0,0,lineLength);

    glEnd();
    glLineWidth(1.0f);
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::drawObbVolume(const Vec<3, Real> &halfExtents, const Vec4f &color)
{
    glBegin(GL_LINES);
    glColor4d(color.x(), color.y(), color.z(), color.w());
    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());

    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glEnd();
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::updateOBBHierarchy()
{
    for (unsigned int i = 0; i < _cluster->numChildren(); i++)
        updateOBBHierarchyRec(_cluster->clusterObb());
}

template <class LGCDataTypes>
void LGCPointClusterPrivate<LGCDataTypes>::updateOBBHierarchyRec(LGCObb<LGCDataTypes> *parent)
{
    if (parent->parent() != NULL)
    {
        std::cout << parent->identifier() << ": old centerOffset = " << parent->lcData()._childOffset;
        parent->lcData()._childOffset = parent->position() - parent->parent()->position();
        std::cout << ", new centerOffset = " << parent->lcData()._childOffset << std::endl;

        parent->lcData()._invParentTransform = parent->parent()->wcData()._localAxes.transposed();
        parent->lcData()._childTransform = parent->wcData()._localAxes;
    }

    if (parent->numOBBChildren() > 0)
    {
        for (unsigned int i = 0; i < parent->numOBBChildren(); i++)
            updateOBBHierarchyRec(parent->child(i));
    }
}

#endif // LGCPOINTCLUSTER_INL
