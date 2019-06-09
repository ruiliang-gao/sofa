#ifndef OBBTREEGPUCOLLISIONDETECTION_H
#define OBBTREEGPUCOLLISIONDETECTION_H

#include "initObbTreeGpuPlugin.h"
#include <SofaBaseCollision/BruteForceDetection.h>

#include "ObbTreeGPUCollisionModel.h"
#include "ObbTreeCPUCollisionModel.h"

#include "ObbTreeGPUIntersection.h"

#include "ObbTreeGPUCollisionDetection_cuda.h"

#define OBBTREE_GPU_STREAMED_COLLISION_QUERIES
#define OBBTREE_GPU_COLLISION_DETECTION_FULL_ALLOCATION_DETECTION
namespace sofa
{
    namespace component
    {
        namespace collision
        {

            class SOFA_OBBTREEGPUPLUGIN_API ObbTreeGPUCollisionDetection: public BruteForceDetection
            {
                public:
                    SOFA_CLASS(ObbTreeGPUCollisionDetection, BruteForceDetection);

                protected:
                    ObbTreeGPUCollisionDetection();
                    ~ObbTreeGPUCollisionDetection();

                    std::map<std::string, ObbTreeGPUCollisionModel<Vec3Types>*> m_obbModels;
                    std::map<std::string, ObbTreeCPUCollisionModel<Vec3Types>*> m_pqpModels;

                    //std::map<std::string, std::string> m_testedModelPairs_AB;
                    //std::map<std::string, std::string> m_testedModelPairs_BA;

                    std::vector<std::pair<std::string, std::string> >  m_testedModelPairs;

                    std::map<std::string, std::vector<std::pair<int,int> > > m_intersectingOBBs;
                    std::map<std::string, std::vector<std::pair<int,int> > > m_intersectingTriangles;

                    std::map<std::string, std::pair<GPUVertex*, GPUVertex*> > m_transformedVertices;

                    OBBTreeGPUDiscreteIntersection* m_intersection;

                    double m_alarmDistance, m_contactDistance;

#ifdef OBBTREE_GPU_STREAMED_COLLISION_QUERIES
                    cudaStream_t m_workerStreams[4];

                    unsigned int m_resultBinSize1, m_resultBinSize2, m_resultBinSize3;

                    gProximityWorkerUnit* m_gProximityWorkerUnits[4];
                    gProximityWorkerResult* m_gProximityWorkerResults_64[4];
                    gProximityWorkerResult* m_gProximityWorkerResults_128[4];
                    gProximityWorkerResult* m_gProximityWorkerResults_256[4];

                    bool m_workerUnitOccupied[4];
                    bool m_workerResultOccupied_64[4];
                    bool m_workerResultOccupied_128[4];
                    bool m_workerResultOccupied_256[4];

                    unsigned int m_numOBBTreePairsTested;


                    Data<bool> _useStreamedCollisionQueries;

                    Data<unsigned int> _numStreamedWorkerUnits;
                    Data<unsigned int> _numStreamedWorkerResultBins;
                    Data<unsigned int> _streamedWorkerResultMinSize;
                    Data<unsigned int> _streamedWorkerResultMaxSize;
                    Data<bool> _useDynamicWorkerScheme;

                    Data<bool> _updateGPUVertexPositions;

                    std::vector<gProximityWorkerUnit*> _streamedWorkerUnits;
                    std::map<unsigned int, std::vector<gProximityWorkerResult*> > _streamedWorkerResults;

                    std::vector<cudaStream_t> _workerStreams;
                    std::vector<cudaEvent_t> _workerEvents;
                    cudaEvent_t _workerStartEvent, _workerEndEvent, _workerBalanceEvent;

                    cudaStream_t _transformStream;
                    cudaStream_t _memoryStream;

                    std::vector<cudaStream_t> _triTestStreams;
                    std::vector<cudaEvent_t> _triTestEvents;
                    cudaEvent_t _triTestStartEvent, _triTestEndEvent;

                    unsigned int _totalResultBinCount;

                    unsigned int* _outputIndices;

                    struct OBBModelContainer
                    {
                        public:
                            OBBContainer _obbContainer;
                            ObbTreeGPUCollisionModel<Vec3Types>* _obbCollisionModel;
                    };

                    std::vector< std::pair<OBBModelContainer,OBBModelContainer> > _narrowPhasePairs;
                    //std::map<std::string,gProximityGPUTransform*> _modelTransforms;
                    std::map<std::string,ObbTreeGPUCollisionModel<Vec3Types>*> _obbTreeGPUModels;

                    gProximityGPUTransform** _gpuTransforms;
                    std::map<std::string, unsigned int> _gpuTransformIndices;
#endif

                    void scheduleBVHTraversals(std::vector<std::pair<OBBModelContainer, OBBModelContainer> > &narrowPhasePairs,
                                               unsigned int iteration, unsigned int numSlots,
                                               std::vector<OBBContainer*> models1, std::vector<OBBContainer*> models2,
                                               std::vector<gProximityGPUTransform*> transforms1, std::vector<gProximityGPUTransform*> transforms2,
                                               std::vector<gProximityWorkerUnit*> workerUnits);
                    void runBVHTraversals();
                    void partitionResultBins();
                    void runTriangleTests();

                public:
                    void init();
                    void reinit();
                    void reset();

                    void bwdInit();

                    void addCollisionModels(const sofa::helper::vector<core::CollisionModel *> collisionModels);
                    void addCollisionModel(core::CollisionModel *cm);
                    void addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair);
                    void addCollisionPairs(const sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >& v);

                    void beginBroadPhase();
                    void endBroadPhase();

                    void beginNarrowPhase();
                    void endNarrowPhase();

                    /* for debugging */
                    void draw(const core::visual::VisualParams* vparams);
            };

            inline bool contactTypeCompare(sofa::core::collision::DetectionOutput i, sofa::core::collision::DetectionOutput j) { return (i.contactType < j.contactType); }
        }
    }
}

#endif // OBBTREEGPUCOLLISIONDETECTION_H
