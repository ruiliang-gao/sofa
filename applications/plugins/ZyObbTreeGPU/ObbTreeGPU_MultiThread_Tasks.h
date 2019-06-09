#ifndef OBBTREEGPU_MULTITHREAD_TASKS_H
#define OBBTREEGPU_MULTITHREAD_TASKS_H

#include "Tasks.h"

#include <vector_functions.h>
#include "ObbTreeGPU_CudaDataStructures.h"

#include <WorkerThreadIface.h>

#include <sofa/defaulttype/Vec3Types.h>

using namespace sofa::defaulttype;

namespace sofa
{
    namespace component
    {
        namespace collision
        {
			template<class VecTypes> class ObbTreeGPUCollisionModel;

            class NarrowPhaseGPUTask: public Zyklio::MultiThreading::PoolTask
            {
                public:
                    virtual bool run(Zyklio::MultiThreading::WorkerThreadIface*);

                    NarrowPhaseGPUTask() { m_localCudaEvents = false; }
                    NarrowPhaseGPUTask(const Zyklio::MultiThreading::TaskStatus* status, unsigned int numStreamedWorkerResults = 4, unsigned int minResultSize = 4096, unsigned int maxResultSize = 32768);
                    ~NarrowPhaseGPUTask();

                    void setAlarmDistance(const double& dist) { m_alarmDistance = dist; }
                    void setContactDistance(const double& dist) { m_contactDistance = dist; }

                    void setupResultBins(unsigned int minSize, unsigned int maxSize);

					void addTriangleCheck(gProximityWorkerUnit* workerUnit, OBBContainer* container1, OBBContainer* container2, ObbTreeGPUCollisionModel<Vec3Types>* model1, ObbTreeGPUCollisionModel<Vec3Types>* model2);
                    void setBVHTraversalResult(unsigned int, unsigned int);

                    void clearWorkList();

                    float getElapsedTime() const { return m_elapsedTime; }
                    const std::vector<float>& getElapsedTimePerTest() const { return m_elapsedTimePerTest; }

                    std::map<unsigned int, std::vector<int> >& getContactIDs() { return m_contactIDs; }
                    std::map<unsigned int, std::vector<gProximityContactType> >& getContactTypes() { return m_contactTypes; }
                    std::map<unsigned int, std::vector<double> >& getContactDistances() { return m_contactDistances; }
                    std::map<unsigned int, std::vector<Vector3> >& getContactPoints_0() { return m_contactPoints_0; }
                    std::map<unsigned int, std::vector<Vector3> >& getContactPoints_1() { return m_contactPoints_1; }
                    std::map<unsigned int, std::vector<Vector3> >& getContactNormals() { return m_contactNormals; }
                    std::map<unsigned int, std::vector<std::pair<int, int> > >& getContactElements() { return m_contactElements; }
                    std::map<unsigned int, std::vector<std::pair<int, int> > >& getContactElementsFeatures() { return m_contactElementsFeatures; }

					ObbTreeGPUCollisionModel<Vec3Types>* getModel1(unsigned int);
					ObbTreeGPUCollisionModel<Vec3Types>* getModel2(unsigned int);

                    const std::string getModelName1(unsigned int) const;
                    const std::string getModelName2(unsigned int) const;

                    std::vector<int> getTriangleIntersectionResults() {
                        return m_triangleIntersectionResults;
                    }

                    int getResultSize() {
                        return m_totalResultsProduced;
                    }

                protected:
                    void distributeResultBins();

                    std::vector<gProximityWorkerUnit*> m_taskWorkerUnits;
                    std::vector<OBBContainer*> m_containers_1;
                    std::vector<OBBContainer*> m_containers_2;

                    std::vector<std::string> m_modelNames_1;
                    std::vector<std::string> m_modelNames_2;

					std::vector<ObbTreeGPUCollisionModel<Vec3Types>*> m_models_1;
					std::vector<ObbTreeGPUCollisionModel<Vec3Types>*> m_models_2;

                    std::vector<cudaStream_t> m_triTestStreams;
                    cudaStream_t m_memStream;

                    std::vector<cudaEvent_t> m_triTestEvents;
                    cudaEvent_t m_startEvent;
                    cudaEvent_t m_stopEvent;

                    float m_elapsedTime;
                    std::vector<float> m_elapsedTimePerTest;

                    std::map<unsigned int, std::vector<gProximityWorkerResult*> > m_streamedWorkerResults;
                    std::vector<std::pair<unsigned int, unsigned int> > m_resultBinSizes;
                    std::vector<unsigned int> m_resultBinStartIndices;

                    std::map<unsigned int, std::vector<gProximityWorkerResult*> > m_freeResultBins;
                    std::map<unsigned int, std::vector<std::pair<unsigned int, unsigned int> > > m_freeResultBinSizes;
                    std::map<unsigned int, std::vector<unsigned int> > m_freeResultBinStartIndices;

                    std::map<unsigned int, bool> m_satisfiablePairChecks;
                    std::map<unsigned int, std::pair<OBBContainer, OBBContainer> > m_markedPairChecks;

                    bool m_localCudaEvents;

                    std::vector<int> m_intersectingTriPairCount;
                    std::vector<int> m_triangleIntersectionResults;

                    std::map<unsigned int, std::vector<int> > m_contactIDs;
                    std::map<unsigned int, std::vector<gProximityContactType> > m_contactTypes;
                    std::map<unsigned int, std::vector<double> > m_contactDistances;
                    std::map<unsigned int, std::vector<Vector3> > m_contactNormals;
                    std::map<unsigned int, std::vector<Vector3> > m_contactPoints_0;
                    std::map<unsigned int, std::vector<Vector3> > m_contactPoints_1;
                    std::map<unsigned int, std::vector<std::pair<int, int> > > m_contactElements;
                    std::map<unsigned int, std::vector<std::pair<int, int> > > m_contactElementsFeatures;

                    double m_contactDistance, m_alarmDistance;
                    unsigned int m_numWorkerUnits;
                    unsigned int m_totalResultBinCount;
                    unsigned int m_numStreamedWorkerResultBins;
                    unsigned int m_minResultSize;
                    unsigned int m_maxResultSize;

					unsigned int m_totalResultsProduced;
            };

            class BVHTraversalTask : public Zyklio::MultiThreading::PoolTask
            {
                public:
                    virtual bool run(Zyklio::MultiThreading::WorkerThreadIface*);

                    BVHTraversalTask() { m_localCudaEvents = false; }
                    BVHTraversalTask(const Zyklio::MultiThreading::TaskStatus* status, unsigned int numWorkerUnits);
                    BVHTraversalTask(const Zyklio::MultiThreading::TaskStatus* status, unsigned int numSmallWorkerUnits, unsigned long sizeSmallWorkerUnit, unsigned int numLargeWorkerUnits, unsigned long sizeLargeWorkerUnit);

                    BVHTraversalTask(const BVHTraversalTask&);
                    BVHTraversalTask& operator=(const BVHTraversalTask&);

                    ~BVHTraversalTask();

					void addTraversal(OBBContainer*, OBBContainer*, gProximityGPUTransform*, gProximityGPUTransform*, ObbTreeGPUCollisionModel<Vec3Types>*, ObbTreeGPUCollisionModel<Vec3Types>*);
                    int getNumTraversals() const { return m_taskWorkerUnits.size(); }

                    gProximityWorkerUnit* getWorkerUnit(unsigned int);
                    OBBContainer* getContainer1(unsigned int);
                    OBBContainer* getContainer2(unsigned int);

					ObbTreeGPUCollisionModel<Vec3Types>* getModel1(unsigned int);
					ObbTreeGPUCollisionModel<Vec3Types>* getModel2(unsigned int);

                    const std::string getModelName1(unsigned int) const;
                    const std::string getModelName2(unsigned int) const;

                    void setAlarmDistance(const double& dist) { m_alarmDistance = dist; }
                    void setContactDistance(const double& dist) { m_contactDistance = dist; }

                    const std::vector<int>& getResults() const { return m_results; }
                    float getElapsedTime() const { return m_elapsedTime; }

					const std::vector<float>& getElapsedTimePerTest() const { return m_elapsedTimePerTest; }
					const std::vector<std::pair<std::string, int64_t> >& getElapsedTimeCPUStep() const { return m_elapsedTime_CPUStep; }

					ctime_t getElapsedTimeInThread() const { return m_elapsedTimeInStep;  }
                    unsigned int getTraversalCalls() { return m_traversalCalls; }

                    void clearWorkList();

                protected:
                    std::vector<gProximityWorkerUnit*> m_taskWorkerUnits;
                    std::vector<OBBContainer*> m_containers_1;
                    std::vector<OBBContainer*> m_containers_2;
                    std::vector<gProximityGPUTransform*> m_transforms_1;
                    std::vector<gProximityGPUTransform*> m_transforms_2;
                    
					std::vector<ObbTreeGPUCollisionModel<Vec3Types>*> m_models_1;
					std::vector<ObbTreeGPUCollisionModel<Vec3Types>*> m_models_2;
					
					std::vector<std::string> m_modelNames_1;
                    std::vector<std::string> m_modelNames_2;

                    std::vector<cudaStream_t> m_workerStreams;
                    std::vector<cudaEvent_t> m_workerEvents;

					std::vector<cudaEvent_t> m_workerStartEvents;
					std::vector<cudaEvent_t> m_workerStopEvents;

                    cudaStream_t m_memStream;
                    cudaEvent_t m_startEvent;
                    cudaEvent_t m_stopEvent;
                    cudaEvent_t m_balanceEvent;

                    int* m_workQueueCounts;

                    float m_elapsedTime;
					std::vector<float> m_elapsedTimePerTest;
					std::vector<std::pair<std::string, int64_t> > m_elapsedTime_CPUStep;
					
					ctime_t m_elapsedTimeInStep;

                    unsigned int m_traversalCalls;

                    bool m_localCudaEvents;

                    double m_contactDistance, m_alarmDistance;
                    unsigned int m_numWorkerUnits;
					unsigned int m_numSmallWorkerUnits, m_numLargeWorkerUnits;
					unsigned long m_smallWorkerUnitSize, m_largeWorkerUnitSize;

                    unsigned int m_numAssignedTraversals;

                    std::vector<int> m_results;
            };
        }
    }
}

#endif // OBBTREEGPU_MULTITHREAD_TASKS_H
