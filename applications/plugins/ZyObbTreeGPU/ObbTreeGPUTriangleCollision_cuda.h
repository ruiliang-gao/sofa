#ifndef OBBTREEGPU_TRIANGLE_COLLISION_CUDA_H
#define OBBTREEGPU_TRIANGLE_COLLISION_CUDA_H

#include "ObbTreeGPU_CudaDataStructures.h"

void ObbTreeGPU_TriangleIntersection(sofa::component::collision::OBBContainer *model1,
                                      sofa::component::collision::OBBContainer *model2,
                                      gProximityWorkerUnit* workerUnit,
                                      gProximityWorkerResult* workerResult,
                                      double alarmDistance,
                                      double contactDistance,
                                      int& nIntersecting);

void ObbTreeGPU_TriangleIntersection_Streams(sofa::component::collision::OBBContainer *model1,
                                              sofa::component::collision::OBBContainer *model2,
                                              gProximityWorkerUnit* workerUnit,
                                              std::vector<gProximityWorkerResult*>& workerResults,
                                              std::vector<std::pair<unsigned int, unsigned int> > &workerResultSizes,
                                              std::vector<unsigned int>& workerResultStartIndices,
                                              std::vector<cudaStream_t> &triTestStreams,
                                              bool modSlotAppended,
                                              double alarmDistance,
                                              double contactDistance,
                                              int& nIntersecting);

//void ObbTreeGPU_PostProcess_TriangleIntersection_Streamed(gProximityWorkerUnit* workerUnit,
//                                                          gProximityWorkerResult* workerResult, cudaStream_t &triTestStream, bool cleanResults);

void ObbTreeGPU_TriangleIntersection_Streams_Batch(sofa::component::collision::OBBContainer *model1,
                                              sofa::component::collision::OBBContainer *model2,
                                              gProximityWorkerUnit* workerUnit,
                                              std::vector<gProximityWorkerResult*>& workerResults,
                                              std::vector<std::pair<unsigned int, unsigned int> > &workerResultSizes,
                                              std::vector<unsigned int>& workerResultStartIndices,
                                              std::vector<cudaStream_t> &triTestStreams,
                                              std::vector<cudaEvent_t> &triTestEvents, cudaStream_t &mem_stream,
                                              cudaEvent_t &startEvent, cudaEvent_t &stopEvent,
                                              double alarmDistance,
                                              double contactDistance,
                                              int& nIntersecting, float &elapsedTime);

void ObbTreeGPU_PostProcess_TriangleIntersection_Streamed_Batch(gProximityWorkerUnit* workerUnit,
                                                                std::vector<gProximityWorkerResult *> &workerResults,
                                                                cudaStream_t& mem_stream);

#endif //OBBTREEGPU_TRIANGLE_COLLISION_CUDA_H
