#ifndef OBBTREEGPUCOLLISIONDETECTION_CUDA_H
#define OBBTREEGPUCOLLISIONDETECTION_CUDA_H

#include "initObbTreeGpuPlugin.h"

#include <vector>
#include <map>
#include <gProximity/transform.h>

#include "ObbTreeGPU_CudaDataStructures.h"

#include <cutil/cutil.h>
#include "gProximity/cuda_defs.h"
#include "gProximity/cuda_collision.h"

#include <gProximity/ObbTreeGPU_LinearAlgebra.cuh>
#include <gProximity/cuda_collision.h>

void ObbTreeGPU_BVHCollide(sofa::component::collision::OBBContainer *model1, sofa::component::collision::OBBContainer *model2, std::vector<std::pair<int, int> >& collisionList
#ifdef OBBTREE_GPU_COLLISION_DETECTION_RECORD_INTERSECTING_OBBS
                                                    , int** obbList, int* nObbs
#endif
#ifdef OBBTREE_GPU_DEBUG_TRANSFORMED_VERTICES
                                                    , void** tfVertices1, void** tfVertices2
#endif
                                                    , gProximityDetectionOutput** contactPoints
                                                    , int* numberOfContacts
                                                    , double alarmDistance
                                                    , double contactDistance
													, int& nIntersecting
                                                   );

void ObbTreeGPU_BVHCollide_Streams(sofa::component::collision::OBBContainer *model1,
                                                           sofa::component::collision::OBBContainer *model2,
                                                           gProximityWorkerUnit* workerUnit,
                                                           gProximityWorkerResult* workerResult,
                                                           double alarmDistance,
                                                           double contactDistance,
														   int& nIntersecting
                                                   );

void ObbTreeGPU_BVH_Traverse(sofa::component::collision::OBBContainer *model1,
                             sofa::component::collision::OBBContainer *model2,
                             gProximityWorkerUnit* workerUnit,
                             double alarmDistance,
                             double contactDistance,
                             int& nIntersecting,
                             int workUnitId = 0);

void ObbTreeGPU_BVH_Traverse_Streamed(sofa::component::collision::OBBContainer *model1,
                             sofa::component::collision::OBBContainer *model2,
                             gProximityGPUTransform* model1_transform,
                             gProximityGPUTransform* model2_transform,
                             gProximityWorkerUnit* workerUnit,
                             double alarmDistance,
                             double contactDistance,
                             int& nIntersecting,
                             int workUnitId = 0);

void ObbTreeGPU_BVH_Traverse_Streamed_Batch(std::vector<sofa::component::collision::OBBContainer*>& models1,
                             std::vector<sofa::component::collision::OBBContainer*>& models2,
                             std::vector<gProximityGPUTransform*>& model1_transforms,
                             std::vector<gProximityGPUTransform*>& model2_transforms,
                             std::vector<gProximityWorkerUnit*>& workerUnits,
                             std::vector<cudaStream_t> &workerStreams,
                             double alarmDistance,
                             double contactDistance,
                             std::vector<int>& nIntersecting,
                             int*& workQueueCounts,
                             cudaStream_t &mem_stream,
                             cudaEvent_t &startEvent, cudaEvent_t &stopEvent, cudaEvent_t &balanceEvent,
                             unsigned int numAssignedTraversals,
							 float& elapsedTime_workers,
							 std::vector<float>& elapsedTime_perWorker,
							 std::vector<std::pair<std::string, int64_t> >& elapsedTime_CPUStep,
							 std::vector<cudaEvent_t>& workerStart_Event, std::vector<cudaEvent_t>& workerStop_Event);

#endif // OBBTREEGPUCOLLISIONDETECTION_CUDA_H
