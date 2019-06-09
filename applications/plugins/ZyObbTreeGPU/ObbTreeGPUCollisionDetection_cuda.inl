#include "ObbTreeGPUCollisionDetection_cuda.h"

#include <cutil/cutil.h>
#include "gProximity/cuda_bvh_constru.h"
#include "gProximity/cuda_intersect_tree.h"
#include "gProximity/cuda_defs.h"
#include "gProximity/cuda_make_grid.h"
#include "gProximity/cuda_timer.h"
#include "gProximity/cuda_collision.h"

#include "gProximity/ObbTreeGPU_LinearAlgebra.cuh"

#ifdef USE_TRUNK_CUDA_THRUST
#include <thrust_v180/tuple.h>
#include <thrust_v180/host_vector.h>
#include <thrust_v180/device_ptr.h>
#include <thrust_v180/device_vector.h>
#endif

//#define GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING

//#define GPROXIMITY_DEBUG_BVH_COLLIDE_CONTACT_POINTS
#define GPROXIMITY_DEBUG_BVH_COLLIDE
//#define OBBTREE_GPU_DEBUG_TRANSFORMED_VERTICES
void ObbTreeGPU_BVHCollide(sofa::component::collision::OBBContainer* model1, sofa::component::collision::OBBContainer* model2, std::vector<std::pair<int, int> >& collisionList
#ifdef OBBTREE_GPU_COLLISION_DETECTION_RECORD_INTERSECTING_OBBS
	, int** obbList, int* nObbs
#endif
#ifdef OBBTREE_GPU_DEBUG_TRANSFORMED_VERTICES
	, void** tfVertices1, void** tfVertices2
#endif
	, gProximityDetectionOutput** contactPoints
	, int* numberOfContacts
	, double alarmDistance, double contactDistance
	, int& nIntersecting)
{
	OBBNode* obbTree1 = (OBBNode*)model1->obbTree;
	OBBNode* obbTree2 = (OBBNode*)model2->obbTree;

	GPUVertex* d_vertices1 = (GPUVertex*)model1->vertexPointer;
	GPUVertex* d_vertices2 = (GPUVertex*)model2->vertexPointer;
	uint3* d_triIndices1 = (uint3*)model1->triIdxPointer;
	uint3* d_triIndices2 = (uint3*)model2->triIdxPointer;

	GPUVertex* d_tfVertices1 = (GPUVertex*)model1->vertexTfPointer;
	GPUVertex* d_tfVertices2 = (GPUVertex*)model2->vertexTfPointer;

	//    dim3 grids1 = makeGrid((int)ceilf(model1->nVerts / (float)GENERAL_THREADS));
	//    dim3 grids2 = makeGrid((int)ceilf(model2->nVerts / (float)GENERAL_THREADS));
	//    dim3 threads(GENERAL_THREADS, 1, 1);

	Matrix3x3_d* d_modelTransform1 = NULL;
	Matrix3x3_d* d_modelTransform2 = NULL;
	float3* d_trVector1 = NULL;
	float3* d_trVector2 = NULL;

	Matrix3x3_d h_modelTransform1;
	Matrix3x3_d h_modelTransform2;

	h_modelTransform1.m_row[0].x = model1->modelTransform.m_R[0][0];
	h_modelTransform1.m_row[0].y = model1->modelTransform.m_R[0][1];
	h_modelTransform1.m_row[0].z = model1->modelTransform.m_R[0][2];
	h_modelTransform1.m_row[1].x = model1->modelTransform.m_R[1][0];
	h_modelTransform1.m_row[1].y = model1->modelTransform.m_R[1][1];
	h_modelTransform1.m_row[1].z = model1->modelTransform.m_R[1][2];
	h_modelTransform1.m_row[2].x = model1->modelTransform.m_R[2][0];
	h_modelTransform1.m_row[2].y = model1->modelTransform.m_R[2][1];
	h_modelTransform1.m_row[2].z = model1->modelTransform.m_R[2][2];

	h_modelTransform2.m_row[0].x = model2->modelTransform.m_R[0][0];
	h_modelTransform2.m_row[0].y = model2->modelTransform.m_R[0][1];
	h_modelTransform2.m_row[0].z = model2->modelTransform.m_R[0][2];
	h_modelTransform2.m_row[1].x = model2->modelTransform.m_R[1][0];
	h_modelTransform2.m_row[1].y = model2->modelTransform.m_R[1][1];
	h_modelTransform2.m_row[1].z = model2->modelTransform.m_R[1][2];
	h_modelTransform2.m_row[2].x = model2->modelTransform.m_R[2][0];
	h_modelTransform2.m_row[2].y = model2->modelTransform.m_R[2][1];
	h_modelTransform2.m_row[2].z = model2->modelTransform.m_R[2][2];

	float3 h_trVector1 = make_float3(model1->modelTransform.m_T[0], model1->modelTransform.m_T[1], model1->modelTransform.m_T[2]);
	float3 h_trVector2 = make_float3(model2->modelTransform.m_T[0], model2->modelTransform.m_T[1], model2->modelTransform.m_T[2]);

	GPUMALLOC((void**)&d_modelTransform1, sizeof(Matrix3x3_d));
	GPUMALLOC((void**)&d_modelTransform2, sizeof(Matrix3x3_d));

	GPUMALLOC((void**)&d_trVector1, sizeof(float3));
	GPUMALLOC((void**)&d_trVector2, sizeof(float3));

	TOGPU(d_trVector1, &h_trVector1, sizeof(float3));
	TOGPU(d_trVector2, &h_trVector2, sizeof(float3));

	TOGPU(d_modelTransform1, &h_modelTransform1, sizeof(Matrix3x3_d));
	TOGPU(d_modelTransform2, &h_modelTransform2, sizeof(Matrix3x3_d));
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << " model1 translation for kernel = " << h_trVector1.x << "," << h_trVector1.y << "," << h_trVector1.z << std::endl;
	std::cout << " model2 translation for kernel = " << h_trVector2.x << "," << h_trVector2.y << "," << h_trVector2.z << std::endl;
#endif

	//    //ThreeDVertexTransform <<< grids1, threads >>> (d_vertices1, d_tfVertices1, d_modelTransform1, d_trVector1, model1->nVerts, false);
	//    //ThreeDVertexTransform <<< grids2, threads >>> (d_vertices2, d_tfVertices2, d_modelTransform2, d_trVector2, model2->nVerts, false);

	//    //cudaDeviceSynchronize();

#ifdef OBBTREE_GPU_DEBUG_TRANSFORMED_VERTICES
	FROMGPU((GPUVertex*)(*tfVertices1), d_tfVertices1, sizeof(GPUVertex)* model1->nVerts);
	FROMGPU((GPUVertex*)(*tfVertices2), d_tfVertices2, sizeof(GPUVertex)* model2->nVerts);

	std::cout << " model1 vertices transformed: " << std::endl;
	GPUVertex* vertsM1 = (GPUVertex*)(*tfVertices1);
	for (int k = 0; k < model1->nVerts; k++)
	{
		std::cout << " * " << k << ": " << vertsM1[k].v.x << "," << vertsM1[k].v.y << "," << vertsM1[k].v.z << std::endl;
	}

	GPUVertex* vertsM2 = (GPUVertex*)(*tfVertices2);
	std::cout << " model2 vertices transformed: " << std::endl;
	for (int k = 0; k < model2->nVerts; k++)
	{
		std::cout << " * " << k << ": " << vertsM2[k].v.x << "," << vertsM2[k].v.y << "," << vertsM2[k].v.z << std::endl;
	}

#endif

	unsigned int* d_collisionPairIndex = NULL;
	unsigned int* d_collisionSync = NULL;
	unsigned int* d_nWorkQueueElements = NULL;
	int2* d_collisionPairs = NULL;
	int2* d_workQueues = NULL, *d_workQueues2 = NULL;
	unsigned int* d_workQueueCounts = NULL;
	int* d_balanceSignal = NULL;

	unsigned int *d_obbOutputIndex = NULL;
	int2* d_intersectingOBBs = NULL;

	// allocate collision list (try to be conservative)
	unsigned int collisionPairCapacity = COLLISION_PAIR_CAPACITY;
	GPUMALLOC((void**)&d_collisionPairs, sizeof(int2)* collisionPairCapacity);

	GPUMALLOC((void **)&d_collisionPairIndex, sizeof(int));
	GPUMALLOC((void **)&d_nWorkQueueElements, sizeof(int));
	GPUMALLOC((void **)&d_collisionSync, sizeof(int));

	unsigned int obbCount = COLLISION_PAIR_CAPACITY;
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << "OBB count for intersection log: " << obbCount << std::endl;
#endif
	GPUMALLOC((void **)&d_intersectingOBBs, sizeof(int2)* obbCount);

	// allocate work queues
	GPUMALLOC((void **)&d_workQueues, sizeof(int2)*QUEUE_NTASKS*QUEUE_SIZE_PER_TASK_GLOBAL);
	GPUMALLOC((void **)&d_workQueues2, sizeof(int2)*QUEUE_NTASKS*QUEUE_SIZE_PER_TASK_GLOBAL);
	GPUMALLOC((void **)&d_workQueueCounts, sizeof(int)*QUEUE_NTASKS);
	GPUMALLOC((void**)&d_balanceSignal, sizeof(int));

	// init first work element:
	GPUMEMSET(d_workQueues, 0, sizeof(int2));
	GPUMEMSET(d_workQueueCounts, 0, sizeof(int)* QUEUE_NTASKS);
	GPUMEMSET(d_collisionPairIndex, 0, sizeof(int));

	GPUMALLOC((void **)&d_obbOutputIndex, sizeof(int));
	GPUMEMSET(d_obbOutputIndex, 0, sizeof(int));

	unsigned int firstCount = 1;
	TOGPU(d_workQueueCounts, &firstCount, sizeof(unsigned int));

	unsigned int nPairs = 0;
	int nActiveSplits = 1;
	int nRuns = 0;
	int bNeedBalance = 0;

	TimerValue balanceTimer, traverseTimer;
	double elapsedBalance = 0, elapsedTraverse = 0, elapsedTraverseTotal = 0, elapsedBalanceTotal = 0;

	bool bstop = false;

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal");

	while (nActiveSplits)
	{
		GPUMEMSET(d_collisionSync, 0, sizeof(int));


		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal_traverse");
		traverseTimer.start();

		traverseTree<OBBNode, OBB, QUEUE_SIZE_PER_TASK, QUEUE_SIZE_PER_TASK_INIT, TRAVERSAL_THREADS> << < QUEUE_NTASKS, TRAVERSAL_THREADS >> >
			(obbTree1, d_vertices1, d_triIndices1, obbTree2, d_vertices2, d_triIndices2,
			d_workQueues, d_workQueueCounts, d_collisionSync, QUEUE_SIZE_PER_TASK_GLOBAL,
			d_collisionPairs, d_collisionPairIndex, NULL,
#ifdef GPROXIMITY_TRAVERSETREE_STORE_OVERLAPPING_OBB_PAIRS
			d_intersectingOBBs, d_obbOutputIndex, obbCount,
#endif
			d_modelTransform1, d_modelTransform2, d_trVector1, d_trVector2, alarmDistance);

		// cudaDeviceSynchronize();

		traverseTimer.stop();
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal_traverse");

		elapsedTraverse = traverseTimer.getElapsedMicroSec();
		elapsedTraverseTotal += elapsedTraverse;
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
		printf("traversal time: %f; total: %f\n", elapsedTraverse, elapsedTraverseTotal);
#endif

#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
		std::cout << " allocate workqueue counts array: " << QUEUE_NTASKS << " elements." << std::endl;
#endif
		unsigned int* workQueueCounts = new unsigned int[QUEUE_NTASKS];
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
		std::cout << " workqueue counts allocated; transfer from GPU memory" << std::endl;
#endif
		FROMGPU(workQueueCounts, d_workQueueCounts, sizeof(unsigned int)* QUEUE_NTASKS);
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
		std::cout << " transferred from GPU memory: " << sizeof(unsigned int)* QUEUE_NTASKS << std::endl;
#endif
		for (int i = 0; i < QUEUE_NTASKS; ++i)
		{
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
			std::cout << " * " << workQueueCounts[i] << " >= " << QUEUE_SIZE_PER_TASK_GLOBAL << ": " << (workQueueCounts[i] >= QUEUE_SIZE_PER_TASK_GLOBAL) << std::endl;
#endif
			if (workQueueCounts[i] >= QUEUE_SIZE_PER_TASK_GLOBAL)
			{
				bstop = true;
				printf("the %d-th global queue is overflow! %d\n", i, workQueueCounts[i]);
				break;
			}
		}

		delete[] workQueueCounts;

		if (bstop)
			break;


		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal_balance");

		balanceTimer.start();
		balanceWorkList<BALANCE_THREADS, QUEUE_NTASKS, int2> << < 1, BALANCE_THREADS >> >(d_workQueues, d_workQueues2, d_workQueueCounts, QUEUE_SIZE_PER_TASK_GLOBAL, d_nWorkQueueElements, d_balanceSignal);
		balanceTimer.stop();

		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal_balance");

		elapsedBalance = balanceTimer.getElapsedMicroSec();
		elapsedBalanceTotal += elapsedBalance;
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
		printf("balance time: %f, total: %f\n", elapsedBalance, elapsedBalanceTotal);
#endif
		FROMGPU(&nActiveSplits, d_nWorkQueueElements, sizeof(unsigned int));
		FROMGPU(&bNeedBalance, d_balanceSignal, sizeof(unsigned int));
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		printf("active splits num: %d\n", nActiveSplits);
#endif
		if (bNeedBalance == 1)
		{
			int2* t = d_workQueues;
			d_workQueues = d_workQueues2;
			d_workQueues2 = t;
		}

		nRuns++;
	}

	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal");

	FROMGPU(&nPairs, d_collisionPairIndex, sizeof(int));

	#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << "Overlapping OBB leaf nodes: " << nPairs << std::endl;
	#endif

#ifdef OBBTREE_GPU_COLLISION_DETECTION_RECORD_INTERSECTING_OBBS
	int nOBBPairs = 0;
	FROMGPU(&nOBBPairs, d_obbOutputIndex, sizeof(int));

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << "Overlapping OBB pairs: " << nOBBPairs << std::endl;
#endif

	if (obbList && nOBBPairs > 0)
	{
		*obbList = new int[2 * nOBBPairs];
		FROMGPU(*obbList, d_intersectingOBBs, sizeof(int2)* nOBBPairs);
		*nObbs = nOBBPairs;
	}
#endif

#ifdef GPROXIMITY_RUN_PRE_INTERSECTION_TESTS
	// run actual collision tests of primitives
	unsigned int nCollision = CUDA_trianglePairCollide(/*d_vertices1*/ d_tfVertices1, d_triIndices1, /*d_vertices2*/ d_tfVertices2, d_triIndices2, d_collisionPairs, nPairs, collisionList);
	int2* h_collidingPairs = new int2[nCollision];

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << "Colliding triangle pairs: " << nCollision << std::endl;
#endif
	unsigned long nCollidingPairs = 0;
	for (std::vector<std::pair<int, int> >::const_iterator it = collisionList.begin(); it != collisionList.end(); it++)
	{
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		std::cout << it->first << " - " << it->second << ";";
#endif
		h_collidingPairs[nCollidingPairs] = make_int2(it->first, it->second);
		nCollidingPairs++;
	}
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << std::endl;
	std::cout << "Colliding triangle pair count = " << nCollidingPairs << std::endl;
#endif

	int2* d_collidingPairs = NULL;
	GPUMALLOC((void**)&d_collidingPairs, sizeof(int2)* nCollidingPairs);
	TOGPU(d_collidingPairs, h_collidingPairs, sizeof(int2)* nCollidingPairs);
#else
	unsigned long nCollidingPairs = nPairs;
#endif

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	static long iterationCount = 0;
	iterationCount++;
	size_t mem_free, mem_total;
	cudaMemGetInfo(&mem_free, &mem_total);

	std::cout << "Iteration " << iterationCount << ", before intersection test -- GPU memory: " << mem_free << " free of total " << mem_total << std::endl;

    std::cout << "BEFORE TRIANGLE INTERSECTION TESTS: nCollidingPairs = " << nCollidingPairs << std::endl;
#endif

	int nContacts = 0;
	if (nCollidingPairs > 0)
	{

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_Allocations");
		gProximityDetectionOutput* h_contacts = new gProximityDetectionOutput;

		h_contacts->valid = NULL;
		h_contacts->contactId = NULL;
		h_contacts->elems = NULL;
		h_contacts->distance = NULL;
		h_contacts->normal = NULL;
		h_contacts->point0 = NULL;
		h_contacts->point1 = NULL;
		h_contacts->contactType = NULL;

		//GPUMALLOC((void **)d_contacts, sizeof(struct DetectionOutput));

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		std::cout << "ObbTreeGPU_BVHCollide(): allocate h_contacts->valid, size = " << sizeof(bool)* nCollidingPairs * CollisionTestElementsSize << " (" << sizeof(bool) << " * " << nCollidingPairs << " * " << CollisionTestElementsSize << ")" << std::endl;
#endif
		GPUMALLOC((void **)&(h_contacts->valid), sizeof(bool)* nCollidingPairs * CollisionTestElementsSize);

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		std::cout << "ObbTreeGPU_BVHCollide(): allocate h_contacts->contactId, size = " << sizeof(int)* nCollidingPairs * CollisionTestElementsSize << " (" << sizeof(int) << " * " << nCollidingPairs << " * " << CollisionTestElementsSize << ")" << std::endl;
#endif
		GPUMALLOC((void **)&(h_contacts->contactId), sizeof(int)* nCollidingPairs * CollisionTestElementsSize);

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		std::cout << "ObbTreeGPU_BVHCollide(): allocate h_contacts->distance, size = " << sizeof(double)* nCollidingPairs * CollisionTestElementsSize << " (" << sizeof(double) << " * " << nCollidingPairs << " * " << CollisionTestElementsSize << ")" << std::endl;
#endif
		GPUMALLOC((void **)&(h_contacts->distance), sizeof(double)* nCollidingPairs * CollisionTestElementsSize);

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		std::cout << "ObbTreeGPU_BVHCollide(): allocate h_contacts->elems, size = " << sizeof(int4)* nCollidingPairs * CollisionTestElementsSize << " (" << sizeof(int4) << " * " << nCollidingPairs << " * " << CollisionTestElementsSize << ")" << std::endl;
#endif
		GPUMALLOC((void **)&(h_contacts->elems), sizeof(int4)* nCollidingPairs * CollisionTestElementsSize);

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		std::cout << "ObbTreeGPU_BVHCollide(): allocate h_contacts->normal, size = " << sizeof(float3)* nCollidingPairs * CollisionTestElementsSize << " (" << sizeof(float3) << " * " << nCollidingPairs << " * " << CollisionTestElementsSize << ")" << std::endl;
#endif
		GPUMALLOC((void **)&(h_contacts->normal), sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		std::cout << "ObbTreeGPU_BVHCollide(): allocate h_contacts->point0, size = " << sizeof(float3)* nCollidingPairs * CollisionTestElementsSize << " (" << sizeof(float3) << " * " << nCollidingPairs << " * " << CollisionTestElementsSize << ")" << std::endl;
#endif
		GPUMALLOC((void **)&(h_contacts->point0), sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		std::cout << "ObbTreeGPU_BVHCollide(): allocate h_contacts->point1, size = " << sizeof(int)* nCollidingPairs * CollisionTestElementsSize << " (" << sizeof(float3) << " * " << nCollidingPairs << " * " << CollisionTestElementsSize << ")" << std::endl;
#endif
		GPUMALLOC((void **)&(h_contacts->point1), sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		std::cout << "ObbTreeGPU_BVHCollide(): allocate h_contacts->contactType, size = " << sizeof(gProximityContactType)* nCollidingPairs * CollisionTestElementsSize << " (" << sizeof(gProximityContactType) << " * " << nCollidingPairs << " * " << CollisionTestElementsSize << ")" << std::endl;
#endif
		GPUMALLOC((void **)&(h_contacts->contactType), sizeof(gProximityContactType)* nCollidingPairs * CollisionTestElementsSize);

		GPUMEMSET(h_contacts->valid, 0, sizeof(bool)* nCollidingPairs * CollisionTestElementsSize);
		GPUMEMSET(h_contacts->contactId, 0, sizeof(int)* nCollidingPairs * CollisionTestElementsSize);
		GPUMEMSET(h_contacts->contactType, (int)COLLISION_INVALID, sizeof(gProximityContactType)* nCollidingPairs * CollisionTestElementsSize);

		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_Allocations");

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer");
		gProximityDetectionOutput* d_contacts = NULL;
		GPUMALLOC((void**)& d_contacts, sizeof(gProximityDetectionOutput));
		TOGPU(d_contacts, h_contacts, sizeof(gProximityDetectionOutput));
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_ToGPU");

#ifdef GPROXIMITY_RUN_PRE_INTERSECTION_TESTS
		CUDA_trianglePairIntersect(d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2,
			d_collidingPairs, nCollidingPairs, d_contacts, nCollidingPairs * CollisionTestElementsSize, alarmDistance, contactDistance);
#else

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_Compute");

		CUDA_trianglePairIntersect(d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2,
			d_collisionPairs, nCollidingPairs, d_contacts, nCollidingPairs * CollisionTestElementsSize, alarmDistance, contactDistance);

		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_Compute");
#endif
		// cudaDeviceSynchronize();

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		std::cout << "Overlapping pairs: " << nCollidingPairs << "; Max. contacts collected: " << nCollidingPairs * CollisionTestElementsSize << std::endl;
		std::cout << " valid array size = " << sizeof(bool)* nCollidingPairs * CollisionTestElementsSize << std::endl;
		std::cout << " distance array size = " << sizeof(double)* nCollidingPairs * CollisionTestElementsSize << std::endl;
		std::cout << " elems array size = " << sizeof(int2)* nCollidingPairs * CollisionTestElementsSize << std::endl;
		std::cout << " point0/point1/normal array size = " << sizeof(float3)* nCollidingPairs * CollisionTestElementsSize << std::endl;
#endif

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU");

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_HostAlloc");
		gProximityDetectionOutput h_detectedContacts;
		h_detectedContacts.valid = new bool[nCollidingPairs * CollisionTestElementsSize];
		h_detectedContacts.contactId = new int[nCollidingPairs * CollisionTestElementsSize];
		h_detectedContacts.distance = new double[nCollidingPairs * CollisionTestElementsSize];
		h_detectedContacts.elems = new int4[nCollidingPairs * CollisionTestElementsSize];
		h_detectedContacts.normal = new float3[nCollidingPairs * CollisionTestElementsSize];
		h_detectedContacts.point0 = new float3[nCollidingPairs * CollisionTestElementsSize];
		h_detectedContacts.point1 = new float3[nCollidingPairs * CollisionTestElementsSize];
		h_detectedContacts.contactType = new gProximityContactType[nCollidingPairs * CollisionTestElementsSize];

		/*cudaMallocHost((void**)&(h_detectedContacts.valid), sizeof(bool) * nCollidingPairs * CollisionTestElementsSize);
		cudaMallocHost((void**)&(h_detectedContacts.contactId), sizeof(int) * nCollidingPairs * CollisionTestElementsSize);
		cudaMallocHost((void**)&(h_detectedContacts.distance), sizeof(double) * nCollidingPairs * CollisionTestElementsSize);
		cudaMallocHost((void**)&(h_detectedContacts.elems), sizeof(int4) * nCollidingPairs * CollisionTestElementsSize);
		cudaMallocHost((void**)&(h_detectedContacts.normal), sizeof(float3) * nCollidingPairs * CollisionTestElementsSize);
		cudaMallocHost((void**)&(h_detectedContacts.point0), sizeof(float3) * nCollidingPairs * CollisionTestElementsSize);
		cudaMallocHost((void**)&(h_detectedContacts.point1), sizeof(float3) * nCollidingPairs * CollisionTestElementsSize);
		cudaMallocHost((void**)&(h_detectedContacts.contactType), sizeof(gProximityContactType) * nCollidingPairs * CollisionTestElementsSize);*/

		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_HostAlloc");

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer");

		/*cudaEvent_t startEvent, stopEvent;
		float ms;

		int nStreams = 8;
		cudaStream_t stream[nStreams];
		for (int i = 0; i < nStreams; ++i)
		cudaStreamCreate(&stream[i]);

		cudaEventCreate(&startEvent);
		cudaEventCreate(&stopEvent);

		cudaEventRecord(startEvent,0);*/

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_valid");
		//cudaMemcpyAsync(h_detectedContacts.valid, h_contacts->valid, sizeof(bool) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[0]);
		FROMGPU(h_detectedContacts.valid, h_contacts->valid, sizeof(bool)* nCollidingPairs * CollisionTestElementsSize);
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_valid");

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_contactId");
		FROMGPU(h_detectedContacts.contactId, h_contacts->contactId, sizeof(int)* nCollidingPairs * CollisionTestElementsSize);
		//cudaMemcpyAsync(h_detectedContacts.contactId, h_contacts->contactId, sizeof(int) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[1]);
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_contactId");

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_distance");
		FROMGPU(h_detectedContacts.distance, h_contacts->distance, sizeof(double)* nCollidingPairs * CollisionTestElementsSize);
		//cudaMemcpyAsync(h_detectedContacts.distance, h_contacts->distance, sizeof(double) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[2]);
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_distance");

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_elems");
		FROMGPU(h_detectedContacts.elems, h_contacts->elems, sizeof(int4)* nCollidingPairs * CollisionTestElementsSize);
		//cudaMemcpyAsync(h_detectedContacts.elems, h_contacts->elems, sizeof(int4) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[3]);
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_elems");

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_normal");
		FROMGPU(h_detectedContacts.normal, h_contacts->normal, sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);
		//cudaMemcpyAsync(h_detectedContacts.normal, h_contacts->normal, sizeof(float3) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[4]);
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_normal");

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_point0");
		FROMGPU(h_detectedContacts.point0, h_contacts->point0, sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);
		//cudaMemcpyAsync(h_detectedContacts.point0, h_contacts->point0, sizeof(float3) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[5]);
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_point0");

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_point1");
		FROMGPU(h_detectedContacts.point1, h_contacts->point1, sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);
		//cudaMemcpyAsync(h_detectedContacts.point1, h_contacts->point1, sizeof(float3) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[6]);
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_point1");

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_contactType");
		FROMGPU(h_detectedContacts.contactType, h_contacts->contactType, sizeof(gProximityContactType)* nCollidingPairs * CollisionTestElementsSize);
		//cudaMemcpyAsync(h_detectedContacts.contactType, h_contacts->contactType, sizeof(gProximityContactType) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[7]);
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_contactType");

		/*cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&ms, startEvent, stopEvent);

		printf("Time for asynchronous transfer (ms): %f\n", ms);

		cudaEventDestroy(startEvent);
		cudaEventDestroy(stopEvent);*/
		/*for (int i = 0; i < nStreams; ++i)
		cudaStreamDestroy(stream[i]);*/

		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer");

		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU");

		for (unsigned int k = 0; k < nCollidingPairs * CollisionTestElementsSize; k++)
		{
			if (h_detectedContacts.valid[k])
			{
				nContacts++;
#ifndef GPROXIMITY_RUN_PRE_INTERSECTION_TESTS
				collisionList.push_back(std::make_pair(h_detectedContacts.elems[k].w, h_detectedContacts.elems[k].y));
#endif
			}
		}

		*numberOfContacts = nContacts;

		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_ToHost");

		{
			*contactPoints = new gProximityDetectionOutput;
			(*contactPoints)->valid = new bool[nContacts];
			(*contactPoints)->contactId = new int[nContacts];
			(*contactPoints)->elems = new int4[nContacts];
			(*contactPoints)->point0 = new float3[nContacts];
			(*contactPoints)->point1 = new float3[nContacts];
			(*contactPoints)->normal = new float3[nContacts];
			(*contactPoints)->distance = new double[nContacts];
			(*contactPoints)->contactType = new gProximityContactType[nContacts];
		}

		long validContactIdx = 0;

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_CONTACT_POINTS
		std::cout << "=== Max. Contact points detected: " << nCollidingPairs * CollisionTestElementsSize << " ===" << std::endl;
#endif
		for (unsigned int k = 0; k < nCollidingPairs * CollisionTestElementsSize; k++)
		{
			if (h_detectedContacts.valid[k])
			{
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_CONTACT_POINTS
				std::cout << " * pos. " << k << ": ";
				std::cout << " type = ";
				if (h_detectedContacts.contactType[k] == COLLISION_LINE_LINE)
					std::cout << "LINE_LINE";
				else if (h_detectedContacts.contactType[k] == COLLISION_VERTEX_FACE)
					std::cout << "VERTEX_FACE";
				else
					std::cout << "INVALID";

				std::cout << ", elems = " << h_detectedContacts.elems[k].x << " - " << h_detectedContacts.elems[k].y << "; distance = " << h_detectedContacts.distance[k] << ", point0 = " << h_detectedContacts.point0[k].x << "," << h_detectedContacts.point0[k].y << "," << h_detectedContacts.point0[k].z <<
					", point1 = " << h_detectedContacts.point1[k].x << "," << h_detectedContacts.point1[k].y << "," << h_detectedContacts.point1[k].z << ", normal = " <<
					h_detectedContacts.normal[k].x << "," << h_detectedContacts.normal[k].y << "," << h_detectedContacts.normal[k].z;
				if (h_detectedContacts.distance[k] < 0.000001)
					std::cout << " --- DISTANCE UNDERRUN";

				std::cout << std::endl;
#endif
				if (h_detectedContacts.distance[k] < 0.000001)
					(*contactPoints)->valid[validContactIdx] = false;
				else
					(*contactPoints)->valid[validContactIdx] = true;

				(*contactPoints)->contactId[validContactIdx] = h_detectedContacts.contactId[k];
				(*contactPoints)->elems[validContactIdx] = make_int4(h_detectedContacts.elems[k].x, h_detectedContacts.elems[k].y, h_detectedContacts.elems[k].z, h_detectedContacts.elems[k].w);
				(*contactPoints)->point0[validContactIdx] = make_float3(h_detectedContacts.point0[k].x, h_detectedContacts.point0[k].y, h_detectedContacts.point0[k].z);
				(*contactPoints)->point1[validContactIdx] = make_float3(h_detectedContacts.point1[k].x, h_detectedContacts.point1[k].y, h_detectedContacts.point1[k].z);
				(*contactPoints)->normal[validContactIdx] = make_float3(h_detectedContacts.normal[k].x, h_detectedContacts.normal[k].y, h_detectedContacts.normal[k].z);
				(*contactPoints)->distance[validContactIdx] = h_detectedContacts.distance[k];
				(*contactPoints)->contactType[validContactIdx] = h_detectedContacts.contactType[k];
				validContactIdx++;
			}
		}

		delete[] h_detectedContacts.valid;
		delete[] h_detectedContacts.contactId;
		delete[] h_detectedContacts.distance;
		delete[] h_detectedContacts.elems;
		delete[] h_detectedContacts.normal;
		delete[] h_detectedContacts.point0;
		delete[] h_detectedContacts.point1;
		delete[] h_detectedContacts.contactType;

		/*cudaFreeHost(h_detectedContacts.valid);
		cudaFreeHost(h_detectedContacts.contactId);
		cudaFreeHost(h_detectedContacts.distance);
		cudaFreeHost(h_detectedContacts.elems);
		cudaFreeHost(h_detectedContacts.normal);
		cudaFreeHost(h_detectedContacts.point0);
		cudaFreeHost(h_detectedContacts.point1);
		cudaFreeHost(h_detectedContacts.contactType);*/

		GPUFREE(h_contacts->valid);
		GPUFREE(h_contacts->contactId);
		GPUFREE(h_contacts->distance);
		GPUFREE(h_contacts->elems);
		GPUFREE(h_contacts->normal);
		GPUFREE(h_contacts->point0);
		GPUFREE(h_contacts->point1);
		GPUFREE(h_contacts->contactType);

		GPUFREE(d_contacts);

		delete h_contacts;

#ifdef GPROXIMITY_RUN_PRE_INTERSECTION_TESTS
		GPUFREE(d_collidingPairs);
		delete[] h_collidingPairs;
#endif

		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_ToHost");
	}
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << "Contact points generated: " << nContacts << std::endl;
#endif

	GPUFREE(d_modelTransform1);
	GPUFREE(d_modelTransform2);
	GPUFREE(d_trVector1);
	GPUFREE(d_trVector2);
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	cudaMemGetInfo(&mem_free, &mem_total);
	std::cout << "Iteration " << iterationCount << ", after intersection test -- GPU memory: " << mem_free << " free of total " << mem_total << std::endl;
#endif

#ifdef GPROXIMITY_INVERSE_VERTICES_TRANSFORM
	GPUMALLOC((void**)&d_trVector1, sizeof(float3));
	GPUMALLOC((void**)&d_trVector2, sizeof(float3));

	GPUMALLOC((void**)&d_modelTransform1, sizeof(Matrix3x3_d));
	GPUMALLOC((void**)&d_modelTransform2, sizeof(Matrix3x3_d));

	TOGPU(d_trVector1, &h_trVector1, sizeof(float3));
	TOGPU(d_trVector2, &h_trVector2, sizeof(float3));

	Matrix3x3_d h_invModelTransform1;
	Matrix3x3_d h_invModelTransform2;

	h_invModelTransform1.m_row[0].x = model1->modelTransform.m_R[0][0];
	h_invModelTransform1.m_row[0].y = model1->modelTransform.m_R[1][0];
	h_invModelTransform1.m_row[0].z = model1->modelTransform.m_R[2][0];
	h_invModelTransform1.m_row[1].x = model1->modelTransform.m_R[0][1];
	h_invModelTransform1.m_row[1].y = model1->modelTransform.m_R[1][1];
	h_invModelTransform1.m_row[1].z = model1->modelTransform.m_R[2][1];
	h_invModelTransform1.m_row[2].x = model1->modelTransform.m_R[0][2];
	h_invModelTransform1.m_row[2].y = model1->modelTransform.m_R[1][2];
	h_invModelTransform1.m_row[2].z = model1->modelTransform.m_R[2][2];

	h_invModelTransform2.m_row[0].x = model2->modelTransform.m_R[0][0];
	h_invModelTransform2.m_row[0].y = model2->modelTransform.m_R[1][0];
	h_invModelTransform2.m_row[0].z = model2->modelTransform.m_R[2][0];
	h_invModelTransform2.m_row[1].x = model2->modelTransform.m_R[0][1];
	h_invModelTransform2.m_row[1].y = model2->modelTransform.m_R[1][1];
	h_invModelTransform2.m_row[1].z = model2->modelTransform.m_R[2][1];
	h_invModelTransform2.m_row[2].x = model2->modelTransform.m_R[0][2];
	h_invModelTransform2.m_row[2].y = model2->modelTransform.m_R[1][2];
	h_invModelTransform2.m_row[2].z = model2->modelTransform.m_R[2][2];

	TOGPU(d_modelTransform1, &h_invModelTransform1, sizeof(Matrix3x3_d));
	TOGPU(d_modelTransform2, &h_invModelTransform2, sizeof(Matrix3x3_d));

	ThreeDVertexTransform << < grids1, threads >> > (d_vertices1, d_tfVertices1, d_modelTransform1, d_trVector1, model1->nVerts, true);
	ThreeDVertexTransform << < grids2, threads >> > (d_vertices2, d_tfVertices2, d_modelTransform2, d_trVector2, model2->nVerts, true);

	cudaThreadSynchronize();

#endif

	GPUFREE(d_workQueueCounts);
	GPUFREE(d_workQueues);
	GPUFREE(d_workQueues2);
	GPUFREE(d_collisionPairs);
	GPUFREE(d_intersectingOBBs);
	GPUFREE(d_collisionPairIndex);
	GPUFREE(d_obbOutputIndex);

	GPUFREE(d_nWorkQueueElements);
	GPUFREE(d_collisionSync);
	GPUFREE(d_balanceSignal);

	//return nPairs;
	nIntersecting = nPairs;
}

void ObbTreeGPU_BVHCollide_Streams(sofa::component::collision::OBBContainer* model1,
	sofa::component::collision::OBBContainer* model2,
	gProximityWorkerUnit* workerUnit,
	gProximityWorkerResult* workerResult,
	double alarmDistance, double contactDistance,
	int& nIntersecting)
{
	OBBNode* obbTree1 = (OBBNode*)model1->obbTree;
	OBBNode* obbTree2 = (OBBNode*)model2->obbTree;

	GPUVertex* d_vertices1 = (GPUVertex*)model1->vertexPointer;
	GPUVertex* d_vertices2 = (GPUVertex*)model2->vertexPointer;
	uint3* d_triIndices1 = (uint3*)model1->triIdxPointer;
	uint3* d_triIndices2 = (uint3*)model2->triIdxPointer;

	GPUVertex* d_tfVertices1 = (GPUVertex*)model1->vertexTfPointer;
	GPUVertex* d_tfVertices2 = (GPUVertex*)model2->vertexTfPointer;

	Matrix3x3_d* d_modelTransform1 = NULL;
	Matrix3x3_d* d_modelTransform2 = NULL;
	float3* d_trVector1 = NULL;
	float3* d_trVector2 = NULL;

	Matrix3x3_d h_modelTransform1;
	Matrix3x3_d h_modelTransform2;

	h_modelTransform1.m_row[0].x = model1->modelTransform.m_R[0][0];
	h_modelTransform1.m_row[0].y = model1->modelTransform.m_R[0][1];
	h_modelTransform1.m_row[0].z = model1->modelTransform.m_R[0][2];
	h_modelTransform1.m_row[1].x = model1->modelTransform.m_R[1][0];
	h_modelTransform1.m_row[1].y = model1->modelTransform.m_R[1][1];
	h_modelTransform1.m_row[1].z = model1->modelTransform.m_R[1][2];
	h_modelTransform1.m_row[2].x = model1->modelTransform.m_R[2][0];
	h_modelTransform1.m_row[2].y = model1->modelTransform.m_R[2][1];
	h_modelTransform1.m_row[2].z = model1->modelTransform.m_R[2][2];

	h_modelTransform2.m_row[0].x = model2->modelTransform.m_R[0][0];
	h_modelTransform2.m_row[0].y = model2->modelTransform.m_R[0][1];
	h_modelTransform2.m_row[0].z = model2->modelTransform.m_R[0][2];
	h_modelTransform2.m_row[1].x = model2->modelTransform.m_R[1][0];
	h_modelTransform2.m_row[1].y = model2->modelTransform.m_R[1][1];
	h_modelTransform2.m_row[1].z = model2->modelTransform.m_R[1][2];
	h_modelTransform2.m_row[2].x = model2->modelTransform.m_R[2][0];
	h_modelTransform2.m_row[2].y = model2->modelTransform.m_R[2][1];
	h_modelTransform2.m_row[2].z = model2->modelTransform.m_R[2][2];

	float3 h_trVector1 = make_float3(model1->modelTransform.m_T[0], model1->modelTransform.m_T[1], model1->modelTransform.m_T[2]);
	float3 h_trVector2 = make_float3(model2->modelTransform.m_T[0], model2->modelTransform.m_T[1], model2->modelTransform.m_T[2]);

	GPUMALLOC((void**)&d_modelTransform1, sizeof(Matrix3x3_d));
	GPUMALLOC((void**)&d_modelTransform2, sizeof(Matrix3x3_d));

	GPUMALLOC((void**)&d_trVector1, sizeof(float3));
	GPUMALLOC((void**)&d_trVector2, sizeof(float3));

	TOGPU(d_trVector1, &h_trVector1, sizeof(float3));
	TOGPU(d_trVector2, &h_trVector2, sizeof(float3));

	TOGPU(d_modelTransform1, &h_modelTransform1, sizeof(Matrix3x3_d));
	TOGPU(d_modelTransform2, &h_modelTransform2, sizeof(Matrix3x3_d));
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << " model1 translation for kernel = " << h_trVector1.x << "," << h_trVector1.y << "," << h_trVector1.z << std::endl;
	std::cout << " model2 translation for kernel = " << h_trVector2.x << "," << h_trVector2.y << "," << h_trVector2.z << std::endl;
#endif

	/*unsigned int* d_collisionPairIndex = NULL;
	unsigned int* d_collisionSync = NULL;
	unsigned int* d_nWorkQueueElements = NULL;
	int2* d_collisionPairs = NULL;
	int2* d_workQueues = NULL, *d_workQueues2 = NULL;
	unsigned int* d_workQueueCounts = NULL;
	int* d_balanceSignal = NULL;

	unsigned int *d_obbOutputIndex = NULL;
	int2* d_intersectingOBBs = NULL;

	// allocate collision list (try to be conservative)
	unsigned int collisionPairCapacity = COLLISION_PAIR_CAPACITY;
	GPUMALLOC((void**)&d_collisionPairs, sizeof(int2) * collisionPairCapacity);

	GPUMALLOC((void **)&d_collisionPairIndex, sizeof(int));
	GPUMALLOC((void **)&d_nWorkQueueElements, sizeof(int));
	GPUMALLOC((void **)&d_collisionSync, sizeof(int));

	unsigned int obbCount = COLLISION_PAIR_CAPACITY;
	GPUMALLOC((void **)&d_intersectingOBBs, sizeof(int2) * obbCount);

	// allocate work queues
	GPUMALLOC((void **)&d_workQueues, sizeof(int2)*QUEUE_NTASKS*QUEUE_SIZE_PER_TASK_GLOBAL);
	GPUMALLOC((void **)&d_workQueues2, sizeof(int2)*QUEUE_NTASKS*QUEUE_SIZE_PER_TASK_GLOBAL);
	GPUMALLOC((void **)&d_workQueueCounts, sizeof(int)*QUEUE_NTASKS);
	GPUMALLOC((void**)&d_balanceSignal, sizeof(int));*/

	// init first work element:
	GPUMEMSET(workerUnit->d_workQueues, 0, sizeof(int2));
	GPUMEMSET(workerUnit->d_workQueueCounts, 0, sizeof(int)* QUEUE_NTASKS);
	GPUMEMSET(workerUnit->d_collisionPairIndex, 0, sizeof(int));

	// GPUMALLOC((void **)&d_obbOutputIndex, sizeof(int));
	GPUMEMSET(workerUnit->d_obbOutputIndex, 0, sizeof(int));

	unsigned int firstCount = 1;
	TOGPU(workerUnit->d_workQueueCounts, &firstCount, sizeof(unsigned int));

	unsigned int nPairs = 0;
	int nActiveSplits = 1;
	int nRuns = 0;
	int bNeedBalance = 0;

	TimerValue balanceTimer, traverseTimer;
	double elapsedBalance = 0, elapsedTraverse = 0, elapsedTraverseTotal = 0, elapsedBalanceTotal = 0;

	bool bstop = false;

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal");

	while (nActiveSplits)
	{
		GPUMEMSET(workerUnit->d_collisionSync, 0, sizeof(int));


		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal_traverse");
		traverseTimer.start();

		traverseTree<OBBNode, OBB, QUEUE_SIZE_PER_TASK, QUEUE_SIZE_PER_TASK_INIT, TRAVERSAL_THREADS> << < QUEUE_NTASKS, TRAVERSAL_THREADS, 0, *(workerUnit->_stream) >> >
			(obbTree1, d_vertices1, d_triIndices1, obbTree2, d_vertices2, d_triIndices2,
			workerUnit->d_workQueues, workerUnit->d_workQueueCounts, workerUnit->d_collisionSync, QUEUE_SIZE_PER_TASK_GLOBAL,
			workerUnit->d_collisionPairs, workerUnit->d_collisionPairIndex, workerUnit->d_collisionLeafs,
#ifdef GPROXIMITY_TRAVERSETREE_STORE_OVERLAPPING_OBB_PAIRS
			d_intersectingOBBs, workerUnit->d_obbOutputIndex, obbCount,
#endif
			d_modelTransform1, d_modelTransform2, d_trVector1, d_trVector2, alarmDistance);

		// cudaDeviceSynchronize();

		traverseTimer.stop();
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal_traverse");

		elapsedTraverse = traverseTimer.getElapsedMicroSec();
		elapsedTraverseTotal += elapsedTraverse;
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
		printf("traversal time (streaming): %f; total: %f\n", elapsedTraverse, elapsedTraverseTotal);
#endif

#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
		std::cout << " allocate workqueue counts array: " << QUEUE_NTASKS << " elements." << std::endl;
#endif
		unsigned int* workQueueCounts = new unsigned int[QUEUE_NTASKS];
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
		std::cout << " workqueue counts allocated; transfer from GPU memory" << std::endl;
#endif
		FROMGPU(workQueueCounts, workerUnit->d_workQueueCounts, sizeof(unsigned int)* QUEUE_NTASKS);
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
		std::cout << " transferred from GPU memory: " << sizeof(unsigned int)* QUEUE_NTASKS << std::endl;
#endif
		for (int i = 0; i < QUEUE_NTASKS; ++i)
		{
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
			std::cout << " * " << workQueueCounts[i] << " >= " << QUEUE_SIZE_PER_TASK_GLOBAL << ": " << (workQueueCounts[i] >= QUEUE_SIZE_PER_TASK_GLOBAL) << std::endl;
#endif
			if (workQueueCounts[i] >= QUEUE_SIZE_PER_TASK_GLOBAL)
			{
				bstop = true;
				printf("the %d-th global queue is overflow! %d\n", i, workQueueCounts[i]);
				break;
			}
		}

		delete[] workQueueCounts;

		if (bstop)
			break;


		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal_balance");

		balanceTimer.start();
		balanceWorkList<BALANCE_THREADS, QUEUE_NTASKS, int2> << < 1, BALANCE_THREADS, 0, *(workerUnit->_stream) >> >
			(workerUnit->d_workQueues, workerUnit->d_workQueues2, workerUnit->d_workQueueCounts, QUEUE_SIZE_PER_TASK_GLOBAL, workerUnit->d_nWorkQueueElements, workerUnit->d_balanceSignal);

		balanceTimer.stop();

		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal_balance");

		elapsedBalance = balanceTimer.getElapsedMicroSec();
		elapsedBalanceTotal += elapsedBalance;
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
		printf("balance time (streaming): %f, total: %f\n", elapsedBalance, elapsedBalanceTotal);
#endif
		FROMGPU(&nActiveSplits, workerUnit->d_nWorkQueueElements, sizeof(unsigned int));
		FROMGPU(&bNeedBalance, workerUnit->d_balanceSignal, sizeof(unsigned int));
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
		printf("active splits num: %d\n", nActiveSplits);
#endif
		if (bNeedBalance == 1)
		{
			int2* t = workerUnit->d_workQueues;
			workerUnit->d_workQueues = workerUnit->d_workQueues2;
			workerUnit->d_workQueues2 = t;
		}

		nRuns++;
	}

	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal");

	FROMGPU(&nPairs, workerUnit->d_collisionPairIndex, sizeof(int));

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << "Overlapping OBB leaf nodes: " << nPairs << std::endl;
#endif

	unsigned long nCollidingPairs = nPairs;

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	static long iterationCount = 0;
	iterationCount++;
	size_t mem_free, mem_total;
	cudaMemGetInfo(&mem_free, &mem_total);

	std::cout << "Iteration " << iterationCount << ", before intersection test -- GPU memory: " << mem_free << " free of total " << mem_total << std::endl;
#endif


	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_Allocations");
	gProximityDetectionOutput* h_contacts = new gProximityDetectionOutput;

	h_contacts->valid = NULL;
	h_contacts->contactId = NULL;
	h_contacts->elems = NULL;
	h_contacts->distance = NULL;
	h_contacts->normal = NULL;
	h_contacts->point0 = NULL;
	h_contacts->point1 = NULL;
	h_contacts->contactType = NULL;

	//GPUMALLOC((void **)d_contacts, sizeof(struct DetectionOutput));
	GPUMALLOC((void **)&(h_contacts->valid), sizeof(bool)* nCollidingPairs * CollisionTestElementsSize);
	GPUMALLOC((void **)&(h_contacts->contactId), sizeof(int)* nCollidingPairs * CollisionTestElementsSize);
	GPUMALLOC((void **)&(h_contacts->distance), sizeof(double)* nCollidingPairs * CollisionTestElementsSize);
	GPUMALLOC((void **)&(h_contacts->elems), sizeof(int4)* nCollidingPairs * CollisionTestElementsSize);
	GPUMALLOC((void **)&(h_contacts->normal), sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);
	GPUMALLOC((void **)&(h_contacts->point0), sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);
	GPUMALLOC((void **)&(h_contacts->point1), sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);
	GPUMALLOC((void **)&(h_contacts->contactType), sizeof(gProximityContactType)* nCollidingPairs * CollisionTestElementsSize);

	GPUMEMSET(h_contacts->valid, 0, sizeof(bool)* nCollidingPairs * CollisionTestElementsSize);
	GPUMEMSET(h_contacts->contactId, 0, sizeof(int)* nCollidingPairs * CollisionTestElementsSize);
	GPUMEMSET(h_contacts->contactType, (int)COLLISION_INVALID, sizeof(gProximityContactType)* nCollidingPairs * CollisionTestElementsSize);

	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_Allocations");

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer");
	gProximityDetectionOutput* d_contacts = NULL;
	GPUMALLOC((void**)& d_contacts, sizeof(gProximityDetectionOutput));
	TOGPU(d_contacts, h_contacts, sizeof(gProximityDetectionOutput));
	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_ToGPU");


	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_Compute");

	CUDA_trianglePairIntersect(d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2,
		workerUnit->d_collisionPairs, nCollidingPairs, d_contacts, nCollidingPairs * CollisionTestElementsSize, alarmDistance, contactDistance);

	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_Compute");

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << "Overlapping pairs: " << nCollidingPairs << "; Max. contacts collected: " << nCollidingPairs * CollisionTestElementsSize << std::endl;
	std::cout << " valid array size = " << sizeof(bool)* nCollidingPairs * CollisionTestElementsSize << std::endl;
	std::cout << " distance array size = " << sizeof(double)* nCollidingPairs * CollisionTestElementsSize << std::endl;
	std::cout << " elems array size = " << sizeof(int2)* nCollidingPairs * CollisionTestElementsSize << std::endl;
	std::cout << " point0/point1/normal array size = " << sizeof(float3)* nCollidingPairs * CollisionTestElementsSize << std::endl;
#endif

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU");

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_HostAlloc");
	gProximityDetectionOutput h_detectedContacts;
	h_detectedContacts.valid = new bool[nCollidingPairs * CollisionTestElementsSize];
	h_detectedContacts.contactId = new int[nCollidingPairs * CollisionTestElementsSize];
	h_detectedContacts.distance = new double[nCollidingPairs * CollisionTestElementsSize];
	h_detectedContacts.elems = new int4[nCollidingPairs * CollisionTestElementsSize];
	h_detectedContacts.normal = new float3[nCollidingPairs * CollisionTestElementsSize];
	h_detectedContacts.point0 = new float3[nCollidingPairs * CollisionTestElementsSize];
	h_detectedContacts.point1 = new float3[nCollidingPairs * CollisionTestElementsSize];
	h_detectedContacts.contactType = new gProximityContactType[nCollidingPairs * CollisionTestElementsSize];

	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_HostAlloc");

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer");


	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_valid");
	//cudaMemcpyAsync(h_detectedContacts.valid, h_contacts->valid, sizeof(bool) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[0]);
	FROMGPU(h_detectedContacts.valid, h_contacts->valid, sizeof(bool)* nCollidingPairs * CollisionTestElementsSize);
	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_valid");

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_contactId");
	FROMGPU(h_detectedContacts.contactId, h_contacts->contactId, sizeof(int)* nCollidingPairs * CollisionTestElementsSize);
	//cudaMemcpyAsync(h_detectedContacts.contactId, h_contacts->contactId, sizeof(int) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[1]);
	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_contactId");

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_distance");
	FROMGPU(h_detectedContacts.distance, h_contacts->distance, sizeof(double)* nCollidingPairs * CollisionTestElementsSize);
	//cudaMemcpyAsync(h_detectedContacts.distance, h_contacts->distance, sizeof(double) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[2]);
	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_distance");

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_elems");
	FROMGPU(h_detectedContacts.elems, h_contacts->elems, sizeof(int4)* nCollidingPairs * CollisionTestElementsSize);
	//cudaMemcpyAsync(h_detectedContacts.elems, h_contacts->elems, sizeof(int4) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[3]);
	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_elems");

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_normal");
	FROMGPU(h_detectedContacts.normal, h_contacts->normal, sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);
	//cudaMemcpyAsync(h_detectedContacts.normal, h_contacts->normal, sizeof(float3) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[4]);
	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_normal");

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_point0");
	FROMGPU(h_detectedContacts.point0, h_contacts->point0, sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);
	//cudaMemcpyAsync(h_detectedContacts.point0, h_contacts->point0, sizeof(float3) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[5]);
	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_point0");

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_point1");
	FROMGPU(h_detectedContacts.point1, h_contacts->point1, sizeof(float3)* nCollidingPairs * CollisionTestElementsSize);
	//cudaMemcpyAsync(h_detectedContacts.point1, h_contacts->point1, sizeof(float3) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[6]);
	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_point1");

	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_contactType");
	FROMGPU(h_detectedContacts.contactType, h_contacts->contactType, sizeof(gProximityContactType)* nCollidingPairs * CollisionTestElementsSize);
	//cudaMemcpyAsync(h_detectedContacts.contactType, h_contacts->contactType, sizeof(gProximityContactType) * nCollidingPairs * CollisionTestElementsSize, cudaMemcpyDeviceToHost, stream[7]);
	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer_contactType");

	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU_Transfer");

	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_FromGPU");

	int nContacts = 0;
	for (unsigned int k = 0; k < nCollidingPairs * CollisionTestElementsSize; k++)
	{
		if (h_detectedContacts.valid[k])
		{
			nContacts++;
		}
	}

	//return nContacts;
	nIntersecting = nContacts;

	// *numberOfContacts = nContacts;
	/*
	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_ToHost");

	{
	*contactPoints = new gProximityDetectionOutput;
	(*contactPoints)->valid = new bool[nContacts];
	(*contactPoints)->contactId = new int[nContacts];
	(*contactPoints)->elems = new int4[nContacts];
	(*contactPoints)->point0 = new float3[nContacts];
	(*contactPoints)->point1 = new float3[nContacts];
	(*contactPoints)->normal = new float3[nContacts];
	(*contactPoints)->distance = new double[nContacts];
	(*contactPoints)->contactType = new gProximityContactType[nContacts];
	}

	long validContactIdx = 0;

	#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_CONTACT_POINTS
	std::cout << "=== Max. Contact points detected: " << nCollidingPairs * CollisionTestElementsSize  << " ===" << std::endl;
	#endif
	for (unsigned int k = 0; k < nCollidingPairs * CollisionTestElementsSize; k++)
	{
	if (h_detectedContacts.valid[k])
	{
	#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_CONTACT_POINTS
	std::cout << " * pos. " << k << ": ";
	std::cout << " type = ";
	if (h_detectedContacts.contactType[k] == COLLISION_LINE_LINE)
	std::cout << "LINE_LINE";
	else if (h_detectedContacts.contactType[k] == COLLISION_VERTEX_FACE)
	std::cout << "VERTEX_FACE";
	else
	std::cout << "INVALID";

	std::cout <<  ", elems = " << h_detectedContacts.elems[k].x << " - " << h_detectedContacts.elems[k].y << "; distance = " << h_detectedContacts.distance[k] << ", point0 = " << h_detectedContacts.point0[k].x << "," << h_detectedContacts.point0[k].y << "," << h_detectedContacts.point0[k].z <<
	", point1 = " << h_detectedContacts.point1[k].x << "," << h_detectedContacts.point1[k].y << "," << h_detectedContacts.point1[k].z << ", normal = " <<
	h_detectedContacts.normal[k].x << "," << h_detectedContacts.normal[k].y << "," << h_detectedContacts.normal[k].z;
	if (h_detectedContacts.distance[k] < 0.000001)
	std::cout << " --- DISTANCE UNDERRUN";

	std::cout << std::endl;
	#endif
	if (h_detectedContacts.distance[k] < 0.000001)
	(*contactPoints)->valid[validContactIdx] = false;
	else
	(*contactPoints)->valid[validContactIdx] = true;

	(*contactPoints)->contactId[validContactIdx] = h_detectedContacts.contactId[k];
	(*contactPoints)->elems[validContactIdx] = make_int4(h_detectedContacts.elems[k].x, h_detectedContacts.elems[k].y, h_detectedContacts.elems[k].z, h_detectedContacts.elems[k].w);
	(*contactPoints)->point0[validContactIdx] = make_float3(h_detectedContacts.point0[k].x, h_detectedContacts.point0[k].y, h_detectedContacts.point0[k].z);
	(*contactPoints)->point1[validContactIdx] = make_float3(h_detectedContacts.point1[k].x, h_detectedContacts.point1[k].y, h_detectedContacts.point1[k].z);
	(*contactPoints)->normal[validContactIdx] = make_float3(h_detectedContacts.normal[k].x, h_detectedContacts.normal[k].y, h_detectedContacts.normal[k].z);
	(*contactPoints)->distance[validContactIdx] = h_detectedContacts.distance[k];
	(*contactPoints)->contactType[validContactIdx] = h_detectedContacts.contactType[k];
	validContactIdx++;
	}
	}

	delete[] h_detectedContacts.valid;
	delete[] h_detectedContacts.contactId;
	delete[] h_detectedContacts.distance;
	delete[] h_detectedContacts.elems;
	delete[] h_detectedContacts.normal;
	delete[] h_detectedContacts.point0;
	delete[] h_detectedContacts.point1;
	delete[] h_detectedContacts.contactType;

	delete h_contacts;

	#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	std::cout << "Contact points generated: " << nContacts << std::endl;
	#endif

	GPUFREE(d_modelTransform1);
	GPUFREE(d_modelTransform2);
	GPUFREE(d_trVector1);
	GPUFREE(d_trVector2);
	#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE
	cudaMemGetInfo(&mem_free, &mem_total);
	std::cout << "Iteration " << iterationCount << ", after intersection test -- GPU memory: " << mem_free/1024/1024 << "GB free of total " << mem_total/1024/1024 << "GB" << std::endl;
	#endif



	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_TriTest_DataTransfer_ToHost");

	return nContacts;
	*/
}

#include <algorithm>
#include <map>

#define GPROXIMITY_DEBUG_BVH_TRAVERSAL
//#define OBBTREEGPU_BVH_TRAVERSE_VERBOSE_PAIR_DUMP

void ObbTreeGPU_BVH_Traverse(sofa::component::collision::OBBContainer *model1,
                             sofa::component::collision::OBBContainer *model2,
                             gProximityWorkerUnit* workerUnit,
                             double alarmDistance,
                             double contactDistance,
                             int& nIntersecting,
                             int workUnitId)
{
    OBBNode* obbTree1 = (OBBNode*)model1->obbTree;
    OBBNode* obbTree2 = (OBBNode*)model2->obbTree;

    GPUVertex* d_vertices1 = (GPUVertex*)model1->vertexPointer;
    GPUVertex* d_vertices2 = (GPUVertex*)model2->vertexPointer;
    uint3* d_triIndices1 = (uint3*)model1->triIdxPointer;
    uint3* d_triIndices2 = (uint3*)model2->triIdxPointer;

    Matrix3x3_d* d_modelTransform1 = NULL;
    Matrix3x3_d* d_modelTransform2 = NULL;
    float3* d_trVector1 = NULL;
    float3* d_trVector2 = NULL;

    Matrix3x3_d h_modelTransform1;
    Matrix3x3_d h_modelTransform2;

    h_modelTransform1.m_row[0].x = model1->modelTransform.m_R[0][0];
    h_modelTransform1.m_row[0].y = model1->modelTransform.m_R[0][1];
    h_modelTransform1.m_row[0].z = model1->modelTransform.m_R[0][2];
    h_modelTransform1.m_row[1].x = model1->modelTransform.m_R[1][0];
    h_modelTransform1.m_row[1].y = model1->modelTransform.m_R[1][1];
    h_modelTransform1.m_row[1].z = model1->modelTransform.m_R[1][2];
    h_modelTransform1.m_row[2].x = model1->modelTransform.m_R[2][0];
    h_modelTransform1.m_row[2].y = model1->modelTransform.m_R[2][1];
    h_modelTransform1.m_row[2].z = model1->modelTransform.m_R[2][2];

    h_modelTransform2.m_row[0].x = model2->modelTransform.m_R[0][0];
    h_modelTransform2.m_row[0].y = model2->modelTransform.m_R[0][1];
    h_modelTransform2.m_row[0].z = model2->modelTransform.m_R[0][2];
    h_modelTransform2.m_row[1].x = model2->modelTransform.m_R[1][0];
    h_modelTransform2.m_row[1].y = model2->modelTransform.m_R[1][1];
    h_modelTransform2.m_row[1].z = model2->modelTransform.m_R[1][2];
    h_modelTransform2.m_row[2].x = model2->modelTransform.m_R[2][0];
    h_modelTransform2.m_row[2].y = model2->modelTransform.m_R[2][1];
    h_modelTransform2.m_row[2].z = model2->modelTransform.m_R[2][2];

    float3 h_trVector1 = make_float3(model1->modelTransform.m_T[0], model1->modelTransform.m_T[1], model1->modelTransform.m_T[2]);
    float3 h_trVector2 = make_float3(model2->modelTransform.m_T[0], model2->modelTransform.m_T[1], model2->modelTransform.m_T[2]);

    GPUMALLOC((void**)&d_modelTransform1, sizeof(Matrix3x3_d));
    GPUMALLOC((void**)&d_modelTransform2, sizeof(Matrix3x3_d));

    GPUMALLOC((void**)&d_trVector1, sizeof(float3));
    GPUMALLOC((void**)&d_trVector2, sizeof(float3));

    TOGPU_ASYNC(d_trVector1, &h_trVector1, sizeof(float3), (*workerUnit->_stream));
    TOGPU_ASYNC(d_trVector2, &h_trVector2, sizeof(float3), (*workerUnit->_stream));

    TOGPU_ASYNC(d_modelTransform1, &h_modelTransform1, sizeof(Matrix3x3_d), (*workerUnit->_stream));
    TOGPU_ASYNC(d_modelTransform2, &h_modelTransform2, sizeof(Matrix3x3_d), (*workerUnit->_stream));
    cudaStreamSynchronize(*(workerUnit->_stream));

#ifdef GPROXIMITY_DEBUG_BVH_TRAVERSAL
    FROMGPU(&h_trVector1, d_trVector1, sizeof(float3));
    FROMGPU(&h_trVector2, d_trVector2, sizeof(float3));

    std::cout << " model1 translation for kernel = " << h_trVector1.x << "," << h_trVector1.y << "," << h_trVector1.z << std::endl;
    std::cout << " model2 translation for kernel = " << h_trVector2.x << "," << h_trVector2.y << "," << h_trVector2.z << std::endl;

    std::cout << " model1 translation re-read (in-place alloc) = " << h_trVector1.x << "," << h_trVector1.y << "," << h_trVector1.z << std::endl;
    std::cout << " model2 translation re-read (in-place alloc) = " << h_trVector2.x << "," << h_trVector2.y << "," << h_trVector2.z << std::endl;
#endif

    // init first work element:
    GPUMEMSET_ASYNC(workerUnit->d_workQueues, 0, sizeof(int2), (*workerUnit->_stream));
    GPUMEMSET_ASYNC(workerUnit->d_workQueueCounts, 0, sizeof(int)* QUEUE_NTASKS, (*workerUnit->_stream));
    GPUMEMSET_ASYNC(workerUnit->d_collisionPairIndex, 0, sizeof(int), (*workerUnit->_stream));

    // GPUMALLOC((void **)&d_obbOutputIndex, sizeof(int));
    GPUMEMSET_ASYNC(workerUnit->d_obbOutputIndex, 0, sizeof(int), (*workerUnit->_stream));

    unsigned int firstCount = 1;
    TOGPU_ASYNC(workerUnit->d_workQueueCounts, &firstCount, sizeof(unsigned int), (*workerUnit->_stream));

    unsigned int nPairs = 0;
    int nActiveSplits = 1;
    int nRuns = 0;
    int bNeedBalance = 0;

    TimerValue balanceTimer, traverseTimer;
    double elapsedBalance = 0, elapsedTraverse = 0, elapsedTraverseTotal = 0, elapsedBalanceTotal = 0;

    bool bstop = false;

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal");
#endif

    // GPUMEMSET_ASYNC(workerUnit->d_collisionLeafs, false, sizeof(bool)* workerUnit->_collisionPairCapacity, (*workerUnit->_stream));

    cudaStreamSynchronize((*workerUnit->_stream));

    while (nActiveSplits)
    {
        GPUMEMSET_ASYNC(workerUnit->d_collisionSync, 0, sizeof(int), (*workerUnit->_stream));
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
        sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal_traverse");
#endif
        traverseTimer.start();

        traverseTree<OBBNode, OBB, QUEUE_SIZE_PER_TASK, QUEUE_SIZE_PER_TASK_INIT, TRAVERSAL_THREADS> << < QUEUE_NTASKS, TRAVERSAL_THREADS, 0, *(workerUnit->_stream) >> >
            (obbTree1, d_vertices1, d_triIndices1, obbTree2, d_vertices2, d_triIndices2,
            workerUnit->d_workQueues, workerUnit->d_workQueueCounts, workerUnit->d_collisionSync, QUEUE_SIZE_PER_TASK_GLOBAL,
            workerUnit->d_collisionPairs, workerUnit->d_collisionPairIndex, workerUnit->d_collisionLeafs,
#ifdef GPROXIMITY_TRAVERSETREE_STORE_OVERLAPPING_OBB_PAIRS
            d_intersectingOBBs, workerUnit->d_obbOutputIndex, obbCount,
#endif
            d_modelTransform1, d_modelTransform2, d_trVector1, d_trVector2,
            alarmDistance /*contactDistance*/);

        cudaStreamSynchronize((*workerUnit->_stream));

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
        traverseTimer.stop();
        sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal_traverse");
#endif
        elapsedTraverse = traverseTimer.getElapsedMicroSec();
        elapsedTraverseTotal += elapsedTraverse;
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
        printf("traversal time (streaming, work unit %i): %f; total: %f\n", workUnitId, elapsedTraverse, elapsedTraverseTotal);
#endif

#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
        std::cout << " allocate workqueue counts array: " << QUEUE_NTASKS << " elements." << std::endl;
#endif
        unsigned int* workQueueCounts = new unsigned int[QUEUE_NTASKS];
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
        std::cout << " workqueue counts allocated; transfer from GPU memory" << std::endl;
#endif
        FROMGPU_ASYNC(workQueueCounts, workerUnit->d_workQueueCounts, sizeof(unsigned int)* QUEUE_NTASKS, (*workerUnit->_stream));
        //FROMGPU(workQueueCounts, workerUnit->d_workQueueCounts, sizeof(unsigned int)* QUEUE_NTASKS);
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
        std::cout << " transferred from GPU memory: " << sizeof(unsigned int)* QUEUE_NTASKS << std::endl;
#endif

        cudaStreamSynchronize((*workerUnit->_stream));

        for (int i = 0; i < QUEUE_NTASKS; ++i)
        {
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
            std::cout << " * " << workQueueCounts[i] << " >= " << QUEUE_SIZE_PER_TASK_GLOBAL << ": " << (workQueueCounts[i] >= QUEUE_SIZE_PER_TASK_GLOBAL) << std::endl;
#endif
            if (workQueueCounts[i] >= QUEUE_SIZE_PER_TASK_GLOBAL)
            {
                bstop = true;
                printf("the %d-th global queue is overflow! %d\n", i, workQueueCounts[i]);
                break;
            }
        }

        delete[] workQueueCounts;

        if (bstop)
            break;

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
        sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal_balance");
        balanceTimer.start();
#endif

        balanceWorkList<BALANCE_THREADS, QUEUE_NTASKS, int2> << < 1, BALANCE_THREADS, 0, *(workerUnit->_stream) >> >
            (workerUnit->d_workQueues, workerUnit->d_workQueues2, workerUnit->d_workQueueCounts, QUEUE_SIZE_PER_TASK_GLOBAL, workerUnit->d_nWorkQueueElements, workerUnit->d_balanceSignal);

        cudaStreamSynchronize((*workerUnit->_stream));

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
        balanceTimer.stop();
        sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal_balance");
        elapsedBalance = balanceTimer.getElapsedMicroSec();
        elapsedBalanceTotal += elapsedBalance;
        printf("balance time (streaming, work unit %i): %f, total: %f\n", workUnitId, elapsedBalance, elapsedBalanceTotal);
#endif
        FROMGPU_ASYNC(&nActiveSplits, workerUnit->d_nWorkQueueElements, sizeof(unsigned int), (*workerUnit->_stream));
        FROMGPU_ASYNC(&bNeedBalance, workerUnit->d_balanceSignal, sizeof(unsigned int), (*workerUnit->_stream));
#ifdef GPROXIMITY_DEBUG_BVH_TRAVERSAL
        printf("active splits num: %d\n", nActiveSplits);
#endif


        if (bNeedBalance == 1)
        {
            int2* t = workerUnit->d_workQueues;
            workerUnit->d_workQueues = workerUnit->d_workQueues2;
            workerUnit->d_workQueues2 = t;
        }

        nRuns++;
    }

    cudaStreamSynchronize(*(workerUnit->_stream));
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal");
#endif
    FROMGPU_ASYNC(&nPairs, workerUnit->d_collisionPairIndex, sizeof(int), (*workerUnit->_stream));

#ifdef GPROXIMITY_DEBUG_BVH_TRAVERSAL
    std::cout << "Traversal result: Overlapping OBB leaf nodes = " << nPairs << std::endl;
#endif
    workerUnit->_nCollidingPairs = nPairs;

    nIntersecting = nPairs;

    GPUFREE(d_trVector1);
    GPUFREE(d_trVector2);

    GPUFREE(d_modelTransform1);
    GPUFREE(d_modelTransform2);

#ifdef OBBTREEGPU_BVH_TRAVERSE_VERBOSE_PAIR_DUMP
    int2* h_collisionPairs = new int2[nPairs];
    bool* h_collisionLeafs = new bool[nPairs];

    FROMGPU(h_collisionPairs, workerUnit->d_collisionPairs, sizeof(int2)* nPairs);
    FROMGPU(h_collisionLeafs, workerUnit->d_collisionLeafs, sizeof(bool)* nPairs);

    std::vector<int> firstNodes;
    std::map<int, std::vector<int> > collisionPairs;
    for (unsigned int k = 0; k < nPairs; k++)
    {
        firstNodes.push_back(h_collisionPairs[k].x);
    }

    std::unique(firstNodes.begin(), firstNodes.end());

    for (std::vector<int>::const_iterator it = firstNodes.begin(); it != firstNodes.end(); it++)
    {
        collisionPairs.insert(std::make_pair((*it), std::vector<int>()));
    }

    for (unsigned int k = 0; k < nPairs; k++)
    {
        collisionPairs[h_collisionPairs[k].x].push_back(h_collisionPairs[k].y);
    }

    for (std::map<int, std::vector<int> >::iterator it = collisionPairs.begin(); it != collisionPairs.end(); it++)
    {
        std::cout << " -> " << it->first << ": ";
        std::sort(it->second.begin(), it->second.end());
        for (std::vector<int>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); it2++)
        {
            std::cout << *it2 << ";";
        }
        std::cout << std::endl;
    }

    std::cout << "Overlapping leaf OBB's (candidates for tri-pair checks): ";
    unsigned int totalOverlappingLeafs = 0;
    for (unsigned int k = 0; k < nPairs; k++)
    {
        std::cout << h_collisionLeafs[k] << ";";
        if (h_collisionLeafs[k] == true)
        {
            std::cout << h_collisionPairs[k].x << " - " << h_collisionPairs[k].y << ";";
            totalOverlappingLeafs++;
        }
    }
    std::cout << std::endl;
    std::cout << "Closely overlapping leafs = " << totalOverlappingLeafs << ", total overlapping leafs = " << nPairs << std::endl;

    delete[] h_collisionPairs;
    delete[] h_collisionLeafs;
#endif

#ifdef OBBTREEGPU_BVH_TRAVERSE_TRIANGLE_INTERSECTION_TEST
    GPUVertex* d_tfVertices1 = (GPUVertex*)model1->vertexTfPointer;
    GPUVertex* d_tfVertices2 = (GPUVertex*)model2->vertexTfPointer;

    std::vector<std::pair<int, int> > collidingTrianglePairs;
    unsigned int nIntersectingTris = CUDA_trianglePairCollide(d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2, workerUnit->d_collisionPairs, nPairs, collidingTrianglePairs);

    std::cout << "Potentially intersecting triangle pairs = " << nIntersectingTris << std::endl;
#endif
}

void ObbTreeGPU_BVH_Traverse_Streamed(sofa::component::collision::OBBContainer *model1,
                             sofa::component::collision::OBBContainer *model2,
                             gProximityGPUTransform* model1_transform,
                             gProximityGPUTransform* model2_transform,
                             gProximityWorkerUnit* workerUnit,
                             double alarmDistance,
                             double contactDistance,
                             int& nIntersecting,
                             int workUnitId)
{
	OBBNode* obbTree1 = (OBBNode*)model1->obbTree;
	OBBNode* obbTree2 = (OBBNode*)model2->obbTree;

	GPUVertex* d_vertices1 = (GPUVertex*)model1->vertexPointer;
	GPUVertex* d_vertices2 = (GPUVertex*)model2->vertexPointer;
	uint3* d_triIndices1 = (uint3*)model1->triIdxPointer;
	uint3* d_triIndices2 = (uint3*)model2->triIdxPointer;

	// init first work element:
    GPUMEMSET_ASYNC(workerUnit->d_workQueues, 0, sizeof(int2), (*workerUnit->_stream));
    GPUMEMSET_ASYNC(workerUnit->d_workQueueCounts, 0, sizeof(int)* QUEUE_NTASKS, (*workerUnit->_stream));
    GPUMEMSET_ASYNC(workerUnit->d_collisionPairIndex, 0, sizeof(int), (*workerUnit->_stream));

    GPUMEMSET_ASYNC(workerUnit->d_obbOutputIndex, 0, sizeof(int), (*workerUnit->_stream));

	unsigned int firstCount = 1;
    TOGPU_ASYNC(workerUnit->d_workQueueCounts, &firstCount, sizeof(unsigned int), (*workerUnit->_stream));

	unsigned int nPairs = 0;
	int nActiveSplits = 1;
	int nRuns = 0;
	int bNeedBalance = 0;

	TimerValue balanceTimer, traverseTimer;
	double elapsedBalance = 0, elapsedTraverse = 0, elapsedTraverseTotal = 0, elapsedBalanceTotal = 0;

	bool bstop = false;

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
	sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal");
#endif

    // GPUMEMSET_ASYNC(workerUnit->d_collisionLeafs, false, sizeof(bool)* workerUnit->_collisionPairCapacity, (*workerUnit->_stream));

    cudaStreamSynchronize((*workerUnit->_stream));

	while (nActiveSplits)
	{
        GPUMEMSET_ASYNC(workerUnit->d_collisionSync, 0, sizeof(int), (*workerUnit->_stream));
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal_traverse");
#endif
		traverseTimer.start();

        traverseTree<OBBNode, OBB, QUEUE_SIZE_PER_TASK, QUEUE_SIZE_PER_TASK_INIT, TRAVERSAL_THREADS> << < QUEUE_NTASKS, TRAVERSAL_THREADS, 0, *(workerUnit->_stream) >> >
            (obbTree1, d_vertices1, d_triIndices1, obbTree2, d_vertices2, d_triIndices2,
            workerUnit->d_workQueues, workerUnit->d_workQueueCounts, workerUnit->d_collisionSync, QUEUE_SIZE_PER_TASK_GLOBAL,
            workerUnit->d_collisionPairs, workerUnit->d_collisionPairIndex, workerUnit->d_collisionLeafs,
#ifdef GPROXIMITY_TRAVERSETREE_STORE_OVERLAPPING_OBB_PAIRS
            d_intersectingOBBs, workerUnit->d_obbOutputIndex, obbCount,
#endif
            model1_transform->modelOrientation, model2_transform->modelOrientation,
            model1_transform->modelTranslation, model2_transform->modelTranslation,
            alarmDistance /*contactDistance*/);

        cudaStreamSynchronize((*workerUnit->_stream));

        /*CUDA_SAFE_CALL(cudaEventRecord(*(workerUnit->_event), *(workerUnit->_stream)));
        CUDA_SAFE_CALL(cudaStreamWaitEvent(*(workerUnit->_stream), *(workerUnit->_event), 0));*/

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
        traverseTimer.stop();
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal_traverse");
#endif
		elapsedTraverse = traverseTimer.getElapsedMicroSec();
		elapsedTraverseTotal += elapsedTraverse;
#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
        printf("traversal time (streaming, work unit %i): %f; total: %f\n", workUnitId, elapsedTraverse, elapsedTraverseTotal);
#endif

#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
		std::cout << " allocate workqueue counts array: " << QUEUE_NTASKS << " elements." << std::endl;
#endif
        unsigned int* workQueueCounts = new unsigned int[QUEUE_NTASKS];
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
		std::cout << " workqueue counts allocated; transfer from GPU memory" << std::endl;
#endif
        //FROMGPU_ASYNC(workQueueCounts, workerUnit->d_workQueueCounts, sizeof(unsigned int)* QUEUE_NTASKS, (*workerUnit->_stream));
        FROMGPU(workQueueCounts, workerUnit->d_workQueueCounts, sizeof(unsigned int)* QUEUE_NTASKS);
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
		std::cout << " transferred from GPU memory: " << sizeof(unsigned int)* QUEUE_NTASKS << std::endl;
#endif

        cudaStreamSynchronize((*workerUnit->_stream));

        for (int i = 0; i < QUEUE_NTASKS; ++i)
		{
#ifdef GPROXIMITY_DEBUG_BVH_WORKQUEUE
			std::cout << " * " << workQueueCounts[i] << " >= " << QUEUE_SIZE_PER_TASK_GLOBAL << ": " << (workQueueCounts[i] >= QUEUE_SIZE_PER_TASK_GLOBAL) << std::endl;
#endif
			if (workQueueCounts[i] >= QUEUE_SIZE_PER_TASK_GLOBAL)
			{
				bstop = true;
				printf("the %d-th global queue is overflow! %d\n", i, workQueueCounts[i]);
				break;
			}
		}

        delete[] workQueueCounts;

		if (bstop)
			break;

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
		sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_BVHCollide_Traversal_balance");
		balanceTimer.start();
#endif

		balanceWorkList<BALANCE_THREADS, QUEUE_NTASKS, int2> << < 1, BALANCE_THREADS, 0, *(workerUnit->_stream) >> >
			(workerUnit->d_workQueues, workerUnit->d_workQueues2, workerUnit->d_workQueueCounts, QUEUE_SIZE_PER_TASK_GLOBAL, workerUnit->d_nWorkQueueElements, workerUnit->d_balanceSignal);

        cudaStreamSynchronize((*workerUnit->_stream));

        /*CUDA_SAFE_CALL(cudaEventRecord(*(workerUnit->_event), *(workerUnit->_stream)));
        CUDA_SAFE_CALL(cudaStreamWaitEvent(*(workerUnit->_stream), *(workerUnit->_event), 0));*/

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
		balanceTimer.stop();
		sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal_balance");
		elapsedBalance = balanceTimer.getElapsedMicroSec();
		elapsedBalanceTotal += elapsedBalance;
        printf("balance time (streaming, work unit %i): %f, total: %f\n", workUnitId, elapsedBalance, elapsedBalanceTotal);
#endif
        FROMGPU_ASYNC(&nActiveSplits, workerUnit->d_nWorkQueueElements, sizeof(unsigned int), (*workerUnit->_stream));
        FROMGPU_ASYNC(&bNeedBalance, workerUnit->d_balanceSignal, sizeof(unsigned int), (*workerUnit->_stream));
#ifdef GPROXIMITY_DEBUG_BVH_TRAVERSAL
		printf("active splits num: %d\n", nActiveSplits);
#endif


		if (bNeedBalance == 1)
		{
			int2* t = workerUnit->d_workQueues;
			workerUnit->d_workQueues = workerUnit->d_workQueues2;
			workerUnit->d_workQueues2 = t;
		}

        cudaStreamSynchronize(*(workerUnit->_stream));

		nRuns++;
	}

#ifdef GPROXIMITY_DEBUG_BVH_COLLIDE_TIMING
	sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_BVHCollide_Traversal");
#endif
    FROMGPU_ASYNC(&nPairs, workerUnit->d_collisionPairIndex, sizeof(int), (*workerUnit->_stream));

    cudaStreamSynchronize(*(workerUnit->_stream));

#ifdef GPROXIMITY_DEBUG_BVH_TRAVERSAL
	std::cout << "Traversal result: Overlapping OBB leaf nodes = " << nPairs << std::endl;
#endif
	workerUnit->_nCollidingPairs = nPairs;

    nIntersecting = nPairs;

#ifdef OBBTREEGPU_BVH_TRAVERSE_VERBOSE_PAIR_DUMP
	int2* h_collisionPairs = new int2[nPairs];
	bool* h_collisionLeafs = new bool[nPairs];

	FROMGPU(h_collisionPairs, workerUnit->d_collisionPairs, sizeof(int2)* nPairs);
	FROMGPU(h_collisionLeafs, workerUnit->d_collisionLeafs, sizeof(bool)* nPairs);

	std::vector<int> firstNodes;
	std::map<int, std::vector<int> > collisionPairs;
	for (unsigned int k = 0; k < nPairs; k++)
	{
		firstNodes.push_back(h_collisionPairs[k].x);
	}

	std::unique(firstNodes.begin(), firstNodes.end());

	for (std::vector<int>::const_iterator it = firstNodes.begin(); it != firstNodes.end(); it++)
	{
		collisionPairs.insert(std::make_pair((*it), std::vector<int>()));
	}

	for (unsigned int k = 0; k < nPairs; k++)
	{
		collisionPairs[h_collisionPairs[k].x].push_back(h_collisionPairs[k].y);
	}

	for (std::map<int, std::vector<int> >::iterator it = collisionPairs.begin(); it != collisionPairs.end(); it++)
	{
		std::cout << " -> " << it->first << ": ";
		std::sort(it->second.begin(), it->second.end());
		for (std::vector<int>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); it2++)
		{
			std::cout << *it2 << ";";
		}
		std::cout << std::endl;
	}

	std::cout << "Overlapping leaf OBB's (candidates for tri-pair checks): ";
	unsigned int totalOverlappingLeafs = 0;
	for (unsigned int k = 0; k < nPairs; k++)
	{
		std::cout << h_collisionLeafs[k] << ";";
		if (h_collisionLeafs[k] == true)
		{
			std::cout << h_collisionPairs[k].x << " - " << h_collisionPairs[k].y << ";";
			totalOverlappingLeafs++;
		}
	}
	std::cout << std::endl;
	std::cout << "Closely overlapping leafs = " << totalOverlappingLeafs << ", total overlapping leafs = " << nPairs << std::endl;

	delete[] h_collisionPairs;
	delete[] h_collisionLeafs;
#endif

#ifdef OBBTREEGPU_BVH_TRAVERSE_TRIANGLE_INTERSECTION_TEST
	GPUVertex* d_tfVertices1 = (GPUVertex*)model1->vertexTfPointer;
	GPUVertex* d_tfVertices2 = (GPUVertex*)model2->vertexTfPointer;

	std::vector<std::pair<int, int> > collidingTrianglePairs;
	unsigned int nIntersectingTris = CUDA_trianglePairCollide(d_tfVertices1, d_triIndices1, d_tfVertices2, d_triIndices2, workerUnit->d_collisionPairs, nPairs, collidingTrianglePairs);

	std::cout << "Potentially intersecting triangle pairs = " << nIntersectingTris << std::endl;
#endif
}

#include <boost/chrono.hpp>

//#define OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
void ObbTreeGPU_BVH_Traverse_Streamed_Batch(std::vector<sofa::component::collision::OBBContainer*>& models1,
                             std::vector<sofa::component::collision::OBBContainer*>& models2,
                             std::vector<gProximityGPUTransform*>& model1_transforms,
                             std::vector<gProximityGPUTransform*>& model2_transforms,
                             std::vector<gProximityWorkerUnit*>& workerUnits, std::vector<cudaStream_t>& workerStreams,
                             double alarmDistance,
                             double contactDistance,
                             std::vector<int>& nIntersecting,
                             int*& workQueueCounts,
                             cudaStream_t& mem_stream,
                             cudaEvent_t& startEvent,
                             cudaEvent_t& stopEvent,
                             cudaEvent_t& balanceEvent,
                             unsigned int numAssignedTraversals,
							 float& elapsedTime_workers,
							 std::vector<float>& elapsedTime_perWorker,
							 std::vector<std::pair<std::string, int64_t> > &elapsedTime_CPUStep,
							 std::vector<cudaEvent_t>& worker_startEvents, std::vector<cudaEvent_t>& worker_stopEvents)
{
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
    std::cout << "ObbTreeGPU_BVH_Traverse_Streamed_Batch(): " << workerUnits.size() << " BVH worker units, " << numAssignedTraversals << " have valid tasks assigned." << std::endl;
#endif

    if (numAssignedTraversals == 0)
    {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
        std::cout << " no valid tasks assigned, returning" << std::endl;
#endif
        return;
    }

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
    for (unsigned int k = 0; k < workerUnits.size(); k++)
    {
        std::cout << " - unit " << k << ": " << workerUnits[k]->_workerUnitIndex << std::endl;
        if (numAssignedTraversals > 0 && k < numAssignedTraversals)
        {
            std::cout << "    in assigned range: yes" << std::endl;
        }
        else
        {
            std::cout << "    in assigned range: no " << std::endl;
        }
    }
#endif

    if (numAssignedTraversals > 0)
        nIntersecting.resize(numAssignedTraversals);
    else
        nIntersecting.resize(workerUnits.size());

    std::vector<int> nActiveSplits;

    if (numAssignedTraversals > 0)
        nActiveSplits.resize(numAssignedTraversals);
    else
        nActiveSplits.resize(workerUnits.size());

    std::vector<unsigned int> bNeedBalancing;

    if (numAssignedTraversals > 0)
        bNeedBalancing.resize(numAssignedTraversals);
    else
        bNeedBalancing.resize(workerUnits.size());

    std::vector<bool> bStopFlags;

    if (numAssignedTraversals > 0)
        bStopFlags.resize(numAssignedTraversals);
    else
        bStopFlags.resize(workerUnits.size());

    std::vector<unsigned int> nPairs;

    if (numAssignedTraversals > 0)
        nPairs.resize(numAssignedTraversals);
    else
        nPairs.resize(workerUnits.size());

	if (numAssignedTraversals > 0)
		elapsedTime_perWorker.resize(numAssignedTraversals, 0.0f);
	else
		elapsedTime_perWorker.resize(workerUnits.size(), 0.0f);

    CUDA_SAFE_CALL(cudaEventRecord(startEvent));

    unsigned int numIterations = workerUnits.size();
    if (numAssignedTraversals > 0)
        numIterations = numAssignedTraversals;

	if (numIterations >= workerUnits.size())
	{
		std::cerr << "numIterations = " << numIterations << " >= workerUnits.size() = " << workerUnits.size() << "; cap to " << workerUnits.size() << std::endl;
		numIterations = workerUnits.size();
		std::cerr << "would need " << (numIterations / workerUnits.size()) << " iterations for processing all assigned tasks; mod = " << (numIterations % workerUnits.size()) << std::endl;
	}

	/*cudaEvent_t worker_startEvent, worker_stopEvent;
	CUDA_SAFE_CALL(cudaEventCreate(&worker_startEvent));
	CUDA_SAFE_CALL(cudaEventCreate(&worker_stopEvent));*/

	float elapsedTime_Traversal = 0.0f;
	float elapsedTime_Balance = 0.0f;

	unsigned long elapsedTime_cpu = 0;
	boost::chrono::high_resolution_clock::time_point start_step, stop_step;

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
    std::cout << " setup traversal units" << std::endl;
#endif

	unsigned int firstCount = 1;
	start_step = boost::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < numIterations; k++)
    {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
        std::cout << " - unit " << k << ": ";
#endif
		CUDA_SAFE_CALL(cudaEventRecord(worker_startEvents[k]));
        // init first work element:
        GPUMEMSET_ASYNC(workerUnits[k]->d_workQueues, 0, sizeof(int2), (*workerUnits[k]->_stream));
        GPUMEMSET_ASYNC(workerUnits[k]->d_workQueueCounts, 0, sizeof(int)* QUEUE_NTASKS, (*workerUnits[k]->_stream));
        GPUMEMSET_ASYNC(workerUnits[k]->d_collisionPairIndex, 0, sizeof(int), (*workerUnits[k]->_stream));

        GPUMEMSET_ASYNC(workerUnits[k]->d_obbOutputIndex, 0, sizeof(int), (*workerUnits[k]->_stream));
        TOGPU_ASYNC(workerUnits[k]->d_workQueueCounts, &firstCount, sizeof(unsigned int), (*workerUnits[k]->_stream));

		CUDA_SAFE_CALL(cudaEventRecord(worker_stopEvents[k]));
		CUDA_SAFE_CALL(cudaEventSynchronize(worker_stopEvents[k]));

		CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime_Traversal, worker_startEvents[k], worker_stopEvents[k]));
		elapsedTime_perWorker[k] += elapsedTime_Traversal;

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
        std::cout << " done." << std::endl;
#endif
    }
	stop_step = boost::chrono::high_resolution_clock::now();
	elapsedTime_CPUStep.push_back(std::make_pair("Setup", (stop_step - start_step).count()));

    unsigned int nRuns = 0;
    bool traversalsActive = true;

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
    std::cout << " first traverseTree loop" << std::endl;
#endif

	start_step = boost::chrono::high_resolution_clock::now();
	for (unsigned int k = 0; k < numIterations; k++)
	{
		OBBNode* obbTree1 = (OBBNode*)models1[k]->obbTree;
		OBBNode* obbTree2 = (OBBNode*)models2[k]->obbTree;

		GPUVertex* d_vertices1 = (GPUVertex*)models1[k]->vertexPointer;
		GPUVertex* d_vertices2 = (GPUVertex*)models2[k]->vertexPointer;

		uint3* d_triIndices1 = (uint3*)models1[k]->triIdxPointer;
		uint3* d_triIndices2 = (uint3*)models2[k]->triIdxPointer;

		CUDA_SAFE_CALL(cudaEventRecord(worker_startEvents[k]));

		traverseTree<OBBNode, OBB, QUEUE_SIZE_PER_TASK, QUEUE_SIZE_PER_TASK_INIT, TRAVERSAL_THREADS> << < QUEUE_NTASKS, TRAVERSAL_THREADS, 0, workerStreams[k] >> >
			(obbTree1, d_vertices1, d_triIndices1, obbTree2, d_vertices2, d_triIndices2,
			workerUnits[k]->d_workQueues, workerUnits[k]->d_workQueueCounts, workerUnits[k]->d_collisionSync, workerUnits[k]->_queueSizePerTaskGlobal /*QUEUE_SIZE_PER_TASK_GLOBAL*/,
			workerUnits[k]->d_collisionPairs, workerUnits[k]->d_collisionPairIndex, workerUnits[k]->d_collisionLeafs,
			model1_transforms[k]->modelOrientation, model2_transforms[k]->modelOrientation,
			model1_transforms[k]->modelTranslation, model2_transforms[k]->modelTranslation,
			alarmDistance);

		CUDA_SAFE_CALL(cudaEventRecord(worker_stopEvents[k]));
		CUDA_SAFE_CALL(cudaEventSynchronize(worker_stopEvents[k]));

		CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime_Traversal, worker_startEvents[k], worker_stopEvents[k]));
		elapsedTime_perWorker[k] += elapsedTime_Traversal;

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
        std::cout << " - unit " << k << std::endl;
#endif
    }
	stop_step = boost::chrono::high_resolution_clock::now();
	elapsedTime_CPUStep.push_back(std::make_pair("traverseTree_1", (stop_step - start_step).count()));

	start_step = boost::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < numIterations; k++)
    {
        if (!bStopFlags[k])
        {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
            std::cout << " - unit " << k << std::endl;
#endif
			CUDA_SAFE_CALL(cudaEventRecord(worker_startEvents[k]));

            balanceWorkList<BALANCE_THREADS, QUEUE_NTASKS, int2> << < 1, BALANCE_THREADS, 0, workerStreams[k] >> >
                (workerUnits[k]->d_workQueues, workerUnits[k]->d_workQueues2, workerUnits[k]->d_workQueueCounts,
				 workerUnits[k]->_queueSizePerTaskGlobal /*QUEUE_SIZE_PER_TASK_GLOBAL*/, workerUnits[k]->d_nWorkQueueElements, workerUnits[k]->d_balanceSignal);

			CUDA_SAFE_CALL(cudaEventRecord(worker_stopEvents[k]));
			CUDA_SAFE_CALL(cudaEventSynchronize(worker_stopEvents[k]));

			CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime_Balance, worker_startEvents[k], worker_stopEvents[k]));
			elapsedTime_perWorker[k] += elapsedTime_Balance;
        }
    }

	CUDA_SAFE_CALL(cudaEventRecord(balanceEvent, mem_stream));
	CUDA_SAFE_CALL(cudaStreamWaitEvent(mem_stream, balanceEvent, 0));

	stop_step = boost::chrono::high_resolution_clock::now();
	elapsedTime_CPUStep.push_back(std::make_pair("balanceWorkList_1", (stop_step - start_step).count()));

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
    std::cout << " eval balanceSignals/workQueueElements" << std::endl;
#endif


	start_step = boost::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < numIterations; k++)
    {
        if (!bStopFlags[k])
        {
			CUDA_SAFE_CALL(cudaEventRecord(worker_startEvents[k]));

            FROMGPU_ASYNC(&(nActiveSplits[k]), workerUnits[k]->d_nWorkQueueElements, sizeof(int), mem_stream);
            FROMGPU_ASYNC(&(bNeedBalancing[k]), workerUnits[k]->d_balanceSignal, sizeof(unsigned int), mem_stream);

			CUDA_SAFE_CALL(cudaEventRecord(worker_stopEvents[k]));
			CUDA_SAFE_CALL(cudaEventSynchronize(worker_stopEvents[k]));

			CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime_Traversal, worker_startEvents[k], worker_stopEvents[k]));
			elapsedTime_perWorker[k] += elapsedTime_Traversal;

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG            
			std::cout << "  - bNeedBalancing[" << k << "] = " << bNeedBalancing[k] << ", nActiveSplits[" << k << "] = " << nActiveSplits[k] << std::endl;
#endif
            if (bNeedBalancing[k] == 1)
            {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
                std::cout << "    needs balancing" << std::endl;
#endif
                int2* t = workerUnits[k]->d_workQueues;
                workerUnits[k]->d_workQueues = workerUnits[k]->d_workQueues2;
                workerUnits[k]->d_workQueues2 = t;
            }
        }
    }

    for (unsigned int k = 0; k < numIterations; k++)
    {
        if (nActiveSplits[k] > 0)
        {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
            std::cout << " active traversals still exist" << std::endl;
#endif
            traversalsActive = true;
            break;
        }
    }
	stop_step = boost::chrono::high_resolution_clock::now();
	elapsedTime_CPUStep.push_back(std::make_pair("workQueue_Check_1", (stop_step - start_step).count()));

    unsigned int loopCount = 0;
	std::stringstream iterStream;
    while (traversalsActive)
    {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
        std::cout << " synchronize traverseTree calls" << std::endl;
#endif
        for (unsigned int k = 0; k < workerUnits.size(); k++)
        {
			cudaError_t streamStatus = cudaStreamQuery(workerStreams[k]);
			while (streamStatus == cudaErrorNotReady)
			{
				cudaStreamSynchronize(workerStreams[k]);
				streamStatus = cudaStreamQuery(workerStreams[k]);
			}
        }

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
        std::cout << " check for overflowing work queues" << std::endl;
#endif


		start_step = boost::chrono::high_resolution_clock::now();
        for (unsigned int k = 0; k < numIterations; k++)
        {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
            std::cout << " - unit " << k << ", loopCount = " << loopCount << std::endl;
#endif
            loopCount++;

			CUDA_SAFE_CALL(cudaEventRecord(worker_startEvents[k]));

            FROMGPU_ASYNC(workQueueCounts, workerUnits[k]->d_workQueueCounts, sizeof(unsigned int) * QUEUE_NTASKS, mem_stream);

			CUDA_SAFE_CALL(cudaEventRecord(worker_stopEvents[k]));
			CUDA_SAFE_CALL(cudaEventSynchronize(worker_stopEvents[k]));

			CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime_Traversal, worker_startEvents[k], worker_stopEvents[k]));
			elapsedTime_perWorker[k] += elapsedTime_Traversal;

            for (int i = 0; i < QUEUE_NTASKS; ++i)
            {
				if (workQueueCounts[i] >= workerUnits[k]->_queueSizePerTaskGlobal /*QUEUE_SIZE_PER_TASK_GLOBAL*/)
                {
                    bStopFlags[k] = true;
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
                    std::cout << " WARNING: task " << k << " -- global queue " << i << " has overflown: " << workQueueCounts[i] << std::endl;
#endif
                    break;
                }
            }
        }

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
        std::cout << " balanceTree calls" << std::endl;
#endif

		iterStream << "WorkQueueCounts_" << loopCount;
		stop_step = boost::chrono::high_resolution_clock::now();
		elapsedTime_CPUStep.push_back(std::make_pair(iterStream.str(), (stop_step - start_step).count()));
		iterStream.str("");

		start_step = boost::chrono::high_resolution_clock::now();
        for (unsigned int k = 0; k < numIterations; k++)
        {
            if (!bStopFlags[k])
            {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
                std::cout << " - unit " << k << std::endl;
#endif
				CUDA_SAFE_CALL(cudaEventRecord(worker_startEvents[k]));

                balanceWorkList<BALANCE_THREADS, QUEUE_NTASKS, int2> << < 1, BALANCE_THREADS, 0, workerStreams[k] >> >
                    (workerUnits[k]->d_workQueues, workerUnits[k]->d_workQueues2, workerUnits[k]->d_workQueueCounts,
					workerUnits[k]->_queueSizePerTaskGlobal /*QUEUE_SIZE_PER_TASK_GLOBAL*/, workerUnits[k]->d_nWorkQueueElements, workerUnits[k]->d_balanceSignal);
            
				CUDA_SAFE_CALL(cudaEventRecord(worker_stopEvents[k]));
				CUDA_SAFE_CALL(cudaEventSynchronize(worker_stopEvents[k]));

				CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime_Balance, worker_startEvents[k], worker_stopEvents[k]));
				elapsedTime_perWorker[k] += elapsedTime_Balance;
			}
        }

		CUDA_SAFE_CALL(cudaEventRecord(balanceEvent, mem_stream));
		CUDA_SAFE_CALL(cudaStreamWaitEvent(mem_stream, balanceEvent, 0));

		iterStream << "balanceWorkList_" << loopCount;
		stop_step = boost::chrono::high_resolution_clock::now();
		elapsedTime_CPUStep.push_back(std::make_pair(iterStream.str(), (stop_step - start_step).count()));
		iterStream.str("");

		start_step = boost::chrono::high_resolution_clock::now();
        for (unsigned int k = 0; k < numIterations; k++)
        {
            if (!bStopFlags[k])
            {
				CUDA_SAFE_CALL(cudaEventRecord(worker_startEvents[k]));

				FROMGPU_ASYNC(&(nActiveSplits[k]), workerUnits[k]->d_nWorkQueueElements, sizeof(unsigned int), mem_stream);
				FROMGPU_ASYNC(&(bNeedBalancing[k]), workerUnits[k]->d_balanceSignal, sizeof(unsigned int), mem_stream);

				CUDA_SAFE_CALL(cudaEventRecord(worker_stopEvents[k]));
				CUDA_SAFE_CALL(cudaEventSynchronize(worker_stopEvents[k]));

				CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime_Traversal, worker_startEvents[k], worker_stopEvents[k]));
				elapsedTime_perWorker[k] += elapsedTime_Traversal;

                if (bNeedBalancing[k] == 1)
                {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
                    std::cout << "    needs balancing" << std::endl;
#endif
                    int2* t = workerUnits[k]->d_workQueues;
                    workerUnits[k]->d_workQueues = workerUnits[k]->d_workQueues2;
                    workerUnits[k]->d_workQueues2 = t;
                }
            }
        }

        traversalsActive = false;
        for (unsigned int k = 0; k < numIterations; k++)
        {
            if (nActiveSplits[k] > 0)
            {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
                std::cout << " active traversals still exist" << std::endl;
#endif
                traversalsActive = true;
                break;
            }
        }

        if (!traversalsActive)
        {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
            std::cout << " no more active traversals, break" << std::endl;
#endif
            break;
        }

        for (unsigned int k = 0; k < numIterations; k++)
        {
            OBBNode* obbTree1 = (OBBNode*) models1[k]->obbTree;
            OBBNode* obbTree2 = (OBBNode*) models2[k]->obbTree;

            GPUVertex* d_vertices1 = (GPUVertex*) models1[k]->vertexPointer;
            GPUVertex* d_vertices2 = (GPUVertex*) models2[k]->vertexPointer;

            uint3* d_triIndices1 = (uint3*) models1[k]->triIdxPointer;
            uint3* d_triIndices2 = (uint3*) models2[k]->triIdxPointer;

			CUDA_SAFE_CALL(cudaEventRecord(worker_startEvents[k]));

            traverseTree<OBBNode, OBB, QUEUE_SIZE_PER_TASK, QUEUE_SIZE_PER_TASK_INIT, TRAVERSAL_THREADS> << < QUEUE_NTASKS, TRAVERSAL_THREADS, 0, workerStreams[k] >> >
                (obbTree1, d_vertices1, d_triIndices1, obbTree2, d_vertices2, d_triIndices2,
				workerUnits[k]->d_workQueues, workerUnits[k]->d_workQueueCounts, workerUnits[k]->d_collisionSync, workerUnits[k]->_queueSizePerTaskGlobal /*QUEUE_SIZE_PER_TASK_GLOBAL*/,
                workerUnits[k]->d_collisionPairs, workerUnits[k]->d_collisionPairIndex, workerUnits[k]->d_collisionLeafs,
                model1_transforms[k]->modelOrientation, model2_transforms[k]->modelOrientation,
                model1_transforms[k]->modelTranslation, model2_transforms[k]->modelTranslation,
                alarmDistance);

			CUDA_SAFE_CALL(cudaEventRecord(worker_stopEvents[k]));
			CUDA_SAFE_CALL(cudaEventSynchronize(worker_stopEvents[k]));

			CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime_Traversal, worker_startEvents[k], worker_stopEvents[k]));
			elapsedTime_perWorker[k] += elapsedTime_Traversal;
        }

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
        std::cout << " synchronize traverseTree calls" << std::endl;
#endif
        for (unsigned int k = 0; k < numIterations; k++)
        {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
            std::cout << " - unit " << k << std::endl;
#endif
			cudaError_t streamStatus = cudaStreamQuery(workerStreams[k]);
			while (streamStatus == cudaErrorNotReady)
			{
				cudaStreamSynchronize(workerStreams[k]);
				streamStatus = cudaStreamQuery(workerStreams[k]);
			}
        }

		iterStream << "traverseTree_" << loopCount;
		stop_step = boost::chrono::high_resolution_clock::now();
		elapsedTime_CPUStep.push_back(std::make_pair(iterStream.str(), (stop_step - start_step).count()));
		iterStream.str("");

        nRuns++;
    }

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
    std::cout << " LAST synchronize traverseTree call, and result query" << std::endl;
#endif

	
	start_step = boost::chrono::high_resolution_clock::now();
	for (unsigned int k = 0; k < numIterations; k++)
    {
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
		std::cout << "  synchronize workerStreams[" << k << "]" << std::endl;
#endif
		cudaError_t streamStatus = cudaStreamQuery(workerStreams[k]);
		while (streamStatus == cudaErrorNotReady)
		{
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
			std::cout << "   stream " << k << " not ready..." << std::endl;
#endif
			cudaStreamSynchronize(workerStreams[k]);
			streamStatus = cudaStreamQuery(workerStreams[k]);
		}
#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
		std::cout << "  stream " << k << " ready!" << std::endl;
#endif
        
		cudaEventRecord(balanceEvent, workerStreams[k]);
        cudaStreamWaitEvent(mem_stream, balanceEvent, 0);

		CUDA_SAFE_CALL(cudaEventRecord(worker_startEvents[k]));

		FROMGPU_ASYNC(&(nPairs[k]), workerUnits[k]->d_collisionPairIndex, sizeof(int), mem_stream);

		CUDA_SAFE_CALL(cudaEventRecord(worker_stopEvents[k]));
		CUDA_SAFE_CALL(cudaEventSynchronize(worker_stopEvents[k]));

		CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime_Traversal, worker_startEvents[k], worker_stopEvents[k]));
		elapsedTime_perWorker[k] += elapsedTime_Traversal;

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
        std::cout << " * Traversal result " << k << ": Overlapping OBB leaf nodes = " << nPairs[k] << std::endl;
#endif
        workerUnits[k]->_nCollidingPairs = nPairs[k];
        nIntersecting[k] = nPairs[k];
    }

	stop_step = boost::chrono::high_resolution_clock::now();
	elapsedTime_CPUStep.push_back(std::make_pair("resultQuery", (stop_step - start_step).count()));
	iterStream.str("");

    CUDA_SAFE_CALL(cudaEventRecord(stopEvent));
    CUDA_SAFE_CALL(cudaEventSynchronize(stopEvent));
    
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime_workers, startEvent, stopEvent));

	/*CUDA_SAFE_CALL(cudaEventDestroy(worker_startEvent));
	CUDA_SAFE_CALL(cudaEventDestroy(worker_stopEvent));*/

#ifdef OBBTREEGPU_BVH_TRAVERSE_STREAMED_BATCH_DEBUG
    std::cout << "==> elapsed time for BVH traversals: " << elapsedTime_workers << " ms" << std::endl;
#endif
}
