#ifndef OBBTREEGPU_MULTITHREAD_TASKS_INL
#define OBBTREEGPU_MULTITHREAD_TASKS_INL

#include "ObbTreeGPU_MultiThread_Tasks.h"

#include <boost/thread.hpp>

using namespace sofa::component::collision;

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "ObbTreeGPUCollisionDetection_cuda.h"
#include "ObbTreeGPUTriangleCollision_cuda.h"

#include "ObbTreeGPUCollisionModel.h"

#include <cuda_runtime.h>

#include <sofa/defaulttype/Vec3Types.h>

#include <sofa/helper/AdvancedTimer.h>

using namespace sofa;
using namespace sofa::defaulttype;

struct gProximityWorkerResultPrivate
{
#ifdef USE_THRUST_HOST_VECTORS_IN_RESULTS
    thrust::host_vector<bool, std::allocator<bool> > h_valid;
    thrust::host_vector<int, std::allocator<int> > h_contactId;
    thrust::host_vector<double, std::allocator<double> > h_distance;
    thrust::host_vector<int4, std::allocator<int4> > h_elems;
    thrust::host_vector<float3, std::allocator<float3> > h_point0;
    thrust::host_vector<float3, std::allocator<float3> > h_point1;
    thrust::host_vector<float3, std::allocator<float3> > h_normal;
    thrust::host_vector<gProximityContactType, std::allocator<gProximityContactType> > h_gProximityContactType;
#else
    bool* h_valid;
    int* h_contactId;
    double* h_distance;
    int4* h_elems;
    float3* h_point0;
    float3* h_point1;
    float3* h_normal;
    gProximityContactType* h_gProximityContactType;
#endif
};

NarrowPhaseGPUTask::NarrowPhaseGPUTask(const Zyklio::MultiThreading::TaskStatus* status, unsigned int numStreamedWorkerResultBins, unsigned int minResultSize, unsigned int maxResultSize) :
	PoolTask(), m_numStreamedWorkerResultBins(numStreamedWorkerResultBins), m_minResultSize(minResultSize), m_maxResultSize(maxResultSize), m_totalResultsProduced(0)
{
    CUDA_SAFE_CALL(cudaEventCreate(&m_startEvent));
    CUDA_SAFE_CALL(cudaEventCreate(&m_stopEvent));

    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&m_memStream, cudaStreamNonBlocking));

    setupResultBins(m_minResultSize, m_maxResultSize);

    m_triTestStreams.resize(m_numStreamedWorkerResultBins);
    m_triTestEvents.resize(m_totalResultBinCount);

    for (unsigned int k = 0; k < m_numStreamedWorkerResultBins; k++)
    {
        CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&(m_triTestStreams[k]), cudaStreamNonBlocking));
    }

    for (unsigned int k = 0; k < m_totalResultBinCount; k++)
    {
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&(m_triTestEvents[k]), cudaEventDisableTiming));
    }

    m_localCudaEvents = true;
}

NarrowPhaseGPUTask::~NarrowPhaseGPUTask()
{
    if (m_localCudaEvents)
    {
        CUDA_SAFE_CALL(cudaEventDestroy(m_startEvent));
        CUDA_SAFE_CALL(cudaEventDestroy(m_stopEvent));

        CUDA_SAFE_CALL(cudaStreamDestroy(m_memStream));

        for (unsigned int k = 0; k < m_numStreamedWorkerResultBins; k++)
        {
            CUDA_SAFE_CALL(cudaStreamDestroy(m_triTestStreams[k]));
        }
        for(unsigned int k = 0; k < m_totalResultBinCount; k++)
        {
            CUDA_SAFE_CALL(cudaEventDestroy(m_triTestEvents[k]));
        }

        for (std::map<unsigned int, std::vector<gProximityWorkerResult*> >::iterator it = m_streamedWorkerResults.begin(); it != m_streamedWorkerResults.end(); it++)
        {
            std::vector<gProximityWorkerResult*>& workerResults = it->second;
            for (unsigned int k = 0; k < workerResults.size(); k++)
            {
                delete workerResults[k];
            }
            workerResults.clear();
        }
    }
}

void NarrowPhaseGPUTask::setupResultBins(unsigned int minSize, unsigned int maxSize)
{
    //std::cout << "NarrowPhaseGPUTask::setupResultBins(" << minSize << "," << maxSize << ")" << std::endl;
	//std::cout << " m_numStreamedWorkerResultBins = " << m_numStreamedWorkerResultBins << std::endl;
    for (unsigned int k = 0; k < m_numStreamedWorkerResultBins; k++)
    {
        m_streamedWorkerResults.insert(std::make_pair(k, std::vector<gProximityWorkerResult*>()));
    }

    unsigned int minMaxRatio = maxSize / minSize;
    unsigned int curBinSize = minSize;

    m_totalResultBinCount = 0;
    for (unsigned int k = 0; k < m_numStreamedWorkerResultBins; k++)
    {
        unsigned int numResultUnitsForBin = m_numStreamedWorkerResultBins * minMaxRatio;
		//std::cout << " - bin level " << k << ": curBinSize = " << curBinSize << ", minMaxRatio = " << minMaxRatio << ", numResultUnitsForBin = " << numResultUnitsForBin << std::endl;
        for (unsigned int l = 0; l < numResultUnitsForBin; l++)
        {
            gProximityWorkerResult* workerResult = new gProximityWorkerResult(curBinSize);
            workerResult->_resultIndex = l;
            workerResult->_resultBin = k;
            workerResult->_outputIndexPosition = m_totalResultBinCount;
            m_streamedWorkerResults[k].push_back(workerResult);
            m_totalResultBinCount++;
        }

        minMaxRatio /= 2;
        if (minMaxRatio == 0)
            minMaxRatio = 1;

        curBinSize *= 2;
    }
}

//#define OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
void NarrowPhaseGPUTask::distributeResultBins()
{
	//std::cout << "NarrowPhaseGPUTask::distributeResultBins()" << std::endl;

    m_freeResultBins.clear();
    m_freeResultBinSizes.clear();
    m_freeResultBinStartIndices.clear();

	//std::cout << "    m_freeResultBins.size() = " << m_freeResultBins.size() << std::endl;
	//std::cout << "    m_freeResultBinSizes.size() = " << m_freeResultBinSizes.size() << std::endl;
	//std::cout << "    m_freeResultBinStartIndices.size() = " << m_freeResultBinStartIndices.size() << std::endl;

    std::map<unsigned int, std::multimap<unsigned int, unsigned int> > requiredSlotsPerBinSize;
    std::map<unsigned int, int> bestDivSlotPositions;
    std::map<unsigned int, int> bestDivSizes;

    std::map<unsigned int, int> freeBinsPerBinLevel;
    std::map<unsigned int, std::multimap<unsigned int, unsigned int> > claimedResultBinsPerPairCheck;

    for (unsigned int k = 0; k < m_numStreamedWorkerResultBins; k++)
    {
        freeBinsPerBinLevel[k] = m_streamedWorkerResults[k].size();
    }

    for (unsigned int l = 0; l < m_taskWorkerUnits.size(); l++)
    {
        unsigned int curBinSize = m_minResultSize;
        int bestDivSlotPosition = -1;
        int bestDivSize = m_maxResultSize;

        gProximityWorkerUnit* workerUnit = m_taskWorkerUnits[l];
        OBBContainer* obbModel1 = m_containers_1[l];
        OBBContainer* obbModel2 = m_containers_2[l];

        // How much is needed actually? The worst-case estimate assumes that each pair of faces from models is in contact,
        // producing maximum number of 15 contacts per pair. This is not the case, unless the two models are identical, and are aligned in space!
        //unsigned int potentialResults = obbModel1->nTris * obbModel2->nTris * CollisionTestElementsSize;

        // So, let's use a more optimistic estimate...
        // unsigned int potentialResults = obbModel1->nTris * obbModel2->nTris;
        unsigned int potentialResults = m_intersectingTriPairCount[l] * CollisionTestElementsSize;

        for (unsigned int p = 0; p < m_numStreamedWorkerResultBins; p++)
        {
            unsigned int sizeDivBinK = potentialResults / curBinSize;
            unsigned int sizeModBinK = potentialResults % curBinSize;

            requiredSlotsPerBinSize[l].insert(std::make_pair(p, sizeDivBinK));

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
            std::cout<< " - task from worker unit " << workerUnit->_workerUnitIndex << std::endl;
            std::cout<< "    bin fitting: DivBin_" << p << " = " << sizeDivBinK << ", ModBin_" << p << " = " << sizeModBinK << std::endl;
    #endif

            if (m_streamedWorkerResults[p].size() > 0)
            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< "    required bins for size " << p << " = " << m_streamedWorkerResults[p][0]->_maxResults << ": " << sizeDivBinK << " + " << (sizeModBinK == 0 ? "0" : "1") << std::endl;
    #endif

                if (sizeDivBinK > 0 && sizeDivBinK < bestDivSize)
                {
                    bestDivSize = sizeDivBinK;
                    bestDivSlotPosition = p;

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout<< "    new minimum div slot size = " << bestDivSize << ", in slot = " << bestDivSlotPosition << std::endl;
    #endif
                }
            }

            curBinSize *= 2;
        }

		//std::cout << " recorded for task " << l << ": bestDivSlotPosition = " << bestDivSlotPosition << ", bestDivSize = " << bestDivSize << std::endl;
        bestDivSlotPositions.insert(std::make_pair(l, bestDivSlotPosition));
        bestDivSizes.insert(std::make_pair(l, bestDivSize));
    }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
    std::cout<< "   Size requirements for pair-checks: " << std::endl;

    for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator it = requiredSlotsPerBinSize.begin(); it != requiredSlotsPerBinSize.end(); it++)
    {
        std::cout<< "    task " << it->first << std::endl;
        std::multimap<unsigned int, unsigned int>& slot_requirements = it->second;
        for (std::multimap<unsigned int, unsigned int>::const_iterator slot_it = slot_requirements.begin(); slot_it != slot_requirements.end(); slot_it++)
        {
            std::cout<< "      - slot " << slot_it->first << ": " << slot_it->second << " bins required." << std::endl;
        }
    }
#endif

    for (unsigned int l = 0; l < m_taskWorkerUnits.size(); l++)
    {
        OBBContainer* obbModel1 = m_containers_1[l];
        OBBContainer* obbModel2 = m_containers_2[l];

        std::pair<OBBContainer,OBBContainer> triCheckPair;

        //unsigned int potentialResults = obbModel1->nTris * obbModel2->nTris * CollisionTestElementsSize;
        //unsigned int potentialResults = obbModel1->nTris * obbModel2->nTris;

        unsigned int potentialResults = m_intersectingTriPairCount[l] * CollisionTestElementsSize;

        unsigned int summedResultSize = 0;
        int bestDivSlotPosition = bestDivSlotPositions[l];
        int bestDivSize = bestDivSizes[l];

        bool triPairCheckAccepted = false;

        if (bestDivSlotPosition >= 0)
        {
            // Possibility 1: Does our test fit into the best slot size (divided) at once?
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
            std::cout<< "  Alternative 1: Check bestDivSlotPosition = " << bestDivSlotPosition << ", if it can fit our pair test alone." << std::endl;
            std::cout<< "   Best div slot location: slot = " << bestDivSlotPosition << ", bins required = " << bestDivSize << ", bins available = " << freeBinsPerBinLevel[bestDivSlotPosition] << std::endl;
    #endif

            if (bestDivSize <= freeBinsPerBinLevel[bestDivSlotPosition])
            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< "  bins required = " << bestDivSize << " <= free bins = " << freeBinsPerBinLevel[bestDivSlotPosition] << std::endl;
    #endif

                if (m_streamedWorkerResults[bestDivSlotPosition].size() > 0)
                {
                    unsigned int freeBinsInSlot = 0, blockedBinsInSlot = 0;
                    bool resultSizeSatisfied = false;
                    for (unsigned int n = 0; n < m_streamedWorkerResults[bestDivSlotPosition].size(); n++)
                    {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout<< "    - block " << n << " state: " << m_streamedWorkerResults[bestDivSlotPosition][n]->_blocked << ";";
    #endif

                        if (m_streamedWorkerResults[bestDivSlotPosition][n]->_blocked == false)
                        {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< " free, marking as blocked";
    #endif

                            claimedResultBinsPerPairCheck[bestDivSlotPosition].insert(std::make_pair(l, n));
                            freeBinsInSlot++;
                            summedResultSize += m_streamedWorkerResults[bestDivSlotPosition][n]->_maxResults;
                        }
                        else
                        {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< " blocked, continue";
    #endif

                            blockedBinsInSlot++;
                        }
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout<< std::endl;
                        std::cout<< "    summedResultSize step " << n << " = " << summedResultSize << std::endl;
    #endif

                        if (summedResultSize >= potentialResults)
                        {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< "    found enough bins in slot " << bestDivSlotPosition << " to fit tri-pair test results" << std::endl;
    #endif

                            triPairCheckAccepted = true;

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< "    blocking tagged bins" << std::endl;
    #endif

                            for (std::map<unsigned int, unsigned int>::const_iterator bit = claimedResultBinsPerPairCheck[bestDivSlotPosition].begin(); bit != claimedResultBinsPerPairCheck[bestDivSlotPosition].end(); bit++)
                            {
                                if (bit->first == l)
                                {
                                    if (m_streamedWorkerResults[bestDivSlotPosition][bit->second]->_blocked == false)
                                    {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                        std::cout<< "     - blocking bin = " << bit->second << "; remaining free on level " << bestDivSlotPosition << " = " << freeBinsPerBinLevel[bestDivSlotPosition] << std::endl;
    #endif

                                        m_streamedWorkerResults[bestDivSlotPosition][bit->second]->_blocked = true;
                                        freeBinsPerBinLevel[bestDivSlotPosition] -= 1;
                                    }
                                }
                            }

                            resultSizeSatisfied = true;
                            break;
                        }
                    }

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout<< "    Free bins marked as blocked: " << freeBinsInSlot << ", already blocked: " << blockedBinsInSlot << std::endl;
                    std::cout<< "    summedResultSize = " << summedResultSize << " for potentialResults = " << potentialResults << std::endl;
                    std::cout<< "    triPairCheckAccepted = " << triPairCheckAccepted << ", resultSizeSatisfied = " << resultSizeSatisfied << std::endl;

                    int remainingSize = potentialResults - summedResultSize;
                    if (remainingSize > 0)
                    {
                        std::cout<< "   remaining gap to potentialResults = " << remainingSize << std::endl;
                        std::cout<< "   need to fit additional bin(s) to accomodate" << std::endl;
                    }
    #endif

                    if (!resultSizeSatisfied)
                    {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout<< "    WARNING: Failed to block enough result bins to satisfy size requirements: summedResultSize = " << summedResultSize << " < " << potentialResults << std::endl;
                        std::cout<< "    Un-blocking already blocked result bins." << std::endl;
    #endif

                        triPairCheckAccepted = false;
                    }
                }

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< "    triPairCheckAccepted = " << triPairCheckAccepted;
    #endif
                if (triPairCheckAccepted)
                {
                    m_satisfiablePairChecks.insert(std::make_pair(l, true));
                    m_markedPairChecks.insert(std::make_pair(l, triCheckPair));

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout<< "; marking for execution.";
    #endif
                }
                else
                {
                    m_satisfiablePairChecks.insert(std::make_pair(l, false));
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout<< "; NOT MARKED FOR EXECUTION.";
    #endif
                }
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< std::endl;
    #endif
            }
        }

        // Possibility 2: Divide the test up amongst several bins
        if (!triPairCheckAccepted)
        {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
            std::cout<< "  Alternative 2: Trying to locate a bin combination that could fit." << std::endl;
            std::cout<< "  requiredSlotsPerBinSize.size() = " << requiredSlotsPerBinSize.size() << std::endl;
    #endif
            unsigned int diffToMaxBinSize = std::abs((int) m_maxResultSize - (int) bestDivSize);
            unsigned int diffToMinBinSize = std::abs((int) m_minResultSize - (int) bestDivSize);

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
            std::cout<< "   difference bestDivSize - max bin size = " << diffToMaxBinSize << ", bestDivSize - min bin size = " << diffToMinBinSize << std::endl;
    #endif
            if (diffToMaxBinSize < diffToMinBinSize)
            {
                bool resultSizeSatisfied = false;

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< "   starting from largest bin size down" << std::endl;
    #endif
                std::multimap<unsigned int, unsigned int>& requiredSlotsPerBinSize_curTask = requiredSlotsPerBinSize[l];
                for (std::multimap<unsigned int, unsigned int>::const_reverse_iterator it = requiredSlotsPerBinSize_curTask.rbegin(); it != requiredSlotsPerBinSize_curTask.rend(); it++)
                {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout<< "   --> alternative " << it->first << ": " << it->second << " bins * " << m_streamedWorkerResults[it->first][0]->_maxResults << " size = " << (it->second * m_streamedWorkerResults[it->first][0]->_maxResults) << std::endl;
    #endif

                    unsigned int freeBinsInSlot = 0, blockedBinsInSlot = 0;
                    if (freeBinsPerBinLevel[it->first] >= it->second)
                    {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout<< "      holds enough free bins to fulfill size requirement, trying to block: " << freeBinsPerBinLevel[it->first] << std::endl;
    #endif

                        for (unsigned int r = 0; r < m_streamedWorkerResults[it->first].size(); r++)
                        {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< "       - block " << r << "state: ";
    #endif

                            if (m_streamedWorkerResults[it->first][r]->_blocked == false)
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< " free, marking as blocked";
    #endif

                                freeBinsInSlot++;
                                claimedResultBinsPerPairCheck[it->first].insert(std::make_pair(l, r));
                                summedResultSize += m_streamedWorkerResults[it->first][r]->_maxResults;
                            }
                            else
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << "  blocked, skipping";
    #endif

                                blockedBinsInSlot++;
                            }
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE

                            std::cout<< std::endl;
                            std::cout<< "    summedResultSize step " << r << " = " << summedResultSize << std::endl;
    #endif

                            if (summedResultSize >= potentialResults)
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    found enough bins in slot " << it->first << " to fit tri-pair test results" << std::endl;
    #endif

                                triPairCheckAccepted = true;

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    blocking tagged bins" << std::endl;
    #endif

                                for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator it = claimedResultBinsPerPairCheck.begin(); it != claimedResultBinsPerPairCheck.end(); it++)
                                {
                                    std::multimap<unsigned int, unsigned int>& claimedResultBinInLevel = claimedResultBinsPerPairCheck[it->first];
                                    for (std::multimap<unsigned int, unsigned int>::const_iterator bit = claimedResultBinInLevel.begin(); bit != claimedResultBinInLevel.end(); bit++)
                                    {
                                        if (bit->first == l)
                                        {
                                            if (m_streamedWorkerResults[it->first][bit->second]->_blocked == false)
                                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout<< "     - blocking bin = " << bit->second << "; remaining free on level " << it->first << " = " << freeBinsPerBinLevel[it->first] << std::endl;
    #endif

                                                m_streamedWorkerResults[it->first][bit->second]->_blocked = true;
                                                freeBinsPerBinLevel[it->first] -= 1;
                                            }
                                        }
                                    }
                                }
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    Free bins marked as blocked: " << freeBinsInSlot << ", already blocked: " << blockedBinsInSlot << std::endl;
    #endif

                                resultSizeSatisfied = true;
                                break;
                            }
                        }
                    }
                    else
                    {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout<< "      slot " << it->first << " does not hold sufficient free bins: " << freeBinsPerBinLevel[it->first] << " out of " << it->second << std::endl;
                        std::cout<< "      trying to fit partially" << std::endl;
    #endif

                        for (unsigned int r = 0; r < m_streamedWorkerResults[it->first].size(); r++)
                        {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< "       - block " << r << " state: ";
    #endif

                            if (m_streamedWorkerResults[it->first][r]->_blocked == false)
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< " free, marking as blocked" << std::endl;
    #endif

                                freeBinsInSlot++;
                                claimedResultBinsPerPairCheck[it->first].insert(std::make_pair(l, r));
                                summedResultSize += m_streamedWorkerResults[it->first][r]->_maxResults;
                            }
                            else
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< " blocked, skipping" << std::endl;
    #endif

                                blockedBinsInSlot++;
                            }

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< "    summedResultSize step " << r << " = " << summedResultSize << std::endl;
    #endif

                            if (summedResultSize >= potentialResults)
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    found enough bins in slot " << it->first << " to fit tri-pair test results" << std::endl;
    #endif

                                triPairCheckAccepted = true;

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    blocking tagged bins" << std::endl;
    #endif

                                for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator it = claimedResultBinsPerPairCheck.begin(); it != claimedResultBinsPerPairCheck.end(); it++)
                                {
                                    std::multimap<unsigned int, unsigned int>& claimedResultBinInLevel = claimedResultBinsPerPairCheck[it->first];
                                    for (std::multimap<unsigned int, unsigned int>::const_iterator bit = claimedResultBinInLevel.begin(); bit != claimedResultBinInLevel.end(); bit++)
                                    {
                                        if (bit->first == l)
                                        {
                                            if (m_streamedWorkerResults[it->first][bit->second]->_blocked == false)
                                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout<< "     - blocking bin = " << bit->second << "; remaining free on level " << it->first << " = " << freeBinsPerBinLevel[it->first] << std::endl;
    #endif

                                                m_streamedWorkerResults[it->first][bit->second]->_blocked = true;
                                                freeBinsPerBinLevel[it->first] -= 1;
                                            }
                                        }
                                    }
                                }
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    Free bins marked as blocked: " << freeBinsInSlot << ", already blocked: " << blockedBinsInSlot << std::endl;
    #endif

                                resultSizeSatisfied = true;
                                break;
                            }
                        }
                    }

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout<< "   --> alternative " << it->first << " resultSizeSatisfied = " << resultSizeSatisfied << std::endl;
    #endif

                    if (resultSizeSatisfied)
                    {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout<< "      sufficient number of bins found, stop search" << std::endl;
    #endif

                        break;
                    }
                }

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< "    summedResultSize = " << summedResultSize << " for potentialResults = " << potentialResults << std::endl;
                std::cout<< "    triPairCheckAccepted = " << triPairCheckAccepted << ", resultSizeSatisfied = " << resultSizeSatisfied << std::endl;
    #endif

                if (!resultSizeSatisfied)
                {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout<< "    WARNING: Failed to block enough result bins to satisfy size requirements: summedResultSize = " << summedResultSize << " < " << potentialResults << std::endl;
                    std::cout<< "    Un-blocking already blocked result bins." << std::endl;
    #endif

                    triPairCheckAccepted = false;
                }
            }
            else
            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< "   starting from smallest bin size up" << std::endl;
    #endif

                bool resultSizeSatisfied = false;
                std::multimap<unsigned int, unsigned int>& requiredSlotsPerBinSize_curTask = requiredSlotsPerBinSize[l];
                for (std::multimap<unsigned int, unsigned int>::const_iterator it = requiredSlotsPerBinSize_curTask.begin(); it != requiredSlotsPerBinSize_curTask.end(); it++)
                {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout<< "   --> alternative " << it->first << ": " << it->second << " bins * " << m_streamedWorkerResults[it->first][0]->_maxResults << " size = " << (it->second * m_streamedWorkerResults[it->first][0]->_maxResults) << std::endl;
    #endif

                    unsigned int freeBinsInSlot = 0, blockedBinsInSlot = 0;
                    if (freeBinsPerBinLevel[it->first] >= it->second)
                    {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout<< "      holds enough free bins to fulfill size requirement, trying to block: " << freeBinsPerBinLevel[it->first] << std::endl;
    #endif

                        for (unsigned int r = 0; r < m_streamedWorkerResults[it->first].size(); r++)
                        {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< "       - block " << r << "state: ";
    #endif

                            if (m_streamedWorkerResults[it->first][r]->_blocked == false)
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< " free, marking as blocked";
    #endif

                                freeBinsInSlot++;
                                claimedResultBinsPerPairCheck[it->first].insert(std::make_pair(l, r));
                                summedResultSize += m_streamedWorkerResults[it->first][r]->_maxResults;
                            }
                            else
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout << " blocked, skipping";
    #endif

                                blockedBinsInSlot++;
                            }
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< std::endl;
                            std::cout<< "    summedResultSize step " << r << " = " << summedResultSize << std::endl;
    #endif

                            if (summedResultSize >= potentialResults)
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    found enough bins in slot " << it->first << " to fit tri-pair test results" << std::endl;
    #endif
                                triPairCheckAccepted = true;

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    blocking tagged bins" << std::endl;
    #endif

                                for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator it = claimedResultBinsPerPairCheck.begin(); it != claimedResultBinsPerPairCheck.end(); it++)
                                {
                                    std::multimap<unsigned int, unsigned int>& claimedResultBinInLevel = claimedResultBinsPerPairCheck[it->first];
                                    for (std::multimap<unsigned int, unsigned int>::const_iterator bit = claimedResultBinInLevel.begin(); bit != claimedResultBinInLevel.end(); bit++)
                                    {
                                        if (bit->first == l)
                                        {
                                            if (m_streamedWorkerResults[it->first][bit->second]->_blocked == false)
                                            {

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout<< "     - blocking bin = " << bit->second << "; remaining free on level " << it->first << " = " << freeBinsPerBinLevel[it->first] << std::endl;
    #endif

                                                m_streamedWorkerResults[it->first][bit->second]->_blocked = true;
                                                freeBinsPerBinLevel[it->first] -= 1;
                                            }
                                        }
                                    }
                                }
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    Free bins marked as blocked: " << freeBinsInSlot << ", already blocked: " << blockedBinsInSlot << std::endl;
    #endif

                                resultSizeSatisfied = true;
                                break;
                            }
                        }
                    }
                    else
                    {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout<< "      slot " << it->first << " does not hold sufficient free bins: " << freeBinsPerBinLevel[it->first] << " out of " << it->second << std::endl;
                        std::cout<< "      trying to fit partially" << std::endl;
    #endif

                        for (unsigned int r = 0; r < m_streamedWorkerResults[it->first].size(); r++)
                        {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< "       - block " << r << " state:";
    #endif

                            if (m_streamedWorkerResults[it->first][r]->_blocked == false)
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< " free, marking as blocked" << std::endl;
    #endif

                                freeBinsInSlot++;
                                claimedResultBinsPerPairCheck[it->first].insert(std::make_pair(l, r));
                                summedResultSize += m_streamedWorkerResults[it->first][r]->_maxResults;
                            }
                            else
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< " blocked, skipping" << std::endl;
    #endif

                                blockedBinsInSlot++;
                            }

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                            std::cout<< "    summedResultSize step " << r << " = " << summedResultSize << std::endl;
    #endif

                            if (summedResultSize >= potentialResults)
                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    found enough bins in slot " << it->first << " to fit tri-pair test results" << std::endl;
    #endif

                                triPairCheckAccepted = true;

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    blocking tagged bins" << std::endl;
    #endif

                                for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator it = claimedResultBinsPerPairCheck.begin(); it != claimedResultBinsPerPairCheck.end(); it++)
                                {
                                    std::multimap<unsigned int, unsigned int>& claimedResultBinInLevel = claimedResultBinsPerPairCheck[it->first];
                                    for (std::multimap<unsigned int, unsigned int>::const_iterator bit = claimedResultBinInLevel.begin(); bit != claimedResultBinInLevel.end(); bit++)
                                    {
                                        if (bit->first == l)
                                        {
                                            if (m_streamedWorkerResults[it->first][bit->second]->_blocked == false)
                                            {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                                std::cout<< "     - blocking bin = " << bit->second << "; remaining free on level " << it->first << " = " << freeBinsPerBinLevel[it->first] << std::endl;
    #endif

                                                m_streamedWorkerResults[it->first][bit->second]->_blocked = true;
                                                freeBinsPerBinLevel[it->first] -= 1;
                                            }
                                        }
                                    }
                                }

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                                std::cout<< "    Free bins marked as blocked: " << freeBinsInSlot << ", already blocked: " << blockedBinsInSlot << std::endl;
    #endif

                                resultSizeSatisfied = true;
                                break;
                            }
                        }
                    }
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout<< "   --> alternative " << it->first << " resultSizeSatisfied = " << resultSizeSatisfied << std::endl;
    #endif

                    if (resultSizeSatisfied)
                    {
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                        std::cout<< "      sufficient number of bins found, stop search" << std::endl;
    #endif
                        break;
                    }
                }

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< "    summedResultSize = " << summedResultSize << " for potentialResults = " << potentialResults << std::endl;
                std::cout<< "    triPairCheckAccepted = " << triPairCheckAccepted << ", resultSizeSatisfied = " << resultSizeSatisfied << std::endl;
    #endif

                if (!resultSizeSatisfied)
                {

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                    std::cout<< "    WARNING: Failed to block enough result bins to satisfy size requirements: summedResultSize = " << summedResultSize << " < " << potentialResults << std::endl;
                    std::cout<< "    Un-blocking already blocked result bins." << std::endl;
    #endif

                    triPairCheckAccepted = false;
                }
            }

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
            std::cout<< "    triPairCheckAccepted = " << triPairCheckAccepted;
    #endif

            if (triPairCheckAccepted)
            {
                m_satisfiablePairChecks.insert(std::make_pair(l, true));
                m_markedPairChecks.insert(std::make_pair(l, triCheckPair));

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< "; marking for execution.";
    #endif

            }
            else
            {
                m_satisfiablePairChecks.insert(std::make_pair(l, false));

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< "; NOT MARKED FOR EXECUTION.";
    #endif
            }

    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
            std::cout<< std::endl;
    #endif
        }
    }


    m_intersectingTriPairCount.resize(m_taskWorkerUnits.size());

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
    std::cout<< "=== Remaining free result bins per level after matching ===" << std::endl;
    for (std::map<unsigned int, int>::const_iterator it = freeBinsPerBinLevel.begin(); it != freeBinsPerBinLevel.end(); it++)
    {
        std::cout<< " - level " << it->first << ": " << it->second << " of " << m_streamedWorkerResults[it->first].size() << std::endl;
    }

    std::cout<< "=== triPairTests requirements === " << std::endl;
    for (std::map<unsigned int, bool>::const_iterator it = m_satisfiablePairChecks.begin(); it != m_satisfiablePairChecks.end(); it++)
    {
        std::cout<< " - check " << it->first << " satisfied = " << it->second << std::endl;
    }
#endif

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
    std::cout<< "=== bin occupancy for tasks ===" << std::endl;
#endif

    std::map<unsigned int, std::multimap<unsigned int, unsigned int> > blocksByTaskAndLevel;
    for (unsigned int l = 0; l < m_intersectingTriPairCount.size(); l++)
    {
        if (m_intersectingTriPairCount[l] > 0)
            blocksByTaskAndLevel.insert(std::make_pair(l, std::multimap<unsigned int, unsigned int>()));
    }

    for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::iterator bit = claimedResultBinsPerPairCheck.begin(); bit != claimedResultBinsPerPairCheck.end(); bit++)
    {
        std::multimap<unsigned int, unsigned int>& claimedResultBinsPerTask = bit->second;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
        std::cout<< " - block " << bit->first << ": ";
#endif

        for (std::multimap<unsigned int, unsigned int>::iterator rit = claimedResultBinsPerTask.begin(); rit != claimedResultBinsPerTask.end(); rit++)
        {
            unsigned int tmp1 = rit->first;
            unsigned int tmp2 = bit->first;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
            std::cout<< " task " << rit->first << ": bin " << rit->second << ";";
#endif

            blocksByTaskAndLevel[tmp1].insert(std::make_pair(tmp2, rit->second));
        }
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
        std::cout<< std::endl;
#endif
    }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
    std::cout<< "=== bin occupancy sorted by tasks ===" << std::endl;
#endif

    for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::const_iterator bit = blocksByTaskAndLevel.begin(); bit != blocksByTaskAndLevel.end(); bit++)
    {
        m_freeResultBins.insert(std::make_pair(bit->first, std::vector<gProximityWorkerResult*>()));
        m_freeResultBinSizes.insert(std::make_pair(bit->first, std::vector<std::pair<unsigned int, unsigned int> >()));
        m_freeResultBinStartIndices.insert(std::make_pair(bit->first, std::vector<unsigned int>()));
    }

    for (std::map<unsigned int, std::multimap<unsigned int, unsigned int> >::const_iterator bit = blocksByTaskAndLevel.begin(); bit != blocksByTaskAndLevel.end(); bit++)
    {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
        std::cout<< "  - task " << bit->first << ": " << std::endl;
#endif
        unsigned int curEnd = 0;
        const std::multimap<unsigned int, unsigned int>& blocksOfTask = bit->second;
        for (std::multimap<unsigned int, unsigned int>::const_iterator task_it = blocksOfTask.begin(); task_it != blocksOfTask.end(); task_it++)
        {
            m_freeResultBins[bit->first].push_back(m_streamedWorkerResults[task_it->first][task_it->second]);
            m_freeResultBinSizes[bit->first].push_back(std::make_pair(0, m_streamedWorkerResults[task_it->first][task_it->second]->_maxResults));
            m_freeResultBinStartIndices[bit->first].push_back(curEnd);
            curEnd += m_streamedWorkerResults[task_it->first][task_it->second]->_maxResults;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
            std::cout<< "   block " << task_it->first << ": bin " << task_it->second << "; " << std::endl;
            std::cout<< "    taskBins.size() = " << m_freeResultBins[bit->first].size() << std::endl;
            std::cout<< "    taskBinSizes.size() = " << m_freeResultBinSizes[bit->first].size() << std::endl;
            std::cout<< "    taskBinStartIndices.size() = " << m_freeResultBinStartIndices[bit->first].size() << std::endl;
            std::cout<< "    curEnd = " << curEnd << std::endl;
#endif
        }


#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
        std::vector<gProximityWorkerResult*>& taskBins_assigned = m_freeResultBins[bit->first];
        std::cout<< " taskBins vector re-read: " << taskBins_assigned.size() << std::endl;
        for (unsigned int k = 0; k < taskBins_assigned.size(); k++)
        {
            std::cout<< "  - entry " << k << ": bin slot " << taskBins_assigned[k]->_resultBin << ", index " << taskBins_assigned[k]->_resultBin << taskBins_assigned[k]->_maxResults << " max. results" << std::endl;
        }

        std::cout<< "    m_freeResultBins.size() = " << m_freeResultBins.size() << std::endl;
        std::cout<< "    m_freeResultBinSizes.size() = " << m_freeResultBinSizes.size() << std::endl;
        std::cout<< "    m_freeResultBinStartIndices.size() = " << m_freeResultBinStartIndices.size() << std::endl;
#endif
    }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
    std::cout<< "=== result bin assignments for triangle checks ===" << std::endl;
    for (std::map<unsigned int, std::vector<gProximityWorkerResult*> >::iterator it = m_freeResultBins.begin(); it != m_freeResultBins.end(); it++)
    {
        std::cout<< " - task " << it->first << ": " << it->second.size() << " result bins" << std::endl;

        std::vector<gProximityWorkerResult*>& resultBins = it->second;
        for (unsigned int k = 0; k < resultBins.size(); k++)
        {
            gProximityWorkerResult* res = resultBins.at(k);
            std::cout<< "    - result bin " << res->_resultIndex << " in bin slot " << res->_resultBin << ", capacity: " << res->_maxResults << std::endl;
        }
    }
#endif

}

void NarrowPhaseGPUTask::addTriangleCheck(gProximityWorkerUnit *workerUnit, OBBContainer *container1, OBBContainer *container2, ObbTreeGPUCollisionModel<Vec3Types>* model1, ObbTreeGPUCollisionModel<Vec3Types>* model2)
{
    m_taskWorkerUnits.push_back(workerUnit);
    m_containers_1.push_back(container1);
    m_containers_2.push_back(container2);
	m_models_1.push_back(model1);
	m_models_2.push_back(model2);
	m_modelNames_1.push_back(model1->getName());
    m_modelNames_2.push_back(model2->getName());
}

void NarrowPhaseGPUTask::setBVHTraversalResult(unsigned int traversalIdx, unsigned int result)
{
   // std::cout<< "setBVHTraversalResult(" << traversalIdx << "," << result << ")" << std::endl;
   // std::cout<< " m_intersectingTriPairCount.size() = " << m_intersectingTriPairCount.size() << std::endl;
    if (traversalIdx >= m_intersectingTriPairCount.size())
    {
        // std::cout<< " resize triPairCount vector to " << traversalIdx + 1 << std::endl;
        m_intersectingTriPairCount.resize(traversalIdx + 1, 0);
    }

    m_intersectingTriPairCount[traversalIdx] = result;
}

void NarrowPhaseGPUTask::clearWorkList()
{
    this->m_taskWorkerUnits.clear();
    this->m_containers_1.clear();
    this->m_containers_2.clear();

	m_models_1.clear();
	m_models_2.clear();

    for (int i = 0; i < m_modelNames_1.size(); i++) {
        std::cout << "MODELNAMES CHECKED: " << m_modelNames_1.at(i) << "---" << m_modelNames_2.at(i) << std::endl;
    }

    m_modelNames_1.clear();
    m_modelNames_2.clear();

    m_contactIDs.clear();
    m_contactDistances.clear();
    m_contactNormals.clear();
    m_contactPoints_0.clear();
    m_contactPoints_1.clear();
    m_contactElements.clear();
    m_contactElementsFeatures.clear();
    m_contactTypes.clear();

	m_elapsedTimePerTest.clear();

	m_totalResultsProduced = 0;
}

bool NarrowPhaseGPUTask::run(Zyklio::MultiThreading::WorkerThreadIface *thread)
{

    std::stringstream tmp;
    tmp << "NarrowPhaseGPUTask::run: " << this->getTaskID();
    sofa::helper::AdvancedTimer::stepBegin(tmp.str().c_str());

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
    std::cout<< "=== NarrowPhaseGPUTask::run(" << this->getTaskID() << ") ===" << std::endl;
    std::cout<< " results from BVH traversal: " << std::endl;
    for (unsigned int k = 0; k < m_intersectingTriPairCount.size(); k++)
        std::cout<< " - traversal task " << k << ": " << m_intersectingTriPairCount[k] << std::endl;

    std::cout<< " distribute result bins among task slots" << std::endl;
#endif
    distributeResultBins();

    //std::cout<< " result bin assignments per task after distribution: " << std::endl;
    for (std::map<unsigned int, std::vector<gProximityWorkerResult*> >::iterator it = m_freeResultBins.begin(); it != m_freeResultBins.end(); it++)
    {
        //std::cout<< " - task " << it->first << ": " << it->second.size() << " result bins" << std::endl;

        std::vector<gProximityWorkerResult*>& resultBins = it->second;
        for (unsigned int k = 0; k < resultBins.size(); k++)
        {
            gProximityWorkerResult* res = resultBins.at(k);
            //std::cout<< "    - result bin " << res->_resultIndex << " in bin " << res->_resultBin << ", capacity: " << res->_maxResults << std::endl;
        }
    }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
	std::cout<< "=============================" << std::endl;
    std::cout<< "==== run triangle checks ====" << std::endl;
	std::cout<< "=============================" << std::endl;
#endif

    m_triangleIntersectionResults.resize(m_taskWorkerUnits.size());
	m_elapsedTime = 0.0f;
    //m_elapsedTimePerTest.resize(m_taskWorkerUnits.size());

	// CUDA_SAFE_CALL(cudaSetDevice(0));

    for (unsigned int k = 0; k < m_taskWorkerUnits.size(); k++)
    {
        //std::cout<< " ==>> Task " << this->getTaskID() << " doing CHECK " << k << ": PAIR " << m_modelNames_1.at(k) << " - " << m_modelNames_2.at(k) << std::endl;

        std::vector<gProximityWorkerResult*>& freeResultBins_Task = m_freeResultBins[k];
        std::vector<std::pair<unsigned int, unsigned int> >& freeResultBinSizes_Task = m_freeResultBinSizes[k];
        std::vector<unsigned int>& freeResultBinStartIndices_Task = m_freeResultBinStartIndices[k];

        OBBContainer*& obbTree1 = m_containers_1[k];
        OBBContainer*& obbTree2 = m_containers_2[k];

		thread->startStepTimer();

		float elapsedTimeInCheck = 0.0f;
        ObbTreeGPU_TriangleIntersection_Streams_Batch(obbTree1, obbTree2,
                                                    m_taskWorkerUnits[k],
                                                    freeResultBins_Task,
                                                    freeResultBinSizes_Task,
                                                    freeResultBinStartIndices_Task,
                                                    m_triTestStreams,
                                                    m_triTestEvents,
                                                    m_memStream,
                                                    m_startEvent,
                                                    m_stopEvent,
                                                    m_alarmDistance, m_contactDistance,
                                                    m_triangleIntersectionResults[k],
													elapsedTimeInCheck
                                                 );

		m_elapsedTimePerTest.push_back(elapsedTimeInCheck);
		m_elapsedTime += elapsedTimeInCheck;

		thread->stopStepTimer();

        unsigned int totalResults = 0;
        unsigned int writtenResults = 0;
        for (unsigned int l = 0; l < freeResultBins_Task.size(); l++)
        {
            freeResultBins_Task[l]->_numResults = freeResultBins_Task[l]->h_outputIndex;

            totalResults += freeResultBins_Task[l]->_numResults;
            writtenResults += freeResultBins_Task[l]->h_outputIndex;

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
            std::cout<< "   * results in bin " << l << ": " << freeResultBins_Task[l]->_numResults << ", outputIndex = " << freeResultBins_Task[l]->h_outputIndex << std::endl;
#endif
        }

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
        std::cout<< "   total results = " << totalResults << ", outputIndices summed = " << writtenResults << std::endl;
#endif

#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
        for (unsigned int l = 0; l < freeResultBins_Task.size(); l++)
        {
            std::cout<< "    - bin " << l << " results: " << std::endl;
            gProximityWorkerResult* workerResultUnit = freeResultBins_Task[l];
            for (int m = 0; m < workerResultUnit->_numResults; m++)
            {
                std::cout<< "     - " << m << ": valid = " << workerResultUnit->d_ptr->h_valid[m]
                          << ", id = " << workerResultUnit->d_ptr->h_contactId[m]
                          << ", type = " << workerResultUnit->d_ptr->h_gProximityContactType[m]
                          << ", distance = " << workerResultUnit->d_ptr->h_distance[m]
                          << ", elements = " << workerResultUnit->d_ptr->h_elems[m].w << "," << workerResultUnit->d_ptr->h_elems[m].x << "," << workerResultUnit->d_ptr->h_elems[m].y << "," << workerResultUnit->d_ptr->h_elems[m].z
                          << ", point0 = " << workerResultUnit->d_ptr->h_point0[m].x << "," << workerResultUnit->d_ptr->h_point0[m].y << "," << workerResultUnit->d_ptr->h_point0[m].z
                          << ", point1 = " << workerResultUnit->d_ptr->h_point1[m].x << "," << workerResultUnit->d_ptr->h_point1[m].y << "," << workerResultUnit->d_ptr->h_point1[m].z
                          << ", normal = " << workerResultUnit->d_ptr->h_normal[m].x << "," << workerResultUnit->d_ptr->h_normal[m].y << "," << workerResultUnit->d_ptr->h_normal[m].z
                          << std::endl;
            }
        }
#endif
        if (totalResults > 0)
        {
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
			std::cout<< " ==>> In Task " << this->getTaskID() << " -- CHECK " << k << ": PAIR " << m_modelNames_1.at(k) << " - " << m_modelNames_2.at(k) << " PRODUCED CONTACTS; this gets stored at result position " << m_totalResultsProduced << std::endl;
            std::cout<< " assemble contact points out of totalResults = " << totalResults << std::endl;
#endif
			m_elapsedTimePerTest.push_back(elapsedTimeInCheck);

			m_contactIDs.insert(std::make_pair(m_totalResultsProduced, std::vector<int>()));
			m_contactDistances.insert(std::make_pair(m_totalResultsProduced, std::vector<double>()));
			m_contactNormals.insert(std::make_pair(m_totalResultsProduced, std::vector<Vector3>()));
			m_contactPoints_0.insert(std::make_pair(m_totalResultsProduced, std::vector<Vector3>()));
			m_contactPoints_1.insert(std::make_pair(m_totalResultsProduced, std::vector<Vector3>()));

			m_contactTypes.insert(std::make_pair(m_totalResultsProduced, std::vector<gProximityContactType>()));
			m_contactElements.insert(std::make_pair(m_totalResultsProduced, std::vector<std::pair<int, int> >()));
			m_contactElementsFeatures.insert(std::make_pair(m_totalResultsProduced, std::vector<std::pair<int, int> >()));

            const double maxContactDist = m_alarmDistance + (m_alarmDistance - m_contactDistance);
            const double maxContactDist2 = maxContactDist * maxContactDist;
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
            std::cout<< " maxContactDist = " << maxContactDist << ", maxContactDist2 = " << maxContactDist2 << std::endl;
#endif
            for (unsigned int l = 0; l < freeResultBins_Task.size(); l++)
            {
                gProximityWorkerResult* workerResultUnit = freeResultBins_Task[l];
#ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
                std::cout<< "  check workerResult " << l << ": " << workerResultUnit->_numResults << " results." << std::endl;
#endif
				if (workerResultUnit->_numResults < workerResultUnit->_maxResults)
				{
					for (unsigned int p = 0; p < workerResultUnit->_numResults; p++)
					{
						const float3& normalVec = workerResultUnit->d_ptr->h_normal[p];
						Vector3 contactNormal(normalVec.x, normalVec.y, normalVec.z);
						const double& contactDistance = workerResultUnit->d_ptr->h_distance[p];
						//std::cout<< "   - contact point " << p << ": distance = " << contactDistance << ", normal = " << contactNormal << std::endl;

						if (contactNormal.norm() >= 1e-06)
						{
							//std::cout<< "    contactNormal.norm() = " << contactNormal.norm() << " >= " << 1e-06 << std::endl;
							//std::cout<< "    contactNormal.norm2() = " << contactNormal.norm2() << " >= " << maxContactDist2 << ": " << (contactNormal.norm2() <= maxContactDist2) << std::endl;
							//std::cout<< "    " << std::fabs(contactDistance - m_contactDistance) << " < " << m_contactDistance << ": " << (std::fabs(contactDistance - m_contactDistance) < m_contactDistance) << std::endl;

							if (contactNormal.norm2() <= maxContactDist2 &&
								std::fabs(contactDistance - m_contactDistance) < m_contactDistance)
							{
								// std::cout << "   add new contact from bin " << l << ", outputPos " << p << std::endl;
								const int& contactId = workerResultUnit->d_ptr->h_contactId[p];

								m_contactIDs[m_totalResultsProduced].push_back(contactId);

								const float3& point0 = workerResultUnit->d_ptr->h_point0[p];
								m_contactPoints_0[m_totalResultsProduced].push_back(Vector3(point0.x, point0.y, point0.z));

								const float3& point1 = workerResultUnit->d_ptr->h_point1[p];
								m_contactPoints_1[m_totalResultsProduced].push_back(Vector3(point1.x, point1.y, point1.z));

								double detectionValue = contactNormal.norm();
								contactNormal /= detectionValue;
								detectionValue -= m_contactDistance;

								m_contactNormals[m_totalResultsProduced].push_back(contactNormal);
								m_contactDistances[m_totalResultsProduced].push_back(detectionValue);

								const gProximityContactType& contactType = workerResultUnit->d_ptr->h_gProximityContactType[p];
								m_contactTypes[m_totalResultsProduced].push_back(contactType);

								const int4& contactElems = workerResultUnit->d_ptr->h_elems[p];

								std::pair<int, int> contactElements;
								std::pair<int, int> contactElementsFeatures;
								if (contactType == COLLISION_LINE_LINE)
								{
									contactElements.first = contactElems.w * 3 + contactElems.y;
									contactElements.second = contactElems.x * 3 + contactElems.z;

									m_contactElements[m_totalResultsProduced].push_back(contactElements);

									contactElementsFeatures.first = contactElems.y;
									contactElementsFeatures.second = contactElems.z;

									m_contactElementsFeatures[m_totalResultsProduced].push_back(contactElementsFeatures);
								}
								else if (contactType == COLLISION_VERTEX_FACE)
								{
									if (contactElems.z == -1)
									{
										contactElements.first = contactElems.x * 3;
										contactElements.second = contactElems.w * 3 + contactElems.y;

										m_contactElements[m_totalResultsProduced].push_back(contactElements);

										contactElementsFeatures.first = contactElems.y;
										contactElementsFeatures.second = -1;

										m_contactElementsFeatures[m_totalResultsProduced].push_back(contactElementsFeatures);
									}
									else if (contactElems.y == -1)
									{
										contactElements.first = contactElems.x * 3;
										contactElements.second = contactElems.w * 3 + contactElems.z;

										m_contactElements[m_totalResultsProduced].push_back(contactElements);

										contactElementsFeatures.first = -1;
										contactElementsFeatures.second = contactElems.z;

										m_contactElementsFeatures[m_totalResultsProduced].push_back(contactElementsFeatures);
									}
								}
								else
								{
									// std::cout << "!!!! filtered out! by contact distance" << std::endl;
								}
							}
							else
							{
								// std::cout << "!!!! filtered out! by 10e-6" << std::endl;
							}
						}
					}
				}
				else
				{
                    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
					std::cout << "  workerResult " << l << ": " << workerResultUnit->_numResults << " results > " << workerResultUnit->_maxResults << " max. results; this should not happen!" << std::endl;
#endif
				}
            }
			m_totalResultsProduced++;
    #ifdef OBBTREEGPU_COLLISION_DETECTION_END_NARROW_PHASE_DEBUG_VERBOSE
			std::cout<< "=========================================================" << std::endl;
			std::cout<< "==> m_totalResultsProduced incremented; value = " << m_totalResultsProduced << std::endl;
			std::cout<< "=========================================================" << std::endl;
#endif
        }
    }

    /*
    std::cout<< " resulting contact points (filtered): " << std::endl;
    std::cout<< "  m_contactIDs.size() = " << m_contactIDs.size() << std::endl;
    std::cout<< "  m_contactDistances.size() = " << m_contactDistances.size() << std::endl;
    std::cout<< "  m_contactPoints_0.size() = " << m_contactPoints_0.size() << std::endl;
    std::cout<< "  m_contactPoints_1.size() = " << m_contactPoints_1.size() << std::endl;
    std::cout<< "  m_contactNormals.size() = " << m_contactNormals.size() << std::endl;
    std::cout<< "  m_contactTypes.size() = " << m_contactTypes.size() << std::endl;
    std::cout<< "  m_contactElements.size() = " << m_contactElements.size() << std::endl;
    std::cout<< "  m_contactElementsFeatures.size() = " << m_contactElementsFeatures.size() << std::endl;

    for (unsigned int k = 0; k < m_taskWorkerUnits.size(); k++)
    {
        std::cout<< " ==>> CHECK " << k << ": PAIR " << m_modelNames_1.at(k) << " - " << m_modelNames_2.at(k) << " CONTACT DATA" << std::endl;
        std::cout<< "   m_contactIDs[" << k << "].size() = " << m_contactIDs[k].size() << std::endl;
        std::cout<< "   m_contactDistances[" << k << "].size() = " << m_contactDistances[k].size() << std::endl;
        std::cout<< "   m_contactPoints_0[" << k << "].size() = " << m_contactPoints_0[k].size() << std::endl;
        std::cout<< "   m_contactPoints_1[" << k << "].size() = " << m_contactPoints_1[k].size() << std::endl;
        std::cout<< "   m_contactNormals[" << k << "].size() = " << m_contactNormals[k].size() << std::endl;
        std::cout<< "   m_contactTypes[" << k << "].size() = " << m_contactTypes[k].size() << std::endl;
        std::cout<< "   m_contactElements[" << k << "].size() = " << m_contactElements[k].size() << std::endl;
        std::cout<< "   m_contactElementsFeatures[" << k << "].size() = " << m_contactElementsFeatures[k].size() << std::endl;
    }

    std::cout<< " clearing result bin distribution containers" << std::endl;
    */

    m_freeResultBins.clear();
    m_freeResultBinSizes.clear();
    m_freeResultBinStartIndices.clear();

    //std::cout<< " removing blocked flags from worker results" << std::endl;

    unsigned int outputIndexReset = 0;
    for (std::map<unsigned int, std::vector<gProximityWorkerResult*> >::iterator it = m_streamedWorkerResults.begin(); it != m_streamedWorkerResults.end(); it++)
    {
        std::vector<gProximityWorkerResult*>& workerResults = it->second;
        for (unsigned int u = 0; u < workerResults.size(); u++)
        {
            workerResults[u]->_blocked = false;
            TOGPU_ASYNC(workerResults[u]->d_outputIndex, &outputIndexReset, sizeof(unsigned int), m_memStream);
            workerResults[u]->h_outputIndex = outputIndexReset;
        }
    }

    //std::cout<< "=== clear triPairCount vector ===" << std::endl;
    m_intersectingTriPairCount.clear();

    sofa::helper::AdvancedTimer::stepBegin(tmp.str().c_str());

    return true;
}

ObbTreeGPUCollisionModel<Vec3Types>* NarrowPhaseGPUTask::getModel1(unsigned int idx)
{
	if (idx < m_models_1.size())
		return m_models_1[idx];

	return NULL;
}

ObbTreeGPUCollisionModel<Vec3Types>* NarrowPhaseGPUTask::getModel2(unsigned int idx)
{
	if (idx < m_models_2.size())
		return m_models_2[idx];

	return NULL;
}

const std::string NarrowPhaseGPUTask::getModelName1(unsigned int idx) const
{
    if (idx < m_modelNames_1.size())
        return m_modelNames_1[idx];

    return std::string("");
}

const std::string NarrowPhaseGPUTask::getModelName2(unsigned int idx) const
{
    if (idx < m_modelNames_2.size())
        return m_modelNames_2[idx];

    return std::string("");
}

BVHTraversalTask::BVHTraversalTask(const Zyklio::MultiThreading::TaskStatus* status, unsigned int numWorkerUnits):
                                   PoolTask(), m_numWorkerUnits(numWorkerUnits), m_numAssignedTraversals(0), m_traversalCalls(0)
{
    std::cout<< "BVHTraversalTask::BVHTraversalTask(): " << m_numWorkerUnits << " local worker units." << std::endl;

    CUDA_SAFE_CALL(cudaEventCreate(&m_startEvent));
    CUDA_SAFE_CALL(cudaEventCreate(&m_stopEvent));
    CUDA_SAFE_CALL(cudaEventCreate(&m_balanceEvent));

    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&m_memStream, cudaStreamNonBlocking));

    m_workerStreams.resize(m_numWorkerUnits);
    m_workerEvents.resize(m_numWorkerUnits);

    for (unsigned int k = 0; k < m_numWorkerUnits; k++)
    {
        gProximityWorkerUnit* workerUnit = new gProximityWorkerUnit();

        CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&(m_workerStreams[k]), cudaStreamNonBlocking));
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&(m_workerEvents[k]), cudaEventDisableTiming));

        workerUnit->_workerUnitIndex = k;
        workerUnit->_stream = &(m_workerStreams[k]);
        m_taskWorkerUnits.push_back(workerUnit);

        std::cout<< " - allocated workerUnit " << workerUnit->_workerUnitIndex << std::endl;
    }

    std::cout<< " task worker units count = " << m_taskWorkerUnits.size() << std::endl;
    for (unsigned int k = 0; k < m_taskWorkerUnits.size(); k++)
    {
        std::cout<< " - worker unit mem-address = " << m_taskWorkerUnits.at(k) << ", workerUnitIndex from vector = " << m_taskWorkerUnits[k]->_workerUnitIndex << std::endl;
    }

    m_workQueueCounts = new int[QUEUE_NTASKS];

    m_localCudaEvents = true;
}

BVHTraversalTask::BVHTraversalTask(const Zyklio::MultiThreading::TaskStatus* status, unsigned int numSmallWorkerUnits, unsigned long sizeSmallWorkerUnit, unsigned int numLargeWorkerUnits, unsigned long sizeLargeWorkerUnit): 
PoolTask(), m_numAssignedTraversals(0), m_traversalCalls(0), m_numSmallWorkerUnits(numSmallWorkerUnits), m_numLargeWorkerUnits(numLargeWorkerUnits), m_smallWorkerUnitSize(sizeSmallWorkerUnit), m_largeWorkerUnitSize(sizeLargeWorkerUnit)
{
	m_numWorkerUnits = numSmallWorkerUnits + numLargeWorkerUnits;

	CUDA_SAFE_CALL(cudaEventCreate(&m_startEvent));
	CUDA_SAFE_CALL(cudaEventCreate(&m_stopEvent));
	CUDA_SAFE_CALL(cudaEventCreate(&m_balanceEvent));

	CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&m_memStream, cudaStreamNonBlocking));

	m_workerStreams.resize(m_numWorkerUnits);
	m_workerEvents.resize(m_numWorkerUnits);

	m_workerStartEvents.resize(m_numWorkerUnits);
	m_workerStopEvents.resize(m_numWorkerUnits);

	unsigned int numWorkerUnitsCreated = 0;
	for (unsigned int k = 0; k < m_numSmallWorkerUnits; k++)
	{
		gProximityWorkerUnit* workerUnit = new gProximityWorkerUnit(m_smallWorkerUnitSize, 128, 1000);

		CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&(m_workerStreams[numWorkerUnitsCreated]), cudaStreamNonBlocking));
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&(m_workerEvents[numWorkerUnitsCreated]), cudaEventDisableTiming));

		CUDA_SAFE_CALL(cudaEventCreate(&(m_workerStartEvents[numWorkerUnitsCreated])));
		CUDA_SAFE_CALL(cudaEventCreate(&(m_workerStopEvents[numWorkerUnitsCreated])));

		workerUnit->_workerUnitIndex = numWorkerUnitsCreated;
		workerUnit->_stream = &(m_workerStreams[numWorkerUnitsCreated]);
		m_taskWorkerUnits.push_back(workerUnit);

		//std::cout << " - allocated small workerUnit " << workerUnit->_workerUnitIndex << std::endl;
		numWorkerUnitsCreated++;
	}

	for (unsigned int k = 0; k < m_numLargeWorkerUnits; k++)
	{
		gProximityWorkerUnit* workerUnit = new gProximityWorkerUnit(m_largeWorkerUnitSize, 128, 1000);

		CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&(m_workerStreams[numWorkerUnitsCreated]), cudaStreamNonBlocking));
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&(m_workerEvents[numWorkerUnitsCreated]), cudaEventDisableTiming));

		CUDA_SAFE_CALL(cudaEventCreate(&(m_workerStartEvents[numWorkerUnitsCreated])));
		CUDA_SAFE_CALL(cudaEventCreate(&(m_workerStopEvents[numWorkerUnitsCreated])));

		workerUnit->_workerUnitIndex = numWorkerUnitsCreated;
		workerUnit->_stream = &(m_workerStreams[numWorkerUnitsCreated]);
		m_taskWorkerUnits.push_back(workerUnit);

		//std::cout  << " - allocated small workerUnit " << workerUnit->_workerUnitIndex << std::endl;
		numWorkerUnitsCreated++;
	}

	/*
	std::cout << " task worker units count = " << m_taskWorkerUnits.size() << std::endl;
	for (unsigned int k = 0; k < m_taskWorkerUnits.size(); k++)
	{
		std::cout << " - worker unit mem-address = " << m_taskWorkerUnits.at(k) << ", workerUnitIndex from vector = " << m_taskWorkerUnits[k]->_workerUnitIndex << std::endl;
	}
	*/

	m_workQueueCounts = new int[QUEUE_NTASKS];

	m_localCudaEvents = true;
}

BVHTraversalTask::~BVHTraversalTask()
{
	//std::cout << "BVHTraversalTask::~BVHTraversalTask(" << this->m_taskId << ")" << std::endl;
    if (m_localCudaEvents)
    {
        CUDA_SAFE_CALL(cudaEventDestroy(m_startEvent));
        CUDA_SAFE_CALL(cudaEventDestroy(m_stopEvent));
        CUDA_SAFE_CALL(cudaEventDestroy(m_balanceEvent));

        for (unsigned int k = 0; k < m_numWorkerUnits; k++)
        {
            CUDA_SAFE_CALL(cudaStreamDestroy(m_workerStreams[k]));
            CUDA_SAFE_CALL(cudaEventDestroy(m_workerEvents[k]));

			CUDA_SAFE_CALL(cudaEventDestroy(m_workerStartEvents[k]));
			CUDA_SAFE_CALL(cudaEventDestroy(m_workerStopEvents[k]));

            gProximityWorkerUnit* workerUnit = m_taskWorkerUnits[k];
            if (workerUnit != NULL)
            {
                delete workerUnit;
            }
        }

        m_taskWorkerUnits.clear();

        CUDA_SAFE_CALL(cudaStreamDestroy(m_memStream));

        delete[] m_workQueueCounts;
    }
}

BVHTraversalTask::BVHTraversalTask(const BVHTraversalTask& other): PoolTask(other)
{
    if (this != &other)
    {
        m_results = other.m_results;

        for (unsigned int i = 0; i < other.m_containers_1.size(); i++)
            this->m_containers_1.push_back(other.m_containers_1[i]);

        for (unsigned int i = 0; i < other.m_containers_2.size(); i++)
            this->m_containers_2.push_back(other.m_containers_2[i]);

        for (unsigned int i = 0; i < other.m_transforms_1.size(); i++)
            this->m_transforms_1.push_back(other.m_transforms_1[i]);

        for (unsigned int i = 0; i < other.m_transforms_2.size(); i++)
            this->m_transforms_2.push_back(other.m_transforms_2[i]);
    }
}

BVHTraversalTask& BVHTraversalTask::operator=(const BVHTraversalTask& other)
{
    if (this != &other)
    {
        PoolTask::operator =(other);
        m_results = other.m_results;

        for (unsigned int i = 0; i < other.m_containers_1.size(); i++)
            this->m_containers_1.push_back(other.m_containers_1[i]);

        for (unsigned int i = 0; i < other.m_containers_2.size(); i++)
            this->m_containers_2.push_back(other.m_containers_2[i]);

        for (unsigned int i = 0; i < other.m_transforms_1.size(); i++)
            this->m_transforms_1.push_back(other.m_transforms_1[i]);

        for (unsigned int i = 0; i < other.m_transforms_2.size(); i++)
            this->m_transforms_2.push_back(other.m_transforms_2[i]);
    }
    return *this;
}

void BVHTraversalTask::addTraversal(OBBContainer* model1, OBBContainer* model2, gProximityGPUTransform* transform1, gProximityGPUTransform* transform2, ObbTreeGPUCollisionModel<Vec3Types>* obb_model1, ObbTreeGPUCollisionModel<Vec3Types>* obb_model2)
{
    // TODO: sort by model size (small / large worker units)
#ifdef BVHTRAVERSALTASK_DEBUG
	std::cout << "BVHTraversalTask::addTraversal(" << this->getTaskID() << "): " << model1->nBVs << " OBBs in model1, " << model2->nBVs << " OBBs in model2" << std::endl;
#endif
    m_containers_1.push_back(model1);
    m_containers_2.push_back(model2);
    m_transforms_1.push_back(transform1);
    m_transforms_2.push_back(transform2);

	m_models_1.push_back(obb_model1);
	m_models_2.push_back(obb_model2);

    m_modelNames_1.push_back(obb_model1->getName());
	m_modelNames_2.push_back(obb_model2->getName());

    m_numAssignedTraversals++;
}


bool BVHTraversalTask::run(Zyklio::MultiThreading::WorkerThreadIface* thread)
{
    std::stringstream tmp;
    tmp << "BVHTraversalTask::run: " << this->getTaskID();
    sofa::helper::AdvancedTimer::stepBegin("BVHTraversalTask", tmp.str().c_str());

    if (m_taskWorkerUnits.size() > 0)
    {
        m_results.resize(m_taskWorkerUnits.size(), 0);

		thread->startThreadTimer();

		m_elapsedTimePerTest.clear();
		m_elapsedTime_CPUStep.clear();

		ObbTreeGPU_BVH_Traverse_Streamed_Batch(m_containers_1, m_containers_2, m_transforms_1, m_transforms_2,
                                               m_taskWorkerUnits, m_workerStreams, m_alarmDistance, m_contactDistance,
                                               m_results, m_workQueueCounts,
                                               m_memStream, m_startEvent, m_stopEvent, m_balanceEvent,
                                               m_numAssignedTraversals,
											   m_elapsedTime,
											   m_elapsedTimePerTest,
											   m_elapsedTime_CPUStep,
											   m_workerStartEvents, m_workerStopEvents);
		
		thread->stopThreadTimer();
		this->m_elapsedTimeInStep = thread->getThreadRunTime();
		thread->resetThreadTimer();

		m_traversalCalls++;
#ifdef BVHTRAVERSALTASK_DEBUG
		std::cout << " elapsed time: CUDA = " << m_elapsedTime << " ms" << std::endl; 
#endif

    }

	sofa::helper::AdvancedTimer::stepEnd(("BVHTraversalTask", tmp.str().c_str()));
    return true;
}

void BVHTraversalTask::clearWorkList()
{
    this->m_containers_1.clear();
    this->m_containers_2.clear();
    this->m_transforms_1.clear();
    this->m_transforms_2.clear();

	m_models_1.clear();
	m_models_2.clear();
	
	m_modelNames_1.clear();
    m_modelNames_2.clear();

    m_numAssignedTraversals = 0;
    m_traversalCalls = 0;
    m_results.clear();
}

gProximityWorkerUnit* BVHTraversalTask::getWorkerUnit(unsigned int idx)
{
    if (idx < m_taskWorkerUnits.size())
        return m_taskWorkerUnits[idx];

    return NULL;
}

OBBContainer* BVHTraversalTask::getContainer1(unsigned int idx)
{
    if (idx < m_containers_1.size())
        return m_containers_1[idx];

    return NULL;
}

OBBContainer* BVHTraversalTask::getContainer2(unsigned int idx)
{
    if (idx < m_containers_2.size())
        return m_containers_2[idx];

    return NULL;
}

ObbTreeGPUCollisionModel<Vec3Types>* BVHTraversalTask::getModel1(unsigned int idx)
{
	if (idx < m_models_1.size())
		return m_models_1[idx];

	return NULL;
}

ObbTreeGPUCollisionModel<Vec3Types>* BVHTraversalTask::getModel2(unsigned int idx)
{
	if (idx < m_models_2.size())
		return m_models_2[idx];

	return NULL;
}

const std::string BVHTraversalTask::getModelName1(unsigned int idx) const
{
    if (idx < m_modelNames_1.size())
        return m_modelNames_1[idx];

    return std::string("");
}

const std::string BVHTraversalTask::getModelName2(unsigned int idx) const
{
    if (idx < m_modelNames_2.size())
        return m_modelNames_2[idx];

    return std::string("");
}

#endif // OBBTREEGPU_MULTITHREAD_TASKS_INL
