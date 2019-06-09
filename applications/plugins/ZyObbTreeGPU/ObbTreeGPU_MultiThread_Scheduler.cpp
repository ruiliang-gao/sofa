#include "ObbTreeGPU_MultiThread_Scheduler.h"
#include "ObbTreeGPU_MultiThread_Tasks.h"

#include "ObbTreeGPU_MultiThread_CPU_Tasks.h"

#include <boost/pool/pool.hpp>

#include "ObbTreeGPU_CudaDataStructures.h"

#include <Tasks.h>
#include <WorkerThread_Pool.h>

#include <sofa/helper/AdvancedTimer.h>

using namespace sofa::simulation;
using namespace sofa::component::collision;

template <class PoolTaskType>
ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::ObbTreeGPU_MultiThread_Scheduler(unsigned int numThreads): m_numThreads(numThreads)
{
    m_scheduler = new TaskSchedulerPool<PoolTaskType>(m_numThreads);
}

template <class PoolTaskType>
ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::~ObbTreeGPU_MultiThread_Scheduler()
{
    delete m_scheduler;
}

template <class PoolTaskType>
TaskSchedulerPool<PoolTaskType>* ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::getScheduler()
{
	return m_scheduler;
}

template <class PoolTaskType>
void ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::suspend()
{
	m_scheduler->pauseThreads();
}

template <class PoolTaskType>
void ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::resume()
{
	m_scheduler->resumeThreads();
}

template <class PoolTaskType>
void ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::init()
{
    std::cout << "ObbTreeGPU_MultiThread_Scheduler::init(): Start " << m_numThreads << " threads." << std::endl;
    m_scheduler->start(m_numThreads);
}

template <class PoolTaskType>
void ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::bwdInit()
{
    std::cout << "ObbTreeGPU_MultiThread_Scheduler::bwdInit()" << std::endl;
}

template <class PoolTaskType>
void ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::cleanup()
{
    std::cout << "ObbTreeGPU_MultiThread_Scheduler::cleanup(): Shutdown worker threads" << std::endl;
    m_scheduler->stop();
}

template <class PoolTaskType>
bool ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::addTask(PoolTaskType* task)
{
    m_scheduler->addTask(task);

    return true;
}

// FA TODO: No busy wait loop! Fine for GPU tasks, but it seems that is very bad for simultaneously running CPU-based tasks!!!
template <class PoolTaskType>
void ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::runTasks()
{
#ifdef OBBTREEGPU_MULTITHREAD_SCHEDULER_DEBUG
    std::cout << "=== ObbTreeGPU_MultiThread_Scheduler::runTasks() ===" << std::endl;
#endif

    sofa::helper::AdvancedTimer::stepBegin("ObbTreeGPU_MultiThread_Scheduler_Traversal");
    unsigned int waitIterations = 0;
    while(m_scheduler->activeThreads() > 0)
    {
#ifdef OBBTREEGPU_MULTITHREAD_SCHEDULER_DEBUG
        std::cout << "  waiting for running tasks; threads active = " << m_scheduler->activeThreads()  << std::endl;
#endif
#ifndef _WIN32
        usleep(50);
#else
		boost::this_thread::sleep_for(boost::chrono::nanoseconds(50));
#endif
        waitIterations++;
    }
    sofa::helper::AdvancedTimer::stepEnd("ObbTreeGPU_MultiThread_Scheduler_Traversal");
#ifdef OBBTREEGPU_MULTITHREAD_SCHEDULER_DEBUG
    std::cout << "Task processing finished: Waited " << waitIterations << " loops." << std::endl;
#endif
}

template <class PoolTaskType>
void ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::clearTasks()
{
    m_scheduler->clearProcessedTasks();
    m_scheduler->clearQueuedTasks();
}

template <class PoolTaskType>
void ObbTreeGPU_MultiThread_Scheduler<PoolTaskType>::dumpProcessedTasks()
{
	m_scheduler->dumpProcessedTasks();
}

template class ObbTreeGPU_MultiThread_Scheduler<BVHTraversalTask>;
template class ObbTreeGPU_MultiThread_Scheduler<NarrowPhaseGPUTask>;

template class ObbTreeGPU_MultiThread_Scheduler<CPUCollisionCheckTask>;
