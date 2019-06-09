#ifndef MULTITHREADING_MULTITHREAD_SCHEDULER_H
#define MULTITHREADING_MULTITHREAD_SCHEDULER_H

#include "Tasks.h"
#include "TaskSchedulerBoostPool.h"

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            struct OBBContainer;

            template <class PoolTaskType = Zyklio::MultiThreading::PoolTask>
            class ObbTreeGPU_MultiThread_Scheduler
            {
                public:
                    ObbTreeGPU_MultiThread_Scheduler(unsigned int numThreads);
                    ~ObbTreeGPU_MultiThread_Scheduler();

                    void init();
                    void bwdInit();
                    void cleanup();

                    bool addTask(PoolTaskType* task);

                    void suspend();
                    void resume();

                    void runTasks();
                    void clearTasks();

					TaskSchedulerPool<PoolTaskType>* getScheduler();
					void dumpProcessedTasks();

					const unsigned int getNumThreads() const { return m_numThreads; }

                private:
                    TaskSchedulerPool<PoolTaskType>* m_scheduler;

                    void initThreadLocalData();

                    unsigned int m_numThreads;
            };
        }
    }
}

#endif // MULTITHREADING_MULTITHREAD_SCHEDULER_H
