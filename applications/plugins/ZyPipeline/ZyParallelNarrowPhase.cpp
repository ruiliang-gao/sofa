#include "ZyParallelNarrowPhase.h"

#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Simulation.h>

#include <SofaConstraint/LocalMinDistance.h>
#include <SofaBaseCollision/MinProximityIntersection.h>
#include <SofaBaseCollision/NewProximityIntersection.h>
#include <SofaConstraint/LMDNewProximityIntersection.h>

using namespace Zyklio::Pipeline;

SOFA_DECL_CLASS(ZyParallelNarrowPhase)
int ZyParallelNarrowPhaseClass = sofa::core::RegisterObject("TruPhysics parallel NarrowPhaseDetection")
.add< ZyParallelNarrowPhase >()
;

ZyParallelNarrowPhase::ZyParallelNarrowPhase(const unsigned int& numWorkerThreads) : m_numWorkerThreads(numWorkerThreads), sofa::core::collision::NarrowPhaseDetection()
{
	m_detectionOutputVectors = new NarrowPhaseDetection::DetectionOutputMap();

	for (unsigned int k = 0; k < m_numWorkerThreads; ++k)
	{
		// Useful for something?
		TaskStatus status;

		std::stringstream idStr;
		idStr << "CPU BVH traversal task " << k;
		// No further setup necessary for CPU tasks as of now; assume 8 'worker units' per task
		CPUCollisionCheckTask* cpu_task = new CPUCollisionCheckTask(&status, 8);
		cpu_task->setTaskID(idStr.str());
		m_cpuBVHTraversalTasks.push_back(cpu_task);
	}
	m_scheduler_traverseBVH = new MultiThread_Scheduler<CPUCollisionCheckTask>(m_numWorkerThreads);

	for (unsigned int i = 0; i < m_numWorkerThreads; i++)
		m_scheduler_traverseBVH->getScheduler()->createWorkerThread(true, "CPU_BVH_Traversal");

	m_scheduler_traverseBVH->getScheduler()->startThreads();
	m_scheduler_traverseBVH->getScheduler()->pauseThreads();
}

ZyParallelNarrowPhase::~ZyParallelNarrowPhase()
{
	if (m_detectionOutputVectors != NULL)
	{
		delete m_detectionOutputVectors;
		m_detectionOutputVectors = NULL;
	}
}

void ZyParallelNarrowPhase::beginNarrowPhase()
{
	NarrowPhaseDetection::beginNarrowPhase();
	
	m_narrowPhasePairs.clear();
	m_narrowPhasePairs_TaskAssignment.clear();
	m_narrowPhasePairs_ModelAssociations.clear();

	if (m_detectionOutputVectors != NULL)
		m_detectionOutputVectors->clear();
}

void ZyParallelNarrowPhase::bwdInit()
{
	searchIntersectionMethodInstances();
}

void ZyParallelNarrowPhase::searchIntersectionMethodInstances()
{	
	std::vector<sofa::component::collision::LocalMinDistance*> lmd_vec;
	sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::collision::LocalMinDistance, std::vector<sofa::component::collision::LocalMinDistance* > > lmd_cb(&lmd_vec);
	sofa::simulation::getSimulation()->getCurrentRootNode()->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::collision::LocalMinDistance>::get(), 
																		lmd_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

	std::vector<sofa::component::collision::LMDNewProximityIntersection*> lmd_npi_vec;
	sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::collision::LMDNewProximityIntersection, std::vector<sofa::component::collision::LMDNewProximityIntersection* > > lmd_npi_cb(&lmd_npi_vec);
	sofa::simulation::getSimulation()->getCurrentRootNode()->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::collision::LMDNewProximityIntersection>::get(),
		lmd_npi_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

	std::vector<sofa::component::collision::MinProximityIntersection*> mpi_vec;
	sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::collision::MinProximityIntersection, std::vector<sofa::component::collision::MinProximityIntersection* > > mpi_cb(&mpi_vec);
	sofa::simulation::getSimulation()->getCurrentRootNode()->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::collision::MinProximityIntersection>::get(),
		mpi_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

	std::vector<sofa::component::collision::NewProximityIntersection*> npi_vec;
	sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::collision::NewProximityIntersection, std::vector<sofa::component::collision::NewProximityIntersection* > > npi_cb(&npi_vec);
	sofa::simulation::getSimulation()->getCurrentRootNode()->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::collision::NewProximityIntersection>::get(),
		npi_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);

    msg_info("ZyParallelNarrowPhase") << "=== Intersection method instances in scene graph ===";
    msg_info("ZyParallelNarrowPhase") << " * LocalMinDistance           : " << lmd_vec.size();
    msg_info("ZyParallelNarrowPhase") << " * LMDNewProximityIntersection: " << lmd_npi_vec.size();
    msg_info("ZyParallelNarrowPhase") << " * MinProximityIntersection   : " << mpi_vec.size();
    msg_info("ZyParallelNarrowPhase") << " * NewProximityIntersection   : " << npi_vec.size();
    msg_info("ZyParallelNarrowPhase") << "=== Intersection method instances in scene graph ===";
}

void ZyParallelNarrowPhase::addCollisionPair(const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>& cmPair)
{
	m_narrowPhasePairs.push_back(std::make_pair(cmPair.first, cmPair.second));
}

void ZyParallelNarrowPhase::addCollisionPairs(const sofa::helper::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& v)
{
	for (sofa::helper::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >::const_iterator it = v.begin(); it != v.end(); it++)
	{
		addCollisionPair(*it);
	}

	createDetectionOutputs(m_narrowPhasePairs);
}

void ZyParallelNarrowPhase::endNarrowPhase()
{
	// Clear task list from last iteration
	m_scheduler_traverseBVH->clearTasks();

	int traversalsAssigned = 0;

	// Round-robin, baby
	for (size_t k = 0; k < m_narrowPhasePairs.size(); ++k)
	{
		std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> collisionPair = m_narrowPhasePairs[k];
		unsigned int traversalIndex = traversalsAssigned % m_scheduler_traverseBVH->getNumThreads();

		sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap::iterator outputVector = m_outputsMap.find(collisionPair);

		if (outputVector != m_outputsMap.end())
		{
            msg_info("ZyParallelNarrowPhase") << "  - " << k << ": " << collisionPair.first->getName() << " <-> " << collisionPair.second->getName() << ": Found pre-created DetectionOutputVector, added to task " << traversalIndex ;
			sofa::core::CollisionModel* model_1 = collisionPair.first;
			sofa::core::CollisionModel* model_2 = collisionPair.second;
			while (model_1->getNext() != NULL)
			{
				model_1 = model_1->getNext();
			}

			while (model_2->getNext() != NULL)
			{
				model_2 = model_2->getNext();
			}

            msg_info("ZyParallelNarrowPhase") << "   --> Resolves to: " << model_1->getName() << " <-> " << model_2->getName() ;

			m_cpuBVHTraversalTasks[traversalIndex]->addCollidingPair(collisionPair, outputVector->second);

			m_narrowPhasePairs_TaskAssignment[traversalIndex].push_back(std::make_pair(model_1->getName(), model_2->getName()));
			m_narrowPhasePairs_ModelAssociations.insert(std::make_pair(collisionPair, std::make_pair(model_1->getName(), model_2->getName())));

			traversalsAssigned++;
		}
		else
		{
            msg_info("ZyParallelNarrowPhase") << "  - " << k << ": " << collisionPair.first->getName() << " <-> " << collisionPair.second->getName() << ": NOT ADDED, no pre-created DetectionOutputVector found!!!" ;
		}
	}

	if (traversalsAssigned > 0)
	{
		// First add CPU traversals; these can run independently
		for (size_t k = 0; k < m_cpuBVHTraversalTasks.size(); ++k)
		{
			m_scheduler_traverseBVH->addTask(m_cpuBVHTraversalTasks[k]);
		}

		m_scheduler_traverseBVH->getScheduler()->distributeTasks();

		m_scheduler_traverseBVH->getScheduler()->resumeThreads();
		m_scheduler_traverseBVH->runTasks();
		m_scheduler_traverseBVH->getScheduler()->pauseThreads();


		for (unsigned int k = 0; k < m_cpuBVHTraversalTasks.size(); k++)
		{
			m_cpuBVHTraversalTasks[k]->setFinished(false);
		}

        msg_info("ZyParallelNarrowPhase") << " ---- CPU scheduler task details ----" ;
		for (unsigned int k = 0; k < m_cpuBVHTraversalTasks.size(); k++)
		{
			CPUCollisionCheckTask* task_k = m_cpuBVHTraversalTasks[k];
			const std::vector<float>& elapsedPerTest = task_k->getElapsedTimePerTest();

            msg_info("ZyParallelNarrowPhase") << " - task " << k << " (" << task_k->getTaskID() << "): total = " << task_k->getElapsedTime() / 1000000.0f << " ms" ;

			if (elapsedPerTest.size() > 0)
			{
                msg_info("ZyParallelNarrowPhase") << "   bin checks: " ;

				for (unsigned int m = 0; m < elapsedPerTest.size(); m++)
				{
					if (m_narrowPhasePairs_TaskAssignment.find(k) != m_narrowPhasePairs_TaskAssignment.end())
					{
						std::vector<std::pair<std::string, std::string> >& assignedPairs = m_narrowPhasePairs_TaskAssignment[k];
						if (m < assignedPairs.size())
                            msg_info("ZyParallelNarrowPhase") << "   * pair " << m << ": " << assignedPairs[m].first << " -- " << assignedPairs[m].second << " runtime = " << elapsedPerTest[m] / 1000000.0f << " ms" ;
					}
				}
			}
            msg_info("ZyParallelNarrowPhase") ;
		}
        msg_info("ZyParallelNarrowPhase") << " ---- CPU scheduler task details ----" ;

        msg_info("ZyParallelNarrowPhase") << " ---- Detection output vectors ----" ;
		for (unsigned int k = 0; k < m_cpuBVHTraversalTasks.size(); k++)
		{
			CPUCollisionCheckTask* task_k = m_cpuBVHTraversalTasks[k];
			std::vector<sofa::core::collision::DetectionOutputVector*>& detection_outputs = task_k->getDetectionOutputs();
			std::vector<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& colliding_pairs = task_k->getCollidingPairs();

            msg_info("ZyParallelNarrowPhase") << " - task " << k << " (" << task_k->getTaskID() << ") detection outputs size =  " << detection_outputs.size() << " for colliding pairs size = " << colliding_pairs.size() ;
			for (size_t l = 0; l < detection_outputs.size(); ++l)
			{
				if (detection_outputs[l] != NULL)
				{
                    msg_info("ZyParallelNarrowPhase") << "  - DetectionOutputVector " << l << " for pair " << colliding_pairs[l].first->getName() << " -- " << colliding_pairs[l].second->getName() << " empty: " << detection_outputs[l]->empty() << ", num. of elements = " << detection_outputs[l]->size() ;
                    msg_info("ZyParallelNarrowPhase") << "    -> Corresponds to: " << m_narrowPhasePairs_ModelAssociations[colliding_pairs[l]].first << " <-> " << m_narrowPhasePairs_ModelAssociations[colliding_pairs[l]].second ;
					
					sofa::core::collision::DetectionOutputVector* do_vec = detection_outputs[l];
                    msg_info("ZyParallelNarrowPhase") << "    -> Type name: " << typeid(do_vec).name() ;
					
					std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> collision_pair = std::make_pair(colliding_pairs[l].first, colliding_pairs[l].second);
					m_detectionOutputVectors->insert(std::make_pair(collision_pair, do_vec));
				}
				else
				{
                    msg_info("ZyParallelNarrowPhase") << "  - DetectionOutputVector " << l << " for pair " << colliding_pairs[l].first->getName() << " -- " << colliding_pairs[l].second->getName() << " NOT INSTANTIATED!" ;
				}
			}
		}
        msg_info("ZyParallelNarrowPhase") << " ---- Detection output vectors ----" ;
	}
}
