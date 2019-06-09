#include "ObbTreeGPU_MultiThread_CPU_Tasks.h"

#include "ObbTreeGPUCollisionDetection_Threaded.h"

#include <sofa/helper/system/thread/CTime.h>
#include <queue>

#include <SofaConstraint/LocalMinDistance.h>
#include <SofaBaseCollision/MinProximityIntersection.h>
#include <SofaBaseCollision/NewProximityIntersection.h>
#include <SofaConstraint/LMDNewProximityIntersection.h>

#include <SofaMeshCollision/MeshNewProximityIntersection.h>
#include <SofaGeneralMeshCollision/MeshMinProximityIntersection.h>

namespace sofa
{
	namespace component
	{
		namespace collision
		{
			class CPUCollisionCheckTaskPrivate
			{
			public:
				CPUCollisionCheckTaskPrivate() : _mnpIntersection(NULL), _mmpIntersection(NULL)
				{}

				~CPUCollisionCheckTaskPrivate()
				{
					if (_mnpIntersection != NULL)
					{
						delete _mnpIntersection;
						_mnpIntersection = NULL;
					}

					if (_mmpIntersection != NULL)
					{
						delete _mmpIntersection;
						_mmpIntersection = NULL;
					}
				}

				BaseProximityIntersection::SPtr _intersectionMethod;
				MeshNewProximityIntersection* _mnpIntersection;
				MeshMinProximityIntersection* _mmpIntersection;
				NewProximityIntersection::SPtr _npIntersection;
			};
		}
	}
}

using namespace sofa::component::collision;

CPUCollisionCheckTask::CPUCollisionCheckTask() : _numWorkerUnits(1), _collisionDetection(NULL), m_d(NULL), m_intersectorType(CPUCollisionCheckIntersector_Type_Default)
{
	m_d = new CPUCollisionCheckTaskPrivate();
}

CPUCollisionCheckTask::~CPUCollisionCheckTask()
{
	delete m_d;
}

CPUCollisionCheckTask::CPUCollisionCheckTask(const Zyklio::MultiThreading::TaskStatus* status, ObbTreeGPUCollisionDetection_Threaded* collisionDetection, unsigned int numWorkerUnits) : 
_numWorkerUnits(numWorkerUnits), _collisionDetection(collisionDetection), m_intersectorType(CPUCollisionCheckIntersector_Type_Default)
{
	m_d = new CPUCollisionCheckTaskPrivate();

	if (m_intersectorType == LocalMinDistance_Type)
		m_d->_intersectionMethod = sofa::core::objectmodel::New<LocalMinDistance>();
	else if (m_intersectorType == LMDNewProximityIntersection_Type)
		m_d->_intersectionMethod = sofa::core::objectmodel::New<LMDNewProximityIntersection>();
	else if (m_intersectorType == MinProximityIntersection_Type)
		m_d->_intersectionMethod = sofa::core::objectmodel::New<MinProximityIntersection>();
	else if (m_intersectorType == NewProximityIntersection_Type)
		m_d->_intersectionMethod = sofa::core::objectmodel::New<NewProximityIntersection>();
	
	if (m_d->_intersectionMethod.get())
		m_d->_intersectionMethod->init();
}


CPUCollisionCheckTask::CPUCollisionCheckTask(const CPUCollisionCheckTask& other)
{
	if (this != &other)
	{
		for (size_t k = 0; k < other._collidingPairs.size(); ++k)
		{
			_collidingPairs.push_back(std::make_pair(other._collidingPairs[k].first, other._collidingPairs[k].second));
		}

		for (size_t k = 0; k < other._detectionOutputs.size(); ++k)
		{
			_detectionOutputs.push_back(other._detectionOutputs[k]);
		}

		m_d = new CPUCollisionCheckTaskPrivate();
		m_d->_intersectionMethod = other.m_d->_intersectionMethod;
		_collisionDetection = other._collisionDetection;

		m_elapsedTime = other.m_elapsedTime;

		for (std::vector<float>::const_iterator it = other.m_elapsedTimePerTest.begin(); it != other.m_elapsedTimePerTest.end(); ++it)
			m_elapsedTimePerTest.push_back(*it);
	}
}

CPUCollisionCheckTask& CPUCollisionCheckTask::operator=(const CPUCollisionCheckTask& other)
{
	if (this != &other)
	{
		for (size_t k = 0; k < other._collidingPairs.size(); ++k)
		{
			_collidingPairs.push_back(std::make_pair(other._collidingPairs[k].first, other._collidingPairs[k].second));
		}

		for (size_t k = 0; k < other._detectionOutputs.size(); ++k)
		{
			_detectionOutputs.push_back(other._detectionOutputs[k]);
		}

		m_d = new CPUCollisionCheckTaskPrivate();
		m_d->_intersectionMethod = other.m_d->_intersectionMethod;
		_collisionDetection = other._collisionDetection;

		m_elapsedTime = other.m_elapsedTime;

		for (std::vector<float>::const_iterator it = other.m_elapsedTimePerTest.begin(); it != other.m_elapsedTimePerTest.end(); ++it)
			m_elapsedTimePerTest.push_back(*it);
	}
	return *this;
}

void CPUCollisionCheckTask::addCollidingPair(const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>& collidingPair, sofa::core::collision::DetectionOutputVector*& outputVector)
{
	_collidingPairs.push_back(collidingPair);
	_detectionOutputs.push_back(outputVector);
}

void CPUCollisionCheckTask::clearWorkList()
{
	_collidingPairs.clear();
	_detectionOutputs.clear();
}

void CPUCollisionCheckTask::processCollidingPair(const std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*>& cmPair, sofa::core::collision::DetectionOutputVector*& outputVector)
{
	typedef std::pair< std::pair<core::CollisionElementIterator, core::CollisionElementIterator>, std::pair<core::CollisionElementIterator, core::CollisionElementIterator> > TestPair;

	core::CollisionModel *cm1 = cmPair.first;
	core::CollisionModel *cm2 = cmPair.second;

	if (!cm1->isSimulated() && !cm2->isSimulated())
		return;

	if (cm1->empty() || cm2->empty())
		return;

	core::CollisionModel *finalcm1 = cm1->getLast(); //get the finnest CollisionModel which is not a CubeModel
	core::CollisionModel *finalcm2 = cm2->getLast();
	
	//sout << "Final phase "<<gettypename(typeid(*finalcm1))<<" - "<<gettypename(typeid(*finalcm2))<<sendl;
	
	bool swapModels = false;
	core::collision::ElementIntersector* finalintersector = m_d->_intersectionMethod->findIntersector(finalcm1, finalcm2, swapModels); //find the method for the finnest CollisionModels
	if (finalintersector == NULL)
	{
		std::cout << "CPUCollisionCheckTask " << this->getTaskID() << ": Error finding FINAL intersector from " << m_d->_intersectionMethod->getName() << " for " << finalcm1->getClassName() << " - " << finalcm2->getClassName() << std::endl;
		return;
	}

	if (swapModels)
	{
		core::CollisionModel* tmp;
		tmp = cm1; cm1 = cm2; cm2 = tmp;
		tmp = finalcm1; finalcm1 = finalcm2; finalcm2 = tmp;
	}

	const bool self = (finalcm1->getContext() == finalcm2->getContext());
	//if (self)
	//    sout << "SELF: Final intersector " << finalintersector->name() << " for "<<finalcm1->getName()<<" - "<<finalcm2->getName()<<sendl;

	// Use a pre-created DetectionOutputVector here
	sofa::core::collision::DetectionOutputVector*& outputs = outputVector; // _collisionDetection->getDetectionOutputs(finalcm1, finalcm2);

	finalintersector->beginIntersect(finalcm1, finalcm2, outputs); //creates outputs if null

	if (finalcm1 == cm1 || finalcm2 == cm2)
	{
		// The last model also contains the root element -> it does not only contains the final level of the tree
		finalcm1 = NULL;
		finalcm2 = NULL;
		finalintersector = NULL;
	}

	std::queue< TestPair > externalCells;

	std::pair<core::CollisionElementIterator, core::CollisionElementIterator> internalChildren1 = cm1->begin().getInternalChildren();
	std::pair<core::CollisionElementIterator, core::CollisionElementIterator> internalChildren2 = cm2->begin().getInternalChildren();
	std::pair<core::CollisionElementIterator, core::CollisionElementIterator> externalChildren1 = cm1->begin().getExternalChildren();
	std::pair<core::CollisionElementIterator, core::CollisionElementIterator> externalChildren2 = cm2->begin().getExternalChildren();
	if (internalChildren1.first != internalChildren1.second)
	{
		if (internalChildren2.first != internalChildren2.second)
			externalCells.push(std::make_pair(internalChildren1, internalChildren2));
		if (externalChildren2.first != externalChildren2.second)
			externalCells.push(std::make_pair(internalChildren1, externalChildren2));
	}

	if (externalChildren1.first != externalChildren1.second)
	{
		if (internalChildren2.first != internalChildren2.second)
			externalCells.push(std::make_pair(externalChildren1, internalChildren2));
		if (externalChildren2.first != externalChildren2.second)
			externalCells.push(std::make_pair(externalChildren1, externalChildren2));
	}

	core::collision::ElementIntersector* intersector = NULL;
	MirrorIntersector mirror;
	cm1 = NULL; // force later init of intersector
	cm2 = NULL;

	while (!externalCells.empty())
	{
		TestPair root = externalCells.front();
		externalCells.pop();

		if (cm1 != root.first.first.getCollisionModel() || cm2 != root.second.first.getCollisionModel())//if the CollisionElements do not belong to cm1 and cm2, update cm1 and cm2
		{
			cm1 = root.first.first.getCollisionModel();
			cm2 = root.second.first.getCollisionModel();
			if (!cm1 || !cm2) continue;
			intersector = m_d->_intersectionMethod->findIntersector(cm1, cm2, swapModels);

			if (intersector == NULL)
			{
				std::cout << "CPUCollisionCheckTask " << this->getTaskID() << ": Error finding intersector " << m_d->_intersectionMethod->getName() << " for " << cm1->getClassName() << " - " << cm2->getClassName() << std::endl;
			}
			else
			{
				std::cout << "CPUCollisionCheckTask " << this->getTaskID() << ": intersector " << intersector->name() << " for " << m_d->_intersectionMethod->getName() << " for " << gettypename(typeid(*cm1)) << " - " << gettypename(typeid(*cm2)) << std::endl;
			}

			if (swapModels)
			{
				mirror.intersector = intersector; intersector = &mirror;
			}
		}
		
		if (intersector == NULL)
		{
			std::cout << "CPUCollisionCheckTask " << this->getTaskID() << ": NO intersector found to process " << cm1->getTypeName() << " <-> " << cm2->getTypeName() << std::endl;
			continue;
		}

		std::stack< TestPair > internalCells;
		internalCells.push(root);

		while (!internalCells.empty())
		{
			TestPair current = internalCells.top();
			internalCells.pop();

			core::CollisionElementIterator begin1 = current.first.first;
			core::CollisionElementIterator end1 = current.first.second;
			core::CollisionElementIterator begin2 = current.second.first;
			core::CollisionElementIterator end2 = current.second.second;

			if (begin1.getCollisionModel() == finalcm1 && begin2.getCollisionModel() == finalcm2)
			{
				// Final collision pairs
				for (core::CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
				{
					for (core::CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
					{
						if (!self || it1.canCollideWith(it2))
							intersector->intersect(it1, it2, outputs);
					}
				}
			}
			else
			{
				for (core::CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
				{
					for (core::CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
					{
						bool b = intersector->canIntersect(it1, it2);
						if (b)
						{
							// Need to test recursively
							// Note that an element cannot have both internal and external children

							TestPair newInternalTests(it1.getInternalChildren(), it2.getInternalChildren());
							TestPair newExternalTests(it1.getExternalChildren(), it2.getExternalChildren());
							if (newInternalTests.first.first != newInternalTests.first.second)
							{
								if (newInternalTests.second.first != newInternalTests.second.second)
								{
									internalCells.push(newInternalTests);
								}
								else
								{
									newInternalTests.second.first = it2;
									newInternalTests.second.second = it2;
									++newInternalTests.second.second;
									internalCells.push(newInternalTests);
								}
							}
							else
							{
								if (newInternalTests.second.first != newInternalTests.second.second)
								{
									newInternalTests.first.first = it1;
									newInternalTests.first.second = it1;
									++newInternalTests.first.second;
									internalCells.push(newInternalTests);
								}
								else
								{
									// end of both internal tree of elements.
									// need to test external children
									if (newExternalTests.first.first != newExternalTests.first.second)
									{
										if (newExternalTests.second.first != newExternalTests.second.second)
										{
											if (newExternalTests.first.first.getCollisionModel() == finalcm1 && newExternalTests.second.first.getCollisionModel() == finalcm2)
											{
												core::CollisionElementIterator begin1 = newExternalTests.first.first;
												core::CollisionElementIterator end1 = newExternalTests.first.second;
												core::CollisionElementIterator begin2 = newExternalTests.second.first;
												core::CollisionElementIterator end2 = newExternalTests.second.second;
												for (core::CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
												{
													for (core::CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
													{
														// Final collision pair
														if (!self || it1.canCollideWith(it2))
															finalintersector->intersect(it1, it2, outputs);
													}
												}
											}
											else
												externalCells.push(newExternalTests);
										}
										else
										{
											// only first element has external children
											// test them against the second element
											newExternalTests.second.first = it2;
											newExternalTests.second.second = it2;
											++newExternalTests.second.second;
											externalCells.push(std::make_pair(newExternalTests.first, newInternalTests.second));
										}
									}
									else if (newExternalTests.second.first != newExternalTests.second.second)
									{
										// only first element has external children
										// test them against the first element
										newExternalTests.first.first = it1;
										newExternalTests.first.second = it1;
										++newExternalTests.first.second;
										externalCells.push(std::make_pair(newExternalTests.first, newExternalTests.second));
									}
									else
									{
										// No child -> final collision pair
										if (!self || it1.canCollideWith(it2))
											intersector->intersect(it1, it2, outputs);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

bool CPUCollisionCheckTask::run(Zyklio::MultiThreading::WorkerThreadIface* thread_iface)
{
	std::cout << "CPUCollisionCheckTask::run(" << this->getTaskID() << "): " << _collidingPairs.size() << " pair checks." << std::endl;

	thread_iface->resetThreadTimer();
	thread_iface->startThreadTimer();

	m_elapsedTime = 0.0f;
	m_elapsedTimePerTest.clear();

	unsigned int numProcessed = 0;

	if (_collisionDetection != NULL)
	{
		for (size_t k = 0; k < _collidingPairs.size(); ++k)
		{
			thread_iface->startStepTimer();
			processCollidingPair(_collidingPairs[k], _detectionOutputs[k]);
			thread_iface->stopStepTimer();

			this->m_elapsedTime += (thread_iface->getStepRunTime());
			this->m_elapsedTimePerTest.push_back((float) thread_iface->getStepRunTime());

			if (numProcessed >= 1)
				break;

			numProcessed++;
		}
	}

	thread_iface->stopThreadTimer();
	return true;
}