#include "ObbTreeGPUIntersection.h"

#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/helper/proximity.h>
#include <sofa/core/collision/Intersection.inl>
#include <SofaBaseCollision/MinProximityIntersection.h>

#include <iostream>
#include <algorithm>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/IntersectorFactory.h>

#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/CubeModel.h>

//#define OBBTREEGPUINTERSECTION_DEBUG
namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace sofa::defaulttype;
            using namespace sofa::core::collision;
            using namespace sofa::component::collision;

            SOFA_DECL_CLASS(OBBTreeGPUDiscreteIntersection)

            IntersectorCreator<DiscreteIntersection, OBBTreeGPUDiscreteIntersection> OBBTreeGPUDiscreteIntersectors("OBBTreeGPU");

            OBBTreeGPUDiscreteIntersection::OBBTreeGPUDiscreteIntersection()
            {
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                msg_info("ObbTreeGPUIntersection") << "OBBTreeGPUDiscreteIntersection::OBBTreeGPUDiscreteIntersection() -- SHOULD NOT BE CALLED";
#endif

                sofa::core::objectmodel::BaseObjectDescription discreteIntersectionDesc("Default Intersection","DiscreteIntersection");
                sofa::core::objectmodel::BaseObject::SPtr obj = sofa::core::ObjectFactory::CreateObject(getContext(), &discreteIntersectionDesc);
                intersection = dynamic_cast<component::collision::DiscreteIntersection*>(obj.get());

                OBBTreeGPUDiscreteIntersection(intersection, true);
            }

            OBBTreeGPUDiscreteIntersection::OBBTreeGPUDiscreteIntersection(DiscreteIntersection* object, bool addSelf): BaseIntersector()
                //: intersection(object)
            {
                /*
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                msg_info("ObbTreeGPUIntersection") << "OBBTreeGPUDiscreteIntersection::OBBTreeGPUDiscreteIntersection(" << object << "," << addSelf << ")";
#endif
                if (intersection && addSelf)
                {

#ifdef OBBTREEGPUINTERSECTION_DEBUG
                    msg_info("ObbTreeGPUIntersection") << "Add intersector for ObbTreeGPUCollisionModel <-> ObbTreeGPUCollisionModel";
#endif
                    intersection->intersectors.add<ObbTreeGPUCollisionModel<>, ObbTreeGPUCollisionModel<>, OBBTreeGPUDiscreteIntersection>(this);

                }*/
            }

            bool OBBTreeGPUDiscreteIntersection::testIntersection(ObbTreeGPUCollisionModelNode& e1, ObbTreeGPUCollisionModelNode &e2)
            {
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                msg_info("ObbTreeGPUIntersection") << "OBBTreeGPUDiscreteIntersection::testIntersection(ObbTreeGPUCollisionModelNode, ObbTreeGPUCollisionModelNode): OBBTreeGPUCollisionModels " << e1.getCollisionModel()->getName() << " -- " << e2.getCollisionModel()->getName() << ": ";
#endif
                const CubeModel::CubeData& cubeData1 = e1.getCollisionModel()->getCubeModel()->getCubeData(0);
                const CubeModel::CubeData& cubeData2 = e2.getCollisionModel()->getCubeModel()->getCubeData(0);
                const Vector3& minVect1 = cubeData1.minBBox;
                const Vector3& minVect2 = cubeData2.minBBox;
                const Vector3& maxVect1 = cubeData1.maxBBox;
                const Vector3& maxVect2 = cubeData2.maxBBox;

                // TODO: Make a data member
                const double alarmDist = 2.0f;
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                msg_info("ObbTreeGPUIntersection") << "  minVect1 = " << minVect1 << ", maxVect1 = " << maxVect1 << "; minVect2 = " << minVect2 << ", maxVect2 = " << maxVect2 << "; alarmDist = " << alarmDist;
#endif
                for (int i = 0; i < 3; i++)
                {
                    if (minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist)
                    {
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                        msg_info("ObbTreeGPUIntersection") << " NO INTERSECTION of AABB's";
#endif
						return false;
                    }
                }
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                msg_info("ObbTreeGPUIntersection") << " INTERSECTION DETECTED!!!";
#endif
                return true;
            }

            int OBBTreeGPUDiscreteIntersection::computeIntersection(ObbTreeGPUCollisionModelNode& e1, ObbTreeGPUCollisionModelNode& e2, OutputVector* contacts)
            {
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                msg_info("ObbTreeGPUIntersection") << "OBBTreeGPUDiscreteIntersection::computeIntersection(): OBBTreeGPUCollisionModels " << e1.getCollisionModel()->getName() << " -- " << e2.getCollisionModel()->getName();
#endif
                return 0;
            }

            int OBBTreeGPUDiscreteIntersectionClass = core::RegisterObject("GPU discrete intersection for gProximity")
                    .add< OBBTreeGPUDiscreteIntersection >()
                    ;

            SOFA_DECL_CLASS(ObbTreeGPULocalMinDistance)

            int ObbTreeGPULocalMinDistanceClass = core::RegisterObject("ObbTreeGPU specialization of LocalMinDistance")
                    .add< ObbTreeGPULocalMinDistance >()
                    ;

            ObbTreeGPULocalMinDistance::ObbTreeGPULocalMinDistance()
                : LocalMinDistance()
            {
            }

            void ObbTreeGPULocalMinDistance::init()
            {
                intersectors.add<ObbTreeGPUCollisionModel<>, ObbTreeGPUCollisionModel<>, ObbTreeGPULocalMinDistance>(this);
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                msg_info("ObbTreeGPUIntersection") << "ObbTreeGPULocalMinDistance::init(" << this->getName() << ")";
				testOutputFilename.setValue(sofa::helper::system::DataRepository.getFirstPath() + "/" + this->getName() + ".log");
                msg_info("ObbTreeGPUIntersection") << "testOutputFilename = " << testOutputFilename.getValue();

				testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::trunc);
				testOutput.close();
#endif
                // intersector with cube model so that a combination of ObbTreeGPU- and Point-/Line-/Triangle-models doesn't cause an error.
                
                intersectors.add<ObbTreeGPUCollisionModel<>, CubeModel, ObbTreeGPULocalMinDistance>(this);

                intersectors.add<CubeModel, CubeModel, ObbTreeGPULocalMinDistance>(this);

				// Not enough to register in super class: Template parameter 3
				intersectors.add<PointModel, PointModel, ObbTreeGPULocalMinDistance>(this); // point-point is always activated

				intersectors.add<LineModel, LineModel, ObbTreeGPULocalMinDistance>(this);
				intersectors.add<LineModel, PointModel, ObbTreeGPULocalMinDistance>(this);
				
				intersectors.add<TriangleModel, PointModel, ObbTreeGPULocalMinDistance>(this);

                // if any of these are needed, implement the corresponding testIntersection(ObbTreeGPUCollisionModelNode&, ...&)
                // and computeIntersection(ObbTreeGPUCollisionModelNode& , ...& e2, OutputVector* contacts) functions
                // in ObbTreeGPULocalMinDistance (ObbTreeGPUIntersection.h)
                /*intersectors.add<ObbTreeGPUCollisionModel, CapsuleModel, ObbTreeGPULocalMinDistance>(this);
                intersectors.add<ObbTreeGPUCollisionModel, RigidCapsuleModel, ObbTreeGPULocalMinDistance>(this);
                intersectors.add<ObbTreeGPUCollisionModel, SphereModel, ObbTreeGPULocalMinDistance>(this);
                intersectors.add<ObbTreeGPUCollisionModel, OBBModel, ObbTreeGPULocalMinDistance>(this);
                intersectors.add<ObbTreeGPUCollisionModel, RigidSphereModel, ObbTreeGPULocalMinDistance>(this);

                IntersectorFactory::getInstance()->addIntersectors(this);*/
                // tst ende

                LocalMinDistance::init();
            }

            bool ObbTreeGPULocalMinDistance::testIntersection(ObbTreeGPUCollisionModelNode& e1, ObbTreeGPUCollisionModelNode &e2)
            {
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                testOutput << "=== OBBTreeGPULocalMinDistance::testIntersection(ObbTreeGPUCollisionModelNode, ObbTreeGPUCollisionModelNode): OBBTreeGPUCollisionModels " << e1.getCollisionModel()->getName() << " -- " << e2.getCollisionModel()->getName()  << ", dt = " << this->getTime() << " ===";
#endif

				checkedCollisionModels.push_back(std::make_pair(e1.getCollisionModel()->getName(), e2.getCollisionModel()->getName()));

                const CubeModel::CubeData& cubeData1 = e1.getCollisionModel()->getCubeModel()->getCubeData(0);
                const CubeModel::CubeData& cubeData2 = e2.getCollisionModel()->getCubeModel()->getCubeData(0);
                const Vector3& minVect1 = cubeData1.minBBox;
                const Vector3& minVect2 = cubeData2.minBBox;
                const Vector3& maxVect1 = cubeData1.maxBBox;
                const Vector3& maxVect2 = cubeData2.maxBBox;

                const double alarmDist = getAlarmDistance();

#ifdef OBBTREEGPUINTERSECTION_DEBUG
                testOutput << "  minVect1 = " << minVect1 << ", maxVect1 = " << maxVect1 << "; minVect2 = " << minVect2 << ", maxVect2 = " << maxVect2 << "; alarmDist = " << alarmDist;
#endif

                for (int i = 0; i < 3; i++)
                {
                    if (minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist)
					{
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                        testOutput << "  NO AABB Overlap detected -- " << minVect1[i] << " > " << maxVect2[i] << " + " << alarmDist << " -- " << minVect2[i] << " > " << maxVect1[i] << " + " << alarmDist;
#endif
                        return false;
                    }
                }

#ifdef OBBTREEGPUINTERSECTION_DEBUG
                testOutput << "  AABB Intersection DETECTED -- model1 is ghost = " << e1.getCollisionModel()->isGhostObject() << ", model2 is ghost = " << e2.getCollisionModel()->isGhostObject();
				if (e1.getCollisionModel()->isGhostObject() && e2.getCollisionModel()->isGhostObject())
                    testOutput << "  ghost-ghost AABB intersection detected!!!";
#endif
				overlappingCollisionModels.push_back(std::make_pair(e1.getCollisionModel()->getName(), e2.getCollisionModel()->getName()));

                return true;
            }

            int ObbTreeGPULocalMinDistance::computeIntersection(ObbTreeGPUCollisionModelNode& e1, ObbTreeGPUCollisionModelNode& e2, OutputVector* contacts)
            {
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                msg_info("ObbTreeGPUIntersection") << "OBBTreeGPULocalMinDistance::computeIntersection(): OBBTreeGPUCollisionModels " << e1.getCollisionModel()->getName() << " -- " << e2.getCollisionModel()->getName();
#endif
                return 0;
            }

			void ObbTreeGPULocalMinDistance::beginBroadPhase()
            {
#ifdef OBBTREEGPUINTERSECTION_DEBUG
				testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::ate | std::ofstream::app);
                testOutput << "=== ObbTreeGPULocalMinDistance::beginBroadPhase(" << this->getName() << "), dt = " << this->getTime() << " ===";
#endif
				checkedCollisionModels.clear();
				overlappingCollisionModels.clear();
			}

			void ObbTreeGPULocalMinDistance::endBroadPhase()
            {
#ifdef OBBTREEGPUINTERSECTION_DEBUG
                testOutput << "=== ObbTreeGPULocalMinDistance::endBroadPhase(" << this->getName() << "), dt = " << this->getTime() << " ===";

                testOutput << "AABB pairs tested in broad-phase: " << checkedCollisionModels.size();
				for (std::vector<std::pair<std::string, std::string> >::const_iterator it = checkedCollisionModels.begin(); it != checkedCollisionModels.end(); it++)
                    testOutput << " - " << it->first << " -- " << it->second;

                testOutput << "AABB pairs overlapping in broad-phase: " << overlappingCollisionModels.size();
				for (std::vector<std::pair<std::string, std::string> >::const_iterator it = overlappingCollisionModels.begin(); it != overlappingCollisionModels.end(); it++)
                    testOutput << " - " << it->first << " -- " << it->second;

                testOutput.close();
#endif
			}

			bool ObbTreeGPULocalMinDistance::testIntersection(Cube& c1, Cube& c2)
			{
				return LocalMinDistance::testIntersection(c1, c2);
			}

			bool ObbTreeGPULocalMinDistance::testIntersection(Point& c1, Point& c2)
			{
				return LocalMinDistance::testIntersection(c1, c2);
			}
			
			bool ObbTreeGPULocalMinDistance::testIntersection(Line& c1, Point& c2)
			{
				return LocalMinDistance::testIntersection(c1, c2);
			}

			bool ObbTreeGPULocalMinDistance::testIntersection(Line& c1, Line& c2)
			{
				return LocalMinDistance::testIntersection(c1, c2);
			}

			bool ObbTreeGPULocalMinDistance::testIntersection(Triangle& c1, Point& c2)
			{
				return LocalMinDistance::testIntersection(c1, c2);
			}

			int ObbTreeGPULocalMinDistance::computeIntersection(Cube& e1, Cube& e2, OutputVector* contacts)
			{
				return LocalMinDistance::computeIntersection(e1, e2, contacts);
			}

			int ObbTreeGPULocalMinDistance::computeIntersection(Point& e1, Point& e2, OutputVector* contacts)
			{
				return LocalMinDistance::computeIntersection(e1, e2, contacts);
			}

			int ObbTreeGPULocalMinDistance::computeIntersection(Line& e1, Point& e2, OutputVector* contacts)
			{
				return LocalMinDistance::computeIntersection(e1, e2, contacts);
			}

			int ObbTreeGPULocalMinDistance::computeIntersection(Line& e1, Line& e2, OutputVector* contacts)
			{
				return LocalMinDistance::computeIntersection(e1, e2, contacts);
			}

			int ObbTreeGPULocalMinDistance::computeIntersection(Triangle& e1, Point& e2, OutputVector* contacts)
			{
				return LocalMinDistance::computeIntersection(e1, e2, contacts);
			}
        }
    }
}
