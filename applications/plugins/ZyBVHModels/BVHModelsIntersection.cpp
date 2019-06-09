#include "BVHModelsIntersection.h"

#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/helper/proximity.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/collision/Intersection.inl>
#include <SofaBaseCollision/MinProximityIntersection.h>

#include <iostream>
#include <algorithm>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/IntersectorFactory.h>

#include <GL/gl.h>

#define BVHMODELSINTERSECTION_DEBUG
namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace sofa::defaulttype;
            using namespace sofa::core::collision;
            using namespace sofa::component::collision;

            SOFA_DECL_CLASS(BVHModelsDiscreteIntersection)

            IntersectorCreator<DiscreteIntersection, BVHModelsDiscreteIntersection> BVHModelsDiscreteIntersectors("BVHModels");

            BVHModelsDiscreteIntersection::BVHModelsDiscreteIntersection()
            {
#ifdef BVHMODELSINTERSECTION_DEBUG
                std::cout << "BVHModelsDiscreteIntersection::BVHModelsDiscreteIntersection() -- SHOULD NOT BE CALLED" << std::endl;
#endif

                sofa::core::objectmodel::BaseObjectDescription discreteIntersectionDesc("Default Intersection","DiscreteIntersection");
                sofa::core::objectmodel::BaseObject::SPtr obj = sofa::core::ObjectFactory::CreateObject(getContext(), &discreteIntersectionDesc);
                intersection = dynamic_cast<component::collision::DiscreteIntersection*>(obj.get());

                BVHModelsDiscreteIntersection(intersection, true);
            }

            BVHModelsDiscreteIntersection::BVHModelsDiscreteIntersection(DiscreteIntersection* object, bool addSelf)
                : intersection(object)
            {

#ifdef BVHMODELSINTERSECTION_DEBUG
                std::cout << "BVHModelsDiscreteIntersection::BVHModelsDiscreteIntersection(" << object << "," << addSelf << ")" << std::endl;
#endif
                if (intersection && addSelf)
                {

#ifdef BVHMODELSINTERSECTION_DEBUG
                    std::cout << "Add intersector for PQPCollisionModelNode <-> PQPCollisionModelNode" << std::endl;
#endif
                    intersection->intersectors.add<PQPCollisionModel<>, PQPCollisionModel<>, BVHModelsDiscreteIntersection>(this);
                }
            }

            bool BVHModelsDiscreteIntersection::testIntersection(PQPCollisionModelNode& e1, PQPCollisionModelNode &e2)
            {
#ifdef BVHMODELSINTERSECTION_DEBUG
                std::cout << "BVHModelsDiscreteIntersection::testIntersection(PQPCollisionModelNode, PQPCollisionModelNode): PQPCollisionModels " << e1.getCollisionModel()->getName() << " -- " << e2.getCollisionModel()->getName() << ": "; // << std::endl;
#endif
                const CubeModel::CubeData& cubeData1 = e1.getCollisionModel()->getCubeModel()->getCubeData(0);
                const CubeModel::CubeData& cubeData2 = e2.getCollisionModel()->getCubeModel()->getCubeData(0);
                const Vector3& minVect1 = cubeData1.minBBox;
                const Vector3& minVect2 = cubeData2.minBBox;
                const Vector3& maxVect1 = cubeData1.maxBBox;
                const Vector3& maxVect2 = cubeData2.maxBBox;

				/// TODO: Parameter für AlarmDistance!
                const double alarmDist = 0.25f;

                std::cout << "  minVect1 = " << minVect1 << ", maxVect1 = " << maxVect1 << "; minVect2 = " << minVect2 << ", maxVect2 = " << maxVect2 << "; alarmDist = " << alarmDist << std::endl;

                for (int i = 0; i < 3; i++)
                {
                    if (minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist)
                    {
                        std::cout << minVect1[i] << " > " << maxVect2[i] << " + " << alarmDist << " -- " << minVect2[i] <<  " > " << maxVect1[i] << " + " << alarmDist << std::endl;
                        std::cout << " NO INTERSECTION of AABB's" << std::endl;
                        return false;
                    }
                }

                std::cout << " INTERSECTION DETECTED!!!" << std::endl;
                return true;
            }

            int BVHModelsDiscreteIntersection::computeIntersection(PQPCollisionModelNode& e1, PQPCollisionModelNode& e2, OutputVector* contacts)
            {
#ifdef BVHMODELSINTERSECTION_DEBUG
                std::cout << "BVHModelsDiscreteIntersection::computeIntersection(): PQPCollisionModels " << e1.getCollisionModel()->getName() << " -- " << e2.getCollisionModel()->getName() << std::endl;
#endif
                return 0;
            }

            int BVHModelsDiscreteIntersectionClass = core::RegisterObject("Discrete intersection for BVHModels")
                    .add< BVHModelsDiscreteIntersection >()
                    ;

            SOFA_DECL_CLASS(ObbTreeGPULocalMinDistance)

            int BVHModelsLocalMinDistanceClass = core::RegisterObject("BVHModels specialization of LocalMinDistance")
                    .add< BVHModelsLocalMinDistance >()
                    ;

            BVHModelsLocalMinDistance::BVHModelsLocalMinDistance()
                : LocalMinDistance(),
                  m_checkMode(initData(&m_checkMode, (unsigned int) 1, "PQPCheckMode", "PQP mode (1 = intersection, 2 = RSS, 4 = tolerance; can be combined with binary AND logic (e. g. 7 for all checks))"))
            {
            }

            void BVHModelsLocalMinDistance::init()
            {
                intersectors.add<PQPCollisionModel<>, PQPCollisionModel<>, BVHModelsLocalMinDistance>(this);

#ifdef BVHMODELSINTERSECTION_DEBUG
                std::cout << "BVHModelsLocalMinDistance::init(" << this->getName() << ")" << std::endl;
                testOutputFilename.setValue(sofa::helper::system::DataRepository.getFirstPath() + "/" + this->getName() + ".log");
                std::cout << " testOutputFilename = " << testOutputFilename.getValue() << std::endl;

                testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::trunc);
                testOutput.close();
#endif

                LocalMinDistance::init();
            }

            bool BVHModelsLocalMinDistance::testIntersection(PQPCollisionModelNode& e1, PQPCollisionModelNode &e2)
            {
#ifdef BVHMODELSINTERSECTION_DEBUG
                testOutput << "=== BVHModelsLocalMinDistance::testIntersection(PQPCollisionModelNode, PQPCollisionModelNode): PQPCollisionModels " << e1.getCollisionModel()->getName() << " -- " << e2.getCollisionModel()->getName()  << ", dt = " << this->getTime() << " ===" << std::endl;
#endif

                checkedCollisionModels.push_back(std::make_pair(e1.getCollisionModel()->getName(), e2.getCollisionModel()->getName()));

                const CubeModel::CubeData& cubeData1 = e1.getCollisionModel()->getCubeModel()->getCubeData(0);
                const CubeModel::CubeData& cubeData2 = e2.getCollisionModel()->getCubeModel()->getCubeData(0);
                const Vector3& minVect1 = cubeData1.minBBox;
                const Vector3& minVect2 = cubeData2.minBBox;
                const Vector3& maxVect1 = cubeData1.maxBBox;
                const Vector3& maxVect2 = cubeData2.maxBBox;

                const double alarmDist = getAlarmDistance();

#ifdef BVHMODELSINTERSECTION_DEBUG
                testOutput << "  minVect1 = " << minVect1 << ", maxVect1 = " << maxVect1 << "; minVect2 = " << minVect2 << ", maxVect2 = " << maxVect2 << "; alarmDist = " << alarmDist << std::endl;
#endif

                for (int i = 0; i < 3; i++)
                {
                    if (minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist)
                    {
#ifdef BVHMODELSINTERSECTION_DEBUG
                        testOutput << "  NO AABB Overlap detected -- " << minVect1[i] << " > " << maxVect2[i] << " + " << alarmDist << " -- " << minVect2[i] << " > " << maxVect1[i] << " + " << alarmDist << std::endl;
#endif
                        return false;
                    }
                }

#ifdef BVHMODELSINTERSECTION_DEBUG
                testOutput << "  AABB Intersection DETECTED -- model1 is ghost = " << e1.getCollisionModel()->isGhostObject() << ", model2 is ghost = " << e2.getCollisionModel()->isGhostObject() << std::endl;
                if (e1.getCollisionModel()->isGhostObject() && e2.getCollisionModel()->isGhostObject())
                    testOutput << "  ghost-ghost AABB intersection detected!!!" << std::endl;
#endif
                overlappingCollisionModels.push_back(std::make_pair(e1.getCollisionModel()->getName(), e2.getCollisionModel()->getName()));

                return true;
            }

            void MxVpV(Vector3& Vr, const PQP_REAL M1[3][3], const PQP_REAL V1[3], const PQP_REAL V2[3])
            {
                Vr.x() = (M1[0][0] * V1[0] +
                        M1[0][1] * V1[1] +
                        M1[0][2] * V1[2] +
                        V2[0]);
                Vr.y() = (M1[1][0] * V1[0] +
                        M1[1][1] * V1[1] +
                        M1[1][2] * V1[2] +
                        V2[1]);
                Vr.z() = (M1[2][0] * V1[0] +
                        M1[2][1] * V1[1] +
                        M1[2][2] * V1[2] +
                        V2[2]);
            }

            int BVHModelsLocalMinDistance::computeIntersection(PQPCollisionModelNode& e1, PQPCollisionModelNode& e2, OutputVector* contacts)
            {
#ifdef BVHMODELSINTERSECTION_DEBUG
                std::cout << "BVHModelsLocalMinDistance::computeIntersection(): PQPCollisionModels " << e1.getCollisionModel()->getName() << " -- " << e2.getCollisionModel()->getName() << std::endl;
                std::cout << " BV counts: " << e1.model->getNumBVs() << " -- " << e2.model->getNumBVs() << std::endl;
#endif
                Vector3 pos_m1, pos_m2;
                e1.model->getPosition(pos_m1);
                e2.model->getPosition(pos_m2);

                Matrix3 rot_m1, rot_m2;
                e1.model->getOrientation(rot_m1);
                e2.model->getOrientation(rot_m2);

                PQP_REAL p1[3], p2[3];
                p1[0] = pos_m1.x(); p1[1] = pos_m1.y(); p1[2] = pos_m1.z();
                p2[0] = pos_m2.x(); p2[1] = pos_m2.y(); p2[2] = pos_m2.z();

                PQP_REAL r1[3][3], r2[3][3];

                for (unsigned short k = 0; k < 3; k++)
                {
                    for (unsigned short l = 0; l < 3; l++)
                    {
                        r1[k][l] = rot_m1(k, l);
                        r2[k][l] = rot_m2(k, l);
                    }
                }

                std::cout << "=== CHECK MODE = " << m_checkMode.getValue() << " ===" << std::endl;
                std::cout << "m_checkMode.getValue() & PQP_MODE_INTERSECTION = " << (m_checkMode.getValue() & PQP_MODE_INTERSECTION) << std::endl;
                std::cout << "m_checkMode.getValue() & PQP_MODE_CLOSEST_POINT = " << (m_checkMode.getValue() & PQP_MODE_CLOSEST_POINT) << std::endl;
                std::cout << "m_checkMode.getValue() & PQP_MODE_TOLERANCE = " << (m_checkMode.getValue() & PQP_MODE_TOLERANCE) << std::endl;

                int ret_val = 0;

                if ((m_checkMode.getValue() & PQP_MODE_INTERSECTION) == PQP_MODE_INTERSECTION)
                {
                    std::cout << " CALL intersection" << std::endl;
                    PQP_CollideResult pqp_res;
                    int pqp_status = PQP_Collide(&pqp_res, r1, p1, e1.model->getPQPModel(), r2, p2, e2.model->getPQPModel(), PQP_ALL_CONTACTS);

                    std::cout << "  result = " << pqp_status << std::endl;

                    if (pqp_status == PQP_OK)
                    {
                        std::cout << "PQP_Collide result " << e1.model->getName() << " - " << e2.model->getName() << " = " << pqp_res.NumPairs() << "; OBB pairs checked = " << pqp_res.NumBVTests() << "; triangle pairs checked = " << pqp_res.NumTriTests() << std::endl;
                        ret_val = pqp_res.NumPairs();
                        std::pair<std::string, std::string> checkedPair = std::make_pair(e1.model->getName(), e2.model->getName());
                        collideResults[checkedPair] = pqp_res;
                    }
                }

                if ((m_checkMode.getValue() & PQP_MODE_CLOSEST_POINT) == PQP_MODE_CLOSEST_POINT)
                {
                    std::cout << " CALL distance" << std::endl;
                    PQP_DistanceResult pqp_res;
                    int pqp_status = PQP_Distance(&pqp_res, r1, p1, e1.model->getPQPModel(), r2, p2, e2.model->getPQPModel(), 0.0f, 0.0f, 512);

                    std::cout << "  result = " << pqp_status << std::endl;

                    if (pqp_status == PQP_OK)
                    {
                        std::pair<std::string, std::string> checkedPair = std::make_pair(e1.model->getName(), e2.model->getName());
                        std::cout << "PQP_Distance result = " << e1.model->getName() << " - " << e2.model->getName() << " = " << pqp_res.Distance() << std::endl;

                        Vector3 p1_tr, p2_tr;
                        MxVpV(p1_tr, r1, pqp_res.p1, p1);
                        MxVpV(p2_tr, r2, pqp_res.p2, p2);

                        pqp_res.p1[0] = p1_tr.x(); pqp_res.p1[1] = p1_tr.y(); pqp_res.p1[2] = p1_tr.z();
                        pqp_res.p2[0] = p2_tr.x(); pqp_res.p2[1] = p2_tr.y(); pqp_res.p2[2] = p2_tr.z();

                        std::cout << " p1 = " << p1_tr << "; " << "p2 = " << p2_tr << "," << std::endl;
                        distanceResults[checkedPair] = pqp_res;
                    }
                }

                if ((m_checkMode.getValue() & PQP_MODE_TOLERANCE) == PQP_MODE_TOLERANCE)
                {
                    std::cout << " CALL tolerance" << std::endl;
                    PQP_ToleranceResult pqp_res;
                    int pqp_status = PQP_Tolerance(&pqp_res, r1, p1, e1.model->getPQPModel(), r2, p2, e2.model->getPQPModel(), 0.25f);

                    std::cout << "  result = " << pqp_status << std::endl;

                    if (pqp_status == PQP_OK)
                    {
                        std::pair<std::string, std::string> checkedPair = std::make_pair(e1.model->getName(), e2.model->getName());
                        toleranceResults[checkedPair] = pqp_res;
                    }
                }

                return ret_val;
            }

            void BVHModelsLocalMinDistance::beginBroadPhase()
            {
#ifdef BVHMODELSINTERSECTION_DEBUG
                testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::ate | std::ofstream::app);
                testOutput << "=== BVHModelsLocalMinDistance::beginBroadPhase(" << this->getName() << "), dt = " << this->getTime() << " ===" << std::endl;
#endif
                checkedCollisionModels.clear();
                overlappingCollisionModels.clear();
                collideResults.clear();
                distanceResults.clear();
                toleranceResults.clear();
            }

            void BVHModelsLocalMinDistance::endBroadPhase()
            {
#ifdef BVHMODELSINTERSECTION_DEBUG
                testOutput << "=== BVHModelsLocalMinDistance::endBroadPhase(" << this->getName() << "), dt = " << this->getTime() << " ===" << std::endl;

                testOutput << "AABB pairs tested in broad-phase: " << checkedCollisionModels.size() << std::endl;
                for (std::vector<std::pair<std::string, std::string> >::const_iterator it = checkedCollisionModels.begin(); it != checkedCollisionModels.end(); it++)
                    testOutput << " - " << it->first << " -- " << it->second << std::endl;

                testOutput << "AABB pairs overlapping in broad-phase: " << overlappingCollisionModels.size() << std::endl;
                for (std::vector<std::pair<std::string, std::string> >::const_iterator it = overlappingCollisionModels.begin(); it != overlappingCollisionModels.end(); it++)
                    testOutput << " - " << it->first << " -- " << it->second << std::endl;

                testOutput.close();
#endif
            }

            void BVHModelsLocalMinDistance::draw(const core::visual::VisualParams *vparams)
            {
                //std::cout << "BVHModelsLocalMinDistance::draw(" << this->getName() << "): distanceResults = " << distanceResults.size() << ", toleranceResults = " << toleranceResults.size() << std::endl;
                glPointSize(25.0f);
                glLineWidth(20.0f);

                glPushMatrix();
                glPushAttrib(GL_ENABLE_BIT);
                glEnable(GL_LIGHTING);
                glEnable(GL_COLOR_MATERIAL);

                for (std::map<std::pair<std::string, std::string>, PQP_DistanceResult>::const_iterator it = distanceResults.begin(); it != distanceResults.end(); it++)
                {
                    /*Vector3 p1_tr, p2_tr;
                    MxVpV(p1_tr, it->second.R, it->second.p1, it->second.T);
                    MxVpV(p2_tr, it->second.R, it->second.p2, it->second.T);*/
                    glBegin(GL_POINTS);
                    glColor4f(0.5f, 0.5f, 1.0f, 1.0f);
                    glVertex3f(it->second.p1[0], it->second.p1[1], it->second.p1[2]);
                    glColor4f(0.5f, 1.0f, 0.5f, 1.0f);
                    glVertex3f(it->second.p2[0], it->second.p2[1], it->second.p2[2]);
                    glEnd();

                    glBegin(GL_LINES);
                    glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
                    glVertex3f(it->second.p1[0], it->second.p1[1], it->second.p1[2]);
                    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                    glVertex3f(it->second.p2[0], it->second.p2[1], it->second.p2[2]);
                    glEnd();
                }

                for (std::map<std::pair<std::string, std::string>, PQP_ToleranceResult>::const_iterator it = toleranceResults.begin(); it != toleranceResults.end(); it++)
                {
                    /*Vector3 p1_tr, p2_tr;
                    MxVpV(p1_tr, it->second.R, it->second.p1, it->second.T);
                    MxVpV(p2_tr, it->second.R, it->second.p2, it->second.T);*/
                    glBegin(GL_POINTS);
                    glColor4f(0.5f, 1.0f, 0.5f, 1.0f);
                    glVertex3f(it->second.p1[0], it->second.p1[1], it->second.p1[2]);
                    glColor4f(1.0f, 0.5f, 0.5f, 1.0f);
                    glVertex3f(it->second.p2[0], it->second.p2[1], it->second.p2[2]);
                    glEnd();

                    glBegin(GL_LINES);
                    if (it->second.closer_than_tolerance)
                        glColor4f(1.0f, 0.5f, 0.5f, 1.0f);
                    else
                        glColor4f(0.5f, 1.0f, 0.5f, 1.0f);

                    glVertex3f(it->second.p1[0], it->second.p1[1], it->second.p1[2]);

                    if (it->second.closer_than_tolerance)
                        glColor4f(1.0f, 0.5f, 0.5f, 1.0f);
                    else
                        glColor4f(0.5f, 1.0f, 0.5f, 1.0f);

                    glVertex3f(it->second.p2[0], it->second.p2[1], it->second.p2[2]);
                    glEnd();
                }

                glPopAttrib();
                glPopMatrix();

                glPointSize(1.0f);
                glLineWidth(1.0f);
            }
        }
    }
}
