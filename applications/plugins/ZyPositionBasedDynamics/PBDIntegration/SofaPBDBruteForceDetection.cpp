#include "SofaPBDBruteForceDetection.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/DetectionOutput.h>

#include <sofa/core/visual/VisualParams.h>

#include <PBDIntegration/SofaPBDPointCollisionModel.h>
#include <PBDIntegration/SofaPBDLineCollisionModel.h>
#include <PBDIntegration/SofaPBDTriangleCollisionModel.h>

using namespace sofa::component::collision;
using namespace sofa::core::collision;
using namespace sofa;

using namespace sofa::simulation::PBDSimulation;

SOFA_DECL_CLASS(SofaPBDBruteForceDetection)

int SofaPBDBruteForceDetectionClass = sofa::core::RegisterObject("Wrapper for SOFA's BruteForceDetection component, linking it to the PBD plugin.")
        .add<SofaPBDBruteForceDetection>()
        ;

SofaPBDBruteForceDetection::SofaPBDBruteForceDetection(): sofa::component::collision::BruteForceDetection()
{
    this->f_printLog.setValue(true);
}

void SofaPBDBruteForceDetection::addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& coll_pair)
{
    if (this->f_printLog.getValue())
        msg_info("SofaPBDBruteForceDetection") << "addCollisionPair(" << coll_pair.first->getName() << ", " << coll_pair.second->getName() << ")";

    std::pair<std::string, std::string> coll_pair_names = std::make_pair(coll_pair.first->getName(), coll_pair.second->getName());
    if (std::find(checkedCollisionModels.begin(), checkedCollisionModels.end(), coll_pair_names) == checkedCollisionModels.end())
        checkedCollisionModels.emplace_back(coll_pair_names);

    BruteForceDetection::addCollisionPair(coll_pair);
}

void SofaPBDBruteForceDetection::beginBroadPhase()
{
    if (this->f_printLog.getValue())
        msg_info("SofaPBDBruteForceDetection") << "beginBroadPhase(" << this->getName() << "), dt = " << this->getTime();

    checkedCollisionModels.clear();
    overlappingCollisionModels.clear();

    BruteForceDetection::beginBroadPhase();
}

void SofaPBDBruteForceDetection::endBroadPhase()
{
    if (this->f_printLog.getValue())
    {
        msg_info("SofaPBDBruteForceDetection") << "endBroadPhase(" << this->getName() << "), dt = " << this->getTime();

        msg_info("SofaPBDBruteForceDetection") << "AABB pairs tested in broad-phase: " << checkedCollisionModels.size();
        for (std::vector<std::pair<std::string, std::string> >::const_iterator it = checkedCollisionModels.begin(); it != checkedCollisionModels.end(); it++)
            msg_info("SofaPBDBruteForceDetection") << " - " << it->first << " -- " << it->second;

        msg_info("SofaPBDBruteForceDetection") << "AABB pairs overlapping in broad-phase: " << overlappingCollisionModels.size();
        for (std::vector<std::pair<std::string, std::string> >::const_iterator it = overlappingCollisionModels.begin(); it != overlappingCollisionModels.end(); it++)
            msg_info("SofaPBDBruteForceDetection") << " - " << it->first << " -- " << it->second;
    }
    BruteForceDetection::endBroadPhase();
}

void SofaPBDBruteForceDetection::beginNarrowPhase()
{
    if (this->f_printLog.getValue())
    {
        msg_info("SofaPBDBruteForceDetection") << "beginNarrowPhase(" << this->getName() << "), dt = " << this->getTime();

        msg_info("SofaPBDBruteForceDetection") << "AABB pairs tested in broad-phase: " << checkedCollisionModels.size();
        for (std::vector<std::pair<std::string, std::string> >::const_iterator it = checkedCollisionModels.begin(); it != checkedCollisionModels.end(); it++)
            msg_info("SofaPBDBruteForceDetection") << " - " << it->first << " -- " << it->second;

        msg_info("SofaPBDBruteForceDetection") << "AABB pairs overlapping in broad-phase: " << overlappingCollisionModels.size();
        for (std::vector<std::pair<std::string, std::string> >::const_iterator it = overlappingCollisionModels.begin(); it != overlappingCollisionModels.end(); it++)
            msg_info("SofaPBDBruteForceDetection") << " - " << it->first << " -- " << it->second;
    }

    collisionOutputs.clear();

    NarrowPhaseDetection::beginNarrowPhase();
}

void SofaPBDBruteForceDetection::endNarrowPhase()
{
    //if (this->f_printLog.getValue())
    {
        msg_info("SofaPBDBruteForceDetection") << "endNarrowPhase(" << this->getName() << "), dt = " << this->getTime();

        msg_info("SofaPBDBruteForceDetection") << "AABB pairs tested in broad-phase: " << checkedCollisionModels.size();
        for (std::vector<std::pair<std::string, std::string> >::const_iterator it = checkedCollisionModels.begin(); it != checkedCollisionModels.end(); it++)
            msg_info("SofaPBDBruteForceDetection") << " - " << it->first << " -- " << it->second;

        msg_info("SofaPBDBruteForceDetection") << "AABB pairs overlapping in broad-phase: " << overlappingCollisionModels.size();
        for (std::vector<std::pair<std::string, std::string> >::const_iterator it = overlappingCollisionModels.begin(); it != overlappingCollisionModels.end(); it++)
            msg_info("SofaPBDBruteForceDetection") << " - " << it->first << " -- " << it->second;

        const DetectionOutputMap& detectionOutputs = getDetectionOutputs();
        msg_info("SofaPBDBruteForceDetection") << "DetectionOutputMap entries: " << detectionOutputs.size();
        for (DetectionOutputMap::const_iterator it = detectionOutputs.begin(); it != detectionOutputs.end(); it++)
        {
            const std::pair< core::CollisionModel*, core::CollisionModel* >& cmPair = it->first;
            const DetectionOutputVector* outputs = it->second;

            const sofa::helper::vector<DetectionOutput>* contactPoints = dynamic_cast<const sofa::helper::vector<DetectionOutput>*>(outputs);

            msg_info("SofaPBDBruteForceDetection") << "Object pair " << cmPair.first->getName() << " -- " << cmPair.second->getName() << ": " << outputs->size() << " contacts.";

            if (contactPoints)
            {
                if (outputs->size() > 0)
                {
                    std::vector<int64_t> unique_contact_ids;
                    sofa::helper::vector<SofaPBDCollisionDetectionOutput> unique_outputs;
                    for (size_t m = 0; m < outputs->size(); m++)
                    {
                        sofa::core::collision::DetectionOutput detection = contactPoints->operator[](m);

                        msg_info("SofaPBDBruteForceDetection") << "Contact " << m << ": ID = " << detection.id << ", point0 = " << detection.point[0] << ", point1 = " << detection.point[1]
                                                               << ", normal = " << detection.normal
                                                               << ", features = " << detection.elem.first.getIndex() << " -- " << detection.elem.second.getIndex()
                                                               << ", distance = " << detection.value;

                        bool cpAlreadyFound = false;
                        for (std::vector<SofaPBDCollisionDetectionOutput>::const_iterator ct_it = unique_outputs.begin(); ct_it != unique_outputs.end(); ct_it++)
                        {
                            // TODO: Distinguish between point, line and triangle models: These report different element indices for contact point pairs with identical point/normal/distance values!
                            if (ct_it->point[0] == detection.point[0] &&
                                ct_it->point[1] == detection.point[1] &&
                                ct_it->normal == detection.normal &&
                                ct_it->value == detection.value /*&&
                                (ct_it->elem.first.getIndex() == detection.elem.first.getIndex() ||
                                 ct_it->elem.first.getIndex() == detection.elem.second.getIndex() ||
                                 ct_it->elem.second.getIndex() == detection.elem.first.getIndex() ||
                                 ct_it->elem.second.getIndex() == detection.elem.second.getIndex())*/)
                            {
                                msg_info("SofaPBDBruteForceDetection") << "Contact point with equal attributes already registered, skipping ID " << detection.id;
                                cpAlreadyFound = true;
                                break;
                            }
                        }

                        if (cpAlreadyFound)
                            continue;

                        if (std::find(unique_contact_ids.begin(), unique_contact_ids.end(), detection.id) == unique_contact_ids.end())
                        {
                            unique_contact_ids.emplace_back(detection.id);
                            SofaPBDCollisionDetectionOutput pbd_output(detection);

                            bool cm1IsPBDPointModel = cmPair.first->hasTag(tagPBDPointCollisionModel);
                            bool cm2IsPBDPointModel = cmPair.second->hasTag(tagPBDPointCollisionModel);
                            bool cm1IsPBDLineModel = cmPair.first->hasTag(tagPBDLineCollisionModel);
                            bool cm2IsPBDLineModel = cmPair.second->hasTag(tagPBDLineCollisionModel);
                            bool cm1IsPBDTriangleModel = cmPair.first->hasTag(tagPBDTriangleCollisionModel);
                            bool cm2IsPBDTriangleModel = cmPair.second->hasTag(tagPBDTriangleCollisionModel);

                            // Distinguish contact types: Rigid/Rigid, Rigid/Line, Rigid/Particle, Solid/Particle
                            SofaPBDPointCollisionModel* pm1 = nullptr;
                            SofaPBDPointCollisionModel* pm2 = nullptr;
                            SofaPBDLineCollisionModel* lm1 = nullptr;
                            SofaPBDLineCollisionModel* lm2 = nullptr;
                            SofaPBDTriangleCollisionModel* tm1 = nullptr;
                            SofaPBDTriangleCollisionModel* tm2 = nullptr;

                            if (cm1IsPBDPointModel)
                            {
                                pm1 = dynamic_cast<SofaPBDPointCollisionModel*>(cmPair.first);
                            }

                            if (cm2IsPBDPointModel)
                            {
                                pm2 = dynamic_cast<SofaPBDPointCollisionModel*>(cmPair.second);
                            }

                            if (cm1IsPBDLineModel)
                            {
                                lm1 = dynamic_cast<SofaPBDLineCollisionModel*>(cmPair.first);
                            }

                            if (cm2IsPBDLineModel)
                            {
                                lm2 = dynamic_cast<SofaPBDLineCollisionModel*>(cmPair.second);
                            }

                            if (cm1IsPBDTriangleModel)
                            {
                                tm1 = dynamic_cast<SofaPBDTriangleCollisionModel*>(cmPair.first);
                            }

                            if (cm2IsPBDTriangleModel)
                            {
                                tm2 = dynamic_cast<SofaPBDTriangleCollisionModel*>(cmPair.second);
                            }

                            // Distinguish contact model pairings

                            // Point vs. point model
                            if (cm1IsPBDPointModel && cm2IsPBDPointModel)
                            {
                                pbd_output.modelPairType = PBD_CONTACT_PAIR_POINT_POINT;
                                // Point vs. point should be rigid-rigid contacts exclusively
                                pbd_output.contactType = PBD_RIGID_RIGID_CONTACT;

                                pbd_output.rigidBodyIndices[0] = pm1->getPBDRigidBodyIndex();
                                pbd_output.rigidBodyIndices[1] = pm2->getPBDRigidBodyIndex();
                            }

                            // Line vs. line model
                            if (cm1IsPBDLineModel && cm2IsPBDLineModel)
                            {
                                pbd_output.modelPairType = PBD_CONTACT_PAIR_LINE_LINE;

                                // Thread segments: Self-intersection - consider this a rigid vs. rigid contact, unless something else is required
                                if (lm1->usesPBDLineModel() && lm2->usesPBDLineModel())
                                {
                                    pbd_output.contactType = PBD_LINE_LINE_CONTACT;
                                }
                                // Rigid body pair
                                else if (lm1->usesPBDRigidBody() && lm2->usesPBDRigidBody())
                                {
                                    pbd_output.contactType = PBD_RIGID_RIGID_CONTACT;
                                    pbd_output.rigidBodyIndices[0] = lm1->getPBDRigidBodyIndex();
                                    pbd_output.rigidBodyIndices[1] = lm2->getPBDRigidBodyIndex();
                                }
                                // Thread segment vs. rigid body
                                else
                                {
                                    pbd_output.contactType = PBD_RIGID_LINE_CONTACT;
                                    if (lm1->usesPBDLineModel() && lm2->usesPBDRigidBody())
                                    {
                                        pbd_output.rigidBodyIndices[1] = lm2->getPBDRigidBodyIndex();
                                    }
                                    else if (lm1->usesPBDRigidBody() && lm2->usesPBDLineModel())
                                    {
                                        pbd_output.rigidBodyIndices[0] = lm1->getPBDRigidBodyIndex();
                                    }
                                }
                            }

                            // Triangle vs. triangle model
                            if (cm1IsPBDTriangleModel && cm2IsPBDTriangleModel)
                            {
                                pbd_output.modelPairType = PBD_CONTACT_PAIR_TRIANGLE_TRIANGLE;
                                // This can only be a rigid-body pair
                                pbd_output.contactType = PBD_RIGID_RIGID_CONTACT;

                                pbd_output.rigidBodyIndices[0] = tm1->getPBDRigidBodyIndex();
                                pbd_output.rigidBodyIndices[1] = tm2->getPBDRigidBodyIndex();
                            }

                            // Point vs. line model
                            if ((cm1IsPBDPointModel && cm2IsPBDLineModel) || (cm2IsPBDPointModel && cm1IsPBDLineModel))
                            {
                                pbd_output.modelPairType = PBD_CONTACT_PAIR_POINT_LINE;

                                // This can also only be rigid vs. rigid body, right?
                                pbd_output.contactType = PBD_RIGID_RIGID_CONTACT;

                                if (cm2IsPBDPointModel && cm1IsPBDLineModel)
                                {
                                    pbd_output.modelTypeMirrored = true;
                                    pbd_output.rigidBodyIndices[0] = pm2->getPBDRigidBodyIndex();
                                    pbd_output.rigidBodyIndices[1] = lm1->getPBDRigidBodyIndex();
                                }
                                else if (cm1IsPBDPointModel && cm2IsPBDLineModel)
                                {
                                    pbd_output.rigidBodyIndices[0] = pm1->getPBDRigidBodyIndex();
                                    pbd_output.rigidBodyIndices[1] = lm2->getPBDRigidBodyIndex();
                                }
                            }

                            if ((cm1IsPBDPointModel && cm2IsPBDTriangleModel) || (cm2IsPBDPointModel && cm1IsPBDTriangleModel))
                            {
                                pbd_output.modelPairType = PBD_CONTACT_PAIR_POINT_TRIANGLE;

                                // This can also only be rigid vs. rigid body, right?
                                pbd_output.contactType = PBD_RIGID_RIGID_CONTACT;

                                if (cm2IsPBDPointModel && cm1IsPBDTriangleModel)
                                {
                                    pbd_output.modelTypeMirrored = true;
                                    pbd_output.rigidBodyIndices[0] = pm2->getPBDRigidBodyIndex();
                                    pbd_output.rigidBodyIndices[1] = tm1->getPBDRigidBodyIndex();
                                }
                                else if (cm1IsPBDPointModel && cm2IsPBDTriangleModel)
                                {
                                    pbd_output.rigidBodyIndices[0] = tm1->getPBDRigidBodyIndex();
                                    pbd_output.rigidBodyIndices[1] = pm2->getPBDRigidBodyIndex();
                                }
                            }

                            // Line vs. triangle models
                            if ((cm1IsPBDLineModel && cm2IsPBDTriangleModel) || (cm2IsPBDLineModel && cm2IsPBDTriangleModel))
                            {
                                pbd_output.modelPairType = PBD_CONTACT_PAIR_LINE_TRIANGLE;

                                if (cm2IsPBDLineModel && cm2IsPBDTriangleModel)
                                {
                                    pbd_output.modelTypeMirrored = true;
                                }

                                // Depending on whether one of the line models represents a thread or a rigid body model, set the contactType accordingly
                                if (cm1IsPBDLineModel)
                                {
                                    if (lm1->usesPBDLineModel())
                                    {
                                        pbd_output.contactType = PBD_RIGID_LINE_CONTACT;
                                        // TODO: Retrieve particle indices involved in collision
                                    }
                                    else if (lm1->usesPBDRigidBody())
                                    {
                                        pbd_output.contactType = PBD_RIGID_RIGID_CONTACT;
                                        pbd_output.rigidBodyIndices[0] = lm1->getPBDRigidBodyIndex();
                                        pbd_output.rigidBodyIndices[1] = tm2->getPBDRigidBodyIndex();
                                    }
                                }
                                if (cm2IsPBDLineModel)
                                {
                                    if (lm2->usesPBDLineModel())
                                    {
                                        pbd_output.contactType = PBD_RIGID_LINE_CONTACT;
                                    }
                                    else if (lm2->usesPBDRigidBody())
                                    {
                                        pbd_output.contactType = PBD_RIGID_RIGID_CONTACT;
                                        pbd_output.rigidBodyIndices[0] = tm1->getPBDRigidBodyIndex();
                                        pbd_output.rigidBodyIndices[1] = lm2->getPBDRigidBodyIndex();
                                    }
                                }
                            }

                            unique_outputs.push_back(pbd_output);
                        }
                    }

                    msg_info("SofaPBDBruteForceDetection") << "Unique contacts after filtering out duplicates: " << unique_contact_ids.size();
                    unsigned long contact_idx = 0;
                    for (std::vector<SofaPBDCollisionDetectionOutput>::const_iterator ct_it = unique_outputs.begin(); ct_it != unique_outputs.end(); ct_it++)
                    {
                        msg_info("SofaPBDBruteForceDetection") << "Contact " << contact_idx << ": ID = " << ct_it->id
                                                               << ", contactType = " << ct_it->contactType
                                                               << ", modelPairType = " << ct_it->modelPairType
                                                               << ", modelTypeMirrored = " << ct_it->modelTypeMirrored
                                                               << ", point0 = " << ct_it->point[0]
                                                               << ", point1 = " << ct_it->point[1]
                                                               << ", normal = " << ct_it->normal
                                                               << ", features = " << ct_it->elem.first.getIndex() << " -- " << ct_it->elem.second.getIndex()
                                                               << ", distance = " << ct_it->value;
                        contact_idx++;
                    }
                    collisionOutputs.insert(std::make_pair(cmPair, unique_outputs));
                }
            }
            // This else branch should not be necessary, if only PBD vs. PBD models are handled
            // It might become relevant if handling collisions between PBD-based and non-PBD-based objects
            /*else
            {
                msg_info("SofaPBDBruteForceDetection") << "Casting to DetectionOutputVector failed. Trying known collision model permutations.";
                const sofa::core::collision::TDetectionOutputVector<SofaPBDTriangleCollisionModel, SofaPBDTriangleCollisionModel>* triPair_Contacts =
                dynamic_cast<const sofa::core::collision::TDetectionOutputVector<SofaPBDTriangleCollisionModel, SofaPBDTriangleCollisionModel>*>(outputs);
                const sofa::core::collision::TDetectionOutputVector<SofaPBDTriangleCollisionModel, SofaPBDPointCollisionModel>* triPoint_Contacts =
                dynamic_cast<const sofa::core::collision::TDetectionOutputVector<SofaPBDTriangleCollisionModel, SofaPBDPointCollisionModel>*>(outputs);
                const sofa::core::collision::TDetectionOutputVector<SofaPBDTriangleCollisionModel, SofaPBDLineCollisionModel>* triLine_Contacts =
                dynamic_cast<const sofa::core::collision::TDetectionOutputVector<SofaPBDTriangleCollisionModel, SofaPBDLineCollisionModel>*>(outputs);

                if (triPair_Contacts)
                {
                    msg_info("SofaPBDBruteForceDetection") << "Triangle - triangle PBD contacts: " << triPair_Contacts->size();
                }
                if (triLine_Contacts)
                {
                    msg_info("SofaPBDBruteForceDetection") << "Triangle - line PBD contacts: " << triLine_Contacts->size();
                }
                if (triPoint_Contacts)
                {
                    msg_info("SofaPBDBruteForceDetection") << "Triangle - point PBD contacts: " << triPoint_Contacts->size();
                }
            }*/
        }
    }

    msg_info("SofaPBDBruteForceDetection") << "collisionOutputs map size: " << collisionOutputs.size();
    for (std::map<std::pair<core::CollisionModel*, core::CollisionModel*>, sofa::helper::vector<sofa::core::collision::SofaPBDCollisionDetectionOutput>>::const_iterator cm_it = collisionOutputs.begin(); cm_it != collisionOutputs.end(); cm_it++)
    {
        msg_info("SofaPBDBruteForceDetection") << cm_it->first.first->getName() << " -- " << cm_it->first.second->getName() << ": " << cm_it->second.size() << " contacts.";
        for (size_t r = 0; r < cm_it->second.size(); r++)
        {
            msg_info("SofaPBDBruteForceDetection") << "Contact: ID = " << cm_it->second[r].id
                                                   << ", point0 = " << cm_it->second[r].point[0]
                                                   << ", point1 = " << cm_it->second[r].point[1]
                                                   << ", normal = " << cm_it->second[r].normal
                                                   << ", features = " << cm_it->second[r].elem.first.getIndex() << " -- " << cm_it->second[r].elem.second.getIndex()
                                                   << ", distance = " << cm_it->second[r].value;
        }
    }

    NarrowPhaseDetection::endNarrowPhase();
}

const std::map<std::pair<core::CollisionModel *, core::CollisionModel *>, sofa::helper::vector<SofaPBDCollisionDetectionOutput> > &SofaPBDBruteForceDetection::getCollisionOutputs() const
{
    return collisionOutputs;
}

void SofaPBDBruteForceDetection::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehavior())
        return;

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->enableLighting();

    sofa::defaulttype::Vec4f normalArrowColor(0.8f, 0.2f, 0.2f, 0.9f);
    for (std::map<std::pair<core::CollisionModel*, core::CollisionModel*>, sofa::helper::vector<sofa::core::collision::SofaPBDCollisionDetectionOutput>>::const_iterator cm_it = collisionOutputs.begin(); cm_it != collisionOutputs.end(); cm_it++)
    {
        for (size_t r = 0; r < cm_it->second.size(); r++)
        {
            vparams->drawTool()->drawSphere(cm_it->second[r].point[0], 0.015f);
            vparams->drawTool()->drawSphere(cm_it->second[r].point[1], 0.015f);
            vparams->drawTool()->drawArrow(cm_it->second[r].point[0], cm_it->second[r].point[0] + cm_it->second[r].normal, 0.01f, 0.01f, normalArrowColor, 8);
        }
    }

    vparams->drawTool()->disableLighting();
    vparams->drawTool()->restoreLastState();
}
