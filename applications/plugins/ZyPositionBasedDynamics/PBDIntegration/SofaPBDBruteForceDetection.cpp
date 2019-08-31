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
                    sofa::helper::vector<DetectionOutput> unique_outputs;
                    for (size_t m = 0; m < outputs->size(); m++)
                    {
                        sofa::core::collision::DetectionOutput detection = contactPoints->operator[](m);
                        /*msg_info("SofaPBDBruteForceDetection") << "Contact " << m << ": ID = " << detection.id << ", point0 = " << detection.point[0] << ", point1 = " << detection.point[1]
                                                               << ", normal = " << detection.normal
                                                               << ", features = " << detection.elem.first.getIndex() << " -- " << detection.elem.second.getIndex()
                                                               << ", distance = " << detection.value;*/

                        if (std::find(unique_contact_ids.begin(), unique_contact_ids.end(), detection.id) == unique_contact_ids.end())
                        {
                            unique_contact_ids.emplace_back(detection.id);
                            unique_outputs.push_back(detection);
                        }
                    }

                    /*msg_info("SofaPBDBruteForceDetection") << "Unique contacts: " << unique_contact_ids.size();
                    for (std::map<int64_t, DetectionOutput>::const_iterator ct_it = unique_outputs.begin(); ct_it != unique_outputs.end(); ct_it++)
                        msg_info("SofaPBDBruteForceDetection") << "Contact: ID = " << ct_it->second.id
                                                               << ", point0 = " << ct_it->second.point[0]
                                                               << ", point1 = " << ct_it->second.point[1]
                                                               << ", normal = " << ct_it->second.normal
                                                               << ", features = " << ct_it->second.elem.first.getIndex() << " -- " << ct_it->second.elem.second.getIndex()
                                                               << ", distance = " << ct_it->second.value;*/

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
    for (std::map<std::pair<core::CollisionModel*, core::CollisionModel*>, sofa::helper::vector<sofa::core::collision::DetectionOutput>>::const_iterator cm_it = collisionOutputs.begin(); cm_it != collisionOutputs.end(); cm_it++)
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

const std::map<std::pair<core::CollisionModel*, core::CollisionModel*>, sofa::helper::vector<sofa::core::collision::DetectionOutput>>& SofaPBDBruteForceDetection::getCollisionOutputs() const
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
    for (std::map<std::pair<core::CollisionModel*, core::CollisionModel*>, sofa::helper::vector<sofa::core::collision::DetectionOutput>>::const_iterator cm_it = collisionOutputs.begin(); cm_it != collisionOutputs.end(); cm_it++)
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
