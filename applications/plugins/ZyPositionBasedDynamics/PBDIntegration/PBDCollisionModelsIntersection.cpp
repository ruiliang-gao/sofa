#include "PBDCollisionModelsIntersection.h"

#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/helper/proximity.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/collision/Intersection.inl>
#include <SofaBaseCollision/MinProximityIntersection.h>
#include <sofa/core/visual/VisualParams.h>

#include <iostream>
#include <algorithm>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/IntersectorFactory.h>

#include <GL/gl.h>

#include "SofaPBDPointCollisionModel.h"
#include "SofaPBDLineCollisionModel.h"
#include "SofaPBDTriangleCollisionModel.h"

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace sofa::component::collision;
using namespace sofa::simulation::PBDSimulation;

int PBDCollisionModelsIntersectionClass = sofa::core::RegisterObject("Wrapper for SOFA's default collision detection, linking it to the PBD plugin.")
        .add< PBDCollisionModelsIntersection >()
        ;

//IntersectorCreator<DiscreteIntersection, PBDCollisionModelsIntersection> PBDCollisionModelsDiscreteIntersectors("PBDCollisionModels");

PBDCollisionModelsIntersection::PBDCollisionModelsIntersection(/*DiscreteIntersection *object*/): BaseProximityIntersection()
{
//    if (object)
//        msg_info("PBDCollisionModelsIntersection") << "DiscreteIntersection instance provided: " << object->getName();
//    else
//        msg_info("PBDCollisionModelsIntersection") << "No DiscreteIntersection instance provided.";
}

void PBDCollisionModelsIntersection::init()
{
    intersectors.add<SofaPBDPointCollisionModel, SofaPBDPointCollisionModel, PBDCollisionModelsIntersection>(this);
    intersectors.add<SofaPBDLineCollisionModel, SofaPBDPointCollisionModel, PBDCollisionModelsIntersection>(this);
    intersectors.add<SofaPBDLineCollisionModel, SofaPBDLineCollisionModel, PBDCollisionModelsIntersection>(this);
    intersectors.add<SofaPBDTriangleCollisionModel, SofaPBDPointCollisionModel, PBDCollisionModelsIntersection>(this);
    intersectors.add<SofaPBDTriangleCollisionModel, SofaPBDLineCollisionModel, PBDCollisionModelsIntersection>(this);
    intersectors.add<SofaPBDTriangleCollisionModel, SofaPBDTriangleCollisionModel, PBDCollisionModelsIntersection>(this);

    const std::vector<std::pair<helper::TypeInfo, helper::TypeInfo> > intersectorPairs = intersectors.getIntersectors();

    msg_info("PBDCollisionModelsIntersection") << "Registered intersector pairings: " << intersectorPairs.size();
    for (size_t k = 0; k < intersectorPairs.size(); k++)
        msg_info("PBDCollisionModelsIntersection") << k << ": " << intersectorPairs[k].first.pt->name() << " -- " << intersectorPairs[k].second.pt->name();
}

//SOFA_DECL_CLASS(PBDCollisionModelsLocalMinDistance)

int PBDCollisionModelsLMDNewDistanceClass = sofa::core::RegisterObject("BVHModels specialization of LocalMinDistance")
        .add< PBDCollisionModelsLocalMinDistance >()
        ;

PBDCollisionModelsLocalMinDistance::PBDCollisionModelsLocalMinDistance()
    : LMDNewProximityIntersection()
{
}

void PBDCollisionModelsLocalMinDistance::init()
{
    msg_info("PBDCollisionModelsIntersection") << "PBDCollisionModelsLocalMinDistance::init(" << this->getName() << ")";

    LMDNewProximityIntersection::init();
}

void PBDCollisionModelsLocalMinDistance::beginBroadPhase()
{
    msg_info("PBDCollisionModelsIntersection") << "PBDCollisionModelsLocalMinDistance::beginBroadPhase(" << this->getName() << "), dt = " << this->getTime();

    checkedCollisionModels.clear();
    overlappingCollisionModels.clear();

    LMDNewProximityIntersection::beginBroadPhase();
}

void PBDCollisionModelsLocalMinDistance::endBroadPhase()
{
    msg_info("PBDCollisionModelsIntersection") << "PBDCollisionModelsLocalMinDistance::endBroadPhase(" << this->getName() << "), dt = " << this->getTime();

    msg_info("PBDCollisionModelsIntersection") << "AABB pairs tested in broad-phase: " << checkedCollisionModels.size();
    for (std::vector<std::pair<std::string, std::string> >::const_iterator it = checkedCollisionModels.begin(); it != checkedCollisionModels.end(); it++)
        msg_info("PBDCollisionModelsIntersection") << " - " << it->first << " -- " << it->second;

    msg_info("PBDCollisionModelsIntersection") << "AABB pairs overlapping in broad-phase: " << overlappingCollisionModels.size();
    for (std::vector<std::pair<std::string, std::string> >::const_iterator it = overlappingCollisionModels.begin(); it != overlappingCollisionModels.end(); it++)
        msg_info("PBDCollisionModelsIntersection") << " - " << it->first << " -- " << it->second;

    LMDNewProximityIntersection::endBroadPhase();
}

bool PBDCollisionModelsIntersection::testIntersection(Point& p1, Point& p2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection point - point: " << p1.getCollisionModel()->getName() << " (" << p1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";
    return false;
}

bool PBDCollisionModelsIntersection::testIntersection(Line& l1, Point& p2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection line - point: " << l1.getCollisionModel()->getName() << " (" << l1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";
    return false;
}

bool PBDCollisionModelsIntersection::testIntersection(Line& l1, Line& l2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection line - line: " << l1.getCollisionModel()->getName() << " (" << l1.getIndex() << ") - " << l2.getCollisionModel()->getName() << " (" << l2.getIndex() << ")";
    return false;
}

bool PBDCollisionModelsIntersection::testIntersection(Triangle& t1, Point& p2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection triangle - point: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";
    return false;
}

bool PBDCollisionModelsIntersection::testIntersection(Triangle& t1, Line& l2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection triangle - line: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << l2.getCollisionModel()->getName() << " (" << l2.getIndex() << ")";
    return false;
}

bool PBDCollisionModelsIntersection::testIntersection(Triangle& t1, Triangle& t2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection triangle - triangle: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << t2.getCollisionModel()->getName() << " (" << t2.getIndex() << ")";
    return false;
}

int PBDCollisionModelsIntersection::computeIntersection(Point& p1, Point& p2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection point - point: " << p1.getCollisionModel()->getName() << " (" << p1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Line& l1, Point& p2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection line - point: " << l1.getCollisionModel()->getName() << " (" << l1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Line& l1, Line& l2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection line - line: " << l1.getCollisionModel()->getName() << " (" << l1.getIndex() << ") - " << l2.getCollisionModel()->getName() << " (" << l2.getIndex() << ")";
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Triangle& t1, Point& p2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection triangle - point: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Triangle& t1, Line& l2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection triangle - line: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << l2.getCollisionModel()->getName() << " (" << l2.getIndex() << ")";
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Triangle& t1, Triangle& t2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection triangle - triangle: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << t2.getCollisionModel()->getName() << " (" << t2.getIndex() << ")";
    return 0;
}

void PBDCollisionModelsLocalMinDistance::draw(const core::visual::VisualParams *vparams)
{
    if (!vparams->displayFlags().getShowCollision())
        return;

    msg_info("PBDCollisionModelsLocalMinDistance") << "draw(" << this->getName() << ")";
    glPointSize(25.0f);
    glLineWidth(20.0f);

    glPushMatrix();
    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);


    glPopAttrib();
    glPopMatrix();

    glPointSize(1.0f);
    glLineWidth(1.0f);
}
