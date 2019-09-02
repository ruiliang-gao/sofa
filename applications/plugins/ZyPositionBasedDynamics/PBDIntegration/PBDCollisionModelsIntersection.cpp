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

SOFA_DECL_CLASS(PBDCollisionModelsIntersection)

int PBDCollisionModelsIntersectionClass = sofa::core::RegisterObject("Wrapper for SOFA's default collision detection, linking it to the PBD plugin.")
        .add<PBDCollisionModelsIntersection>()
        ;

//IntersectorCreator<DiscreteIntersection, PBDCollisionModelsIntersection> PBDCollisionModelsDiscreteIntersectors("PBDCollisionModels");

void GetPosOfEdgeVertexOnTriangle(Vector3& pv1, Vector3& pv2, int edge_number, TPBDTriangle<sofa::defaulttype::Vec3Types> &t)
{
    /*sofa::core::topology::BaseMeshTopology::Edge edge = t.getCollisionModel()->getTopology()->getEdge(edge_number);
    core::behavior::MechanicalState<Vec3Types>* mState= t.getCollisionModel()->getMechanicalState();
    pv1= (mState->read(core::ConstVecCoordId::position())->getValue())[edge[0]];
    pv2= (mState->read(core::ConstVecCoordId::position())->getValue())[edge[1]];*/

    const sofa::helper::fixed_array<unsigned int, 3> t_edges = t.getCollisionModel()->getEdgesInTriangle(t.getIndex());
    if (edge_number == t_edges[0])
    {
        pv1 = t.getCollisionModel()->getVertex1(t.getIndex());
        pv2 = t.getCollisionModel()->getVertex2(t.getIndex());
    }
    else if (edge_number == t_edges[1])
    {
        pv1 = t.getCollisionModel()->getVertex2(t.getIndex());
        pv2 = t.getCollisionModel()->getVertex3(t.getIndex());
    }
    else if (edge_number == t_edges[2])
    {
        pv1 = t.getCollisionModel()->getVertex3(t.getIndex());
        pv2 = t.getCollisionModel()->getVertex1(t.getIndex());
    }
}

PBDCollisionModelsIntersection::PBDCollisionModelsIntersection(/*DiscreteIntersection *object*/): LMDNewProximityIntersection()
{
    this->f_printLog.setValue(true);
}

void PBDCollisionModelsIntersection::init()
{
    if (this->f_printLog.getValue())
        msg_info("PBDCollisionModelsIntersection") << "PBDCollisionModelsLocalMinDistance::init(" << this->getName() << ")";


    intersectors.add<SofaPBDPointCollisionModel, SofaPBDPointCollisionModel, PBDCollisionModelsIntersection>(this);
    intersectors.add<SofaPBDLineCollisionModel, SofaPBDPointCollisionModel, PBDCollisionModelsIntersection>(this);
    intersectors.add<SofaPBDLineCollisionModel, SofaPBDLineCollisionModel, PBDCollisionModelsIntersection>(this);
    intersectors.add<SofaPBDTriangleCollisionModel, SofaPBDPointCollisionModel, PBDCollisionModelsIntersection>(this);
    intersectors.add<SofaPBDTriangleCollisionModel, SofaPBDLineCollisionModel, PBDCollisionModelsIntersection>(this);
    intersectors.add<SofaPBDTriangleCollisionModel, SofaPBDTriangleCollisionModel, PBDCollisionModelsIntersection>(this);

    const std::vector<std::pair<helper::TypeInfo, helper::TypeInfo> > intersectorPairs = intersectors.getIntersectors();

    if (this->f_printLog.getValue())
    {
        msg_info("PBDCollisionModelsIntersection") << "Registered intersector pairings: " << intersectorPairs.size();
        for (size_t k = 0; k < intersectorPairs.size(); k++)
            msg_info("PBDCollisionModelsIntersection") << k << ": " << intersectorPairs[k].first.pt->name() << " -- " << intersectorPairs[k].second.pt->name();
    }
}

bool PBDCollisionModelsIntersection::testIntersection(Line& /*l1*/, Line& /*l2*/)
{
    return false;
}

bool PBDCollisionModelsIntersection::testIntersection(Triangle& /*t1*/, Line& /*l1*/)
{
    return false;
}

bool PBDCollisionModelsIntersection::testIntersection(Triangle& /*t1*/, Triangle& /*t2*/)
{
    return false;
}

int PBDCollisionModelsIntersection::computeIntersection(Line& /*l1*/, Line& /*l2*/, OutputVector* /*o*/)
{
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Triangle&, TPBDPoint<sofa::defaulttype::Vector3>&, OutputVector*)
{
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Line&, TPBDPoint<sofa::defaulttype::Vec3Types>&, OutputVector*)
{
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Triangle&, TPBDPoint<sofa::defaulttype::Vec3Types>&, OutputVector*)
{
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Triangle& /*t1*/, Point& /*p2*/, OutputVector* /*o*/)
{
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Triangle& /*t1*/, Line& /*l2*/, OutputVector* /*o*/)
{
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(Triangle& /*t1*/, Triangle& /*t2*/, OutputVector* /*o*/)
{
    return 0;
}

bool PBDCollisionModelsIntersection::testIntersection(Line&, TPBDPoint<sofa::defaulttype::Vec3Types>&)
{
    return false;
}

bool PBDCollisionModelsIntersection::testIntersection(Triangle&, TPBDPoint<sofa::defaulttype::Vec3Types>&)
{
    return false;
}

bool PBDCollisionModelsIntersection::testIntersection(Triangle& /*t1*/, Point& /*p1*/)
{
    return false;
}

bool PBDCollisionModelsIntersection::testIntersection(TPBDPoint<sofa::defaulttype::Vec3Types>& p1, TPBDPoint<sofa::defaulttype::Vec3Types>& p2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection point - point: " << p1.getCollisionModel()->getName() << " (" << p1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";
    OutputVector contacts;
    const double alarmDist = getAlarmDistance() + p1.getProximity() + p2.getProximity();

    int n = m_intersector.doIntersectionPointPoint(alarmDist*alarmDist, p1.p(), p2.p(), &contacts, -1, p1.getIndex(), p2.getIndex(), *(p1.getCollisionModel()->getFilter()), *(p2.getCollisionModel()->getFilter()));
    return (n > 0);
}

bool PBDCollisionModelsIntersection::testIntersection(TPBDLine<Vec3Types>& l1, TPBDPoint<Vec3Types>& p2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection line - point: " << l1.getCollisionModel()->getName() << " (" << l1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";
    return true;
}

bool PBDCollisionModelsIntersection::testIntersection(TPBDLine<Vec3Types> &l1, TPBDLine<Vec3Types> &l2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection line - line: " << l1.getCollisionModel()->getName() << " (" << l1.getIndex() << ") - " << l2.getCollisionModel()->getName() << " (" << l2.getIndex() << ")";
    return true;
}

bool PBDCollisionModelsIntersection::testIntersection(TPBDTriangle<sofa::defaulttype::Vec3Types>& t1, TPBDPoint<Vec3Types> &p2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection triangle - point: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";
    return true;
}

bool PBDCollisionModelsIntersection::testIntersection(TPBDTriangle<Vec3Types> &t1, TPBDLine<Vec3Types> &l2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection triangle - line: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << l2.getCollisionModel()->getName() << " (" << l2.getIndex() << ")";
    return true;
}

bool PBDCollisionModelsIntersection::testIntersection(TPBDTriangle<Vec3Types> &t1, TPBDTriangle<Vec3Types> &t2)
{
    msg_info("PBDCollisionModelsIntersection") << "testIntersection triangle - triangle: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << t2.getCollisionModel()->getName() << " (" << t2.getIndex() << ")";
    return true;
}

int PBDCollisionModelsIntersection::computeIntersection(TPBDPoint<sofa::defaulttype::Vec3Types>& p1, TPBDPoint<sofa::defaulttype::Vec3Types>& p2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection point - point: " << p1.getCollisionModel()->getName() << " (" << p1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";

    // msg_info("PBDCollisionModelsIntersection") << "Retrieving p1 proximity value.";
    const double proximity_p1 = p1.getProximity();
    // msg_info("PBDCollisionModelsIntersection") << "Retrieving p2 proximity value.";
    const double proximity_p2 = p2.getProximity();

    // msg_info("PBDCollisionModelsIntersection") << "proximity values: p1 = " << proximity_p1 << ", proximity_p2 = " << proximity_p2;
    const double alarmDist = getAlarmDistance() + proximity_p1 + proximity_p2;
    // msg_info("PBDCollisionModelsIntersection") << "alarmDistance = " << alarmDist;

    Vec3 pt1 = p1.p();
    Vec3 pt2 = p2.p();

    // msg_info("PBDCollisionModelsIntersection") << "Calling doIntersectionPointPoint(" << pt1 << "," << pt2 << ")";
    int n = m_intersector.doIntersectionPointPoint(alarmDist * alarmDist, pt1, pt2, o
            , (p1.getCollisionModel()->getSize() > p2.getCollisionModel()->getSize()) ? p1.getIndex() : p2.getIndex()
            , p1.getIndex(), p2.getIndex(), *(p1.getCollisionModel()->getFilter())
            , *(p2.getCollisionModel()->getFilter()));

    // msg_info("PBDCollisionModelsIntersection") << "doIntersectionPointPoint result: " << n;

    if (n > 0)
    {
        const double contactDist = getContactDistance() + p1.getProximity() + p2.getProximity();
        // msg_info("PBDCollisionModelsIntersection") << "Contact distance: " << contactDist;

        for (OutputVector::iterator detection = o->end() - n; detection != o->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(p1, p2);
            detection->value -= contactDist;
        }
    }

    msg_info("PBDCollisionModelsIntersection") << "Intersections detected: " << n;
    return n;
}

int PBDCollisionModelsIntersection::computeIntersection(TPBDLine<Vec3Types> &l1, TPBDPoint<Vec3Types> &p2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection line - point: " << l1.getCollisionModel()->getName() << " (" << l1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";
    const double alarmDist = getAlarmDistance() + l1.getProximity() + p2.getProximity();

    int id = p2.getIndex();

    Vector3 pt1 = l1.p1();
    Vector3 pt2 = l1.p2();
    Vector3 pt3 = p2.p();

    int n = m_intersector.doIntersectionLinePoint(alarmDist * alarmDist, pt1, pt2, pt3, o, id
            , l1.getIndex(), p2.getIndex(), *(l1.getCollisionModel()->getFilter())
            , *(p2.getCollisionModel()->getFilter()));

    if (n > 0)
    {
        const double contactDist = getContactDistance() + l1.getProximity() + p2.getProximity();
        for (OutputVector::iterator detection = o->end() - n; detection != o->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(l1, p2);
            detection->value -= contactDist;
        }
    }
    msg_info("PBDCollisionModelsIntersection") << "Intersections detected: " << n;

    return n;
}

int PBDCollisionModelsIntersection::computeIntersection(TPBDLine<Vec3Types> &l1, TPBDLine<Vec3Types> &l2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection line - line: " << l1.getCollisionModel()->getName() << " (" << l1.getIndex() << ") - " << l2.getCollisionModel()->getName() << " (" << l2.getIndex() << ")";
    const double alarmDist = getAlarmDistance() + l1.getProximity() + l2.getProximity();
    const double dist2 = alarmDist * alarmDist;
    const int id = (l1.getCollisionModel()->getSize() > l2.getCollisionModel()->getSize()) ? l1.getIndex() : l2.getIndex();

    int n = m_intersector.doIntersectionLineLine(dist2, l1.p1(), l1.p2(), l2.p1(), l2.p2(), o, id
            , l1.getIndex(), l2.getIndex(), true, l1.getCollisionModel()->getFilter()
            , l2.getCollisionModel()->getFilter());


    if (n > 0)
    {
        const double contactDist = getContactDistance() + l1.getProximity() + l2.getProximity();
        for (OutputVector::iterator detection = o->end() - n; detection != o->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(l1, l2);
            detection->value -= contactDist;
        }
    }

    msg_info("PBDCollisionModelsIntersection") << "Intersections detected: " << n;
    return n;
}

int PBDCollisionModelsIntersection::computeIntersection(TPBDTriangle<Vec3Types> &t1, TPBDPoint<Vec3Types> &p2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection triangle - point: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << p2.getCollisionModel()->getName() << " (" << p2.getIndex() << ")";

    const sofa::helper::fixed_array<unsigned int,3> edgesInTriangle1 = t1.getCollisionModel()->getEdgesInTriangle(t1.getIndex());
    unsigned int E1edge1verif, E1edge2verif, E1edge3verif;
    E1edge1verif=0; E1edge2verif=0; E1edge3verif=0;

    // verify the edge ordering //
    sofa::core::topology::BaseMeshTopology::Edge edge[3];
    for (int i=0; i<3; i++)
    {
        // Verify for E1: convention: Edge1 = P1 P2    Edge2 = P2 P3    Edge3 = P3 P1
        edge[i] = t1.getCollisionModel()->getTopology()->getEdge(edgesInTriangle1[i]);
        if(((int)edge[i][0] == t1.p1Index() && (int)edge[i][1] == t1.p2Index()) || ((int)edge[i][0] == t1.p2Index() && (int)edge[i][1] == t1.p1Index()))
        {
            E1edge1verif = edgesInTriangle1[i]; /*msg_info("LMDNewProximityIntersection")<<"- e1 1: "<<i ;*/
        }
        if(((int)edge[i][0] == t1.p2Index() && (int)edge[i][1] == t1.p3Index()) || ((int)edge[i][0] == t1.p3Index() && (int)edge[i][1] == t1.p2Index()))
        {
            E1edge2verif = edgesInTriangle1[i]; /*msg_info("LMDNewProximityIntersection")<<"- e1 2: "<<i ;*/
        }
        if(((int)edge[i][0] == t1.p1Index() && (int)edge[i][1] == t1.p3Index()) || ((int)edge[i][0] == t1.p3Index() && (int)edge[i][1] == t1.p1Index()))
        {
            E1edge3verif = edgesInTriangle1[i]; /*msg_info("LMDNewProximityIntersection")<<"- e1 3: "<<i ;*/
        }
    }

    unsigned int e1_edgesIndex[3];
    e1_edgesIndex[0] = E1edge1verif; e1_edgesIndex[1] = E1edge2verif; e1_edgesIndex[2] = E1edge3verif;

    // msg_info("PBDCollisionModelsIntersection") << "computeIntersection(Triangle& e1, Point& e2... is called";
    const double alarmDist = getAlarmDistance() + t1.getProximity() + p2.getProximity();
    const double dist2 = alarmDist*alarmDist;

    int id = p2.getIndex();
    int n = m_intersector.doIntersectionTrianglePoint(dist2, t1.flags(), t1.p1(), t1.p2(), t1.p3(), t1.n(), p2.p(), o, id
            , t1, e1_edgesIndex, p2.getIndex() , *(t1.getCollisionModel()->getFilter())
            /*, *(p2.getCollisionModel()->getFilter())*/);

    if (n > 0)
    {
        const double contactDist = getContactDistance() + t1.getProximity() + p2.getProximity();
        for (OutputVector::iterator detection = o->end() - n; detection != o->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(t1, p2);
            detection->value -= contactDist;
        }
    }

    msg_info("PBDCollisionModelsIntersection") << "Intersections detected (triangle - point): " << n;
    return n;
}

int PBDCollisionModelsIntersection::computeIntersection(TPBDTriangle<Vec3Types> &t1, TPBDLine<Vec3Types> &l2, OutputVector* o)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection triangle - line: " << t1.getCollisionModel()->getName() << " (" << t1.getIndex() << ") - " << l2.getCollisionModel()->getName() << " (" << l2.getIndex() << ")";
    return 0;
}

int PBDCollisionModelsIntersection::computeIntersection(TPBDTriangle<Vec3Types> &e1, TPBDTriangle<Vec3Types> &e2, OutputVector* contacts)
{
    msg_info("PBDCollisionModelsIntersection") << "computeIntersection triangle - triangle: " << e1.getCollisionModel()->getName() << " (" << e1.getIndex() << ") - " << e2.getCollisionModel()->getName() << " (" << e2.getIndex() << ")";
    if (e1.getIndex() >= e1.getCollisionModel()->getSize())
    {
        msg_warning("PBDCollisionModelsIntersection") << "Invalid e1 index "
                << e1.getIndex() << " on CM " << e1.getCollisionModel()->getName() << " of size " << e1.getCollisionModel()->getSize();
        return 0;
    }

    if (e2.getIndex() >= e2.getCollisionModel()->getSize())
    {
        msg_warning("PBDCollisionModelsIntersection")  << "Invalid e2 index "
                << e2.getIndex() << " on CM " << e2.getCollisionModel()->getName() << " of size " << e2.getCollisionModel()->getSize()<<sendl;
        return 0;
    }

    // index of lines:
    const sofa::helper::fixed_array<unsigned int,3> edgesInTriangle1 = e1.getCollisionModel()->getEdgesInTriangle(e1.getIndex());
    const sofa::helper::fixed_array<unsigned int,3> edgesInTriangle2 = e2.getCollisionModel()->getEdgesInTriangle(e2.getIndex());

    unsigned int E1edge1verif, E1edge2verif, E1edge3verif;
    unsigned int E2edge1verif, E2edge2verif, E2edge3verif;
    E1edge1verif = 0; E1edge2verif = 0; E1edge3verif = 0;
    E2edge1verif = 0; E2edge2verif = 0; E2edge3verif = 0;

    // verify the edge ordering //
    sofa::core::topology::BaseMeshTopology::Edge edge[3];

    int vtx_idx_1_e1 = e1.getCollisionModel()->getVertex1Idx(e1.getIndex());
    int vtx_idx_2_e1 = e1.getCollisionModel()->getVertex2Idx(e1.getIndex());
    int vtx_idx_3_e1 = e1.getCollisionModel()->getVertex3Idx(e1.getIndex());

    int vtx_idx_1_e2 = e2.getCollisionModel()->getVertex1Idx(e2.getIndex());
    int vtx_idx_2_e2 = e2.getCollisionModel()->getVertex2Idx(e2.getIndex());
    int vtx_idx_3_e2 = e2.getCollisionModel()->getVertex3Idx(e2.getIndex());

    for (int i = 0; i < 3; i++)
    {
        // Verify for E1: convention: Edge1 = P1 P2    Edge2 = P2 P3    Edge3 = P3 P1
        // edge[i] = e1.getCollisionModel()->getTopology()->getEdge(edgesInTriangle1[i]);
        if (i == 0)
            edge[i] = sofa::core::topology::BaseMeshTopology::Edge(vtx_idx_1_e1, vtx_idx_2_e1);
        else if (i == 1)
            edge[i] = sofa::core::topology::BaseMeshTopology::Edge(vtx_idx_1_e1, vtx_idx_3_e1);
        else if (i == 2)
            edge[i] = sofa::core::topology::BaseMeshTopology::Edge(vtx_idx_2_e1, vtx_idx_3_e1);

        if(((int)edge[i][0]==e1.p1Index() && (int)edge[i][1]==e1.p2Index()) || ((int)edge[i][0]==e1.p2Index() && (int)edge[i][1]==e1.p1Index()))
        {
            E1edge1verif = edgesInTriangle1[i];
            msg_info("PBDCollisionModelsIntersection") << "- e1 1: " << i;
        }
        if(((int)edge[i][0]==e1.p2Index() && (int)edge[i][1]==e1.p3Index()) || ((int)edge[i][0]==e1.p3Index() && (int)edge[i][1]==e1.p2Index()))
        {
            E1edge2verif = edgesInTriangle1[i];
            msg_info("PBDCollisionModelsIntersection") << "- e1 2: " << i;
        }
        if(((int)edge[i][0]==e1.p1Index() && (int)edge[i][1]==e1.p3Index()) || ((int)edge[i][0]==e1.p3Index() && (int)edge[i][1]==e1.p1Index()))
        {
            E1edge3verif = edgesInTriangle1[i];
            msg_info("PBDCollisionModelsIntersection") << "- e1 3: " << i;
        }

        // Verify for E2: convention: Edge1 = P1 P2    Edge2 = P2 P3    Edge3 = P3 P1
        // edge[i] = e2.getCollisionModel()->getTopology()->getEdge(edgesInTriangle2[i]);
        if (i == 0)
            edge[i] = sofa::core::topology::BaseMeshTopology::Edge(vtx_idx_1_e2, vtx_idx_2_e2);
        else if (i == 1)
            edge[i] = sofa::core::topology::BaseMeshTopology::Edge(vtx_idx_1_e2, vtx_idx_3_e2);
        else if (i == 2)
            edge[i] = sofa::core::topology::BaseMeshTopology::Edge(vtx_idx_2_e2, vtx_idx_3_e2);

        if(((int)edge[i][0]==e2.p1Index() && (int)edge[i][1]==e2.p2Index()) || ((int)edge[i][0]==e2.p2Index() && (int)edge[i][1]==e2.p1Index()))
        {
            E2edge1verif = edgesInTriangle2[i];
            msg_info("PBDCollisionModelsIntersection") << "- e2 1: " << i;
        }
        if(((int)edge[i][0]==e2.p2Index() && (int)edge[i][1]==e2.p3Index()) || ((int)edge[i][0]==e2.p3Index() && (int)edge[i][1]==e2.p2Index()))
        {
            E2edge2verif = edgesInTriangle2[i];
            msg_info("PBDCollisionModelsIntersection") << "- e2 2: " << i;
        }
        if(((int)edge[i][0]==e2.p1Index() && (int)edge[i][1]==e2.p3Index()) || ((int)edge[i][0]==e2.p3Index() && (int)edge[i][1]==e2.p1Index()))
        {
            E2edge3verif = edgesInTriangle2[i];
            msg_info("PBDCollisionModelsIntersection") << "- e2 3: " << i;
        }
    }

    unsigned int e1_edgesIndex[3],e2_edgesIndex[3];
    e1_edgesIndex[0]=E1edge1verif; e1_edgesIndex[1]=E1edge2verif; e1_edgesIndex[2]=E1edge3verif;
    e2_edgesIndex[0]=E2edge1verif; e2_edgesIndex[1]=E2edge2verif; e2_edgesIndex[2]=E2edge3verif;

    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const double dist2 = alarmDist*alarmDist;
    const Vector3& p1 = e1.p1();
    const Vector3& p2 = e1.p2();
    const Vector3& p3 = e1.p3();
    const Vector3& pn = e1.n();
    const Vector3& q1 = e2.p1();
    const Vector3& q2 = e2.p2();
    const Vector3& q3 = e2.p3();
    const Vector3& qn = e2.n();

    msg_info("PBDCollisionModelsIntersection") << "p1 - p2 - p3: " << p1 << " - " << p2 << " - " << p3;
    msg_info("PBDCollisionModelsIntersection") << "q1 - q2 - q3: " << q1 << " - " << q2 << " - " << q3;

    const int f1 = e1.flags();
    const int f2 = e2.flags();

    const int id1 = e1.getIndex() * 3; // index of contacts involving points in e1
    const int id2 = e1.getCollisionModel()->getSize() * 3 + e2.getIndex() * 12; // index of contacts involving points in e2

    msg_info("PBDCollisionModelsIntersection") << "id1 - id2: " << id1 << " - " << id2;
    msg_info("PBDCollisionModelsIntersection") << "f1 - f2: " << f1 << " - " << f2;

    int n = 0;

    if (f1 & TriangleModel::FLAG_P1)
    {
        msg_info("PBDCollisionModelsIntersection") << "Testing p1 = " << p1 << " against triangle 2";
        n += m_intersector.doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p1, contacts, id1+0, e2, e2_edgesIndex, e1.p1Index(), *(e2.getCollisionModel()->getFilter())/*, *(e1.getCollisionModel()->getFilter())*/, true);
    }
    if (f1 & TriangleModel::FLAG_P2)
    {
        msg_info("PBDCollisionModelsIntersection") << "Testing p2 = " << p2 << " against triangle 2";
        n += m_intersector.doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p2, contacts, id1+1, e2, e2_edgesIndex, e1.p2Index(), *(e2.getCollisionModel()->getFilter())/*, *(e1.getCollisionModel()->getFilter())*/, true);
    }
    if (f1 & TriangleModel::FLAG_P3)
    {
        msg_info("PBDCollisionModelsIntersection") << "Testing p3 = " << p3 << " against triangle 2";
        n += m_intersector.doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p3, contacts, id1+2, e2, e2_edgesIndex, e1.p3Index(), *(e2.getCollisionModel()->getFilter())/*, *(e1.getCollisionModel()->getFilter())*/, true);
    }

    if (f2 & TriangleModel::FLAG_P1)
    {
        msg_info("PBDCollisionModelsIntersection") << "Testing q1 = " << q1 << " against triangle 1";
        n += m_intersector.doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q1, contacts, id2+0, e1, e1_edgesIndex, e2.p1Index(), *(e1.getCollisionModel()->getFilter())/*, *(e1.getCollisionModel()->getFilter())*/, false);
    }
    if (f2 & TriangleModel::FLAG_P2)
    {
        msg_info("PBDCollisionModelsIntersection") << "Testing q2 = " << q2 << " against triangle 1";
        n += m_intersector.doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q2, contacts, id2+1, e1, e1_edgesIndex, e2.p2Index(), *(e1.getCollisionModel()->getFilter())/*, *(e1.getCollisionModel()->getFilter())*/, false);
    }
    if (f2 & TriangleModel::FLAG_P3)
    {
        msg_info("PBDCollisionModelsIntersection") << "Testing q3 = " << q3 << " against triangle 1";
        n += m_intersector.doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q3, contacts, id2+2, e1, e1_edgesIndex, e2.p3Index(), *(e1.getCollisionModel()->getFilter())/*, *(e1.getCollisionModel()->getFilter())*/, false);
    }

    if (useLineLine.getValue())
    {
        msg_info("PBDCollisionModelsIntersection") << "Doing line-line checks.";
        Vector3 e1_p1, e1_p2, e1_p3, e2_q1, e2_q2, e2_q3;

        if (f1 & TriangleModel::FLAG_E12)
        {
            GetPosOfEdgeVertexOnTriangle(e1_p1,e1_p2,edgesInTriangle1[0],e1);

            if (f2 & TriangleModel::FLAG_E12)
            {
                // look for the first edge of the triangle (given by edgesInTriangle1[0])
                GetPosOfEdgeVertexOnTriangle(e2_q1,e2_q2,edgesInTriangle2[0],e2);
                msg_info("PBDCollisionModelsIntersection") << "Testing e1_p1/e1_p2 = " << e1_p1 << "/" << e1_p2 << " against e2_q1/e2_q2 = " << e2_q1 << "/" << e2_q2;
                n += m_intersector.doIntersectionLineLine(dist2, e1_p1, e1_p2, e2_q1, e2_q2, contacts, id2+3, edgesInTriangle1[0], edgesInTriangle2[0], false);
            }
            if (f2 & TriangleModel::FLAG_E23)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q2,e2_q3,edgesInTriangle2[1],e2);
                msg_info("PBDCollisionModelsIntersection") << "Testing e1_p1/e1_p2 = " << e1_p1 << "/" << e1_p2 << " against e2_q2/e2_q3 = " << e2_q2 << "/" << e2_q3;
                n += m_intersector.doIntersectionLineLine(dist2, e1_p1, e1_p2, e2_q2, e2_q3, contacts, id2+4, edgesInTriangle1[0], edgesInTriangle2[1], false);
            }
            if (f2 & TriangleModel::FLAG_E31)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q3,e2_q1,edgesInTriangle2[2],e2);
                msg_info("PBDCollisionModelsIntersection") << "Testing e1_p1/e1_p2 = " << e1_p1 << "/" << e1_p2 << " against e2_q3/e2_q1 = " << e2_q3 << "/" << e2_q1;
                n += m_intersector.doIntersectionLineLine(dist2, e1_p1, e1_p2, e2_q3, e2_q1, contacts, id2+5, edgesInTriangle1[0], edgesInTriangle2[2], false);
            }
        }

        if (f1 & TriangleModel::FLAG_E23)
        {
            GetPosOfEdgeVertexOnTriangle(e1_p2,e1_p3,edgesInTriangle1[1],e1);

            if (f2 & TriangleModel::FLAG_E12)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q1,e2_q2,edgesInTriangle2[0],e2);
                msg_info("PBDCollisionModelsIntersection") << "Testing e1_p2/e1_p3 = " << e1_p2 << "/" << e1_p3 << " against e2_q1/e2_q2 = " << e2_q1 << "/" << e2_q2;
                n += m_intersector.doIntersectionLineLine(dist2, e1_p2, e1_p3, e2_q1, e2_q2, contacts, id2+6, edgesInTriangle1[1], edgesInTriangle2[0], false);
            }
            if (f2 & TriangleModel::FLAG_E23)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q2,e2_q3,edgesInTriangle2[1],e2);
                msg_info("PBDCollisionModelsIntersection") << "Testing e1_p2/e1_p3 = " << e1_p2 << "/" << e1_p3 << " against e2_q2/e2_q3 = " << e2_q2 << "/" << e2_q3;
                n += m_intersector.doIntersectionLineLine(dist2, e1_p2, e1_p3, e2_q2, e2_q3, contacts, id2+7, edgesInTriangle1[1], edgesInTriangle2[1], false);
            }
            if (f2 & TriangleModel::FLAG_E31)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q3,e2_q1,edgesInTriangle2[2],e2);
                msg_info("PBDCollisionModelsIntersection") << "Testing e1_p2/e1_p3 = " << e1_p2 << "/" << e1_p3 << " against e2_q3/e2_q1 = " << e2_q3 << "/" << e2_q1;
                n += m_intersector.doIntersectionLineLine(dist2, e1_p2, e1_p3, e2_q3, e2_q1, contacts, id2+8, edgesInTriangle1[1], edgesInTriangle2[2], false);
            }
        }

        if (f1 & TriangleModel::FLAG_E31)
        {
            GetPosOfEdgeVertexOnTriangle(e1_p3,e1_p1,edgesInTriangle1[2],e1);
            if (f2 & TriangleModel::FLAG_E12)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q1,e2_q2,edgesInTriangle2[0],e2);
                msg_info("PBDCollisionModelsIntersection") << "Testing e1_p3/e1_p1 = " << e1_p3 << "/" << e1_p1 << " against e2_q1/e2_q2 = " << e2_q1 << "/" << e2_q2;
                n += m_intersector.doIntersectionLineLine(dist2, e1_p3, e1_p1, e2_q1, e2_q2, contacts, id2+9, edgesInTriangle1[2], edgesInTriangle2[0], false);
            }
            if (f2 & TriangleModel::FLAG_E23)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q2,e2_q3,edgesInTriangle2[1],e2);
                msg_info("PBDCollisionModelsIntersection") << "Testing e1_p3/e1_p1 = " << e1_p3 << "/" << e1_p1 << " against e2_q2/e2_q3 = " << e2_q2 << "/" << e2_q3;
                n += m_intersector.doIntersectionLineLine(dist2, e1_p3, e1_p1, e2_q2, e2_q3, contacts, id2+10, edgesInTriangle1[2], edgesInTriangle2[1], false);
            }
            if (f2 & TriangleModel::FLAG_E31)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q3,e2_q1,edgesInTriangle2[2],e2);
                msg_info("PBDCollisionModelsIntersection") << "Testing e1_p3/e1_p1 = " << e1_p3 << "/" << e1_p1 << " against e2_q3/e2_q1 = " << e2_q3 << "/" << e2_q1;
                n += m_intersector.doIntersectionLineLine(dist2, e1_p3, e1_p1, e2_q3, e2_q1, contacts, id2+11, edgesInTriangle1[2], edgesInTriangle2[2], false);
            }
        }
    }

    if (n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (int i = 0; i < n; ++i)
        {
            (*contacts)[contacts->size()-n+i].elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            (*contacts)[contacts->size()-n+i].value -= contactDist;
        }
    }

    msg_info("PBDCollisionModelsIntersection") << "Intersections detected (triangle - triangle): " << n;
    return n;
}
