#include "SofaPBDTriangleCollisionModel.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/TriangleLocalMinDistanceFilter.h>

#include "PBDModels/SofaPBDRigidBodyModel.h"
#include "PBDModels/SofaPBDLineModel.h"

#include "PBDIntegration/SofaPBDCollisionDetectionOutput.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTriangleCollisionModelPrivate
            {
                public:
                    SofaPBDTriangleCollisionModelPrivate(): m_pbdRigidBody(nullptr), m_cubeModel(nullptr),
                        m_initialized(false), m_numTriangles(0)
                    {

                    }

                    std::string m_pbdRigidBodyModelName;

                    SofaPBDRigidBodyModel* m_pbdRigidBody;

                    bool m_useMState;
                    bool m_initialized;

                    sofa::component::collision::CubeModel* m_cubeModel;

                    // Lookup: Triangle points in vertex array of PBD rigid body geometry
                    std::vector<sofa::defaulttype::Vec3> m_vertex_1;
                    std::vector<sofa::defaulttype::Vec3> m_vertex_2;
                    std::vector<sofa::defaulttype::Vec3> m_vertex_3;

                    // Lookup: Index of triangle points in vertex array of PBD rigid body geometry
                    std::vector<int> m_vertexToIndex_1;
                    std::vector<int> m_vertexToIndex_2;
                    std::vector<int> m_vertexToIndex_3;

                    // Lookup: Index of triangle edges to edge index in PBD rigid body mesh
                    std::vector<int> m_edgeToIndex_1;
                    std::vector<int> m_edgeToIndex_2;
                    std::vector<int> m_edgeToIndex_3;

                    unsigned int m_numTriangles;
            };
        }
    }
}

using namespace sofa::simulation::PBDSimulation;
using namespace sofa::core;
using namespace sofa::component::collision;

#include "SofaPBDTriangleCollisionModel.inl"

template class TPBDTriangle<sofa::defaulttype::Vec3Types>;

SOFA_DECL_CLASS(SofaPBDTriangleCollisionModel)

int SofaPBDTriangleCollisionModelClass = sofa::core::RegisterObject("PBD plugin adapter class for triangle collision models.")
                            .add< SofaPBDTriangleCollisionModel >()
                            .addDescription("PBD plugin adapter class for triangle collision models.");

SofaPBDTriangleCollisionModel::SofaPBDTriangleCollisionModel(): sofa::component::collision::TriangleModel(),
    showIndices(initData(&showIndices, true, "showIndices", "Show indices. (default=false)")),
    showIndicesScale(initData(&showIndicesScale, (float) 0.02, "showIndicesScale", "Scale for indices display. (default=0.02)"))

{
    m_d = new SofaPBDTriangleCollisionModelPrivate();
    this->f_printLog.setValue(true);
    this->addTag(sofa::core::collision::tagPBDTriangleCollisionModel);
}

SofaPBDTriangleCollisionModel::~SofaPBDTriangleCollisionModel()
{
    if (m_d)
    {
        delete m_d;
        m_d = nullptr;
    }
}

void SofaPBDTriangleCollisionModel::parse(BaseObjectDescription* arg)
{
    if (arg->getAttribute("rigidBodyModel"))
    {
        std::string valueString(arg->getAttribute("rigidBodyModel"));

        if (this->f_printLog.getValue())
            msg_info("SofaPBDTriangleCollisionModel") << "'rigidBodyModel' attribute given for SofaPBDPointCollisionModel '" << this->getName() << ": " << valueString;

        if (valueString[0] != '@')
        {
            msg_error("SofaPBDTriangleCollisionModel") <<"'rigidBodyModel' attribute value should be a link using '@'";
        }
        else
        {
            if (this->f_printLog.getValue())
                msg_info("SofaPBDTriangleCollisionModel") << "'rigidBodyModel' attribute: " << valueString;

            m_d->m_pbdRigidBodyModelName = valueString;
        }
    }

    BaseObject::parse(arg);
}

void SofaPBDTriangleCollisionModel::init()
{
    behavior::BaseMechanicalState* ms = getContext()->getMechanicalState();
    if (ms != nullptr)
    {
        if (this->f_printLog.getValue())
            msg_info("SofaPBDPointCollisionModel") << "BaseMechanicalState instance: " << ms->getName() << " of class " << ms->getClassName();

        m_d->m_useMState = true;
    }
    else
    {
        if (this->f_printLog.getValue())
            msg_info("SofaPBDPointCollisionModel") << "Could not locate valid MechanicalState in context. Will use PBD point model directly.";
    }
}

void SofaPBDTriangleCollisionModel::bwdInit()
{
    BaseContext* bc = this->getContext();
    std::vector<SofaPBDRigidBodyModel*> pbdRigidBodies = bc->getObjects<SofaPBDRigidBodyModel>(BaseContext::SearchDown);
    std::vector<SofaPBDLineModel*> pbdLineModels = bc->getObjects<SofaPBDLineModel>(BaseContext::SearchDown);

    if (this->f_printLog.getValue())
    {
        msg_info("SofaPBDTriangleCollisionModel") << "SofaPBDRigidBodyModel instances on peer/child level: " << pbdRigidBodies.size();
        msg_info("SofaPBDTriangleCollisionModel") << "SofaPBDLineBodyModel instances on peer/child level: " << pbdLineModels.size();
    }

    if (!m_d->m_pbdRigidBodyModelName.empty())
    {
        std::string targetRigidBodyName = m_d->m_pbdRigidBodyModelName.substr(1);

        if (this->f_printLog.getValue())
            msg_info("SofaPBDTriangleCollisionModel") << "Searching for target SofaPBDRigidBody named: " << targetRigidBodyName;

        if (pbdRigidBodies.size() > 0)
        {
            for (size_t k = 0; k < pbdRigidBodies.size(); k++)
            {
                if (this->f_printLog.getValue())
                    msg_info("SofaPBDTriangleCollisionModel") << "Comparing: " << pbdRigidBodies[k]->getName() << " == " << targetRigidBodyName;

                if (pbdRigidBodies[k]->getName().compare(targetRigidBodyName) == 0)
                {
                    if (this->f_printLog.getValue())
                        msg_info("SofaPBDTriangleCollisionModel") << "Found specified SofaPBDRigidBody instance: " << pbdRigidBodies[k]->getName();

                    m_d->m_pbdRigidBody = pbdRigidBodies[k];
                    break;
                }
            }
        }

        unsigned int ntriangles = 0;
        if (m_d->m_pbdRigidBody)
        {
            ntriangles = m_d->m_pbdRigidBody->getRigidBodyGeometry().getMesh().numFaces();

            if (this->f_printLog.getValue())
            {
                msg_info("SofaPBDTriangleCollisionModel") << "rigidBodyModel link specified and found.";
            }

            if (ntriangles == 0)
            {
                msg_warning("SofaPBDTriangleCollisionModel") << "PBD rigid body geometry reports 0 triangles! This is most likely incorrect. Have its init/bwdInit not run yet? Calling them now.";
                m_d->m_pbdRigidBody->init();
                m_d->m_pbdRigidBody->bwdInit();

                ntriangles = m_d->m_pbdRigidBody->getRigidBodyGeometry().getMesh().numFaces();
            }
        }
        else
        {
            msg_error("SofaPBDTriangleCollisionModel") << "rigidBodyModel link specified, but no SofaPBDRigidBody instance found on peer/child level!";
        }

        if (this->f_printLog.getValue())
            msg_info("SofaPBDTriangleCollisionModel") << "Size of collision model (triangles): " << ntriangles;

        if (ntriangles != size)
        {
            if (this->f_printLog.getValue())
                msg_info("SofaPBDTriangleCollisionModel") << "Resizing collision model to: " << ntriangles;

            resize(ntriangles);
        }

        // Initialize lookup for vertex to index data to avoid excessive queries to PBD rigid body geometry
        if (this->f_printLog.getValue())
        {
            msg_info("SofaPBDTriangleCollisionModel") << "===================================================";
            msg_info("SofaPBDTriangleCollisionModel") << "SofaPBDTriangleCollisionModel aux. data filled HERE";
            msg_info("SofaPBDTriangleCollisionModel") << "===================================================";
        }
        m_d->m_vertexToIndex_1.resize(ntriangles);
        m_d->m_vertexToIndex_2.resize(ntriangles);
        m_d->m_vertexToIndex_3.resize(ntriangles);

        m_d->m_edgeToIndex_1.resize(ntriangles);
        m_d->m_edgeToIndex_2.resize(ntriangles);
        m_d->m_edgeToIndex_3.resize(ntriangles);

        m_d->m_vertex_1.resize(ntriangles);
        m_d->m_vertex_2.resize(ntriangles);
        m_d->m_vertex_3.resize(ntriangles);

        m_d->m_numTriangles = ntriangles;

        const PBDVertexData& vertexData = m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData();
        const Utilities::PBDIndexedFaceMesh& rbGeometry = m_d->m_pbdRigidBody->getRigidBodyGeometry().getMesh();
        const Utilities::PBDIndexedFaceMesh::FaceData& meshFaces = rbGeometry.getFaceData();
        const Utilities::PBDIndexedFaceMesh::Edges& meshEdges = rbGeometry.getEdges();

        if (this->f_printLog.getValue())
        {
            msg_info("SofaPBDTriangleCollisionModel") << "Size of PBD triangle mesh (number of faces): " << meshFaces.size();
            msg_info("SofaPBDTriangleCollisionModel") << "Size of PBD triangle mesh (number of edges): " << meshEdges.size();
        }

        // Hard coded for triangle meshes - this should serve most cases, however, even if it is a loss of generality
        if (rbGeometry.getNumVerticesPerFace() == 3)
        {
            for (unsigned int k = 0; k < meshFaces.size(); k++)
            {
                /// TODO: Check for equal sequences: 1 == 2, 1 == 3, 2 == 3 in edge vertex index order - edges might not be in clock- or counter-clockwise order!
                const Utilities::PBDIndexedFaceMesh::Face& f = meshFaces[k];

                for (unsigned int l = 0; l < rbGeometry.getNumVerticesPerFace(); l++)
                {
                    const Utilities::PBDIndexedFaceMesh::Edge& e = meshEdges[f.m_edges[l]];

                    if (this->f_printLog.getValue())
                        msg_info("SofaPBDLineCollisionModel") << this->getName() << ": Edge " << l << " indices -- " << e.m_vert[0] << " - " << e.m_vert[1];
                }

                for (unsigned int l = 0; l < rbGeometry.getNumVerticesPerFace(); l++)
                {
                    const Utilities::PBDIndexedFaceMesh::Edge& e = meshEdges[f.m_edges[l]];
                    if (l == 0)
                    {
                        if (this->f_printLog.getValue())
                            msg_info("SofaPBDLineCollisionModel") << this->getName() << ": Face " << k << " vertex " << l << ": Index " << e.m_vert[0];

                        m_d->m_vertexToIndex_1[k] = e.m_vert[0];
                        Vector3r vt = vertexData.getPosition(e.m_vert[0]);
                        m_d->m_vertex_1[k] = sofa::defaulttype::Vec3(vt[0], vt[1], vt[2]);
                    }
                    else if (l == 1)
                    {
                        if (this->f_printLog.getValue())
                            msg_info("SofaPBDLineCollisionModel") << this->getName() << ": Face " << k << " vertex " << l << ": Index " << e.m_vert[0];

                        m_d->m_vertexToIndex_2[k] = e.m_vert[0];
                        Vector3r vt = vertexData.getPosition(e.m_vert[0]);
                        m_d->m_vertex_2[k] = sofa::defaulttype::Vec3(vt[0], vt[1], vt[2]);
                    }
                    else if (l == 2)
                    {
                        if (this->f_printLog.getValue())
                            msg_info("SofaPBDLineCollisionModel") << this->getName() << ": Face " << k << " vertex " << l << ": Index " << e.m_vert[0];

                        m_d->m_vertexToIndex_3[k] = e.m_vert[0];
                        Vector3r vt = vertexData.getPosition(e.m_vert[0]);
                        m_d->m_vertex_3[k] = sofa::defaulttype::Vec3(vt[0], vt[1], vt[2]);
                    }
                }

                bool idx_1_eq_2 = (m_d->m_vertexToIndex_1[k] == m_d->m_vertexToIndex_2[k]);
                bool idx_1_eq_3 = (m_d->m_vertexToIndex_1[k] == m_d->m_vertexToIndex_3[k]);
                bool idx_2_eq_3 = (m_d->m_vertexToIndex_2[k] == m_d->m_vertexToIndex_3[k]);

                if (idx_1_eq_2 || idx_1_eq_3 || idx_2_eq_3)
                {
                    if (idx_1_eq_2)
                    {
                        msg_warning("SofaPBDTriangleCollisionModel") << "Mesh index 1 and 2 (" << m_d->m_vertexToIndex_1[k] << " - " << m_d->m_vertexToIndex_2[k] << ") are equal! Explicit index swap required.";
                        const Utilities::PBDIndexedFaceMesh::Edge& e = meshEdges[f.m_edges[0]];
                        m_d->m_vertexToIndex_1[k] = e.m_vert[1];

                        Vector3r vt = vertexData.getPosition(e.m_vert[1]);
                        m_d->m_vertex_1[k] = sofa::defaulttype::Vec3(vt[0], vt[1], vt[2]);

                        bool idx_1_eq_2_a = (m_d->m_vertexToIndex_1[k] == m_d->m_vertexToIndex_2[k]);
                        bool idx_1_eq_3_a = (m_d->m_vertexToIndex_1[k] == m_d->m_vertexToIndex_3[k]);
                        bool idx_2_eq_3_a = (m_d->m_vertexToIndex_2[k] == m_d->m_vertexToIndex_3[k]);

                        if (idx_1_eq_2_a || idx_1_eq_3_a || idx_2_eq_3_a)
                        {
                            msg_warning("SofaPBDTriangleCollisionModel") << "Mesh indices still not unique after swap! idx_1_eq_2_a = " << idx_1_eq_2_a << ", idx_1_eq_3_a = " << idx_1_eq_3_a << ", idx_2_eq_3_a = " << idx_2_eq_3_a;
                        }
                        else
                        {
                            if (this->f_printLog.getValue())
                                msg_info("SofaPBDTriangleCollisionModel") << "Corrected face indices for face " << k << ": " << m_d->m_vertexToIndex_1[k] << "," << m_d->m_vertexToIndex_2[k] << "," << m_d->m_vertexToIndex_3[k];
                        }
                    }

                    if (idx_1_eq_3)
                    {
                        msg_warning("SofaPBDTriangleCollisionModel") << "Mesh index 1 and 3 (" << m_d->m_vertexToIndex_1[k] << " - " << m_d->m_vertexToIndex_3[k] << ") are equal! Explicit index swap required.";
                        const Utilities::PBDIndexedFaceMesh::Edge& e = meshEdges[f.m_edges[2]];
                        m_d->m_vertexToIndex_3[k] = e.m_vert[1];

                        Vector3r vt = vertexData.getPosition(e.m_vert[1]);
                        m_d->m_vertex_3[k] = sofa::defaulttype::Vec3(vt[0], vt[1], vt[2]);

                        bool idx_1_eq_2_a = (m_d->m_vertexToIndex_1[k] == m_d->m_vertexToIndex_2[k]);
                        bool idx_1_eq_3_a = (m_d->m_vertexToIndex_1[k] == m_d->m_vertexToIndex_3[k]);
                        bool idx_2_eq_3_a = (m_d->m_vertexToIndex_2[k] == m_d->m_vertexToIndex_3[k]);

                        if (idx_1_eq_2_a || idx_1_eq_3_a || idx_2_eq_3_a)
                        {
                            msg_warning("SofaPBDTriangleCollisionModel") << "Mesh indices still not unique after swap! idx_1_eq_2_a = " << idx_1_eq_2_a << ", idx_1_eq_3_a = " << idx_1_eq_3_a << ", idx_2_eq_3_a = " << idx_2_eq_3_a;
                        }
                        else
                        {
                            if (this->f_printLog.getValue())
                                msg_info("SofaPBDTriangleCollisionModel") << "Corrected face indices for face " << k << ": " << m_d->m_vertexToIndex_1[k] << "," << m_d->m_vertexToIndex_2[k] << "," << m_d->m_vertexToIndex_3[k];
                        }
                    }

                    if (idx_2_eq_3)
                    {
                        msg_warning("SofaPBDTriangleCollisionModel") << "Mesh index 2 and 3 (" << m_d->m_vertexToIndex_2[k] << " - " << m_d->m_vertexToIndex_3[k] << ") are equal! Explicit index swap required.";
                        const Utilities::PBDIndexedFaceMesh::Edge& e = meshEdges[f.m_edges[0]];
                        m_d->m_vertexToIndex_2[k] = e.m_vert[1];

                        Vector3r vt = vertexData.getPosition(e.m_vert[1]);
                        m_d->m_vertex_2[k] = sofa::defaulttype::Vec3(vt[0], vt[1], vt[2]);

                        bool idx_1_eq_2_a = (m_d->m_vertexToIndex_1[k] == m_d->m_vertexToIndex_2[k]);
                        bool idx_1_eq_3_a = (m_d->m_vertexToIndex_1[k] == m_d->m_vertexToIndex_3[k]);
                        bool idx_2_eq_3_a = (m_d->m_vertexToIndex_2[k] == m_d->m_vertexToIndex_3[k]);

                        if (idx_1_eq_2_a || idx_1_eq_3_a || idx_2_eq_3_a)
                        {
                            msg_warning("SofaPBDTriangleCollisionModel") << "Mesh indices still not unique after swap! idx_1_eq_2_a = " << idx_1_eq_2_a << ", idx_1_eq_3_a = " << idx_1_eq_3_a << ", idx_2_eq_3_a = " << idx_2_eq_3_a;
                        }
                        else
                        {
                            if (this->f_printLog.getValue())
                                msg_info("SofaPBDTriangleCollisionModel") << "Corrected face indices for face " << k << ": " << m_d->m_vertexToIndex_1[k] << "," << m_d->m_vertexToIndex_2[k] << "," << m_d->m_vertexToIndex_3[k];
                        }
                    }
                }
            }
        }

        for (unsigned int k = 0; k < meshFaces.size(); k++)
        {
            const Utilities::PBDIndexedFaceMesh::Face& f = meshFaces[k];
            for (unsigned int l = 0; l < rbGeometry.getNumVerticesPerFace(); l++)
            {
                if (l == 0)
                {
                    if (this->f_printLog.getValue())
                        msg_info("SofaPBDLineCollisionModel") << this->getName() << ": Face " << k << " edge " << l << ": Index " << f.m_edges[l];

                    m_d->m_edgeToIndex_1[k] = f.m_edges[l];
                }
                else if (l == 1)
                {
                    if (this->f_printLog.getValue())
                        msg_info("SofaPBDLineCollisionModel") << this->getName() << ": Face " << k << " edge " << l << ": Index " << f.m_edges[l];

                    m_d->m_edgeToIndex_2[k] = f.m_edges[l];
                }
                else if (l == 2)
                {
                    if (this->f_printLog.getValue())
                        msg_info("SofaPBDLineCollisionModel") << this->getName() << ": Face " << k << " edge " << l << ": Index " << f.m_edges[l];

                    m_d->m_edgeToIndex_3[k] = f.m_edges[l];
                }
            }
        }

        if (this->f_printLog.getValue())
        {
            msg_info("SofaPBDLineCollisionModel") << this->getName() << ": Face edge indices vector sizes:" << m_d->m_edgeToIndex_1.size() << "," << m_d->m_edgeToIndex_2.size() << "," << m_d->m_edgeToIndex_3.size();
            for (unsigned int k = 0; k < m_d->m_numTriangles; k++)
                msg_info("SofaPBDLineCollisionModel") << this->getName() << ": Face " << k << " edge indices:" << m_d->m_edgeToIndex_1[k] << "," << m_d->m_edgeToIndex_2[k] << "," << m_d->m_edgeToIndex_3[k];
        }
    }
}

void SofaPBDTriangleCollisionModel::draw(const core::visual::VisualParams* vparams)
{
    if (m_d->m_cubeModel)
        m_d->m_cubeModel->draw(vparams);

    if (showIndices.getValue())
    {
        defaulttype::Vec4f color(1.0, 1.0, 1.0, 1.0);

        const Utilities::PBDIndexedFaceMesh::FaceData& meshFaceData = this->getPBDRigidBody()->getGeometry().getMesh().getFaceData();

        vparams->drawTool()->saveLastState();
        std::ostringstream oss;
        for (unsigned int j = 0; j < meshFaceData.size(); j++)
        {
            const Vec3 pt1 = this->getCoord(m_d->m_vertexToIndex_1[j]);
            const Vec3 pt2 = this->getCoord(m_d->m_vertexToIndex_2[j]);
            const Vec3 pt3 = this->getCoord(m_d->m_vertexToIndex_3[j]);

            oss.str("");
            oss << "Face " << j << " vtx 1: " << pt1;
            vparams->drawTool()->draw3DText((1.0 + ((j + 1) * 0.05)) * pt1, showIndicesScale.getValue(), color, oss.str().c_str());

            oss.str("");
            oss << "Face " << j << " vtx 2: " << pt2;
            vparams->drawTool()->draw3DText((1.0 + ((j + 1) * 0.05)) * pt2, showIndicesScale.getValue(), color, oss.str().c_str());

            oss.str("");
            oss << "Face " << j << " vtx 3: " << pt3;
            vparams->drawTool()->draw3DText((1.0 + ((j + 1) * 0.05)) * pt3, showIndicesScale.getValue(), color, oss.str().c_str());

            const int e_idx1 = m_d->m_edgeToIndex_1[j];
            const int e_idx2 = m_d->m_edgeToIndex_2[j];
            const int e_idx3 = m_d->m_edgeToIndex_3[j];

            const Vec3 e1_center((pt1.x() + pt2.x()) / 2.0,
                                 (pt1.y() + pt2.y()) / 2.0,
                                 (pt1.z() + pt2.z()) / 2.0);

            const Vec3 e2_center((pt1.x() + pt3.x()) / 2.0,
                                 (pt1.y() + pt3.y()) / 2.0,
                                 (pt1.z() + pt3.z()) / 2.0);

            const Vec3 e3_center((pt2.x() + pt3.x()) / 2.0,
                                 (pt2.y() + pt3.y()) / 2.0,
                                 (pt2.z() + pt3.z()) / 2.0);

            oss.str("");
            oss << "Face " << j << " edge 1: " << e_idx1;
            vparams->drawTool()->draw3DText((1.0 + ((j + 1) * 0.05)) * e1_center, showIndicesScale.getValue(), color, oss.str().c_str());

            oss.str("");
            oss << "Face " << j << " edge 2: " << e_idx2;
            vparams->drawTool()->draw3DText((1.0 + ((j + 1) * 0.05)) * e2_center, showIndicesScale.getValue(), color, oss.str().c_str());

            oss.str("");
            oss << "Face " << j << " edge 3: " << e_idx3;
            vparams->drawTool()->draw3DText((1.0 + ((j + 1) * 0.05)) * e3_center, showIndicesScale.getValue(), color, oss.str().c_str());

            const Vec3 t_center((pt1.x() + pt2.x() + pt3.x()) / 3.0,
                                (pt1.y() + pt2.y() + pt3.y()) / 3.0,
                                (pt1.z() + pt2.z() + pt3.z()) / 3.0);

            // msg_info("SofaPBDTriangleCollisionModel") << "Drawing face: " << j << " at: " << t_center;

            oss.str("");
            oss << "Face " << j;
            vparams->drawTool()->draw3DText(t_center, showIndicesScale.getValue(), color, oss.str().c_str());
        }

        vparams->drawTool()->restoreLastState();
    }
}

const sofa::core::CollisionModel *SofaPBDTriangleCollisionModel::toCollisionModel() const
{
    return Base::toCollisionModel();
}

sofa::core::CollisionModel *SofaPBDTriangleCollisionModel::toCollisionModel()
{
    return Base::toCollisionModel();
}

bool SofaPBDTriangleCollisionModel::insertInNode(objectmodel::BaseNode *node)
{
    return BaseObject::insertInNode(node);
}

bool SofaPBDTriangleCollisionModel::removeInNode(objectmodel::BaseNode *node)
{
    return BaseObject::removeInNode(node);
}

void SofaPBDTriangleCollisionModel::computeBoundingTree(int maxDepth)
{
    m_d->m_cubeModel = createPrevious<CubeModel>();

    if (!m_d->m_initialized)
    {
        unsigned int ntriangles = m_d->m_pbdRigidBody->getRigidBodyGeometry().getMesh().getFaces().size();
        msg_info("SofaPBDTriangleCollisionModel") << "Use SofaPBDRigidBody geometry for BVH update.";

        if (ntriangles != size)
        {
            if (this->f_printLog.getValue())
                msg_info("SofaPBDTriangleCollisionModel") << "Resizing collision model to: " << ntriangles;

            resize(ntriangles);
        }

        m_d->m_initialized = true;
    }

    // TODO: (Base)MeshTopology integration for SofaPBDTriangleCollisionModel
    // check first that topology didn't change
    /*if (m_topology->getRevision() != m_topologyRevision)
        updateFromTopology();*/

    if (!isMoving() && !m_d->m_cubeModel->empty() /*&& !m_needsUpdate*/)
        return; // No need to recompute BBox if immobile nor if mesh didn't change.


    if (/*m_needsUpdate &&*/ !m_d->m_cubeModel->empty())
        m_d->m_cubeModel->resize(0);


    // set to false to avoid excessive loop
    // m_needsUpdate=false;

    // TODO: Recalculating normals redundant or not?
    // const bool calcNormals = d_computeNormals.getValue();

    m_d->m_cubeModel->resize(size);  // size = number of triangles
    if (!empty())
    {
        const PBDRigidBodyGeometry& rbGeometry = m_d->m_pbdRigidBody->getRigidBodyGeometry();
        const Utilities::PBDIndexedFaceMesh::Faces& faces = rbGeometry.getMesh().getFaces();
        const PBDVertexData& vertices = rbGeometry.getVertexData();

        const unsigned int numVerticesPerFace = rbGeometry.getMesh().getNumVerticesPerFace();
        const unsigned int numFaces = rbGeometry.getMesh().numFaces();
        const SReal distance = (SReal)this->proximity.getValue();

        if (numFaces != size)
        {
            if (this->f_printLog.getValue())
                msg_info("SofaPBDTriangleCollisionModel") << "Resizing collision model to: " << numFaces;

            resize(numFaces);
        }

        defaulttype::Vector3 minElem, maxElem;
        defaulttype::Vector3 pt1, pt2, pt3;

        for (int i = 0; i < numFaces * numVerticesPerFace; i += numVerticesPerFace)
        {
            unsigned int face_idx1 = faces[i];
            unsigned int face_idx2 = faces[i + 1];
            unsigned int face_idx3 = faces[i + 2];

            const Vector3r& pt1_tr = vertices.getPosition(face_idx1);
            const Vector3r& pt2_tr = vertices.getPosition(face_idx2);
            const Vector3r& pt3_tr = vertices.getPosition(face_idx3);

            pt1 = defaulttype::Vector3(pt1_tr[0], pt1_tr[1], pt1_tr[2]);
            pt2 = defaulttype::Vector3(pt2_tr[0], pt2_tr[1], pt2_tr[2]);
            pt3 = defaulttype::Vector3(pt3_tr[0], pt3_tr[1], pt3_tr[2]);

            if (this->f_printLog.getValue())
            {
                msg_info("SofaPBDTriangleCollisionModel") << "Triangle: " << face_idx1 << "," << face_idx2 << "," << face_idx3;
                msg_info("SofaPBDTriangleCollisionModel") << "Triangle vertices: " << pt1 << " - " << pt2 << " - " << pt3;
            }

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c])
                    maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c])
                    minElem[c] = pt2[c];
                if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                else if (pt3[c] < minElem[c])
                    minElem[c] = pt3[c];

                minElem[c] -= distance;
                maxElem[c] += distance;
            }

            // TODO: Recalculating normals redundant or not?
            /*if (calcNormals)
            {
                // Also recompute normal vector
                t.n() = cross(pt2-pt1,pt3-pt1);
                t.n().normalize();
            }*/
            m_d->m_cubeModel->setParentOf(i / numVerticesPerFace, minElem, maxElem); // define the bounding box of the current triangle
        }
        m_d->m_cubeModel->computeBoundingTree(maxDepth);

        if (this->f_printLog.getValue())
        {
            msg_info("SofaPBDTriangleCollisionModel") << "Cells in BVH: " << m_d->m_cubeModel->getNumberCells();
            sofa::helper::vector< std::pair< sofa::defaulttype::Vector3, sofa::defaulttype::Vector3> > model_bvh;

            m_d->m_cubeModel->getBoundingTree(model_bvh);
            for (size_t k = 0; k < model_bvh.size(); k++)
            {
                msg_info("SofaPBDTriangleCollisionModel") << "BVH cell " << k << ": " << model_bvh[k].first << " -- " << model_bvh[k].second;
            }
        }
    }
    else
    {
        msg_warning("SofaPBDTriangleCollisionModel") << "Model marked as empty, no BVH update possible.";
    }


    if (m_lmdFilter != 0)
    {
        m_lmdFilter->invalidate();
    }
}

const PBDRigidBody* SofaPBDTriangleCollisionModel::getPBDRigidBody() const
{
    if (m_d->m_pbdRigidBody)
        return m_d->m_pbdRigidBody->getPBDRigidBody();

    return nullptr;
}

PBDRigidBody* SofaPBDTriangleCollisionModel::getPBDRigidBody()
{
    if (m_d->m_pbdRigidBody)
        return m_d->m_pbdRigidBody->getPBDRigidBody();

    return nullptr;
}

const int SofaPBDTriangleCollisionModel::getPBDRigidBodyIndex() const
{
    if (m_d->m_pbdRigidBody)
        return m_d->m_pbdRigidBody->getPBDRigidBodyIndex();

    return -1;
}

const sofa::defaulttype::Vec3& SofaPBDTriangleCollisionModel::getVertex1(const unsigned int idx) const
{
    static sofa::defaulttype::Vector3 zeroVec(0,0,0);

    if (idx < m_d->m_vertex_1.size())
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex1(" << idx << ") = " << m_d->m_vertex_1[idx];
        return m_d->m_vertex_1[idx];
    }
    else
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex1(" << idx << ") - index " << idx << " larger than triangle model size " << m_d->m_vertex_1.size() << ", or smaller than 0!";
    }

    return zeroVec;
}

const sofa::defaulttype::Vec3& SofaPBDTriangleCollisionModel::getVertex2(const unsigned int idx) const
{
    static sofa::defaulttype::Vector3 zeroVec(0,0,0);

    if (idx < m_d->m_vertex_2.size())
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex2(" << idx << ") = " << m_d->m_vertex_2[idx];
        return m_d->m_vertex_2[idx];
    }
    else
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex2(" << idx << ") - index " << idx << " larger than triangle model size " << m_d->m_vertex_2.size() << ", or smaller than 0!";
    }

    return zeroVec;
}

const sofa::defaulttype::Vec3& SofaPBDTriangleCollisionModel::getVertex3(const unsigned int idx) const
{
    static sofa::defaulttype::Vector3 zeroVec(0,0,0);

    if (idx < m_d->m_vertex_3.size())
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex3(" << idx << ") = " << m_d->m_vertex_3[idx];
        return m_d->m_vertex_3[idx];
    }
    else
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex3(" << idx << ") - index " << idx << " larger than triangle model size " << m_d->m_vertex_3.size() << ", or smaller than 0!";
    }

    return zeroVec;
}

const int SofaPBDTriangleCollisionModel::getVertex1Idx(const unsigned int faceIdx) const
{
    if (faceIdx < m_d->m_numTriangles && faceIdx < m_d->m_vertexToIndex_1.size())
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex1Idx(" << faceIdx << ") = " << m_d->m_vertexToIndex_1[faceIdx];
        return m_d->m_vertexToIndex_1[faceIdx];
    }
    else
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex1Idx(" << faceIdx << ") - index larger than mesh size " << m_d->m_vertexToIndex_1.size() << ", or smaller than 0!";
        return -1;
    }
}

const int SofaPBDTriangleCollisionModel::getVertex2Idx(const unsigned int faceIdx) const
{
    if (faceIdx < m_d->m_numTriangles && faceIdx < m_d->m_vertexToIndex_2.size())
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex2Idx(" << faceIdx << ") = " << m_d->m_vertexToIndex_2[faceIdx];
        return m_d->m_vertexToIndex_2[faceIdx];
    }
    else
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex2Idx(" << faceIdx << ") - index larger than mesh size " << m_d->m_vertexToIndex_2.size() << ", or smaller than 0!";
        return -1;
    }
}

const int SofaPBDTriangleCollisionModel::getVertex3Idx(const unsigned int faceIdx) const
{
    if (faceIdx < m_d->m_numTriangles && faceIdx < m_d->m_vertexToIndex_3.size())
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex3Idx(" << faceIdx << ") = " << m_d->m_vertexToIndex_3[faceIdx];
        return m_d->m_vertexToIndex_3[faceIdx];
    }
    else
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getVertex3Idx(" << faceIdx << ") - index larger than mesh size " << m_d->m_vertexToIndex_3.size() << ", or smaller than 0!";
        return -1;
    }
}

const int SofaPBDTriangleCollisionModel::getEdge1Idx(const unsigned int faceIdx) const
{
    if (faceIdx < m_d->m_numTriangles && faceIdx < m_d->m_edgeToIndex_1.size())
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getEdge1Idx(" << faceIdx << ") = " << m_d->m_edgeToIndex_1[faceIdx];
        return m_d->m_edgeToIndex_1[faceIdx];
    }
    else
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getEdge1Idx(" << faceIdx << ") - index larger than mesh size " << m_d->m_edgeToIndex_1.size() << ", or smaller than 0!";
        return -1;
    }
}

const int SofaPBDTriangleCollisionModel::getEdge2Idx(const unsigned int faceIdx) const
{
    if (faceIdx < m_d->m_numTriangles && faceIdx < m_d->m_edgeToIndex_2.size())
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getEdge2Idx(" << faceIdx << ") = " << m_d->m_edgeToIndex_2[faceIdx];
        return m_d->m_edgeToIndex_2[faceIdx];
    }
    else
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getEdge2Idx(" << faceIdx << ") - index larger than mesh size " << m_d->m_edgeToIndex_2.size() << ", or smaller than 0!";
        return -1;
    }
}

const int SofaPBDTriangleCollisionModel::getEdge3Idx(const unsigned int faceIdx) const
{
    if (faceIdx < m_d->m_numTriangles && faceIdx < m_d->m_edgeToIndex_3.size())
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getEdge3Idx(" << faceIdx << ") = " << m_d->m_edgeToIndex_3[faceIdx];
        return m_d->m_edgeToIndex_3[faceIdx];
    }
    else
    {
        msg_info("SofaPBDTriangleCollisionModel") << "getEdge3Idx(" << faceIdx << ") - index larger than mesh size " << m_d->m_edgeToIndex_3.size() << ", or smaller than 0!";
        return -1;
    }
}

const sofa::helper::fixed_array<unsigned int, 3> SofaPBDTriangleCollisionModel::getEdgesInTriangle(unsigned int idx) const
{
    sofa::helper::fixed_array<unsigned int, 3> edgesInTriangle;
    edgesInTriangle[0] = 0, edgesInTriangle[1] = 0, edgesInTriangle[2] = 0;
    if (idx < m_d->m_numTriangles)
    {
        edgesInTriangle[0] = m_d->m_edgeToIndex_1[idx];
        edgesInTriangle[1] = m_d->m_edgeToIndex_2[idx];
        edgesInTriangle[2] = m_d->m_edgeToIndex_3[idx];

        if (this->f_printLog.getValue())
            msg_info("SofaPBDTriangleCollisionModel") << this->getName() << " -- Edges indices in triangle " << idx << ": " << edgesInTriangle[0] << ", " << edgesInTriangle[1] << ", " << edgesInTriangle[2];
    }
    else
    {
        msg_warning("SofaPBDTriangleCollisionModel") << "Triangle index " << idx << " outside range for mesh size: " << m_d->m_numTriangles;
    }

    return edgesInTriangle;
}

const sofa::defaulttype::Vec3 SofaPBDTriangleCollisionModel::getCoord(unsigned int idx) const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    // msg_info("SofaPBDTriangleCollisionModel") << "getCoord(" << idx << ")";

    if (m_d->m_pbdRigidBody)
    {
          if (idx >= m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().size())
        {
            msg_info("SofaPBDTriangleCollisionModel") << "Index " << idx << " lies beyond RB geometry size " << m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().size() << " returning zeroVec.";
            return zeroVec;
        }
        Vector3r pt = m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().getPosition(idx);
        return sofa::defaulttype::Vector3(pt[0], pt[1], pt[2]);
    }

    return zeroVec;
}

void SofaPBDTriangleCollisionModel::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    if( !onlyVisible )
        return;

    static const Real max_real = std::numeric_limits<Real>::max();
    static const Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};

    for (int i=0; i<size; i++)
    {
        Element t(this,i);
        const defaulttype::Vector3& pt1 = t.p1();
        const defaulttype::Vector3& pt2 = t.p2();
        const defaulttype::Vector3& pt3 = t.p3();

        for (int c=0; c<3; c++)
        {
            if (pt1[c] > maxBBox[c])
                maxBBox[c] = (Real)pt1[c];
            else if (pt1[c] < minBBox[c])
                minBBox[c] = (Real)pt1[c];

            if (pt2[c] > maxBBox[c])
                maxBBox[c] = (Real)pt2[c];
            else if (pt2[c] < minBBox[c])
                minBBox[c] = (Real)pt2[c];

            if (pt3[c] > maxBBox[c])
                maxBBox[c] = (Real)pt3[c];
            else if (pt3[c] < minBBox[c])
                minBBox[c] = (Real)pt3[c];
        }
    }

    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(minBBox,maxBBox));
}

int SofaPBDTriangleCollisionModel::getTriangleFlags(Topology::TriangleID i)
{
    int f = 0;

    if (this->f_printLog.getValue())
        msg_info("SofaPBDTriangleCollisionModel") << "getTriangleFlags(" << i << ")";

    if (i < m_d->m_numTriangles)
    {
#if 1
        for (unsigned int j = 0; j < 3; ++j)
        {
            f |= (FLAG_P1 << j);
            f |= (FLAG_E23 << j);
        }
#else
        // The PBD has the required topology information to determine the flags from the mesh structure
        // But the PBD integration requires more thorough investigation testing to find out how to translate between PBD and SOFA mesh topology data.
        // This is the experimental variant: Vertex flags work, edge flags don't

        const Utilities::PBDIndexedFaceMesh::VerticesFaces& vf_info = m_d->m_pbdRigidBody->getRigidBodyGeometry().getMesh().getVertexFaces();
        const Utilities::PBDIndexedFaceMesh::FaceData& face_data = m_d->m_pbdRigidBody->getRigidBodyGeometry().getMesh().getFaceData();
        const Utilities::PBDIndexedFaceMesh::Edges& edges_data = m_d->m_pbdRigidBody->getRigidBodyGeometry().getMesh().getEdges();

        for (unsigned int j = 0; j < 3; ++j)
        {
            /*const sofa::core::topology::BaseMeshTopology::TrianglesAroundVertex& tav = m_topology->getTrianglesAroundVertex(t[j]);
            if (tav[0] == (sofa::core::topology::BaseMeshTopology::TriangleID)i)*/

            unsigned int faces_for_vertex = vf_info[t[j]].m_numFaces;
            for (unsigned int k = 0; k < faces_for_vertex; k++)
            {
                if (vf_info[t[j]].m_fIndices[k] == i)
                {
                    f |= (FLAG_P1 << j);
                    if (j == 0)
                        msg_info("SofaPBDTriangleCollisionModel") << "Setting FLAG_P1";
                    else if (j == 1)
                        msg_info("SofaPBDTriangleCollisionModel") << "Setting FLAG_P2";
                    else if (j == 2)
                        msg_info("SofaPBDTriangleCollisionModel") << "Setting FLAG_P3";
                }
            }
        }

        //const sofa::core::topology::BaseMeshTopology::EdgesInTriangle& e = m_topology->getEdgesInTriangle(i);
        const sofa::helper::fixed_array<unsigned int, 3> e = this->getEdgesInTriangle(i);

        for (unsigned int j = 0; j < 3; ++j)
        {
            // const sofa::core::topology::BaseMeshTopology::TrianglesAroundEdge& tae = m_topology->getTrianglesAroundEdge(e[j]);
            const Utilities::PBDIndexedFaceMesh::Edge& edge = edges_data[e[j]];

            if (edge.m_face[0] == i)
            {
                if (j == 0)
                    msg_info("SofaPBDTriangleCollisionModel") << "Setting FLAG_E23";
                else if (j == 1)
                    msg_info("SofaPBDTriangleCollisionModel") << "Setting FLAG_E31";
                else if (j == 2)
                    msg_info("SofaPBDTriangleCollisionModel") << "Setting FLAG_E12";
                f |= (FLAG_E23 << j);
            }
            if (edge.m_face[1] == 0xffffffff)
                f |= (FLAG_BE23 << j);
        }
#endif
    }
    return f;
}
