#include "SofaPBDTriangleCollisionModel.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/TriangleLocalMinDistanceFilter.h>

#include "PBDModels/SofaPBDRigidBodyModel.h"
#include "PBDModels/SofaPBDLineModel.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTriangleCollisionModelPrivate
            {
                public:
                    SofaPBDTriangleCollisionModelPrivate(): m_pbdRigidBody(nullptr), m_pbdLineModel(nullptr), m_cubeModel(nullptr),
                        m_initialized(false)
                    {

                    }

                    std::string m_pbdRigidBodyModelName;

                    SofaPBDRigidBodyModel* m_pbdRigidBody;
                    SofaPBDLineModel* m_pbdLineModel;

                    bool m_useMState;
                    bool m_initialized;

                    sofa::component::collision::CubeModel* m_cubeModel;
            };
        }
    }
}

using namespace sofa::simulation::PBDSimulation;
using namespace sofa::core;
using namespace sofa::component::collision;

SOFA_DECL_CLASS(SofaPBDTriangleCollisionModel)

int SofaPBDTriangleCollisionModelClass = sofa::core::RegisterObject("PBD plugin adapter class for triangle collision models.")
                            .add< SofaPBDTriangleCollisionModel >()
                            .addDescription("PBD plugin adapter class for triangle collision models.");

SofaPBDTriangleCollisionModel::SofaPBDTriangleCollisionModel(): sofa::component::collision::TriangleModel()
{
    m_d = new SofaPBDTriangleCollisionModelPrivate();
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

        msg_info("SofaPBDTriangleCollisionModel") << "'rigidBodyModel' attribute given for SofaPBDPointCollisionModel '" << this->getName() << ": " << valueString;

        if (valueString[0] != '@')
        {
            msg_error("SofaPBDTriangleCollisionModel") <<"'rigidBodyModel' attribute value should be a link using '@'";
        }
        else
        {
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
        msg_info("SofaPBDPointCollisionModel") << "BaseMechanicalState instance: " << ms->getName() << " of class " << ms->getClassName();
        m_d->m_useMState = true;
    }
    else
    {
        msg_info("SofaPBDPointCollisionModel") << "Could not locate valid MechanicalState in context. Will use PBD point model directly.";
    }
}

void SofaPBDTriangleCollisionModel::bwdInit()
{
    BaseContext* bc = this->getContext();
    std::vector<SofaPBDRigidBodyModel*> pbdRigidBodies = bc->getObjects<SofaPBDRigidBodyModel>(BaseContext::SearchDown);
    std::vector<SofaPBDLineModel*> pbdLineModels = bc->getObjects<SofaPBDLineModel>(BaseContext::SearchDown);

    msg_info("SofaPBDTriangleCollisionModel") << "SofaPBDRigidBodyModel instances on peer/child level: " << pbdRigidBodies.size();
    msg_info("SofaPBDTriangleCollisionModel") << "SofaPBDLineBodyModel instances on peer/child level: " << pbdLineModels.size();

    if (!m_d->m_pbdRigidBodyModelName.empty())
    {
        std::string targetRigidBodyName = m_d->m_pbdRigidBodyModelName.substr(1);

        msg_info("SofaPBDTriangleCollisionModel") << "Searching for target SofaPBDRigidBody named: " << targetRigidBodyName;
        if (pbdRigidBodies.size() > 0)
        {
            for (size_t k = 0; k < pbdRigidBodies.size(); k++)
            {
                msg_info("SofaPBDTriangleCollisionModel") << "Comparing: " << pbdRigidBodies[k]->getName() << " == " << targetRigidBodyName;
                if (pbdRigidBodies[k]->getName().compare(targetRigidBodyName) == 0)
                {
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
            msg_info("SofaPBDTriangleCollisionModel") << "rigidBodyModel link specified and found.";
        }
        else
        {
            msg_error("SofaPBDTriangleCollisionModel") << "rigidBodyModel link specified, but no SofaPBDRigidBody instance found on peer/child level!";
        }

        msg_info("SofaPBDLineCollisionModel") << "Size of collision model (triangles): " << ntriangles;

        if (ntriangles != size)
        {
            msg_info("SofaPBDTriangleCollisionModel") << "Resizing collision model to: " << ntriangles;
            resize(ntriangles);
        }
    }
}

void SofaPBDTriangleCollisionModel::draw(const core::visual::VisualParams* vparams)
{
    if (m_d->m_cubeModel)
        m_d->m_cubeModel->draw(vparams);
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

            msg_info("SofaPBDTriangleCollisionModel") << "Triangle: " << face_idx1 << "," << face_idx2 << "," << face_idx3;
            msg_info("SofaPBDTriangleCollisionModel") << "Triangle vertices: " << pt1 << " - " << pt2 << " - " << pt3;

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

        msg_info("SofaPBDTriangleCollisionModel") << "Cells in BVH: " << m_d->m_cubeModel->getNumberCells();
        sofa::helper::vector< std::pair< sofa::defaulttype::Vector3, sofa::defaulttype::Vector3> > model_bvh;

        m_d->m_cubeModel->getBoundingTree(model_bvh);
        for (size_t k = 0; k < model_bvh.size(); k++)
        {
            msg_info("SofaPBDTriangleCollisionModel") << "BVH cell " << k << ": " << model_bvh[k].first << " -- " << model_bvh[k].second;
        }
    }
    else
    {
        msg_info("SofaPBDTriangleCollisionModel") << "Model marked as empty, no BVH update possible.";
    }


    if (m_lmdFilter != 0)
    {
        m_lmdFilter->invalidate();
    }
}
