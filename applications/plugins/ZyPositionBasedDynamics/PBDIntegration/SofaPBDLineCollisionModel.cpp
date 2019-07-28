#include "SofaPBDLineCollisionModel.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/LineLocalMinDistanceFilter.h>

#include "PBDModels/SofaPBDRigidBodyModel.h"
#include "PBDModels/SofaPBDLineModel.h"
#include "PBDMain/SofaPBDSimulation.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDLineCollisionModelPrivate
            {
                public:
                    SofaPBDLineCollisionModelPrivate(): m_pbdRigidBody(nullptr), m_pbdLineModel(nullptr), m_cubeModel(nullptr),
                        m_usePBDLineModel(false), m_usePBDRigidBody(false), m_initialized(false)
                    {

                    }

                    std::string m_pbdRigidBodyModelName;
                    std::string m_pbdLineModelName;

                    SofaPBDRigidBodyModel* m_pbdRigidBody;
                    SofaPBDLineModel* m_pbdLineModel;

                    bool m_useMState;
                    bool m_initialized;

                    bool m_usePBDRigidBody;
                    bool m_usePBDLineModel;

                    sofa::component::collision::CubeModel* m_cubeModel;
            };
        }
    }
}

using namespace sofa::simulation::PBDSimulation;
using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::component::collision;

SOFA_DECL_CLASS(SofaPBDLineCollisionModel)

int SofaPBDLineCollisionModelClass = sofa::core::RegisterObject("PBD plugin adapter class for line collision models.")
                            .add< SofaPBDLineCollisionModel >()
                            .addDescription("PBD plugin adapter class for line collision models.");


SofaPBDLineCollisionModel::SofaPBDLineCollisionModel(): /*sofa::core::CollisionModel(),*/ sofa::component::collision::LineModel()
{
    m_d = new SofaPBDLineCollisionModelPrivate();
}

SofaPBDLineCollisionModel::~SofaPBDLineCollisionModel()
{
    if (m_d)
    {
        delete m_d;
        m_d = NULL;
    }
}

void SofaPBDLineCollisionModel::parse(BaseObjectDescription* arg)
{
    if (arg->getAttribute("rigidBodyModel"))
    {
        std::string valueString(arg->getAttribute("rigidBodyModel"));

        msg_info("SofaPBDLineCollisionModel") << "'rigidBodyModel' attribute given for SofaPBDPointCollisionModel '" << this->getName() << ": " << valueString;

        if (valueString[0] != '@')
        {
            msg_error("SofaPBDLineCollisionModel") <<"'rigidBodyModel' attribute value should be a link using '@'";
        }
        else
        {
            msg_info("SofaPBDLineCollisionModel") << "'rigidBodyModel' attribute: " << valueString;
            m_d->m_pbdRigidBodyModelName = valueString;
        }
    }

    if (arg->getAttribute("lineModel"))
    {
        std::string valueString(arg->getAttribute("lineModel"));

        msg_info("SofaPBDLineCollisionModel") << "'lineModel' attribute given for SofaPBDPointCollisionModel '" << this->getName() << ": " << valueString;

        if (valueString[0] != '@')
        {
            msg_error("SofaPBDLineCollisionModel") <<"'lineModel' attribute value should be a link using '@'";
        }
        else
        {
            msg_info("SofaPBDLineCollisionModel") << "'lineModel' attribute: " << valueString;
            m_d->m_pbdLineModelName = valueString;
        }
    }

    BaseObject::parse(arg);
}

void SofaPBDLineCollisionModel::init()
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

// TODO: De-duplicate bwdInit in PBDPoint/LineCollisionModel
void SofaPBDLineCollisionModel::bwdInit()
{
    BaseContext* bc = this->getContext();
    std::vector<SofaPBDRigidBodyModel*> pbdRigidBodies = bc->getObjects<SofaPBDRigidBodyModel>(BaseContext::SearchDown);
    std::vector<SofaPBDLineModel*> pbdLineModels = bc->getObjects<SofaPBDLineModel>(BaseContext::SearchDown);

    msg_info("SofaPBDLineCollisionModel") << "SofaPBDRigidBodyModel instances on peer/child level: " << pbdRigidBodies.size();
    msg_info("SofaPBDLineCollisionModel") << "SofaPBDLineBodyModel instances on peer/child level: " << pbdLineModels.size();

    if (!m_d->m_pbdRigidBodyModelName.empty())
    {
        std::string targetRigidBodyName = m_d->m_pbdRigidBodyModelName.substr(1);
        msg_info("SofaPBDLineCollisionModel") << "Searching for target SofaPBDRigidBody named: " << targetRigidBodyName;

        if (pbdRigidBodies.size() > 0)
        {
            for (size_t k = 0; k < pbdRigidBodies.size(); k++)
            {
                msg_info("SofaPBDLineCollisionModel") << "Comparing: " << pbdRigidBodies[k]->getName() << " == " << targetRigidBodyName;
                if (pbdRigidBodies[k]->getName().compare(targetRigidBodyName) == 0)
                {
                    msg_info("SofaPBDLineCollisionModel") << "Found specified SofaPBDRigidBody instance: " << pbdRigidBodies[k]->getName();
                    m_d->m_pbdRigidBody = pbdRigidBodies[k];
                    break;
                }
            }
        }

        if (m_d->m_pbdRigidBody)
        {
            msg_info("SofaPBDLineCollisionModel") << "rigidBodyModel link specified and found.";
            m_d->m_usePBDRigidBody = true;
        }
        else
        {
            msg_error("SofaPBDLineCollisionModel") << "rigidBodyModel link specified, but no SofaPBDRigidBody instance found on peer/child level!";
        }
    }

    if (!m_d->m_pbdLineModelName.empty())
    {
        std::string targetLineModel = m_d->m_pbdLineModelName.substr(1);

        for (size_t k = 0; k < pbdLineModels.size(); k++)
        {
            if (pbdLineModels[k]->getName().compare(targetLineModel) == 0)
            {
                msg_info("SofaPBDLineCollisionModel") << "Found specified SofaPBDLineModel instance: " << pbdLineModels[k]->getName();
                m_d->m_pbdLineModel = pbdLineModels[k];
                break;
            }
        }

        if (m_d->m_pbdLineModel)
        {
            msg_info("SofaPBDLineCollisionModel") << "lineModel link specified and found.";
            m_d->m_usePBDLineModel = true;
        }
        else
        {
            msg_error("SofaPBDLineCollisionModel") << "lineModel link specified, but no SofaPBDLineModel instance found on peer/child level!";
        }
    }

    if (m_d->m_usePBDLineModel && m_d->m_usePBDRigidBody)
    {
        msg_warning("SofaPBDLineCollisionModel") << "Specified both lineModel and rigidBodyModel to use. Prioritizing lineModel.";
        m_d->m_usePBDRigidBody = false;
    }

    unsigned int nlines = 0;
    if (m_d->m_usePBDRigidBody && !m_d->m_usePBDLineModel)
    {
        nlines = m_d->m_pbdRigidBody->getRigidBodyGeometry().getMesh().getEdges().size();
    }
    if (m_d->m_usePBDLineModel && !m_d->m_usePBDRigidBody)
    {
        nlines = m_d->m_pbdLineModel->getPBDLineModel()->getEdges().size();
    }

    msg_info("SofaPBDLineCollisionModel") << "Size of collision model (lines): " << nlines;

    if (nlines != size)
    {
        msg_info("SofaPBDLineCollisionModel") << "Resizing collision model to: " << nlines;
        resize(nlines);
    }
}

void SofaPBDLineCollisionModel::draw(const core::visual::VisualParams* vparams)
{
    if (m_d->m_cubeModel)
        m_d->m_cubeModel->draw(vparams);
}

const sofa::core::CollisionModel *SofaPBDLineCollisionModel::toCollisionModel() const
{
    return Base::toCollisionModel();
}

sofa::core::CollisionModel *SofaPBDLineCollisionModel::toCollisionModel()
{
    return Base::toCollisionModel();
}

bool SofaPBDLineCollisionModel::insertInNode(objectmodel::BaseNode *node)
{
    return BaseObject::insertInNode(node);
}

bool SofaPBDLineCollisionModel::removeInNode(objectmodel::BaseNode *node)
{
    return BaseObject::removeInNode(node);
}

void SofaPBDLineCollisionModel::computeBoundingTree(int maxDepth)
{
    m_d->m_cubeModel = createPrevious<CubeModel>();

    if (!m_d->m_initialized)
    {
        unsigned int nlines = 0;
        if (m_d->m_usePBDRigidBody && !m_d->m_usePBDLineModel)
        {
            msg_info("SofaPBDLineCollisionModel") << "Use SofaPBDRigidBody geometry for BVH update.";
            const PBDRigidBodyGeometry& pbdRBGeom = m_d->m_pbdRigidBody->getRigidBodyGeometry();
            const Utilities::PBDIndexedFaceMesh::Edges& edges = pbdRBGeom.getMesh().getEdges();

            nlines = edges.size();

            if (nlines != size)
            {
                msg_info("SofaPBDLineCollisionModel") << "Resizing collision model to: " << nlines;
                resize(nlines);
            }
        }

        if (m_d->m_usePBDLineModel && !m_d->m_usePBDRigidBody)
        {
            msg_info("SofaPBDLineCollisionModel") << "Use SofaPBDLineModel geometry for BVH update.";
            PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
            const PBDParticleData &pd = model->getParticles();

            PBDLineModel::Edges& lm_edges = m_d->m_pbdLineModel->getPBDLineModel()->getEdges();

            nlines = lm_edges.size();

            if (nlines != size)
            {
                msg_info("SofaPBDLineCollisionModel") << "Resizing collision model to: " << nlines;
                resize(nlines);
            }
        }

        m_d->m_initialized = true;
    }

    // TODO: (Base)MeshToplogy support to take topological changes into account!
    /*updateFromTopology();

    if (needsUpdate)
        cubeModel->resize(0);*/

    if (!isMoving() && !m_d->m_cubeModel->empty() /*&& !needsUpdate*/)
        return; // No need to recompute BBox if immobile

    // needsUpdate = false;
    defaulttype::Vector3 minElem, maxElem;

    unsigned int nlines = 0;
    m_d->m_cubeModel->resize(size);
    if (!empty())
    {
        const SReal distance = (SReal)this->proximity.getValue();

        defaulttype::Vector3 pt1;
        defaulttype::Vector3 pt2;

        if (m_d->m_usePBDRigidBody && !m_d->m_usePBDLineModel)
        {
            msg_info("SofaPBDLineCollisionModel") << "Use SofaPBDRigidBody geometry for BVH update.";
            const PBDRigidBodyGeometry& pbdRBGeom = m_d->m_pbdRigidBody->getRigidBodyGeometry();
            const Utilities::PBDIndexedFaceMesh::Edges& edges = pbdRBGeom.getMesh().getEdges();

            nlines = edges.size();

            if (nlines != size)
            {
                msg_info("SofaPBDLineCollisionModel") << "Resizing collision model to: " << nlines;
                resize(nlines);
            }

            for (int i = 0; i < edges.size(); i++)
            {
                const Utilities::PBDIndexedFaceMesh::Edge& edge = edges[i];
                const Vector3r& pt_ln1 = pbdRBGeom.getVertexData().getPosition(edge.m_vert[0]);
                const Vector3r& pt_ln2 = pbdRBGeom.getVertexData().getPosition(edge.m_vert[1]);

                pt1 = defaulttype::Vector3(pt_ln1[0], pt_ln1[1], pt_ln1[2]);
                pt2 = defaulttype::Vector3(pt_ln2[0], pt_ln2[1], pt_ln2[2]);

                for (int c = 0; c < 3; c++)
                {
                    minElem[c] = pt1[c];
                    maxElem[c] = pt1[c];

                    if (pt2[c] > maxElem[c])
                        maxElem[c] = pt2[c];
                    else if (pt2[c] < minElem[c])
                        minElem[c] = pt2[c];

                    minElem[c] -= distance;
                    maxElem[c] += distance;
                }

                m_d->m_cubeModel->setParentOf(i, minElem, maxElem);
            }
        }

        if (m_d->m_usePBDLineModel && !m_d->m_usePBDRigidBody)
        {
            msg_info("SofaPBDLineCollisionModel") << "Use SofaPBDLineModel geometry for BVH update.";
            PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
            const PBDParticleData &pd = model->getParticles();

            PBDLineModel::Edges& lm_edges = m_d->m_pbdLineModel->getPBDLineModel()->getEdges();

            nlines = lm_edges.size();

            if (nlines != size)
            {
                msg_info("SofaPBDLineCollisionModel") << "Resizing collision model to: " << nlines;
                resize(nlines);
            }

            for (int i = 0; i < lm_edges.size(); i++)
            {
                const PBDLineModel::OrientedEdge& edge = lm_edges[i];
                const Vector3r& lpt_i1 = pd.getPosition(edge.m_vert[0]);
                const Vector3r& lpt_i2 = pd.getPosition(edge.m_vert[1]);

                pt1 = defaulttype::Vector3(lpt_i1[0], lpt_i1[1], lpt_i1[2]);
                pt2 = defaulttype::Vector3(lpt_i2[0], lpt_i2[1], lpt_i2[2]);

                for (int c = 0; c < 3; c++)
                {
                    minElem[c] = pt1[c];
                    maxElem[c] = pt1[c];

                    if (pt2[c] > maxElem[c])
                        maxElem[c] = pt2[c];
                    else if (pt2[c] < minElem[c])
                        minElem[c] = pt2[c];

                    minElem[c] -= distance;
                    maxElem[c] += distance;
                }

                m_d->m_cubeModel->setParentOf(i, minElem, maxElem);
            }
        }

        m_d->m_cubeModel->computeBoundingTree(maxDepth);

        msg_info("SofaPBDLineCollisionModel") << "Cells in BVH: " << m_d->m_cubeModel->getNumberCells();
        sofa::helper::vector< std::pair< sofa::defaulttype::Vector3, sofa::defaulttype::Vector3> > model_bvh;

        m_d->m_cubeModel->getBoundingTree(model_bvh);
        for (size_t k = 0; k < model_bvh.size(); k++)
        {
            msg_info("SofaPBDLineCollisionModel") << "BVH cell " << k << ": " << model_bvh[k].first << " -- " << model_bvh[k].second;
        }
    }
    else
    {
        msg_info("SofaPBDLineCollisionModel") << "Model marked as empty, no BVH update possible.";
    }

    if (m_lmdFilter != 0)
    {
        m_lmdFilter->invalidate();
    }
}
