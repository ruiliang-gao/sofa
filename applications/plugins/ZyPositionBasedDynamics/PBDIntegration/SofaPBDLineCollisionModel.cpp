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

#include "SofaPBDLineCollisionModel.inl"

template class TPBDLine<sofa::defaulttype::Vec3Types>;

SOFA_DECL_CLASS(SofaPBDLineCollisionModel)

int SofaPBDLineCollisionModelClass = sofa::core::RegisterObject("PBD plugin adapter class for line collision models.")
                            .add< SofaPBDLineCollisionModel >()
                            .addDescription("PBD plugin adapter class for line collision models.");


SofaPBDLineCollisionModel::SofaPBDLineCollisionModel(): sofa::component::collision::LineModel(),
    m_initCalled(false), m_initCallCount(0)
{
    m_d = new SofaPBDLineCollisionModelPrivate();

    this->addTag(sofa::core::collision::tagPBDLineCollisionModel);
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

    if (!m_initCalled)
    {
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
        m_initCalled = true;
        m_initCallCount++;
    }
    else
    {
        msg_warning("SofaPBDLineCollisionModel") << "init/bwdInit have already been called " << m_initCallCount << " times, not initializing again.";
        m_initCallCount++;
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
            RigidBodyGeometry& pbdRBGeom = m_d->m_pbdRigidBody->getRigidBodyGeometry();
            Utilities::IndexedFaceMesh::Edges& edges = pbdRBGeom.getMesh().getEdges();

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
            PBD::LineModel::Edges& lm_edges = m_d->m_pbdLineModel->getPBDLineModel()->getEdges();

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
            RigidBodyGeometry& pbdRBGeom = m_d->m_pbdRigidBody->getRigidBodyGeometry();
            Utilities::IndexedFaceMesh::Edges& edges = pbdRBGeom.getMesh().getEdges();

            nlines = edges.size();

            if (nlines != size)
            {
                msg_info("SofaPBDLineCollisionModel") << "Resizing collision model to: " << nlines;
                resize(nlines);
            }

            for (int i = 0; i < edges.size(); i++)
            {
                const Utilities::IndexedFaceMesh::Edge& edge = edges[i];
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
            SimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
            const ParticleData &pd = model->getParticles();

            PBD::LineModel::Edges& lm_edges = m_d->m_pbdLineModel->getPBDLineModel()->getEdges();

            nlines = lm_edges.size();

            if (nlines != size)
            {
                msg_info("SofaPBDLineCollisionModel") << "Resizing collision model to: " << nlines;
                resize(nlines);
            }

            for (int i = 0; i < lm_edges.size(); i++)
            {
                const PBD::LineModel::OrientedEdge& edge = lm_edges[i];
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

bool SofaPBDLineCollisionModel::usesPBDRigidBody() const
{
    return m_d->m_usePBDRigidBody;
}

bool SofaPBDLineCollisionModel::usesPBDLineModel() const
{
    return m_d->m_usePBDLineModel;
}

const RigidBody* SofaPBDLineCollisionModel::getPBDRigidBody() const
{
    if (m_d->m_pbdRigidBody)
        return m_d->m_pbdRigidBody->getPBDRigidBody();

    return nullptr;
}

RigidBody *SofaPBDLineCollisionModel::getPBDRigidBody()
{
    if (m_d->m_pbdRigidBody)
        return m_d->m_pbdRigidBody->getPBDRigidBody();

    return nullptr;
}

const int SofaPBDLineCollisionModel::getPBDRigidBodyIndex() const
{
    if (m_d->m_pbdRigidBody)
        return m_d->m_pbdRigidBody->getPBDRigidBodyIndex();

    return -1;
}

const PBD::LineModel* SofaPBDLineCollisionModel::getPBDLineModel() const
{
    if (m_d->m_pbdLineModel)
        return m_d->m_pbdLineModel->getPBDLineModel().get();

    return nullptr;
}

PBD::LineModel* SofaPBDLineCollisionModel::getPBDLineModel()
{
    if (m_d->m_pbdLineModel)
        return m_d->m_pbdLineModel->getPBDLineModel().get();

    return nullptr;
}

const sofa::defaulttype::Vec3 SofaPBDLineCollisionModel::getCoord(unsigned int idx) const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);
    msg_info("SofaPBDLineCollisionModel") << "getCoord(" << idx << ")";

    if (m_d->m_usePBDRigidBody)
    {
        msg_info("SofaPBDLineCollisionModel") << "Using PBDRigidBody.";
        if (idx >= m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().size())
        {
            msg_info("SofaPBDLineCollisionModel") << "Index " << idx << " lies beyond RB geometry size " << m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().size() << " returning zeroVec.";
            return zeroVec;
        }
        Vector3r pt = m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().getPosition(idx);
        msg_info("SofaPBDLineCollisionModel") << "RB vertex at index: " << idx << ": " << pt;
        return sofa::defaulttype::Vector3(pt[0], pt[1], pt[2]);
    }

    if (m_d->m_usePBDLineModel)
    {
        SimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
        const ParticleData &pd = model->getParticles();

        PBD::LineModel::Edges& lm_edges = m_d->m_pbdLineModel->getPBDLineModel()->getEdges();
        if (idx > lm_edges.size())
        {
            msg_info("SofaPBDLineCollisionModel") << "Index " << idx << " lies beyond line model size " << lm_edges.size() << ", returning zeroVec.";
            return zeroVec;
        }

        if (idx == lm_edges.size())
        {
            const PBD::LineModel::OrientedEdge& edge = lm_edges.back();
            const Vector3r& lpt = pd.getPosition(edge.m_vert[1]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
        else
        {
            const PBD::LineModel::OrientedEdge& edge = lm_edges[idx];
            const Vector3r& lpt = pd.getPosition(edge.m_vert[0]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
    }

    return zeroVec;
}

const sofa::defaulttype::Vec3 SofaPBDLineCollisionModel::getDeriv(unsigned int idx) const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    if (m_d->m_usePBDRigidBody)
    {
        if (idx >= m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().size())
            return zeroVec;

        Vector3r tr_v = m_d->m_pbdRigidBody->getPBDRigidBody()->getVelocity();
        Vector3r rot_v = m_d->m_pbdRigidBody->getPBDRigidBody()->getAngularVelocity();

        Vector3r rb_pos = m_d->m_pbdRigidBody->getPBDRigidBody()->getPosition();
        Vector3r pt = m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().getPosition(idx);

        Vector3r pt_in_rb_pos = rb_pos - pt;

        Vec3 pt_in_rb(pt_in_rb_pos[0], pt_in_rb_pos[1], pt_in_rb_pos[2]);

        Vec3 tr_vel(tr_v[0], tr_v[1], tr_v[2]);
        Vec3 rot_vel(rot_v[0], rot_v[1], rot_v[2]);

        Vec3 pt_in_rb_vel = tr_vel + rot_vel.cross(pt_in_rb);

        return pt_in_rb_vel;
    }

    if (m_d->m_usePBDLineModel)
    {
        SimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
        const ParticleData &pd = model->getParticles();

        PBD::LineModel::Edges& lm_edges = m_d->m_pbdLineModel->getPBDLineModel()->getEdges();
        if (idx > lm_edges.size())
            return zeroVec;

        if (idx == lm_edges.size())
        {
            const PBD::LineModel::OrientedEdge& edge = lm_edges.back();
            const Vector3r& lpt = pd.getVelocity(edge.m_vert[1]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
        else
        {
            const PBD::LineModel::OrientedEdge& edge = lm_edges[idx];
            const Vector3r& lpt = pd.getVelocity(edge.m_vert[0]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
    }

    return zeroVec;
}

void SofaPBDLineCollisionModel::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    if (!onlyVisible)
        return;

    static const Real max_real = std::numeric_limits<Real>::max();
    static const Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};

    for (int i=0; i<size; i++)
    {
        Element e(this,i);
        const defaulttype::Vector3& pt1 = e.p1();
        const defaulttype::Vector3& pt2 = e.p2();

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
        }
    }

    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(minBBox,maxBBox));
}
