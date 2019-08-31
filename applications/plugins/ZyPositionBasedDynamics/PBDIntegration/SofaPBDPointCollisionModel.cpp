#include "SofaPBDPointCollisionModel.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/PointLocalMinDistanceFilter.h>

#include "PBDModels/SofaPBDRigidBodyModel.h"
#include "PBDModels/SofaPBDLineModel.h"
#include "PBDMain/SofaPBDSimulation.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDPointCollisionModelPrivate
            {
                public:
                    SofaPBDPointCollisionModelPrivate(): m_pbdRigidBody(nullptr), m_pbdLineModel(nullptr), m_cubeModel(nullptr),
                        m_useMState(false), m_usePBDLineModel(false), m_usePBDRigidBody(false)
                    {

                    }

                    std::string m_pbdRigidBodyModelName;
                    std::string m_pbdLineModelName;

                    SofaPBDRigidBodyModel* m_pbdRigidBody;
                    SofaPBDLineModel* m_pbdLineModel;

                    bool m_useMState;

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

#include "SofaPBDPointCollisionModel.inl"

template class TPBDPoint<sofa::defaulttype::Vec3Types>;

SOFA_DECL_CLASS(SofaPBDPointCollisionModel)

int SofaPBDPointCollisionModelClass = sofa::core::RegisterObject("PBD plugin adapter class for point collision models.")
                            .add< SofaPBDPointCollisionModel >()
                            .addDescription("PBD plugin adapter class for point collision models.");

SofaPBDPointCollisionModel::SofaPBDPointCollisionModel(): /*sofa::core::CollisionModel(),*/ sofa::component::collision::PointModel()
{
    m_d = new SofaPBDPointCollisionModelPrivate();
}

SofaPBDPointCollisionModel::~SofaPBDPointCollisionModel()
{
    msg_info("SofaPBDPointCollisionModel") << "Destructor.";
    if (m_d)
    {
        msg_info("SofaPBDPointCollisionModel") << "Deleting pimpl pointer.";
        delete m_d;
        m_d = NULL;
    }
}

void SofaPBDPointCollisionModel::parse(BaseObjectDescription* arg)
{
    if (arg->getAttribute("rigidBodyModel"))
    {
        std::string valueString(arg->getAttribute("rigidBodyModel"));

        msg_info("SofaPBDPointCollisionModel") << "'rigidBodyModel' attribute given for SofaPBDPointCollisionModel '" << this->getName() << ": " << valueString;

        if (valueString[0] != '@')
        {
            msg_error("SofaPBDPointCollisionModel") <<"'rigidBodyModel' attribute value should be a link using '@'";
        }
        else
        {
            msg_info("SofaPBDPointCollisionModel") << "'rigidBodyModel' attribute: " << valueString;
            m_d->m_pbdRigidBodyModelName = valueString;
        }
    }

    if (arg->getAttribute("lineModel"))
    {
        std::string valueString(arg->getAttribute("lineModel"));

        msg_info("SofaPBDPointCollisionModel") << "'lineModel' attribute given for SofaPBDPointCollisionModel '" << this->getName() << ": " << valueString;

        if (valueString[0] != '@')
        {
            msg_error("SofaPBDPointCollisionModel") <<"'lineModel' attribute value should be a link using '@'";
        }
        else
        {
            msg_info("SofaPBDPointCollisionModel") << "'lineModel' attribute: " << valueString;
            m_d->m_pbdLineModelName = valueString;
        }
    }

    BaseObject::parse(arg);
}

void SofaPBDPointCollisionModel::init()
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
void SofaPBDPointCollisionModel::bwdInit()
{
    BaseContext* bc = this->getContext();
    std::vector<SofaPBDRigidBodyModel*> pbdRigidBodies = bc->getObjects<SofaPBDRigidBodyModel>(BaseContext::SearchDown);
    std::vector<SofaPBDLineModel*> pbdLineModels = bc->getObjects<SofaPBDLineModel>(BaseContext::SearchDown);

    msg_info("SofaPBDPointCollisionModel") << "SofaPBDRigidBodyModel instances on peer/child level: " << pbdRigidBodies.size();
    msg_info("SofaPBDPointCollisionModel") << "SofaPBDLineBodyModel instances on peer/child level: " << pbdLineModels.size();

    if (!m_d->m_pbdRigidBodyModelName.empty())
    {
        std::string targetRigidBodyName = m_d->m_pbdRigidBodyModelName.substr(1);

        msg_info("SofaPBDPointCollisionModel") << "Searching for target SofaPBDRigidBody named: " << targetRigidBodyName;
        if (pbdRigidBodies.size() > 0)
        {
            for (size_t k = 0; k < pbdRigidBodies.size(); k++)
            {
                msg_info("SofaPBDPointCollisionModel") << "Comparing: " << pbdRigidBodies[k]->getName() << " == " << targetRigidBodyName;
                if (pbdRigidBodies[k]->getName().compare(targetRigidBodyName) == 0)
                {
                    msg_info("SofaPBDPointCollisionModel") << "Found specified SofaPBDRigidBody instance: " << pbdRigidBodies[k]->getName();
                    m_d->m_pbdRigidBody = pbdRigidBodies[k];
                    break;
                }
            }
        }

        if (m_d->m_pbdRigidBody)
        {
            msg_info("SofaPBDPointCollisionModel") << "rigidBodyModel link specified and found.";
            m_d->m_usePBDRigidBody = true;
        }
        else
        {
            msg_error("SofaPBDPointCollisionModel") << "rigidBodyModel link specified, but no SofaPBDRigidBody instance found on peer/child level!";
        }
    }

    if (!m_d->m_pbdLineModelName.empty())
    {
        std::string targetLineModel = m_d->m_pbdLineModelName.substr(1);
        for (size_t k = 0; k < pbdLineModels.size(); k++)
        {
            if (pbdLineModels[k]->getName().compare(targetLineModel) == 0)
            {
                msg_info("SofaPBDPointCollisionModel") << "Found specified SofaPBDLineModel instance: " << pbdLineModels[k]->getName();
                m_d->m_pbdLineModel = pbdLineModels[k];
                break;
            }
        }

        if (m_d->m_pbdLineModel)
        {
            msg_info("SofaPBDPointCollisionModel") << "lineModel link specified and found.";
            m_d->m_usePBDLineModel = true;
        }
        else
        {
            msg_error("SofaPBDPointCollisionModel") << "lineModel link specified, but no SofaPBDLineModel instance found on peer/child level!";
        }
    }

    if (m_d->m_usePBDLineModel && m_d->m_usePBDRigidBody)
    {
        msg_warning("SofaPBDPointCollisionModel") << "Specified both lineModel and rigidBodyModel to use. Prioritizing lineModel.";
        m_d->m_usePBDRigidBody = false;
    }
}

const int SofaPBDPointCollisionModel::getPBDRigidBodyIndex() const
{
    if (m_d->m_pbdRigidBody)
        return m_d->m_pbdRigidBody->getPBDRigidBodyIndex();

    return -1;
}

void SofaPBDPointCollisionModel::draw(const core::visual::VisualParams* vparams)
{
    if (m_d->m_cubeModel)
        m_d->m_cubeModel->draw(vparams);
}

const sofa::core::CollisionModel* SofaPBDPointCollisionModel::toCollisionModel() const
{
    return Base::toCollisionModel();
}

sofa::core::CollisionModel* SofaPBDPointCollisionModel::toCollisionModel()
{
    return Base::toCollisionModel();
}

bool SofaPBDPointCollisionModel::insertInNode(objectmodel::BaseNode *node)
{
    return BaseObject::insertInNode(node);
}

bool SofaPBDPointCollisionModel::removeInNode(objectmodel::BaseNode *node)
{
    return BaseObject::removeInNode(node);
}

void SofaPBDPointCollisionModel::computeBoundingTree(int maxDepth)
{
    if (m_d->m_useMState && !this->mstate)
    {
        msg_warning("SofaPBDPointCollisionModel") << "No valid MechanicalState instance provided, can't continue.";
        return;
    }

    m_d->m_cubeModel = createPrevious<CubeModel>();
    int npoints = 0;

    if (m_d->m_useMState)
    {
        npoints = mstate->getSize();
    }
    else
    {
        if (m_d->m_usePBDRigidBody && !m_d->m_usePBDLineModel)
            npoints = m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().size();

        if (m_d->m_usePBDLineModel && !m_d->m_usePBDRigidBody)
            npoints = m_d->m_pbdLineModel->getPBDLineModel()->getNumPoints();
    }

    msg_info("SofaPBDPointCollisionModel") << "Number of points: " << npoints;

    bool updated = false;

    if (npoints != size)
    {
        msg_info("SofaPBDPointCollisionModel") << "Resizing collision model to: " << npoints;
        resize(npoints);
        updated = true;
    }

    if (updated)
    {
        msg_info("SofaPBDPointCollisionModel") << "Marking cubeModel for re-computation.";
        m_d->m_cubeModel->resize(0);
    }
    if (!isMoving() && !m_d->m_cubeModel->empty() && !updated)
    {
        msg_info("SofaPBDPointCollisionModel") << "Suppressing BVH update, model is static.";
        return; // No need to recompute BBox if immobile
    }
    if (computeNormals.getValue())
    {
        msg_info("SofaPBDPointCollisionModel") << "Updating normals.";
        updateNormals();
    }

    m_d->m_cubeModel->resize(size);

    if (!empty())
    {
        msg_info("SofaPBDPointCollisionModel") << "Model is not empty, updating cubeModel.";
        const SReal distance = this->proximity.getValue();

        if (m_d->m_usePBDRigidBody)
        {
            msg_info("SofaPBDPointCollisionModel") << "Use SofaPBDRigidBody geometry for BVH update.";
            const PBDRigidBodyGeometry& pbdRBGeom = m_d->m_pbdRigidBody->getRigidBodyGeometry();
            for (int i = 0; i < npoints; i++)
            {
                const Vector3r& rbp = pbdRBGeom.getVertexData().getPosition(i);
                defaulttype::Vector3 pt(rbp[0], rbp[1], rbp[2]);

                m_d->m_cubeModel->setParentOf(i, pt - defaulttype::Vector3(distance,distance,distance), pt + defaulttype::Vector3(distance,distance,distance));
            }
        }

        if (m_d->m_usePBDLineModel)
        {
            msg_info("SofaPBDPointCollisionModel") << "Use SofaPBDLineModel geometry for BVH update.";
            PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
            const PBDParticleData &pd = model->getParticles();

            PBDLineModel::Edges& lm_edges = m_d->m_pbdLineModel->getPBDLineModel()->getEdges();
            unsigned int pt_idx = 0;
            for (int i = 0; i < lm_edges.size(); i++)
            {
                const PBDLineModel::OrientedEdge& edge = lm_edges[i];
                const Vector3r& lpt_i1 = pd.getPosition(edge.m_vert[0]);
                const Vector3r& lpt_i2 = pd.getPosition(edge.m_vert[1]);
                defaulttype::Vector3 pt1(lpt_i1[0], lpt_i1[1], lpt_i1[2]);
                defaulttype::Vector3 pt2(lpt_i2[0], lpt_i2[1], lpt_i2[2]);

                m_d->m_cubeModel->setParentOf(pt_idx, pt1 - defaulttype::Vector3(distance,distance,distance), pt1 + defaulttype::Vector3(distance,distance,distance));
                m_d->m_cubeModel->setParentOf(pt_idx + 1, pt2 - defaulttype::Vector3(distance,distance,distance), pt2 + defaulttype::Vector3(distance,distance,distance));
                pt_idx += 2;
            }
        }

        m_d->m_cubeModel->computeBoundingTree(maxDepth);
        msg_info("SofaPBDPointCollisionModel") << "Cells in BVH: " << m_d->m_cubeModel->getNumberCells();
        sofa::helper::vector< std::pair< sofa::defaulttype::Vector3, sofa::defaulttype::Vector3> > model_bvh;

        m_d->m_cubeModel->getBoundingTree(model_bvh);
        for (size_t k = 0; k < model_bvh.size(); k++)
        {
            msg_info("SofaPBDPointCollisionModel") << "BVH cell " << k << ": " << model_bvh[k].first << " -- " << model_bvh[k].second;
        }
    }
    else
    {
        msg_info("SofaPBDPointCollisionModel") << "Model marked as empty, no BVH update possible.";
    }

    if (m_lmdFilter != 0)
    {
        m_lmdFilter->invalidate();
    }
}

const sofa::defaulttype::Vec3 SofaPBDPointCollisionModel::getCoord(unsigned int idx) const
{
    static sofa::defaulttype::Vec3 zeroVec(0, 0, 0);

    //msg_info("SofaPBDPointCollisionModel") << "getCoord(" << idx << ")";

    if (m_d->m_usePBDRigidBody)
    {
        //msg_info("SofaPBDPointCollisionModel") << "Using PBDRigidBody.";
        if (idx >= m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().size())
        {
            msg_info("SofaPBDPointCollisionModel") << "Index " << idx << " lies beyond RB geometry size " << m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().size() << " returning zeroVec.";
            return zeroVec;
        }
        Vector3r pt = m_d->m_pbdRigidBody->getRigidBodyGeometry().getVertexData().getPosition(idx);
        //msg_info("SofaPBDPointCollisionModel") << "RB vertex at index: " << idx << ": " << pt;
        return sofa::defaulttype::Vector3(pt[0], pt[1], pt[2]);
    }

    if (m_d->m_usePBDLineModel)
    {
        PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
        const PBDParticleData &pd = model->getParticles();

        PBDLineModel::Edges& lm_edges = m_d->m_pbdLineModel->getPBDLineModel()->getEdges();
        if (idx > lm_edges.size())
        {
            msg_info("SofaPBDPointCollisionModel") << "Index " << idx << " lies beyond line model size " << lm_edges.size() << ", returning zeroVec.";
            return zeroVec;
        }

        if (idx == lm_edges.size())
        {
            const PBDLineModel::OrientedEdge& edge = lm_edges.back();
            const Vector3r& lpt = pd.getPosition(edge.m_vert[1]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
        else
        {
            const PBDLineModel::OrientedEdge& edge = lm_edges[idx];
            const Vector3r& lpt = pd.getPosition(edge.m_vert[0]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
    }

    return zeroVec;
}

const sofa::defaulttype::Vec3 SofaPBDPointCollisionModel::getDeriv(unsigned int idx) const
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
        PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
        const PBDParticleData &pd = model->getParticles();

        PBDLineModel::Edges& lm_edges = m_d->m_pbdLineModel->getPBDLineModel()->getEdges();
        if (idx > lm_edges.size())
            return zeroVec;

        if (idx == lm_edges.size())
        {
            const PBDLineModel::OrientedEdge& edge = lm_edges.back();
            const Vector3r& lpt = pd.getVelocity(edge.m_vert[1]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
        else
        {
            const PBDLineModel::OrientedEdge& edge = lm_edges[idx];
            const Vector3r& lpt = pd.getVelocity(edge.m_vert[0]);
            return sofa::defaulttype::Vector3(lpt[0], lpt[1], lpt[2]);
        }
    }

    return zeroVec;
}

void SofaPBDPointCollisionModel::computeBBox(const core::ExecParams* params, bool onlyVisible)
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
        const defaulttype::Vector3& p = e.p();

        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c])
                maxBBox[c] = (Real)p[c];
            else if (p[c] < minBBox[c])
                minBBox[c] = (Real)p[c];
        }
    }

    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(minBBox,maxBBox));
}
