#include "SofaPBDLineModel.h"
#include "PBDMain/SofaPBDSimulation.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/loader/MeshLoader.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDLineModelPrivate
            {
                public:
                    SofaPBDLineModelPrivate(): m_drawFrames(true), m_srcLoaderObject(NULL)
                    {
                        m_numPoints = 0;
                        m_numEdges = 0;
                        m_numQuaternions = 0;
                        m_pbdLineModel.reset(new PBDLineModel());
                    }

                    std::shared_ptr<PBDLineModel> m_pbdLineModel;

                    std::string m_srcLoader;
                    sofa::core::loader::BaseLoader* m_srcLoaderObject;

                    unsigned int m_numPoints;
                    unsigned int m_numEdges;
                    unsigned int m_numQuaternions;

                    std::vector<Vector3r> m_points;
                    std::vector<Quaternionr> m_quaternions;

                    std::vector<unsigned int> m_indices;
                    std::vector<unsigned int> m_indicesQuaternions;

                    bool m_drawFrames;
            };
        }
    }
}

using namespace sofa::core::objectmodel;
using namespace sofa::simulation::PBDSimulation;

SOFA_DECL_CLASS(SofaPBDLineModel)

int SofaPBDLineModelClass = sofa::core::RegisterObject("Wrapper class for PBD LineModels.")
                            .add< SofaPBDLineModel >()
                            .addDescription("Encapsulates sets of particles connected in a chain.");

SofaPBDLineModel::SofaPBDLineModel(): SofaPBDModelBase()
{
    m_d.reset(new SofaPBDLineModelPrivate());
}

void SofaPBDLineModel::parse(BaseObjectDescription* arg)
{
    if (arg->getAttribute("src"))
    {
        std::string valueString(arg->getAttribute("src"));

        msg_info("PBDLineModel") << "'src' attribute given for PBDLineModel '" << this->getName() << ": " << valueString;

        if (valueString[0] != '@')
        {
            msg_error("PBDLineModel") <<"'src' attribute value should be a link using '@'";
        }
        else
        {
            msg_info("PBDLineModel") << "src attribute: " << valueString;
            m_d->m_srcLoader = valueString;
        }
    }

    if (arg->getAttribute("scale") != NULL)
    {
        SReal s = (SReal)arg->getAttributeAsFloat("scale", 1.0);
        scale.setValue(Vector3(s, s, s));

        arg->removeAttribute("scale");
    }

    if (arg->getAttribute("sx") != NULL || arg->getAttribute("sy") != NULL || arg->getAttribute("sz") != NULL)
    {
        scale.setValue(Vector3((SReal)arg->getAttributeAsFloat("sx",1.0),
                               (SReal)arg->getAttributeAsFloat("sy",1.0),
                               (SReal)arg->getAttributeAsFloat("sz",1.0)));

        if (arg->getAttribute("sx") != NULL)
            arg->removeAttribute("sx");

        if (arg->getAttribute("sy") != NULL)
            arg->removeAttribute("sy");

        if (arg->getAttribute("sz") != NULL)
            arg->removeAttribute("sz");
    }

    if (arg->getAttribute("rx") != NULL || arg->getAttribute("ry") != NULL || arg->getAttribute("rz") != NULL)
    {
        rotation.setValue(Vector3((SReal)arg->getAttributeAsFloat("rx",0.0),
                                  (SReal)arg->getAttributeAsFloat("ry",0.0),
                                  (SReal)arg->getAttributeAsFloat("rz",0.0)));

        if (arg->getAttribute("rx") != NULL)
            arg->removeAttribute("rx");

        if (arg->getAttribute("ry") != NULL)
            arg->removeAttribute("ry");

        if (arg->getAttribute("rz") != NULL)
            arg->removeAttribute("rz");

        msg_info("SofaPBDLineModel") << "Euler angles: " << rotation;

        sofa::defaulttype::Quaternion quat = Quaternion::createQuaterFromEuler(rotation.getValue() * M_PI / 180.0);
        msg_info("SofaPBDLineModel") << "Resulting quaternion: " << quat;

        rotationQuat.setValue(quat);
    }

    if (arg->getAttribute("dx") != NULL || arg->getAttribute("dy") != NULL || arg->getAttribute("dz") != NULL)
    {
        translation.setValue(Vector3((Real)arg->getAttributeAsFloat("dx",0.0),
                                     (Real)arg->getAttributeAsFloat("dy",0.0),
                                     (Real)arg->getAttributeAsFloat("dz",0.0)));

        if (arg->getAttribute("dx") != NULL)
            arg->removeAttribute("dx");

        if (arg->getAttribute("dy") != NULL)
            arg->removeAttribute("dy");

        if (arg->getAttribute("dz") != NULL)
            arg->removeAttribute("dz");
    }

    BaseObject::parse(arg);
}

std::shared_ptr<PBDLineModel> SofaPBDLineModel::getPBDLineModel() const
{
    return m_d->m_pbdLineModel;
}

void SofaPBDLineModel::init()
{
    m_d->m_pbdLineModel.reset(new PBDLineModel());
}

void SofaPBDLineModel::bwdInit()
{
    msg_info("SofaPBDLineModel") << "bwdInit() " << this->getName();
    buildModel();
    initializeModel();
}

void SofaPBDLineModel::cleanup()
{

}

void SofaPBDLineModel::buildModel()
{
    if (m_d->m_srcLoader != "")
    {
        msg_info("SofaPBDLineModel") << "Found source loader instance: " << m_d->m_srcLoader;
        if (this->getContext())
        {
            std::vector<sofa::core::loader::BaseLoader*> loaderObjects = this->getContext()->getObjects<sofa::core::loader::BaseLoader>(BaseContext::SearchRoot);
            msg_info("SofaPBDLineModel") << "BaseLoader object instances in scene: " << loaderObjects.size();
            if (loaderObjects.size() > 0)
            {
                std::string targetLoaderName = m_d->m_srcLoader;
                if (targetLoaderName[0] == '@')
                    targetLoaderName = targetLoaderName.substr(1);

                for (size_t k = 0; k < loaderObjects.size(); k++)
                {
                    std::string loaderName = loaderObjects[k]->getName();
                    msg_info("SofaPBDLineModel") << "Comparing names: " << loaderName << " <-> " << targetLoaderName;
                    if (loaderName.compare(targetLoaderName) == 0)
                    {
                        msg_info("SofaPBDLineModel") << "Found matching source loader object: " << loaderObjects[k]->getName() << " of type: " << loaderObjects[k]->getTypeName();
                        m_d->m_srcLoaderObject = loaderObjects[k];
                        break;
                    }
                }

                if (m_d->m_srcLoaderObject != NULL)
                {
                    msg_info("SofaPBDLineModel") << "Found a loader object to read geometry data from.";
                    if (dynamic_cast<sofa::core::loader::MeshLoader*>(m_d->m_srcLoaderObject))
                    {
                        msg_info("SofaPBDLineModel") << "Cast to MeshLoader instance successful.";
                        sofa::core::loader::MeshLoader* meshLoader = dynamic_cast<sofa::core::loader::MeshLoader*>(m_d->m_srcLoaderObject);
                        msg_info("SofaPBDLineModel") << "Vertex count in mesh: " << meshLoader->d_positions.getValue().size();
                        msg_info("SofaPBDLineModel") << "Edge count in mesh  : " << meshLoader->d_edges.getValue().size();

                        int nPoints = meshLoader->d_positions.getValue().size();
                        int nEdges = meshLoader->d_edges.getValue().size();
                        int nQuaternions = nPoints - 1;

                        /*std::vector<Vector3r> points(nPoints);
                        std::vector<Quaternionr> quaternions(nQuaternions);*/

                        m_d->m_points.resize(nPoints);
                        m_d->m_quaternions.resize(nQuaternions);

                        m_d->m_numPoints = nPoints;
                        m_d->m_numEdges = nEdges;
                        m_d->m_numQuaternions = nQuaternions;

                        const helper::vector<Vec3>& linePoints = meshLoader->d_positions.getValue();
                        const helper::vector<sofa::core::topology::Topology::Edge> edges = meshLoader->d_edges.getValue();

                        Vector3r position(translation.getValue()[0], translation.getValue()[1], translation.getValue()[2]);
                        Quaternionr orientation(rotationQuat.getValue()[3], rotationQuat.getValue()[0], rotationQuat.getValue()[1], rotationQuat.getValue()[2]);

                        msg_info("SofaPBDLineModel") << "Position: " << position;
                        msg_info("SofaPBDLineModel") << "Orientation: (" << orientation.x() << "," << orientation.y() << "," << orientation.z() << "," << orientation.w() << ")";

                        // init particles
                        for (int i = 0; i < nPoints; i++)
                        {
                            m_d->m_points[i].x() = linePoints[i][0];
                            m_d->m_points[i].y() = linePoints[i][1];
                            m_d->m_points[i].z() = linePoints[i][2];

                            m_d->m_points[i] = orientation * m_d->m_points[i] + position;

                            // Line collision primitives
                            /* if (i > 0)
                            {
                                cd.addCollisionLine(2, CollisionDetection::CollisionObject::LineModelCollisionObjectType,
                                                    points[i-1], points[i], i-1, i, Real(0.1));
                            }*/
                        }

                        // init quaternions
                        Vector3r from(m_d->m_points[0].x(), m_d->m_points[0].y(), m_d->m_points[0].z());
                        for(int i = 0; i < nQuaternions; i++)
                        {
                            Vector3r to = (m_d->m_points[i + 1] - m_d->m_points[i]).normalized();
                            Quaternionr dq = Quaternionr::FromTwoVectors(from, to);
                            if (i == 0)
                                m_d->m_quaternions[i] = dq;
                            else
                                m_d->m_quaternions[i] = dq * m_d->m_quaternions[i - 1];

                            from = to;
                        }

                        /*std::vector<unsigned int> indices(2 * nPoints - 1);
                        std::vector<unsigned int> indicesQuaternions(nQuaternions);*/

                        m_d->m_indices.resize(2 * nPoints - 1);
                        m_d->m_indicesQuaternions.resize(nQuaternions);

                        for(int i = 0; i < nEdges; i++)
                        {
                            m_d->m_indices[2 * i] = edges[i][0];
                            m_d->m_indices[2 * i + 1] = edges[i][1];
                        }

                        for (int i = 0; i < nQuaternions; i++)
                        {
                            m_d->m_indicesQuaternions[i] = i;
                        }
                    }
                }
            }
        }
    }
    else
    {
        msg_warning("SofaPBDLineModel") << "Did not find source loader instance for SofaPBDLineModel " << this->getName();
    }
}

void SofaPBDLineModel::initializeModel()
{
    PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();

    model->addLineModel(m_d->m_pbdLineModel.get(), m_d->m_numPoints, m_d->m_numQuaternions, &(m_d->m_points[0]), &(m_d->m_quaternions)[0], &(m_d->m_indices)[0], &(m_d->m_indicesQuaternions)[0]);

    // TODO: Set mass from SOFA classes, and/or from given mass in SofaPBDBaseModel
    PBDParticleData &pd = model->getParticles();
    const int nPointsTotal = pd.getNumberOfParticles();
    for (int i = nPointsTotal - 1; i > nPointsTotal - m_d->m_numPoints; i--)
    {
        pd.setMass(i, 1.0);
    }

    // Set mass of first point to zero => make it static
    // TODO: Are free-moving line models sensible? Include an option to fix first point, or not?
    pd.setMass(nPointsTotal - m_d->m_numPoints, 0.0);

    PBDOrientationData &od = model->getOrientations();
    const unsigned int nQuaternionsTotal = od.getNumberOfQuaternions();
    for (unsigned int i = nQuaternionsTotal - 1; i > nQuaternionsTotal - m_d->m_numQuaternions; i--)
    {
        od.setMass(i, 1.0);
    }

    // Set mass of quaternions to zero => make it static
    // TODO: Are free-moving line models sensible? Include an option to fix first point, or not?
    od.setMass(nQuaternionsTotal - m_d->m_numQuaternions, 0.0);

    // init constraints
    const size_t rodNumber = model->getLineModels().size() - 1;
    const unsigned int offset = model->getLineModels()[rodNumber]->getIndexOffset();
    const unsigned int offsetQuaternions = model->getLineModels()[rodNumber]->getIndexOffsetQuaternions();
    const size_t nEdges = model->getLineModels()[rodNumber]->getEdges().size();
    const PBDLineModel::Edges &edges = model->getLineModels()[rodNumber]->getEdges();

    msg_info("SofaPBDLineModel") << "Adding constraints: startIndex = " << offset << ", startIndexQuaternions = " << offsetQuaternions;
    msg_info("SofaPBDLineModel") << "Constraints to add: stretchSchear = " << m_d->m_numEdges << ", bendTwist" << m_d->m_numEdges - 1;

    // stretchShear constraints
    for (unsigned int i = 0; i < nEdges; i++)
    {
        const unsigned int v1 = edges[i].m_vert[0] + offset;
        const unsigned int v2 = edges[i].m_vert[1] + offset;
        const unsigned int q1 = edges[i].m_quat + offsetQuaternions;

        msg_info("SofaPBDLineModel") << "Adding stretchShear constraint between particle indices: " << v1 << " - " << v2 << ", using quaternion index: " << q1;

        model->addStretchShearConstraint(v1, v2, q1);
    }

    // bendTwist constraints
    for (unsigned int i = 0; i < nEdges - 1; i++)
    {
        const unsigned int q1 = edges[i].m_quat + offsetQuaternions;
        const unsigned int q2 = edges[i + 1].m_quat + offsetQuaternions;

        msg_info("SofaPBDLineModel") << "Adding bendTwist constraint between quaternion indices: " << q1 << " - " << q2;

        model->addBendTwistConstraint(q1, q2);
    }

    msg_info("SofaPBDLineModel") << "Number of particles in PBDLineModel: " << m_d->m_numPoints;
    msg_info("SofaPBDLineModel") << "Number of quaternions in PBDLineModel: " << m_d->m_numQuaternions;
}

void SofaPBDLineModel::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    // Draw simulation model
    PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
    const PBDParticleData &pd = model->getParticles();
    const PBDOrientationData &od = model->getOrientations();
    sofa::defaulttype::Vec4f red(0.8f, 0.0f, 0.0f, 1);
    sofa::defaulttype::Vec4f green(0.0f, 0.8f, 0.0f, 1);
    sofa::defaulttype::Vec4f blue(0.0f, 0.0f, 0.8f, 1);

    const unsigned int indexOffset = m_d->m_pbdLineModel->getIndexOffset();
    const unsigned int indexOffsetQuaternions = m_d->m_pbdLineModel->getIndexOffsetQuaternions();

    // msg_info("SofaPBDLineModel") << "draw() -- edges = " << m_d->m_pbdLineModel->getEdges().size() << "; indexOffset = " << indexOffset << ", indexOffsetQuaternions = " << indexOffsetQuaternions;

    for(unsigned int e = 0; e < m_d->m_pbdLineModel->getEdges().size(); e++)
    {
        const unsigned int i1 = m_d->m_pbdLineModel->getEdges()[e].m_vert[0] + indexOffset;
        const unsigned int i2 = m_d->m_pbdLineModel->getEdges()[e].m_vert[1] + indexOffset;
        const unsigned int iq = m_d->m_pbdLineModel->getEdges()[e].m_quat + indexOffsetQuaternions;

        // msg_info("SofaPBDLineModel") << "Edge " << e << ": " << i1 << " - " << i2 << ", quaternion idx = " << iq;

        const Vector3r &v1 = pd.getPosition(i1);
        const Vector3r &v2 = pd.getPosition(i2);
        const Quaternionr &q = od.getQuaternion(iq);

        sofa::defaulttype::Vec3f pos1(v1[0], v1[1], v1[2]);
        sofa::defaulttype::Vec3f pos2(v2[0], v2[1], v2[2]);

        vparams->drawTool()->setMaterial(blue);
        vparams->drawTool()->drawSphere(pos1, 0.02f);
        if(e == m_d->m_pbdLineModel->getEdges().size() - 1)
            vparams->drawTool()->drawSphere(pos2, 0.02f);

        if(m_d->m_drawFrames)
            vparams->drawTool()->drawCylinder(pos1, pos2, 0.02f, blue);
        else
            vparams->drawTool()->drawCylinder(pos1, pos2, 0.01f, blue);

        //draw coordinate frame at the center of the edges
        if(m_d->m_drawFrames)
        {
            sofa::defaulttype::Vec3f vm = 0.5 * (pos1 + pos2);
            Real scale = static_cast<Real>(0.15);
            Vector3r d1 = q._transformVector(Vector3r(1, 0, 0)) * scale;
            Vector3r d2 = q._transformVector(Vector3r(0, 1, 0)) * scale;
            Vector3r d3 = q._transformVector(Vector3r(0, 0, 1)) * scale;

            sofa::defaulttype::Vec3f axis1(d1[0], d1[1], d1[2]);
            sofa::defaulttype::Vec3f axis2(d2[0], d2[1], d2[2]);
            sofa::defaulttype::Vec3f axis3(d3[0], d3[1], d3[2]);

            vparams->drawTool()->drawCylinder(vm, vm + axis1, 0.01f, red);
            vparams->drawTool()->drawCylinder(vm, vm + axis2, 0.01f, green);
            vparams->drawTool()->drawCylinder(vm, vm + axis3, 0.01f, blue);
        }
    }
}
