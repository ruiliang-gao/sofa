#include "SofaPBDTriangleModel.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#include <PBDMain/SofaPBDSimulation.h>
#include <PBDUtils/PBDIndexedFaceMesh.h>

#ifdef _WIN32
#include <gl/glut.h>
#else
#include <GL/glut.h>
#endif

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTriangleModelPrivate
            {
                public:
                    SofaPBDTriangleModelPrivate()
                    {
                        m_pbdTriangleModel.reset(new PBDTriangleModel());
                    }

                    std::shared_ptr<PBDTriangleModel> m_pbdTriangleModel;
                    std::string m_srcLoader;
            };
        }
    }
}

using namespace sofa::simulation::PBDSimulation;
using namespace sofa::simulation::PBDSimulation::Utilities;
using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;

int SofaPBDTriangleModelClass = sofa::core::RegisterObject("Wrapper class for PBD TriangleModels.")
                            .add< SofaPBDTriangleModel >()
                            .addDescription("Encapsulates sets of particles in an indexed triangle mesh.");

SofaPBDTriangleModel::SofaPBDTriangleModel(): SofaPBDModelBase()
{
    m_d.reset(new SofaPBDTriangleModelPrivate());
}

void SofaPBDTriangleModel::parse(BaseObjectDescription* arg)
{
    if (arg->getAttribute("src"))
    {
        std::string valueString(arg->getAttribute("src"));

        msg_info("SofaPBDTriangleModel") << "'src' attribute given for SofaPBDTriangleModel '" << this->getName() << ": " << valueString;

        if (valueString[0] != '@')
        {
            msg_error("SofaPBDTriangleModel") <<"'src' attribute value should be a link using '@'";
        }
        else
        {
            msg_info("SofaPBDTriangleModel") << "src attribute: " << valueString;
            m_d->m_srcLoader = valueString;
        }
    }
    BaseObject::parse(arg);
}

void SofaPBDTriangleModel::init()
{
    m_d->m_pbdTriangleModel.reset(new PBDTriangleModel());
}

void SofaPBDTriangleModel::bwdInit()
{
    buildModel();
    applyInitialTransform();
}

void SofaPBDTriangleModel::buildModel()
{
    const Base::MapData& datas = this->getDataAliases();
    Base::MapData::const_iterator src_it = datas.find("src");
    if (src_it != datas.end())
    {
        msg_info("SofaPBDLineModel") << "Found src Data entry.";
        BaseData* src_data = src_it->second;
        msg_info("SofaPBDLineModel") << "src points to loader: " << src_data->getValueString();
    }
}

void SofaPBDTriangleModel::applyInitialTransform()
{

}

void SofaPBDTriangleModel::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowCollisionModels())
        return;

    PBDSimulationModel* simModel = SofaPBDSimulation::getCurrent()->getModel();
    PBDParticleData& particleData = simModel->getParticles();
    const PBDIndexedFaceMesh &mesh = m_d->m_pbdTriangleModel->getParticleMesh();
    const PBDIndexedFaceMesh::Edges& meshEdges = mesh.getEdges();
    const unsigned int vertexCount = mesh.numVertices();
    const unsigned int offset = m_d->m_pbdTriangleModel->getIndexOffset();

    Vec4f colour(1,0,0,0.5);
    Vec4f colour2(0,0,1,0.5);

    glPushMatrix();
    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_COLOR_MATERIAL);

    // Draw mesh particles
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    for (unsigned int k = offset; k < offset + vertexCount; k++)
    {
        glColor4f(1,1,0,0.75);
        Vector3r particle = particleData.getPosition(k);
        glVertex3d(particle[0], particle[1], particle[2]);
    }
    glEnd();
    glPointSize(1.0f);

    // Draw mesh lines
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    for (unsigned int k = 0; k < mesh.numEdges(); k++)
    {
        const PBDIndexedFaceMesh::Edge& edge = meshEdges[k];
        const Vector3r& pt1 = particleData.getPosition(edge.m_vert[0]);
        const Vector3r& pt2 = particleData.getPosition(edge.m_vert[1]);
        glColor4f(colour.x(), colour.y(), colour.z(), colour.w());
        glVertex3d(pt1[0], pt1[1], pt1[2]);
        glColor4f(colour2.x(), colour2.y(), colour2.z(), colour2.w());
        glVertex3d(pt2[0], pt2[1], pt2[2]);
    }
    glLineWidth(1.0f);
    glEnd();

    glPopAttrib();
    glPopMatrix();
}
