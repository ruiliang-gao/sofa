#include "SofaPBDLineModel.h"
#include "PBDMain/SofaPBDSimulation.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDLineModelPrivate
            {
                public:
                    SofaPBDLineModelPrivate(): m_drawFrames(true)
                    {
                        m_pbdLineModel.reset(new PBDLineModel());
                    }

                    std::shared_ptr<PBDLineModel> m_pbdLineModel;

                    std::string m_srcLoader;
                    bool m_drawFrames;
            };
        }
    }
}

using namespace sofa::core::objectmodel;
using namespace sofa::simulation::PBDSimulation;

int SofaPBDLineModelClass = sofa::core::RegisterObject("Wrapper class for PBD LineModels.")
                            .add< SofaPBDLineModel >()
                            .addDescription("Encapsulates sets of particles connected in a chain.");

SofaPBDLineModel::SofaPBDLineModel(): sofa::core::objectmodel::BaseObject()
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
    BaseObject::parse(arg);
}

void SofaPBDLineModel::init()
{
    m_d->m_pbdLineModel.reset(new PBDLineModel());
}

void SofaPBDLineModel::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowCollisionModels())
        return;

    // Draw simulation model
    PBDSimulationModel *model = SofaPBDSimulation::getCurrent()->getModel();
    const PBDParticleData &pd = model->getParticles();
    const PBDOrientationData &od = model->getOrientations();
    sofa::defaulttype::Vec4f red(0.8f, 0.0f, 0.0f, 1);
    sofa::defaulttype::Vec4f green(0.0f, 0.8f, 0.0f, 1);
    sofa::defaulttype::Vec4f blue(0.0f, 0.0f, 0.8f, 1);

    for (unsigned int i = 0; i < model->getLineModels().size(); i++)
    {
        for(unsigned int e = 0; e < m_d->m_pbdLineModel->getEdges().size(); e++)
        {
            const unsigned int indexOffset = m_d->m_pbdLineModel->getIndexOffset();
            const unsigned int indexOffsetQuaternions = m_d->m_pbdLineModel->getIndexOffsetQuaternions();
            const unsigned int i1 = m_d->m_pbdLineModel->getEdges()[e].m_vert[0] + indexOffset;
            const unsigned int i2 = m_d->m_pbdLineModel->getEdges()[e].m_vert[1] + indexOffset;
            const unsigned int iq = m_d->m_pbdLineModel->getEdges()[e].m_quat + indexOffsetQuaternions;
            const Vector3r &v1 = pd.getPosition(i1);
            const Vector3r &v2 = pd.getPosition(i2);
            const Quaternionr &q = od.getQuaternion(iq);

            sofa::defaulttype::Vec3f pos1(v1[0], v1[1], v1[2]);
            sofa::defaulttype::Vec3f pos2(v2[0], v2[1], v2[2]);

            vparams->drawTool()->setMaterial(blue);
            vparams->drawTool()->drawSphere(pos1, 0.07f);
            if(e == m_d->m_pbdLineModel->getEdges().size() - 1)
                vparams->drawTool()->drawSphere(pos2, 0.07f);

            if(m_d->m_drawFrames)
                vparams->drawTool()->drawCylinder(pos1, pos2, 0.01f, blue);
            else
                vparams->drawTool()->drawCylinder(pos1, pos2, 0.07f, blue);

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
}
