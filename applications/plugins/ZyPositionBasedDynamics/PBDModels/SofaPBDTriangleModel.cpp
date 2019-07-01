#include "SofaPBDTriangleModel.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#include <PBDUtils/PBDIndexedFaceMesh.h>

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

int SofaPBDTriangleModelClass = sofa::core::RegisterObject("Wrapper class for PBD TriangleModels.")
                            .add< SofaPBDTriangleModel >()
                            .addDescription("Encapsulates sets of particles in an indexed triangle mesh.");

SofaPBDTriangleModel::SofaPBDTriangleModel(): sofa::core::objectmodel::BaseObject()
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

void SofaPBDTriangleModel::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowCollisionModels())
        return;

    const PBDIndexedFaceMesh &mesh = m_d->m_pbdTriangleModel->getParticleMesh();
    const unsigned int offset = m_d->m_pbdTriangleModel->getIndexOffset();
    //Visualization::drawTexturedMesh(pd, mesh, offset, surfaceColor);

//    glPushMatrix();
//    glPushAttrib(GL_ENABLE_BIT);
//    glEnable(GL_COLOR_MATERIAL);

//    Vec4f colour(1,0,0,0.5);
//    Vec4f colour2(0,0,1,0.5);
//#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW
//    glBegin(GL_LINES);
//    glColor4d(colour2.x(), colour2.y(), colour2.z(), colour2.w());
//    glVertex3d(0, 0, 0);
//    glColor4d(colour.x(), colour.y(), colour.z(), colour.w());
//    glVertex3d(newTr.x(), newTr.y(), newTr.z());
//    glEnd();
//#endif //OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW

//    glTranslated(newTr.x(), newTr.y(), newTr.z());

//    BVHDrawHelpers::drawCoordinateMarkerGL(0.5f, 1.0f, colour, colour * 0.5, colour * 0.25);

//    //std::cout << " rotate to obj. orientation = " << newOrientation.transposed() << std::endl;
//    glMultMatrixd(modelGlOrientation.transposed().ptr());


//    glBegin(GL_LINES);
//    glColor4d(0, 1, 0, 0.5);
//    glVertex3d(0, 0, 0);
//    glColor4d(colour.x(), colour.y(), colour.z(), colour.w());
//    glVertex3d(m_pqp_tree->b->To[0], m_pqp_tree->b->To[1], m_pqp_tree->b->To[2]);
//    glEnd();

//    //std::cout << " translate to OBB center = " << m_pqp_tree->b->To[0] << "," << m_pqp_tree->b->To[1] << "," << m_pqp_tree->b->To[2] << std::endl;
//    glTranslated(m_pqp_tree->b->To[0], m_pqp_tree->b->To[1], m_pqp_tree->b->To[2]);

//    Matrix3 obbRotation; obbRotation.identity();
//    obbRotation[0] = Vector3(m_pqp_tree->b->R[0][0], m_pqp_tree->b->R[1][0], m_pqp_tree->b->R[2][0]);
//    obbRotation[1] = Vector3(m_pqp_tree->b->R[0][1], m_pqp_tree->b->R[1][1], m_pqp_tree->b->R[2][1]);
//    obbRotation[2] = Vector3(m_pqp_tree->b->R[0][2], m_pqp_tree->b->R[1][2], m_pqp_tree->b->R[2][2]);

//    Matrix4 glOrientation; glOrientation.identity();
//    for (int k = 0; k < 3; k++)
//    {
//        for (int l = 0; l < 3; l++)
//        {
//            glOrientation[k][l] = obbRotation[k][l];
//        }
//    }

//    //drawObbVolume(sofa::defaulttype::Vector3(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]), colour2);

//    //std::cout << " rotate to OBB orientation = " << glOrientation << std::endl;
//    BVHDrawHelpers::drawCoordinateMarkerGL(0.75f, 4.0f, colour2, colour2, colour);

//    glMultMatrixd(glOrientation.ptr());

//    BVHDrawHelpers::drawCoordinateMarkerGL(1.0f, 6.0f, colour, colour2, colour2);

//    BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]), colour);

//    float extent_x = m_pqp_tree->b->d[0]; float extent_y = m_pqp_tree->b->d[1]; float extent_z = m_pqp_tree->b->d[2];
//    if (m_pqp_tree->b->min_dimension == 0)
//        extent_x = m_pqp_tree->b->min_dimension_val;
//    else if (m_pqp_tree->b->min_dimension == 1)
//        extent_y = m_pqp_tree->b->min_dimension_val;
//    else if (m_pqp_tree->b->min_dimension == 2)
//        extent_z = m_pqp_tree->b->min_dimension_val;

//    BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(extent_x, extent_y, extent_z), Vec4f(0,1,0,1), true);

//    glMultMatrixd(glOrientation.transposed().ptr());

//    glTranslated(-m_pqp_tree->b->To[0], -m_pqp_tree->b->To[1], -m_pqp_tree->b->To[2]);

//    if (m_pqp_tree->b->first_child > 0)
//    {
//        BV* child1 = m_pqp_tree->child(m_pqp_tree->b->first_child);

//        Matrix3 childRotation; childRotation.identity();
//        childRotation[0] = Vector3(child1->R[0][0], child1->R[1][0], child1->R[2][0]);
//        childRotation[1] = Vector3(child1->R[0][1], child1->R[1][1], child1->R[2][1]);
//        childRotation[2] = Vector3(child1->R[0][2], child1->R[1][2], child1->R[2][2]);

//        Matrix4 glOrientation; glOrientation.identity();
//        for (int k = 0; k < 3; k++)
//        {
//            for (int l = 0; l < 3; l++)
//            {
//                glOrientation[k][l] = childRotation[k][l];
//            }
//        }

//        glTranslated(child1->To[0], child1->To[1], child1->To[2]);

//        glMultMatrixd(glOrientation.ptr());
//        BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(child1->d[0], child1->d[1], child1->d[2]), Vec4f(1, 1, 1, 0.5), true);
//        glMultMatrixd(glOrientation.transposed().ptr());

//        glTranslated(-child1->To[0], -child1->To[1], -child1->To[2]);
//    }

//    if (m_pqp_tree->b->first_child + 1 > 0)
//    {
//        BV* child2 = m_pqp_tree->child(m_pqp_tree->b->first_child + 1);

//        Matrix3 childRotation; childRotation.identity();
//        childRotation[0] = Vector3(child2->R[0][0], child2->R[1][0], child2->R[2][0]);
//        childRotation[1] = Vector3(child2->R[0][1], child2->R[1][1], child2->R[2][1]);
//        childRotation[2] = Vector3(child2->R[0][2], child2->R[1][2], child2->R[2][2]);

//        Matrix4 glOrientation; glOrientation.identity();
//        for (int k = 0; k < 3; k++)
//        {
//            for (int l = 0; l < 3; l++)
//            {
//                glOrientation[k][l] = childRotation[k][l];
//            }
//        }

//        glTranslated(child2->To[0], child2->To[1], child2->To[2]);

//        glMultMatrixd(glOrientation.ptr());
//        BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(child2->d[0], child2->d[1], child2->d[2]), Vec4f(1, 1, 1, 1), true);
//        glMultMatrixd(glOrientation.transposed().ptr());

//        glTranslated(-child2->To[0], -child2->To[1], -child2->To[2]);
//    }

//    if (m_drawOBBHierarchy.getValue() > 1)
//    {
//        if (m_pqp_tree->num_bvs > 2)
//        {
//            BVHDrawHelpers::drawRec(m_pqp_tree, m_pqp_tree->child(m_pqp_tree->b->first_child), vparams, colour2, 1, false);
//            BVHDrawHelpers::drawRec(m_pqp_tree, m_pqp_tree->child(m_pqp_tree->b->first_child+1), vparams, colour2, 1, false);
//        }
//    }

//    glPopAttrib();
//    glPopMatrix();

    vparams->drawTool()->drawTriangles();
}
