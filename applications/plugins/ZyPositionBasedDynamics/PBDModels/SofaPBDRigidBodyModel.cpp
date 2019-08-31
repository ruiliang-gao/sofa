#include "SofaPBDRigidBodyModel.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/loader/MeshLoader.h>
#include <SofaLoader/MeshObjLoader.h>

#include <PBDMain/SofaPBDSimulation.h>
#include <PBDUtils/PBDIndexedFaceMesh.h>

#include <PBDSimulation/PBDRigidBodyGeometry.h>

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
            class SofaPBDRigidBodyModelPrivate
            {
                public:
                    SofaPBDRigidBodyModelPrivate(): m_srcLoaderObject(NULL)
                    {
                        m_numPoints = 0;
                        m_numFaces = 0;
                        m_pbdIndexedFaceMesh.reset(new Utilities::PBDIndexedFaceMesh());
                        m_densitySet = m_massSet = false;

                        m_pbdRBIndex = -1;
                    }

                    std::shared_ptr<Utilities::PBDIndexedFaceMesh> m_pbdIndexedFaceMesh;
                    PBDVertexData m_vertexData;

                    std::string m_srcLoader;
                    sofa::core::loader::BaseLoader* m_srcLoaderObject;

                    int m_pbdRBIndex;

                    unsigned int m_numPoints;
                    unsigned int m_numFaces;

                    bool m_densitySet;
                    bool m_massSet;
            };
        }
    }
}

using namespace sofa::simulation::PBDSimulation;
using namespace sofa::simulation::PBDSimulation::Utilities;
using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(SofaPBDRigidBodyModel)

int SofaPBDRigidBodyModelClass = sofa::core::RegisterObject("Wrapper class for PBD TriangleModels.")
                            .add< SofaPBDRigidBodyModel >()
                            .addDescription("Encapsulates sets of particles in an indexed triangle mesh.");

SofaPBDRigidBodyModel::SofaPBDRigidBodyModel(): SofaPBDModelBase(),
    mass(initData(&mass, 0.0, "mass", "Rigid body total mass.")),
    inertiaTensor(initData(&inertiaTensor, Vec3d(1.0, 1.0, 1.0), "inertiaTensor", "Rigid body inertia tensor.")),
    density(initData(&density, 1.0, "density", "Rigid body material density.")),
    frictionCoefficient(initData(&frictionCoefficient, 0.1, "frictionCoefficient", "Rigid body friction coefficient."))
{
    m_d.reset(new SofaPBDRigidBodyModelPrivate());
}

void SofaPBDRigidBodyModel::parse(BaseObjectDescription* arg)
{
    if (arg->getAttribute("src"))
    {
        std::string valueString(arg->getAttribute("src"));

        msg_info("SofaPBDRigidBodyModel") << "'src' attribute given for SofaPBDRigidBodyModel '" << this->getName() << ": " << valueString;

        if (valueString[0] != '@')
        {
            msg_error("SofaPBDRigidBodyModel") <<"'src' attribute value should be a link using '@'";
        }
        else
        {
            msg_info("SofaPBDRigidBodyModel") << "src attribute: " << valueString;
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

    if (arg->getAttribute("ixx") != NULL || arg->getAttribute("iyy") != NULL || arg->getAttribute("izz") != NULL)
    {
        inertiaTensor.setValue(Vector3((SReal)arg->getAttributeAsFloat("ixx", 1.0),
                                  (SReal)arg->getAttributeAsFloat("iyy", 1.0),
                                  (SReal)arg->getAttributeAsFloat("izz", 1.0)));

        if (arg->getAttribute("ixx") != NULL)
            arg->removeAttribute("ixx");

        if (arg->getAttribute("iyy") != NULL)
            arg->removeAttribute("iyy");

        if (arg->getAttribute("izz") != NULL)
            arg->removeAttribute("izz");
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

        msg_info("SofaPBDRigidBodyModel") << "Euler angles: " << rotation;

        sofa::defaulttype::Quaternion quat = Quaternion::createQuaterFromEuler(rotation.getValue() * M_PI / 180.0);
        msg_info("SofaPBDRigidBodyModel") << "Resulting quaternion: " << quat;

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

    if (arg->getAttribute("mass") != NULL)
    {
        SReal m = (SReal)arg->getAttributeAsFloat("mass", 0.0);
        msg_info("SofaPBDRigidBodyModel") << "mass attribute specified. value = " << m;

        mass.setValue(m);
        m_d->m_massSet = true;

        arg->removeAttribute("mass");
    }

    if (arg->getAttribute("density") != NULL)
    {
        SReal d = (SReal)arg->getAttributeAsFloat("density", 1.0);
        msg_info("SofaPBDRigidBodyModel") << "density attribute specified. value = " << d;

        density.setValue(d);
        m_d->m_densitySet = true;

        arg->removeAttribute("density");
    }

    if (arg->getAttribute("frictionCoefficient") != NULL)
    {
        SReal f = (SReal)arg->getAttributeAsFloat("frictionCoefficient", 0.1);
        frictionCoefficient.setValue(f);

        arg->removeAttribute("frictionCoefficient");
    }

    BaseObject::parse(arg);
}

const int SofaPBDRigidBodyModel::getPBDRigidBodyIndex() const
{
    return m_d->m_pbdRBIndex;
}

void SofaPBDRigidBodyModel::init()
{

}

void SofaPBDRigidBodyModel::bwdInit()
{
    buildModel();
    initializeModel();
}

void SofaPBDRigidBodyModel::buildModel()
{
    msg_info("SofaPBDRigidBodyModel") << "buildModel() " << this->getName();
    if (m_d->m_srcLoader != "")
    {
        msg_info("SofaPBDRigidBodyModel") << "Found source loader instance: " << m_d->m_srcLoader;
        if (this->getContext())
        {
            std::vector<sofa::core::loader::BaseLoader*> loaderObjects = this->getContext()->getObjects<sofa::core::loader::BaseLoader>(BaseContext::SearchUp);
            msg_info("SofaPBDRigidBodyModel") << "BaseLoader object instances in scene: " << loaderObjects.size();
            if (loaderObjects.size() > 0)
            {
                std::string targetLoaderName = m_d->m_srcLoader;
                if (targetLoaderName[0] == '@')
                    targetLoaderName = targetLoaderName.substr(1);

                for (size_t k = 0; k < loaderObjects.size(); k++)
                {
                    std::string loaderName = loaderObjects[k]->getName();
                    msg_info("SofaPBDRigidBodyModel") << "Comparing names: " << loaderName << " <-> " << targetLoaderName;
                    if (loaderName.compare(targetLoaderName) == 0)
                    {
                        msg_info("SofaPBDRigidBodyModel") << "Found matching source loader object: " << loaderObjects[k]->getName() << " of type: " << loaderObjects[k]->getTypeName();
                        m_d->m_srcLoaderObject = loaderObjects[k];
                        break;
                    }
                }

                if (m_d->m_srcLoaderObject != NULL)
                {
                    msg_info("SofaPBDRigidBodyModel") << "Found a loader object to read geometry data from.";
                    if (dynamic_cast<sofa::core::loader::MeshLoader*>(m_d->m_srcLoaderObject))
                    {
                        msg_info("SofaPBDRigidBodyModel") << "Cast to MeshLoader instance successful.";

                        sofa::core::loader::MeshLoader* meshLoader = dynamic_cast<sofa::core::loader::MeshLoader*>(m_d->m_srcLoaderObject);
                        sofa::component::loader::MeshObjLoader* meshObjLoader = dynamic_cast<sofa::component::loader::MeshObjLoader*>(m_d->m_srcLoaderObject);

                        msg_info("SofaPBDRigidBodyModel") << "Vertex count in mesh  : " << meshLoader->d_positions.getValue().size();
                        msg_info("SofaPBDRigidBodyModel") << "Normal count in mesh  : " << meshLoader->d_normals.getValue().size();
                        msg_info("SofaPBDRigidBodyModel") << "Triangle count in mesh: " << meshLoader->d_triangles.getValue().size();

                        if (meshObjLoader)
                            msg_info("SofaPBDRigidBodyModel") << "Texture coords in mesh: " << meshObjLoader->texCoords.getValue().size();

                        PBDIndexedFaceMesh& mesh = *(m_d->m_pbdIndexedFaceMesh);
                        PBDVertexData& vd = m_d->m_vertexData;

                        mesh.release();

                        const unsigned int nPoints = (unsigned int) meshLoader->d_positions.getValue().size();
                        const unsigned int nFaces = (unsigned int) meshLoader->d_triangles.getValue().size();

                        m_d->m_numPoints = nPoints;
                        m_d->m_numFaces = nFaces;

                        unsigned int nTexCoords = 0;

                        const helper::vector<Vec3>& x = meshLoader->d_positions.getValue();
                        const helper::vector<sofa::core::topology::Topology::Triangle>& faces = meshLoader->d_triangles.getValue();

                        if (meshObjLoader)
                            nTexCoords = (unsigned int) meshObjLoader->texCoords.getValue().size();

                        mesh.initMesh(nPoints, nFaces * 2, nFaces);
                        vd.reserve(nPoints);

                        msg_info("SofaPBDRigidBodyModel") << "Adding vertices: " << nPoints;
                        for (unsigned int i = 0; i < nPoints; i++)
                        {
                            msg_info("SofaPBDRigidBodyModel") << "Vertex " << i << ": " << x[i];
                            vd.addVertex(Vector3r(x[i][0], x[i][1], x[i][2]));
                        }

                        if (nTexCoords > 0)
                        {
                            const helper::vector<sofa::defaulttype::Vector2>& texCoords = meshObjLoader->texCoords.getValue();
                            const helper::SVector<helper::SVector<int>>& texIndices = meshObjLoader->texIndexList.getValue();
                            for (unsigned int i = 0; i < nTexCoords; i++)
                            {
                                mesh.addUV(texCoords[i][0], texCoords[i][1]);
                            }

                            unsigned int nTexIndices = texIndices.size();

                            if (nTexCoords > 0)
                            {
                                for (unsigned int j = 0; j < nFaces; j++)
                                {
                                    int texIndicesFace[3];
                                    for (unsigned int k = 0; k < 3; k++)
                                    {
                                        texIndicesFace[k] = texIndices[j][k];
                                        mesh.addUVIndex(texIndicesFace[k]);
                                    }
                                }
                            }
                        }

                        msg_info("SofaPBDRigidBodyModel") << "Adding faces: " << nFaces;
                        for (unsigned int i = 0; i < nFaces; i++)
                        {
                            int posIndices[3];
                            for (int j = 0; j < 3; j++)
                            {
                                posIndices[j] = faces[i][j];
                            }

                            msg_info("SofaPBDRigidBodyModel") << "Face " << i << ": " << posIndices[0] << "," << posIndices[1] << "," << posIndices[2];
                            mesh.addFace(&posIndices[0]);
                        }

                        mesh.buildNeighbors();

                        mesh.updateNormals(vd, 0);
                        mesh.updateVertexNormals(vd);

                        msg_info("SofaPBDRigidBodyModel") << "Number of triangles in PBDIndexedMesh: " << mesh.numFaces();
                        msg_info("SofaPBDRigidBodyModel") << "Number of vertices in PBDIndexedMesh : " << mesh.numVertices();
                    }
                }
            }
        }
    }
    else
    {
        msg_warning("SofaPBDRigidBodyModel") << "Did not find source loader instance for SofaPBDRigidBodyModel " << this->getName();
    }
}

void SofaPBDRigidBodyModel::initializeModel()
{
    msg_info("SofaPBDRigidBodyModel") << "initializeModel() " << this->getName();
    PBDIndexedFaceMesh& mesh = *(m_d->m_pbdIndexedFaceMesh);

    msg_info("SofaPBDRigidBodyModel") << "Instantiating PBD rigid body object.";
    m_pbdRigidBody = new PBDRigidBody();

    msg_info("SofaPBDRigidBodyModel") << "Calling initBody on PBD rigid body object.";
    msg_info("SofaPBDRigidBodyModel") << "Position: " << translation.getValue() << ", orientation: " << rotationQuat.getValue();

    if (m_d->m_massSet && m_d->m_densitySet)
    {
        msg_info("SofaPBDRigidBodyModel") << "Both density and mass properties set. Using specified mass " << mass.getValue() << " to initialize PBDRigidBody.";
        m_pbdRigidBody->initBody(mass.getValue(),
            Vector3r(translation.getValue()[0], translation.getValue()[1], translation.getValue()[2]),
            Vector3r(inertiaTensor.getValue()[0], inertiaTensor.getValue()[1], inertiaTensor.getValue()[2]),
            Quaternionr(rotationQuat.getValue()[3], rotationQuat.getValue()[0], rotationQuat.getValue()[1], rotationQuat.getValue()[2]),
            m_d->m_vertexData, mesh,
            Vector3r(scale.getValue()[0], scale.getValue()[1], scale.getValue()[2]));
    }
    else if (m_d->m_massSet && !m_d->m_densitySet)
    {
        msg_info("SofaPBDRigidBodyModel") << "Only mass property specified. Using specified mass " << mass.getValue() << " to initialize PBDRigidBody.";
        m_pbdRigidBody->initBody(mass.getValue(),
            Vector3r(translation.getValue()[0], translation.getValue()[1], translation.getValue()[2]),
            Vector3r(inertiaTensor.getValue()[0], inertiaTensor.getValue()[1], inertiaTensor.getValue()[2]),
            Quaternionr(rotationQuat.getValue()[3], rotationQuat.getValue()[0], rotationQuat.getValue()[1], rotationQuat.getValue()[2]),
            m_d->m_vertexData, mesh,
            Vector3r(scale.getValue()[0], scale.getValue()[1], scale.getValue()[2]));
    }
    else if (!m_d->m_massSet && m_d->m_densitySet)
    {
        msg_info("SofaPBDRigidBodyModel") << "Only density property specified. Using specified density " << density.getValue() << " to initialize PBDRigidBody.";
        m_pbdRigidBody->initBody(density.getValue(),
            Vector3r(translation.getValue()[0], translation.getValue()[1], translation.getValue()[2]),
            Quaternionr(rotationQuat.getValue()[3], rotationQuat.getValue()[0], rotationQuat.getValue()[1], rotationQuat.getValue()[2]),
            m_d->m_vertexData, mesh,
            Vector3r(scale.getValue()[0], scale.getValue()[1], scale.getValue()[2]));
    }
    else
    {
        msg_info("SofaPBDRigidBodyModel") << "Neither density nor mass properties specified. Using default density 1.0 to initialize PBDRigidBody.";
        m_pbdRigidBody->initBody(1.0,
            Vector3r(translation.getValue()[0], translation.getValue()[1], translation.getValue()[2]),
            Quaternionr(rotationQuat.getValue()[3], rotationQuat.getValue()[0], rotationQuat.getValue()[1], rotationQuat.getValue()[2]),
            m_d->m_vertexData, mesh,
            Vector3r(scale.getValue()[0], scale.getValue()[1], scale.getValue()[2]));
    }

    /*m_pbdRigidBody->initBody(density.getValue(),
        Vector3r(translation.getValue()[0], translation.getValue()[1], translation.getValue()[2]),
        Quaternionr(rotationQuat.getValue()[3], rotationQuat.getValue()[0], rotationQuat.getValue()[1], rotationQuat.getValue()[2]),
        m_d->m_vertexData, mesh,
        Vector3r(scale.getValue()[0], scale.getValue()[1], scale.getValue()[2]));*/

    if (m_d->m_massSet)
    {
        msg_info("SofaPBDRigidBodyModel") << "Setting user-defined mass value: " << mass.getValue();
        m_pbdRigidBody->setMass(mass.getValue());
    }
    else
    {
        msg_info("SofaPBDRigidBodyModel") << "Using computed mass value: " << m_pbdRigidBody->getMass();
    }

    msg_info("SofaPBDRigidBodyModel") << "Setting friction coefficient: " << frictionCoefficient.getValue();
    m_pbdRigidBody->setFrictionCoeff(static_cast<Real>(frictionCoefficient.getValue()));

    PBDSimulationModel::RigidBodyVector& rigidBodies = SofaPBDSimulation::getCurrent()->getModel()->getRigidBodies();
    msg_info("SofaPBDRigidBodyModel") << "Adding rigid body to PBDSimulationModel. Number of rigid bodies before insertion: " << rigidBodies.size();

    rigidBodies.emplace_back(m_pbdRigidBody);
    msg_info("SofaPBDRigidBodyModel") << "Number of rigid bodies after insertion: " << rigidBodies.size();

    if (rigidBodies.size() == 1)
        m_d->m_pbdRBIndex = 0;
    else
        m_d->m_pbdRBIndex = rigidBodies.size() - 1;

    PBDRigidBodyGeometry& rbGeometry = m_pbdRigidBody->getGeometry();

    msg_info("SofaPBDRigidBodyModel") << "initializeModel() done: vertexCount = " << rbGeometry.getVertexData().size() << ", edgeCount = " << rbGeometry.getMesh().numEdges();

    msg_info("SofaPBDRigidBodyModel") << "=== Vertex data ===";
    for (unsigned int k = 0; k < rbGeometry.getVertexData().size(); k++)
    {
        msg_info("SofaPBDRigidBodyModel") << "Vertex " << k << ": (" << rbGeometry.getVertexData().getPosition(k)[0] << "," << rbGeometry.getVertexData().getPosition(k)[1] << "," << rbGeometry.getVertexData().getPosition(k)[2] << ")";
    }

    msg_info("SofaPBDRigidBodyModel") << "=== Local vertex data ===";
    for (unsigned int k = 0; k < rbGeometry.getVertexDataLocal().size(); k++)
    {
        msg_info("SofaPBDRigidBodyModel") << "Vertex " << k << ": (" << rbGeometry.getVertexDataLocal().getPosition(k)[0] << "," << rbGeometry.getVertexDataLocal().getPosition(k)[1] << "," << rbGeometry.getVertexDataLocal().getPosition(k)[2] << ")";
    }

    msg_info("SofaPBDRigidBodyModel") << "initializeModel() end";
}

void SofaPBDRigidBodyModel::draw(const core::visual::VisualParams* vparams)
{
    /*if (!vparams->displayFlags().getShowCollisionModels())
        return;*/

    PBDRigidBodyGeometry& rbGeometry = m_pbdRigidBody->getGeometry();

    Vec4f colour(1,0,0,0.5);
    Vec4f colour2(0,0,1,0.5);

    glPushMatrix();
    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_COLOR_MATERIAL);

    // Draw mesh vertices
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    for (unsigned int k = 0; k < rbGeometry.getVertexData().size(); k++)
    {
        glColor4f(1,1,0,0.75);
        Vector3r vt = rbGeometry.getVertexData().getPosition(k);
        // msg_info("SofaPBDRigidBodyModel") << "vertex[" << k << "] = " << vt;
        glVertex3d(vt[0], vt[1], vt[2]);
    }
    glEnd();
    glPointSize(1.0f);

    Utilities::PBDIndexedFaceMesh::Edges& meshEdges = rbGeometry.getMesh().getEdges();
    // Draw mesh lines
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    for (unsigned int k = 0; k < meshEdges.size(); k++)
    {
        const PBDIndexedFaceMesh::Edge& edge = meshEdges[k];
        const Vector3r& pt1 = rbGeometry.getVertexData().getPosition(edge.m_vert[0]);
        const Vector3r& pt2 = rbGeometry.getVertexData().getPosition(edge.m_vert[1]);
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
