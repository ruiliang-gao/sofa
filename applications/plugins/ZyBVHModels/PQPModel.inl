#ifndef PQPMODEL_INL
#define PQPMODEL_INL

#include "PQPModel.h"

#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/Node.h>
#include <SofaLoader/MeshObjLoader.h>

#include "BVHDrawHelpers.h"
#include "SofaMeshCollision/Triangle.h"

#include <GL/gl.h>

using namespace sofa::component::collision;
using namespace sofa::core::objectmodel;
using namespace sofa;

template <class DataTypes>
PQPCollisionModel<DataTypes>::PQPCollisionModel(): core::CollisionModel(), m_modelLoaded(false),
                                                   m_cubeModel(NULL),
                                                   _mState(NULL), _mObject(NULL), _objectMState(NULL), m_meshLoader(NULL)
{

}

template <class DataTypes>
PQPCollisionModel<DataTypes>::~PQPCollisionModel()
{
    if (m_pqp_tree)
    {
        delete m_pqp_tree;
        m_pqp_tree = NULL;
    }
}

template <class DataTypes>
void PQPCollisionModel<DataTypes>::init()
{
    using namespace sofa::component::container;

    m_lastTimestep = -1.0;

    _mState = dynamic_cast< core::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());

    Vector3 scale(1,1,1);
    Real uniformScale = 1.0f;

    component::container::MechanicalObject<DataTypes>* mechanicalObject = dynamic_cast< component::container::MechanicalObject<DataTypes>* >(_mState);
    if (mechanicalObject != NULL)
    {

        typedef MechanicalObject<RigidTypes> mType;

        std::vector<mType*> mechObjects;
        sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<mType, std::vector<mType*> > cbMech(&mechObjects);

        getContext()->getObjects(TClassInfo<mType>::get(), cbMech, TagSet(), BaseContext::SearchParents);

        for (std::vector<mType*>::iterator it = mechObjects.begin(); it != mechObjects.end(); it++)
        {
            mType* current = (*it);
            Rigid3dTypes::VecCoord coords = current->getPosition();
            m_initialPosition = Vec3d(coords[0][0], coords[0][1], coords[0][2]);
            m_initialOrientation = Quat(coords[0][3], coords[0][4], coords[0][5], coords[0][6]);
            m_scale = uniformScale; // TODO! m_scale should be a Vec3d!!!!!!        mscale = current->getScale();
            scale = current->getScale();
            if (scale.x() == scale.y() && scale.y() == scale.z()) {
                  m_scale = scale.x();
            }
            _objectMState = current;
        }

        simulation::Node* parentNode = dynamic_cast<simulation::Node*>(this->getContext());

        if (parentNode)
        {
            core::objectmodel::BaseNode::Parents grandParents = parentNode->getParents();
            if (grandParents.size() == 1)
            {
                simulation::Node* grandParentNode = dynamic_cast<simulation::Node*>(grandParents[0]);

                std::vector<sofa::component::container::MechanicalObject<DataTypes>* > mo_vec;
                sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::container::MechanicalObject<DataTypes>, std::vector<sofa::component::container::MechanicalObject<DataTypes>* > > mo_cb(&mo_vec);

                grandParentNode->getObjects(TClassInfo<sofa::component::container::MechanicalObject<DataTypes> >::get(), mo_cb, TagSet(), BaseContext::SearchDown);

                if (mo_vec.size() > 0)
                    _mObject = mo_vec.at(0);
            }
        }
    }

    if (getContext() && getContext()->getRootContext())
    {
        std::vector<sofa::component::loader::MeshObjLoader* > molV;
        sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::loader::MeshObjLoader,std::vector<sofa::component::loader::MeshObjLoader* > > mol_cb(&molV);

        getContext()->getObjects(sofa::core::objectmodel::TClassInfo<sofa::component::loader::MeshObjLoader>::get(), mol_cb, sofa::core::objectmodel::TagSet(), sofa::core::objectmodel::BaseContext::SearchDown);
        if (molV.size() > 0)
        {
            std::cout << "MeshOBJLoader objects found: " << molV.size() << std::endl;
            for (std::vector<sofa::component::loader::MeshObjLoader*>::iterator it = molV.begin(); it != molV.end(); it++)
            {
                std::cout << " * OBJ loader: " << (*it)->getName() << std::endl;
                sofa::core::objectmodel::BaseData* fnData = (*it)->findData("filename");
                if (fnData)
                {
                    const void* txtValue = fnData->getValueVoidPtr();
                    m_modelFile = fnData->getValueTypeInfo()->getTextValue(txtValue, 0);
                    std::cout << "       File name: " << m_modelFile << std::endl;

                    bool fileFound = sofa::helper::system::DataRepository.findFile(m_modelFile);
                    std::string absFilePath;
                    if (fileFound)
                    {
                        std::cout << "     File found." << std::endl;
                        absFilePath = sofa::helper::system::DataRepository.getFile(m_modelFile);
                        m_modelFile = absFilePath;
                    }
                    else
                    {
                        std::cerr << "       ERROR: Failed to resolve absolute path of " << m_modelFile << "!" << std::endl;
                        m_modelFile.clear();
                    }

                    std::cout << "       Absolute path: " << m_modelFile << std::endl;
                }
            }
        }
    }

    if (!m_modelFile.empty())
        computeOBBHierarchy(true);
    else
        computeOBBHierarchy(false);
}

template <class DataTypes>
void PQPCollisionModel<DataTypes>::cleanup()
{

}

template <class DataTypes>
bool PQPCollisionModel<DataTypes>::getPosition(Vector3& position) const
{
    position = m_cachedPosition;
    return true;
}

template <class DataTypes>
bool PQPCollisionModel<DataTypes>::getOrientation(Matrix3& orientation) const
{
    m_cachedOrientation.toMatrix(orientation);
    return true;
}

template <class DataTypes>
bool PQPCollisionModel<DataTypes>::getOrientation(Quaternion& orientation) const
{
    orientation = m_cachedOrientation;
    return true;
}

template <class DataTypes>
void PQPCollisionModel<DataTypes>::setCachedPosition(const Vec3d &position)
{
    m_cachedPosition = position;
}

template <class DataTypes>
void PQPCollisionModel<DataTypes>::setCachedOrientation(const Quaternion &orientation)
{
    m_cachedOrientation = orientation;
}

template <class DataTypes>
void PQPCollisionModel<DataTypes>::computeOBBHierarchy(bool fromFile)
{
    std::cout << "PQPCollisionModel<DataTypes>::computeOBBHierarchy(" << this->getName() << ")" << std::endl;
    if (fromFile && !m_modelFile.empty())
    {
        if (m_modelLoaded)
            return;

        if (!m_meshLoader || m_modelFile.empty())
            return;

        sofa::helper::vector<Vector3> meshVertices = this->m_meshLoader->d_positions.getValue();
        sofa::helper::vector<sofa::core::topology::Triangle> meshTriangles = this->m_meshLoader->d_triangles.getValue(); // added namespace and header after upstream merge

        m_pqp_tree = new PQP_Model();
        m_pqp_tree->BeginModel(meshTriangles.size());

        PQP_REAL v1[3], v2[3], v3[3];
        for (unsigned int i = 0; i < meshTriangles.size(); i++)
        {
            /*const Triangle& tri = meshTriangles[i];

            v1[0] = meshVertices[tri[0]].x();
            v1[1] = meshVertices[tri[0]].y();
            v1[2] = meshVertices[tri[0]].z();

            v2[0] = meshVertices[tri[1]].x();
            v2[1] = meshVertices[tri[1]].y();
            v2[2] = meshVertices[tri[1]].z();

            v3[0] = meshVertices[tri[2]].x();
            v3[1] = meshVertices[tri[2]].y();
            v3[2] = meshVertices[tri[2]].z();*/ // did not work this way after upstream merge

            const sofa::core::topology::Triangle& tri = meshTriangles[i]; // added namespace and header after upstream merge

            v1[0] = meshVertices[tri[0]].x();
            v1[1] = meshVertices[tri[0]].y();
            v1[2] = meshVertices[tri[0]].z();

            v2[0] = meshVertices[tri[1]].x();
            v2[1] = meshVertices[tri[1]].y();
            v2[2] = meshVertices[tri[1]].z();

            v3[0] = meshVertices[tri[2]].x();
            v3[1] = meshVertices[tri[2]].y();
            v3[2] = meshVertices[tri[2]].z();

            m_pqp_tree->AddTri(v1, v2, v3, i);
        }

        m_pqp_tree->EndModel(0.25, false, true);

        m_modelLoaded = true;
    }
    else
    {
        BaseMeshTopology* meshTopology = this->getMeshTopology();
        sofa::core::behavior::MechanicalState<DataTypes>* mechanicalState = this->getMechanicalState();

        if (meshTopology->getNbQuads() > 0) { serr << meshTopology->getNbQuads() << " Quads in Mesh Topology ignored!" << sendl; };
        if (meshTopology->getNbHexas() > 0) { serr << meshTopology->getNbHexas() << " Hexas in Mesh Topology ignored!" << sendl; };
        if (meshTopology->getNbTetras() > 0) { serr << meshTopology->getNbTetras() << " Tetras in Mesh Topology ignored!" << sendl; };
        if (meshTopology->getNbHexahedra() > 0) { serr << meshTopology->getNbHexahedra() << " Hexahedra in Mesh Topology ignored!" << sendl; };
        if (meshTopology->getNbTetrahedra() > 0 ) { serr << meshTopology->getNbTetrahedra() << " Tetrahedra in Mesh Topology ignored!" << sendl; };

        const typename DataTypes::VecCoord& x = mechanicalState->read(core::ConstVecCoordId::position())->getValue();


        if (x.size() > 0 && meshTopology->getNbTriangles() > 0)
        {
            m_pqp_tree = new PQP_Model();
            m_pqp_tree->BeginModel(meshTopology->getNbTriangles());

            PQP_REAL v_1[3], v_2[3], v_3[3];
            for (int k = 0; k < meshTopology->getNbTriangles(); k++)
            {
                BaseMeshTopology::Triangle meshTri = meshTopology->getTriangle(k);

                Vector3 v1 = x[meshTri[0]];
                Vector3 v2 = x[meshTri[1]];
                Vector3 v3 = x[meshTri[2]];

                v1 -= m_initialPosition;
                v1 = m_initialOrientation.inverseRotate(v1);

                v2 -= m_initialPosition;
                v2 = m_initialOrientation.inverseRotate(v2);

                v3 -= m_initialPosition;
                v3 = m_initialOrientation.inverseRotate(v3);

                v_1[0] = v1.x(); v_1[1] = v1.y(); v_1[2] = v1.z();
                v_2[0] = v2.x(); v_2[1] = v2.y(); v_2[2] = v2.z();
                v_3[0] = v3.x(); v_3[1] = v3.y(); v_3[2] = v3.z();

                m_pqp_tree->AddTri(v_1, v_2, v_3, k);
            }

            m_pqp_tree->EndModel(0.25, false, true);

#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_COMPUTEOBBHIERARCHY
            std::cout << "==== PQP BVH tree ====" << std::endl;
            std::cout << " * Number of BV's     : " << m_pqp_tree->num_bvs << std::endl;
            std::cout << " * Number of triangles: " << m_pqp_tree->num_tris << std::endl;
            std::cout << " * Top BV position    : (" << m_pqp_tree->b->To[0] << "," << m_pqp_tree->b->To[1] << "," << m_pqp_tree->b->To[2] << ")"
                      << ", extents: (" << m_pqp_tree->b->d[0] << "," << m_pqp_tree->b->d[1] << "," << m_pqp_tree->b->d[2] << "), triangle range: " << m_pqp_tree->b->child_range_min << " - " << m_pqp_tree->b->child_range_max << std::endl;
            for (unsigned long p = 0; p < m_pqp_tree->num_bvs; p++)
            {
                std::cout << "    * BV " << p << " position: (" << m_pqp_tree->child(p)->To[0] << "," << m_pqp_tree->child(p)->To[1] << "," << m_pqp_tree->child(p)->To[2] << ")"
                                              << ", extents: (" << m_pqp_tree->child(p)->d[0] << "," << m_pqp_tree->child(p)->d[1] << "," << m_pqp_tree->child(p)->d[2] << ")"
                                              << ", triangle range: " << m_pqp_tree->child(p)->child_range_min << " - " << m_pqp_tree->child(p)->child_range_max
                                              << std::endl;
            }
#endif
            m_modelLoaded = true;
        }
    }
}

template <class DataTypes>
void PQPCollisionModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (_mObject)
    {
        glPushMatrix();
        typename DataTypes::VecCoord objTransform = _mObject->getPosition();
        Vector3 position(objTransform[0][0], objTransform[0][1], objTransform[0][2]);
        glTranslated(position.x(), position.y(), position.z());

        BVHDrawHelpers::drawCoordinateMarkerGL(6.0, 2.0);
        glPopMatrix();
    }

    if (m_cubeModel)
        m_cubeModel->draw(vparams);

    if ( (m_drawOBBHierarchy.getValue())
         && m_modelLoaded && m_pqp_tree)
    {
        Vector3 newTr;// Quaternion newRot;
        Matrix3 newOrientation;
        if (true)
        {
            getCachedPositionAndOrientation();

            newTr = m_modelTranslation;
            newOrientation = m_modelOrientation;
            //std::cout << " model pos. = " << newTr << ", model rot. = " << newRot << std::endl;
            //sout << "draw() 1 m_d->m_modelTranslation" <<  m_d->m_modelTranslation << sendl;
            //sout << "draw() 1 m_cachedPosition" << m_cachedPosition << sendl;


            Matrix4 modelGlOrientation; modelGlOrientation.identity();
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    modelGlOrientation[k][l] = newOrientation[k][l];
                }
            }

            glPushMatrix();
            glPushAttrib(GL_ENABLE_BIT);
            glEnable(GL_COLOR_MATERIAL);

            Vec4f colour(1,0,0,0.5);
            Vec4f colour2(0,0,1,0.5);
#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW
            glBegin(GL_LINES);
            glColor4d(colour2.x(), colour2.y(), colour2.z(), colour2.w());
            glVertex3d(0, 0, 0);
            glColor4d(colour.x(), colour.y(), colour.z(), colour.w());
            glVertex3d(newTr.x(), newTr.y(), newTr.z());
            glEnd();
#endif //OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW

            glTranslated(newTr.x(), newTr.y(), newTr.z());

            BVHDrawHelpers::drawCoordinateMarkerGL(0.5f, 1.0f, colour, colour * 0.5, colour * 0.25);

            //std::cout << " rotate to obj. orientation = " << newOrientation.transposed() << std::endl;
            glMultMatrixd(modelGlOrientation.transposed().ptr());

            glBegin(GL_LINES);
            glColor4d(0, 1, 0, 0.5);
            glVertex3d(0, 0, 0);
            glColor4d(colour.x(), colour.y(), colour.z(), colour.w());
            glVertex3d(m_pqp_tree->b->To[0], m_pqp_tree->b->To[1], m_pqp_tree->b->To[2]);
            glEnd();

            //std::cout << " translate to OBB center = " << m_pqp_tree->b->To[0] << "," << m_pqp_tree->b->To[1] << "," << m_pqp_tree->b->To[2] << std::endl;
            glTranslated(m_pqp_tree->b->To[0], m_pqp_tree->b->To[1], m_pqp_tree->b->To[2]);

            Matrix3 obbRotation; obbRotation.identity();
            obbRotation[0] = Vector3(m_pqp_tree->b->R[0][0], m_pqp_tree->b->R[1][0], m_pqp_tree->b->R[2][0]);
            obbRotation[1] = Vector3(m_pqp_tree->b->R[0][1], m_pqp_tree->b->R[1][1], m_pqp_tree->b->R[2][1]);
            obbRotation[2] = Vector3(m_pqp_tree->b->R[0][2], m_pqp_tree->b->R[1][2], m_pqp_tree->b->R[2][2]);

            Matrix4 glOrientation; glOrientation.identity();
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    glOrientation[k][l] = obbRotation[k][l];
                }
            }

            //drawObbVolume(sofa::defaulttype::Vector3(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]), colour2);

            //std::cout << " rotate to OBB orientation = " << glOrientation << std::endl;
            BVHDrawHelpers::drawCoordinateMarkerGL(0.75f, 4.0f, colour2, colour2, colour);

            glMultMatrixd(glOrientation.ptr());

            BVHDrawHelpers::drawCoordinateMarkerGL(1.0f, 6.0f, colour, colour2, colour2);

            BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]), colour);

            float extent_x = m_pqp_tree->b->d[0]; float extent_y = m_pqp_tree->b->d[1]; float extent_z = m_pqp_tree->b->d[2];
            if (m_pqp_tree->b->min_dimension == 0)
                extent_x = m_pqp_tree->b->min_dimension_val;
            else if (m_pqp_tree->b->min_dimension == 1)
                extent_y = m_pqp_tree->b->min_dimension_val;
            else if (m_pqp_tree->b->min_dimension == 2)
                extent_z = m_pqp_tree->b->min_dimension_val;

            BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(extent_x, extent_y, extent_z), Vec4f(0,1,0,1), true);

            glMultMatrixd(glOrientation.transposed().ptr());

            glTranslated(-m_pqp_tree->b->To[0], -m_pqp_tree->b->To[1], -m_pqp_tree->b->To[2]);

            if (m_pqp_tree->b->first_child > 0)
            {
                BV* child1 = m_pqp_tree->child(m_pqp_tree->b->first_child);

                Matrix3 childRotation; childRotation.identity();
                childRotation[0] = Vector3(child1->R[0][0], child1->R[1][0], child1->R[2][0]);
                childRotation[1] = Vector3(child1->R[0][1], child1->R[1][1], child1->R[2][1]);
                childRotation[2] = Vector3(child1->R[0][2], child1->R[1][2], child1->R[2][2]);

                Matrix4 glOrientation; glOrientation.identity();
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        glOrientation[k][l] = childRotation[k][l];
                    }
                }

                glTranslated(child1->To[0], child1->To[1], child1->To[2]);

                glMultMatrixd(glOrientation.ptr());
                BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(child1->d[0], child1->d[1], child1->d[2]), Vec4f(1, 1, 1, 0.5), true);

                glMultMatrixd(glOrientation.transposed().ptr());

                glTranslated(-child1->To[0], -child1->To[1], -child1->To[2]);
            }

            if (m_pqp_tree->b->first_child + 1 > 0)
            {
                BV* child2 = m_pqp_tree->child(m_pqp_tree->b->first_child + 1);

                Matrix3 childRotation; childRotation.identity();
                childRotation[0] = Vector3(child2->R[0][0], child2->R[1][0], child2->R[2][0]);
                childRotation[1] = Vector3(child2->R[0][1], child2->R[1][1], child2->R[2][1]);
                childRotation[2] = Vector3(child2->R[0][2], child2->R[1][2], child2->R[2][2]);

                Matrix4 glOrientation; glOrientation.identity();
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        glOrientation[k][l] = childRotation[k][l];
                    }
                }

                glTranslated(child2->To[0], child2->To[1], child2->To[2]);

                glMultMatrixd(glOrientation.ptr());

                BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(child2->d[0], child2->d[1], child2->d[2]), Vec4f(1, 1, 1, 1), true);
                glMultMatrixd(glOrientation.transposed().ptr());

                glTranslated(-child2->To[0], -child2->To[1], -child2->To[2]);
            }

            if (m_drawOBBHierarchy.getValue() > 1)
            {
                if (m_pqp_tree->num_bvs > 2)
                {
                    BVHDrawHelpers::drawRec(m_pqp_tree, m_pqp_tree->child(m_pqp_tree->b->first_child), vparams, colour2, 1, false);
                    BVHDrawHelpers::drawRec(m_pqp_tree, m_pqp_tree->child(m_pqp_tree->b->first_child+1), vparams, colour2, 1, false);
                }
            }

            glPopAttrib();
            glPopMatrix();
        }
    }

#ifdef RSS_EXPERIMENTAL

    if (vparams->displayFlags().getRssTreeHierarchies()
         && m_modelLoaded && m_pqp_tree)
    {
        Vector3 newTr; Quaternion newRot;
        if (getOrientation(newRot) && getPosition(newTr))
        {
            //std::cout << " model pos. = " << newTr << ", model rot. = " << newRot << std::endl;

            Matrix3 newOrientation;
            newRot.toMatrix(newOrientation);

            Matrix4 modelGlOrientation; modelGlOrientation.identity();
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    modelGlOrientation[k][l] = newOrientation[k][l];
                }
            }

            glPushMatrix();
            glPushAttrib(GL_ENABLE_BIT);
            glEnable(GL_COLOR_MATERIAL);

            Vec4f sphereColour(0.7,0,0,1);

            Vec4f colour(1,0,0,1);
            Vec4f colour2(0,0,1,1);
#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW
            glBegin(GL_LINES);
            glColor4d(colour2.x(), colour2.y(), colour2.z(), colour2.w());
            glVertex3d(0, 0, 0);
            glColor4d(colour.x(), colour.y(), colour.z(), colour.w());
            glVertex3d(newTr.x(), newTr.y(), newTr.z());
            glEnd();
#endif //OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW

            //std::cout << " translate to obj. coord. = " << newTr << std::endl;
            glTranslated(newTr.x(), newTr.y(), newTr.z());

            BVHDrawHelpers::drawCoordinateMarkerGL(0.5f, 4.0f, colour, colour * 0.5, colour * 0.25);

            //std::cout << " rotate to obj. orientation = " << newOrientation.transposed() << std::endl;
            glMultMatrixd(modelGlOrientation.transposed().ptr());

            glBegin(GL_LINES);
            glColor4d(0, 1, 0, 0.5);
            glVertex3d(0, 0, 0);
            glColor4d(colour.x(), colour.y(), colour.z(), colour.w());
            glVertex3d(m_pqp_tree->b->Tr[0], m_pqp_tree->b->Tr[1], m_pqp_tree->b->Tr[2]);
            glEnd();

            //std::cout << " translate to OBB center = " << m_pqp_tree->b->Tr[0] << "," << m_pqp_tree->b->Tr[1] << "," << m_pqp_tree->b->Tr[2] << std::endl;
            glTranslated(m_pqp_tree->b->Tr[0], m_pqp_tree->b->Tr[1], m_pqp_tree->b->Tr[2]);

            Matrix3 obbRotation; obbRotation.identity();
            obbRotation[0] = Vector3(m_pqp_tree->b->R[0][0], m_pqp_tree->b->R[1][0], m_pqp_tree->b->R[2][0]);
            obbRotation[1] = Vector3(m_pqp_tree->b->R[0][1], m_pqp_tree->b->R[1][1], m_pqp_tree->b->R[2][1]);
            obbRotation[2] = Vector3(m_pqp_tree->b->R[0][2], m_pqp_tree->b->R[1][2], m_pqp_tree->b->R[2][2]);

            Matrix4 glOrientation; glOrientation.identity();
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    glOrientation[k][l] = obbRotation[k][l];
                }
            }

            //drawObbVolume(sofa::defaulttype::Vector3(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]), colour2);

            //std::cout << " rotate to OBB orientation = " << glOrientation << std::endl;
            BVHDrawHelpers::drawCoordinateMarkerGL(0.75f, 4.0f, colour2, colour2, colour);

            glMultMatrixd(glOrientation.ptr());

            BVHDrawHelpers::drawCoordinateMarkerGL(1.0f, 6.0f, colour, colour2, colour2);

            //banane
            //ASD_BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]), colour);
            BVHDrawHelpers::drawRssVolume(sofa::defaulttype::Vector2(m_pqp_tree->b->l[0], m_pqp_tree->b->l[1]), (SReal)(m_pqp_tree->b->r), colour, sphereColour, true);

            float extent_x = m_pqp_tree->b->d[0]; float extent_y = m_pqp_tree->b->d[1]; float extent_z = m_pqp_tree->b->d[2];
            if (m_pqp_tree->b->min_dimension == 0)
                extent_x = m_pqp_tree->b->min_dimension_val;
            else if (m_pqp_tree->b->min_dimension == 1)
                extent_y = m_pqp_tree->b->min_dimension_val;
            else if (m_pqp_tree->b->min_dimension == 2)
                extent_z = m_pqp_tree->b->min_dimension_val;

            //banane
            //ASD_BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(extent_x, extent_y, extent_z), Vec4f(0,1,0,1), true);
            BVHDrawHelpers::drawRssVolume(sofa::defaulttype::Vector2(m_pqp_tree->b->l[0], m_pqp_tree->b->l[1]), (SReal)(m_pqp_tree->b->r), Vec4f(1.0, 0.0, 0.0, 1), Vec4f(1.0, 0.2, 0.2, 1), true);

            glMultMatrixd(glOrientation.transposed().ptr());

            glTranslated(-m_pqp_tree->b->Tr[0], -m_pqp_tree->b->Tr[1], -m_pqp_tree->b->Tr[2]);

            if (m_pqp_tree->b->first_child > 0)
            {
                BV* child1 = m_pqp_tree->child(m_pqp_tree->b->first_child);

                Matrix3 childRotation; childRotation.identity();
                childRotation[0] = Vector3(child1->R[0][0], child1->R[1][0], child1->R[2][0]);
                childRotation[1] = Vector3(child1->R[0][1], child1->R[1][1], child1->R[2][1]);
                childRotation[2] = Vector3(child1->R[0][2], child1->R[1][2], child1->R[2][2]);

                Matrix4 glOrientation; glOrientation.identity();
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        glOrientation[k][l] = childRotation[k][l];
                    }
                }

                glTranslated(child1->Tr[0], child1->Tr[1], child1->Tr[2]);

                glMultMatrixd(glOrientation.ptr());
                //banane
                //ASD_BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(child1->d[0], child1->d[1], child1->d[2]), Vec4f(1, 1, 1, 1), true);
                BVHDrawHelpers::drawRssVolume(sofa::defaulttype::Vector2(child1->l[0], child1->l[1]), (SReal)(child1->r), Vec4f(0.0, 1.0, 0.0, 1), Vec4f(0.2, 1.0, 0.2, 1), true);
                glMultMatrixd(glOrientation.transposed().ptr());

                glTranslated(-child1->Tr[0], -child1->Tr[1], -child1->Tr[2]);
            }

            if (m_pqp_tree->b->first_child + 1 > 0)
            {
                BV* child2 = m_pqp_tree->child(m_pqp_tree->b->first_child + 1);

                Matrix3 childRotation; childRotation.identity();
                childRotation[0] = Vector3(child2->R[0][0], child2->R[1][0], child2->R[2][0]);
                childRotation[1] = Vector3(child2->R[0][1], child2->R[1][1], child2->R[2][1]);
                childRotation[2] = Vector3(child2->R[0][2], child2->R[1][2], child2->R[2][2]);

                Matrix4 glOrientation; glOrientation.identity();
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        glOrientation[k][l] = childRotation[k][l];
                    }
                }

                glTranslated(child2->Tr[0], child2->Tr[1], child2->Tr[2]);

                glMultMatrixd(glOrientation.ptr());
                BVHDrawHelpers::drawRssVolume(sofa::defaulttype::Vector2(child2->l[0], child2->l[1]), (SReal)(child2->r), Vec4f(0.0, 0.0, 1.0, 1), Vec4f(0.2, 0.2, 1.0, 1), true);
                glMultMatrixd(glOrientation.transposed().ptr());

                glTranslated(-child2->Tr[0], -child2->Tr[1], -child2->Tr[2]);
            }

            if (m_drawOBBHierarchy.getValue() > 1)
            {
                if (m_pqp_tree->num_bvs > 2)
                {
                    BVHDrawHelpers::drawRec(m_pqp_tree, m_pqp_tree->child(m_pqp_tree->b->first_child), vparams, colour2, 1, false);
                    BVHDrawHelpers::drawRec(m_pqp_tree, m_pqp_tree->child(m_pqp_tree->b->first_child+1), vparams, colour2, 1, false);
                }
            }

            glPopAttrib();
            glPopMatrix();
        }
    }
#endif
}

template <class DataTypes>
void PQPCollisionModel<DataTypes>::getCachedPositionAndOrientation()
{
    double timestep = this->getContext()->getTime();
    if (timestep != m_lastTimestep)
    {
        m_lastTimestep = timestep;
        // POSITION

        sofa::simulation::Node * node1 = (sofa::simulation::Node*)this->getContext();

        sofa::simulation::Node* parent1 = (sofa::simulation::Node*)(node1->getParents()[0]);
        std::vector<sofa::component::container::MechanicalObject<Rigid3Types> *> mobj_1;
        sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::container::MechanicalObject<Rigid3Types>, std::vector<sofa::component::container::MechanicalObject<Rigid3Types>* > > mobj_cb_1(&mobj_1);
        parent1->getObjects(TClassInfo<sofa::component::container::MechanicalObject<Rigid3Types> >::get(), mobj_cb_1, TagSet(), BaseContext::SearchDown);

        const Rigid3Types::VecCoord c = mobj_1.at(0)->getPosition();
        Vec3d currentPos_From(c[0][0], c[0][1], c[0][2]);
        m_modelTranslation = currentPos_From;

        // ORIENTATION
        Quat quat_From(c[0][3], c[0][4], c[0][5], c[0][6]);
        quat_From.toMatrix(m_modelOrientation);
    }
}

#define PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
template <class DataTypes>
void PQPCollisionModel<DataTypes>::computeBoundingTree(int maxDepth)
{
#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
    std::cout << "=== PQPCollisionModel::computeBoundingTree(" << this->getName() << "," << maxDepth << ") ===" << std::endl;
#endif
    if (!m_cubeModel)
        m_cubeModel = createPrevious<CubeModel>();

    if (!isMoving() && !m_cubeModel->empty())
        return; // No need to recompute BBox if immobile

    Vector3 modelPosition = getCachedPosition();
    Quaternion modelOrientation = getCachedOrientation();

    m_cubeModel->resize(0);

    sofa::defaulttype::Vector3 treeCenter(m_pqp_tree->b->To[0], m_pqp_tree->b->To[1], m_pqp_tree->b->To[2]);
    sofa::defaulttype::Vector3 he(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]);
    sofa::defaulttype::Vector3 heTransform = modelPosition + modelOrientation.rotate(treeCenter);

    sofa::defaulttype::Matrix3 treeOrientation;
    for (short k = 0; k < 3; k++)
        for (short l = 0; l < 3; l++)
            treeOrientation[k][l] = m_pqp_tree->b->R[k][l];

    if (maxDepth == 0)
    {
        if (this->empty())
        {
            m_cubeModel->resize(0);
        }
        else
        {
            m_cubeModel->resize(1);
            sofa::defaulttype::Vector3 treeMin, treeMax;

            Vector3 tc0 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(he.x(), he.y(), he.z()));
            Vector3 tc1 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(he.x(), he.y(), -he.z()));
            Vector3 tc2 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(he.x(), -he.y(), he.z()));
            Vector3 tc3 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(he.x(), -he.y(), -he.z()));
            Vector3 tc4 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(-he.x(), he.y(), he.z()));
            Vector3 tc5 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(-he.x(), he.y(), -he.z()));
            Vector3 tc6 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(-he.x(), -he.y(), he.z()));
            Vector3 tc7 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(-he.x(), -he.y(), -he.z()));
            for (int c = 0; c < 3; c++)
            {
                treeMin[c] = tc0[c];
                treeMax[c] = tc0[c];

                if (tc1[c] > treeMax[c])
                    treeMax[c] = tc1[c];
                else if (tc1[c] < treeMin[c])
                    treeMin[c] = tc1[c];

                if (tc2[c] > treeMax[c])
                    treeMax[c] = tc2[c];
                else if (tc2[c] < treeMin[c])
                    treeMin[c] = tc2[c];

                if (tc3[c] > treeMax[c])
                    treeMax[c] = tc3[c];
                else if (tc3[c] < treeMin[c])
                    treeMin[c] = tc3[c];

                if (tc4[c] > treeMax[c])
                    treeMax[c] = tc4[c];
                else if (tc4[c] < treeMin[c])
                    treeMin[c] = tc4[c];

                if (tc5[c] > treeMax[c])
                    treeMax[c] = tc5[c];
                else if (tc5[c] < treeMin[c])
                    treeMin[c] = tc5[c];

                if (tc6[c] > treeMax[c])
                    treeMax[c] = tc6[c];
                else if (tc6[c] < treeMin[c])
                    treeMin[c] = tc6[c];

                if (tc7[c] > treeMax[c])
                    treeMax[c] = tc7[c];
                else if (tc7[c] < treeMin[c])
                    treeMin[c] = tc7[c];
            }
#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
        std::cout << " Tree level min/max: " << treeMin << " / " << treeMax << std::endl;
#endif
            m_cubeModel->setLeafCube(0, std::make_pair(this->begin(),this->end()), treeMin, treeMax);
        }
    }
    else
    {
#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
        std::cout << "RESIZE m_cubeModel to " << m_pqp_tree->num_bvs << std::endl;
#endif
        m_cubeModel->resize(1 /*m_pqp_tree->num_bvs*/);
        Vector3 treeMin, treeMax;

        Vector3 tc0 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(he.x(), he.y(), he.z()));
        Vector3 tc1 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(he.x(), he.y(), -he.z()));
        Vector3 tc2 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(he.x(), -he.y(), he.z()));
        Vector3 tc3 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(he.x(), -he.y(), -he.z()));
        Vector3 tc4 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(-he.x(), he.y(), he.z()));
        Vector3 tc5 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(-he.x(), he.y(), -he.z()));
        Vector3 tc6 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(-he.x(), -he.y(), he.z()));
        Vector3 tc7 = heTransform + modelOrientation.rotate(treeOrientation * Vector3(-he.x(), -he.y(), -he.z()));

        for (int c = 0; c < 3; c++)
        {
            treeMin[c] = tc0[c];
            treeMax[c] = tc0[c];

            if (tc1[c] > treeMax[c])
                treeMax[c] = tc1[c];
            else if (tc1[c] < treeMin[c])
                treeMin[c] = tc1[c];

            if (tc2[c] > treeMax[c])
                treeMax[c] = tc2[c];
            else if (tc2[c] < treeMin[c])
                treeMin[c] = tc2[c];

            if (tc3[c] > treeMax[c])
                treeMax[c] = tc3[c];
            else if (tc3[c] < treeMin[c])
                treeMin[c] = tc3[c];

            if (tc4[c] > treeMax[c])
                treeMax[c] = tc4[c];
            else if (tc4[c] < treeMin[c])
                treeMin[c] = tc4[c];

            if (tc5[c] > treeMax[c])
                treeMax[c] = tc5[c];
            else if (tc5[c] < treeMin[c])
                treeMin[c] = tc5[c];

            if (tc6[c] > treeMax[c])
                treeMax[c] = tc6[c];
            else if (tc6[c] < treeMin[c])
                treeMin[c] = tc6[c];

            if (tc7[c] > treeMax[c])
                treeMax[c] = tc7[c];
            else if (tc7[c] < treeMin[c])
                treeMin[c] = tc7[c];
        }

#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
        std::cout << " Tree level min/max: " << treeMin << " / " << treeMax << std::endl;
        std::cout << " total OBB count = " << m_pqp_tree->num_bvs << std::endl;
#endif
        m_cubeModel->setParentOf(0, treeMin, treeMax);

#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
        std::cout << " tree node 1st child = " << m_pqp_tree->b->first_child << ", 2nd child = " << m_pqp_tree->b->first_child + 1;
#endif

#if 0
        if (m_pqp_tree->num_bvs > 1)
        {
            for (unsigned int i = 0; i <= 1; i++)
            {
                //if (m_pqp_tree->b->first_child <= maxDepth)
                if (m_pqp_tree->b->first_child + i >= 0)
                {
                    BV* childNode = m_pqp_tree->child(m_pqp_tree->b->first_child + i);

#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
                    std::cout << "  accessed child node " << i << " = " << m_pqp_tree->b->first_child + i << std::endl;
#endif

                    Vector3 childTranslation(childNode->To[0], childNode->To[1], childNode->To[2]);
                    sofa::defaulttype::Matrix3 childOrientation;
                    for (short k = 0; k < 3; k++)
                    for (short l = 0; l < 3; l++)
                        childOrientation[k][l] = childNode->R[k][l];

                    Vector3 childHeTransform = modelPosition + modelOrientation.rotate(treeCenter) +
                        (childOrientation * childTranslation);
#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
                    std::cout << "  node " << i << ": computeBoundingTreeRec(" << (m_pqp_tree->b->first_child + i) - 1 << ")" << std::endl;
#endif
                    computeBoundingTreeRec(m_pqp_tree, childNode, treeOrientation, childHeTransform, (m_pqp_tree->b->first_child + i), 0, 1 /*maxDepth*/);
                }
            }
        }
#endif
        m_cubeModel->computeBoundingTree(maxDepth);
    }

#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
    std::cout << "=== CubeModel new size: " << m_cubeModel->getNumberCells() << " ===" << std::endl;
    for (unsigned int i = 0; i < m_cubeModel->getNumberCells(); i++)
    {
        std::cout << "  * cell " << i << ": ";
        std::pair<core::CollisionElementIterator,core::CollisionElementIterator> ic = m_cubeModel->getInternalChildren(i);
        //if (ic.first.valid())
            std::cout << " child 1 = " << ic.first.getIndex() << ", ";

        //if (ic.second.valid())
            std::cout << " child 2 = " << ic.second.getIndex();

        std::cout << std::endl;
    }
#endif
}

template <class DataTypes>
void PQPCollisionModel<DataTypes>::computeBoundingTreeRec(PQP_Model* tree, BV* obb, Matrix3 treeOrientation, Vector3& accumulatedChildOffsets, int boxIndex, int currentDepth, int maxDepth)
{
#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
    std::cout << "  computeBoundingTreeRec(" << obb << "), boxIndex = " << boxIndex << std::endl;
#endif
    if (currentDepth > maxDepth)
    {
#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
        std::cout << "  computeBoundingTreeRec(" << obb << "), currentDepth = " << currentDepth << ", maxDepth = " << maxDepth << ", return" << std::endl;
#endif
        return;
    }

    Vector3 childHe(obb->d[0], obb->d[1], obb->d[2]);
    Vector3 cMin, cMax;

    Vector3 childTranslation(obb->To[0], obb->To[1], obb->To[2]);
    sofa::defaulttype::Matrix3 childOrientation;
    for (short k = 0; k < 3; k++)
        for (short l = 0; l < 3; l++)
            childOrientation[k][l] = obb->R[k][l];


    Vector3 tc0 = accumulatedChildOffsets + treeOrientation * (childOrientation * Vector3(childHe.x(), childHe.y(), childHe.z()));
    Vector3 tc1 = accumulatedChildOffsets + treeOrientation * (childOrientation * Vector3(childHe.x(), childHe.y(), -childHe.z()));
    Vector3 tc2 = accumulatedChildOffsets + treeOrientation * (childOrientation * Vector3(childHe.x(), -childHe.y(), childHe.z()));
    Vector3 tc3 = accumulatedChildOffsets + treeOrientation * (childOrientation * Vector3(childHe.x(), -childHe.y(), -childHe.z()));
    Vector3 tc4 = accumulatedChildOffsets + treeOrientation * (childOrientation * Vector3(-childHe.x(), childHe.y(), childHe.z()));
    Vector3 tc5 = accumulatedChildOffsets + treeOrientation * (childOrientation * Vector3(-childHe.x(), childHe.y(), -childHe.z()));
    Vector3 tc6 = accumulatedChildOffsets + treeOrientation * (childOrientation * Vector3(-childHe.x(), -childHe.y(), childHe.z()));
    Vector3 tc7 = accumulatedChildOffsets + treeOrientation * (childOrientation * Vector3(-childHe.x(), -childHe.y(), -childHe.z()));

    for (int c = 0; c < 3; c++)
    {
        cMin[c] = tc0[c];
        cMax[c] = tc0[c];

        if (tc1[c] > cMax[c])
            cMax[c] = tc1[c];
        else if (tc1[c] < cMin[c])
            cMin[c] = tc1[c];

        if (tc2[c] > cMax[c])
            cMax[c] = tc2[c];
        else if (tc2[c] < cMin[c])
            cMin[c] = tc2[c];

        if (tc3[c] > cMax[c])
            cMax[c] = tc3[c];
        else if (tc3[c] < cMin[c])
            cMin[c] = tc3[c];

        if (tc4[c] > cMax[c])
            cMax[c] = tc4[c];
        else if (tc4[c] < cMin[c])
            cMin[c] = tc4[c];

        if (tc5[c] > cMax[c])
            cMax[c] = tc5[c];
        else if (tc5[c] < cMin[c])
            cMin[c] = tc5[c];

        if (tc6[c] > cMax[c])
            cMax[c] = tc6[c];
        else if (tc6[c] < cMin[c])
            cMin[c] = tc6[c];

        if (tc7[c] > cMax[c])
            cMax[c] = tc7[c];
        else if (tc7[c] < cMin[c])
            cMin[c] = tc7[c];
    }

#ifdef PQPCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
    std::cout << "  AABB " << boxIndex << ": min / max = " << cMin << " / " << cMax << std::endl;
#endif

    if (boxIndex < m_pqp_tree->num_bvs)
    {
        m_cubeModel->setParentOf(boxIndex, cMin, cMax);

        for (int i = 0; i < 2; i++)
        {
            if (currentDepth < maxDepth)
            {
                if (obb->first_child + i < m_pqp_tree->num_bvs)
                {
                    Vector3 childHeTransform = accumulatedChildOffsets + (treeOrientation * childTranslation);

                    boxIndex += 1;
                    computeBoundingTreeRec(tree, m_pqp_tree->child(obb->first_child + i), treeOrientation, childHeTransform, boxIndex, currentDepth + 1, maxDepth);
                }
            }
        }
    }
}

#endif // PQPMODEL_INL
