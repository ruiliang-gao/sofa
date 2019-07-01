#ifndef OBBTREEGPU_COLLISIONMODEL_INL
#define OBBTREEGPU_COLLISIONMODEL_INL

#include "ObbTreeGPUCollisionModel.h"

#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/ClassInfo.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaLoader/MeshObjLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/Node.h>

#ifdef __APPLE__
#include <GL/glew.h>
#include <GLUT/glut.h>
#else
#ifdef _WIN32
#include <gl/glew.h>
#else
#include <GL/glew.h>
#include <GL/glut.h>
#endif
#endif

#include "BVHDrawHelpers.h"
#include <sofa/core/visual/VisualParams.h>

#include "gProximity/transform.h"
#include "gProximity/geometry.h"
#include "gProximity/cuda_defs.h"
#include "gProximity/cuda_vertex.h"
#include "gProximity/cuda_vectors.h"
#include "gProximity/cpu_bvh_constru.h"
#include "gProximity/cuda_bvh_constru.h"

#include <PQP.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>

#include <cuda_runtime_api.h>
#include <cutil/cutil.h>
#include <cuda_gl_interop.h>

#ifdef _WIN32
#include <gl/glut.h>
#else
#include <GL/glut.h>
#endif
#include "ObbTreeGPUCollisionModel_cuda.h"

#include "ObbTreeGPUIntersection.h"

#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>

using sofa::component::collision::ObbTreeGPULocalMinDistance;

using sofa::helper::system::DataRepository;

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            class ObbTreeGPUCollisionModelPrivate
            {
                public:
                    ObbTreeGPUCollisionModelPrivate(ModelInstance* gpModel): m_gpModel(gpModel)
                    {
                        m_transformedVertices = new GPUVertex[m_gpModel->nVerts];
                    }

                    ~ObbTreeGPUCollisionModelPrivate()
                    {
                        delete[] m_transformedVertices;
                    }

                    ModelInstance* m_gpModel;
                    GPUVertex* m_transformedVertices;

                    gProximityGPUTransform** m_modelTransform;

                    Matrix3 m_modelOrientation;
                    Vector3 m_modelTranslation;
            };
        }
    }
}

using namespace sofa::core::objectmodel;
using namespace sofa::component::collision;
using namespace sofa;


template <class DataTypes>
ObbTreeGPUCollisionModel<DataTypes>::ObbTreeGPUCollisionModel(): core::CollisionModel(),
                                                                 m_gpModel(NULL), m_pqp_tree(NULL),
                                                                 m_scale(1.0f), m_modelLoaded(false),
                                                                 m_objectMechanicalState(NULL), m_objectBaseMState(NULL), m_mechanicalObject(NULL),
                                                                 m_cubeModel(NULL), m_d(NULL),
                                                                 m_drawOBBHierarchy(initData(&m_drawOBBHierarchy, 0, "drawOBBHierarchy", "Draw the model's OBB hierarchy 0=off 1=box 2=full")),
                                                                 m_useVertexBuffer(initData(&m_useVertexBuffer, false, "useVertexBuffers", "Use vertex buffers to store mesh data in GPU device memory")), 
                                                                 m_updateVerticesInModel(initData(&m_updateVerticesInModel, true, "updateVerticesInModel", "Update GPU array of vertex coordinates from model")),
                                                                 m_edgeLabelScaleFactor(initData(&m_edgeLabelScaleFactor, 0.01, "edgeLabelScaleFactor", "Scale for edge labels", true, false)),
																 m_oldCachedPosition(), m_oldCachedOrientation(),
                                                                 m_cachedPositionChange(0), m_cachedOrientationChange(0),
                                                                 useContactManifolds(initData(&useContactManifolds, false, "useContactManifolds", "Create contact manifolds from detection output. Must be true in both collision models.")),
                                                                 maxNumberOfLineLineManifolds(initData(&maxNumberOfLineLineManifolds, 1u, "maxNumberOfLineLineManifolds", "Use instead of maxNumberOfManifolds. Maximum number of Line/Line contact manifolds that should be created. The lower number defined in both models is used. Cannot be smaller than 1.")),
                                                                 maxNumberOfFaceVertexManifolds(initData(&maxNumberOfFaceVertexManifolds, 1u, "maxNumberOfFaceVertexManifolds", "Use instead of maxNumberOfManifolds. Maximum number of Face/Vertex contact manifolds that should be created. The lower number defined in both models is used. Cannot be smaller than 1.")),
                                                                 m_uuid(boost::uuids::random_generator()()),
																 m_innerMapping(NULL), m_fromObject(NULL)
{
}

template <class DataTypes>
ObbTreeGPUCollisionModel<DataTypes>::~ObbTreeGPUCollisionModel()
{
    disposeHierarchy();
    if (m_gpModel)
    {
        delete m_gpModel;
        m_gpModel = NULL;
    }

    if (m_pqp_tree)
    {
        delete m_pqp_tree;
        m_pqp_tree = NULL;
    }

    if (m_d)
        delete m_d;
}

template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::init()
{
    using namespace sofa::component::container;

    m_lastTimestep = -1.0;

    m_objectMechanicalState = dynamic_cast< core::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());

    Vector3 scale(1,1,1);
    Real uniformScale = 1.0f;

    std::vector<ObbTreeGPULocalMinDistance* > lmdNodes;
    sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<ObbTreeGPULocalMinDistance, std::vector<ObbTreeGPULocalMinDistance* > > cb(&lmdNodes);
    getContext()->getObjects(TClassInfo<ObbTreeGPULocalMinDistance>::get(), cb, TagSet(), BaseContext::SearchRoot);
    if (lmdNodes.size() > 0)
    {
           m_alarmDistance = lmdNodes.at(0)->getAlarmDistance();
    }


    component::container::MechanicalObject<DataTypes>* mechanicalObject = dynamic_cast< component::container::MechanicalObject<DataTypes>* >(m_objectMechanicalState);
    if (mechanicalObject != NULL)
    {

		typedef MechanicalObject<RigidTypes> mType;
		 
		std::vector<mType*> mechObjects;
		sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<mType, std::vector<mType*> > cbMech(&mechObjects);

		getContext()->getObjects(TClassInfo<mType>::get(), cbMech, TagSet(), BaseContext::SearchParents);

		for (std::vector<mType*>::iterator it = mechObjects.begin(); it != mechObjects.end(); it++) {
			mType* current = (*it);
			Rigid3dTypes::VecCoord coords = current->getPosition();
            //std::cout << "         NEW POSITION VECTOR: " << coords << std::endl;
			m_initialPosition = Vec3d(coords[0][0], coords[0][1], coords[0][2]);
			m_initialOrientation = Quat(coords[0][3], coords[0][4], coords[0][5], coords[0][6]);
			m_scale = uniformScale; // TODO! m_scale should be a Vec3d!!!!!!        mscale = current->getScale();
			scale = current->getScale();
			if (scale.x() == scale.y() && scale.y() == scale.z()) {
				  m_scale = scale.x();
			}
			m_objectBaseMState = current;
		}

        //std::cout << "WHAT AM I: " << this->getTypeName() << ", class = " << this->getClassName() << std::endl;
        //std::cout << "WHAT IS MY PARENT, named " << this->getContext()->getName() << ": " << this->getContext()->getTypeName() << ", class = " << this->getContext()->getClassName() << std::endl;
		simulation::Node* parentNode = dynamic_cast<simulation::Node*>(this->getContext());

		if (parentNode)
		{
			core::objectmodel::BaseNode::Parents grandParents = parentNode->getParents();
            //std::cout << " PARENT NODE parents count = " << grandParents.size() << std::endl;
            /*for (int k = 0; k < grandParents.size(); k++)
			{
				std::cout << " * " << k << ": " << grandParents[k]->getName() << std::endl;
            }*/
			if (grandParents.size() == 1)
			{
				simulation::Node* grandParentNode = dynamic_cast<simulation::Node*>(grandParents[0]);

				std::vector<sofa::component::container::MechanicalObject<DataTypes>* > mo_vec;
				sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::container::MechanicalObject<DataTypes>, std::vector<sofa::component::container::MechanicalObject<DataTypes>* > > mo_cb(&mo_vec);

				grandParentNode->getObjects(TClassInfo<sofa::component::container::MechanicalObject<DataTypes> >::get(), mo_cb, TagSet(), BaseContext::SearchDown);

                //std::cout << "MECHANICAL OBJECT SEARCH 1 YIELDS: " << mo_vec.size() << " MODELS!!!" << std::endl;

				if (mo_vec.size() > 0)
					m_mechanicalObject = mo_vec.at(0);
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

                    bool fileFound = DataRepository.findFile(m_modelFile);
                    std::string absFilePath;
                    if (fileFound)
                    {
                        std::cout << "     File found." << std::endl;
                        absFilePath = DataRepository.getFile(m_modelFile);
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
void ObbTreeGPUCollisionModel<DataTypes>::cleanup()
{

}

template <class DataTypes>
gProximityContactType ObbTreeGPUCollisionModel<DataTypes>::getContactType(const boost::uuids::uuid& model_uuid, const int contact_index)
{
    return COLLISION_INVALID;
}

template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::setContactType(const boost::uuids::uuid& model_uuid, const int contact_index, const gProximityContactType contact_type)
{

}

template <class DataTypes>
std::pair<int,int> ObbTreeGPUCollisionModel<DataTypes>::getContactFeatures(const boost::uuids::uuid& model_uuid, const int contact_index)
{
    return std::pair<int, int>(-1, -1);
}

template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::setContactFeatures(const boost::uuids::uuid& model_uuid, const int contact_index, const int feature_1, const int feature_2)
{

}

// Please call with care: This requires void* parameter to avoid CUDA compilation issues, so only pass in suitably cast'ed pointer!

template <class DataTypes>
void* ObbTreeGPUCollisionModel<DataTypes>::getGPUModelTransform()
{
    return *m_modelTransform;
}

template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::setGPUModelTransform(void** tr_ptr)
{
    this->m_modelTransform = /*(gProximityGPUTransform**)*/ tr_ptr;

    Matrix3x3_d h_modelTransform;
    float3 h_trVector;

    std::cout << "ObbTreeGPUCollisionModel<DataTypes>::setGPUModelTransform(" << this->getName() << ")" << std::endl;

    gProximityGPUTransform* tmp = (gProximityGPUTransform*) *(this->m_modelTransform);

    FROMGPU(&h_modelTransform, tmp->modelOrientation, sizeof(Matrix3x3_d));
    FROMGPU(&h_trVector, tmp->modelTranslation, sizeof(float3));

    Matrix3 cachedOri;
    this->getCachedOrientation().toMatrix(cachedOri);
    std::cout << " model translation in host mem. = " << this->getCachedPosition() << std::endl;
    std::cout << " model orientation in host mem. = " << cachedOri << std::endl;
    std::cout << " model translation from GPU device = " << h_trVector.x << "," << h_trVector.y << "," << h_trVector.z << std::endl;
    std::cout << " model orientation from GPU device = " << h_modelTransform.m_row[0].x << "," << h_modelTransform.m_row[0].y << "," << h_modelTransform.m_row[0].z
              << "/" << h_modelTransform.m_row[1].x << "," << h_modelTransform.m_row[1].y << "," << h_modelTransform.m_row[1].z
              << "/" << h_modelTransform.m_row[2].x << "," << h_modelTransform.m_row[2].y << "," << h_modelTransform.m_row[2].z
              << std::endl;
}

template <class DataTypes>
void* ObbTreeGPUCollisionModel<DataTypes>::getTransformedVerticesPtr()
{
    return (void*) m_d->m_transformedVertices;
}

/*
template <class DataTypes>
const Vector3& ObbTreeGPUCollisionModel<DataTypes>::getCachedModelPosition()
{
    return m_d->m_modelTranslation;
}

template <class DataTypes>
const Matrix3& ObbTreeGPUCollisionModel<DataTypes>::getCachedModelOrientation()
{
    return m_d->m_modelOrientation;
}
*/

template <class DataTypes>
unsigned int ObbTreeGPUCollisionModel<DataTypes>::numTriangles() const
{
    return m_gpModel->nTris;
}

template <class DataTypes>
unsigned int ObbTreeGPUCollisionModel<DataTypes>::numVertices() const
{
    return m_gpModel->nVerts;
}

template <class DataTypes>
unsigned int ObbTreeGPUCollisionModel<DataTypes>::numOBBs() const
{
    return m_pqp_tree->num_bvs;
}

template <class DataTypes>
void* ObbTreeGPUCollisionModel<DataTypes>::obbTree_device()
{
    return m_gpModel->obbTree;
}

template <class DataTypes>
void* ObbTreeGPUCollisionModel<DataTypes>::vertexPointer_device()
{
    return m_gpModel->vertexPointer;
}

template <class DataTypes>
void* ObbTreeGPUCollisionModel<DataTypes>::vertexTfPointer_device()
{
    return m_gpModel->vertexTfPointer;
}

template <class DataTypes>
void* ObbTreeGPUCollisionModel<DataTypes>::triIndexPointer_device()
{
    return m_gpModel->triIdxPointer;
}

template <class DataTypes>
bool ObbTreeGPUCollisionModel<DataTypes>::getVertex(const int& idx, Vector3& outVertex)
{
    if (idx < m_gpModel->nVerts)
    {
        outVertex = Vector3(m_gpModel->verlist[idx].x(), m_gpModel->verlist[idx].y(), m_gpModel->verlist[idx].z());
        return true;
    }
    else
    {
        return false;
    }
}

template <class DataTypes>
bool ObbTreeGPUCollisionModel<DataTypes>::getTriangle(const int& idx, sofa::core::topology::Triangle& outTri)
{
    if (idx < m_gpModel->nTris)
    {
        outTri = sofa::core::topology::Triangle(m_gpModel->trilist[idx].p[0], m_gpModel->trilist[idx].p[1], m_gpModel->trilist[idx].p[2]);
        return true;
    }
    else
    {
        return false;
    }
}

template <class DataTypes>
bool ObbTreeGPUCollisionModel<DataTypes>::getPosition(Vector3& position) const
{
    //std::cout << "getPosition(" << this->getName() << "): ";
    position = m_cachedPosition;
    //position = getCachedPosition();
    return true;
}

template <class DataTypes>
bool ObbTreeGPUCollisionModel<DataTypes>::getOrientation(Matrix3& orientation) const
{
    m_cachedOrientation.toMatrix(orientation);
    //getCachedOrientation().toMatrix(orientation);
	//orientation = getCachedOrientationMatrix();
    return true;
}

template <class DataTypes>
bool ObbTreeGPUCollisionModel<DataTypes>::getOrientation(Quaternion& orientation) const
{
    orientation = m_cachedOrientation;
    //orientation = getCachedOrientation();
    return true;
}


template <class DataTypes>
defaulttype::Quaternion ObbTreeGPUCollisionModel<DataTypes>::getCachedOrientation()
{
    getCachedPositionAndOrientation();
    return m_cachedOrientation;
}

template <class DataTypes>
defaulttype::Matrix3 ObbTreeGPUCollisionModel<DataTypes>::getCachedOrientationMatrix()
{
    getCachedPositionAndOrientation();

    Matrix3 orientation;
    m_cachedOrientation.toMatrix(orientation);
    return orientation;
}

template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::setCachedPosition(const Vec3d &position)
{
    m_cachedPositionChange = (m_oldCachedPosition != position);
#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_UPDATE_INTERNAL_GEOMETRY
    if (!cachedPositionChange) { std::cout << "the position of " << this->getName() << "hasn't changed! old: " << m_oldCachedPosition << " new: " << position << " difference: " << (m_oldCachedPosition - position) << std::endl; }
#endif

	if (m_modelLoaded)
	{
		m_oldCachedPosition = m_cachedPosition;
		m_cachedPosition = position;
		m_d->m_modelTranslation = position;
	}
}

template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::setCachedOrientation(const Quaternion &orientation)
{
    m_cachedOrientationChange = (m_oldCachedOrientation != orientation);
#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_UPDATE_INTERNAL_GEOMETRY
    if (!cachedOrientationChange) { std::cout << "the orientation of " << this->getName() << "hasn't changed! old: " << m_oldCachedOrientation << " new: " << orientation << std::endl; }
#endif

	if (m_modelLoaded)
	{
		m_oldCachedOrientation = m_cachedOrientation;
		m_cachedOrientation = orientation;
		Matrix3 mat;
		orientation.toMatrix(mat);
		m_d->m_modelOrientation = mat;
	}
}

template <class DataTypes>
bool ObbTreeGPUCollisionModel<DataTypes>::hasModelPositionChanged() const
{
    // Bootstrap!
    if (this->getContext()->getTime() == 0.0) return true;

    return (m_cachedPositionChange || m_cachedOrientationChange) || (m_fromObject != NULL && m_innerMapping != NULL);
}

template <class DataTypes>
void  ObbTreeGPUCollisionModel<DataTypes>::getCachedPositionAndOrientation() {

    // ATTACHED OBJECTS
    if (m_fromObject != NULL && m_innerMapping != NULL)
    {
        // POSITION
        const Rigid3Types::VecCoord c = m_fromObject->getPosition();
        Vec3d currentPos_From(c[0][0], c[0][1], c[0][2]);
        Quat quat_From(c[0][3], c[0][4], c[0][5], c[0][6]);
        Vec3d newPos = currentPos_From + quat_From.rotate(this->m_appliedTranslation.getValue());

        setCachedPosition(newPos);

        // ORIENTATION
        defaulttype::Quat quat_to = this->m_appliedRotation.getValue();

        setCachedOrientation(quat_From*quat_to);


        sout << "getCachedP&O() (LINKED)" << m_cachedPosition << sendl;
    }
    else {

        double timestep = this->getContext()->getTime();
        if (timestep != m_lastTimestep) {
            m_lastTimestep = timestep;
            // POSITION

            sofa::simulation::Node * node1 = (sofa::simulation::Node*)this->getContext();

            sofa::simulation::Node* parent1 = (sofa::simulation::Node*)(node1->getParents()[0]);
            std::vector<sofa::component::container::MechanicalObject<Rigid3Types> *> mobj_1;
            sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::container::MechanicalObject<Rigid3Types>, std::vector<sofa::component::container::MechanicalObject<Rigid3Types>* > > mobj_cb_1(&mobj_1);
            parent1->getObjects(TClassInfo<sofa::component::container::MechanicalObject<Rigid3Types> >::get(), mobj_cb_1, TagSet(), BaseContext::SearchDown);

            const Rigid3Types::VecCoord c = mobj_1.at(0)->getPosition();
            Vec3d currentPos_From(c[0][0], c[0][1], c[0][2]);

            setCachedPosition(currentPos_From);

            // ORIENTATION
            Quat quat_From(c[0][3], c[0][4], c[0][5], c[0][6]);

            setCachedOrientation(quat_From);

            sout << "getCachedP&O() (NORMAL)" << m_cachedPosition << sendl;
        }
    }
}

template <class DataTypes>
defaulttype::Vec3d ObbTreeGPUCollisionModel<DataTypes>::getCachedPosition()
{
    getCachedPositionAndOrientation();
    return m_cachedPosition;
#if 0
	const sofa::core::objectmodel::BaseContext* myContext = this->getContext();
	if (myContext)
	{
		std::cout << "ObbTreeGPUCollisionModel<DataTypes>::getCachedPosition(" << this->getName() << "): Got BaseContext " << myContext->getName() << " of type " << myContext->getTypeName() << std::endl;
		
		

		sofa::simulation::Node* myNode = (sofa::simulation::Node*)(myContext);
		sofa::simulation::Node* parentNode = (sofa::simulation::Node*)(myNode->getParents()[0]);
		std::cout << " parentNode = " << parentNode->getName() << std::endl;
		
		std::vector<sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>*> inner_mapping;
		sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>, std::vector<sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types>* > > inner_mapping_cb(&inner_mapping);
		parentNode->getObjects(TClassInfo<sofa::component::mapping::RigidMapping<Rigid3Types, Vec3Types> >::get(), inner_mapping_cb, TagSet(), BaseContext::SearchDown);

		std::cout << " RigidMapping instances = " << inner_mapping.size() << std::endl;
		if (inner_mapping.size() > 0)
		{
			std::cout << "  got " << inner_mapping.size() << " RigidMapping's" << std::endl;
			for (unsigned int k = 0; k < inner_mapping.size(); k++)
			{
				if (inner_mapping[k]->getFromModel() != NULL)
				{
					std::cout << "   - " << inner_mapping[k]->getName() << " from " << inner_mapping[k]->getFromModel()->getName() << " to " << inner_mapping[k]->getToModel()->getName() << std::endl;
					sofa::component::container::MechanicalObject<Rigid3Types>* fromModel = (sofa::component::container::MechanicalObject<Rigid3Types>*)(inner_mapping[k]->getFromModel());

					Vec3d currentPos_From(fromModel->getPosition()[0][0], fromModel->getPosition()[0][1], fromModel->getPosition()[0][2]);
					std::cout << "     current position of fromModel '" << fromModel->getName() << "' = " << currentPos_From << std::endl;
					
					if (boost::algorithm::ends_with(inner_mapping[k]->getName(), "_Gripping"))
					{
						std::cout << "      Warning: This needs special position offset handling: Add position offset = " << inner_mapping[k]->appliedTranslation.getValue() << std::endl;
						if (inner_mapping[k]->getToModel() != NULL)
						{
							sofa::component::container::MechanicalObject<Vec3Types>* toModel = (sofa::component::container::MechanicalObject<Vec3Types>*)(inner_mapping[k]->getToModel());
							Vec3d currentPos_To(toModel->getPosition()[0][0], toModel->getPosition()[0][1], toModel->getPosition()[0][2]);
							std::cout << "      current position of toModel '" << toModel->getName() << "' = " << currentPos_To << std::endl;
						}
						std::cout << "      position offset from RigidMapping '" << inner_mapping[k]->getName() << "' = " << inner_mapping[k]->appliedTranslation.getValue() << std::endl;
					
						return (currentPos_From + inner_mapping[k]->appliedTranslation.getValue());
					}
				}
			}
		}
	}
#endif
    //return m_cachedPosition;
}



//#define OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW
//#define OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW_TRANSFORMED_VERTICES
template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW
    if (_mObject)
    {
        glPushMatrix();
        typename DataTypes::VecCoord objTransform = _mObject->getPosition();
        Vector3 position(objTransform[0][0], objTransform[0][1], objTransform[0][2]);
        glTranslated(position.x(), position.y(), position.z());

        BVHDrawHelpers::drawCoordinateMarkerGL(6.0, 2.0);
        glPopMatrix();
    }

#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW_TRANSFORMED_VERTICES
    if (m_d && m_d->m_transformedVertices)
    {
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (int i = 0; i < m_gpModel->nVerts; i++)
        {
            glColor4f(1,1,0,0.75);
            glVertex3d(m_d->m_transformedVertices[i].v.x, m_d->m_transformedVertices[i].v.y, m_d->m_transformedVertices[i].v.z);
        }
        glEnd();
        glPointSize(1.0f);
    }
#endif

#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_DRAW_EDGE_LABELS
    Mat<4,4, GLfloat> modelviewM;
    const VecCoord& coords = *(this->_mState->getX());
    const sofa::defaulttype::Vec3f& color = Vec3f(0.5,0.5,0.5);
    glColor3f(color[0], color[1], color[2]);
    glDisable(GL_LIGHTING);
    const sofa::defaulttype::BoundingBox& bbox = this->getContext()->f_bbox.getValue();
    float scale = (float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.1f);

    scale = scale * m_edgeLabelScaleFactor.getValue();

    const sofa::helper::vector <Edge>& edgeArray = this->getMeshTopology()->getEdges();

    for (unsigned int i = 0; i < edgeArray.size(); i++)
    {

        Edge the_edge = edgeArray[i];
        Vector3 vertex1 = coords[ the_edge[0] ];
        Vector3 vertex2 = coords[ the_edge[1] ];
        sofa::defaulttype::Vec3f center; center = (DataTypes::getCPos(vertex1)+DataTypes::getCPos(vertex2))/2;

        std::ostringstream oss;
        oss << i;
        std::string tmp = oss.str();
        const char* s = tmp.c_str();
        glPushMatrix();

        glTranslatef(center[0], center[1], center[2]);
        glScalef(scale,scale,scale);

        // Makes text always face the viewer by removing the scene rotation
        // get the current modelview matrix
        glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
        modelviewM.transpose();

        sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

        //glLoadMatrixf(modelview);
        glLoadIdentity();

        glTranslatef(temp[0], temp[1], temp[2]);
        glScalef(scale,scale,scale);

        while(*s)
        {
            glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
            s++;
        }

        glPopMatrix();
    }
#endif
#endif
	//std::cout << "ObbTreeGPUCollisionModel<DataTypes>::draw(): m_modelLoaded = " << m_modelLoaded << ", m_pqp_tree = " << m_pqp_tree << std::endl;
	//std::cout << "Draw _mObject: " << _objectMState->getName() << ", is-a " << _objectMState->getTypeName() << std::endl;

    if (m_cubeModel)
        m_cubeModel->draw(vparams);

    if ( (m_drawOBBHierarchy.getValue())
         && m_modelLoaded && m_pqp_tree)
    {
        Vector3 newTr;// Quaternion newRot;
        Matrix3 newOrientation;
        getCachedPositionAndOrientation();

        newTr = m_d->m_modelTranslation;
        newOrientation = m_d->m_modelOrientation;
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

#define OBBTREEGPUCOLLISIONMODEL_DEBUG_UPDATE_INTERNAL_GEOMETRY
template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::updateInternalGeometry()
{
    return;
#if 0

    Matrix3 rotMatrix;
    //m_cachedOrientation.toMatrix(rotMatrix);
	getCachedOrientation().toMatrix(rotMatrix);
    m_d->m_modelOrientation = rotMatrix;
    
	//m_d->m_modelTranslation = m_cachedPosition;
	m_d->m_modelTranslation = getCachedPosition();

#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_UPDATE_INTERNAL_GEOMETRY
    std::cout << "ObbTreeGPUCollisionModel<DataTypes>::updateInternalGeometry(" << this->getName() << ")" << std::endl;
    std::cout << "  POSITION: " << m_cachedPosition << std::endl;
    std::cout << "  MATRIX: " << rotMatrix  << ", QUATERNION: " << m_cachedOrientation << std::endl;
#endif

    if (!m_updateVerticesInModel.getValue())
    {
        std::cout << "updateVerticesInModel = false: NOT UPDATING position for " << this->getName() << std::endl;
        return;
    }
    else
    {
        std::cout << "updateVerticesInModel = true: Updating position for " << this->getName() << std::endl;
    }
    /*if (!this->isMoving())
        return;*/

#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_UPDATE_INTERNAL_GEOMETRY
    std::cout << "MATRIX: " << rotMatrix  << ", QUATERNION: " << getCachedOrientation() << std::endl;
    /*GPUVertex* tfVertices1 = new GPUVertex[m_gpModel->nVerts];

    FROMGPU(tfVertices1, m_gpModel->vertexTfPointer, sizeof(GPUVertex) * m_gpModel->nVerts);
    std::cout << "== VERTICES BEFORE TRANSFORM ==" << std::endl;
    for (int k = 0; k < m_gpModel->nVerts; k++)
        std::cout << " * " << tfVertices1[k].v.x << "," << tfVertices1[k].v.y << "," << tfVertices1[k].v.z << std::endl;*/
#endif

    gpTransform modelTransform;
    //modelTransform.m_T[0] = m_cachedPosition.x(); modelTransform.m_T[1] = m_cachedPosition.y(); modelTransform.m_T[2] = m_cachedPosition.z();
	Vector3 cachedPosition = getCachedPosition();
	modelTransform.m_T[0] = cachedPosition.x(); modelTransform.m_T[1] = cachedPosition.y(); modelTransform.m_T[2] = cachedPosition.z();

    for (short k = 0; k < 3; k++)
    {
        for (short l = 0; l < 3; l++)
        {
            modelTransform.m_R[k][l] = rotMatrix(k,l);
        }
    }

#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_UPDATE_INTERNAL_GEOMETRY
	std::cout << "updateInternalGeometry_cuda is about to be called for " << this->getName() << "." << std::endl;
    std::cout << "cachedPositionChange = " << cachedPositionChange << ", cachedOrientationChange = " << cachedOrientationChange << std::endl;
#endif

    //updateInternalGeometry_cuda(this->m_gpModel, m_d->m_transformedVertices, modelTransform, (cachedPositionChange || cachedOrientationChange));

#ifdef OBBTREEGPUCOLLISIONMODEL_DEBUG_UPDATE_INTERNAL_GEOMETRY
    /*FROMGPU(tfVertices1, m_gpModel->vertexTfPointer, sizeof(GPUVertex) * m_gpModel->nVerts);
    std::cout << "== VERTICES AFTER TRANSFORM ==" << std::endl;
    for (int k = 0; k < m_gpModel->nVerts; k++)
        std::cout << " * " << tfVertices1[k].v.x << "," << tfVertices1[k].v.y << "," << tfVertices1[k].v.z << std::endl;

    delete[] tfVertices1;*/
#endif

#endif

}


template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::computeBoundingTreeRec(PQP_Model* tree, BV* obb, Matrix3 treeOrientation, Vector3& accumulatedChildOffsets, int boxIndex, int currentDepth, int maxDepth)
{
    std::cout << "  computeBoundingTreeRec(" << obb << "), boxIndex = " << boxIndex << std::endl;
    
	if (currentDepth > maxDepth)
	{
		std::cout << "  computeBoundingTreeRec(" << obb << "), currentDepth = " << currentDepth << ", maxDepth = " << maxDepth << ", return" << std::endl;
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

    std::cout << "  AABB " << boxIndex << ": min / max = " << cMin << " / " << cMax << std::endl;
    
	if (boxIndex < m_pqp_tree->num_bvs)
	{
		m_cubeModel->setParentOf(boxIndex, cMin, cMax);

		//std::cout << "  children: " << obb->numOBBChildren() << std::endl;
		for (unsigned int i = 0; i < 2; i++)
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

//#define OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
#define OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE
template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::computeBoundingTree(int maxDepth)
{
#ifdef OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE
#ifdef OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
    std::cout << "=== ObbTreeGPUCollisionModel::computeBoundingTree(" << this->getName() << "," << maxDepth << ") ===" << std::endl;
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
#ifdef OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
        std::cout << " Tree level min/max: " << treeMin << " / " << treeMax << std::endl;
#endif
            m_cubeModel->setLeafCube(0, std::make_pair(this->begin(),this->end()), treeMin, treeMax);
        }
    }
    else
    {
#ifdef OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
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

#ifdef OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
        std::cout << " Tree level min/max: " << treeMin << " / " << treeMax << std::endl;
		std::cout << " total OBB count = " << m_pqp_tree->num_bvs << std::endl;
#endif
        m_cubeModel->setParentOf(0, treeMin, treeMax);

#ifdef OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
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

#ifdef OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
					std::cout << "  accessed child node " << i << " = " << m_pqp_tree->b->first_child + i << std::endl;
#endif

					Vector3 childTranslation(childNode->To[0], childNode->To[1], childNode->To[2]);
					sofa::defaulttype::Matrix3 childOrientation;
					for (short k = 0; k < 3; k++)
					for (short l = 0; l < 3; l++)
						childOrientation[k][l] = childNode->R[k][l];

					Vector3 childHeTransform = modelPosition + modelOrientation.rotate(treeCenter) +
						(childOrientation * childTranslation);
#ifdef OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
					std::cout << "  node " << i << ": computeBoundingTreeRec(" << (m_pqp_tree->b->first_child + i) - 1 << ")" << std::endl;
#endif
					computeBoundingTreeRec(m_pqp_tree, childNode, treeOrientation, childHeTransform, (m_pqp_tree->b->first_child + i), 0, 1 /*maxDepth*/);
				}
			}
		}
#endif   
		m_cubeModel->computeBoundingTree(maxDepth);
    }

#ifdef OBBTREEGPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
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
#endif
}

//#define OBBTREEGPU_COLLISIONMODEL_DEBUG_COMPUTEOBBHIERARCHY
template <class DataTypes>
void ObbTreeGPUCollisionModel<DataTypes>::computeOBBHierarchy(bool fromFile)
{
    std::cout << "ObbTreeGPUCollisionModel<LGCDataTypes>::computeOBBHierarchy(" << this->getName() << ")" << std::endl;
    if (fromFile && !m_modelFile.empty())
    {
        if (m_modelLoaded)
            return;

        m_gpModel = new ModelInstance();

        gpTransform modelTransform;
        modelTransform.set_identity();
#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_COMPUTEOBBHIERARCHY
        std::cout << "Scale to apply: " << m_scale << std::endl;
#endif
        int loadResult = m_gpModel->load(m_modelFile.c_str(), m_scale, modelTransform);
        std::cout << "Load result: " << loadResult << std::endl;
        if (loadResult != -1 && m_gpModel->nTris > 0 && m_gpModel->nVerts > 0)
        {
            m_pqp_tree = (PQP_Model*)createPQPModel(m_gpModel, m_alarmDistance, false);
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
    else
    {
        BaseMeshTopology* meshTopology = this->getMeshTopology();
        sofa::core::behavior::MechanicalState<DataTypes>* mechanicalState = this->getMechanicalState();

        if (meshTopology->getNbQuads() > 0) { serr << meshTopology->getNbQuads() << " Quads in Mesh Topology ignored!" << sendl; };
        if (meshTopology->getNbHexas() > 0) { serr << meshTopology->getNbHexas() << " Hexas in Mesh Topology ignored!" << sendl; };
        if (meshTopology->getNbTetras() > 0) { serr << meshTopology->getNbTetras() << " Tetras in Mesh Topology ignored!" << sendl; };
        if (meshTopology->getNbHexahedra() > 0) { serr << meshTopology->getNbHexahedra() << " Hexahedra in Mesh Topology ignored!" << sendl; };
        if (meshTopology->getNbTetrahedra() > 0 ) { serr << meshTopology->getNbTetrahedra() << " Tetrahedra in Mesh Topology ignored!" << sendl; };

        typename core::behavior::MechanicalState<DataTypes>::ReadVecCoord pos = mechanicalState->readPositions();

        if (pos.size() > 0 && meshTopology->getNbTriangles() > 0)
        {
            m_gpModel = new ModelInstance();
            m_gpModel->nVerts = pos.size();
            m_gpModel->nTris = meshTopology->getNbTriangles();

#ifdef _WIN32
            m_gpModel->verlist = (SSEVertex *)_aligned_malloc(sizeof(SSEVertex) * m_gpModel->nVerts, 16);
#else
            posix_memalign((void**) &(m_gpModel->verlist), 16, sizeof(SSEVertex) * m_gpModel->nVerts);
#endif

            /*Vector3 objPosition;
            this->getPosition(objPosition);
            Quaternion objOrientation;
            this->getOrientation(objOrientation);*/

            //std::cout << "=== Build ObbTreeGPU model " << this->getName() << " ===" << std::endl;
            //std::cout << " add vertices: " << mechanicalState->getX()->size() << std::endl;
            for (int k = 0; k < pos.size(); k++)
            {
                Vector3 objVertex = pos[k];
                objVertex -= m_initialPosition;
                objVertex = m_initialOrientation.inverseRotate(objVertex);
                m_gpModel->verlist[k] = SSEVertex(objVertex.x(), objVertex.y(), objVertex.z());
                //std::cout << "  * " << k << ": " << m_gpModel->verlist[k].e[0] << "," << m_gpModel->verlist[k].e[1] << "," << m_gpModel->verlist[k].e[2] << "; original vertex = " << origVertex << std::endl;
            }

            m_gpModel->trilist = new Triangle_t[m_gpModel->nTris];

#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_COMPUTEOBBHIERARCHY
            std::cout << " add triangles: " << meshTopology->getNbTriangles() << std::endl;
#endif

            for (int k = 0; k < meshTopology->getNbTriangles(); k++)
            {
                Triangle_t tri;
                BaseMeshTopology::Triangle meshTri = meshTopology->getTriangle(k);
                tri.p[0] = meshTri[0];
                tri.p[1] = meshTri[1];
                tri.p[2] = meshTri[2];

                Vector3 v1 = pos[tri.p[0]];
                Vector3 v2 = pos[tri.p[1]];
                Vector3 v3 = pos[tri.p[2]];

                v1 -= m_initialPosition;
                v1 = m_initialOrientation.inverseRotate(v1);

                v2 -= m_initialPosition;
                v2 = m_initialOrientation.inverseRotate(v2);

                v3 -= m_initialPosition;
                v3 = m_initialOrientation.inverseRotate(v3);

#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_COMPUTEOBBHIERARCHY
                std::cout << "  * " << k << ": " << tri.p[0] << " (" << v1 << "), " << tri.p[1] << " (" << v2 << "), " << tri.p[2] << " (" << v3 << ")" << std::endl;
#endif

                tri.updateFromVerticesNoReorder(m_gpModel->verlist[tri.p[0]],
                    m_gpModel->verlist[tri.p[1]],
                    m_gpModel->verlist[tri.p[2]]);

                tri.n = -tri.n;

                m_gpModel->trilist[k] = tri;
            }
            m_pqp_tree = (PQP_Model*)createPQPModel(m_gpModel, m_alarmDistance, false);
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

    if (m_modelLoaded)
    {
        initHierarchy();

        m_d = new ObbTreeGPUCollisionModelPrivate(m_gpModel);
    }
}


//#define OBBTREEGPU_COLLISIONMODEL_DEBUG_INITHIERARCHY
template <class LGCDataTypes>
void ObbTreeGPUCollisionModel<LGCDataTypes>::initHierarchy()
{
#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_INITHIERARCHY
    std::cout << "=== ObbTreeGPUCollisionModel<LGCDataTypes>::initHierarchy(" << this->getName() << ") ===" << std::endl;
#endif

    int maxV, maxI;
    glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &maxV);
    glGetIntegerv(GL_MAX_ELEMENTS_INDICES, &maxI);

#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_INITHIERARCHY
    std::cout << "OpenGL VBO max vertices " << maxV << " " << "max indices " << maxI << std::endl;
    std::cout << "Model vertices count: " << m_gpModel->nVerts << ", indices count: " << m_gpModel->nTris * 3 << std::endl;
#endif

    VBOVertex* modelVertices = new VBOVertex[m_gpModel->nVerts];
    unsigned int* modelIndices = new unsigned int[m_gpModel->nTris * 3];

#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_INITHIERARCHY
    std::cout << " VERTICES FOR gpModel " << this->getName() << std::endl;
#endif

    for(unsigned int t = 0; t < m_gpModel->nVerts; ++t)
    {
        modelVertices[t].v = m_gpModel->verlist[t];

#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_INITHIERARCHY
        std::cout << " * " << t << ": " << modelVertices[t].v.x() << "," << modelVertices[t].v.y() << "," << modelVertices[t].v.z() << std::endl;
#endif

    }

    for(unsigned int t = 0; t < m_gpModel->nTris; ++t)
    {
        modelIndices[3 * t] = m_gpModel->trilist[t].p[0];
        modelIndices[3 * t + 1] = m_gpModel->trilist[t].p[1];
        modelIndices[3 * t + 2] = m_gpModel->trilist[t].p[2];
    }

    glGenBuffers(1, &m_gpModel->vbo_Vertex);
    glBindBuffer(GL_ARRAY_BUFFER, m_gpModel->vbo_Vertex);
    glBufferData(GL_ARRAY_BUFFER, m_gpModel->nVerts * sizeof(VBOVertex), modelVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (glGetError() != GL_NO_ERROR)
    {
#ifdef USE_GLU_ERROR_STRING
       const GLubyte *errString = gluErrorString(errCode);
       std::cerr << "OpenGL Error: " << errString << std::endl;
#else
        std::cerr << "OpenGL error: " << glGetError() << std::endl;
#endif
    }

#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_INITHIERARCHY
    std::cout << "n_gpModel->vbo_Vertex = " << m_gpModel->vbo_Vertex << std::endl;
#endif

    glGenBuffers(1, &m_gpModel->vbo_TriIndex);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gpModel->vbo_TriIndex);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_gpModel->nTris * 3 * sizeof(unsigned int), modelIndices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    if (glGetError() != GL_NO_ERROR)
    {
#ifdef USE_GLU_ERROR_STRING
        const GLubyte *errString = gluErrorString(glGetError());
        std::cerr << "OpenGL Error: " << errString << std::endl;
#else
        std::cerr << "OpenGL error: " << glGetError() << std::endl;
#endif
    }

#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_INITHIERARCHY
    std::cout << "m_gpModel->vbo_TriIndex = " << m_gpModel->vbo_TriIndex << std::endl;
#endif

    delete [] modelIndices;
    modelIndices = NULL;
    delete [] modelVertices;
    modelVertices = NULL;


    GPUVertex* vertexPointer = NULL;
    uint3* triIdxPointer = NULL;

#ifdef USE_DEPRECATED_CUDA_API
	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(m_gpModel->vbo_Vertex));
	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(m_gpModel->vbo_TriIndex));
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&vertexPointer, m_gpModel->vbo_Vertex));
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&triIdxPointer, m_gpModel->vbo_TriIndex));
#else
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(m_gpModel->vbo_Vertex));
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(m_gpModel->vbo_TriIndex));
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&vertexPointer, m_gpModel->vbo_Vertex));
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&triIdxPointer, m_gpModel->vbo_TriIndex));
#endif

    


    if (m_useVertexBuffer.getValue())
    {
        m_gpModel->vertexPointer = vertexPointer;
        m_gpModel->triIdxPointer = triIdxPointer;
    }
    else
    {
        GPUVertex* global_vertexPointer = NULL;
        uint3* global_triIdxPointer = NULL;
        GPUMALLOC((void**)&global_vertexPointer, sizeof(GPUVertex) * m_gpModel->nVerts);
        GPUMALLOC((void**)&global_triIdxPointer, sizeof(uint3) * m_gpModel->nTris);
        GPUTOGPU(global_vertexPointer, vertexPointer, sizeof(GPUVertex) * m_gpModel->nVerts);
        GPUTOGPU(global_triIdxPointer, triIdxPointer, sizeof(uint3) * m_gpModel->nTris);
        m_gpModel->vertexPointer = global_vertexPointer;
        m_gpModel->triIdxPointer = global_triIdxPointer;

        GPUVertex* global_tfVertexPointer = NULL;

        GPUMALLOC((void**)&global_tfVertexPointer, sizeof(GPUVertex) * m_gpModel->nVerts);
        GPUTOGPU(global_tfVertexPointer, vertexPointer, sizeof(GPUVertex) * m_gpModel->nVerts);
        m_gpModel->vertexTfPointer = global_tfVertexPointer;

        CUDA_SAFE_CALL(cudaGLUnmapBufferObject(m_gpModel->vbo_Vertex));
        CUDA_SAFE_CALL(cudaGLUnmapBufferObject(m_gpModel->vbo_TriIndex));
        CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(m_gpModel->vbo_Vertex));
        CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(m_gpModel->vbo_TriIndex));
    }


    OBBNode_host* object = PQP_createOBBTree(m_gpModel, m_alarmDistance);
    OBBNode* d_obbTree = NULL;
    GPUMALLOC((void**)&d_obbTree, sizeof(OBBNode) * (2 * m_gpModel->nTris - 1));
    OBBNode* h_obbTree = new OBBNode[2 * m_gpModel->nTris - 1];

    std::cout << "Create h_obbTree: " << 2 * m_gpModel->nTris - 1 << " OBB nodes." << std::endl;
    for(int j = 0; j < 2 * m_gpModel->nTris - 1; ++j)
    {
        h_obbTree[j].bbox.axis1 = object[j].bbox.axis1;
        h_obbTree[j].bbox.axis2 = object[j].bbox.axis2;
        h_obbTree[j].bbox.axis3 = object[j].bbox.axis3;
        h_obbTree[j].bbox.center = object[j].bbox.center;
        h_obbTree[j].bbox.extents = object[j].bbox.extents;
        h_obbTree[j].left = object[j].left;
        h_obbTree[j].right = object[j].right;

        h_obbTree[j].bbox.idx = object[j].bbox.idx;
        h_obbTree[j].bbox.min_dimension = object[j].bbox.min_dimension;
        h_obbTree[j].bbox.min_dimension_val = object[j].bbox.min_dimension_val;

        //std::cout << " * Node " << j << ": min_dimension = " << h_obbTree[j].bbox.min_dimension << ", min_dimension_val = " << h_obbTree[j].bbox.min_dimension_val << std::endl;
    }

    TOGPU(d_obbTree, h_obbTree, sizeof(OBBNode) * (2 * m_gpModel->nTris - 1));
    m_gpModel->obbTree = d_obbTree;

    delete [] object;
    delete [] h_obbTree;

    this->size = this->numTriangles();
}

template <class LGCDataTypes>
void ObbTreeGPUCollisionModel<LGCDataTypes>::disposeHierarchy()
{
#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_DISPOSEHIERARCHY
    std::cout << "=== ObbTreeGPUCollisionModel<LGCDataTypes>::disposeHierarchy(" << this->getName() << ") ===" << std::endl;
    std::cout << " m_modelLoaded = " << m_modelLoaded << ", m_gpModel = " << m_gpModel << std::endl;
#endif

    if (m_modelLoaded && m_gpModel)
    {
        if (m_useVertexBuffer.getValue())
        {
            cudaError err;
            {
#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_DISPOSEHIERARCHY
                std::cout << "m_gpModel->vbo_Vertex = " << m_gpModel->vbo_Vertex << std::endl;
#endif

                err = cudaGLUnmapBufferObject(m_gpModel->vbo_Vertex);
                if (cudaSuccess != err)
                {
                    std::cerr << "cudaGLUnmapBufferObject(d->_gpModel->vbo_Vertex): Cuda error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
                }
                err = cudaGLUnregisterBufferObject(m_gpModel->vbo_Vertex);
                if (cudaSuccess != err)
                {
                    std::cerr << "cudaGLUnregisterBufferObject(d->_gpModel->vbo_Vertex): Cuda error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
                }
            }

            {
                std::cout << "m_gpModel->vbo_Vertex = " << m_gpModel->vbo_TriIndex << std::endl;
                err = cudaGLUnmapBufferObject(m_gpModel->vbo_TriIndex);
                if (cudaSuccess != err)
                {
                    std::cerr << "cudaGLUnmapBufferObject(d->_gpModel->vbo_TriIndex): Cuda error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
                }
                err = cudaGLUnregisterBufferObject(m_gpModel->vbo_TriIndex);
                if (cudaSuccess != err)
                {
                    std::cerr << "cudaGLUnregisterBufferObject(d->_gpModel->vbo_TriIndex): Cuda error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;
                }
            }
        }
        else
        {
            GPUFREE(m_gpModel->vertexPointer);
            GPUFREE(m_gpModel->vertexTfPointer);
            GPUFREE(m_gpModel->triIdxPointer);
        }

        if (m_gpModel->obbTree != NULL)
            GPUFREE(m_gpModel->obbTree);
#ifdef OBBTREEGPU_COLLISIONMODEL_DEBUG_DISPOSEHIERARCHY
        std::cout << " unregistered vertex/index buffers, free'd obbTree structure" << std::endl;
#endif
    }
}

#endif //OBBTREEGPU_COLLISIONMODEL_INL
