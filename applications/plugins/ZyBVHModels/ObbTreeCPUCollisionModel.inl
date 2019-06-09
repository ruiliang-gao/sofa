#ifndef OBBTREECPU_COLLISIONMODEL_INL
#define OBBTREECPU_COLLISIONMODEL_INL

#include "ObbTreeCPUCollisionModel.h"

#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/ClassInfo.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/Node.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaLoader/MeshObjLoader.h>

#include "BVHDrawHelpers.h"
#include "ObbTree.h"

using namespace sofa::component::collision;
using namespace sofa::core::objectmodel;
using namespace sofa;

template <class DataTypes>
ObbTreeCPUCollisionModel<DataTypes>::ObbTreeCPUCollisionModel(): core::CollisionModel(),
                                                                 m_scale(1.0f), m_modelLoaded(false),
                                                                 _mState(NULL), _objectMState(NULL), _mObject(NULL),
                                                                 m_drawOBBHierarchy(initData(&m_drawOBBHierarchy, true, "drawOBBHierarchy", "Draw the model's OBB hierarchy")),
                                                                 m_minDrawDepth(initData(&m_minDrawDepth, (unsigned int) 0, "minDrawDepth", "Minimum depth of OBB hierarchy to draw")),
                                                                 m_maxDrawDepth(initData(&m_maxDrawDepth, (unsigned int) 5, "maxDrawDepth", "Maximum depth of OBB hierarchy to draw"))

{
    m_minDrawDepth.setGroup("Visualization");
    m_maxDrawDepth.setGroup("Visualization");
}

template <class DataTypes>
ObbTreeCPUCollisionModel<DataTypes>::~ObbTreeCPUCollisionModel()
{

}

template <class DataTypes>
void ObbTreeCPUCollisionModel<DataTypes>::init()
{
    using namespace sofa::component::container;

    _mState = dynamic_cast< core::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());

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
            _objectMState = current;
        }

        sofa::simulation::Node* parentNode = dynamic_cast<simulation::Node*>(this->getContext());

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

    computeOBBHierarchy();
}

template <class DataTypes>
void ObbTreeCPUCollisionModel<DataTypes>::cleanup()
{

}

template <class DataTypes>
unsigned int ObbTreeCPUCollisionModel<DataTypes>::numTriangles()
{
    //return m_gpModel->nTris;
    return this->m_obbTree.getTopology()->getNbTriangles();
}

template <class DataTypes>
unsigned int ObbTreeCPUCollisionModel<DataTypes>::numVertices()
{
    const Vec3Types::VecCoord& x = this->m_obbTree.getMState()->read(core::ConstVecCoordId::position())->getValue();
    return x.size();
}

template <class DataTypes>
unsigned int ObbTreeCPUCollisionModel<DataTypes>::numOBBs()
{
    //return m_pqp_tree->num_bvs;
    return this->m_obbTree.getObbNodes().size();
}

template <class DataTypes>
bool ObbTreeCPUCollisionModel<DataTypes>::getVertex(unsigned int idx, Vector3& outVertex)
{
    /*if (idx < m_gpModel->nVerts)
    {
        outVertex = Vector3(m_gpModel->verlist[idx].x(), m_gpModel->verlist[idx].y(), m_gpModel->verlist[idx].z());
        return true;
    }
    else
    {
        return false;
    }*/
    const Vec3Types::VecCoord& x = this->m_obbTree.getMState()->read(core::ConstVecCoordId::position())->getValue();
    if (idx < x.size())
    {
        outVertex = x[idx];
        return true;
    }
    return false;
}

template <class DataTypes>
bool ObbTreeCPUCollisionModel<DataTypes>::getTriangle(unsigned int idx, sofa::core::topology::Triangle& outTri)
{
    if (idx < this->m_obbTree.getTopology()->getNbTriangles())
    {
        outTri = m_obbTree.getTopology()->getTriangle(idx);
        return true;
    }
    return false;
}

template <class DataTypes>
bool ObbTreeCPUCollisionModel<DataTypes>::getPosition(Vector3& position) const
{
    const sofa::core::objectmodel::BaseData* posData = _objectMState->baseRead(core::ConstVecCoordId::position());
    if (posData)
    {
        const void* posValues = posData->getValueVoidPtr();
        double t0 = posData->getValueTypeInfo()->getScalarValue(posValues, 0);
        double t1 = posData->getValueTypeInfo()->getScalarValue(posValues, 1);
        double t2 = posData->getValueTypeInfo()->getScalarValue(posValues, 2);
        position = Vector3(t0,t1,t2);

        return true;
    }
    return false;
}

template <class DataTypes>
bool ObbTreeCPUCollisionModel<DataTypes>::getOrientation(Matrix3& orientation) const
{
    const sofa::core::objectmodel::BaseData* posData = _objectMState->baseRead(core::ConstVecCoordId::position());
    if (posData)
    {
        const void* posValues = posData->getValueVoidPtr();
        double r0 = posData->getValueTypeInfo()->getScalarValue(posValues, 3);
        double r1 = posData->getValueTypeInfo()->getScalarValue(posValues, 4);
        double r2 = posData->getValueTypeInfo()->getScalarValue(posValues, 5);
        double r3 = posData->getValueTypeInfo()->getScalarValue(posValues, 6);

        Quaternion ori(r0,r1,r2,r3);
        ori.toMatrix(orientation);

        return true;
    }
    return false;
}

template <class DataTypes>
bool ObbTreeCPUCollisionModel<DataTypes>::getOrientation(Quaternion& orientation) const
{
    const sofa::core::objectmodel::BaseData* posData = _objectMState->baseRead(core::ConstVecCoordId::position());
    if (posData)
    {
        const void* posValues = posData->getValueVoidPtr();
        double r0 = posData->getValueTypeInfo()->getScalarValue(posValues, 3);
        double r1 = posData->getValueTypeInfo()->getScalarValue(posValues, 4);
        double r2 = posData->getValueTypeInfo()->getScalarValue(posValues, 5);
        double r3 = posData->getValueTypeInfo()->getScalarValue(posValues, 6);

        orientation = Quaternion(r0,r1,r2,r3);

        return true;
    }
    return false;
}


#define OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE
//#define OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
template <class DataTypes>
void ObbTreeCPUCollisionModel<DataTypes>::computeBoundingTree(int maxDepth)
{
#ifdef OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE
#ifdef OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
	std::cout << "=== ObbTreeCPUCollisionModel::computeBoundingTree(" << this->getName() << "," << maxDepth << ") ===" << std::endl;
#endif

	if (!m_cubeModel)
		m_cubeModel = createPrevious<CubeModel>();

	if (!isMoving() && !m_cubeModel->empty())
		return; // No need to recompute BBox if immobile

    Vector3 modelPosition = getCachedPosition();
    Quaternion modelOrientation = getCachedOrientation();

	m_cubeModel->resize(0);

	sofa::defaulttype::Vector3 treeCenter(m_obbTree.getPosition().x(), m_obbTree.getPosition().y(), m_obbTree.getPosition().z());
	sofa::defaulttype::Vector3 he(m_obbTree.getHalfExtents().x(), m_obbTree.getHalfExtents().y(), m_obbTree.getHalfExtents().z());
	sofa::defaulttype::Vector3 heTransform = modelPosition + modelOrientation.rotate(treeCenter);

	sofa::defaulttype::Matrix3 treeOrientation = m_obbTree.getOrientation();
	
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
#ifdef OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
			std::cout << " Tree level min/max: " << treeMin << " / " << treeMax << std::endl;
#endif
			m_cubeModel->setLeafCube(0, std::make_pair(this->begin(), this->end()), treeMin, treeMax);
		}
	}
	else
	{
#ifdef OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
		std::cout << "RESIZE m_cubeModel to " << m_obbTree.getObbNodes().size() << std::endl;
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

#ifdef OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
		std::cout << " Tree level min/max: " << treeMin << " / " << treeMax << std::endl;
		std::cout << " total OBB count = " << m_obbTree.getObbNodes().size() << std::endl;
#endif
		m_cubeModel->setParentOf(0, treeMin, treeMax);

#ifdef OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
		std::cout << " tree node 1st child = " << m_obbTree.getFirstChild() << ", 2nd child = " << m_obbTree.getSecondChild() << std::endl;
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

#ifdef OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
					std::cout << "  accessed child node " << i << " = " << m_pqp_tree->b->first_child + i << std::endl;
#endif

					Vector3 childTranslation(childNode->To[0], childNode->To[1], childNode->To[2]);
					sofa::defaulttype::Matrix3 childOrientation;
					for (short k = 0; k < 3; k++)
					for (short l = 0; l < 3; l++)
						childOrientation[k][l] = childNode->R[k][l];

					Vector3 childHeTransform = modelPosition + modelOrientation.rotate(treeCenter) +
						(childOrientation * childTranslation);
#ifdef OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
					std::cout << "  node " << i << ": computeBoundingTreeRec(" << (m_pqp_tree->b->first_child + i) - 1 << ")" << std::endl;
#endif
					computeBoundingTreeRec(m_pqp_tree, childNode, treeOrientation, childHeTransform, (m_pqp_tree->b->first_child + i), 0, 1 /*maxDepth*/);
				}
			}
		}
#endif   
		m_cubeModel->computeBoundingTree(maxDepth);
	}

#ifdef OBBTREECPUCOLLISIONMODEL_COMPUTE_BOUNDING_TREE_DEBUG
	std::cout << "=== CubeModel new size: " << m_cubeModel->getNumberCells() << " ===" << std::endl;
	for (unsigned int i = 0; i < m_cubeModel->getNumberCells(); i++)
	{
		std::cout << "  * cell " << i << ": ";
		std::pair<core::CollisionElementIterator, core::CollisionElementIterator> ic = m_cubeModel->getInternalChildren(i);
		std::cout << " child 1 = " << ic.first.getIndex() << ", ";

		std::cout << " child 2 = " << ic.second.getIndex();

		std::cout << std::endl;
	}
#endif
#endif
}

template <class DataTypes>
void ObbTreeCPUCollisionModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (m_obbTree.getMinDrawDepth() != m_minDrawDepth.getValue())
        m_obbTree.setMinDrawDepth(m_minDrawDepth.getValue());

    if (m_obbTree.getMaxDrawDepth() != m_maxDrawDepth.getValue())
        m_obbTree.setMaxDrawDepth(m_maxDrawDepth.getValue());

    if (this->m_drawOBBHierarchy.getValue())
        m_obbTree.draw(vparams);

    //std::cout << "ObbTreeGPUCollisionModel<DataTypes>::draw(): m_modelLoaded = " << m_modelLoaded << ", m_pqp_tree = " << m_pqp_tree << std::endl;
    /*if (m_drawOBBHierarchy.getValue() && m_modelLoaded && m_pqp_tree)
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

            Vec4f colour(1,1,0,1);
            Vec4f colour2(0,1,1,1);
            glBegin(GL_LINES);
            glColor4d(colour2.x(), colour2.y(), colour2.z(), colour2.w());
            glVertex3d(0, 0, 0);
            glColor4d(colour.x(), colour.y(), colour.z(), colour.w());
            glVertex3d(newTr.x(), newTr.y(), newTr.z());
            glEnd();

            //std::cout << " translate to obj. coord. = " << newTr << std::endl;
            glTranslated(newTr.x(), newTr.y(), newTr.z());

            BVHDrawHelpers::drawCoordinateMarkerGL(0.5f, 4.0f, colour, colour * 0.5, colour * 0.25);

            //std::cout << " rotate to obj. orientation = " << newOrientation.transposed() << std::endl;
            glMultMatrixd(modelGlOrientation.transposed().ptr());

            glBegin(GL_LINES);
            glColor4d(1, 1, 0, 0.5);
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

            bool emphasize = false;
            if (std::find(m_emphasizedIndices.begin(), m_emphasizedIndices.end(), 0) != m_emphasizedIndices.end())
                emphasize = true;

            BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]), colour, emphasize);

            glMultMatrixd(glOrientation.transposed().ptr());

            glTranslated(-m_pqp_tree->b->To[0], -m_pqp_tree->b->To[1], -m_pqp_tree->b->To[2]);

            if (m_pqp_tree->num_bvs > 2)
            {
                bool emphasize1 = false, emphasize2 = false;

                if (std::find(m_emphasizedIndices.begin(), m_emphasizedIndices.end(), m_pqp_tree->b->first_child) != m_emphasizedIndices.end())
                    emphasize1 = true;

                if (std::find(m_emphasizedIndices.begin(), m_emphasizedIndices.end(), m_pqp_tree->b->first_child+1) != m_emphasizedIndices.end())
                    emphasize2 = true;

                BVHDrawHelpers::drawRec(m_pqp_tree, m_pqp_tree->child(m_pqp_tree->b->first_child), vparams, colour2, 1, emphasize1);
                BVHDrawHelpers::drawRec(m_pqp_tree, m_pqp_tree->child(m_pqp_tree->b->first_child+1), vparams, colour2, 1, emphasize2);
            }

            glPopAttrib();
            glPopMatrix();
        }
    }*/
}

template <class DataTypes>
void ObbTreeCPUCollisionModel<DataTypes>::computeOBBHierarchy()
{
    sofa::core::topology::BaseMeshTopology* meshTopology = getContext()->getMeshTopology();

    try
    {
        Vector3 pos; Matrix3 rot;
        getPosition(pos);
        getOrientation(rot);
        std::cout << "ObbTreeBuilder instantiation: passing obbTree pointer = " << &m_obbTree << " from ObbTreeCPUCollisionModel " << this->getName() << std::endl;

        m_obbTree = ObbTree(this->getName(), meshTopology, (sofa::core::behavior::MechanicalState<Vec3Types>*)_mState);
        ObbTreeBuilder obbBuilder(&m_obbTree, meshTopology, (sofa::core::behavior::MechanicalState<Vec3Types>*)_mState);

        obbBuilder.buildTree();


        m_obbTree.m_position = m_obbTree.getObbNodes()[0].m_position;
        m_obbTree.m_halfExtents = m_obbTree.getObbNodes()[0].m_halfExtents;
        m_obbTree.m_localAxes = m_obbTree.getObbNodes()[0].m_localAxes;
        m_obbTree.t_rel_top = m_obbTree.getObbNodes()[0].t_rel_top;

        m_obbTree.translate(pos);
        m_obbTree.rotate(rot);

        m_obbTree.assignOBBNodeColors();

        std::vector<ObbVolume>& obbNodes = m_obbTree.getObbNodes();
        std::cout << "OBB tree for " << this->getName() << ": " << obbNodes.size() << " OBBs" << std::endl;

        if (obbNodes.size() > 0)
        {
            std::cout << " at: " << m_obbTree.getWorldPosition() << ", orientation = " << m_obbTree.getWorldOrientation() << ", children = " << m_obbTree.getFirstChild() << " + " << m_obbTree.getSecondChild() << std::endl;
            for (std::vector<ObbVolume>::iterator it = obbNodes.begin(); it != obbNodes.end(); it++)
            {
                ObbVolume& obb = *it;
                std::cout << " * " << obb.identifier() << " t_rel_top = " << obb.t_rel_top << ", position: " << obb.getPosition() << ", half-extents: " << obb.getHalfExtents() << ", local axes: " << obb.getOrientation() << ", children: " << obb.getFirstChild() << " of type " << obb.getChildType(0) << " + " << obb.getSecondChild() << " of type " << obb.getChildType(1) << std::endl;
            }
        }
    }
    catch (std::out_of_range& ex)
    {
        std::cout << "out_of_range exception in obbBuilder.buildTree(): " << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Exception in obbBuilder.buildTree()" << std::endl;
    }
}

#endif //OBBTREEGPU_COLLISIONMODEL_INL
