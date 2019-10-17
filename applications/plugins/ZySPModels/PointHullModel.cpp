#include "PointHullModel.h"

#include "BVHDrawHelpers.h"
#include <ZyPQP/include/PQP.h>

namespace sofa
{
	namespace component
	{
		namespace collision
		{
			SOFA_DECL_CLASS(PointHullModel)

				int PointHullModelClass = sofa::core::RegisterObject("Surface point hull model")
				.add< PointHullModel >();
		}
	}
}

#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/ClassInfo.h>

#ifdef _WIN32
#include <gl/glut.h>
#else
#include <GL/glut.h>
#endif

#include <Math/MathUtils.h>
#include <Query/Query2.h>

using namespace sofa;
using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::component::collision;
using namespace sofa::component::container;

HullPoint::HullPoint(PointHullModel *model, int index) : sofa::core::TCollisionElementIterator<PointHullModel>(model, index)
{
	std::cout << "HullPoint::HullPoint(model = " << model->getName() << ", index = " << index << ")" << std::endl;
	//_point = model->surfacePoint(index);
}

HullPoint::HullPoint(sofa::core::CollisionElementIterator &i) :
TCollisionElementIterator<PointHullModel>(static_cast<PointHullModel*>(i.getCollisionModel()), i.getIndex())
{
	std::cout << "HullPoint::HullPoint(model = " << i.getCollisionModel()->getName() << ", CollisionElementIterator = " << i.getIndex() << ")" << std::endl;
	PointHullModel* model = static_cast<PointHullModel*>(i.getCollisionModel());
	//_point = model->surfacePoint(i.getIndex());
}

PointHullModel::PointHullModel() : sofa::core::CollisionModel(), m_storage(NULL), m_computation(NULL)
{

}

PointHullModel::~PointHullModel()
{
	if (m_computation != NULL)
		delete m_computation;

	if (m_storage != NULL)
		delete m_storage;

	glDeleteLists(m_pointHullList, 1);
}

void PointHullModel::init()
{
	std::cout << "=== PointHullModel::init(" << this->getName() << ") BEGIN ===" << std::endl;

	core::behavior::MechanicalState<Vec3Types>* mState = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* >(getContext()->getMechanicalState());

	if (mState)
	{
		std::cout << "  got valid mState = " << mState->getName() << std::endl;
		component::container::MechanicalObject<Vec3Types>* mechanicalObject = dynamic_cast< component::container::MechanicalObject<Vec3Types>* >(mState);
		if (mechanicalObject != NULL)
		{
			std::cout << "  got valid mechanicalObject = " << mechanicalObject->getName() << std::endl;
			m_mechanicalObject = mechanicalObject;

			typedef MechanicalObject<Rigid3Types> mType;

			std::vector<mType*> mechObjects;
			sofa::core::objectmodel::BaseContext::GetObjectsCallBackT<mType, std::vector<mType*> > cbMech(&mechObjects);

			getContext()->getObjects(TClassInfo<mType>::get(), cbMech, TagSet(), BaseContext::SearchParents);

			sofa::core::behavior::BaseMechanicalState* objectMState = NULL;
			std::cout << "  objectMState alternatives = " << mechObjects.size() << std::endl;
 			if (mechObjects.size() == 1)
			{
				objectMState = mechObjects[0];
				std::cout << "  got valid objectMState = " << objectMState->getName() << std::endl;
			}

			sofa::core::topology::BaseMeshTopology* mTopology = getContext()->getMeshTopology();
			if (objectMState != NULL && mTopology != NULL)
			{
				std::cout << "  got valid mTopology = " << mTopology->getName() << std::endl;
				std::cout << "  construct storage and computation instances" << std::endl;
				m_storage = new PointHullStorage_Full(mState, objectMState, mTopology);
				m_computation = new PointHullComputation_CPU(m_storage);

				std::cout << "  computePointHull() call" << std::endl;
				m_computation->computePointHull();
			}
		}
	}

	m_pointHullList = glGenLists(1);
	glNewList(m_pointHullList, GL_COMPILE);
	initDrawList();
	glEndList();

	std::cout << "=== PointHullModel::init(" << this->getName() << ") END   ===" << std::endl;
}

void PointHullModel::cleanup()
{

}

void PointHullModel::computeBoundingTree(int maxDepth)
{

}

void PointHullModel::draw(const sofa::core::visual::VisualParams* vparams)
{
	if (vparams->displayFlags().getShowCollisionModels() &&
		(m_storage->numVertices() > 0 || m_storage->numEdgePoints() > 0))
	{
		Vector3 newTr = this->getCachedPosition();
		Quaternion newRot = this->getCachedOrientation();
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

		glTranslated(newTr.x(), newTr.y(), newTr.z());
		glMultMatrixd(modelGlOrientation.transposed().ptr());

		glCallList(m_pointHullList);
		
		glPopMatrix();

		glPushMatrix();
		glPushAttrib(GL_ENABLE_BIT);
		glEnable(GL_COLOR_MATERIAL);

		Mat<4, 4, GLfloat> modelviewM;
		//const sofa::defaulttype::BoundingBox& bbox = this->getContext()->f_bbox.getValue();
		float scale = 0.00025f;

		const std::map<unsigned long, Triangle3d>& surfaceTriangles = m_computation->getSurfaceTriangles();
		const SurfaceGridMap& surfaceGrid = m_storage->getSurfaceGridData();

		for (std::map<unsigned long, Triangle3d>::const_iterator it = surfaceTriangles.begin(); it != surfaceTriangles.end(); it++)
		{
			//std::cout << " triangle " << it->first << " (" << it->second.V[0] << "," << it->second.V[1] << "," << it->second.V[2] << ")" << std::endl;

			if (surfaceGrid.find(it->first) != surfaceGrid.end())
			{
				const SurfaceGrid& triangle_grid = surfaceGrid.find(it->first)->second;

				const Vector3 center = (it->second.V[0] + it->second.V[1] + it->second.V[2]);

				const char* s = triangle_grid.m_gridID.c_str();

				glPushMatrix();

				glTranslatef(center[0], center[1], center[2]);
				glScalef(scale, scale, scale);

				// Makes text always face the viewer by removing the scene rotation
				// get the current modelview matrix
				glGetFloatv(GL_MODELVIEW_MATRIX, modelviewM.ptr());
				modelviewM.transpose();

				sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

				glLoadIdentity();
				glColor4d(1, 1, 1, 0.5);
				glTranslatef(temp[0], temp[1], temp[2]);
				glScalef(scale * 10, scale * 10, scale * 10);

				while (*s)
				{
					glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
					s++;
				}

				glPopMatrix();

				for (unsigned int p = 0; p <= triangle_grid.m_sizeX; p++)
				{
					/*if (p % 2 != 0)
						continue;*/

					for (unsigned int q = 0; q <= triangle_grid.m_sizeY; q++)
					{
						/*if (q % 2 != 0)
							continue;*/

						Vector3 center = triangle_grid.m_surfacePoints.at(p)[q];

						std::ostringstream oss;
						oss << "(" << p << "," << q << "): " << center.x() << "," << center.y() << "," << center.z();
						std::string tmp = oss.str();
						const char* s = tmp.c_str();

						//std::cout << " label text " << tmp << " with scale = " << scale << std::endl;
						glPushMatrix();

						glTranslatef(center[0], center[1], center[2]);
						glScalef(scale, scale, scale);

						// Makes text always face the viewer by removing the scene rotation
						// get the current modelview matrix
						glGetFloatv(GL_MODELVIEW_MATRIX, modelviewM.ptr());
						modelviewM.transpose();
						//std::cout << " modelViewM = " << modelviewM << std::endl;

						sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

						glLoadIdentity();
						glColor4d(1, 1, 1, 0.5);
						glTranslatef(temp[0], temp[1], temp[2]);
						glScalef(scale, scale, scale);

						while (*s)
						{
							glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
							//std::cout << *s << std::endl;
							s++;
						}
						//std::cout << std::endl;

						glPopMatrix();
					}
				}
			}
		}
		glPopMatrix();
		glPopAttrib();

//#define POINTHULLMODEL_SHOW_OBB_CORNER_DATA
#ifdef POINTHULLMODEL_SHOW_OBB_CORNER_DATA

		PQP_Model* m_pqp_tree = m_computation->getPqpModel();
		if (m_pqp_tree)
		{
			Vector3 newTr = this->getCachedPosition();
			Quaternion newRot = this->getCachedOrientation();

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

				std::vector<Vector3> triangleObbPoints;

				glPushMatrix();
				glPushAttrib(GL_ENABLE_BIT);
				glEnable(GL_COLOR_MATERIAL);

				{
					Vector3 triangle_obb_center(m_pqp_tree->b->To[0], m_pqp_tree->b->To[1], m_pqp_tree->b->To[2]);
					Vector3 triangle_obb_extents(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]);

					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));

				}

				{
					Mat<4, 4, GLfloat> modelviewM;
					const sofa::defaulttype::BoundingBox& bbox = this->getContext()->f_bbox.getValue();
					float scale = (float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.001f);

					for (unsigned int k = 0; k < triangleObbPoints.size(); k++)
					{

						sofa::defaulttype::Vec3f center = triangleObbPoints[k];

						std::ostringstream oss;
						oss << "Corner " << k << ": " << center.x() << "," << center.y() << "," << center.z();
						std::string tmp = oss.str();
						const char* s = tmp.c_str();
						glPushMatrix();

						glTranslatef(center[0], center[1], center[2]);
						glScalef(scale, scale, scale);

						// Makes text always face the viewer by removing the scene rotation
						// get the current modelview matrix
						glGetFloatv(GL_MODELVIEW_MATRIX, modelviewM.ptr());
						modelviewM.transpose();

						sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

						//glLoadMatrixf(modelview);
						glLoadIdentity();

						glTranslatef(temp[0], temp[1], temp[2]);
						glScalef(scale, scale, scale);

						while (*s)
						{
							glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
							s++;
						}

						glPopMatrix();
					}

					for (unsigned int k = 0; k < m_pqp_tree->num_bvs; k++)
					{
						BV* obb_k = m_pqp_tree->child(k);
						Vector3 triangle_obb_center(obb_k->To[0], obb_k->To[1], obb_k->To[2]);
						Vector3 triangle_obb_extents(obb_k->d[0], obb_k->d[1], obb_k->d[2]);

						std::vector<Vector3> triangleObbPoints;

						triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
						triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));
						triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
						triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));
						triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
						triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));
						triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
						triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));

						Mat<4, 4, GLfloat> modelviewM;
						const sofa::defaulttype::BoundingBox& bbox = this->getContext()->f_bbox.getValue();
						float scale = (float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.001f);

						for (unsigned int k = 0; k < triangleObbPoints.size(); k++)
						{

							sofa::defaulttype::Vec3f center = triangleObbPoints[k];

							std::ostringstream oss;
							oss << "Corner " << k << ": " << center.x() << "," << center.y() << "," << center.z();
							std::string tmp = oss.str();
							const char* s = tmp.c_str();
							glPushMatrix();

							glTranslatef(center[0], center[1], center[2]);
							glScalef(scale, scale, scale);

							// Makes text always face the viewer by removing the scene rotation
							// get the current modelview matrix
							glGetFloatv(GL_MODELVIEW_MATRIX, modelviewM.ptr());
							modelviewM.transpose();

							sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

							//glLoadMatrixf(modelview);
							glLoadIdentity();

							glTranslatef(temp[0], temp[1], temp[2]);
							glScalef(scale, scale, scale);

							while (*s)
							{
								glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
								s++;
							}

							glPopMatrix();
						}
					}
				}

				glPopMatrix();
				glPopAttrib();
			}
		}
#endif
	}
}

void PointHullModel::draw(const sofa::core::visual::VisualParams* vparams, int index)
{

}

bool PointIsInTriangle(const Vector3& point, const Vector3& tri_vert0, const Vector3& tri_vert1, const Vector3& tri_vert2)
{
	Plane3d trianglePlane(tri_vert0, tri_vert1, tri_vert2);
	Vector3 U0, U1;

	BVHModels::Mathd::GenerateComplementBasis(U0, U1, trianglePlane.Normal);

	Vector3 PmV0 = point - tri_vert0;
	Vector3 V1mV0 = tri_vert1 - tri_vert0;
	Vector3 V2mV0 = tri_vert2 - tri_vert0;

	Vector2 ProjV[3] =
	{
		//TRU Vec<2,Real>(0,0,0),
		Vector2((double) 0.0, (double) 0.0),
		Vector2((U0 * V1mV0), (U1 * V1mV0)),
		Vector2((U0 * V2mV0), (U1 * V2mV0))
	};

	Vector2 ProjP((U0 * PmV0), (U1 * PmV0));

	BVHModels::Query2d tmpQuery(3, ProjV);
	if (tmpQuery.ToTriangle(ProjP, 0, 1, 2) <= 0)
	{
		//std::cout << "   point = " << point << "; position = INSIDE triangle" << std::endl;
		return true;
	}
	else
	{
		//std::cout << "   point = " << point << "; position = OUTSIDE triangle" << std::endl;
		return false;
	}
}

void PointHullModel::initDrawList()
{
#if 0
	if ((m_storage->numVertices() > 0 || m_storage->numEdgePoints() > 0))
	{
		Vector3 model_position = this->getCachedPosition();
		Matrix4 model_orientation;
		this->getCachedOrientation().writeOpenGlMatrix(model_orientation.ptr());

		glPointSize(4.0f);
		glPushMatrix();

		glTranslated(model_position.x(), model_position.y(), model_position.z());
		glMultMatrixd(model_orientation.ptr());

		if (m_storage->numVertices() > 0)
		{
			unsigned int numVertices = m_storage->numVertices();
			glBegin(GL_POINTS);
			for (unsigned int k = 0; k < numVertices; k++)
			{
				glColor4d(0.0f, 1.0f, 0.0f, 0.75f);
				glVertex3d(m_storage->vertex(k).x(), m_storage->vertex(k).y(), m_storage->vertex(k).z());
			}
			glEnd();
		}

		if (m_storage->numEdgePoints() > 0)
		{
			unsigned int numEdgePoints = m_storage->numEdgePoints();
			glBegin(GL_POINTS);
			for (unsigned int k = 0; k < numEdgePoints; k++)
			{
				glColor4d(0.8f, 0.8f, 0.0f, 0.75f);
				glVertex3d(m_storage->edgePoint(k).x(), m_storage->edgePoint(k).y(), m_storage->edgePoint(k).z());
			}
			glEnd();
		}
		glPopMatrix();
		glPointSize(1.0f);
	}

	PQP_Model* m_pqp_tree = m_computation->getPqpModel();

	if (m_pqp_tree)
	{
		Vector3 newTr = this->getCachedPosition();
		Quaternion newRot = this->getCachedOrientation();

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

			std::vector<Vector3> triangleObbPoints;

			glPushMatrix();
			glPushAttrib(GL_ENABLE_BIT);
			glEnable(GL_COLOR_MATERIAL);

			{
				Vector3 triangle_obb_center(m_pqp_tree->b->To[0], m_pqp_tree->b->To[1], m_pqp_tree->b->To[2]);
				Vector3 triangle_obb_extents(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]);

				triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
				triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));
				triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
				triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));
				triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
				triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));
				triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
				triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));

				glPointSize(1.0f);
				glBegin(GL_POINTS);
				for (unsigned int k = 0; k < triangleObbPoints.size(); k++)
				{
					glColor4d(1.0, 0.2, 0.2, 0.75);
					glVertex3d(triangleObbPoints[k].x(), triangleObbPoints[k].y(), triangleObbPoints[k].z());
				}
				glEnd();
				glPointSize(1.0f);
			}
#define POINTHULLMODEL_SHOW_OBB_CORNER_DATA
#ifdef POINTHULLMODEL_SHOW_OBB_CORNER_DATA
			{
				Mat<4, 4, GLfloat> modelviewM;
				const sofa::defaulttype::BoundingBox& bbox = this->getContext()->f_bbox.getValue();
				float scale = (float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.001f);

				for (unsigned int k = 0; k < triangleObbPoints.size(); k++)
				{

					sofa::defaulttype::Vec3f center = triangleObbPoints[k];

					std::ostringstream oss;
					oss << "Corner " << k << ": " << center.x() << "," << center.y() << "," << center.z();
					std::string tmp = oss.str();
					const char* s = tmp.c_str();
					glPushMatrix();

					glTranslatef(center[0], center[1], center[2]);
					glScalef(scale, scale, scale);

					// Makes text always face the viewer by removing the scene rotation
					// get the current modelview matrix
					glGetFloatv(GL_MODELVIEW_MATRIX, modelviewM.ptr());
					modelviewM.transpose();

					sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

					//glLoadMatrixf(modelview);
					glLoadIdentity();

					glTranslatef(temp[0], temp[1], temp[2]);
					glScalef(scale, scale, scale);

					while (*s)
					{
						glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
						s++;
					}

					glPopMatrix();
				}

				for (unsigned int k = 0; k < m_pqp_tree->num_bvs; k++)
				{
					BV* obb_k = m_pqp_tree->child(k);
					Vector3 triangle_obb_center(obb_k->To[0], obb_k->To[1], obb_k->To[2]);
					Vector3 triangle_obb_extents(obb_k->d[0], obb_k->d[1], obb_k->d[2]);

					std::vector<Vector3> triangleObbPoints;

					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
					triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));

					glPointSize(10.0f);
					glBegin(GL_POINTS);
					for (unsigned int k = 0; k < triangleObbPoints.size(); k++)
					{
						glColor4d(1.0, 0.2, 0.2, 0.75);
						glVertex3d(triangleObbPoints[k].x(), triangleObbPoints[k].y(), triangleObbPoints[k].z());
					}
					glEnd();
					glPointSize(1.0f);

					Mat<4, 4, GLfloat> modelviewM;
					const sofa::defaulttype::BoundingBox& bbox = this->getContext()->f_bbox.getValue();
					float scale = (float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.001f);

					for (unsigned int k = 0; k < triangleObbPoints.size(); k++)
					{

						sofa::defaulttype::Vec3f center = triangleObbPoints[k];

						std::ostringstream oss;
						oss << "Corner " << k << ": " << center.x() << "," << center.y() << "," << center.z();
						std::string tmp = oss.str();
						const char* s = tmp.c_str();
						glPushMatrix();

						glTranslatef(center[0], center[1], center[2]);
						glScalef(scale, scale, scale);

						// Makes text always face the viewer by removing the scene rotation
						// get the current modelview matrix
						glGetFloatv(GL_MODELVIEW_MATRIX, modelviewM.ptr());
						modelviewM.transpose();

						sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

						//glLoadMatrixf(modelview);
						glLoadIdentity();

						glTranslatef(temp[0], temp[1], temp[2]);
						glScalef(scale, scale, scale);

						while (*s)
						{
							glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
							s++;
						}

						glPopMatrix();
					}
				}
			}
#endif
			Vec4f colour(1, 0, 0, 1);
			Vec4f colour2(0, 0, 1, 1);

			//std::cout << " translate to obj. coord. = " << newTr << std::endl;
			glTranslated(newTr.x(), newTr.y(), newTr.z());

			BVHDrawHelpers::drawCoordinateMarkerGL(0.5f, 4.0f, colour, colour * 0.5, colour * 0.25);

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


			//std::cout << " rotate to OBB orientation = " << glOrientation << std::endl;
			BVHDrawHelpers::drawCoordinateMarkerGL(0.75f, 4.0f, colour2, colour2, colour);

			glMultMatrixd(glOrientation.ptr());

			{
				Vector3 triangle_obb_center(m_pqp_tree->b->To[0], m_pqp_tree->b->To[1], m_pqp_tree->b->To[2]);
				Vector3 triangle_obb_extents(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]);

				std::vector<Vector3> triangleObbPoints;

				triangleObbPoints.push_back(/*triangle_obb_center + */ Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z()));
				triangleObbPoints.push_back(/*triangle_obb_center + */ Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z()));
				triangleObbPoints.push_back(/*triangle_obb_center + */ Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z()));
				triangleObbPoints.push_back(/*triangle_obb_center + */ Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z()));

				triangleObbPoints.push_back(/*triangle_obb_center + */ Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z()));
				triangleObbPoints.push_back(/*triangle_obb_center + */ Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z()));
				triangleObbPoints.push_back(/*triangle_obb_center + */ Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z()));
				triangleObbPoints.push_back(/*triangle_obb_center + */ Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z()));

				glPointSize(2.0f);
				glBegin(GL_POINTS);
				for (unsigned int k = 0; k < triangleObbPoints.size(); k++)
				{
					glColor4d(0.2, 1.0, 0.2, 0.75);
					glVertex3d(triangleObbPoints[k].x(), triangleObbPoints[k].y(), triangleObbPoints[k].z());
				}
				glEnd();
				glPointSize(1.0f);
			}

			BVHDrawHelpers::drawCoordinateMarkerGL(1.0f, 6.0f, colour, colour2, colour2);

			BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(m_pqp_tree->b->d[0], m_pqp_tree->b->d[1], m_pqp_tree->b->d[2]), colour);

			float extent_x = m_pqp_tree->b->d[0]; float extent_y = m_pqp_tree->b->d[1]; float extent_z = m_pqp_tree->b->d[2];
			if (m_pqp_tree->b->min_dimension == 0)
				extent_x = m_pqp_tree->b->min_dimension_val;
			else if (m_pqp_tree->b->min_dimension == 1)
				extent_y = m_pqp_tree->b->min_dimension_val;
			else if (m_pqp_tree->b->min_dimension == 2)
				extent_z = m_pqp_tree->b->min_dimension_val;

			BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(extent_x, extent_y, extent_z), Vec4f(0, 1, 0, 1), true);

			glMultMatrixd(glOrientation.transposed().ptr());

			glTranslated(-m_pqp_tree->b->To[0], -m_pqp_tree->b->To[1], -m_pqp_tree->b->To[2]);

			if (m_pqp_tree->num_bvs > 2)
			{
				BVHDrawHelpers::drawRec(m_pqp_tree, m_pqp_tree->child(m_pqp_tree->b->first_child), NULL, colour2, 1, false);
				BVHDrawHelpers::drawRec(m_pqp_tree, m_pqp_tree->child(m_pqp_tree->b->first_child + 1), NULL, colour2, 1, false);
			}

			const PointHullComputation::FacetDiagonalsMap& facetDiagonals = this->m_computation->getFacetDiagonals();
			const PointHullComputation::FacetBordersMap& facetLongestSides = this->m_computation->getFacetLongestSides();
			const PointHullComputation::FacetBordersMap& facetBroadestSides = this->m_computation->getFacetBroadestSides();
			//std::cout << "FacetDiagonals draw: " << facetDiagonals.size() << std::endl;
			glLineWidth(2.0f);
			glBegin(GL_LINES);
			for (PointHullComputation::FacetDiagonalsMap::const_iterator diag_it = facetDiagonals.begin(); diag_it != facetDiagonals.end(); diag_it++)
			{
				//std::cout << " * " << diag_it->second.first << " -- " << diag_it->second.second << std::endl;
				glColor4d(1, 1, 1, 0.75);
				glVertex3d(diag_it->second.first.x(), diag_it->second.first.y(), diag_it->second.first.z());
				glColor4d(1, 1, 1, 0.75);
				glVertex3d(diag_it->second.second.x(), diag_it->second.second.y(), diag_it->second.second.z());
			}
			glEnd();

			glBegin(GL_LINES);
			for (PointHullComputation::FacetBordersMap::const_iterator diag_it = facetLongestSides.begin(); diag_it != facetLongestSides.end(); diag_it++)
			{
				//std::cout << " * " << diag_it->second.first << " -- " << diag_it->second.second << std::endl;
				glColor4d(0, 0.8, 0, 0.75);
				glVertex3d(diag_it->second.first.x(), diag_it->second.first.y(), diag_it->second.first.z());
				glColor4d(1, 1, 1, 0.75);
				glVertex3d(diag_it->second.second.x(), diag_it->second.second.y(), diag_it->second.second.z());
			}
			glEnd();

			glBegin(GL_LINES);
			for (PointHullComputation::FacetBordersMap::const_iterator diag_it = facetBroadestSides.begin(); diag_it != facetBroadestSides.end(); diag_it++)
			{
				//std::cout << " * " << diag_it->second.first << " -- " << diag_it->second.second << std::endl;
				glColor4d(0, 0, 0.8, 0.75);
				glVertex3d(diag_it->second.first.x(), diag_it->second.first.y(), diag_it->second.first.z());
				glColor4d(1, 1, 1, 0.75);
				glVertex3d(diag_it->second.second.x(), diag_it->second.second.y(), diag_it->second.second.z());
			}
			glEnd();

			glLineWidth(1.0f);

			glPointSize(6.0f);
			glBegin(GL_POINTS);
			/*for (unsigned int k = 0; k < m_storage->numSurfacePoints(); k++)
			{
			const Vector3& sf_pt = m_storage->surfacePoint(k);
			glColor4d(0.3, 0.9, 0.3, 0.75);
			glVertex3d(sf_pt.x(), sf_pt.y(), sf_pt.z());
			}*/
			//std::cout << " draw m_storage->numFacets() = " << m_storage->numFacets() << std::endl;
			for (unsigned long k = 0; k < m_storage->numFacets(); k++)
			{
				const std::pair<int, int>& grid_size = m_storage->getSurfaceGridSize(k);
				//std::cout << "  - facet " << k << " grid_size = " << grid_size.first << " x " << grid_size.second << std::endl;
				if (grid_size.first > 0 && grid_size.second > 0)
				{
					for (int l = 0; l < grid_size.first; l++)
					{
						for (int m = 0; m < grid_size.second; m++)
						{
							SurfacePointType pt_type = m_storage->getSurfacePointType(k, l, m);
							Vector3 surface_pt = m_storage->getSurfacePoint(k, l, m);

							//std::cout << "    - surface point " << l << "," << m << " type = " << pt_type << ", coords = " << surface_pt << std::endl;
							if (pt_type != UNKNOWN_SURFACE)
							{
								if (pt_type == INSIDE_SURFACE)
									glColor4d(0.2, 0.8, 0.2, 0.75);
								else if (pt_type == EDGE_OF_SURFACE)
									glColor4d(0.2, 0.8, 0.8, 0.75);
								else if (pt_type == OUTSIDE_SURFACE)
									glColor4d(0.8, 0.2, 0.2, 0.75);

								glVertex3d(surface_pt.x(), surface_pt.y(), surface_pt.z());
							}
						}
					}
				}
			}

			glEnd();
			glPointSize(1.0f);

			glPopAttrib();
			glPopMatrix();
		}
	}
#else
	const SurfaceGridMap& surfaceGrid = m_storage->getSurfaceGridData();
	if (surfaceGrid.size() > 0)
	{
		Vector3 model_position = this->getCachedPosition();
		Matrix4 model_orientation;
		this->getCachedOrientation().writeOpenGlMatrix(model_orientation.ptr());

		glPointSize(4.0f);
		glPushMatrix();

		glTranslated(model_position.x(), model_position.y(), model_position.z());
		glMultMatrixd(model_orientation.ptr());

		for (SurfaceGridMap::const_iterator it = surfaceGrid.begin(); it != surfaceGrid.end(); it++)
		{
			glBegin(GL_POINTS);
			for (VerticesMap::const_iterator vt_it = it->second.m_vertices.begin(); vt_it != it->second.m_vertices.end(); vt_it++)
			{
				glColor4d(0.0f, 1.0f, 0.0f, 0.75f);
				glVertex3d(vt_it->second.x(), vt_it->second.y(), vt_it->second.z());
			}
			glEnd();

			glBegin(GL_POINTS);
			for (EdgePointsMap::const_iterator edge_it = it->second.m_edgePoints.begin(); edge_it != it->second.m_edgePoints.end(); edge_it++)
			{
				for (unsigned int k = 0; k < edge_it->second.size(); k++)
				{
					glColor4d(1.0f, 1.0f, 0.0f, 0.75f);
					glVertex3d(edge_it->second[k].x(), edge_it->second[k].y(), edge_it->second[k].z());
				}
			}
			glEnd();

			/*glBegin(GL_POINTS);
			unsigned int rowIdx = 0;
			for (SurfaceGridPointMap::const_iterator sf_it = it->second.m_surfacePoints.begin(); sf_it != it->second.m_surfacePoints.end(); sf_it++)
			{
				const std::vector<SurfacePointType>& sf_pt_types = it->second.m_surfacePointTypes.at(rowIdx);
				for (unsigned int k = 0; k < sf_it->second.size(); k++)
				{
					//std::cout << "   surface point type " << rowIdx << "," << k << " = " << sf_pt_types.at(k) << std::endl;
					
					if (sf_pt_types[k] == INSIDE_SURFACE)
						glColor4d(0.2f, 1.0f, 0.2f, 0.75f);
					else if (sf_pt_types[k] == EDGE_OF_SURFACE)
						glColor4d(0.2f, 1.0f, 1.0f, 0.75f);
					else if (sf_pt_types[k] == OUTSIDE_SURFACE)
						glColor4d(1.0f, 0.2f, 0.2f, 0.75f);
					else
						glColor4d(0.0f, 0.0f, 1.0f, 0.75f);
					
					glVertex3d(sf_it->second[k].x(), sf_it->second[k].y(), sf_it->second[k].z());
				}
				rowIdx++;
			}
			glEnd();*/
		}

		glLineWidth(3.0f);
		glPointSize(5.0f);
		const std::map<unsigned long, Triangle3d>& surfaceTriangles = m_computation->getSurfaceTriangles();
		
		std::cout << "=== Triangle surface points ===" << std::endl;
		for (std::map<unsigned long, Triangle3d>::const_iterator it = surfaceTriangles.begin(); it != surfaceTriangles.end(); it++)
		{
			if (surfaceGrid.find(it->first) != surfaceGrid.end())
			{
				const SurfaceGrid& triangle_grid = surfaceGrid.find(it->first)->second;
				std::cout << " - triangle " << it->first << ": " << " vertices = " << it->second.V[0] << "/" << it->second.V[1] << "/" << it->second.V[2] << ", Grid size = " << triangle_grid.m_sizeX << " x " << triangle_grid.m_sizeY << std::endl;
				for (unsigned int p = 0; p < triangle_grid.m_sizeX; p++)
				{
					for (unsigned int q = 0; q < triangle_grid.m_sizeY; q++)
					{
						Vector3 point = triangle_grid.m_surfacePoints.at(p)[q];
						std::cout << "   * " << p << "," << q << ": " << point << std::endl;
					}
				}
			}
		}

		MechanicalObject<Rigid3Types>* mechanicalObject = dynamic_cast<MechanicalObject<Rigid3Types>*>(m_storage->getObjectMechanicalState());
		Rigid3dTypes::VecCoord coords = mechanicalObject->getPosition();

		const SurfaceGridMap& surfaceGridData = m_storage->getSurfaceGridData();

		Vector3 initialPosition(coords[0][0], coords[0][1], coords[0][2]);
		Quat initialOrientation(coords[0][3], coords[0][4], coords[0][5], coords[0][6]);

        /*typename*/ core::behavior::MechanicalState<Vec3Types>::ReadVecCoord pos = m_mechanicalObject->readPositions();

		for (std::map<unsigned long, Triangle3d>::const_iterator it = surfaceTriangles.begin(); it != surfaceTriangles.end(); it++)
		{
			std::cout << " triangle " << it->first << " (" << it->second.V[0] << "," << it->second.V[1] << "," << it->second.V[2] << ")" << std::endl;

			sofa::core::topology::BaseMeshTopology::Triangle tri = this->getMeshTopology()->getTriangles()[it->first];
            Vector3 tri_vert0 = pos[tri[0]];
            Vector3 tri_vert1 = pos[tri[1]];
            Vector3 tri_vert2 = pos[tri[2]];

			tri_vert0 -= initialPosition;
			tri_vert1 -= initialPosition;
			tri_vert2 -= initialPosition;

			tri_vert0 = initialOrientation.inverseRotate(tri_vert0);
			tri_vert1 = initialOrientation.inverseRotate(tri_vert1);
			tri_vert2 = initialOrientation.inverseRotate(tri_vert2);

			const SurfaceGrid& tri_grid = surfaceGridData.at(it->first);

#if 1
			bool vert0_in = PointIsInTriangle(tri_vert0, tri_vert0, tri_vert1, tri_vert2);
			bool vert1_in = PointIsInTriangle(tri_vert1, tri_vert0, tri_vert1, tri_vert2);
			bool vert2_in = PointIsInTriangle(tri_vert2, tri_vert0, tri_vert1, tri_vert2);

			Vector3 tri_center = (tri_vert0 + tri_vert1 + tri_vert2) / 3.0f;
			bool center_in = PointIsInTriangle(tri_center, tri_vert0, tri_vert1, tri_vert2);

			Vector3 pt_outside(-100, -100, -100);
			bool outside_test = PointIsInTriangle(pt_outside, tri_vert0, tri_vert1, tri_vert2);
			
			std::cout << " ==== VERTICES TEST ====" << std::endl;
			std::cout << "  tri_vert0 in triangle = " << vert0_in << std::endl;
			std::cout << "  tri_vert1 in triangle = " << vert1_in << std::endl;
			std::cout << "  tri_vert2 in triangle = " << vert2_in << std::endl;
			std::cout << "  center " << tri_center << " in triangle =    " << center_in << std::endl;
			std::cout << "  outside pt. " << pt_outside << " in triangle =    " << outside_test << std::endl;
			

			glPointSize(10.0f);
			glBegin(GL_POINTS);
			if (vert0_in)
				glColor4d(0.2, 0.9, 0.2, 0.75);
			else
				glColor4d(0.9, 0.2, 0.2, 0.75);

			glVertex3d(tri_vert0.x(), tri_vert0.y(), tri_vert0.z());

			if (vert1_in)
				glColor4d(0.2, 0.9, 0.2, 0.75);
			else
				glColor4d(0.9, 0.2, 0.2, 0.75);

			glVertex3d(tri_vert1.x(), tri_vert1.y(), tri_vert1.z());

			if (vert2_in)
				glColor4d(0.2, 0.9, 0.2, 0.75);
			else
				glColor4d(0.9, 0.2, 0.2, 0.75);

			glVertex3d(tri_vert2.x(), tri_vert2.y(), tri_vert2.z());
			
			if (center_in)
				glColor4d(0.2, 0.9, 0.2, 0.75);
			else
				glColor4d(0.9, 0.2, 0.2, 0.75);

			glVertex3d(tri_center.x(), tri_center.y(), tri_center.z());

			Vector3 tr_offsetX = /*initialOrientation.inverseRotate*/(tri_grid.m_offsetX);
			Vector3 tr_offsetY = /*initialOrientation.inverseRotate*/(tri_grid.m_offsetY);
			std::cout << " tr_offsetX = " << tr_offsetX << ", tr_offsetY = " << tr_offsetY << std::endl;

			Vector3 tri_center_xy = tri_center + tr_offsetX + tr_offsetY;
			bool center_in_xy = PointIsInTriangle(tri_center_xy, tri_vert0, tri_vert1, tri_vert2);

			if (center_in_xy)
				glColor4d(0.2, 0.9, 0.2, 0.75);
			else
				glColor4d(0.9, 0.2, 0.2, 0.75);

			glVertex3d(tri_center_xy.x(), tri_center_xy.y(), tri_center_xy.z());

			glEnd();
			glPointSize(1.0f);

			for (unsigned int u = 0; u <= tri_grid.m_sizeX; u++)
			{
				for (unsigned int v = 0; v <= tri_grid.m_sizeY; v++)
				{
					Vector3 next_grid_pt = tri_grid.m_gridOrigin + (tr_offsetX * u) + (tr_offsetY * v);
					bool isInside = PointIsInTriangle(next_grid_pt, tri_vert0, tri_vert1, tri_vert2);

					std::cout << " next_grid_pt = " << next_grid_pt << " = " << tri_grid.m_gridOrigin << " + " << u << " * " << tr_offsetX << " + " << v << " * " << tr_offsetY << " isInside = " << isInside << std::endl;

					if (isInside)
						glColor4d(0.2, 0.9, 0.2, 0.75);
					else
						glColor4d(0.9, 0.2, 0.2, 0.75);
					
					glPointSize(10.0f);
					glBegin(GL_POINTS);
					glVertex3d(next_grid_pt.x(), next_grid_pt.y(), next_grid_pt.z());
					glEnd();
					glPointSize(1.0f);

					glLineWidth(5.0f);
					glBegin(GL_LINES);
					glColor4d(0.2, 0.9, 0.2, 0.75);
					glVertex3d(tri_grid.m_gridOrigin.x(), tri_grid.m_gridOrigin.y(), tri_grid.m_gridOrigin.z());
					glColor4d(0.2, 0.9, 0.2, 0.75);
					glVertex3d(tri_grid.m_gridOrigin.x() + (tr_offsetX * u).x(), tri_grid.m_gridOrigin.y() + (tr_offsetX * u).y(), tri_grid.m_gridOrigin.z() + (tr_offsetX * u).z());

					glColor4d(0.2, 0.9, 0.2, 0.75);
					glVertex3d(tri_grid.m_gridOrigin.x() + (tr_offsetX * u).x(), tri_grid.m_gridOrigin.y() + (tr_offsetX * u).y(), tri_grid.m_gridOrigin.z() + (tr_offsetX * u).z());
					glColor4d(0.2, 0.9, 0.2, 0.75);
					glVertex3d(tri_grid.m_gridOrigin.x() + (tr_offsetX * u).x() + (tr_offsetY * v).x(), tri_grid.m_gridOrigin.y() + (tr_offsetX * u).y() + (tr_offsetY * v).y(), tri_grid.m_gridOrigin.z() + (tr_offsetX * u).z() + (tr_offsetY * v).z());

					glEnd();
					glLineWidth(1.0f);

				}
			}
			

			glBegin(GL_LINES);
			glColor4d(0.2, 0.9, 0.2, 0.75);
			glVertex3d(tri_center.x(), tri_center.y(), tri_center.z());
			glColor4d(0.2, 0.9, 0.2, 0.75);
			glVertex3d(tri_center.x() + tr_offsetX.x(), tri_center.y() + tr_offsetX.y(), tri_center.z() + tr_offsetX.z());

			glColor4d(0.2, 0.9, 0.2, 0.75);
			glVertex3d(tri_center.x() + tr_offsetX.x(), tri_center.y() + tr_offsetX.y(), tri_center.z() + tr_offsetX.z());
			glColor4d(0.2, 0.9, 0.2, 0.75);
			glVertex3d(tri_center.x() + tr_offsetX.x() + tr_offsetY.x(), tri_center.y() + tr_offsetX.y() + tr_offsetY.y(), tri_center.z() + tr_offsetX.z() + tr_offsetY.z());

			glEnd();

			std::cout << " ==== VERTICES TEST ====" << std::endl;
#endif

			
			std::cout << " === surface grid in-/outside test: " << tri_grid.m_sizeX << "x" << tri_grid.m_sizeY << " ===" << std::endl;
			for (unsigned int u = 0; u < tri_grid.m_sizeX; u++)
			{
				std::cout << "  ";
				for (unsigned int v = 0; v < tri_grid.m_sizeY; v++)
				{
					const Vector3& sf_pt = tri_grid.m_surfacePoints.at(u)[v];
					bool inside = PointIsInTriangle(sf_pt, tri_vert0, tri_vert1, tri_vert2);
					if (inside)
						std::cout << "1";
					else
						std::cout << "0";
				}
				std::cout << std::endl;
			}
			std::cout << " === surface grid in-/outside test: " << tri_grid.m_sizeX << "x" << tri_grid.m_sizeY << " ===" << std::endl;

			Plane3d trianglePlane(tri_vert0, tri_vert1, tri_vert2);
			Vector3 U0, U1;

			BVHModels::Mathd::GenerateComplementBasis(U0, U1, trianglePlane.Normal);
			U0.normalize(); U1.normalize(); trianglePlane.Normal.normalize();
			
			if (surfaceGrid.find(it->first) != surfaceGrid.end())
			{
				const SurfaceGrid& triangle_grid = surfaceGrid.find(it->first)->second;

				Vector3 V1mV0 = tri_vert1 - tri_vert0;
				Vector3 V2mV0 = tri_vert2 - tri_vert0;

				glColor4d(0.8, 0.4, 0.2, 0.75);
				glVertex3d(V1mV0.x(), V1mV0.y(), V1mV0.z());
				glColor4d(0.4, 0.8, 0.2, 0.75);
				glVertex3d(V2mV0.x(), V2mV0.y(), V2mV0.z());

				std::cout << " triangle " << it->first << " (" << it->second.V[0] << "," << it->second.V[1] << "," << it->second.V[2] << ") grid size: " << triangle_grid.m_sizeX << " x " << triangle_grid.m_sizeY << std::endl;

				glBegin(GL_POINTS);
				for (unsigned int p = 0; p < triangle_grid.m_sizeX; p++)
				{
					/*if (p % 2 != 0)
						continue;*/

					for (unsigned int q = 0; q < triangle_grid.m_sizeY; q++)
					{
						/*if (q % 2 != 0)
							continue;*/

						Vector3 point = triangle_grid.m_surfacePoints.at(p)[q];
						bool isInTri = PointIsInTriangle(point, tri_vert0, tri_vert1, tri_vert2);

						if (isInTri)
							glColor4d(0.2, 0.9, 0.2, 0.75);
						else
							glColor4d(0.9, 0.2, 0.2, 0.75);
						
						glVertex3d(point.x(), point.y(), point.z());

						std::cout << std::endl;
					}
				}
				glEnd();
			}

			Vector3 triangleCenter = (it->second.V[0] + it->second.V[1] + it->second.V[2]) / 3.0f;
			glBegin(GL_LINES);
			glColor4d(1.0, 0.0, 0.0, 0.75);
			glVertex3d(triangleCenter.x(), triangleCenter.y(), triangleCenter.z());
			glVertex3d(triangleCenter.x() + U0.x(), triangleCenter.y() + U0.y(), triangleCenter.z() + U0.z());

			glColor4d(0.0, 1.0, 0.0, 0.75);
			glVertex3d(triangleCenter.x(), triangleCenter.y(), triangleCenter.z());
			glVertex3d(triangleCenter.x() + U1.x(), triangleCenter.y() + U1.y(), triangleCenter.z() + U1.z());

			glColor4d(0.0, 0.0, 1.0, 0.75);
			glVertex3d(triangleCenter.x(), triangleCenter.y(), triangleCenter.z());
			glVertex3d(triangleCenter.x() + trianglePlane.Normal.x(), triangleCenter.y() + trianglePlane.Normal.y(), triangleCenter.z() + trianglePlane.Normal.z());

			glColor4d(0.2, 0.8, 0.2, 0.75);
			glVertex3d(it->second.V[0].x(), it->second.V[0].y(), it->second.V[0].z());
			glVertex3d(it->second.V[1].x(), it->second.V[1].y(), it->second.V[1].z());
			
			glColor4d(0.2, 0.8, 0.2, 0.75);
			glVertex3d(it->second.V[1].x(), it->second.V[1].y(), it->second.V[1].z());
			glVertex3d(it->second.V[2].x(), it->second.V[2].y(), it->second.V[2].z());

			glColor4d(0.2, 0.8, 0.2, 0.75);
			glVertex3d(it->second.V[2].x(), it->second.V[2].y(), it->second.V[2].z());
			glVertex3d(it->second.V[0].x(), it->second.V[0].y(), it->second.V[0].z());
			glEnd();
		}
		glLineWidth(1.0f);
		glPointSize(1.0f);

		glPopMatrix();
	}
#endif
}
