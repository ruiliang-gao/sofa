#include "PBDRigidBodyGeometry.h"

#include <sofa/helper/logging/Messaging.h>

using namespace sofa::simulation::PBDSimulation;

PBDRigidBodyGeometry::PBDRigidBodyGeometry() :
    m_mesh()
{
}

PBDRigidBodyGeometry::~PBDRigidBodyGeometry(void)
{
    m_mesh.release();
}

PBDRigidBodyGeometry::Mesh &PBDRigidBodyGeometry::getMesh()
{
    return m_mesh;
}

const PBDRigidBodyGeometry::Mesh &PBDRigidBodyGeometry::getMesh() const
{
    return m_mesh;
}

void PBDRigidBodyGeometry::initMesh(const unsigned int nVertices, const unsigned int nFaces, const Vector3r *vertices, const unsigned int* indices, const Mesh::UVIndices& uvIndices, const Mesh::UVs& uvs, const Vector3r &scale)
{
    msg_info("PBDRigidBodyGeometry") << "initMesh() -- nVertices = " << nVertices << ", nFaces = " << nFaces << ", scale = " << scale;

    for (unsigned int k = 0;  k < nVertices; k++)
        msg_info("PBDRigidBodyGeometry") << "Vertex " << k << ": (" << vertices[k][0] << "," << vertices[k][1] << "," << vertices[k][2] << ")";

    m_mesh.release();
    m_mesh.initMesh(nVertices, nFaces * 2, nFaces);
    m_vertexData_local.resize(nVertices);
    m_vertexData.resize(nVertices);
    for (unsigned int i = 0; i < nVertices; i++)
    {
        m_vertexData_local.getPosition(i) = vertices[i].cwiseProduct(scale);
        m_vertexData.getPosition(i) = m_vertexData_local.getPosition(i);

         msg_info("PBDRigidBodyGeometry") << "Vertex " << i << " after applying scale: (" << m_vertexData.getPosition(i)[0] << "," << m_vertexData.getPosition(i)[1] << "," << m_vertexData.getPosition(i)[2] << ")";
    }

    for (unsigned int i = 0; i < nFaces; i++)
    {
        msg_info("PBDRigidBodyGeometry") << "Adding face: " << indices[3 * i] << "," << indices[3 * i + 1] << "," << indices[3 * i + 2];
        m_mesh.addFace(&indices[3 * i]);
    }
    m_mesh.copyUVs(uvIndices, uvs);
    m_mesh.buildNeighbors();
    updateMeshNormals(m_vertexData);

    const Utilities::PBDIndexedFaceMesh::Faces& meshFaces = m_mesh.getFaces();
    const Utilities::PBDIndexedFaceMesh::Edges& meshEdges = m_mesh.getEdges();
    const Utilities::PBDIndexedFaceMesh::FaceData& meshFaceData = m_mesh.getFaceData();

    msg_info("PBDRigidBodyGeometry") << "=======================================";
    msg_info("PBDRigidBodyGeometry") << "Mesh structure data for PBDRigidBody   ";
    msg_info("PBDRigidBodyGeometry") << "=======================================";

    msg_info("PBDRigidBodyGeometry") << "Vertex count after mesh init: " << m_vertexData.size();
    for (unsigned int k = 0; k < m_vertexData.size(); k++)
        msg_info("PBDRigidBodyGeometry") << m_vertexData.getPosition(k)[0] << "," << m_vertexData.getPosition(k)[1] << "," << m_vertexData.getPosition(k)[2];

    msg_info("PBDRigidBodyGeometry") << "Local vertex count after mesh init: " << m_vertexData_local.size();
    for (unsigned int k = 0; k < m_vertexData_local.size(); k++)
        msg_info("PBDRigidBodyGeometry") << m_vertexData_local.getPosition(k)[0] << "," << m_vertexData_local.getPosition(k)[1] << "," << m_vertexData_local.getPosition(k)[2];

    msg_info("PBDRigidBodyGeometry") << "Mesh indices count: " << meshFaces.size();
    for (unsigned int k = 0; k < meshFaces.size(); k++)
        msg_info("PBDRigidBodyGeometry") << "Index " << k << ": " << meshFaces.at(k);

    msg_info("PBDRigidBodyGeometry") << "Size of PBD triangle mesh (number of faces): " << meshFaces.size();

    for (unsigned int k = 0; k < meshFaceData.size(); k++)
    {
        const Utilities::PBDIndexedFaceMesh::Face& f = meshFaceData[k];
        for (unsigned int l = 0; l < this->getMesh().getNumVerticesPerFace(); l++)
        {
            const Utilities::PBDIndexedFaceMesh::Edge& e = meshEdges[f.m_edges[l]];
            msg_info("PBDRigidBodyGeometry") << "Face " << k << " edge " << l << " vertices: " << e.m_vert[0] << " -- " << e.m_vert[1];
        }
    }

    msg_info("PBDRigidBodyGeometry") << "Size of PBD triangle mesh (number of edges): " << meshEdges.size();

    for (unsigned int k = 0; k < meshFaceData.size(); k++)
    {
        const Utilities::PBDIndexedFaceMesh::Face& f = meshFaceData[k];
        for (unsigned int l = 0; l < this->getMesh().getNumVerticesPerFace(); l++)
        {
            // const Utilities::PBDIndexedFaceMesh::Edge& e = meshEdges[f.m_edges[l]];
            msg_info("PBDRigidBodyGeometry") << "Face " << k << " edge " << l << ": Index " << f.m_edges[l];
        }
    }

    msg_info("PBDRigidBodyGeometry") << "=======================================";
}

void PBDRigidBodyGeometry::updateMeshNormals(const PBDVertexData &vd)
{
    m_mesh.updateNormals(vd, 0);
    m_mesh.updateVertexNormals(vd);
}

void PBDRigidBodyGeometry::updateMeshTransformation(const Vector3r &x, const Matrix3r &R)
{
    for (unsigned int i = 0; i < m_vertexData_local.size(); i++)
    {
        m_vertexData.getPosition(i) = R * m_vertexData_local.getPosition(i) + x;
    }
    updateMeshNormals(m_vertexData);
}

PBDVertexData & PBDRigidBodyGeometry::getVertexData()
{
    return m_vertexData;
}

const PBDVertexData& PBDRigidBodyGeometry::getVertexData() const
{
    return m_vertexData;
}

PBDVertexData& PBDRigidBodyGeometry::getVertexDataLocal()
{
    return m_vertexData_local;
}

const PBDVertexData& PBDRigidBodyGeometry::getVertexDataLocal() const
{
    return m_vertexData_local;
}
