#include "PBDTriangleModel.h"

//#include "PositionBasedDynamics/PositionBasedRigidBodyDynamics.h"
//#include "PositionBasedDynamics/PositionBasedDynamics.h"

using namespace sofa::simulation::PBDSimulation;

PBDTriangleModel::PBDTriangleModel() :
    m_particleMesh()
{
    m_restitutionCoeff = static_cast<Real>(0.6);
    m_frictionCoeff = static_cast<Real>(0.2);
}

PBDTriangleModel::~PBDTriangleModel(void)
{
    cleanupModel();
}

void PBDTriangleModel::cleanupModel()
{
    m_particleMesh.release();
}

void PBDTriangleModel::updateMeshNormals(const PBDParticleData &pd)
{
    m_particleMesh.updateNormals(pd, m_indexOffset);
    m_particleMesh.updateVertexNormals(pd);
}

PBDTriangleModel::ParticleMesh &PBDTriangleModel::getParticleMesh()
{
    return m_particleMesh;
}

void PBDTriangleModel::initMesh(const unsigned int nPoints, const unsigned int nFaces, const unsigned int indexOffset, unsigned int* indices, const ParticleMesh::UVIndices& uvIndices, const ParticleMesh::UVs& uvs)
{
    m_indexOffset = indexOffset;
    m_particleMesh.release();

    m_particleMesh.initMesh(nPoints, nFaces * 2, nFaces);

    for (unsigned int i = 0; i < nFaces; i++)
    {
        m_particleMesh.addFace(&indices[3 * i]);
    }
    m_particleMesh.copyUVs(uvIndices, uvs);
    m_particleMesh.buildNeighbors();
}

unsigned int PBDTriangleModel::getIndexOffset() const
{
    return m_indexOffset;
}

