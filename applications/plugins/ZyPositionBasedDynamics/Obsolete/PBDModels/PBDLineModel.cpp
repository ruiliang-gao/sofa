#include "PBDLineModel.h"

#include "PBDynamics/PositionBasedRigidBodyDynamics.h"
#include "PBDynamics/PositionBasedDynamics.h"
#include "PBDTriangleModel.h"

using namespace sofa::simulation::PBDSimulation;

PBDLineModel::PBDLineModel(): m_indexOffset(0), m_indexOffsetQuaternions(0), m_nPoints(0), m_nQuaternions(0)
{
    m_restitutionCoeff = static_cast<Real>(0.6);
    m_frictionCoeff = static_cast<Real>(0.2);
}

PBDLineModel::~PBDLineModel(void)
{

}

PBDLineModel::Edges& PBDLineModel::getEdges()
{
    return m_edges;
}

void PBDLineModel::initMesh(const unsigned int nPoints, const unsigned int nQuaternions, const unsigned int indexOffset, const unsigned int indexOffsetQuaternions, unsigned int* indices, unsigned int* indicesQuaternions)
{
    msg_info("PBDLineModel") << "initMes() - nPoints = " << nPoints << ", nQuaternions = " << nQuaternions << ", indexOffset = " << indexOffset << ", indexOffsetQuaternions = " << indexOffsetQuaternions;
    m_nPoints = nPoints;
    m_nQuaternions = nQuaternions;
    m_indexOffset = indexOffset;
    m_indexOffsetQuaternions = indexOffsetQuaternions;

    msg_info("PBDLineModel") << "Resizing edges vector to: " << (nPoints - 1);
    m_edges.resize(nPoints - 1);

    for (unsigned int i = 0; i < nPoints - 1; i++)
    {
        msg_info("PBDLineModel") << "Adding edge: Particle indices = " << indices[2*i] << " - " << indices[2*i + 1] << ", quaternion index = " << indicesQuaternions[i];
        m_edges[i] = OrientedEdge(indices[2*i], indices[2*i + 1], indicesQuaternions[i]);
    }
}

unsigned int PBDLineModel::getIndexOffset() const
{
    return m_indexOffset;
}

unsigned PBDLineModel::getIndexOffsetQuaternions() const
{
    return m_indexOffsetQuaternions;
}

unsigned int PBDLineModel::getNumPoints() const
{
    return m_nPoints;
}
