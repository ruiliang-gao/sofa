#include "PointHull_Container.h"

using namespace sofa::component::collision;

PointHullStorage::PointHullStorage(sofa::core::behavior::MechanicalState<Vec3Types>* state, sofa::core::behavior::BaseMechanicalState* objectMState, sofa::core::topology::BaseMeshTopology* topology) : m_state(state), m_topology(topology), m_objectState(objectMState)
{

}

sofa::core::behavior::BaseMechanicalState* PointHullStorage::getObjectMechanicalState()
{
	return m_objectState;
}

sofa::core::behavior::MechanicalState<Vec3Types>* PointHullStorage::getMechanicalState()
{
	return m_state;
}

sofa::core::topology::BaseMeshTopology* PointHullStorage::getTopology()
{
	return m_topology;
}

PointHullStorage_Full::PointHullStorage_Full(sofa::core::behavior::MechanicalState<Vec3Types>* state, sofa::core::behavior::BaseMechanicalState* objectMState, sofa::core::topology::BaseMeshTopology* topology) : PointHullStorage(state, objectMState, topology)
{
	
}

unsigned long PointHullStorage_Full::numFacets()
{
	return m_surfacePointsCountPerFacet.size();
}

bool PointHullStorage_Full::addVertex(const Vector3& point, const unsigned long& pointIdx, const unsigned long& facetIdx)
{
	if (std::find(m_vertices.begin(), m_vertices.end(), point) == m_vertices.end())
	{
		m_vertices.push_back(point);
		m_vertexFacetIndices.insert(std::make_pair(m_vertices.size() - 1, std::list<unsigned long>()));
		m_vertexFacetIndices[m_vertices.size() - 1].push_back(facetIdx);
		m_vertexIndices.insert(std::make_pair(m_vertices.size() - 1, pointIdx));
		m_vertexRegistryTimeAdjacency.insert(std::make_pair(point, m_vertices.size() - 1));

		if (m_surfaceGrids.find(facetIdx) == m_surfaceGrids.end())
		{
			std::stringstream idStr;
			idStr << "SurfaceGrid " << facetIdx;
			m_surfaceGrids.insert(std::make_pair(facetIdx, SurfaceGrid(idStr.str())));
		}
		m_surfaceGrids[facetIdx].m_vertices[pointIdx] = point;

		return true;
	}
	else
	{
		std::list<unsigned long>& adjFacets = m_vertexFacetIndices[m_vertexRegistryTimeAdjacency[point]];
		if (std::find(adjFacets.begin(), adjFacets.end(), facetIdx) == adjFacets.end())
		{
			// std::cout << " vertex " << point << " already known, but not with facet index " << facetIdx << ", registering." << std::endl;
			adjFacets.push_back(facetIdx);
		}
		return false;
	}
}

unsigned long PointHullStorage_Full::numVertices() const 
{ 
	return m_vertices.size();
}

const Vector3& PointHullStorage_Full::vertex(const unsigned long& idx) const
{
	return m_vertices.at(idx);
}

Vector3& PointHullStorage_Full::vertex(const unsigned long& idx)
{
	return m_vertices[idx];
}

bool PointHullStorage_Full::addEdgePoint(const Vector3& point, const unsigned long& pointIdx, const unsigned long& edgeIdx, const unsigned long& facetIdx)
{
	std::cout << " addEdgePoint(" << point << ", index " << pointIdx << ", facet index " << facetIdx << ", edge index " << edgeIdx << ")" << std::endl;
	std::deque<Vector3>::iterator pPos = std::find(m_edgePoints.begin(), m_edgePoints.end(), point);
	if (pPos == m_edgePoints.end())
	{
		m_edgePoints.push_back(point);
		setEdgePointEdgeIndex(m_edgePoints.size() - 1, edgeIdx);
		setEdgePointFacetIndex(m_edgePoints.size() - 1, facetIdx);
		m_edgePointIndices.insert(std::make_pair(m_edgePoints.size() - 1, pointIdx));

		if (m_surfaceGrids.find(facetIdx) == m_surfaceGrids.end())
		{
			std::stringstream idStr;
			idStr << "SurfaceGrid " << facetIdx;
			m_surfaceGrids.insert(std::make_pair(facetIdx, SurfaceGrid(idStr.str())));
		}

		std::cout << "  m_surfaceGrids[" << facetIdx << "].m_edgePoints[" << edgeIdx << "].size() = " << m_surfaceGrids[facetIdx].m_edgePoints[edgeIdx].size() << std::endl;
		m_surfaceGrids[facetIdx].m_edgePoints[edgeIdx].push_back(point);

		return true;
	}
	return false;
}

unsigned long PointHullStorage_Full::numEdgePoints() const
{
	return m_edgePoints.size();
}

const Vector3& PointHullStorage_Full::edgePoint(const unsigned long& idx) const
{
	return m_edgePoints[idx];
}

Vector3& PointHullStorage_Full::edgePoint(const unsigned long& idx)
{
	return m_edgePoints[idx];
}

const std::list<unsigned long>& PointHullStorage_Full::edgePointFacetIndex(const unsigned long& idx)
{
	return m_edgePointFacetIndices[idx];
}

unsigned long PointHullStorage_Full::edgePointEdgeIndex(const unsigned long& idx)
{
	return m_edgePointEdgeIndices[idx];
}

void PointHullStorage_Full::setEdgePointFacetIndex(const unsigned long& idx, const unsigned long& value)
{
	if (m_edgePointFacetIndices.find(idx) == m_edgePointFacetIndices.end())
	{
		m_edgePointFacetIndices.insert(std::make_pair(idx, std::list<unsigned long>()));
		m_edgePointFacetIndices[idx].push_back(value);
	}
	else
	{
		if (std::find(m_edgePointFacetIndices[idx].begin(), m_edgePointFacetIndices[idx].end(), value) == m_edgePointFacetIndices[idx].end())
			m_edgePointFacetIndices[idx].push_back(value);
	}
}

void PointHullStorage_Full::setEdgePointEdgeIndex(const unsigned long& idx, const unsigned long& value)
{
	// std::cout << "     setEdgePointEdgeIndex(" << idx << "," << value << ")" << std::endl;
	if (m_edgePointEdgeIndices.find(idx) == m_edgePointEdgeIndices.end())
		m_edgePointEdgeIndices.insert(std::make_pair(idx, value));
	else
		m_edgePointEdgeIndices[idx] = value;
}


bool PointHullStorage_Full::addSurfacePoint(const Vector3& point, const unsigned int& x, const unsigned int& y, const unsigned long& facetIdx, const SurfacePointType& type) //const unsigned long& pointIdx, const unsigned long& facetIdx)
{
	if (std::find(m_clusterPoints.begin(), m_clusterPoints.end(), point) == m_clusterPoints.end())
	{
		m_clusterPoints.push_back(point);
		m_surfacePointsPerFacet.insert(std::make_pair(facetIdx, std::make_pair(std::make_pair(x,y), point)));
		m_surfacePointFacetIndices.insert(std::make_pair(facetIdx, std::make_pair(x, y)));
		m_surfacePointFacetTypes.insert(std::make_pair(facetIdx, std::make_pair(std::make_pair(x, y), type)));

		m_surfacePointsCountPerFacet[facetIdx] += 1;

		if (m_surfaceGrids.find(facetIdx) == m_surfaceGrids.end())
		{
			std::stringstream idStr;
			idStr << "SurfaceGrid " << facetIdx;
			m_surfaceGrids.insert(std::make_pair(facetIdx, SurfaceGrid(idStr.str())));
		}
		
		std::cout << " m_surfaceGrids[" << facetIdx << "].m_surfacePoints[" << x << "].size() = " << m_surfaceGrids[facetIdx].m_surfacePoints[x].size() << ", y = " << y;
		if (m_surfaceGrids[facetIdx].m_surfacePoints[x].size() <= y || m_surfaceGrids[facetIdx].m_surfacePoints[x].size() == 0)
		{
			m_surfaceGrids[facetIdx].m_surfacePoints[x].resize(y + 1);
			std::cout << ", resized = " << m_surfaceGrids[facetIdx].m_surfacePoints[x].size();
		}
		std::cout << std::endl;

		m_surfaceGrids[facetIdx].m_surfacePoints[x][y] = point;

		if (m_surfaceGrids[facetIdx].m_surfacePointTypes[x].size() <= y || m_surfaceGrids[facetIdx].m_surfacePointTypes[x].size() == 0)
		{
			m_surfaceGrids[facetIdx].m_surfacePointTypes[x].resize(y + 1);
		}

		m_surfaceGrids[facetIdx].m_surfacePointTypes[x][y] = type;

		//std::cout << " * Add surface point " <<  _clusterPoints.size() - 1 << " for facet " << facetIdx << " (count " << pointIdx << "): " << point << std::endl;
		//m_surfacePointFacetIndices.insert(std::make_pair(m_clusterPoints.size() - 1, facetIdx));
		//m_surfacePointIndices.insert(std::make_pair(m_clusterPoints.size() - 1, pointIdx));
		return true;
	}
	return false;
}

unsigned long PointHullStorage_Full::numSurfacePoints() const 
{ 
	return m_clusterPoints.size(); 
}

const Vector3& PointHullStorage_Full::surfacePoint(const unsigned long& idx) const
{
	return m_clusterPoints[idx];
}

Vector3& PointHullStorage_Full::surfacePoint(const unsigned long& idx)
{
	return m_clusterPoints[idx];
}

unsigned long PointHullStorage_Full::surfacePointIndex(const unsigned long& idx)
{
	return m_surfacePointIndices[idx];
}

/*unsigned long PointHullStorage_Full::surfacePointFacetIndex(const unsigned long& idx)
{
	return m_surfacePointFacetIndices[idx];
}

void PointHullStorage_Full::setSurfacePointFacetIndex(const unsigned long& idx, const unsigned long& value)
{
	if (m_surfacePointFacetIndices.find(idx) == m_surfacePointFacetIndices.end())
		m_surfacePointFacetIndices.insert(std::make_pair(idx, value));
	else
		m_surfacePointFacetIndices[idx] = value;
}*/

const std::pair<int, int>& PointHullStorage_Full::getSurfaceGridSize(const unsigned long& facetIdx)
{
	if (m_surfacePointGridSizes.find(facetIdx) != m_surfacePointGridSizes.end())
		return m_surfacePointGridSizes[facetIdx];

	return std::make_pair(-1, -1);
}

void PointHullStorage_Full::setSurfaceGridSize(const unsigned long& facetIdx, const int& x, const int& y)
{
	m_surfacePointGridSizes[facetIdx] = std::make_pair(x, y);
	
	if (m_surfaceGrids.find(facetIdx) == m_surfaceGrids.end())
	{
		m_surfaceGrids.insert(std::make_pair(facetIdx, SurfaceGrid()));
	}

	m_surfaceGrids[facetIdx].m_sizeX = x;
	m_surfaceGrids[facetIdx].m_sizeY = y;

	//if (m_surfaceGrids[facetIdx].m_gridID.empty())
	{
		std::stringstream idStr;
		idStr << "SurfaceGrid " << facetIdx << ": Size " << m_surfaceGrids[facetIdx].m_sizeX << " x " << m_surfaceGrids[facetIdx].m_sizeY << ", distances = " << m_surfaceGrids[facetIdx].m_distX << " x " << m_surfaceGrids[facetIdx].m_distY << ", origin = " << m_surfaceGrids[facetIdx].m_gridOrigin;
	
		m_surfaceGrids[facetIdx].m_gridID = idStr.str();
	}
}

void PointHullStorage_Full::setSurfaceGridDistance(const unsigned long& facetIdx, const double& size_x, const double& size_y)
{
	if (m_surfaceGrids.find(facetIdx) == m_surfaceGrids.end())
	{
		m_surfaceGrids.insert(std::make_pair(facetIdx, SurfaceGrid()));
	}

	m_surfaceGrids[facetIdx].m_distX = size_x;
	m_surfaceGrids[facetIdx].m_distY = size_y;

	//if (m_surfaceGrids[facetIdx].m_gridID.empty())
	{
		std::stringstream idStr;
		idStr << "SurfaceGrid " << facetIdx << ": Size " << m_surfaceGrids[facetIdx].m_sizeX << " x " << m_surfaceGrids[facetIdx].m_sizeY << ", distances = " << m_surfaceGrids[facetIdx].m_distX << " x " << m_surfaceGrids[facetIdx].m_distY << ", origin = " << m_surfaceGrids[facetIdx].m_gridOrigin;
	
		m_surfaceGrids[facetIdx].m_gridID = idStr.str();
	}
}

void PointHullStorage_Full::setSurfaceGridOffset(const unsigned long& facetIdx, const Vector3& offset_x, const Vector3& offset_y)
{
	if (m_surfaceGrids.find(facetIdx) == m_surfaceGrids.end())
	{
		m_surfaceGrids.insert(std::make_pair(facetIdx, SurfaceGrid()));
	}

	m_surfaceGrids[facetIdx].m_offsetX = offset_x;
	m_surfaceGrids[facetIdx].m_offsetY = offset_y;

	//if (m_surfaceGrids[facetIdx].m_gridID.empty())
	{
		std::stringstream idStr;
		idStr << "SurfaceGrid " << facetIdx << ": Size " << m_surfaceGrids[facetIdx].m_sizeX << " x " << m_surfaceGrids[facetIdx].m_sizeY << ", distances = " << m_surfaceGrids[facetIdx].m_distX << " x " << m_surfaceGrids[facetIdx].m_distY << ", origin = " << m_surfaceGrids[facetIdx].m_gridOrigin;
	
		m_surfaceGrids[facetIdx].m_gridID = idStr.str();
	}
}

void PointHullStorage_Full::setSurfaceGridOrigin(const unsigned long& facetIdx, const Vector3& origin)
{
	if (m_surfaceGrids.find(facetIdx) == m_surfaceGrids.end())
	{
		m_surfaceGrids.insert(std::make_pair(facetIdx, SurfaceGrid()));
	}

	m_surfaceGrids[facetIdx].m_gridOrigin = origin;

	//if (m_surfaceGrids[facetIdx].m_gridID.empty())
	{
		std::stringstream idStr;
		idStr << "SurfaceGrid " << facetIdx << ": Size " << m_surfaceGrids[facetIdx].m_sizeX << " x " << m_surfaceGrids[facetIdx].m_sizeY << ", distances = " << m_surfaceGrids[facetIdx].m_distX << " x " << m_surfaceGrids[facetIdx].m_distY << ", origin = " << m_surfaceGrids[facetIdx].m_gridOrigin;
	
		m_surfaceGrids[facetIdx].m_gridID = idStr.str();
	}
}

const Vector3 PointHullStorage_Full::getSurfacePoint(const unsigned long& facetIdx, const unsigned int& x, const unsigned int& y)
{
	if (m_surfacePointFacetIndices.find(facetIdx) != m_surfacePointFacetIndices.end())
	{
		if (m_surfacePointGridSizes[facetIdx].first >= x && m_surfacePointGridSizes[facetIdx].second >= y)
		{
			std::pair<std::multimap<unsigned long, std::pair<std::pair<unsigned int, unsigned int>, Vector3> >::iterator, std::multimap<unsigned long, std::pair<std::pair<unsigned int, unsigned int>, Vector3> >::iterator> sp_it = m_surfacePointsPerFacet.equal_range(facetIdx);
			for (std::multimap<unsigned long, std::pair<std::pair<unsigned int, unsigned int>, Vector3> >::iterator it = sp_it.first; it != sp_it.second; it++)
			{
				if (it->second.first.first == x && it->second.first.second == y)
					return it->second.second;
			}
		}
	}
	return Vector3(0, 0, 0);
}

const SurfacePointType PointHullStorage_Full::getSurfacePointType(const unsigned long& facetIdx, const unsigned int& x, const unsigned int& y)
{
	if (m_surfacePointFacetIndices.find(facetIdx) != m_surfacePointFacetIndices.end())
	{
		if (m_surfacePointGridSizes[facetIdx].first >= x && m_surfacePointGridSizes[facetIdx].second >= y)
		{
			std::pair<std::multimap<unsigned long, std::pair<std::pair<unsigned int, unsigned int>, SurfacePointType> >::iterator, std::multimap<unsigned long, std::pair<std::pair<unsigned int, unsigned int>, SurfacePointType> >::iterator> sp_it = m_surfacePointFacetTypes.equal_range(facetIdx);
			for (std::multimap<unsigned long, std::pair<std::pair<unsigned int, unsigned int>, SurfacePointType> >::iterator it = sp_it.first; it != sp_it.second; it++)
			{
				if (it->second.first.first == x && it->second.first.second == y)
					return it->second.second;
			}
		}
	}
	return UNKNOWN_SURFACE;
}

const SurfaceGridMap& PointHullStorage_Full::getSurfaceGridData()
{
	return m_surfaceGrids;
}