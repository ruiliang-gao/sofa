#ifndef POINTHULLCONTAINER_H
#define POINTHULLCONTAINER_H

#include "initSPModels.h"

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <map>
#include <deque>

using namespace sofa::defaulttype;

namespace sofa
{
	namespace component
	{
		namespace collision
		{
			enum SurfacePointType
			{
				INSIDE_SURFACE,
				EDGE_OF_SURFACE,
				OUTSIDE_SURFACE,
				UNKNOWN_SURFACE
			};

			typedef std::map<unsigned int, std::vector<Vector3> > SurfaceGridPointMap;
			typedef std::map<unsigned int, std::vector<SurfacePointType> > SurfaceGridPointTypeMap;
			typedef std::map<unsigned int, Vector3> VerticesMap;
			typedef std::map<unsigned int, std::vector<Vector3> > EdgePointsMap;

			struct SurfaceGrid
			{
				public:

					SurfaceGrid(const std::string& gridId = "",  int size_x = -1, const int size_y = -1) : m_sizeX(size_x), m_sizeY(size_y), m_distX(0.0f), m_distY(0.0f)
					{
						if (gridId.empty())
							m_gridID = "Unknown";
						else
							m_gridID = gridId;
					}

					SurfaceGrid(const SurfaceGrid& other)
					{
						if (&other != this)
						{
							m_sizeX = other.m_sizeX;
							m_sizeY = other.m_sizeY;
							m_distX = other.m_distX;
							m_distY = other.m_distY;

							m_offsetX = other.m_offsetX;
							m_offsetY = other.m_offsetY;

							m_gridOrigin = other.m_gridOrigin;

							m_gridID = other.m_gridID;

							for (VerticesMap::const_iterator it = other.m_vertices.begin(); it != other.m_vertices.end(); it++)
							{
								m_vertices.insert(std::make_pair(it->first, it->second));
							}

							for (EdgePointsMap::const_iterator it = other.m_edgePoints.begin(); it != other.m_edgePoints.end(); it++)
							{
								//m_edgePoints.insert(std::make_pair(it->first, std::vector<Vector3>(it->second.begin(), it->second.end())));
								m_edgePoints.insert(std::make_pair(it->first, std::vector<Vector3>()));
								for (std::vector<Vector3>::const_iterator line_it = it->second.begin(); line_it != it->second.end(); line_it++)
									m_edgePoints[it->first].push_back((*line_it));
							}

							for (SurfaceGridPointMap::const_iterator it = other.m_surfacePoints.begin(); it != other.m_surfacePoints.end(); it++)
							{
								//m_surfacePoints.insert(std::make_pair(it->first, std::vector<Vector3>(it->second.begin(), it->second.end())));
								m_surfacePoints.insert(std::make_pair(it->first, std::vector<Vector3>()));
								for (std::vector<Vector3>::const_iterator line_it = it->second.begin(); line_it != it->second.end(); line_it++)
									m_surfacePoints[it->first].push_back((*line_it));
							}

							for (SurfaceGridPointTypeMap::const_iterator it = other.m_surfacePointTypes.begin(); it != other.m_surfacePointTypes.end(); it++)
							{
								//m_surfacePointTypes.insert(std::make_pair(it->first, std::vector<SurfacePointType>(it->second.begin(), it->second.end())));
								m_surfacePointTypes.insert(std::make_pair(it->first, std::vector<SurfacePointType>()));
								for (std::vector<SurfacePointType>::const_iterator line_it = it->second.begin(); line_it != it->second.end(); line_it++)
									m_surfacePointTypes[it->first].push_back((*line_it));
							}
						}
					}

					SurfaceGrid& operator=(const SurfaceGrid& other)
					{
						if (&other != this)
						{
							m_sizeX = other.m_sizeX;
							m_sizeY = other.m_sizeY;
							m_distX = other.m_distX;
							m_distY = other.m_distY;
						
							m_offsetX = other.m_offsetX;
							m_offsetY = other.m_offsetY;

							m_gridOrigin = other.m_gridOrigin;

							m_gridID = other.m_gridID;

							for (VerticesMap::const_iterator it = other.m_vertices.begin(); it != other.m_vertices.end(); it++)
							{
								m_vertices.insert(std::make_pair(it->first, it->second));
							}

							for (EdgePointsMap::const_iterator it = other.m_edgePoints.begin(); it != other.m_edgePoints.end(); it++)
							{
								m_edgePoints.insert(std::make_pair(it->first, std::vector<Vector3>(it->second.begin(), it->second.end())));
							}

							for (SurfaceGridPointMap::const_iterator it = other.m_surfacePoints.begin(); it != other.m_surfacePoints.end(); it++)
							{
								m_surfacePoints.insert(std::make_pair(it->first, std::vector<Vector3>(it->second.begin(), it->second.end())));
							}

							for (SurfaceGridPointTypeMap::const_iterator it = other.m_surfacePointTypes.begin(); it != other.m_surfacePointTypes.end(); it++)
							{
								m_surfacePointTypes.insert(std::make_pair(it->first, std::vector<SurfacePointType>(it->second.begin(), it->second.end())));
							}
						}
						return *this;
					}

					int m_sizeX, m_sizeY;
					double m_distX, m_distY;

					Vector3 m_offsetX, m_offsetY;

					Vector3 m_gridOrigin;

					VerticesMap m_vertices;
					EdgePointsMap m_edgePoints;
					SurfaceGridPointMap m_surfacePoints;
					SurfaceGridPointTypeMap m_surfacePointTypes;

					std::string m_gridID;
			};


			typedef std::map<unsigned long, SurfaceGrid> SurfaceGridMap;

			class SOFA_SPMODELSPLUGIN_API PointHullStorage
			{
				public:
					
					PointHullStorage(sofa::core::behavior::MechanicalState<Vec3Types>* state, sofa::core::behavior::BaseMechanicalState* objectMState, sofa::core::topology::BaseMeshTopology* topology);

					virtual unsigned long numFacets() = 0;

					virtual const SurfaceGridMap& getSurfaceGridData() = 0;

					virtual bool addVertex(const Vector3& point, const unsigned long& pointIdx, const unsigned long& facetIdx) = 0;
					
					virtual unsigned long numVertices() const = 0;

					virtual const Vector3& vertex(const unsigned long& idx) const = 0;
					virtual Vector3& vertex(const unsigned long& idx) = 0;

					virtual bool addEdgePoint(const Vector3& point, const unsigned long& pointIdx, const unsigned long& edgeIdx, const unsigned long& facetIdx) = 0;
					
					virtual unsigned long numEdgePoints() const = 0;

					virtual const Vector3& edgePoint(const unsigned long& idx) const = 0;
					virtual Vector3& edgePoint(const unsigned long& idx) = 0;

					virtual const std::list<unsigned long>& edgePointFacetIndex(const unsigned long& idx) = 0;
					virtual unsigned long edgePointEdgeIndex(const unsigned long& idx) = 0;
					//virtual void setEdgePointEdgeIndex(const unsigned long& idx, const unsigned long& value) = 0;
					//virtual void setEdgePointFacetIndex(const unsigned long& idx, const unsigned long& value) = 0;

					virtual bool addSurfacePoint(const Vector3& point, const unsigned int& x, const unsigned int& y, const unsigned long& facetIdx, const SurfacePointType& type) = 0; //(const Vector3& point, const unsigned long& pointIdx, const unsigned long& facetIdx) = 0;

					virtual const Vector3 getSurfacePoint(const unsigned long& facetIdx, const unsigned int& x, const unsigned int& y) { return Vector3(0, 0, 0); }
					virtual const SurfacePointType getSurfacePointType(const unsigned long& facetIdx, const unsigned int& x, const unsigned int& y) { return UNKNOWN_SURFACE; }

					virtual unsigned long numSurfacePoints() const = 0;

					virtual const Vector3& surfacePoint(const unsigned long& idx) const = 0;
					virtual Vector3& surfacePoint(const unsigned long& idx) = 0;

					virtual const std::pair<int, int>& getSurfaceGridSize(const unsigned long& facetIdx) { return std::make_pair(-1, -1); }
					virtual void setSurfaceGridSize(const unsigned long& facetIdx, const int& x, const int& y) {}

					virtual void setSurfaceGridDistance(const unsigned long&, const double&, const double&) {}
					virtual void setSurfaceGridOffset(const unsigned long&, const Vector3&, const Vector3&) {}
					virtual void setSurfaceGridOrigin(const unsigned long& facetIdx, const Vector3&) {}

					virtual unsigned long surfacePointIndex(const unsigned long& idx) = 0;
					//virtual unsigned long surfacePointFacetIndex(const unsigned long& idx) = 0;
					//virtual void setSurfacePointFacetIndex(const unsigned long& idx, const unsigned long& value) = 0;

					sofa::core::behavior::BaseMechanicalState* getObjectMechanicalState();
					sofa::core::behavior::MechanicalState<Vec3Types>* getMechanicalState();
					sofa::core::topology::BaseMeshTopology* getTopology();

				protected:
					sofa::core::behavior::MechanicalState<Vec3Types>* m_state;
					sofa::core::behavior::BaseMechanicalState* m_objectState;
					sofa::core::topology::BaseMeshTopology* m_topology;
			};

			class SOFA_SPMODELSPLUGIN_API PointHullStorage_Full : public PointHullStorage
			{
				public:
					PointHullStorage_Full(sofa::core::behavior::MechanicalState<Vec3Types>* state, sofa::core::behavior::BaseMechanicalState* objectMState, sofa::core::topology::BaseMeshTopology* topology);

					unsigned long numFacets();

					const SurfaceGridMap& getSurfaceGridData();

					bool addVertex(const Vector3& point, const unsigned long& pointIdx, const unsigned long& facetIdx);

					virtual unsigned long numVertices() const;

					virtual const Vector3& vertex(const unsigned long& idx) const;
					virtual Vector3& vertex(const unsigned long& idx);

					bool addEdgePoint(const Vector3& point, const unsigned long& pointIdx, const unsigned long& edgeIdx, const unsigned long& facetIdx);

					unsigned long numEdgePoints() const;

					virtual const Vector3& edgePoint(const unsigned long& idx) const;
					virtual Vector3& edgePoint(const unsigned long& idx);

					const std::list<unsigned long>& edgePointFacetIndex(const unsigned long& idx);
					unsigned long edgePointEdgeIndex(const unsigned long& idx);
					void setEdgePointEdgeIndex(const unsigned long& idx, const unsigned long& value);
					void setEdgePointFacetIndex(const unsigned long& idx, const unsigned long& value);
					
					bool addSurfacePoint(const Vector3& point, const unsigned int& x, const unsigned int& y, const unsigned long& facetIdx, const SurfacePointType& type); //const unsigned long& pointIdx, const unsigned long& facetIdx);
				
					unsigned long numSurfacePoints() const; 

					const Vector3& surfacePoint(const unsigned long& idx) const;
					Vector3& surfacePoint(const unsigned long& idx);

					const Vector3 getSurfacePoint(const unsigned long& facetIdx, const unsigned int& x, const unsigned int& y);
					const SurfacePointType getSurfacePointType(const unsigned long& facetIdx, const unsigned int& x, const unsigned int& y);
					
					const std::pair<int, int>& getSurfaceGridSize(const unsigned long& facetIdx);
					
					void setSurfaceGridSize(const unsigned long& facetIdx, const int& x, const int& y);

					void setSurfaceGridDistance(const unsigned long&, const double&, const double&);
					void setSurfaceGridOffset(const unsigned long&, const Vector3&, const Vector3&);
					void setSurfaceGridOrigin(const unsigned long& facetIdx, const Vector3&);

					unsigned long surfacePointIndex(const unsigned long& idx);
					//unsigned long surfacePointFacetIndex(const unsigned long& idx);
					//void setSurfacePointFacetIndex(const unsigned long& idx, const unsigned long& value);

				private:
					std::deque<Vector3> m_clusterPoints;
					std::deque<Vector3> m_edgePoints;
					std::deque<Vector3> m_vertices;

					std::map<unsigned long, unsigned int> m_surfacePointsCountPerFacet;

					std::multimap<unsigned long, std::pair<std::pair<unsigned int, unsigned int>, Vector3> > m_surfacePointsPerFacet;
					std::multimap<unsigned long, std::pair<unsigned int, unsigned int> > m_surfacePointFacetIndices;
					std::multimap<unsigned long, std::pair<std::pair<unsigned int, unsigned int>, SurfacePointType> > m_surfacePointFacetTypes;
					std::map<unsigned long, std::pair<int, int> > m_surfacePointGridSizes;

					std::map<unsigned long, std::list<unsigned long> > m_edgePointFacetIndices;
					std::map<unsigned long, std::list<unsigned long> > m_vertexFacetIndices;

					std::map<unsigned long, unsigned long> m_edgePointEdgeIndices;

					std::map<unsigned long, unsigned long> m_surfacePointIndices;
					std::map<unsigned long, unsigned long> m_edgePointIndices;
					std::map<unsigned long, unsigned long> m_vertexIndices;
					std::map<Vector3, unsigned long> m_vertexRegistryTimeAdjacency;


					SurfaceGridMap m_surfaceGrids;
			};

			class SOFA_SPMODELSPLUGIN_API PointHullStorage_Compressed //: public PointHullStorage
			{

			};

		}
	}
}

#endif //POINTHULLCONTAINER_H