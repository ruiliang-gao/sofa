#ifndef POINTHULL_COMPUTATION_H
#define POINTHULL_COMPUTATION_H

#include "initSPModels.h"
#include "PointHull_Container.h"

#include <sofa/defaulttype/Vec.h>

class PQP_Model;

#include <Primitives/Segment3.h>
#include <Primitives/Triangle3.h>

using namespace BVHModels;

namespace sofa
{
	namespace component
	{
		namespace collision
		{
			using namespace sofa::defaulttype;

			class SOFA_SPMODELSPLUGIN_API PointHullComputation
			{
				public:
                    typedef std::map<unsigned long, std::pair<Vector3, Vector3> > FacetDiagonalsMap;
					typedef std::map<unsigned long, std::pair<Vector3, Vector3> > FacetBordersMap;
					typedef std::map < unsigned long, std::pair<int, int> > FacetSurfaceGridSizeMap;

					PointHullComputation(PointHullStorage* storage);

					virtual void computePointHull() = 0;

					unsigned int getModelGridSize();
					unsigned int getModelTileSize();

					unsigned int getEdgePointsPerEdge();

                    const FacetDiagonalsMap& getFacetDiagonals();
					const FacetBordersMap& getFacetLongestSides();
					const FacetBordersMap& getFacetBroadestSides();
					const FacetSurfaceGridSizeMap& getFacetSurfaceGridSizes();

				protected:
					unsigned int m_edgePointsPerEdge;
					unsigned int m_modelGridSize;
					unsigned int m_modelTileSize;

                    FacetDiagonalsMap m_facetDiagonals;
					FacetBordersMap m_facetLongestSides;
					FacetBordersMap m_facetBroadestSides;
					FacetSurfaceGridSizeMap m_facetGridSizes;

					PointHullStorage* m_storage;
			};

			class PointHullComputation_CPU_Private;
			class SOFA_SPMODELSPLUGIN_API PointHullComputation_CPU : public PointHullComputation
			{
				public:
					PointHullComputation_CPU(PointHullStorage* storage);
					virtual ~PointHullComputation_CPU();

					void computePointHull();

					PQP_Model* getPqpModel();

					const std::map<unsigned long, Triangle3d>& getSurfaceTriangles();

				private:
					void computeObbHierarchy();
					void computeGridSizes();
					PointHullComputation_CPU_Private* m_d;
			};
		}
	}
}

#endif //POINTHULL_COMPUTATION_H
