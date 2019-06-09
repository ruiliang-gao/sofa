#include "PointHull_Computation.h"

#include <sofa/helper/accessor.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseMechanics/MechanicalObject.h>

#include <PQP/include/PQP.h>

namespace sofa
{
	namespace component
	{
		namespace collision
		{
			class PointHullComputation_CPU_Private
			{
				public:
					PointHullComputation_CPU_Private() : m_pqp_tree(NULL)
					{}

					PQP_Model* m_pqp_tree;
					std::map<int, BV*> m_leaf_obbs;

					std::map<unsigned long, Triangle3d> m_surfaceTriangles;
			};
		}
	}
}

using namespace sofa::helper;
using namespace sofa::core::topology;
using namespace sofa::component::collision;
using namespace sofa::component::topology;
using namespace sofa::component::container;


PointHullComputation::PointHullComputation(PointHullStorage* storage) : m_storage(storage), m_edgePointsPerEdge(5), m_modelGridSize(16), m_modelTileSize(8)
{

}

const PointHullComputation::FacetDiagonalsMap& PointHullComputation::getFacetDiagonals()
{
	return this->m_facetDiagonals;
}

const PointHullComputation::FacetBordersMap& PointHullComputation::getFacetLongestSides()
{
	return this->m_facetLongestSides;
}

const PointHullComputation::FacetBordersMap& PointHullComputation::getFacetBroadestSides()
{
	return this->m_facetBroadestSides;
}

const PointHullComputation::FacetSurfaceGridSizeMap& PointHullComputation::getFacetSurfaceGridSizes()
{
	return this->m_facetGridSizes;
}

PointHullComputation_CPU::PointHullComputation_CPU(PointHullStorage* storage) : PointHullComputation(storage), m_d(NULL)
{
	m_d = new PointHullComputation_CPU_Private;
}

PointHullComputation_CPU::~PointHullComputation_CPU()
{
	if (m_d->m_pqp_tree != NULL)
		delete m_d->m_pqp_tree;
}

PQP_Model* PointHullComputation_CPU::getPqpModel()
{
	return m_d->m_pqp_tree;
}

const std::map<unsigned long, Triangle3d>& PointHullComputation_CPU::getSurfaceTriangles()
{
	return m_d->m_surfaceTriangles;
}

void PointHullComputation_CPU::computePointHull()
{
	std::cout << "PointHullComputation_CPU::computePointHull()" << std::endl;
	std::cout << "  computeObbHierarchy() call" << std::endl;
	
	m_d->m_leaf_obbs.clear();
	
	computeObbHierarchy();

	m_facetDiagonals.clear();
	m_facetGridSizes.clear();
	m_facetLongestSides.clear();
	m_facetBroadestSides.clear();

	m_d->m_surfaceTriangles.clear();

	const sofa::core::topology::BaseMeshTopology::SeqTriangles& meshTriangles = m_storage->getTopology()->getTriangles();
	MeshTopology* meshTopology = dynamic_cast<MeshTopology*>(m_storage->getTopology());
	ReadAccessor<Data<Vec3Types::VecCoord> > meshPoints(m_storage->getMechanicalState()->read(core::ConstVecCoordId::position()));

	unsigned int numTiles = (m_modelGridSize * m_modelGridSize) / (m_modelTileSize * m_modelTileSize);
	unsigned int tilesPerSide = std::sqrt(numTiles);
	unsigned int pmSize = tilesPerSide * m_modelTileSize;

	std::cout << "  Add vertices, edge points, surface points" << std::endl;

	unsigned long vertexIndex = 0;
	unsigned long edgePointIndex = 0;
	unsigned long facetIndex = 0;

	MechanicalObject<Rigid3Types>* mechanicalObject = dynamic_cast<MechanicalObject<Rigid3Types>*>(m_storage->getObjectMechanicalState());
	Rigid3dTypes::VecCoord coords = mechanicalObject->getPosition();

	Vector3 initialPosition(coords[0][0], coords[0][1], coords[0][2]);
	Quat initialOrientation(coords[0][3], coords[0][4], coords[0][5], coords[0][6]);

	for (unsigned long k = 0; k < meshTriangles.size(); k++)
	{
		Vector3 corner0 = meshPoints[meshTriangles[k][0]];
		Vector3 corner1 = meshPoints[meshTriangles[k][1]];
		Vector3 corner2 = meshPoints[meshTriangles[k][2]];

		corner0 -= initialPosition;
		corner1 -= initialPosition;
		corner2 -= initialPosition;

		corner0 = initialOrientation.inverseRotate(corner0);
		corner1 = initialOrientation.inverseRotate(corner1);
		corner2 = initialOrientation.inverseRotate(corner2);

		std::vector<Vector3> facetPoints;
		facetPoints.push_back(corner0);
		facetPoints.push_back(corner1);
		facetPoints.push_back(corner2);
	
		std::cout << "   - triangle " << k << ": " << meshTriangles[k][0] << "," << meshTriangles[k][1] << "," << meshTriangles[k][2] << " vertices: 0 = " << corner0 << ", 1 = " << corner1 << ", 2 = " << corner2 << std::endl;

		if (m_storage->addVertex(corner0, vertexIndex, meshTriangles[k][0]))
			vertexIndex++;

		if (m_storage->addVertex(corner1, vertexIndex, meshTriangles[k][1]))
			vertexIndex++;

		if (m_storage->addVertex(corner2, vertexIndex, meshTriangles[k][2]))
			vertexIndex++;

		sofa::core::topology::BaseMeshTopology::EdgesInTriangle triangleEdges = meshTopology->getEdgesInTriangle(k);
		
		unsigned int edgeCount = 0;
		Vector3 e_vtx_1, e_vtx_2;
		for (unsigned int l = 0; l < 3; l++)
		{
			if (edgeCount == 0)
			{
				e_vtx_1 = corner0;
				e_vtx_2 = corner1;
			}
			else if (edgeCount == 1)
			{
				e_vtx_1 = corner1;
				e_vtx_2 = corner2;
			}
			else if (edgeCount == 2)
			{
				e_vtx_1 = corner2;
				e_vtx_2 = corner0;
			}

			for (unsigned int p = 1; p < m_edgePointsPerEdge; p++)
			{
				Vector3 ep = e_vtx_1 + ((e_vtx_2 - e_vtx_1) * (1.0f * p / m_edgePointsPerEdge));
				if (m_storage->addEdgePoint(ep, edgePointIndex, triangleEdges[l], k))
				{
					//m_storage->setEdgePointFacetIndex(edgePointIndex, k);
					edgePointIndex++;
				}
				std::cout << "   - edge point " << p << ": point " << ep  << "/edge index " << triangleEdges[l] << "/facet index " << k << std::endl;
			}
			edgeCount++;
		}


		std::cout << "Compute surface point grid for triangle " << k << std::endl;

        std::map<unsigned int, Vector3> coincidentPoints;
		bool foundCoincident = false;

		BV* triangle_obb = m_d->m_leaf_obbs[k];
		
		Matrix3 obbRotation; obbRotation.identity();
		obbRotation[0] = Vector3(triangle_obb->R[0][0], triangle_obb->R[1][0], triangle_obb->R[2][0]);
		obbRotation[1] = Vector3(triangle_obb->R[0][1], triangle_obb->R[1][1], triangle_obb->R[2][1]);
		obbRotation[2] = Vector3(triangle_obb->R[0][2], triangle_obb->R[1][2], triangle_obb->R[2][2]);

		Matrix4 glOrientation; glOrientation.identity();
		for (int r = 0; r < 3; r++)
		{
			for (int s = 0; s < 3; s++)
			{
				glOrientation[r][s] = obbRotation[r][s];
			}
		}
		
		Vector3 triangle_obb_center(triangle_obb->To[0], triangle_obb->To[1], triangle_obb->To[2]);
		Vector3 triangle_obb_extents(triangle_obb->d[0], triangle_obb->d[1], triangle_obb->d[2]);

		std::vector<Vector3> triangleObbPoints;

		//0
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));
		//1
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
		//2
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
		//3
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
		
		//4
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
		//5
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));
		//6
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));
		//7
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));

		std::cout << " triangle obb corners (untransformed): " << std::endl;
		for (unsigned int p = 0; p < triangleObbPoints.size(); p++)
		{
			std::cout << p << " = " << triangleObbPoints[p] << ";" << std::endl;
		}
		
		for (unsigned int p = 0; p < triangleObbPoints.size(); p++)
		{
			std::cout << "  - triangle OBB point " << triangleObbPoints[p] << std::endl;
			for (unsigned int q = 0; q < facetPoints.size(); q++)
			{
				std::cout << "      facet point " << facetPoints[q] << "; equal triangleObbPoint = " << (triangleObbPoints[p] == facetPoints[q]) << std::endl;
				if (triangleObbPoints[p] == facetPoints[q])
				{
                    coincidentPoints[q] = facetPoints[q];
					foundCoincident = true;
					break;
				}
			}
		}

        std::cout << " found coincident points: ";
		Vector3 upperCorner;
		Vector3 lowerCorner;
		if (foundCoincident)
        {
            std::cout << "YES: "  << coincidentPoints.size() << " -- ";

            Vector3 coincidentPoint;
            unsigned int coincidentIdx = 0;
            for (std::map<unsigned int, Vector3>::const_iterator it = coincidentPoints.begin(); it != coincidentPoints.end(); it++)
            {
                std::cout << it->first << ": " << it->second << ";";
                if (coincidentIdx == 0)
                    coincidentPoint = it->second;
            }
            std::cout << std::endl;

            unsigned int sourceIdx = 8;
            for (unsigned int p = 0; p < triangleObbPoints.size() ; p++)
            {
                if (triangleObbPoints[p] == coincidentPoint)
                {
                    sourceIdx = p;
                }
            }

            std::cout << " sourceIdx = " << sourceIdx << std::endl;
			
            if (sourceIdx < 8)
            {
                double maxDist = 0.0f;
                
                unsigned int upperCornerIdx = 0;
				std::cout << " upperCorner search" << std::endl;
                for (unsigned int i = 0; i < triangleObbPoints.size(); i++)
                {
                    BVHModels::Segment3d coVtSeg(coincidentPoint, triangleObbPoints[i]);
					std::cout << " - " << i << ": maxDist = " << maxDist << ", coVtSeg.SquaredLength() = " << coVtSeg.SquaredLength() << std::endl;
                    if (coVtSeg.SquaredLength() > maxDist)
                    {
                        maxDist = coVtSeg.SquaredLength();
                        upperCorner = triangleObbPoints[i];
                        upperCornerIdx = i;
						std::cout << " new upperCorner = " << upperCorner << ", upperCornerIdx = " << upperCornerIdx << ", maxDist = " << maxDist << std::endl;
                    }
                }

                std::cout << " upperCorner = " << upperCorner << ", upperCornerIdx = " << upperCornerIdx << std::endl;

				unsigned int lowerCornerIdx = 0;
				if (upperCornerIdx == 0)
				{
					lowerCorner = triangleObbPoints[3];
					lowerCornerIdx = 3;
				}
				else if (upperCornerIdx == 1)
				{
					lowerCorner = triangleObbPoints[2];
					lowerCornerIdx = 2;
				}
				else if (upperCornerIdx == 2)
				{
					lowerCorner = triangleObbPoints[1];
					lowerCornerIdx = 1;
				}
				else if (upperCornerIdx == 3)
				{
					lowerCorner = triangleObbPoints[0];
					lowerCornerIdx = 0;
				}
				else if (upperCornerIdx == 4)
				{
					lowerCorner = triangleObbPoints[7];
					lowerCornerIdx = 7;
				}
				else if (upperCornerIdx == 5)
				{
					lowerCorner = triangleObbPoints[6];
					lowerCornerIdx = 6;
				}
				else if (upperCornerIdx == 6)
				{
					lowerCorner = triangleObbPoints[5];
					lowerCornerIdx = 5;
				}
				else if (upperCornerIdx == 7)
				{
					lowerCorner = triangleObbPoints[4];
					lowerCornerIdx = 4;
				}

                std::cout << " lowerCorner = " << lowerCorner << ", lowerCornerIdx = " << lowerCornerIdx << std::endl;

				m_facetDiagonals.insert(std::make_pair(k, std::make_pair(lowerCorner, upperCorner)));
            }
        }
		else
        {
			std::cout << "NO; using corner 0 to corner 3 as diagonal: " << triangleObbPoints[0] << " -- " << triangleObbPoints[3];
			m_facetDiagonals.insert(std::make_pair(k, std::make_pair(triangleObbPoints[0], triangleObbPoints[3])));
			lowerCorner = triangleObbPoints[0];
			upperCorner = triangleObbPoints[3];
        }

		m_storage->setSurfaceGridOrigin(k, lowerCorner);

		std::cout << std::endl;

		Vector3 obb_side_0 = triangleObbPoints[1] - triangleObbPoints[0];
		Vector3 obb_side_1 = triangleObbPoints[2] - triangleObbPoints[0];
		Vector3 obb_side_2 = triangleObbPoints[4] - triangleObbPoints[0];

		std::cout << " obb_side_0 = " << obb_side_0 << ", length = " << obb_side_0.norm() << std::endl;
		std::cout << " obb_side_1 = " << obb_side_1 << ", length = " << obb_side_1.norm() << std::endl;
		std::cout << " obb_side_2 = " << obb_side_2 << ", length = " << obb_side_2.norm() << std::endl;

		Vector3 smallest, biggest;
		unsigned int smallest_idx, biggest_idx;
		if (obb_side_0.norm() < obb_side_1.norm())
		{
			smallest = obb_side_0;
			smallest_idx = 0;
			biggest = obb_side_1;
			biggest_idx = 1;
		}
		else
		{
			smallest = obb_side_1;
			smallest_idx = 1;
			biggest = obb_side_0;
			biggest_idx = 0;
		}

		if (obb_side_2.norm() < smallest.norm())
		{
			smallest = obb_side_2;
			smallest_idx = 2;
		}
		else if (obb_side_2.norm() > biggest.norm())
		{
			biggest = obb_side_2;
			biggest_idx = 2;
		}

		std::cout << " longest OBB side = " << biggest << " (index " << biggest_idx << "), shortest OBB side = " << smallest << " (index " << smallest_idx << ")" << std::endl;

		Vector3 biggest_pt1, biggest_pt2;
		if (biggest_idx == 0)
		{
			biggest_pt1 = triangleObbPoints[0];
			biggest_pt2 = triangleObbPoints[1];
		}
		else if (biggest_idx == 1)
		{
			biggest_pt1 = triangleObbPoints[0];
			biggest_pt2 = triangleObbPoints[2];
		}
		else if (biggest_idx == 2)
		{
			biggest_pt1 = triangleObbPoints[0];
			biggest_pt2 = triangleObbPoints[4];
		}

		unsigned int broadest_idx = 0;
		if (biggest_idx == 0 && smallest_idx == 1)
		{
			broadest_idx = 2;
		}
		else if (biggest_idx == 1 && smallest_idx == 2)
		{
			broadest_idx = 0;
		}
		if (biggest_idx == 0 && smallest_idx == 2)
		{
			broadest_idx = 1;
		}

		Vector3 broadest_pt1, broadest_pt2;
		if (broadest_idx == 0)
		{
			broadest_pt1 = triangleObbPoints[0];
			broadest_pt2 = triangleObbPoints[1];
		}
		else if (broadest_idx == 1)
		{
			broadest_pt1 = triangleObbPoints[0];
			broadest_pt2 = triangleObbPoints[2];
		}
		else if (broadest_idx == 2)
		{
			broadest_pt1 = triangleObbPoints[0];
			broadest_pt2 = triangleObbPoints[4];
		}

		std::cout << " longest OBB side = " << biggest_pt1 << " -- " << biggest_pt2 << " (index " << biggest_idx << ")" << std::endl;
		std::cout << " broadest OBB side = " << broadest_pt1 << " -- " << broadest_pt2 << " (index " << broadest_idx << ")" << std::endl;

		m_facetLongestSides.insert(std::make_pair(k, std::make_pair(biggest_pt1, biggest_pt2)));
		m_facetBroadestSides.insert(std::make_pair(k, std::make_pair(broadest_pt1, broadest_pt2)));

		Segment3d lowerLine(lowerCorner, lowerCorner + (biggest_pt2 - biggest_pt1));
		Segment3d upperLine(upperCorner, upperCorner + (broadest_pt2 - broadest_pt1));

		Vector3 bvt1 = lowerLine.ProjectOnSegment(upperCorner);
		Vector3 bvt2 = upperLine.ProjectOnSegment(lowerCorner);

		double prLen = (biggest_pt2 - biggest_pt1).norm();
		double supLen = (broadest_pt2 - broadest_pt1).norm();

		std::cout << " bvt1 = " << bvt1 << ", bvt2 = " << bvt2 << ", prLen = " << prLen << ", supLen = " << supLen << std::endl;

		if (supLen != 0.0f)
		{
			double prSupRatio = prLen / supLen;
			double prSupArea = prLen * supLen;

			std::cout << " prSupRatio = " << prSupRatio << ", prSupArea = " << prSupArea << std::endl;

			double ratioDenominatorX = 10.0f;
			double ratioDenominatorY = 10.0f;
			unsigned long iterationCountX = 10;
			unsigned long iterationCountY = 10;

			if (prSupRatio >= 1.0f)
			{
				ratioDenominatorX *= prSupRatio;
				iterationCountX = (unsigned long)std::fabs(prSupRatio * iterationCountX);
			}
			else
			{
				ratioDenominatorY *= (1.0f + prSupRatio);
				iterationCountY = (unsigned long)std::fabs((1.0f + prSupRatio) * iterationCountY);
			}

			double iteratedArea = (prLen / iterationCountX) * (supLen / iterationCountY);
			double iteratedAreaPercentage = (iteratedArea / prSupArea) * 100.0f;

			std::cout << " iteratedArea = " << iteratedArea << ", iteratedAreaPercentage = " << iteratedAreaPercentage << std::endl;

			int steps_grid_fit = 0;
			do
			{
				if (iterationCountX <= 1 || iterationCountY <= 1)
					break;

				iterationCountX -= 1;
				iterationCountY -= 1;

				ratioDenominatorX *= ((iterationCountX * 1.0f) / ((iterationCountX + 1) * 1.0f));
				ratioDenominatorY *= ((iterationCountY * 1.0f) / ((iterationCountY + 1) * 1.0f));

				iteratedArea = (prLen / iterationCountX) * (supLen / iterationCountY);
				iteratedAreaPercentage = (iteratedArea / prSupArea) * 100.0f;

				std::cout << "  step " << steps_grid_fit << ": iteratedArea = " << iteratedArea << ", iteratedAreaPercentage = " << iteratedAreaPercentage << ", iterationCountX = " << iterationCountX << ", iterationCountY = " << iterationCountY << std::endl;
				steps_grid_fit++;

			} while (iteratedAreaPercentage < 1.0f && iterationCountX >= 8 && iterationCountY >= 8);

			int surfaceGridSizeX = iterationCountX + 1;
			int surfaceGridSizeY = iterationCountY + 1;

			std::cout << " Triangle " << k << " surface grid size = " << surfaceGridSizeX << " x " << surfaceGridSizeY << std::endl;

			Vector3 totalDiffX = bvt2 - lowerCorner;
			Vector3 totalDiffY = upperCorner - bvt2;

			Vector3 oneStepX = ((1 / ratioDenominatorX) * totalDiffX);
			Vector3 oneStepY = ((1 / ratioDenominatorY) * totalDiffY);

			if (surfaceGridSizeX <= 2)
			{
				double oneStepRatioX = surfaceGridSizeX / 8.0f;
				oneStepX *= oneStepRatioX;

				surfaceGridSizeX = 8;
			}

			if (surfaceGridSizeY <= 2)
			{
				double oneStepRatioY = surfaceGridSizeY / 8.0f;
				oneStepY *= oneStepRatioY;

				surfaceGridSizeY = 8;
			}

			m_facetGridSizes.insert(std::make_pair(k, std::make_pair(surfaceGridSizeX, surfaceGridSizeY)));

			this->m_storage->setSurfaceGridSize(k, surfaceGridSizeX, surfaceGridSizeY);

			this->m_storage->setSurfaceGridDistance(k, oneStepX.norm(), oneStepY.norm());
			this->m_storage->setSurfaceGridOffset(k, oneStepX, oneStepY);

			Triangle3d current_triangle(facetPoints[0], facetPoints[1], facetPoints[2]);
			m_d->m_surfaceTriangles.insert(std::make_pair(k, current_triangle));

			for (unsigned int q = 0; q <= surfaceGridSizeX; q++)
			{
				Vector3 partStepX = ((q / ratioDenominatorX) * totalDiffX);
				Vector3 ipp1 = lowerCorner + partStepX;

				Vector3 prevP = ipp1;
				Vector3 nextP = ipp1;
				for (unsigned int r = 0; r <= surfaceGridSizeY; r++)
				{
					Vector3 partStepY = ((r / ratioDenominatorY) * totalDiffY);
					Vector3 partStepYP1 = (((r + 1) / ratioDenominatorY) * totalDiffY);

					Vector3 ipp2 = ipp1 + partStepY;
					nextP = ipp1 + partStepYP1;

					if (current_triangle.ContainsPoint(ipp2))
					{
						m_storage->addSurfacePoint(ipp2, q, r, k, INSIDE_SURFACE);
						//addSurfacePoint(q, r, ipp2, INSIDE_SURFACE);
						if (prevP != ipp1)
						{
							if (!current_triangle.ContainsPoint(prevP))
							{
								m_storage->addSurfacePoint(prevP, q, r - 1, k, EDGE_OF_SURFACE);
							}
						}
						if (nextP != ipp1)
						{
							if (!current_triangle.ContainsPoint(nextP))
							{
								m_storage->addSurfacePoint(nextP, q, r + 1, k, EDGE_OF_SURFACE);
							}
						}
					}
					else
					{
						if (current_triangle.ContainsPoint(prevP))
						{
							m_storage->addSurfacePoint(ipp2, q, r, k, EDGE_OF_SURFACE);
						}
						else if (current_triangle.ContainsPoint(nextP))
						{
							m_storage->addSurfacePoint(ipp2, q, r, k, EDGE_OF_SURFACE);
						}
						else
						{
							m_storage->addSurfacePoint(ipp2, q, r, k, OUTSIDE_SURFACE);
						}
					}

					prevP = ipp2;
				}
			}
		}
		else
		{
			std::cout << " WARNING: Triangle " << k << " has degenerate OBB? length of broadest OBB side = " << supLen << ", of longest OBB side = " << prLen << std::endl;
			m_facetGridSizes.insert(std::make_pair(k, std::make_pair(-1, -1)));
		}
	}

	std::cout << "====== FACET DATA ======" << std::endl;
	std::cout << "Leaf OBB's      = " << m_d->m_leaf_obbs.size() << std::endl;
	std::cout << "Facet diagonals = " << m_facetDiagonals.size() << std::endl;
	for (std::map<int, BV*>::const_iterator obb_it = m_d->m_leaf_obbs.begin(); obb_it != m_d->m_leaf_obbs.end(); obb_it++)
	{
		if (m_facetDiagonals.find((unsigned long)obb_it->first) != m_facetDiagonals.end())
			std::cout << "  - leaf obb " << obb_it->first << ": Diagonal = " << m_facetDiagonals[(unsigned long)obb_it->first].first << " -- " << m_facetDiagonals[(unsigned long)obb_it->first].second << std::endl;
		else
			std::cout << "  - leaf obb " << obb_it->first << ": NO FACET DIAGONAL EXISTS!" << std::endl;
	}

	std::cout << "Surface grid sizes = " << m_facetGridSizes.size() << std::endl;
	for (FacetSurfaceGridSizeMap::const_iterator it = m_facetGridSizes.begin(); it != m_facetGridSizes.end(); it++)
		std::cout << "  - Triangle " << it->first << ": " << it->second.first << " x " << it->second.second << std::endl;

	std::cout << "====== FACET DATA ======" << std::endl;
}

void PointHullComputation_CPU::computeGridSizes()
{
	const sofa::core::topology::BaseMeshTopology::SeqTriangles& meshTriangles = m_storage->getTopology()->getTriangles();
	MeshTopology* meshTopology = dynamic_cast<MeshTopology*>(m_storage->getTopology());
	ReadAccessor<Data<Vec3Types::VecCoord> > meshPoints(m_storage->getMechanicalState()->read(core::ConstVecCoordId::position()));

	MechanicalObject<Rigid3Types>* mechanicalObject = dynamic_cast<MechanicalObject<Rigid3Types>*>(m_storage->getObjectMechanicalState());
	Rigid3dTypes::VecCoord coords = mechanicalObject->getPosition();

	Vector3 initialPosition(coords[0][0], coords[0][1], coords[0][2]);
	Quat initialOrientation(coords[0][3], coords[0][4], coords[0][5], coords[0][6]);

	for (unsigned long k = 0; k < meshTriangles.size(); k++)
	{
		Vector3 corner0 = meshPoints[meshTriangles[k][0]];
		Vector3 corner1 = meshPoints[meshTriangles[k][1]];
		Vector3 corner2 = meshPoints[meshTriangles[k][2]];

		corner0 -= initialPosition;
		corner1 -= initialPosition;
		corner2 -= initialPosition;

		corner0 = initialOrientation.inverseRotate(corner0);
		corner1 = initialOrientation.inverseRotate(corner1);
		corner2 = initialOrientation.inverseRotate(corner2);

		std::vector<Vector3> facetPoints;
		facetPoints.push_back(corner0);
		facetPoints.push_back(corner1);
		facetPoints.push_back(corner2);

		std::cout << "   - triangle " << k << ": " << meshTriangles[k][0] << "," << meshTriangles[k][1] << "," << meshTriangles[k][2] << " vertices: 0 = " << corner0 << ", 1 = " << corner1 << ", 2 = " << corner2 << std::endl;

		sofa::core::topology::BaseMeshTopology::EdgesInTriangle triangleEdges = meshTopology->getEdgesInTriangle(k);

		unsigned int edgeCount = 0;
		Vector3 e_vtx_1, e_vtx_2;
		for (unsigned int l = 0; l < 3; l++)
		{
			if (edgeCount == 0)
			{
				e_vtx_1 = corner0;
				e_vtx_2 = corner1;
			}
			else if (edgeCount == 1)
			{
				e_vtx_1 = corner1;
				e_vtx_2 = corner2;
			}
			else if (edgeCount == 2)
			{
				e_vtx_1 = corner2;
				e_vtx_2 = corner0;
			}

			edgeCount++;
		}


		std::cout << "Compute surface point grid for triangle " << k << std::endl;

		std::map<unsigned int, Vector3> coincidentPoints;
		bool foundCoincident = false;

		BV* triangle_obb = m_d->m_leaf_obbs[k];

		Matrix3 obbRotation; obbRotation.identity();
		obbRotation[0] = Vector3(triangle_obb->R[0][0], triangle_obb->R[1][0], triangle_obb->R[2][0]);
		obbRotation[1] = Vector3(triangle_obb->R[0][1], triangle_obb->R[1][1], triangle_obb->R[2][1]);
		obbRotation[2] = Vector3(triangle_obb->R[0][2], triangle_obb->R[1][2], triangle_obb->R[2][2]);

		Matrix4 glOrientation; glOrientation.identity();
		for (int r = 0; r < 3; r++)
		{
			for (int s = 0; s < 3; s++)
			{
				glOrientation[r][s] = obbRotation[r][s];
			}
		}

		Vector3 triangle_obb_center(triangle_obb->To[0], triangle_obb->To[1], triangle_obb->To[2]);
		Vector3 triangle_obb_extents(triangle_obb->d[0], triangle_obb->d[1], triangle_obb->d[2]);

		std::vector<Vector3> triangleObbPoints;

		//0
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));
		//1
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
		//2
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));
		//3
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), triangle_obb_extents.z())));

		//4
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), -triangle_obb_extents.y(), triangle_obb_extents.z())));
		//5
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), -triangle_obb_extents.y(), -triangle_obb_extents.z())));
		//6
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(-triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));
		//7
		triangleObbPoints.push_back(triangle_obb_center + glOrientation.transposed().transform(Vector3(triangle_obb_extents.x(), triangle_obb_extents.y(), -triangle_obb_extents.z())));

		std::cout << " triangle obb corners: " << std::endl;
		for (unsigned int p = 0; p < triangleObbPoints.size(); p++)
		{
			std::cout << p << " = " << triangleObbPoints[p] << ";" << std::endl;
		}

		for (unsigned int p = 0; p < triangleObbPoints.size(); p++)
		{
			std::cout << "  - triangle OBB point " << triangleObbPoints[p] << std::endl;
			for (unsigned int q = 0; q < facetPoints.size(); q++)
			{
				std::cout << "      facet point " << facetPoints[q] << "; equal triangleObbPoint = " << (triangleObbPoints[p] == facetPoints[q]) << std::endl;
				if (triangleObbPoints[p] == facetPoints[q])
				{
					coincidentPoints[q] = facetPoints[q];
					foundCoincident = true;
					break;
				}
			}
		}

		std::cout << " found coincident points: ";
		Vector3 upperCorner;
		Vector3 lowerCorner;
		if (foundCoincident)
		{
			std::cout << "YES: " << coincidentPoints.size() << " -- ";

			Vector3 coincidentPoint;
			unsigned int coincidentIdx = 0;
			for (std::map<unsigned int, Vector3>::const_iterator it = coincidentPoints.begin(); it != coincidentPoints.end(); it++)
			{
				std::cout << it->first << ": " << it->second << ";";
				if (coincidentIdx == 0)
					coincidentPoint = it->second;
			}
			std::cout << std::endl;

			unsigned int sourceIdx = 8;
			for (unsigned int p = 0; p < triangleObbPoints.size(); p++)
			{
				if (triangleObbPoints[p] == coincidentPoint)
				{
					sourceIdx = p;
				}
			}

			std::cout << " sourceIdx = " << sourceIdx << std::endl;

			if (sourceIdx < 8)
			{
				double maxDist = 0.0f;

				unsigned int upperCornerIdx = 0;
				std::cout << " upperCorner search" << std::endl;
				for (unsigned int i = 0; i < triangleObbPoints.size(); i++)
				{
					BVHModels::Segment3d coVtSeg(coincidentPoint, triangleObbPoints[i]);
					std::cout << " - " << i << ": maxDist = " << maxDist << ", coVtSeg.SquaredLength() = " << coVtSeg.SquaredLength() << std::endl;
					if (coVtSeg.SquaredLength() > maxDist)
					{
						maxDist = coVtSeg.SquaredLength();
						upperCorner = triangleObbPoints[i];
						upperCornerIdx = i;
						std::cout << " new upperCorner = " << upperCorner << ", upperCornerIdx = " << upperCornerIdx << ", maxDist = " << maxDist << std::endl;
					}
				}

				std::cout << " upperCorner = " << upperCorner << ", upperCornerIdx = " << upperCornerIdx << std::endl;

				unsigned int lowerCornerIdx = 0;
				if (upperCornerIdx == 0)
				{
					lowerCorner = triangleObbPoints[3];
					lowerCornerIdx = 3;
				}
				else if (upperCornerIdx == 1)
				{
					lowerCorner = triangleObbPoints[2];
					lowerCornerIdx = 2;
				}
				else if (upperCornerIdx == 2)
				{
					lowerCorner = triangleObbPoints[1];
					lowerCornerIdx = 1;
				}
				else if (upperCornerIdx == 3)
				{
					lowerCorner = triangleObbPoints[0];
					lowerCornerIdx = 0;
				}
				else if (upperCornerIdx == 4)
				{
					lowerCorner = triangleObbPoints[7];
					lowerCornerIdx = 7;
				}
				else if (upperCornerIdx == 5)
				{
					lowerCorner = triangleObbPoints[6];
					lowerCornerIdx = 6;
				}
				else if (upperCornerIdx == 6)
				{
					lowerCorner = triangleObbPoints[5];
					lowerCornerIdx = 5;
				}
				else if (upperCornerIdx == 7)
				{
					lowerCorner = triangleObbPoints[4];
					lowerCornerIdx = 4;
				}

				std::cout << " lowerCorner = " << lowerCorner << ", lowerCornerIdx = " << lowerCornerIdx << std::endl;

				m_facetDiagonals.insert(std::make_pair(k, std::make_pair(lowerCorner, upperCorner)));
			}
		}
		else
		{
			std::cout << "NO; using corner 0 to corner 3 as diagonal: " << triangleObbPoints[0] << " -- " << triangleObbPoints[3];
			m_facetDiagonals.insert(std::make_pair(k, std::make_pair(triangleObbPoints[0], triangleObbPoints[3])));
			lowerCorner = triangleObbPoints[0];
			upperCorner = triangleObbPoints[3];
		}

		std::cout << std::endl;

		Vector3 obb_side_0 = triangleObbPoints[1] - triangleObbPoints[0];
		Vector3 obb_side_1 = triangleObbPoints[2] - triangleObbPoints[0];
		Vector3 obb_side_2 = triangleObbPoints[4] - triangleObbPoints[0];

		std::cout << " obb_side_0 = " << obb_side_0 << ", length = " << obb_side_0.norm() << std::endl;
		std::cout << " obb_side_1 = " << obb_side_1 << ", length = " << obb_side_1.norm() << std::endl;
		std::cout << " obb_side_2 = " << obb_side_2 << ", length = " << obb_side_2.norm() << std::endl;

		Vector3 smallest, biggest;
		unsigned int smallest_idx, biggest_idx;
		if (obb_side_0.norm() < obb_side_1.norm())
		{
			smallest = obb_side_0;
			smallest_idx = 0;
			biggest = obb_side_1;
			biggest_idx = 1;
		}
		else
		{
			smallest = obb_side_1;
			smallest_idx = 1;
			biggest = obb_side_0;
			biggest_idx = 0;
		}

		if (obb_side_2.norm() < smallest.norm())
		{
			smallest = obb_side_2;
			smallest_idx = 2;
		}
		else if (obb_side_2.norm() > biggest.norm())
		{
			biggest = obb_side_2;
			biggest_idx = 2;
		}

		std::cout << " longest OBB side = " << biggest << " (index " << biggest_idx << "), shortest OBB side = " << smallest << " (index " << smallest_idx << ")" << std::endl;

		Vector3 biggest_pt1, biggest_pt2;
		if (biggest_idx == 0)
		{
			biggest_pt1 = triangleObbPoints[0];
			biggest_pt2 = triangleObbPoints[1];
		}
		else if (biggest_idx == 1)
		{
			biggest_pt1 = triangleObbPoints[0];
			biggest_pt2 = triangleObbPoints[2];
		}
		else if (biggest_idx == 2)
		{
			biggest_pt1 = triangleObbPoints[0];
			biggest_pt2 = triangleObbPoints[4];
		}

		unsigned int broadest_idx = 0;
		if (biggest_idx == 0 && smallest_idx == 1)
		{
			broadest_idx = 2;
		}
		else if (biggest_idx == 1 && smallest_idx == 2)
		{
			broadest_idx = 0;
		}
		if (biggest_idx == 0 && smallest_idx == 2)
		{
			broadest_idx = 1;
		}

		Vector3 broadest_pt1, broadest_pt2;
		if (broadest_idx == 0)
		{
			broadest_pt1 = triangleObbPoints[0];
			broadest_pt2 = triangleObbPoints[1];
		}
		else if (broadest_idx == 1)
		{
			broadest_pt1 = triangleObbPoints[0];
			broadest_pt2 = triangleObbPoints[2];
		}
		else if (broadest_idx == 2)
		{
			broadest_pt1 = triangleObbPoints[0];
			broadest_pt2 = triangleObbPoints[4];
		}

		std::cout << " longest OBB side = " << biggest_pt1 << " -- " << biggest_pt2 << " (index " << biggest_idx << ")" << std::endl;
		std::cout << " broadest OBB side = " << broadest_pt1 << " -- " << broadest_pt2 << " (index " << broadest_idx << ")" << std::endl;

		m_facetLongestSides.insert(std::make_pair(k, std::make_pair(biggest_pt1, biggest_pt2)));
		m_facetBroadestSides.insert(std::make_pair(k, std::make_pair(broadest_pt1, broadest_pt2)));

		Segment3d lowerLine(lowerCorner, lowerCorner + (biggest_pt2 - biggest_pt1));
		Segment3d upperLine(upperCorner, upperCorner + (broadest_pt2 - broadest_pt1));

		Vector3 bvt1 = lowerLine.ProjectOnSegment(upperCorner);
		Vector3 bvt2 = upperLine.ProjectOnSegment(lowerCorner);

		double prLen = (biggest_pt2 - biggest_pt1).norm();
		double supLen = (broadest_pt2 - broadest_pt1).norm();

		std::cout << " bvt1 = " << bvt1 << ", bvt2 = " << bvt2 << ", prLen = " << prLen << ", supLen = " << supLen << std::endl;

		if (supLen != 0.0f)
		{
			double prSupRatio = prLen / supLen;
			double prSupArea = prLen * supLen;

			std::cout << " prSupRatio = " << prSupRatio << ", prSupArea = " << prSupArea << std::endl;

			double ratioDenominatorX = 10.0f;
			double ratioDenominatorY = 10.0f;
			unsigned long iterationCountX = 10;
			unsigned long iterationCountY = 10;

			if (prSupRatio >= 1.0f)
			{
				ratioDenominatorX *= prSupRatio;
				iterationCountX = (unsigned long)std::fabs(prSupRatio * iterationCountX);
			}
			else
			{
				ratioDenominatorY *= (1.0f + prSupRatio);
				iterationCountY = (unsigned long)std::fabs((1.0f + prSupRatio) * iterationCountY);
			}

			double iteratedArea = (prLen / iterationCountX) * (supLen / iterationCountY);
			double iteratedAreaPercentage = (iteratedArea / prSupArea) * 100.0f;

			std::cout << " iteratedArea = " << iteratedArea << ", iteratedAreaPercentage = " << iteratedAreaPercentage << std::endl;

			int steps_grid_fit = 0;
			do
			{
				iterationCountX -= 1;
				iterationCountY -= 1;

				ratioDenominatorX *= ((iterationCountX * 1.0f) / ((iterationCountX + 1) * 1.0f));
				ratioDenominatorY *= ((iterationCountY * 1.0f) / ((iterationCountY + 1) * 1.0f));

				iteratedArea = (prLen / iterationCountX) * (supLen / iterationCountY);
				iteratedAreaPercentage = (iteratedArea / prSupArea) * 100.0f;

				std::cout << "  step " << steps_grid_fit << ": iteratedArea = " << iteratedArea << ", iteratedAreaPercentage = " << iteratedAreaPercentage << ", iterationCountX = " << iterationCountX << ", iterationCountY = " << iterationCountY << std::endl;
				steps_grid_fit++;

			} while (iteratedAreaPercentage < 1.0f && iterationCountX > 0 && iterationCountY > 0);

			int surfaceGridSizeX = iterationCountX + 1;
			int surfaceGridSizeY = iterationCountY + 1;

			if (surfaceGridSizeX <= 2)
				surfaceGridSizeX = 8;

			if (surfaceGridSizeY <= 2)
				surfaceGridSizeY = 8;

			std::cout << " Triangle " << k << " surface grid size = " << surfaceGridSizeX << " x " << surfaceGridSizeY << std::endl;
			m_facetGridSizes.insert(std::make_pair(k, std::make_pair(surfaceGridSizeX, surfaceGridSizeY)));

			this->m_storage->setSurfaceGridSize(k, surfaceGridSizeX, surfaceGridSizeY);
		}
	}
}

void PointHullComputation_CPU::computeObbHierarchy()
{
	std::cout << "PointHullComputation_CPU::computeObbHierarchy()" << std::endl;
	BaseMeshTopology* meshTopology = m_storage->getTopology();
	sofa::core::behavior::MechanicalState<Vec3Types>* mechanicalState = m_storage->getMechanicalState();

#ifdef _WIN32
    core::behavior::MechanicalState<Vec3Types>::ReadVecCoord pos = mechanicalState->readPositions();
#else
	typename core::behavior::MechanicalState<Vec3Types>::ReadVecCoord pos = mechanicalState->readPositions();
#endif

    std::cout << "  mechanicalState->getX()->size() = " << pos.size() << ", meshTopology->getNbTriangles() = " << meshTopology->getNbTriangles() << std::endl;
    if (pos.size() > 0 && meshTopology->getNbTriangles() > 0)
	{
		m_d->m_pqp_tree = new PQP_Model;
		m_d->m_pqp_tree->BeginModel(meshTopology->getNbTriangles());
		float p1[3], p2[3], p3[3];

		MechanicalObject<Rigid3Types>* mechanicalObject = dynamic_cast<MechanicalObject<Rigid3Types>*>(m_storage->getObjectMechanicalState());
		Rigid3dTypes::VecCoord coords = mechanicalObject->getPosition();
		
		Vector3 initialPosition(coords[0][0], coords[0][1], coords[0][2]);
		Quat initialOrientation(coords[0][3], coords[0][4], coords[0][5], coords[0][6]);

		for (unsigned int i = 0; i < meshTopology->getNbTriangles(); ++i)
		{
			sofa::core::topology::BaseMeshTopology::Triangle triangle = meshTopology->getTriangle(i);
			
			int id = triangle[0];
            Vector3 v1 = pos[id];
			v1 -= initialPosition;
			v1 = initialOrientation.inverseRotate(v1);

			p1[0] = v1.x();
			p1[1] = v1.y();
			p1[2] = v1.z();

			id = triangle[1];
            Vector3 v2 = pos[id];
			v2 -= initialPosition;
			v2 = initialOrientation.inverseRotate(v2);

			p2[0] = v2.x();
			p2[1] = v2.y();
			p2[2] = v2.z();

			id = triangle[2];
            Vector3 v3 = pos[id];
			v3 -= initialPosition;
			v3 = initialOrientation.inverseRotate(v3);
			
			p3[0] = v3.x();
			p3[1] = v3.y();
			p3[2] = v3.z();

			m_d->m_pqp_tree->AddTri(p1, p2, p3, i);
		}

        m_d->m_pqp_tree->EndModel(0.0f, false, false);

		std::cout << "OBB tree size: " << m_d->m_pqp_tree->num_bvs << " BVs, " << m_d->m_pqp_tree->num_tris << " triangles." << std::endl;
		for (unsigned int k = 0; k < m_d->m_pqp_tree->num_bvs; k++)
		{
			if (m_d->m_pqp_tree->child(k)->Leaf())
			{
				BV* obb = m_d->m_pqp_tree->child(k);
				std::cout << " - OBB node " << k << ": LEAF node -- position = " << obb->To[0] << "," << obb->To[1] << "," << obb->To[2] << ", extents = " << obb->d[0] << "," << obb->d[1] << "," << obb->d[2];
				std::cout << ": triangle range = " << obb->child_range_min << ", " << obb->child_range_max << std::endl;
				m_d->m_leaf_obbs.insert(std::make_pair(obb->child_range_min, obb));
			}
			else
			{
				std::cout << " - OBB node " << k << ": INNER node." << std::endl;
			}
		}
	}
}
