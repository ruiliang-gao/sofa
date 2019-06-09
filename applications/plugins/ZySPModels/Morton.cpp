#include "Morton.h"

#include <cmath>

using namespace sofa::component::collision;

Morton::Morton(const Vector3& point, const Vector3& minPoint, const Vector3& maxPoint) : m_cellX(0), m_cellY(0), m_cellZ(0), m_minPoint(minPoint)
{
	m_dimensions = maxPoint - minPoint;
	m_cellX = std::min((int)std::floor(depth_mult * (point.x() - m_minPoint.x()) / m_dimensions.x()), depth_mult - 1);
	m_cellY = std::min((int)std::floor(depth_mult * (point.y() - m_minPoint.y()) / m_dimensions.y()), depth_mult - 1);
	m_cellZ = std::min((int)std::floor(depth_mult * (point.z() - m_minPoint.z()) / m_dimensions.z()), depth_mult - 1);

	m_code = createCode(m_cellX, m_cellY, m_cellZ);
}

Morton::Morton(const Morton& other)
{
	if (this != &other)
	{
		m_minPoint = other.m_minPoint;
		m_dimensions = other.m_dimensions;
		m_cellX = other.m_cellX;
		m_cellY = other.m_cellY;
		m_cellZ = other.m_cellZ;
		m_code = createCode(m_cellX, m_cellY, m_cellZ);
	}
}

Morton& Morton::operator=(const Morton& other)
{
	if (this != &other)
	{
		m_minPoint = other.m_minPoint;
		m_dimensions = other.m_dimensions;
		m_cellX = other.m_cellX;
		m_cellY = other.m_cellY;
		m_cellZ = other.m_cellZ;
		m_code = createCode(m_cellX, m_cellY, m_cellZ);
	}
	return *this;
}

int Morton::spreadBits(int x, int offset)
{
	//......................9876543210
	x = (x | (x << 10)) & 0x000f801f; //............98765..........43210
	x = (x | (x << 4)) & 0x00e181c3; //........987....56......432....10
	x = (x | (x << 2)) & 0x03248649; //......98..7..5..6....43..2..1..0
	x = (x | (x << 2)) & 0x09249249; //....9..8..7..5..6..4..3..2..1..0

	return x << offset;
}

int Morton::compactBits(int x, int offset)
{
	x = (x >> offset) & 0x09249249;  //....9..8..7..5..6..4..3..2..1..0
	x = (x | (x >> 2)) & 0x03248649;  //......98..7..5..6....43..2..1..0                                          
	x = (x | (x >> 2)) & 0x00e181c3;  //........987....56......432....10                                       
	x = (x | (x >> 4)) & 0x000f801f;  //............98765..........43210                                          
	x = (x | (x >> 10)) & 0x000003FF;  //......................9876543210        

	return x;
}

Morton::code_t Morton::createCode(int cell_x, int cell_y, int cell_z)
{
	return spreadBits(cell_x, 0) | spreadBits(cell_y, 1) | spreadBits(cell_z, 2);
}

void Morton::decomposeCode(code_t code, int& cell_x, int& cell_y, int& cell_z)
{
	cell_x = compactBits(code, 0);
	cell_y = compactBits(code, 1);
	cell_z = compactBits(code, 2);
}

Morton::code_t Morton::extractLevelCode(code_t code, int level)
{
	return (code >> (nbits - 3 * (level + 1))) & 7;
}

Morton::code_t Morton::shiftLevelCode(code_t level_code, int level)
{
	return level_code << (nbits - 3 * (level + 1));
}
