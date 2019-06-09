#ifndef MORTON_H
#define MORTON_H

#include "initSPModels.h"

#include <sofa/defaulttype/Vec.h>

using namespace sofa::defaulttype;

namespace sofa
{
	namespace component
	{
		namespace collision
		{
			class Morton
			{
				public:
					const static int levels = 10;
					const static int bits_per_level = 3;
					const static int nbits = levels * bits_per_level;

					const static int depth_mult = 1 << levels;

					typedef int code_t;

					Morton(const Vector3& point, const Vector3& minPoint, const Vector3& maxPoint);
					Morton(const Morton& other);
					Morton& operator=(const Morton& other);

					void decomposeCode(code_t code, int& cell_x, int& cell_y, int& cell_z);
					code_t extractLevelCode(code_t code, int level);
					code_t shiftLevelCode(code_t level_code, int level);

				private:
					int m_cellX, m_cellY, m_cellZ;
					code_t m_code;
					Vector3 m_minPoint, m_dimensions;

					int spreadBits(int x, int offset);
					int compactBits(int x, int offset);

					code_t createCode(int cell_x, int cell_y, int cell_z);
			};
		}
	}
}

#endif //MORTON_H

