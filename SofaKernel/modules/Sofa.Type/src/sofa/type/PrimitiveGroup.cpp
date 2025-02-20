/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/type/PrimitiveGroup.h>

#include <istream>
#include <ostream>

namespace sofa::type
{

std::ostream& operator << (std::ostream& out, const PrimitiveGroup &g)
{
    out << g.groupName << " " << g.materialName << " " << g.materialId << " " << g.p0 << " " << g.nbp;
    return out;
}

std::istream& operator >> (std::istream& in, PrimitiveGroup &g)
{
    in >> g.groupName >> g.materialName >> g.materialId >> g.p0 >> g.nbp;
    return in;
}

bool PrimitiveGroup::operator <(const PrimitiveGroup& p) const
{
    return p0 < p.p0;
}

PrimitiveGroup::PrimitiveGroup() : p0(0), nbp(0), materialId(-1) {}

PrimitiveGroup::PrimitiveGroup(int p0, int nbp, std::string materialName, std::string groupName, int materialId) : p0(p0), nbp(nbp), materialName(materialName), groupName(groupName), materialId(materialId) {}

} /// namespace sofa::type
