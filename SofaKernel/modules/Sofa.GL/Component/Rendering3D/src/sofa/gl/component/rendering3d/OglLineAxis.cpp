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

#include "OglLineAxis.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/gl/gl.h>


namespace sofa::gl::component::rendering3d
{

int OglLineAxisClass = core::RegisterObject("Display scene axis")
        .add< OglLineAxis >()
        ;

using namespace sofa::defaulttype;

OglLineAxis::OglLineAxis()
    : axis(initData(&axis, std::string("xyz"),  "axis", "Axis to draw"))
    , size(initData(&size, (float)(10.0),  "size", "Size of the squared grid"))
    , thickness(initData(&thickness, (float)(1.0),  "thickness", "Thickness of the lines in the grid"))
    , draw(initData(&draw, true,  "draw", "Display the grid or not"))
    , drawX(true), drawY(true), drawZ(true)
{}

void OglLineAxis::init()
{
    updateVisual();
}

void OglLineAxis::reinit()
{
    updateVisual();
}

void OglLineAxis::updateVisual()
{
    std::string a = axis.getValue();

    drawX = a.find_first_of("xX")!=std::string::npos;
    drawY = a.find_first_of("yY")!=std::string::npos;
    drawZ = a.find_first_of("zZ")!=std::string::npos;
}

void OglLineAxis::drawVisual(const core::visual::VisualParams* /*vparams*/)
{
    if (!draw.getValue()) return;

    GLfloat s = size.getValue();

    glPushAttrib( GL_ALL_ATTRIB_BITS);

    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    if(drawX)
    {
        glColor4f( 1.0f, 0.0f, 0.0f, 1.0f );
        glVertex3f(-s*0.5f, 0.0f, 0.0f);
        glVertex3f( s*0.5f, 0.0f, 0.0f);
    }
    if (drawY)
    {
        glColor4f( 0.0f, 1.0f, 0.0f, 1.0f );
        glVertex3f(0.0f, -s*0.5f, 0.0f);
        glVertex3f(0.0f,  s*0.5f, 0.0f);
    }
    if (drawZ)
    {
        glColor4f( 0.0f, 0.0f, 1.0f, 1.0f );
        glVertex3f(0.0f, 0.0f, -s*0.5f);
        glVertex3f(0.0f, 0.0f, s*0.5f);
    }
    glEnd();


    glPopAttrib();

}


} // namespace sofa::gl::component::rendering3d
