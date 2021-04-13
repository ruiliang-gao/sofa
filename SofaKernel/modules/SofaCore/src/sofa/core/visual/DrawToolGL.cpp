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
#define SOFA_HELPER_GL_DRAWTOOLGL_CPP

#include <sofa/core/visual/DrawToolGL.h>

#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/helper/gl/BasicShapesGL.inl>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Cylinder.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/glText.inl>
#include <cmath>

namespace sofa
{

namespace helper
{

namespace gl
{

template class SOFA_CORE_API BasicShapesGL_Sphere< sofa::defaulttype::Vector3 >;
template class SOFA_CORE_API BasicShapesGL_FakeSphere< sofa::defaulttype::Vector3 >;

} // namespace gl

} // namespace helper

namespace core
{

namespace visual
{

using namespace sofa::defaulttype;
using namespace sofa::helper::gl;

DrawToolGL::DrawToolGL()
{
    clear();
    mLightEnabled = false;
    mWireFrameEnabled = false;
    mPolygonMode = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DrawToolGL::~DrawToolGL()
{
}

void DrawToolGL::init()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPoints(const std::vector<Vector3> &points, float size, const RGBAColor& color=RGBAColor(1.0f,1.0f,1.0f,1.0f))
{
    setMaterial(color);
    glPointSize(size);
    if (getLightEnabled()) disableLighting();
    glBegin(GL_POINTS);
    {
        for (std::size_t i=0; i<points.size(); ++i)
        {
            internalDrawPoint(points[i], color);
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    resetMaterial(color);
    glPointSize(1);
}

void DrawToolGL::drawPoints(const std::vector<Vector3> &points, float size, const std::vector<RGBAColor>& color)
{
    glPointSize(size);
    if (getLightEnabled()) disableLighting();
    glBegin(GL_POINTS);
    {
        for (std::size_t i=0; i<points.size(); ++i)
        {
            setMaterial(color[i]);
            internalDrawPoint(points[i], color[i]);
            resetMaterial(color[i]);
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    glPointSize(1);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::internalDrawLine(const Vector3 &p1, const Vector3 &p2, const RGBAColor& color)
{
    internalDrawPoint(p1, color );
    internalDrawPoint(p2, color );
}

void DrawToolGL::drawLine(const Vector3 &p1, const Vector3 &p2, const RGBAColor& color)
{
    glBegin(GL_LINES);
    internalDrawLine(p1,p2,color);
    glEnd();
}

void DrawToolGL::drawInfiniteLine(const Vector3 &point, const Vector3 &direction, const RGBAColor& color)
{
    glBegin(GL_LINES);
    glColor4f(color[0],color[1],color[2],color[3]);
    glVertex4d(point[0], point[1], point[2], 1.0);
    glVertex4d(direction[0], direction[1], direction[2], 0.0);
    glEnd();
}

void DrawToolGL::drawLines(const std::vector<Vector3> &points, float size, const RGBAColor& color)
{
    setMaterial(color);
    glLineWidth(size);
    if (getLightEnabled()) disableLighting();
    glBegin(GL_LINES);
    {
        for (std::size_t i=0; i<points.size()/2; ++i)
        {
            internalDrawLine(points[2*i],points[2*i+1]  , color );
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    glLineWidth(1);
    resetMaterial(color);
}

void DrawToolGL::drawLines(const std::vector<Vector3> &points, float size, const std::vector<RGBAColor>& colors)
{
    if (points.size() != colors.size()*2)
    {
        msg_warning("DrawToolGL") << "Sizes mismatch in drawLines method, points.size(): " << points.size() << " should be equal to colors.size()*2: " << colors.size()*2;
        return drawLines(points, size, RGBAColor::red());
    }

    glLineWidth(size);
    if (getLightEnabled()) disableLighting();
    glBegin(GL_LINES);
    {
        for (std::size_t i=0; i<points.size()/2; ++i)
        {
            setMaterial(colors[i]);
            internalDrawLine(points[2*i],points[2*i+1], colors[i] );
            resetMaterial(colors[i]);
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    glLineWidth(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawLines(const std::vector<Vector3> &points, const std::vector< defaulttype::Vec<2,int> > &index, float size, const RGBAColor& color=RGBAColor(1.0f,1.0f,1.0f,1.0f))
{
    setMaterial(color);
    glLineWidth(size);
    if (getLightEnabled()) disableLighting();
    glBegin(GL_LINES);
    {
        for (std::size_t i=0; i<index.size(); ++i)
        {
            internalDrawLine(points[ index[i][0] ],points[ index[i][1] ], color );
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    resetMaterial(color);
    glLineWidth(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawLineStrip(const std::vector<Vector3> &points, float size, const RGBAColor& color)
{
    setMaterial(color);
    glLineWidth(size);
    if (getLightEnabled()) disableLighting();
    glBegin(GL_LINE_STRIP);
    {
        for (std::size_t i=0; i<points.size(); ++i)
        {
            internalDrawPoint(points[i]  , color );
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    resetMaterial(color);
    glLineWidth(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawLineLoop(const std::vector<Vector3> &points, float size, const RGBAColor& color)
{
    setMaterial(color);
    glLineWidth(size);
    if (getLightEnabled()) disableLighting();
    glBegin(GL_LINE_LOOP);
    {
        for (std::size_t i=0; i<points.size(); ++i)
        {
            internalDrawPoint(points[i]  , color );
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    resetMaterial(color);
    glLineWidth(1);
}

void DrawToolGL::drawDisk(float radius, double from, double to, int resolution, const RGBAColor& color)
{
    if (from > to)
        to += 2.0 * M_PI;
    glBegin(GL_TRIANGLES);
    {
        glColor4f(color.r(), color.g(), color.b(), color.a());
        bool first = true;
        float prev_alpha = 0;
        float prev_beta = 0;
        bool stop = false;
        for (int i  = 0 ; i <= resolution ; ++i)
        {
            double angle = (double(i) / double(resolution) * 2.0 * M_PI) + from;
            if(angle >= to)
            {
                angle = to;
                stop = true;
            }
            float alpha = float(std::sin(angle));
            float beta = float(std::cos(angle));

            if (first)
            {
                first = false;
                prev_alpha = alpha;
                prev_beta = beta;
            }
            glVertex3f(0.0, 0.0, 0.0);
            glVertex3f(radius * prev_alpha, radius * prev_beta, 0.0);
            glVertex3f(radius * alpha, radius * beta, 0.0);
            if (stop)
                break;
            prev_alpha = alpha;
            prev_beta = beta;
        }
    }
    glEnd();
}

void DrawToolGL::drawCircle(float radius, float lineThickness, int resolution, const RGBAColor& color)
{
    glLineWidth(lineThickness);
    glEnable(GL_LINE_SMOOTH);

    glBegin(GL_LINE_STRIP);
    {
        glColor4f(color.r(), color.g(), color.b(), color.a());
        for (int i  = 0 ; i <= resolution ; ++i)
        {
            float angle = float(double(i) / double(resolution) * 2.0 * M_PI);
            float alpha = std::sin(angle);
            float beta = std::cos(angle);

            glVertex3f(radius * alpha, radius * beta, 0.0);
        }
    }
    glEnd();

    glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.0f);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points, const RGBAColor& color)
{
    setMaterial(color);
    glBegin(GL_TRIANGLES);
    {
        for (std::size_t i=0; i<points.size()/3; ++i)
        {
            const Vector3& a = points[ 3*i+0 ];
            const Vector3& b = points[ 3*i+1 ];
            const Vector3& c = points[ 3*i+2 ];
            Vector3 n = cross((b-a),(c-a));
            n.normalize();
            internalDrawTriangle(a,b,c,n,color);
        }
    } glEnd();
    resetMaterial(color);
}

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points, const std::vector< RGBAColor > &color)
{
    std::vector<Vector3> normal;
    normal.clear();
    this->drawTriangles(points,normal,color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points, const Vector3& normal, const RGBAColor& color)
{
    setMaterial(color);
    glBegin(GL_TRIANGLES);
    {
        for (std::size_t i=0; i<points.size()/3; ++i)
            internalDrawTriangle(points[ 3*i+0 ],points[ 3*i+1 ],points[ 3*i+2 ], normal, color);
    } glEnd();
    resetMaterial(color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points, const std::vector< defaulttype::Vec<3,int> > &index,
        const std::vector<Vector3> &normal, const RGBAColor& color=RGBAColor(1.0f,1.0f,1.0f,1.0f))
{
    setMaterial(color);
    glBegin(GL_TRIANGLES);
    {
        for (std::size_t i=0; i<index.size(); ++i)
        {
            internalDrawTriangle(points[ index[i][0] ],points[ index[i][1] ],points[ index[i][2] ],normal[i],color);
        }
    } glEnd();
    resetMaterial(color);
}

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points,
    const std::vector< Vec3i > &index,
    const std::vector<Vector3> &normal,
    const std::vector<RGBAColor>& colour)
{
    //todo !
    SOFA_UNUSED(points);
    SOFA_UNUSED(index);
    SOFA_UNUSED(normal);
    SOFA_UNUSED(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points,
        const std::vector<Vector3> &normal, const std::vector< RGBAColor > &color)
{
    const std::size_t nbTriangles=points.size()/3;
    bool computeNormals= (normal.size() != nbTriangles);
    if (nbTriangles == 0) return;
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    setMaterial(color[0]);
    glBegin(GL_TRIANGLES);
    {
        for (std::size_t i=0; i<nbTriangles; ++i)
        {
            if (!computeNormals)
            {
                internalDrawTriangle(points[3*i+0],points[3*i+1],points[3*i+2],normal[i],
                        color[3*i+0],color[3*i+1],color[3*i+2]);
            }
            else
            {
                const Vector3& a = points[ 3*i+0 ];
                const Vector3& b = points[ 3*i+1 ];
                const Vector3& c = points[ 3*i+2 ];
                Vector3 n = cross((b-a),(c-a));
                n.normalize();

                internalDrawPoint(a,n,color[3*i+0]);
                internalDrawPoint(b,n,color[3*i+1]);
                internalDrawPoint(c,n,color[3*i+2]);

            }
        }
    } glEnd();
    glDisable(GL_COLOR_MATERIAL);
    resetMaterial(color[0]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangleStrip(const std::vector<Vector3> &points,
        const std::vector<Vector3>  &normal,
        const RGBAColor& color)
{
    setMaterial(color);
    glBegin(GL_TRIANGLE_STRIP);
    {
        for (std::size_t i=0; i<normal.size(); ++i)
        {
            glNormalT(normal[i]);
            glVertexNv<3>(points[2*i].ptr());
            glVertexNv<3>(points[2*i+1].ptr());
        }
    } glEnd();
    resetMaterial(color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangleFan(const std::vector<Vector3> &points,
        const std::vector<Vector3>  &normal,
        const RGBAColor& color)
{
    if (points.size() < 3) return;
    setMaterial(color);
    glBegin(GL_TRIANGLE_FAN);

    glNormalT(normal[0]);
    glVertexNv<3>(points[0].ptr());
    glVertexNv<3>(points[1].ptr());
    glVertexNv<3>(points[2].ptr());

    for (std::size_t i=3; i<points.size(); ++i)
    {
        glNormalT(normal[i]);
        glVertexNv<3>(points[i].ptr());
    }

    glEnd();
    resetMaterial(color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawFrame(const Vector3& position, const Quaternion &orientation, const Vec<3,float> &size)
{
    setPolygonMode(0,false);
    helper::gl::Axis::draw(position, orientation, size);
}
void DrawToolGL::drawFrame(const Vector3& position, const Quaternion &orientation, const Vec<3,float> &size, const RGBAColor &color)
{
    setPolygonMode(0,false);
    helper::gl::Axis::draw(position, orientation, size, color, color, color);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawSpheres(const std::vector<Vector3> &points, float radius, const RGBAColor& color)
{
    setMaterial(color);

    m_sphereUtil.draw(points, radius);

    resetMaterial(color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawSpheres(const std::vector<Vector3> &points, const std::vector<float>& radius, const RGBAColor& color)
{
    setMaterial(color);

    m_sphereUtil.draw(points, radius);

    resetMaterial(color);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawFakeSpheres(const std::vector<Vector3> &points, float radius, const RGBAColor& color)
{
    setMaterial(color);

    m_fakeSphereUtil.draw(points, radius);

    resetMaterial(color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawFakeSpheres(const std::vector<Vector3> &points, const std::vector<float>& radius, const RGBAColor& color)
{
    setMaterial(color);

    m_fakeSphereUtil.draw(points, radius);

    resetMaterial(color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawCapsule(const Vector3& p1, const Vector3 &p2, float radius,const RGBAColor& color, int subd){
    Vector3 tmp = p2-p1;
    setMaterial(color);
    /* create Vectors p and q, co-planar with the capsules's cross-sectional disk */
    Vector3 p=tmp;
    if (fabs(p[0]) + fabs(p[1]) < 0.00001*tmp.norm())
        p[0] += 1.0;
    else
        p[2] += 1.0;
    Vector3 q;
    q = p.cross(tmp);
    p = tmp.cross(q);
    /* do the normalization outside the segment loop */
    p.normalize();
    q.normalize();

    int i2;
    /* build the cylinder part of the capsule from rectangular subd */
    std::vector<Vector3> points;
    std::vector<Vector3> normals;

    for (i2=0 ; i2<=subd ; i2++)
    {
        /* sweep out a circle */
        float theta =  (float)( i2 * 2.0f * M_PI / subd );
        float st = sin(theta);
        float ct = cos(theta);
        /* construct normal */
        tmp = p*ct+q*st;
        /* set the normal for the two subseqent points */
        normals.push_back(tmp);

        Vector3 w(p1);
        w += tmp*fabs(radius);
        points.push_back(w);

        w=p2;
        w += tmp*fabs(radius);
        points.push_back(w);
    }

    //we draw here the cylinder part
    drawTriangleStrip(points, normals,color);

    //now we must draw the two hemispheres
    //but it's easier to draw spheres...
    drawSphere(p1,radius);
    drawSphere(p2,radius);

    resetMaterial(color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawCone(const Vector3& p1, const Vector3 &p2, float radius1, float radius2, const RGBAColor& color, int subd)
{
    Vector3 tmp = p2-p1;
    setMaterial(color);
    /* create Vectors p and q, co-planar with the cylinder's cross-sectional disk */
    Vector3 p=tmp;
    if (fabs(p[0]) + fabs(p[1]) < 0.00001*tmp.norm())
        p[0] += 1.0;
    else
        p[2] += 1.0;
    Vector3 q;
    q = p.cross(tmp);
    p = tmp.cross(q);
    /* do the normalization outside the segment loop */
    p.normalize();
    q.normalize();

    int i2;

    /* build the cylinder from rectangular subd */
    std::vector<Vector3> points;
    std::vector<Vector3> normals;

    std::vector<Vector3> pointsCloseCylinder1;
    std::vector<Vector3> normalsCloseCylinder1;
    std::vector<Vector3> pointsCloseCylinder2;
    std::vector<Vector3> normalsCloseCylinder2;

    Vector3 dir=p1-p2; dir.normalize();
    pointsCloseCylinder1.push_back(p1);
    normalsCloseCylinder1.push_back(dir);
    pointsCloseCylinder2.push_back(p2);
    normalsCloseCylinder2.push_back(-dir);


    for (i2=0 ; i2<=subd ; i2++)
    {
        /* sweep out a circle */
        float theta =  (float)( i2 * 2.0f * M_PI / subd );
        float st = sin(theta);
        float ct = cos(theta);
        /* construct normal */
        tmp = p*ct+q*st;
        /* set the normal for the two subseqent points */
        normals.push_back(tmp);

        /* point on disk 1 */
        Vector3 w(p1);
        w += tmp*fabs(radius1);
        points.push_back(w);
        pointsCloseCylinder1.push_back(w);
        normalsCloseCylinder1.push_back(dir);

        /* point on disk 2 */
        w=p2;
        w += tmp*fabs(radius2);
        points.push_back(w);
        pointsCloseCylinder2.push_back(w);
        normalsCloseCylinder2.push_back(-dir);
    }
    pointsCloseCylinder1.push_back(pointsCloseCylinder1[1]);
    normalsCloseCylinder1.push_back(normalsCloseCylinder1[1]);
    pointsCloseCylinder2.push_back(pointsCloseCylinder2[1]);
    normalsCloseCylinder2.push_back(normalsCloseCylinder2[1]);


    drawTriangleStrip(points, normals,color);
    if (radius1 > 0) drawTriangleFan(pointsCloseCylinder1, normalsCloseCylinder1,color);
    if (radius2 > 0) drawTriangleFan(pointsCloseCylinder2, normalsCloseCylinder2,color);

    resetMaterial(color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawCube( const float& radius, const RGBAColor& color, const int& subd)
{
    // X Axis
    drawCylinder( Vector3(-1.0, -1.0, -1.0), Vector3(1.0, -1.0, -1.0), radius, color, subd);
    drawCylinder( Vector3(-1.0,  1.0, -1.0), Vector3(1.0,  1.0, -1.0), radius, color, subd);
    drawCylinder( Vector3(-1.0, -1.0,  1.0), Vector3(1.0, -1.0,  1.0), radius, color, subd);
    drawCylinder( Vector3(-1.0,  1.0,  1.0), Vector3(1.0,  1.0,  1.0), radius, color, subd);
    // Y Axis
    drawCylinder( Vector3(-1.0, -1.0, -1.0), Vector3(-1.0, 1.0, -1.0), radius, color, subd);
    drawCylinder( Vector3(-1.0, -1.0,  1.0), Vector3(-1.0, 1.0,  1.0), radius, color, subd);
    drawCylinder( Vector3( 1.0, -1.0, -1.0), Vector3( 1.0, 1.0, -1.0), radius, color, subd);
    drawCylinder( Vector3( 1.0, -1.0,  1.0), Vector3( 1.0, 1.0,  1.0), radius, color, subd);
    // Z Axis
    drawCylinder( Vector3(-1.0, -1.0, -1.0), Vector3(-1.0, -1.0, 1.0), radius, color, subd);
    drawCylinder( Vector3(-1.0,  1.0, -1.0), Vector3(-1.0,  1.0, 1.0), radius, color, subd);
    drawCylinder( Vector3( 1.0, -1.0, -1.0), Vector3( 1.0, -1.0, 1.0), radius, color, subd);
    drawCylinder( Vector3( 1.0,  1.0, -1.0), Vector3( 1.0,  1.0, 1.0), radius, color, subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawCylinder(const Vector3& p1, const Vector3 &p2, float radius, const RGBAColor& color, int subd)
{
    drawCone( p1,p2,radius,radius,color,subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawArrow(const Vector3& p1, const Vector3 &p2, float radius, const RGBAColor& color, int subd)
{
    Vector3 p3 = p1*.2+p2*.8;
    drawCylinder( p1,p3,radius,color,subd);
    drawCone( p3,p2,radius*2.5f,0,color,subd);
}


void DrawToolGL::drawArrow(const Vector3& p1, const Vector3 &p2, float radius, float coneLength, const RGBAColor& color, int subd)
{
    drawArrow(p1, p2, radius, coneLength, radius * 2.5f, color, subd);
}

void DrawToolGL::drawArrow   (const Vector3& p1, const Vector3 &p2, float radius, float coneLength, float coneRadius, const RGBAColor& color, int subd)
{
    // fixed coneLength ; cone can be stretched or when its length depends on the total arrow length

    Vector3 a = p2 - p1;
    SReal n = a.norm();
    if( coneLength >= n )
        drawCone( p1,p2,coneRadius,0,color,subd);
    else
    {
        a /= n; // normalizing
        Vector3 p3 = p2 - coneLength*a;
        drawCylinder( p1,p3,radius,color,subd);
        drawCone( p3,p2,coneRadius,0,color,subd);
    }
}


void DrawToolGL::drawCross(const Vector3&p, float length, const RGBAColor& color)
{
    std::vector<sofa::defaulttype::Vector3> bounds;

    for ( unsigned int i=0 ; i<3 ; i++ )
    {
        sofa::defaulttype::Vector3 p0 = p;
        sofa::defaulttype::Vector3 p1 = p;

        p0[i] -= length;
        p1[i] += length;

        bounds.push_back(p0);
        bounds.push_back(p1);
    }
    drawLines(bounds, 1, color);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPlus ( const float& radius, const RGBAColor& color, const int& subd)
{
    drawCylinder( Vector3(-1.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), radius, color, subd);
    drawCylinder( Vector3(0.0, -1.0, 0.0), Vector3(0.0, 1.0, 0.0), radius, color, subd);
    drawCylinder( Vector3(0.0, 0.0, -1.0), Vector3(0.0, 0.0, 1.0), radius, color, subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::internalDrawPoint(const Vector3 &p, const RGBAColor &c)
{
    glColor4f(c[0],c[1],c[2],c[3]);
    glVertexNv<3>(p.ptr());
}

void DrawToolGL::internalDrawPoint(const Vector3 &p, const Vector3 &n, const RGBAColor &c)
{
    glColor4f(c[0],c[1],c[2],c[3]);
    glNormalT(n);
    glVertexNv<3>(p.ptr());
}


void DrawToolGL::drawPoint(const Vector3 &p, const RGBAColor &c)
{
    glBegin(GL_POINTS);
    internalDrawPoint(p,c);
    glEnd();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPoint(const Vector3 &p, const Vector3 &n, const RGBAColor &c)
{
    glBegin(GL_POINTS);
    internalDrawPoint(p, n, c);
    glEnd();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::internalDrawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3)
{
    glNormalT(normal);
    glColor4fv(c1.array());
    glVertexNv<3>(p1.ptr());
    glColor4fv(c2.array());
    glVertexNv<3>(p2.ptr());
    glColor4fv(c3.array());
    glVertexNv<3>(p3.ptr());
}


void DrawToolGL::internalDrawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3)
{
    glNormalT(normal1);
    glColor4fv(c1.array());
    glVertexNv<3>(p1.ptr());
    glNormalT(normal2);
    glColor4fv(c2.array());
    glVertexNv<3>(p2.ptr());
    glNormalT(normal3);
    glColor4fv(c3.array());
    glVertexNv<3>(p3.ptr());
}


void DrawToolGL::internalDrawTriangle( const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,
        const Vector3 &normal, const  RGBAColor &c)
{
    glNormalT(normal);
    glColor4fv(c.array());
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
}


void DrawToolGL::internalDrawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal)
{
    glNormalT(normal);
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
}

void DrawToolGL::drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3)
{
    glBegin(GL_TRIANGLES);
    internalDrawTriangle(p1, p2, p3, normal, c1, c2, c3);
    glEnd();
}


void DrawToolGL::drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3)
{
    glBegin(GL_TRIANGLES);
    internalDrawTriangle(p1, p2, p3, normal1, normal2, normal3, c1, c2, c3);
    glEnd();
}


void DrawToolGL::drawTriangle( const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,
        const Vector3 &normal, const  RGBAColor &c)
{
    glBegin(GL_TRIANGLES);
    internalDrawTriangle(p1, p2, p3, normal, c);
    glEnd();
}


void DrawToolGL::drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal)
{
    glBegin(GL_TRIANGLES);
    internalDrawTriangle(p1, p2, p3, normal);
    glEnd();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::internalDrawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal)
{
    glNormalT(normal);
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
    glVertexNv<3>(p4.ptr());
}

void DrawToolGL::internalDrawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal, const RGBAColor &c)
{
    glNormalT(normal);
    glColor4fv(c.array());
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
    glVertexNv<3>(p4.ptr());
}

void DrawToolGL::internalDrawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3, const RGBAColor &c4)
{
    glNormalT(normal);
    glColor4fv(c1.array());
    glVertexNv<3>(p1.ptr());
    glColor4fv(c2.array());
    glVertexNv<3>(p2.ptr());
    glColor4fv(c3.array());
    glVertexNv<3>(p3.ptr());
    glColor4fv(c4.array());
    glVertexNv<3>(p4.ptr());
}

void DrawToolGL::internalDrawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3, const Vector3 &normal4,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3, const RGBAColor &c4)
{
    glNormalT(normal1);
    glColor4fv(c1.array());
    glVertexNv<3>(p1.ptr());
    glNormalT(normal2);
    glColor4fv(c2.array());
    glVertexNv<3>(p2.ptr());
    glNormalT(normal3);
    glColor4fv(c3.array());
    glVertexNv<3>(p3.ptr());
    glNormalT(normal4);
    glColor4fv(c4.array());
    glVertexNv<3>(p4.ptr());
}


void DrawToolGL::drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3, const RGBAColor &c4)
{
    glBegin(GL_QUADS);
    internalDrawQuad(p1, p2, p3, p4, normal, c1, c2, c3, c4);
    glEnd();
}


void DrawToolGL::drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3, const Vector3 &normal4,
        const RGBAColor &c1, const RGBAColor &c2, const RGBAColor &c3, const RGBAColor &c4)
{
    glBegin(GL_QUADS);
    internalDrawQuad(p1, p2, p3, p4, normal1, normal2, normal3, normal4, c1, c2, c3, c4);
    glEnd();
}


void DrawToolGL::drawQuad( const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal, const  RGBAColor &c)
{
    glBegin(GL_QUADS);
    internalDrawQuad(p1, p2, p3, p4, normal, c);
    glEnd();
}


void DrawToolGL::drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal)
{
    glBegin(GL_QUADS);
    internalDrawQuad(p1, p2, p3, p4, normal);
    glEnd();
}

void DrawToolGL::drawQuads(const std::vector<Vector3> &points, const RGBAColor& color)
{
    setMaterial(color);
    glBegin(GL_QUADS);
    {
        for (std::size_t i=0; i<points.size()/4; ++i)
        {
            const Vector3& a = points[ 4*i+0 ];
            const Vector3& b = points[ 4*i+1 ];
            const Vector3& c = points[ 4*i+2 ];
            const Vector3& d = points[ 4*i+3 ];
            Vector3 n = cross((b-a),(c-a));
            n.normalize();
            internalDrawQuad(a,b,c,d,n,color);
        }
    } glEnd();
    resetMaterial(color);
}

void DrawToolGL::drawQuads(const std::vector<Vector3> &points, const std::vector<RGBAColor>& colors)
{
    glBegin(GL_QUADS);
    {
        for (std::size_t i=0; i<points.size()/4; ++i)
        {
            const Vector3& a = points[ 4*i+0 ];
            const Vector3& b = points[ 4*i+1 ];
            const Vector3& c = points[ 4*i+2 ];
            const Vector3& d = points[ 4*i+3 ];

            const RGBAColor& col_a = colors[ 4*i+0 ];
            const RGBAColor& col_b = colors[ 4*i+1 ];
            const RGBAColor& col_c = colors[ 4*i+2 ];
            const RGBAColor& col_d = colors[ 4*i+3 ];

            RGBAColor average_color;
            for(int i=0; i<4; i++)
            {
                average_color[i] = (col_a[i]+col_b[i]+col_c[i]+col_d[i])*0.25f;
            }

            Vector3 n = cross((b-a),(c-a));
            n.normalize();
            internalDrawQuad(a,b,c,d,n,average_color);
        }
    } glEnd();
}

void DrawToolGL::drawTetrahedron(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &p3, const RGBAColor &color)
{
    setMaterial(color);
    glBegin(GL_TRIANGLES);
    {
        this->internalDrawTriangle(p0,p1,p2, cross((p1-p0),(p2-p0)), color);
        this->internalDrawTriangle(p0,p1,p3, cross((p1-p0),(p3-p0)), color);
        this->internalDrawTriangle(p0,p2,p3, cross((p2-p0),(p3-p0)), color);
        this->internalDrawTriangle(p1,p2,p3, cross((p2-p1),(p3-p1)), color);
    } glEnd();
    resetMaterial(color);
}

void DrawToolGL::drawTetrahedra(const std::vector<Vector3> &points, const RGBAColor &color)
{
    setMaterial(color);
    glBegin(GL_TRIANGLES);
    for (std::vector<Vector3>::const_iterator it = points.begin(), end = points.end(); it != end;)
    {
        const Vector3& p0 = *(it++);
        const Vector3& p1 = *(it++);
        const Vector3& p2 = *(it++);
        const Vector3& p3 = *(it++);

        //this->drawTetrahedron(p0,p1,p2,p3,color); // not recommanded as it will call glBegin/glEnd <number of tetra> times
        this->internalDrawTriangle(p0, p1, p2, cross((p1 - p0), (p2 - p0)), color);
        this->internalDrawTriangle(p0, p1, p3, cross((p1 - p0), (p3 - p0)), color);
        this->internalDrawTriangle(p0, p2, p3, cross((p2 - p0), (p3 - p0)), color);
        this->internalDrawTriangle(p1, p2, p3, cross((p2 - p1), (p3 - p1)), color);
    }
    glEnd();
    resetMaterial(color);
}

void DrawToolGL::drawScaledTetrahedra(const std::vector<Vector3> &points, const RGBAColor &color, const float scale)
{
    setMaterial(color);
    glBegin(GL_TRIANGLES);
    for (std::vector<Vector3>::const_iterator it = points.begin(), end = points.end(); it != end;)
    {
        const Vector3& p0 = *(it++);
        const Vector3& p1 = *(it++);
        const Vector3& p2 = *(it++);
        const Vector3& p3 = *(it++);

        Vector3 center = (p0 + p1 + p2 + p3) / 4.0;

        Vector3 np0 = ((p0 - center)*scale) + center;
        Vector3 np1 = ((p1 - center)*scale) + center;
        Vector3 np2 = ((p2 - center)*scale) + center;
        Vector3 np3 = ((p3 - center)*scale) + center;

        //this->drawTetrahedron(p0,p1,p2,p3,color); // not recommanded as it will call glBegin/glEnd <number of tetra> times
        this->internalDrawTriangle(np0, np1, np2, cross((p1 - p0), (p2 - p0)), color);
        this->internalDrawTriangle(np0, np1, np3, cross((p1 - p0), (p3 - p0)), color);
        this->internalDrawTriangle(np0, np2, np3, cross((p2 - p0), (p3 - p0)), color);
        this->internalDrawTriangle(np1, np2, np3, cross((p2 - p1), (p3 - p1)), color);
    }
    glEnd();
    resetMaterial(color);
}


void DrawToolGL::drawHexahedron(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,
                                const Vector3 &p4, const Vector3 &p5, const Vector3 &p6, const Vector3 &p7,
                                const RGBAColor &color)
{
    //{{0,1,2,3}, {4,7,6,5}, {1,0,4,5},{1,5,6,2},  {2,6,7,3}, {0,3,7,4}}
    setMaterial(color);
    glBegin(GL_QUADS);
    {
        this->internalDrawQuad(p0, p1, p2, p3, cross((p1 - p0), (p2 - p0)), color);
        this->internalDrawQuad(p4, p7, p6, p5, cross((p7 - p5), (p6 - p5)), color);
        this->internalDrawQuad(p1, p0, p4, p5, cross((p0 - p1), (p4 - p1)), color);
        this->internalDrawQuad(p1, p5, p6, p2, cross((p5 - p1), (p6 - p1)), color);
        this->internalDrawQuad(p2, p6, p7, p3, cross((p6 - p2), (p7 - p2)), color);
        this->internalDrawQuad(p0, p3, p7, p4, cross((p3 - p0), (p7 - p0)), color);
    } glEnd();
    resetMaterial(color);
}

void DrawToolGL::drawHexahedra(const std::vector<Vector3> &points, const RGBAColor& color)
{
    setMaterial(color);

    glBegin(GL_QUADS);
    for (std::vector<Vector3>::const_iterator it = points.begin(), end = points.end(); it != end;)
    {
        const Vector3& p0 = *(it++);
        const Vector3& p1 = *(it++);
        const Vector3& p2 = *(it++);
        const Vector3& p3 = *(it++);
        const Vector3& p4 = *(it++);
        const Vector3& p5 = *(it++);
        const Vector3& p6 = *(it++);
        const Vector3& p7 = *(it++);

        //this->drawHexahedron(p0,p1,p2,p3,p4,p5,p6,p7,color); // not recommanded as it will call glBegin/glEnd <number of hexa> times
        this->internalDrawQuad(p0, p1, p2, p3, cross((p1 - p0), (p2 - p0)), color);
        this->internalDrawQuad(p4, p7, p6, p5, cross((p7 - p5), (p6 - p5)), color);
        this->internalDrawQuad(p1, p0, p4, p5, cross((p0 - p1), (p4 - p1)), color);
        this->internalDrawQuad(p1, p5, p6, p2, cross((p5 - p1), (p6 - p1)), color);
        this->internalDrawQuad(p2, p6, p7, p3, cross((p6 - p2), (p7 - p2)), color);
        this->internalDrawQuad(p0, p3, p7, p4, cross((p3 - p0), (p7 - p0)), color);
    }
    glEnd();
    resetMaterial(color);
}

void DrawToolGL::drawScaledHexahedra(const std::vector<Vector3> &points, const RGBAColor& color, const float scale)
{
    setMaterial(color);

    glBegin(GL_QUADS);
    for (std::vector<Vector3>::const_iterator it = points.begin(), end = points.end(); it != end;)
    {
        const Vector3& p0 = *(it++);
        const Vector3& p1 = *(it++);
        const Vector3& p2 = *(it++);
        const Vector3& p3 = *(it++);
        const Vector3& p4 = *(it++);
        const Vector3& p5 = *(it++);
        const Vector3& p6 = *(it++);
        const Vector3& p7 = *(it++);

        //barycenter
        Vector3 center = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7)/8.0;

        Vector3 np0 = ((p0 - center)*scale) + center;
        Vector3 np1 = ((p1 - center)*scale) + center;
        Vector3 np2 = ((p2 - center)*scale) + center;
        Vector3 np3 = ((p3 - center)*scale) + center;
        Vector3 np4 = ((p4 - center)*scale) + center;
        Vector3 np5 = ((p5 - center)*scale) + center;
        Vector3 np6 = ((p6 - center)*scale) + center;
        Vector3 np7 = ((p7 - center)*scale) + center;

        //this->drawHexahedron(p0,p1,p2,p3,p4,p5,p6,p7,color); // not recommanded as it will call glBegin/glEnd <number of hexa> times
        this->internalDrawQuad(np0, np1, np2, np3, cross((p1 - p0), (p2 - p0)), color);
        this->internalDrawQuad(np4, np7, np6, np5, cross((p7 - p5), (p6 - p5)), color);
        this->internalDrawQuad(np1, np0, np4, np5, cross((p0 - p1), (p4 - p1)), color);
        this->internalDrawQuad(np1, np5, np6, np2, cross((p5 - p1), (p6 - p1)), color);
        this->internalDrawQuad(np2, np6, np7, np3, cross((p6 - p2), (p7 - p2)), color);
        this->internalDrawQuad(np0, np3, np7, np4, cross((p3 - p0), (p7 - p0)), color);
    }
    glEnd();
    resetMaterial(color);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawSphere( const Vector3 &p, float radius)
{
    glPushMatrix();
    helper::gl::drawSphere(p, radius, 32, 16);
    glPopMatrix();
}

void DrawToolGL::drawSphere(const Vector3 &p, float radius, const RGBAColor &color)
{
    setMaterial(color);
    glPushMatrix();
    helper::gl::drawSphere(p, radius, 32, 16);
    glPopMatrix();
    resetMaterial(color);
}

void DrawToolGL::drawEllipsoid(const Vector3 &p, const Vector3 &radii)
{
    glPushMatrix();
    helper::gl::drawEllipsoid(p, (float)radii[0], (float)radii[1], (float)radii[2], 32, 16);
    glPopMatrix();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawBoundingBox( const Vector3 &min, const Vector3 &max, float size)
{
    glLineWidth(size);
    glBegin( GL_LINES );

    // 0-1
    glVertex3f( (float)min[0], (float)min[1], (float)min[2] );
    glVertex3f( (float)max[0], (float)min[1], (float)min[2] );
    // 2-3
    glVertex3f( (float)max[0], (float)max[1], (float)min[2] );
    glVertex3f( (float)min[0], (float)max[1], (float)min[2] );
    // 4-5
    glVertex3f( (float)min[0], (float)min[1], (float)max[2] );
    glVertex3f( (float)max[0], (float)min[1], (float)max[2] );
    // 6-7
    glVertex3f( (float)max[0], (float)max[1], (float)max[2] );
    glVertex3f( (float)min[0], (float)max[1], (float)max[2] );
    // 0-3
    glVertex3f( (float)min[0], (float)min[1], (float)min[2] );
    glVertex3f( (float)min[0], (float)max[1], (float)min[2] );
    // 1-2
    glVertex3f( (float)max[0], (float)min[1], (float)min[2] );
    glVertex3f( (float)max[0], (float)max[1], (float)min[2] );
    // 4-7
    glVertex3f( (float)min[0], (float)min[1], (float)max[2] );
    glVertex3f( (float)min[0], (float)max[1], (float)max[2] );
    // 5-6
    glVertex3f( (float)max[0], (float)min[1], (float)max[2] );
    glVertex3f( (float)max[0], (float)max[1], (float)max[2] );
    // 0-4
    glVertex3f( (float)min[0], (float)min[1], (float)min[2] );
    glVertex3f( (float)min[0], (float)min[1], (float)max[2] );
    // 1-5
    glVertex3f( (float)max[0], (float)min[1], (float)min[2] );
    glVertex3f( (float)max[0], (float)min[1], (float)max[2] );
    // 2-6
    glVertex3f( (float)max[0], (float)max[1], (float)min[2] );
    glVertex3f( (float)max[0], (float)max[1], (float)max[2] );
    // 3-7
    glVertex3f( (float)min[0], (float)max[1], (float)min[2] );
    glVertex3f( (float)min[0], (float)max[1], (float)max[2] );

    glEnd();
    glLineWidth(1.0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::setPolygonMode(int _mode, bool _wireframe)
{
    mPolygonMode=_mode;
    mWireFrameEnabled=_wireframe;
    if (!mPolygonMode)
    {
        if (mWireFrameEnabled) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        else                  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    else if (mPolygonMode == 1)
    {
        if (mWireFrameEnabled) glPolygonMode(GL_FRONT, GL_LINE);
        else                  glPolygonMode(GL_FRONT, GL_FILL);
    }
    else if (mPolygonMode == 2)
    {
        if (mWireFrameEnabled) glPolygonMode(GL_BACK, GL_LINE);
        else                  glPolygonMode(GL_BACK, GL_FILL);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::setLightingEnabled(bool _isAnabled)
{
    mLightEnabled = _isAnabled;
    if (this->getLightEnabled()) enableLighting();
    else disableLighting();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::setMaterial(const RGBAColor &color)
{
    glColor4f(color[0],color[1],color[2],color[3]);
    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, &color[0]);
    static const float emissive[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
    static const float specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
    glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive);
    glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular);
    glMaterialf  (GL_FRONT_AND_BACK, GL_SHININESS, 20);
    if (color[3] < 1)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(0);
    }
    else
    {
        glDisable(GL_BLEND);
        glDepthMask(1);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::resetMaterial(const RGBAColor &color)
{
    if (color[3] < 1)
    {
        resetMaterial();
    }
}

void DrawToolGL::resetMaterial()
{
    glDisable(GL_BLEND);
    glDepthMask(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::clear()
{
    helper::gl::Axis::clear();
    helper::gl::Cylinder::clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::pushMatrix()
{
    glPushMatrix();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::popMatrix()
{
    glPopMatrix();
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::multMatrix(float* glTransform )
{
    glMultMatrix(glTransform);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::scale( float s )
{
    glScale(s,s,s);
}

void DrawToolGL::translate(float x, float y, float z)
{
    glTranslatef(x, y, z);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::writeOverlayText( int x, int y, unsigned fontSize, const RGBAColor &color, const char* text )
{
    GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );

    static const float letterSize = 0.5;

    float scale = fontSize / letterSize;

    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);

    glPushAttrib( GL_LIGHTING_BIT );
    glEnable( GL_COLOR_MATERIAL );

    glPushAttrib( GL_ENABLE_BIT );

    glDisable(GL_CULL_FACE);

    glColor4f( color[0], color[1], color[2], color[3] );

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, viewport[2], viewport[3], 0 );

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glTranslated(x,y,0);

    glScalef( scale, scale, scale );

    helper::gl::GlText::textureDraw_Overlay(text);

    glPopAttrib(); // GL_ENABLE_BIT
    glPopAttrib(); // GL_LIGHTING_BIT

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

}

//UF - DS
typedef sofa::helper::fixed_array<float, 3> Float3;
typedef sofa::helper::fixed_array<float, 2> Float2;

void DrawToolGL::DrawTextureQuad(int x, int y, int width, int height, sofa::helper::gl::Texture* InTexture, bool bFromRightCorner)
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);

    glPushAttrib(GL_LIGHTING_BIT);
    glEnable(GL_COLOR_MATERIAL);

    glPushAttrib(GL_ENABLE_BIT);

    glDisable(GL_CULL_FACE);

    glColor4f(1, 1, 1, 1);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, viewport[2], 0, viewport[3]);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    if (bFromRightCorner)
    {
        glTranslated(viewport[2] - x - width, y, 0);
        glScalef(width, height, 1.0f);
    }
    else
    {
        glTranslated(x, y, 0);
        glScalef(width, height, 1.0f);
    }

    {
        glPushAttrib(GL_TEXTURE_BIT);
        glEnable(GL_TEXTURE_2D);

        InTexture->bind();

        Float3 Quad[4];
        Quad[0] = Float3(0, 0, 0);
        Quad[1] = Float3(0, 1, 0);
        Quad[2] = Float3(1, 1, 0);
        Quad[3] = Float3(1, 0, 0);

        Float2 UVs[4];
        UVs[0] = Float2(0, 1);
        UVs[1] = Float2(0, 0);
        UVs[2] = Float2(1, 0);
        UVs[3] = Float2(1, 1);

        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_GREATER, 0.0);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glBegin(GL_QUADS);
        for (int j = 0; j < 4; j++)
        {
            glTexCoord2fv(UVs[j].data());
            glVertex3fv(Quad[j].data());
        }
        glEnd();

        glDisable(GL_ALPHA_TEST);

        InTexture->unbind();
        glPopAttrib();
    }

    glPopAttrib(); // GL_ENABLE_BIT
    glPopAttrib(); // GL_LIGHTING_BIT

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

}


void DrawToolGL::enableBlending()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void DrawToolGL::disableBlending()
{
    glDisable(GL_BLEND);
}
void DrawToolGL::enablePolygonOffset( float factor, float units)
{
    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(factor, units);
}
void DrawToolGL::disablePolygonOffset()
{
    glDisable(GL_POLYGON_OFFSET_LINE);
}
void DrawToolGL::enableLighting()
{
    glEnable(GL_LIGHTING);
}

void DrawToolGL::disableLighting()
{
    glDisable(GL_LIGHTING);
}

void DrawToolGL::enableDepthTest()
{
    glEnable(GL_DEPTH_TEST);
}

void DrawToolGL::disableDepthTest()
{
    glDisable(GL_DEPTH_TEST);
}


//UF - DS
void DrawToolGL::enableStencilTest()
{
    glEnable(GL_STENCIL_TEST);
}
void DrawToolGL::disableStencilTest()
{
    glDisable(GL_STENCIL_TEST);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::draw3DText(const Vector3 &p, float scale, const RGBAColor &color, const char* text)
{
    glColor4fv(color.array());

    sofa::helper::gl::GlText::draw(text, p, (double)scale);
}

void DrawToolGL::draw3DText_Indices(const std::vector<Vector3> &positions, float scale, const RGBAColor &color)
{
    glColor4f(color[0], color[1], color[2], color[3]);

    sofa::helper::gl::GlText::textureDraw_Indices(positions, scale);
}

void DrawToolGL::saveLastState()
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
}

void DrawToolGL::restoreLastState()
{
    glPopAttrib();
}

void DrawToolGL::readPixels(int x, int y, int w, int h, float* rgb, float* z)
{
    if(rgb != nullptr && sizeof(*rgb) == 3 * sizeof(float) * w * h)
        glReadPixels(x, y, w, h, GL_RGB, GL_FLOAT, rgb);

    if(z != nullptr && sizeof(*z) == sizeof(float) * w * h)
        glReadPixels(x, y, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, z);
}

} // namespace visual

} // namespace core

} // namespace sofa
