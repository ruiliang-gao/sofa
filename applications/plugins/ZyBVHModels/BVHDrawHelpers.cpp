#include "BVHDrawHelpers.h"

#include <GL/gl.h>

#include <sofa/defaulttype/Mat.h>

void BVHDrawHelpers::drawObbVolume(const sofa::defaulttype::Vector3 &halfExtents, const Vec4f &color, bool emphasize)
{
    if (emphasize)
        glLineWidth(1.0);

    glBegin(GL_LINES);
    glColor4d(color.x(), color.y(), color.z(), color.w());
    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());

    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), halfExtents.z());

    glVertex3d(-halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(-halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glVertex3d(halfExtents.x(), halfExtents.y(), -halfExtents.z());
    glVertex3d(halfExtents.x(), -halfExtents.y(), -halfExtents.z());

    glEnd();
/*
    if (emphasize)
        glLineWidth(1.0);
        */
}

#ifdef RSS_EXPERIMENTAL
void BVHDrawHelpers::drawRssVolume(const sofa::defaulttype::Vector2 &lengths, const SReal &radius, const Vec4f &color, const Vec4f &sphereColor, bool emphasize)
{
    //corner UP LEFT    = (0         ,lengths[1],0)
    //corner UP RIGHT   = (lengths[0],lengths[1],0)
    //corner DOWN LEFT  = (0         ,0         ,0)
    //corner DOWN RIGHT = (lengths[0],0         ,0)
#ifdef DEBUG_DRAWRSS
    std::cout << "Drawing RSS: " << std::endl;
    std::cout << "lengths: " << lengths << std::endl;
    std::cout << "radius: " << radius << std::endl << std::endl;
#endif

    if (emphasize)
        glLineWidth(5.0);

    glBegin(GL_LINES);
    glColor4d(color.x(), color.y(), color.z(), color.w());
    // rectangle
    glVertex3d(0         ,lengths[1],0);
    glVertex3d(lengths[0],lengths[1],0);

    glVertex3d(lengths[0],lengths[1],0);
    glVertex3d(lengths[0],0         ,0);

    glVertex3d(lengths[0],0         ,0);
    glVertex3d(0         ,0         ,0);

    glVertex3d(0         ,0         ,0);
    glVertex3d(0         ,lengths[1],0);

    // swept sphere radii
    glColor4d(sphereColor.x(), sphereColor.y(), sphereColor.z(), sphereColor.w());
    // UL
    glVertex3d(0                  ,lengths[1]         ,0-radius);
    glVertex3d(0                  ,lengths[1]         ,0+radius);

    //glVertex3d(0                  ,lengths[1]-radius  ,0       );
    glVertex3d(0                  ,lengths[1]+radius  ,0       );
    glVertex3d(0                  ,lengths[1]         ,0       );

    glVertex3d(0-radius           ,lengths[1]         ,0       );
    //glVertex3d(0+radius           ,lengths[1]         ,0       );
    glVertex3d(0                  ,lengths[1]         ,0       );

    // UR
    glVertex3d(lengths[0]         ,lengths[1]         ,0-radius);
    glVertex3d(lengths[0]         ,lengths[1]         ,0+radius);

    //glVertex3d(lengths[0]         ,lengths[1]-radius  ,0       );
    glVertex3d(lengths[0]         ,lengths[1]+radius  ,0       );
    glVertex3d(lengths[0]         ,lengths[1]         ,0       );

    //glVertex3d(lengths[0]-radius  ,lengths[1]         ,0       );
    glVertex3d(lengths[0]+radius  ,lengths[1]         ,0       );
    glVertex3d(lengths[0]         ,lengths[1]         ,0       );

    // DR
    glVertex3d(lengths[0]         ,0                  ,0-radius);
    glVertex3d(lengths[0]         ,0                  ,0+radius);

    glVertex3d(lengths[0]         ,0-radius           ,0       );
    //glVertex3d(lengths[0]         ,0+radius           ,0       );
    glVertex3d(lengths[0]         ,0                  ,0       );

    //glVertex3d(lengths[0]-radius  ,0                  ,0       );
    glVertex3d(lengths[0]+radius  ,0                  ,0       );
    glVertex3d(lengths[0]         ,0                  ,0       );

    // DL
    glVertex3d(0                  ,0                  ,0-radius);
    glVertex3d(0                  ,0                  ,0+radius);

    glVertex3d(0                  ,0-radius           ,0       );
    //glVertex3d(0                  ,0+radius           ,0       );
    glVertex3d(0                  ,0                  ,0       );

    glVertex3d(0-radius           ,0                  ,0       );
    //glVertex3d(0+radius           ,0                  ,0       );
    glVertex3d(0                  ,0                  ,0       );

    glEnd();

    // swept sphere outer boxes
    if (emphasize)
        glLineWidth(1.0);

    glBegin(GL_LINES);

    glVertex3d(0                  ,lengths[1]         ,0-radius);
    glVertex3d(lengths[0]         ,lengths[1]         ,0-radius);

    glVertex3d(lengths[0]         ,lengths[1]         ,0-radius);
    glVertex3d(lengths[0]         ,0                  ,0-radius);

    glVertex3d(lengths[0]         ,0                  ,0-radius);
    glVertex3d(0                  ,0                  ,0-radius);

    glVertex3d(0                  ,0                  ,0-radius);
    glVertex3d(0                  ,lengths[1]         ,0-radius);

    glVertex3d(0                  ,lengths[1]         ,0+radius);
    glVertex3d(lengths[0]         ,lengths[1]         ,0+radius);

    glVertex3d(lengths[0]         ,lengths[1]         ,0+radius);
    glVertex3d(lengths[0]         ,0                  ,0+radius);

    glVertex3d(lengths[0]         ,0                  ,0+radius);
    glVertex3d(0                  ,0                  ,0+radius);

    glVertex3d(0                  ,0                  ,0+radius);
    glVertex3d(0                  ,lengths[1]         ,0+radius);


    glVertex3d(0-radius           ,lengths[1]         ,0       );
    glVertex3d(0                  ,lengths[1]+radius  ,0       );

    glVertex3d(0                  ,lengths[1]+radius  ,0       );
    glVertex3d(lengths[0]         ,lengths[1]+radius  ,0       );

    glVertex3d(lengths[0]         ,lengths[1]+radius  ,0       );
    glVertex3d(lengths[0]+radius  ,lengths[1]         ,0       );

    glVertex3d(lengths[0]+radius  ,lengths[1]         ,0       );
    glVertex3d(lengths[0]+radius  ,0                  ,0       );

    glVertex3d(lengths[0]+radius  ,0                  ,0       );
    glVertex3d(lengths[0]         ,0-radius           ,0       );

    glVertex3d(lengths[0]         ,0-radius           ,0       );
    glVertex3d(0                  ,0-radius           ,0       );

    glVertex3d(0                  ,0-radius           ,0       );
    glVertex3d(0-radius           ,0                  ,0       );

    glVertex3d(0-radius           ,0                  ,0       );
    glVertex3d(0-radius           ,lengths[1]         ,0       );

    glEnd();

    if (emphasize)
        glLineWidth(1.0);

    // spheres at the rectangle edges
    GLUquadricObj* GLUquad = gluNewQuadric();
    if (GLUquad != NULL)
    {
        bool glBlendDisabled = false;
        if (!glIsEnabled(GL_BLEND))
        {
            glBlendDisabled = true;
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }

        glColor4d(sphereColor.x(), sphereColor.y(), sphereColor.z(), 0.15*sphereColor.w());

        //gluSphere(GLUquad,radius,numSlices,numStacks);

        unsigned int numSlices = 24;
        unsigned int numStacks = 8;

        gluSphere(GLUquad,radius,numSlices,numStacks);

        glTranslatef(lengths[0],0.0,0.0);
        gluSphere(GLUquad,radius,numSlices,numStacks);

        glTranslatef(0.0,lengths[1],0.0);
        gluSphere(GLUquad,radius,numSlices,numStacks);

        glTranslatef(-lengths[0],0.0,0.0);
        gluSphere(GLUquad,radius,numSlices,numStacks);

        glTranslatef(0.0,-lengths[1],0.0);

        if (glBlendDisabled)
        {
            glDisable(GL_BLEND);
        }
    }
}
#endif

void BVHDrawHelpers::drawCoordinateMarkerGL(float lineLength, float lineWidth, const Vec4f& xColor, const Vec4f& yColor, const Vec4f& zColor)
{
    glLineWidth(lineWidth);
    glBegin(GL_LINES);

    glColor4f(xColor.x(), xColor.y(), xColor.z(), xColor.w());
    glVertex3d(0, 0, 0);
    glVertex3d(lineLength,0,0);

    glColor4f(yColor.x(), yColor.y(), yColor.z(), yColor.w());
    glVertex3d(0, 0, 0);
    glVertex3d(0,lineLength,0);

    glColor4f(zColor.x(), zColor.y(), zColor.z(), zColor.w());
    glVertex3d(0, 0, 0);
    glVertex3d(0,0,lineLength);

    glEnd();
    glLineWidth(1.0f);
}

Vec4f BVHDrawHelpers::randomVec4()
{
#ifndef _WIN32
    double r = drand48();
    double g = drand48();
    double b = drand48();
    double a = drand48();
# else
    double r = (rand() / (RAND_MAX + 1.0));
    double g = (rand() / (RAND_MAX + 1.0));
    double b = (rand() / (RAND_MAX + 1.0));
    double a = (rand() / (RAND_MAX + 1.0));
#endif

    if (r < 0.5f)
        r = 0.5f + r;

    if (g < 0.5f)
        g = 0.5f + g;

    if (b < 0.5f)
        b = 0.5f + b;

    if (a < 0.5f)
        a = 0.5f + a;

    return Vec4f(r,g,b,a);
}

void BVHDrawHelpers::drawRec(PQP_Model* model, BV *drawable, const sofa::core::visual::VisualParams *vparams, const Vec4f &color, unsigned long depth, bool emphasized)
{
    glTranslated(drawable->To[0],drawable->To[1],drawable->To[2]);
    {
        glBegin(GL_LINES);
        glColor4d(color.x(), color.y(), color.z(), color.w());
        glVertex3d(0, 0, 0);
        glColor4d(color.x(), color.y(), color.z(), color.w());
        glVertex3d(-drawable->To[0],-drawable->To[1],-drawable->To[2]);
        glEnd();

        if (emphasized)
            glLineWidth(1.0f);
    }

    Matrix3 obbRotation; obbRotation.identity();
    obbRotation[0] = sofa::defaulttype::Vector3(drawable->R[0][0], drawable->R[1][0], drawable->R[2][0]);
    obbRotation[1] = sofa::defaulttype::Vector3(drawable->R[0][1], drawable->R[1][1], drawable->R[2][1]);
    obbRotation[2] = sofa::defaulttype::Vector3(drawable->R[0][2], drawable->R[1][2], drawable->R[2][2]);

    Matrix4 glOrientation; glOrientation.identity();
    for (int k = 0; k < 3; k++)
    {
        for (int l = 0; l < 3; l++)
        {
            glOrientation[k][l] = obbRotation[k][l];
        }
    }

    glMultMatrixd(glOrientation.ptr());
    {
        /*if (emphasized)
            drawCoordinateMarkerGL(0.5f, 3.0f, color, color, color);
        else
            drawCoordinateMarkerGL(0.5f, 1.0f, color, color, color);
        */
        if (emphasized)
            glLineWidth(4.0f);

        BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(drawable->d[0], drawable->d[1], drawable->d[2]), color);

        float extent_x = drawable->d[0]; float extent_y = drawable->d[1]; float extent_z = drawable->d[2];
        if (drawable->min_dimension == 0)
            extent_x = drawable->min_dimension_val;
        else if (drawable->min_dimension == 1)
            extent_y = drawable->min_dimension_val;
        else if (drawable->min_dimension == 2)
            extent_z = drawable->min_dimension_val;

        BVHDrawHelpers::drawObbVolume(sofa::defaulttype::Vector3(extent_x, extent_y, extent_z), Vec4f(0,1,0,1), true);

        if (emphasized)
            glLineWidth(1.0f);
    }
    glMultMatrixd(glOrientation.transposed().ptr());


    glTranslated(-drawable->To[0],-drawable->To[1],-drawable->To[2]);
    {
        if (drawable->first_child > 0)
        {
            drawRec(model, model->child(drawable->first_child), vparams, color, depth + 1, false);
            drawRec(model, model->child(drawable->first_child + 1), vparams, color, depth + 1, false);
        }
    }
}
