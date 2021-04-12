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
#ifndef SOFA_COMPONENT_VISUALMODEL_OGLMODEL_H
#define SOFA_COMPONENT_VISUALMODEL_OGLMODEL_H
#include "config.h"

#include <vector>
#include <string>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseVisual/VisualModelImpl.h>

#define   NB_MAX_TEXTURES 16

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief Main class for rendering 3D model in SOFA.
 *
 *  This class implements VisuelModelImpl with rendering functions
 *  using OpenGL.
 *
 */

class SOFA_OPENGL_VISUAL_API OglModel : public VisualModelImpl
{
public:
    using Inherit = VisualModelImpl;

    SOFA_CLASS(OglModel, Inherit);


    Data<bool> blendTransparency; ///< Blend transparent parts
protected:
    Data<bool> premultipliedAlpha; ///< is alpha premultiplied ?
    Data<bool> writeZTransparent; ///< Write into Z Buffer for Transparent Object
    Data<bool> alphaBlend; ///< Enable alpha blending
    Data<bool> depthTest; ///< Enable depth testing
    Data<int> cullFace; ///< Face culling (0 = no culling, 1 = cull back faces, 2 = cull front faces)
    Data<GLfloat> lineWidth; ///< Line width (set if != 1, only for lines rendering)
    Data<GLfloat> pointSize; ///< Point size (set if != 1, only for points rendering)
    Data<bool> lineSmooth; ///< Enable smooth line rendering
    Data<bool> pointSmooth; ///< Enable smooth point rendering
    /// Suppress field for save as function
    Data < bool > isEnabled;

    // primitive types
    Data<sofa::helper::OptionsGroup> primitiveType; ///< Select types of primitives to send (necessary for some shader types such as geometry or tesselation)

    //alpha blend function
    Data<sofa::helper::OptionsGroup> blendEquation; ///< if alpha blending is enabled this specifies how source and destination colors are combined
    Data<sofa::helper::OptionsGroup> sourceFactor; ///< if alpha blending is enabled this specifies how the red, green, blue, and alpha source blending factors are computed
    Data<sofa::helper::OptionsGroup> destFactor; ///< if alpha blending is enabled this specifies how the red, green, blue, and alpha destination blending factors are computed
    GLenum blendEq, sfactor, dfactor;

    helper::gl::Texture *tex; //this texture is used only if a texture name is specified in the scn
    bool is Tex3d; //true while using 3d texture
    GLuint target; //tex2d or tex3d
    GLuint vbo, iboEdges, iboTriangles, iboQuads;
    bool VBOGenDone, initDone, useEdges, useTriangles, useQuads, canUsePatches;
    size_t oldVerticesSize, oldNormalsSize, oldTexCoordsSize, oldTangentsSize, oldBitangentsSize, oldEdgesSize, oldTrianglesSize, oldQuadsSize;

    /// These two buffers are used to convert the data field to float type before being sent to
    /// opengl
    std::vector<sofa::defaulttype::Vec3f> verticesTmpBuffer;
    std::vector<sofa::defaulttype::Vec3f> normalsTmpBuffer;

    void internalDraw(const core::visual::VisualParams* vparams, bool transparent) override;

    void drawGroup(int ig, bool transparent);
    void drawGroups(bool transparent);

    virtual void pushTransformMatrix(float* matrix) { glPushMatrix(); glMultMatrixf(matrix); }
    virtual void popTransformMatrix() { glPopMatrix(); }

    std::vector<helper::gl::Texture*> textures;

    std::map<int, int> materialTextureIdMap; //link between a material and a texture

    GLenum getGLenum(const char* c ) const;


    OglModel();

    ~OglModel() override;
public:

    bool loadTexture(const std::string& filename) override;
    bool loadTextures() override;

    void initTextures();
    void initVisual() override;

    void init() override { VisualModelImpl::init(); }
    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    void updateBuffers() override;

    bool hasTransparent() override;
    bool hasTexture();

public:
    bool isUseEdges()	{ return useEdges; }
    bool isUseTriangles()	{ return useTriangles; }
    bool isUseQuads()	{ return useQuads; }

    helper::gl::Texture* getTex() const	{ return tex; }
    GLuint getVbo()	{ return vbo;	}
    GLuint getIboEdges() { return iboEdges; }
    GLuint getIboTriangles() { return iboTriangles; }
    GLuint getIboQuads()    { return iboQuads; }
    const std::vector<helper::gl::Texture*>& getTextures() const { return textures;	}

    void createVertexBuffer();
    void createEdgesIndicesBuffer();
    void createTrianglesIndicesBuffer();
    void createQuadsIndicesBuffer();
    void initVertexBuffer();
    void initEdgesIndicesBuffer();
    void initTrianglesIndicesBuffer();
    void initQuadsIndicesBuffer();
    void updateVertexBuffer();
    void updateEdgesIndicesBuffer();
    void updateTrianglesIndicesBuffer();
    void updateQuadsIndicesBuffer();
};

typedef sofa::defaulttype::Vec<3,GLfloat> GLVec3f;

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
