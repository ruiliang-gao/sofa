#ifndef BVHDRAWHELPERS_H
#define BVHDRAWHELPERS_H


#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/visual/VisualParams.h>

#include <PQP.h>

using namespace sofa::defaulttype;
namespace BVHDrawHelpers
{
    void drawCoordinateMarkerGL(float lineLength, float lineWidth, const Vec4f& xColor = Vec4f(1,0,0,1), const Vec4f& yColor = Vec4f(0,1,0,1), const Vec4f& zColor = Vec4f(0,0,1,1));
    void drawObbVolume(const sofa::defaulttype::Vector3 &halfExtents, const Vec4f &color, bool emphasize = false);

    Vec4f randomVec4();

    void drawRec(PQP_Model* model, BV *drawable, const sofa::core::visual::VisualParams *vparams, const Vec4f &color, unsigned long depth, bool emphasized);
}

#endif // BVHDRAWHELPERS_H
