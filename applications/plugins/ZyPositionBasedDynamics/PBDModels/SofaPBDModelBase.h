#ifndef SOFAPBDMODELBASE_H
#define SOFAPBDMODELBASE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/objectmodel/Data.h>

#include <PBDRigidBody.h>
#include <PBDRigidBodyGeometry.h>

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDModelBase: public BaseObject
            {
                public:
                    SofaPBDModelBase();
                    virtual ~SofaPBDModelBase();

                    /// @name Initial transformations accessors.
                    /// @{

                    virtual void setTranslation(SReal dx, SReal dy, SReal dz) {translation.setValue(Vector3(dx,dy,dz));}
                    virtual void setRotation(SReal rx, SReal ry, SReal rz) {rotation.setValue(Vector3(rx,ry,rz));}
                    virtual void setScale(SReal sx, SReal sy, SReal sz) {scale.setValue(Vector3(sx,sy,sz));}

                    virtual Vector3 getTranslation() const {return translation.getValue();}
                    virtual Vector3 getRotation() const {return rotation.getValue();}
                    virtual Vector3 getScale() const {return scale.getValue();}

                    virtual void setRotationQuat(double qx, double qy, double qz, double qw) { rotationQuat.setValue(Quaternion(qx,qy,qz,qw)); }
                    virtual void setRotationQuat(const Quaternion &q) { rotationQuat.setValue(q);}

                    virtual void resetTransformations();

                    virtual Quaternion getRotationQuat() const { return rotationQuat.getValue(); }

                    virtual bool hasPBDRigidBody() const;
                    virtual const PBDRigidBodyGeometry& getRigidBodyGeometry() const;

                    /// @}
                protected:
                    /// Initial model transform data
                    Data< Vector3 > translation; ///< Translation of the DOFs
                    Data< Vector3 > rotation; ///< Rotation of the DOFs
                    Data< Vector3 > scale; ///< Scale of the DOFs in 3 dimensions
                    Data< Quaternion > rotationQuat; /// Initial rotation as quaternion

                    virtual void buildModel() = 0;
                    virtual void initializeModel() = 0;

                    PBDRigidBody* m_pbdRigidBody;

            };
        }
    }
}

#endif // SOFAPBDMODELBASE_H
