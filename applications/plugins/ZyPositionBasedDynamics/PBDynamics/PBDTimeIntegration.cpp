#include "PBDTimeIntegration.h"

#include <sofa/helper/logging/Messaging.h>

using namespace sofa::simulation::PBDSimulation;

// ----------------------------------------------------------------------------------------------
void PBDTimeIntegration::semiImplicitEuler(
    const Real h,
    const Real mass,
    Vector3r &position,
    Vector3r &velocity,
    const Vector3r &acceleration)
{
    if (mass != 0.0)
    {
        msg_info("PBDTimeIntegration") << "semiImplicitEuler: velocity = (" << velocity[0] << "," << velocity[1] << "," << velocity[2] << ") + " << "((" << acceleration[0] << "," << acceleration[1] << "," << acceleration[2] << ") * " << h << ")";
        velocity += acceleration * h;
        msg_info("PBDTimeIntegration") << "semiImplicitEuler: position = (" << position[0] << "," << position[1] << "," << position[2] << ") + " << "((" << velocity[0] << "," << "," << velocity[1] << "," << velocity[2] << ") * " << h << ")";
        position += velocity * h;
    }
}

// ----------------------------------------------------------------------------------------------
void PBDTimeIntegration::semiImplicitEulerRotation(
    const Real h,
    const Real mass,
    const Matrix3r &invInertiaW,
    Quaternionr &rotation,
    Vector3r &angularVelocity,
    const Vector3r &torque)
{
    if (mass != 0.0)
    {
        // simple form without nutation effect
        msg_info("PBDTimeIntegration") << "semiImplicitEulerRotation: angularVelocity = (" << angularVelocity[0] << "," << angularVelocity[1] << "," << angularVelocity[2] << "," << ") + " << "((" << invInertiaW << " * " << torque[0] << "," << torque[1] << "," << torque[2] << ") * " << h << ")";
        angularVelocity += h * invInertiaW * torque;

        Quaternionr angVelQ(0.0, angularVelocity[0], angularVelocity[1], angularVelocity[2]);
        rotation.coeffs() += h * 0.5 * (angVelQ * rotation).coeffs();
        rotation.normalize();
        msg_info("PBDTimeIntegration") << "semiImplicitEulerRotation: rotation = (" << rotation.x() << "," << rotation.y() << "," << rotation.z() << "," << rotation.w() << ")";
    }
}

// ----------------------------------------------------------------------------------------------
void PBDTimeIntegration::velocityUpdateFirstOrder(
    const Real h,
    const Real mass,
    const Vector3r &position,
    const Vector3r &oldPosition,
    Vector3r &velocity)
{
    if (mass != 0.0)
    {
        velocity = (1.0 / h) * (position - oldPosition);
    }
}

// ----------------------------------------------------------------------------------------------
void PBDTimeIntegration::angularVelocityUpdateFirstOrder(
    const Real h,
    const Real mass,
    const Quaternionr &rotation,
    const Quaternionr &oldRotation,
    Vector3r &angularVelocity)
{
    if (mass != 0.0)
    {
        const Quaternionr relRot = (rotation * oldRotation.conjugate());
        angularVelocity = relRot.vec() *(2.0 / h);
    }
}

// ----------------------------------------------------------------------------------------------
void PBDTimeIntegration::velocityUpdateSecondOrder(
    const Real h,
    const Real mass,
    const Vector3r &position,
    const Vector3r &oldPosition,
    const Vector3r &positionOfLastStep,
    Vector3r &velocity)
{
    if (mass != 0.0)
    {
        velocity = (1.0 / h) * (1.5*position - 2.0*oldPosition + 0.5*positionOfLastStep);
    }
}

// ----------------------------------------------------------------------------------------------
void PBDTimeIntegration::angularVelocityUpdateSecondOrder(
    const Real h,
    const Real mass,
    const Quaternionr &rotation,
    const Quaternionr &oldRotation,
    const Quaternionr &rotationOfLastStep,
    Vector3r &angularVelocity)
{
    // ToDo: is still first order
    if (mass != 0.0)
    {
        const Quaternionr relRot = (rotation * oldRotation.conjugate());
        angularVelocity = relRot.vec() *(2.0 / h);
    }
}
