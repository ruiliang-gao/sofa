#ifndef SOFAPBDCOLLISIONDETECTIONOUTPUT_H
#define SOFAPBDCOLLISIONDETECTIONOUTPUT_H

#include "initZyPositionBasedDynamicsPlugin.h"
#include <sofa/core/collision/DetectionOutput.h>

#include <sofa/core/objectmodel/Tag.h>

namespace sofa
{
    namespace core
    {
        namespace collision
        {
            enum SofaPBDContactType
            {
                PBD_RIGID_RIGID_CONTACT = 0,
                PBD_RIGID_LINE_CONTACT = 1,
                PBD_LINE_LINE_CONTACT = 2,
                PBD_PARTICLE_RIGID_CONTACT = 3,
                PBD_PARTICLE_SOLID_CONTACT = 4,
                PBD_CONTACT_TYPE_INVALID = 5
            };

            enum SofaPBDContactModelPairType
            {
                PBD_CONTACT_PAIR_POINT_POINT = 0,
                PBD_CONTACT_PAIR_POINT_LINE = 1,
                PBD_CONTACT_PAIR_POINT_TRIANGLE = 2,
                PBD_CONTACT_PAIR_LINE_LINE = 3,
                PBD_CONTACT_PAIR_LINE_TRIANGLE = 4,
                PBD_CONTACT_PAIR_TRIANGLE_TRIANGLE = 5,
                PBD_CONTACT_PAIR_INVALID = 6
            };

            static sofa::core::objectmodel::Tag tagPBDPointCollisionModel = sofa::core::objectmodel::Tag("SofaPBDPointCollisionModel");
            static sofa::core::objectmodel::Tag tagPBDLineCollisionModel = sofa::core::objectmodel::Tag("SofaPBDLineCollisionModel");
            static sofa::core::objectmodel::Tag tagPBDTriangleCollisionModel = sofa::core::objectmodel::Tag("SofaPBDTriangleCollisionModel");

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API SofaPBDCollisionDetectionOutput: public DetectionOutput
            {
                public:
                    SofaPBDCollisionDetectionOutput(): DetectionOutput()
                    {
                        contactType = PBD_CONTACT_TYPE_INVALID;
                        modelPairType = PBD_CONTACT_PAIR_INVALID;
                        modelTypeMirrored = false;
                        rigidBodyIndices[0] = rigidBodyIndices[1] = -1;
                        particleIndices[0] = particleIndices[1] = -1;
                    }

                    SofaPBDCollisionDetectionOutput(const DetectionOutput& other): DetectionOutput()
                    {
                        this->point[0] = other.point[0];
                        this->point[1] = other.point[1];
                        this->id = other.id;
                        this->value = other.value;
                        this->deltaT = other.deltaT;
                        this->normal = other.normal;

                        this->elem.first = other.elem.first;
                        this->elem.second = other.elem.second;

                        contactType = PBD_CONTACT_TYPE_INVALID;
                        modelPairType = PBD_CONTACT_PAIR_INVALID;
                        modelTypeMirrored = false;

                        rigidBodyIndices[0] = rigidBodyIndices[1] = -1;
                        particleIndices[0] = particleIndices[1] = -1;
                    }

                    SofaPBDContactType contactType;
                    SofaPBDContactModelPairType modelPairType;
                    bool modelTypeMirrored;

                    int rigidBodyIndices[2];
                    int particleIndices[2];
            };
        }
    }
}

#endif // SOFAPBDCOLLISIONDETECTIONOUTPUT_H
