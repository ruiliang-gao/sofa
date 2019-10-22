#ifndef PBDFIXEDCONSTRAINT_H
#define PBDFIXEDCONSTRAINT_H

#include "PBDConstraintBase.h"
#include <PBDCommon/PBDCommon.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class PBDFixedConstraint: public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;

                    PBDFixedConstraint();

                    bool initConstraint(PBDSimulationModel& model, int rbIndex1, const Vector3r& pos, const Quaternionr& rot);
                    bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);

                private:
                    Vector3r m_pos;
                    Quaternionr m_rot;
            };
        }
    }
}

#endif // PBDFIXEDCONSTRAINT_H
