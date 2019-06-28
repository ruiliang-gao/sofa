#ifndef PBDTIMESTEPCONTROLLER_H
#define PBDTIMESTEPCONTROLLER_H

#include "PBDCommon/PBDCommon.h"
#include "SofaPBDTimeStep.h"
#include "PBDModels/PBDSimulationModel.h"

#include <sofa/helper/OptionsGroup.h>

#include "CollisionDetection.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTimeStepController : public SofaPBDTimeStep
            {
            public:
                Data<int> MAX_ITERATIONS;
                Data<int> MAX_ITERATIONS_V;
                Data<sofa::helper::OptionsGroup> VELOCITY_UPDATE_METHOD;

            protected:
                int m_velocityUpdateMethod;
                unsigned int m_iterations;
                unsigned int m_iterationsV;
                unsigned int m_maxIterations;
                unsigned int m_maxIterationsV;

                virtual void initParameters();

                void positionConstraintProjection(PBDSimulationModel &model);
                void velocityConstraintProjection(PBDSimulationModel &model);


            public:
                SOFA_CLASS(SofaPBDTimeStepController, sofa::core::objectmodel::BaseObject);

                SofaPBDTimeStepController();
                virtual ~SofaPBDTimeStepController(void);

                virtual void step(PBDSimulationModel &model);
                virtual void reset();
            };
        }
    }
}


#endif // PBDTIMESTEPCONTROLLER_H
