#ifndef PBDTIMEMANAGER_H
#define PBDTIMEMANAGER_H

#include "PBDCommon/PBDCommon.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class PBDTimeManager
            {
                private:
                    Real time;
                    static PBDTimeManager *current;
                    Real h;

                public:
                    PBDTimeManager();
                    ~PBDTimeManager();

                    // Singleton
                    static PBDTimeManager* getCurrent ();
                    static void setCurrent (PBDTimeManager* tm);
                    static bool hasCurrent();

                    Real getTime();
                    void setTime(Real t);
                    Real getTimeStepSize();
                    void setTimeStepSize(Real tss);
            };
        }
    }
}

#endif // PBDTIMEMANAGER_H
