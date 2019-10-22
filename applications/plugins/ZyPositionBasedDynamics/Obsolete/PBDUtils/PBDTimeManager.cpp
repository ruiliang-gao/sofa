#include "PBDTimeManager.h"

#include <sofa/helper/logging/Messaging.h>

using namespace sofa::simulation::PBDSimulation;

// #include "Utils/Timing.h"

PBDTimeManager* PBDTimeManager::current = 0;

PBDTimeManager::PBDTimeManager ()
{
    time = 0;
    h = static_cast<Real>(0.005);
}

PBDTimeManager::~PBDTimeManager ()
{
    current = 0;
}

PBDTimeManager* PBDTimeManager::getCurrent ()
{
    if (current == 0)
    {
        msg_info("PBDTimeManager") << "Instantiating PBDTimeManager singleton.";
        current = new PBDTimeManager ();
    }
    return current;
}

void PBDTimeManager::setCurrent (PBDTimeManager* tm)
{
    current = tm;
}

bool PBDTimeManager::hasCurrent()
{
    return (current != 0);
}

Real PBDTimeManager::getTime()
{
    return time;
}

void PBDTimeManager::setTime(Real t)
{
    time = t;
}

Real PBDTimeManager::getTimeStepSize()
{
    return h;
}

void PBDTimeManager::setTimeStepSize(Real tss)
{
    h = tss;
}

