#include "IdFactory.h"

using namespace sofa::simulation::PBDSimulation;

int IDFactory::id = 0;

int IDFactory::getId()
{
    return id++;
}
