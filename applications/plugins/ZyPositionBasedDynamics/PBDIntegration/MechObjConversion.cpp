#include "MechObjConversion.h"

using namespace sofa::simulation::PBDSimulation;

MechObjConversion::MechObjConversion(PBDSimulationModel* model): sofa::core::objectmodel::BaseObject(), m_model(model)
{

}

bool MechObjConversion::convertToPBDObjects()
{
    return true;
}
