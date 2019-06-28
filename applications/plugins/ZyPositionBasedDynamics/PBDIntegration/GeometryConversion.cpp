#include "GeometryConversion.h"

using namespace sofa::simulation::PBDSimulation;

GeometryConversion::GeometryConversion(PBDSimulationModel* model): sofa::core::objectmodel::BaseObject(), m_model(model)
{

}

bool GeometryConversion::convertToPBDObjects()
{
    return true;
}
