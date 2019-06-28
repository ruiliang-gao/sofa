#include "SofaPBDLineModel.h"

#include <sofa/core/ObjectFactory.h>

using namespace sofa::simulation::PBDSimulation;

int SofaPBDLineModelClass = sofa::core::RegisterObject("Wrapper class for PBD LineModels.")
                            .add< SofaPBDLineModel >()
                            .addDescription("Encapsulates sets of particles connected in a chain.");

SofaPBDLineModel::SofaPBDLineModel(): sofa::core::objectmodel::BaseObject()
{

}
