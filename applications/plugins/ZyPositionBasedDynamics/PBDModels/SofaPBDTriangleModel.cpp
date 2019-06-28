#include "SofaPBDTriangleModel.h"

#include <sofa/core/ObjectFactory.h>

using namespace sofa::simulation::PBDSimulation;

int SofaPBDTriangleModelClass = sofa::core::RegisterObject("Wrapper class for PBD TriangleModels.")
                            .add< SofaPBDTriangleModel >()
                            .addDescription("Encapsulates sets of particles in an indexed triangle mesh.");

SofaPBDTriangleModel::SofaPBDTriangleModel(): sofa::core::objectmodel::BaseObject()
{

}
