#include "SofaPBDTriangleModel.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTriangleModelPrivate
            {
                public:
                    SofaPBDTriangleModelPrivate()
                    {
                        m_pbdTriangleModel.reset(new PBDTriangleModel());
                    }

                    std::shared_ptr<PBDTriangleModel> m_pbdTriangleModel;
            };
        }
    }
}

using namespace sofa::simulation::PBDSimulation;

int SofaPBDTriangleModelClass = sofa::core::RegisterObject("Wrapper class for PBD TriangleModels.")
                            .add< SofaPBDTriangleModel >()
                            .addDescription("Encapsulates sets of particles in an indexed triangle mesh.");

SofaPBDTriangleModel::SofaPBDTriangleModel(): sofa::core::objectmodel::BaseObject()
{
    m_d.reset(new SofaPBDTriangleModelPrivate());
}
