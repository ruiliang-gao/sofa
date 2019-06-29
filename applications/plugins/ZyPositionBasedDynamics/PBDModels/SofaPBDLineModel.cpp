#include "SofaPBDLineModel.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDLineModelPrivate
            {
                public:
                    SofaPBDLineModelPrivate()
                    {
                        m_pbdLineModel.reset(new PBDLineModel());
                    }

                    std::shared_ptr<PBDLineModel> m_pbdLineModel;
            };
        }
    }
}
using namespace sofa::simulation::PBDSimulation;

int SofaPBDLineModelClass = sofa::core::RegisterObject("Wrapper class for PBD LineModels.")
                            .add< SofaPBDLineModel >()
                            .addDescription("Encapsulates sets of particles connected in a chain.");

SofaPBDLineModel::SofaPBDLineModel(): sofa::core::objectmodel::BaseObject()
{
    m_d.reset(new SofaPBDLineModelPrivate());
}
