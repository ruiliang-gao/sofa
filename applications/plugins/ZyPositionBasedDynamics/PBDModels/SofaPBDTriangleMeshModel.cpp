#include "SofaPBDTriangleMeshModel.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTriangleMeshModelPrivate
            {
                public:
                    SofaPBDTriangleMeshModelPrivate()
                    {

                    }
            };
        }
    }
}

using namespace sofa::simulation::PBDSimulation;

SofaPBDTriangleMeshModel::SofaPBDTriangleMeshModel(SimulationModel* model)
{
    m_d = new SofaPBDTriangleMeshModelPrivate();
}

SofaPBDTriangleMeshModel::~SofaPBDTriangleMeshModel()
{
    if (m_d)
    {
        delete m_d;
        m_d = nullptr;
    }
}
