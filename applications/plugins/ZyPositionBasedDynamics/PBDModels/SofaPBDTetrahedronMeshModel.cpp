#include "SofaPBDTetrahedronMeshModel.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTetrahedronMeshModelPrivate
            {
                public:
                    SofaPBDTetrahedronMeshModelPrivate()
                    {

                    }
            };
        }
    }
}

using namespace sofa::simulation::PBDSimulation;

SofaPBDTetrahedronMeshModel::SofaPBDTetrahedronMeshModel(SimulationModel* model)
{
    m_d = new SofaPBDTetrahedronMeshModelPrivate();
}

SofaPBDTetrahedronMeshModel::~SofaPBDTetrahedronMeshModel()
{
    if (m_d)
    {
        delete m_d;
        m_d = nullptr;
    }
}
