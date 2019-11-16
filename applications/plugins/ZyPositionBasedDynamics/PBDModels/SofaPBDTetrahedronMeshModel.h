#ifndef SOFAPBDTETRAHEDRONMESHMODEL_H
#define SOFAPBDTETRAHEDRONMESHMODEL_H

#include "TetModel.h"
#include "SofaPBDModelBase.h"

using namespace sofa::defaulttype;
using namespace PBD;

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTetrahedronMeshModelPrivate;
            class SofaPBDTetrahedronMeshModel
            {
                public:
                    SofaPBDTetrahedronMeshModel(SimulationModel* model);
                    virtual ~SofaPBDTetrahedronMeshModel();

                private:
                    SofaPBDTetrahedronMeshModelPrivate* m_d;
            };
        }
    }
}

#endif // SOFAPBDTETRAHEDRONMESHMODEL_H
