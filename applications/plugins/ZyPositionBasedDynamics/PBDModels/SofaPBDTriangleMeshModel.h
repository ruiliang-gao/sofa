#ifndef SOFAPBDTRIANGLEMESHMODEL_H
#define SOFAPBDTRIANGLEMESHMODEL_H

#include "TriangleModel.h"
#include "SofaPBDModelBase.h"

using namespace sofa::defaulttype;
using namespace PBD;

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class SofaPBDTriangleMeshModelPrivate;
            class SofaPBDTriangleMeshModel
            {
                public:
                    SofaPBDTriangleMeshModel(SimulationModel* model = nullptr);
                    virtual ~SofaPBDTriangleMeshModel();

                private:
                    SofaPBDTriangleMeshModelPrivate* m_d;
            };
        }
    }
}

#endif // SOFAPBDTRIANGLEMESHMODEL_H
