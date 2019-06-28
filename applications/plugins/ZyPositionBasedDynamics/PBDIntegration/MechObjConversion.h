#ifndef MECHOBJCONVERSION_H
#define MECHOBJCONVERSION_H

#include "initZyPositionBasedDynamicsPlugin.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <SofaBaseMechanics/MechanicalObject.h>

#include <PBDModels/PBDSimulationModel.h>

#include <vector>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API MechObjConversion: public sofa::core::objectmodel::BaseObject
            {
                public:
                    MechObjConversion(PBDSimulationModel*);

                    const std::vector<sofa::component::container::MechanicalObject<sofa::defaulttype::Vec3Types>*>& getMechanicalObjects() const { return m_mechObjects; }
                    void setMechanicalObjects(std::vector<sofa::component::container::MechanicalObject<sofa::defaulttype::Vec3Types>*>& mechObjects) { m_mechObjects = mechObjects; }

                    bool convertToPBDObjects();

                private:
                    std::shared_ptr<PBDSimulationModel> m_model;
                    std::vector<sofa::component::container::MechanicalObject<sofa::defaulttype::Vec3Types>*> m_mechObjects;
            };
        }
    }
}

#endif // MECHOBJCONVERSION_H
