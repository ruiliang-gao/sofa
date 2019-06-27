#ifndef CONSTRAINTBASE_H
#define CONSTRAINTBASE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>

#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class PBDSimulationModel;

            class PBDConstraintBase: public virtual sofa::core::objectmodel::BaseObject
            {
                public:
                    SOFA_ABSTRACT_CLASS(PBDConstraintBase, sofa::core::objectmodel::BaseObject);

                    typedef sofa::defaulttype::Vec3Types::Coord       Coord;
                    typedef sofa::helper::vector<Coord>               VecCoord;
                    typedef sofa::core::objectmodel::Data<VecCoord>   Coordinates;
                    typedef sofa::helper::ReadAccessor  <Coordinates> ReadCoord;
                    typedef sofa::helper::WriteAccessor <Coordinates> WriteCoord;

                    typedef sofa::defaulttype::Vec3Types::Deriv       Deriv;
                    typedef sofa::helper::vector<Deriv>               VecDeriv;
                    typedef sofa::core::objectmodel::Data<VecDeriv>   Derivatives;
                    typedef sofa::helper::ReadAccessor  <Derivatives> ReadDeriv;
                    typedef sofa::helper::WriteAccessor <Derivatives> WriteDeriv;

                    typedef sofa::core::objectmodel::Data<sofa::helper::vector<uint>> IndexSet;
                    typedef sofa::defaulttype::BaseMatrix Matrix;

                    unsigned int m_numberOfBodies;
                    /** indices of the linked bodies */
                    unsigned int *m_bodies;

                    PBDConstraintBase(const unsigned int numberOfBodies)
                    {
                            m_numberOfBodies = numberOfBodies;
                            m_bodies = new unsigned int[numberOfBodies];
                    }

                    virtual ~PBDConstraintBase() { delete[] m_bodies; }
                    virtual int &getTypeId() const = 0;

                    virtual bool initConstraintBeforeProjection(PBDSimulationModel &model) { return true; }
                    virtual bool updateConstraint(PBDSimulationModel &model) { return true; }
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter) { return true; }
                    virtual bool solveVelocityConstraint(PBDSimulationModel &model, const unsigned int iter) { return true; }
            };
        }
    }
}

#endif // CONSTRAINTBASE_H
