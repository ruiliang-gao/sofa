#ifndef PBDLINEMODEL_H
#define PBDLINEMODEL_H

#include "PBDCommon/PBDCommon.h"
#include <vector>
#include "PBDSimulation/PBDRigidBody.h"
#include "PBDUtils/PBDIndexedFaceMesh.h"
#include "PBDSimulation/PBDParticleData.h"
#include "PBDConstraints/PBDConstraints.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class PBDLineModel: public sofa::core::objectmodel::BaseObject
            {
                struct OrientedEdge
                {
                    OrientedEdge(){}
                    OrientedEdge(unsigned int p0, unsigned int p1, unsigned int q0)
                    {
                        m_vert[0] = p0;
                        m_vert[1] = p1;
                        m_quat = q0;
                    }
                    unsigned int m_vert[2];
                    unsigned int m_quat;
                };

                public:
                    typedef std::vector<OrientedEdge> Edges;

                    PBDLineModel();
                    virtual ~PBDLineModel();

                protected:
                    /** offset which must be added to get the correct index in the particles array */
                    unsigned int m_indexOffset;
                    /** offset which must be added to get the correct index in the quaternions array */
                    unsigned int m_indexOffsetQuaternions;
                    unsigned int m_nPoints, m_nQuaternions;
                    Edges m_edges;
                    Real m_restitutionCoeff;
                    Real m_frictionCoeff;

                public:
                    void updateConstraints();

                    Edges &getEdges();

                    unsigned int getIndexOffset() const;
                    unsigned int getIndexOffsetQuaternions() const;

                    void initMesh(const unsigned int nPoints, const unsigned int nQuaternions, const unsigned int indexOffset, const unsigned int indexOffsetQuaternions, unsigned int* indices, unsigned int* indicesQuaternions);

                    FORCE_INLINE Real getRestitutionCoeff() const
                    {
                        return m_restitutionCoeff;
                    }

                    FORCE_INLINE void setRestitutionCoeff(Real val)
                    {
                        m_restitutionCoeff = val;
                    }

                    FORCE_INLINE Real getFrictionCoeff() const
                    {
                        return m_frictionCoeff;
                    }

                    FORCE_INLINE void setFrictionCoeff(Real val)
                    {
                        m_frictionCoeff = val;
                    }
            };
        }
    }
}

#endif // PBDLINEMODEL_H
