#ifndef PBD_TRIANGLEMODEL_H
#define PBD_TRIANGLEMODEL_H

#include <vector>

#include "PBDCommon/PBDCommon.h"
#include "PBDSimulation/PBDRigidBody.h"
#include "PBDUtils/PBDIndexedFaceMesh.h"
#include "PBDSimulation/PBDParticleData.h"

//#include "Constraints.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class PBDTriangleModel
            {
                public:
                    PBDTriangleModel();
                    virtual ~PBDTriangleModel();

                    typedef Utilities::PBDIndexedFaceMesh ParticleMesh;

                protected:
                    /** offset which must be added to get the correct index in the particles array */
                    unsigned int m_indexOffset;
                    /** Face mesh of particles which represents the simulation model */
                    ParticleMesh m_particleMesh;
                    Real m_restitutionCoeff;
                    Real m_frictionCoeff;

                public:
                    void updateConstraints();

                    ParticleMesh &getParticleMesh();
                    void cleanupModel();

                    unsigned int getIndexOffset() const;

                    void initMesh(const unsigned int nPoints, const unsigned int nFaces, const unsigned int indexOffset, unsigned int* indices, const ParticleMesh::UVIndices& uvIndices, const ParticleMesh::UVs& uvs);
                    void updateMeshNormals(const ParticleData &pd);

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


#endif // PBD_TRIANGLEMODEL_H
