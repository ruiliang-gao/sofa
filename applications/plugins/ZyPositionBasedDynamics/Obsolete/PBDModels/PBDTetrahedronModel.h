#ifndef PBDTETRAHEDRONMODEL_H
#define PBDTETRAHEDRONMODEL_H

#include "PBDCommon/PBDCommon.h"
#include "PBDUtils/PBDIndexedFaceMesh.h"
#include "PBDUtils/PBDIndexedTetrahedronMesh.h"
#include "PBDSimulation/PBDParticleData.h"
#include <vector>

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class PBDTetrahedronModel
            {
                public:
                    PBDTetrahedronModel();
                    virtual ~PBDTetrahedronModel();

                    typedef Utilities::PBDIndexedFaceMesh SurfaceMesh;
                    typedef Utilities::PBDIndexedTetrahedronMesh ParticleMesh;

                    struct Attachment
                    {
                        unsigned int m_index;
                        unsigned int m_triIndex;
                        Real m_bary[3];
                        Real m_dist;
                        Real m_minError;
                    };

                    Vector3r& getInitialX() { return m_initialX; }
                    void setInitialX(const Vector3r &val) { m_initialX = val; }
                    Matrix3r& getInitialR() { return m_initialR; }
                    void setInitialR(const Matrix3r &val) { m_initialR = val; }
                    Vector3r& getInitialScale() { return m_initialScale; }
                    void setInitialScale(const Vector3r &val) { m_initialScale = val; }

                protected:
                    /** offset which must be added to get the correct index in the particles array */
                    unsigned int m_indexOffset;
                    /** Tet mesh of particles which represents the simulation model */
                    ParticleMesh m_particleMesh;
                    SurfaceMesh m_surfaceMesh;
                    PBDVertexData m_visVertices;
                    SurfaceMesh m_visMesh;
                    Real m_restitutionCoeff;
                    Real m_frictionCoeff;
                    std::vector<Attachment> m_attachments;
                    Vector3r m_initialX;
                    Matrix3r m_initialR;
                    Vector3r m_initialScale;

                    void createSurfaceMesh();
                    void solveQuadraticForZero(const Vector3r& F, const Vector3r& Fu,
                        const Vector3r& Fv, const Vector3r& Fuu,
                        const Vector3r&Fuv, const Vector3r& Fvv,
                        Real& u, Real& v);
                    bool pointInTriangle(const Vector3r& p0, const Vector3r& p1, const Vector3r& p2,
                        const Vector3r& p, Vector3r& inter, Vector3r &bary);


                public:
                    void updateConstraints();

                    SurfaceMesh &getSurfaceMesh();
                    PBDVertexData &getVisVertices();
                    SurfaceMesh &getVisMesh();
                    ParticleMesh &getParticleMesh();
                    void cleanupModel();

                    unsigned int getIndexOffset() const;

                    void initMesh(const unsigned int nPoints, const unsigned int nTets, const unsigned int indexOffset, unsigned int* indices);
                    void updateMeshNormals(const PBDParticleData &pd);

                    /** Attach a visualization mesh to the surface of the body.
                     * Important: The vertex normals have to be updated before
                     * calling this function by calling updateMeshNormals().
                     */
                    void attachVisMesh(const PBDParticleData &pd);

                    /** Update the visualization mesh of the body.
                    * Important: The vertex normals have to be updated before
                    * calling this function by calling updateMeshNormals().
                    */
                    void updateVisMesh(const PBDParticleData &pd);

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

#endif // PBDTETRAHEDRONMODEL_H
