#ifndef PBDRIGIDBODYGEOMETRY_H
#define PBDRIGIDBODYGEOMETRY_H

#include <vector>

#include "PBDCommon/PBDCommon.h"
#include "PBDUtils/PBDIndexedFaceMesh.h"
#include "PBDSimulation/PBDParticleData.h"

// TODO: Replace with corresponding SOFA classes

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class PBDRigidBodyGeometry
            {
                public:
                    PBDRigidBodyGeometry();
                    virtual ~PBDRigidBodyGeometry();

                    typedef Utilities::PBDIndexedFaceMesh Mesh;

                protected:
                    Mesh m_mesh;
                    PBDVertexData m_vertexData_local;
                    PBDVertexData m_vertexData;

                public:
                    Mesh &getMesh();
                    PBDVertexData &getVertexData();
                    const PBDVertexData &getVertexData() const;
                    PBDVertexData &getVertexDataLocal();
                    const PBDVertexData &getVertexDataLocal() const;

                    void initMesh(const unsigned int nVertices, const unsigned int nFaces, const Vector3r *vertices, const unsigned int* indices, const Mesh::UVIndices& uvIndices, const Mesh::UVs& uvs, const Vector3r &scale = Vector3r(1.0, 1.0, 1.0));
                    void updateMeshTransformation(const Vector3r &x, const Matrix3r &R);
                    void updateMeshNormals(const PBDVertexData &vd);

            };
        }
    }
}

#endif // PBDRIGIDBODYGEOMETRY_H
