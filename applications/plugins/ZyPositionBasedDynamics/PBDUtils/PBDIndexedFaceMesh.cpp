#include "PBDIndexedFaceMesh.h"

#include <sofa/helper/logging/Messaging.h>

using namespace sofa::simulation::PBDSimulation::Utilities;

PBDIndexedFaceMesh& PBDIndexedFaceMesh::operator=(PBDIndexedFaceMesh const& other)
{
    m_numPoints		   = other.m_numPoints;
    m_indices          = other.m_indices;
    m_edges            = other.m_edges;
    m_faces            = other.m_faces;
    m_closed           = other.m_closed;
    m_uvIndices        = other.m_uvIndices;
    m_uvs              = other.m_uvs;
    m_verticesPerFace  = other.m_verticesPerFace;
    m_normals          = other.m_normals;
    m_vertexNormals    = other.m_vertexNormals;

    for (size_t i(0u); i < m_faces.size(); ++i)
    {
        m_faces[i].m_edges = new unsigned int[m_verticesPerFace];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64)
        std::copy(other.m_faces[i].m_edges, other.m_faces[i].m_edges + m_verticesPerFace,
            stdext::unchecked_array_iterator<unsigned int*>(m_faces[i].m_edges));
#else
        std::copy(other.m_faces[i].m_edges, other.m_faces[i].m_edges + m_verticesPerFace, m_faces[i].m_edges);
#endif
    }

    m_verticesEdges.resize(other.m_verticesEdges.size());
    for (size_t i(0u); i < m_verticesEdges.size(); ++i)
        m_verticesEdges[i] = other.m_verticesEdges[i];

    m_verticesFaces.resize(other.m_verticesFaces.size());
    for (size_t i(0u); i < m_verticesFaces.size(); ++i)
        m_verticesFaces[i] = other.m_verticesFaces[i];

    return *this;
}

PBDIndexedFaceMesh::PBDIndexedFaceMesh(PBDIndexedFaceMesh const& other)
{
    *this = other;
}

PBDIndexedFaceMesh::PBDIndexedFaceMesh(const unsigned int verticesPerFace)
{
    m_verticesPerFace = verticesPerFace;
    m_closed=false;
}

PBDIndexedFaceMesh::~PBDIndexedFaceMesh()
{
    release();
}

bool PBDIndexedFaceMesh::isClosed() const
{
    return m_closed;
}

void PBDIndexedFaceMesh::initMesh(const unsigned int nPoints, const unsigned int nEdges, const unsigned int nFaces)
{
    m_numPoints = nPoints;
    m_indices.reserve(nFaces*m_verticesPerFace);
    m_edges.reserve(nEdges);
    m_faces.reserve(nFaces);
    m_uvIndices.reserve(nFaces);
    m_uvs.reserve(nPoints);
    m_verticesFaces.reserve(nPoints);
    m_verticesEdges.reserve(nPoints);
    m_normals.reserve(nFaces);
    m_vertexNormals.reserve(nPoints);
}

void PBDIndexedFaceMesh::release()
{
    m_indices.clear();
    m_edges.clear();
    for(unsigned int i=0; i < m_faces.size(); i++)
        delete [] m_faces[i].m_edges;
    m_faces.clear();
    m_uvIndices.clear();
    m_uvs.clear();
    m_verticesFaces.clear();
    m_verticesEdges.clear();
    m_normals.clear();
    m_vertexNormals.clear();
}

/** Add a new face. Indices must be an array of size m_verticesPerFace.
    */
void PBDIndexedFaceMesh::addFace(const unsigned int * const indices)
{
    for (unsigned int i = 0u; i < m_verticesPerFace; i++)
        m_indices.push_back(indices[i]);
}

/** Add a new face. Indices must be an array of size m_verticesPerFace.
    */
void PBDIndexedFaceMesh::addFace(const int * const indices)
{
    for (unsigned int i=0u; i < m_verticesPerFace; i++)
        m_indices.push_back((unsigned int) indices[i]);
}

void PBDIndexedFaceMesh::addUV(const Real u, const Real v)
{
    Vector2r uv;
    uv[0] = u;
    uv[1] = v;
    m_uvs.push_back(uv);
}

void PBDIndexedFaceMesh::addUVIndex(const unsigned int index)
{
    m_uvIndices.push_back(index);
}

void PBDIndexedFaceMesh::buildNeighbors()
{
    typedef std::vector<unsigned int> PEdges;
    typedef std::vector<unsigned int> VertexFE;

    msg_info("PBDIndexedFaceMesh") << "buildNeighbors(): numVertices = " << numVertices() << ", numFaces() = " << numFaces();

    PEdges* pEdges = new PEdges[numVertices()];
    VertexFE* vFaces = new VertexFE[numVertices()];
    VertexFE* vEdges = new VertexFE[numVertices()];

    for(unsigned int i=0; i < m_faces.size(); i++)
        delete [] m_faces[i].m_edges;

    m_edges.clear();
    m_faces.resize(numFaces());

    unsigned int *v = new unsigned int[m_verticesPerFace];
    unsigned int *edges = new unsigned int[m_verticesPerFace*2];

    for(unsigned int i = 0; i < numFaces(); i++)
    {
        msg_info("PBDIndexedFaceMesh") << "Face " << i;

        m_faces[i].m_edges = new unsigned int[m_verticesPerFace];
        for (unsigned int j = 0u; j < m_verticesPerFace; j++)
        {
            msg_info("PBDIndexedFaceMesh") << "Assigning vertex index " << m_indices[m_verticesPerFace * i + j] << " as face vertex " << j;
            v[j] = m_indices[m_verticesPerFace * i + j];
        }

        /*edges[0] = v[0]; edges[1] = v[1];
        edges[2] = v[2]; edges[3] = v[3];
        edges[4] = v[2]; edges[5] = v[0];*/

        /*for (unsigned int j = 0u; j < m_verticesPerFace - 1u; j++)
        {
            msg_info("PBDIndexedFaceMesh") << "Assigning vertex index " << v[j] << " and " << v[j + 1] << " to edge " << j;
            edges[2*j] = v[j];
            edges[2*j+1] = v[j+1];
        }*/

        // This fails for (if read from Wavefront OBJ?)
        /*msg_info("PBDIndexedFaceMesh") << "==== Face " << i << " edge loop closing ====";

        msg_info("PBDIndexedFaceMesh") << "Setting second-last face vertex index to: " << v[m_verticesPerFace - 1];
        msg_info("PBDIndexedFaceMesh") << "Setting last face vertex index " << ((2 * m_verticesPerFace - 1) + 1) << " to: " << v[0];

        edges[2*(m_verticesPerFace-1)] = v[m_verticesPerFace-1];
        edges[2*(m_verticesPerFace-1)+1] = v[0];

        msg_info("PBDIndexedFaceMesh") << "==== Face " << i << " edge-vertex indices ====";*/

        //unsigned int edgeIdx = 0;
        /*for (unsigned int k = 0; k < 2 * m_verticesPerFace; k += 2)
        {
            msg_info("PBDIndexedFaceMesh") << "Edge " << edgeIdx << ": " << edges[2 * k] << " - " << edges[2 * k + 1];
            edgeIdx++;
        }*/

        msg_info("PBDIndexedFaceMesh") << "=== CLUSTERFUCK STARTS ===";
        for (unsigned int j=0u; j < m_verticesPerFace-1u; j++)
        {
            edges[2 * j] = v[j];
            edges[2 * j + 1] = v[j + 1];
            msg_info("PBDIndexedFaceMesh") << "Edge index " << j << ": " << edges[2 * j] <<  " to " << edges[2 * j + 1];
        }

        edges[2*(m_verticesPerFace-1)] = v[m_verticesPerFace-1];
        edges[2*(m_verticesPerFace-1)+1] = v[0];

        msg_info("PBDIndexedFaceMesh") << "Edge list dump for face " << i;
        for (int r = 0; r < m_verticesPerFace*2; r++)
            msg_info("PBDIndexedFaceMesh") << " - " << edges[r];

        msg_info("PBDIndexedFaceMesh") << "=== CLUSTERFUCK ENDS ===";

        msg_info("PBDIndexedFaceMesh") << "=== Edge index list ===";
        msg_info("PBDIndexedFaceMesh") << "Edge 0: " << edges[0] << " - " << edges[1];
        msg_info("PBDIndexedFaceMesh") << "Edge 1: " << edges[2] << " - " << edges[3];
        msg_info("PBDIndexedFaceMesh") << "Edge 2: " << edges[4] << " - " << edges[5];

        for(unsigned int j = 0u; j < m_verticesPerFace; j++)
        {
            // add vertex-face connection
            const unsigned int vIndex = m_indices[m_verticesPerFace * i + j];
            bool found = false;
            for(unsigned int k = 0; k < vFaces[vIndex].size(); k++)
            {
                if (vFaces[vIndex][k] == i)
                {
                    msg_info("PBDIndexedFaceMesh") << "Vertex index " << vIndex << " already assigned as vertex " << k << " in face " << i;
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                vFaces[vIndex].push_back(i);
                msg_info("PBDIndexedFaceMesh") << "Assigning vertex index " << vIndex << " as vertex " << vFaces[vIndex].size() << " in face " << i;
            }

            // add edge information
            const unsigned int a = edges[j*2+0];
            const unsigned int b = edges[j*2+1];

            msg_info("PBDIndexedFaceMesh") << "Edge association with vertices and faces - edge " << a << " - " << b;

            unsigned int edge = 0xffffffff;

            // find edge
            for(unsigned int k = 0; k < pEdges[a].size(); k++)
            {
                const Edge& e = m_edges[pEdges[a][k]];
                if(((e.m_vert[0] == a) || (e.m_vert[0] == b)) &&
                    ((e.m_vert[1] == a) || (e.m_vert[1] == b)))
                {
                    msg_info("PBDIndexedFaceMesh") << "Edge " << a << " - " << b << " already registered, at index " << k << " in pEdges[" << a << "]";
                    edge = pEdges[a][k];
                    break;
                }
            }

            if (edge == 0xffffffff)
            {
                // create new
                Edge e;
                e.m_vert[0] = a;
                e.m_vert[1] = b;
                e.m_face[0] = i;
                e.m_face[1] = 0xffffffff;
                m_edges.push_back(e);
                edge = (unsigned int) m_edges.size() - 1u;

                msg_info("PBDIndexedFaceMesh") << "Creating new edge-face-vertex entry. Edge " << a << " - " << b
                                               << " vertex indices: " << a << ", " << b
                                               << " face indices: " << i << ", 0xffffffff";


                // add vertex-edge connection
                vEdges[a].push_back(edge);
                vEdges[b].push_back(edge);
            }
            else
            {
                Edge& e = m_edges[edge];
                e.m_face[1] = i;
                msg_info("PBDIndexedFaceMesh") << "Updating existing edge-face-entry for edge " << a << " - " << b << "."
                                               << " Face indices now: " << e.m_face[0] << ", " << e.m_face[1];
            }

            // append to points
            pEdges[a].push_back(edge);
            pEdges[b].push_back(edge);

            // append face
            msg_info("PBDIndexedFaceMesh") << "Committing edge record: Face " << i << ", edge " << j;
            m_faces[i].m_edges[j] = edge;

            msg_info("PBDIndexedFaceMesh") << "==== Face " << i << " edge-vertex indices after committing ====";

            for (unsigned int k = 0; k < 2 * m_verticesPerFace; k++)
            {
                msg_info("PBDIndexedFaceMesh") << k << ": " << edges[k];
            }
        }
    }
    delete [] v;
    delete [] edges;

    // build vertex-face structure
    m_verticesFaces.clear(); // to delete old pointers
    m_verticesFaces.resize(numVertices());
    m_verticesEdges.clear(); // to delete old pointers
    m_verticesEdges.resize(numVertices());
    for(unsigned int i = 0; i < numVertices(); i++)
    {
        m_verticesFaces[i].m_numFaces = (unsigned int) vFaces[i].size();
        m_verticesFaces[i].m_fIndices = new unsigned int[m_verticesFaces[i].m_numFaces];
        memcpy(m_verticesFaces[i].m_fIndices, vFaces[i].data(), sizeof(unsigned int)*m_verticesFaces[i].m_numFaces);

        m_verticesEdges[i].m_numEdges = (unsigned int) vEdges[i].size();
        m_verticesEdges[i].m_eIndices = new unsigned int[m_verticesEdges[i].m_numEdges];
        memcpy(m_verticesEdges[i].m_eIndices, vEdges[i].data(), sizeof(unsigned int)*m_verticesEdges[i].m_numEdges);
    }

    // check for boundary
    m_closed = true;
    for (unsigned int i = 0; i < (unsigned int)m_edges.size(); i++)
    {
        Edge& e = m_edges[i];
        if(e.m_face[1] == 0xffffffff)
        {
            msg_warning("PBDIndexedFaceMesh") << "Mesh is not closed because of edge " << i << " - adjacendy is only confirmed to face[0] = " << e.m_face[0];
            m_closed = false;
            break;
        }
    }

    msg_info("PBDIndexedFaceMesh") << "Mesh is closed: " << m_closed;

    delete [] pEdges;
    delete [] vFaces;
    delete [] vEdges;
}

void PBDIndexedFaceMesh::copyUVs(const UVIndices& uvIndices, const UVs& uvs)
{
    m_uvs.clear();
    m_uvs.resize(uvs.size());

    for (unsigned int i = 0; i < uvs.size(); i++)
    {
        m_uvs[i] = uvs[i];
    }

    m_uvIndices.clear();
    m_uvIndices.resize(uvIndices.size());

    for (unsigned int i = 0; i < uvIndices.size(); i++)
    {
        m_uvIndices[i] = uvIndices[i];
    }
}

unsigned int PBDIndexedFaceMesh::getVerticesPerFace() const
{
    msg_info("PBDIndexedFaceMesh") << "VerticesPerFace = " << m_verticesPerFace;
    return m_verticesPerFace;
}

const unsigned int PBDIndexedFaceMesh::getNumVerticesPerFace() const
{
    msg_info("PBDIndexedFaceMesh") << "VerticesPerFace = " << m_verticesPerFace;
    return m_verticesPerFace;
}

unsigned int PBDIndexedFaceMesh::numVertices() const
{
    msg_info("PBDIndexedFaceMesh") << "NumVertices = " << m_numPoints;
    return m_numPoints;
}

unsigned int PBDIndexedFaceMesh::numFaces() const
{
    msg_info("PBDIndexedFaceMesh") << "NumFaces = " << ((unsigned int)m_indices.size() / m_verticesPerFace);
    return (unsigned int)m_indices.size() / m_verticesPerFace;
}

unsigned int PBDIndexedFaceMesh::numEdges() const
{
    msg_info("PBDIndexedFaceMesh") << "NumEdges = " << m_edges.size();
    return (unsigned int)m_edges.size();
}

unsigned int PBDIndexedFaceMesh::numUVs() const
{
    return (unsigned int)m_uvs.size();
}
