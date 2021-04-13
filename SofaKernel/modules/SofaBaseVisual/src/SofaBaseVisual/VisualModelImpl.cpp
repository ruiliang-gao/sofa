/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaBaseVisual/VisualModelImpl.h>

#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <SofaBaseTopology/QuadSetTopologyModifier.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>
#include <SofaBaseTopology/HexahedronSetTopologyModifier.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/SparseGridTopology.h>
#include <SofaBaseTopology/CommonAlgorithms.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/system/FileRepository.h>

#include <sstream>
#include <map>
#include <memory>

namespace sofa::component::visualmodel
{
using sofa::helper::types::RGBAColor;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;
using namespace sofa::core::loader;
using helper::vector;

Vec3State::Vec3State()
    : m_positions(initData(&m_positions, "position", "Vertices coordinates"))
    , m_restPositions(initData(&m_restPositions, "restPosition", "Vertices rest coordinates"))
    , m_vnormals (initData (&m_vnormals, "normal", "Normals of the model"))
    , modified(false)
{
    m_positions.setGroup("Vector");
    m_restPositions.setGroup("Vector");
    m_vnormals.setGroup("Vector");
}

void Vec3State::resize(Size vsize)
{
    helper::WriteOnlyAccessor< Data<VecCoord > > positions = m_positions;
    if( positions.size() == vsize ) return;
    helper::WriteOnlyAccessor< Data<VecCoord > > restPositions = m_restPositions;
    helper::WriteOnlyAccessor< Data<VecDeriv > > normals = m_vnormals;

    positions.resize(vsize);
    restPositions.resize(vsize); // todo allocate restpos only when it is necessary
    normals.resize(vsize);

    modified = true;
}

Size Vec3State::getSize() const { return Size(m_positions.getValue().size()); }

Data<Vec3State::VecCoord>* Vec3State::write(     core::VecCoordId  v )
{
    modified = true;

    if( v == core::VecCoordId::position() )
        return &m_positions;
    if( v == core::VecCoordId::restPosition() )
        return &m_restPositions;

    return nullptr;
}

const Data<Vec3State::VecCoord>* Vec3State::read(core::ConstVecCoordId  v )  const
{
    if( v == core::VecCoordId::position() )
        return &m_positions;
    if( v == core::VecCoordId::restPosition() )
        return &m_restPositions;

    return nullptr;
}

Data<Vec3State::VecDeriv>*	Vec3State::write(core::VecDerivId v )
{
    if( v == core::VecDerivId::normal() )
        return &m_vnormals;

    return nullptr;
}

const Data<Vec3State::VecDeriv>* Vec3State::read(core::ConstVecDerivId v ) const
{
    if( v == core::VecDerivId::normal() )
        return &m_vnormals;

    return nullptr;
}


void VisualModelImpl::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->core::visual::VisualModel::parse(arg);

    VisualModelImpl* obj = this;

    if (arg->getAttribute("normals")!=nullptr)
        obj->setUseNormals(arg->getAttributeAsInt("normals", 1)!=0);

    if (arg->getAttribute("castshadow")!=nullptr)
        obj->setCastShadow(arg->getAttributeAsInt("castshadow", 1)!=0);

    if (arg->getAttribute("flip")!=nullptr)
        obj->flipFaces();

    if (arg->getAttribute("color"))
        obj->setColor(arg->getAttribute("color"));

    if (arg->getAttribute("su")!=nullptr || arg->getAttribute("sv")!=nullptr)
        m_scaleTex = TexCoord(arg->getAttributeAsFloat("su",1.0),
                              arg->getAttributeAsFloat("sv",1.0));

    if (arg->getAttribute("du")!=nullptr || arg->getAttribute("dv")!=nullptr)
        m_translationTex = TexCoord(arg->getAttributeAsFloat("du",0.0),
                                    arg->getAttributeAsFloat("dv",0.0));

    if (arg->getAttribute("rx")!=nullptr || arg->getAttribute("ry")!=nullptr || arg->getAttribute("rz")!=nullptr)
        m_rotation.setValue(Vec3Real((Real)arg->getAttributeAsFloat("rx",0.0),
                                     (Real)arg->getAttributeAsFloat("ry",0.0),
                                     (Real)arg->getAttributeAsFloat("rz",0.0)));

    if (arg->getAttribute("dx")!=nullptr || arg->getAttribute("dy")!=nullptr || arg->getAttribute("dz")!=nullptr)
        m_translation.setValue(Vec3Real((Real)arg->getAttributeAsFloat("dx",0.0),
                                        (Real)arg->getAttributeAsFloat("dy",0.0),
                                        (Real)arg->getAttributeAsFloat("dz",0.0)));

    if (arg->getAttribute("scale")!=nullptr)
    {
        m_scale.setValue(Vec3Real((Real)arg->getAttributeAsFloat("scale",1.0),
                                  (Real)arg->getAttributeAsFloat("scale",1.0),
                                  (Real)arg->getAttributeAsFloat("scale",1.0)));
    }
    else if (arg->getAttribute("sx")!=nullptr || arg->getAttribute("sy")!=nullptr || arg->getAttribute("sz")!=nullptr)
    {
        m_scale.setValue(Vec3Real((Real)arg->getAttributeAsFloat("sx",1.0),
                                  (Real)arg->getAttributeAsFloat("sy",1.0),
                                  (Real)arg->getAttributeAsFloat("sz",1.0)));
    }
}

int VisualModelImplClass = core::RegisterObject("Generic visual model. If a viewer is active it will replace the VisualModel alias, otherwise nothing will be displayed.")
        .add< VisualModelImpl >()
        .addAlias("VisualModel")
        ;

VisualModelImpl::VisualModelImpl() //const std::string &name, std::string filename, std::string loader, std::string textureName)
    :  useTopology(false)
    , lastMeshRev(-1)
    , castShadow(true)
    , m_initRestPositions(initData  (&m_initRestPositions, false, "initRestPositions", "True if rest positions must be initialized with initial positions"))
    , m_useNormals		(initData	(&m_useNormals, true, "useNormals", "True if normal smoothing groups should be read from file"))
    , m_updateNormals   (initData   (&m_updateNormals, true, "updateNormals", "True if normals should be updated at each iteration"))
    , m_computeTangents (initData   (&m_computeTangents, false, "computeTangents", "True if tangents should be computed at startup"))
    , m_updateTangents  (initData   (&m_updateTangents, true, "updateTangents", "True if tangents should be updated at each iteration"))
    , m_handleDynamicTopology (initData   (&m_handleDynamicTopology, true, "handleDynamicTopology", "True if topological changes should be handled"))
    , m_fixMergedUVSeams (initData   (&m_fixMergedUVSeams, true, "fixMergedUVSeams", "True if UV seams should be handled even when duplicate UVs are merged"))
    , m_keepLines (initData   (&m_keepLines, false, "keepLines", "keep and draw lines (false by default)"))
    , m_genTex3d(initData(&m_genTex3d, false, "genTex3d", "True to enable computation for 3d texture"))
    , m_vertices2       (initData   (&m_vertices2, "vertices", "vertices of the model (only if vertices have multiple normals/texcoords, otherwise positions are used)"))
    , m_vtexcoords      (initData   (&m_vtexcoords, "texcoords", "coordinates of the texture"))
    , m_vtexcoords3     (initData(&m_vtexcoords3, "texcoords3", "3d coordinates of the texture"))
    , m_vtangents       (initData   (&m_vtangents, "tangents", "tangents for normal mapping"))
    , m_vbitangents     (initData   (&m_vbitangents, "bitangents", "tangents for normal mapping"))
    , m_edges           (initData   (&m_edges, "edges", "edges of the model"))
    , m_triangles       (initData   (&m_triangles, "triangles", "triangles of the model"))
    , m_quads           (initData   (&m_quads, "quads", "quads of the model"))
    , m_vertPosIdx      (initData   (&m_vertPosIdx, "vertPosIdx", "If vertices have multiple normals/texcoords stores vertices position indices"))
    , m_vertNormIdx     (initData   (&m_vertNormIdx, "vertNormIdx", "If vertices have multiple normals/texcoords stores vertices normal indices"))
    , fileMesh          (initData   (&fileMesh, "filename"," Path to an ogl model"))
    , texturename       (initData   (&texturename, "texturename", "Name of the Texture"))
    , m_translation     (initData   (&m_translation, Vec3Real(), "translation", "Initial Translation of the object"))
    , m_rotation        (initData   (&m_rotation, Vec3Real(), "rotation", "Initial Rotation of the object"))
    , m_scale           (initData   (&m_scale, Vec3Real(1.0,1.0,1.0), "scale3d", "Initial Scale of the object"))
    , m_scaleTex        (initData   (&m_scaleTex, TexCoord(1.0,1.0), "scaleTex", "Scale of the texture"))
    , m_scaleTex3(initData(&m_scaleTex3, TexCoord3(1.0, 1.0, 1.0), "scaleTex3", "Scale of the 3d texture"))
    , m_translationTex3(initData(&m_translationTex3, TexCoord3(0.0, 0.0, 0.0), "translationTex3", "Translation of the 3d texture"))
    , m_translationTex  (initData   (&m_translationTex, TexCoord(0.0,0.0), "translationTex", "Translation of the texture"))
    , material			(initData	(&material, "material", "Material")) // tex(nullptr)
    , putOnlyTexCoords	(initData	(&putOnlyTexCoords, (bool) false, "putOnlyTexCoords", "Give Texture Coordinates without the texture binding"))
    , srgbTexturing		(initData	(&srgbTexturing, (bool) false, "srgbTexturing", "When sRGB rendering is enabled, is the texture in sRGB colorspace?"))
    , materials			(initData	(&materials, "materials", "List of materials"))
    , groups			(initData	(&groups, "groups", "Groups of triangles and quads using a given material"))
    , l_topology        (initLink   ("topology", "link to the topology container"))
    , xformsModified(false)
{
    m_topology = nullptr;

    //material.setDisplayed(false);
    addAlias(&fileMesh, "fileMesh");

    m_vertices2     .setGroup("Vector");
    m_vnormals      .setGroup("Vector");
    m_vtexcoords    .setGroup("Vector");
    m_vtexcoords3.setGroup("Vector");
    m_vtangents     .setGroup("Vector");
    m_vbitangents   .setGroup("Vector");
    m_edges         .setGroup("Vector");
    m_triangles     .setGroup("Vector");
    m_quads         .setGroup("Vector");

    m_translation   .setGroup("Transformation");
    m_rotation      .setGroup("Transformation");
    m_scale         .setGroup("Transformation");

    m_edges.setAutoLink(false); // disable linking of edges by default

    // add one identity matrix
    xforms.resize(1);
}

VisualModelImpl::~VisualModelImpl()
{
}

bool VisualModelImpl::hasTransparent()
{
    const Material& material = this->material.getValue();
    helper::ReadAccessor< Data< helper::vector<FaceGroup> > > groups = this->groups;
    helper::ReadAccessor< Data< helper::vector<Material> > > materials = this->materials;
    if (groups.empty())
        return (material.useDiffuse && material.diffuse[3] < 1.0);
    else
    {
        for (std::size_t i = 0; i < groups.size(); ++i)
        {
            const Material& m = (groups[i].materialId == -1) ? material : materials[groups[i].materialId];
            if (m.useDiffuse && m.diffuse[3] < 1.0)
                return true;
        }
    }
    return false;
}

bool VisualModelImpl::hasOpaque()
{
    const Material& material = this->material.getValue();
    helper::ReadAccessor< Data< helper::vector<FaceGroup> > > groups = this->groups;
    helper::ReadAccessor< Data< helper::vector<Material> > > materials = this->materials;
    if (groups.empty())
        return !(material.useDiffuse && material.diffuse[3] < 1.0);
    else
    {
        for (std::size_t i = 0; i < groups.size(); ++i)
        {
            const Material& m = (groups[i].materialId == -1) ? material : materials[groups[i].materialId];
            if (!(m.useDiffuse && m.diffuse[3] < 1.0))
                return true;
        }
    }
    return false;
}

void VisualModelImpl::drawVisual(const core::visual::VisualParams* vparams)
{
    //Update external buffers (like VBO) if the mesh change AFTER doing the updateVisual() process
    if(m_vertices2.isDirty())
    {
        updateBuffers();
    }

    if (hasOpaque())
        internalDraw(vparams,false);
}

void VisualModelImpl::drawTransparent(const core::visual::VisualParams* vparams)
{
    if (hasTransparent())
        internalDraw(vparams,true);
}

void VisualModelImpl::drawShadow(const core::visual::VisualParams* vparams)
{
    if (hasOpaque() && getCastShadow())
        internalDraw(vparams, false);
}

void VisualModelImpl::setMesh(helper::io::Mesh &objLoader, bool tex)
{
    const auto &facetsImport = objLoader.getFacets();
    const vector< Vector3 > &verticesImport = objLoader.getVertices();
    const vector< Vector3 > &normalsImport = objLoader.getNormals();
    const vector< Vector3 > &texCoordsImport = objLoader.getTexCoords();

    const Material &materialImport = objLoader.getMaterial();

    if (!material.isSet() && materialImport.activated)
    {
        Material M;
        M = materialImport;
        material.setValue(M);
    }

    if (!objLoader.getGroups().empty())
    {
        // Get informations about the multiple materials
        helper::WriteAccessor< Data< helper::vector<Material> > > materials = this->materials;
        helper::WriteAccessor< Data< helper::vector<FaceGroup> > > groups = this->groups;
        materials.resize(objLoader.getMaterials().size());
        for (std::size_t i=0; i<materials.size(); ++i)
            materials[i] = objLoader.getMaterials()[i];

        // compute the edge / triangle / quad index corresponding to each facet
        // convert the groups info
        enum { NBE = 0, NBT = 1, NBQ = 2 };
        helper::fixed_array<visual_index_type, 3> nbf{ 0,0,0 };
        helper::vector< helper::fixed_array<visual_index_type, 3> > facet2tq;
        facet2tq.resize(facetsImport.size()+1);
        for (std::size_t i = 0; i < facetsImport.size(); i++)
        {
            facet2tq[i] = nbf;
            const auto& vertNormTexIndex = facetsImport[i];
            const auto& verts = vertNormTexIndex[0];
            if (verts.size() < 2)
                ; // ignore points
            else if (verts.size() == 2)
                nbf[NBE] += 1;
            else if (verts.size() == 4)
                nbf[NBQ] += 1;
            else
                nbf[NBT] += visual_index_type(verts.size()-2);
        }
        facet2tq[facetsImport.size()] = nbf;
        groups.resize(objLoader.getGroups().size());
        for (std::size_t ig = 0; ig < groups.size(); ig++)
        {
            const PrimitiveGroup& g0 = objLoader.getGroups()[ig];
            FaceGroup& g = groups[ig];
            if (g0.materialName.empty()) g.materialName = "defaultMaterial";
            else                         g.materialName = g0.materialName;
            if (g0.groupName.empty())    g.groupName = "defaultGroup";
            else                         g.groupName = g0.groupName;
            g.materialId = g0.materialId;
            g.edge0 = facet2tq[g0.p0][NBE];
            g.nbe = facet2tq[g0.p0+g0.nbp][NBE] - g.edge0;
            g.tri0 = facet2tq[g0.p0][NBT];
            g.nbt = facet2tq[g0.p0+g0.nbp][NBT] - g.tri0;
            g.quad0 = facet2tq[g0.p0][NBQ];
            g.nbq = facet2tq[g0.p0+g0.nbp][NBQ] - g.quad0;
            if (g.materialId == -1 && !g0.materialName.empty())
                msg_info() << "face group " << ig << " name " << g0.materialName << " uses missing material " << g0.materialName << "   ";
        }
    }

    std::size_t nbVIn = verticesImport.size();
    // First we compute for each point how many pair of normal/texcoord indices are used
    // The map store the final index of each combinaison
    vector< std::map< std::pair<Index,Index>, Index > > vertTexNormMap;
    vertTexNormMap.resize(nbVIn);
    for (std::size_t i = 0; i < facetsImport.size(); i++)
    {
        const auto& vertNormTexIndex = facetsImport[i];
        if (vertNormTexIndex[0].size() < 3 && !m_keepLines.getValue() ) continue; // ignore lines
        const auto& verts = vertNormTexIndex[0];
        const auto& texs = vertNormTexIndex[1];
        const auto& norms = vertNormTexIndex[2];
        for (std::size_t j = 0; j < verts.size(); j++)
        {
            vertTexNormMap[verts[j]][std::make_pair((tex ? texs[j] : sofa::InvalidID), (m_useNormals.getValue() ? norms[j] : 0))] = 0;
        }
    }

    // Then we can compute how many vertices are created
    std::size_t nbVOut = 0;
    bool vsplit = false;
    for (std::size_t i = 0; i < nbVIn; i++)
    {
        nbVOut += vertTexNormMap[i].size();
    }

    msg_info() << nbVIn << " input positions, " << nbVOut << " final vertices.   ";

    if (nbVIn != nbVOut)
        vsplit = true;

    // Then we can create the final arrays
    VecCoord& restPositions = *(m_restPositions.beginEdit());
    VecCoord& positions = *(m_positions.beginEdit());
    VecCoord& vertices2 = *(m_vertices2.beginEdit());
    VecDeriv& vnormals = *(m_vnormals.beginEdit());
    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    auto& vertPosIdx = (*m_vertPosIdx.beginEdit());
    auto& vertNormIdx = (*m_vertNormIdx.beginEdit());;

    positions.resize(nbVIn);

    if (m_initRestPositions.getValue())
        restPositions.resize(nbVIn);

    if (vsplit)
    {
        vertices2.resize(nbVOut);
        if( m_useNormals.getValue() ) vnormals.resize(nbVOut);
        vtexcoords.resize(nbVOut);
        vertPosIdx.resize(nbVOut);
        vertNormIdx.resize(nbVOut);
    }
    else
    {
        //vertices2.resize(nbVIn);
        if( m_useNormals.getValue() ) vnormals.resize(nbVIn);
        vtexcoords.resize(nbVIn);
    }

    sofa::Size nbNOut = 0; /// Number of different normals
    for (sofa::Index i = 0, j = 0; i < nbVIn; i++)
    {
        positions[i] = verticesImport[i];

        if (m_initRestPositions.getValue())
            restPositions[i] = verticesImport[i];

        std::map<sofa::Index, sofa::Index> normMap;
        for (auto it = vertTexNormMap[i].begin();
             it != vertTexNormMap[i].end(); ++it)
        {
            Index t = it->first.first;
            Index n = it->first.second;
            if ( m_useNormals.getValue() && n < normalsImport.size())
                vnormals[j] = normalsImport[n];
            if (t < texCoordsImport.size())
                vtexcoords[j] = texCoordsImport[t];

            if (vsplit)
            {
                vertices2[j] = verticesImport[i];
                vertPosIdx[j] = i;
                if (normMap.count(n))
                    vertNormIdx[j] = normMap[n];
                else
                {
                    vertNormIdx[j] = nbNOut;
                    normMap[n] = nbNOut++;
                }
            }
            it->second = j++;
        }
    }

    if( vsplit && nbNOut == nbVOut )
        vertNormIdx.resize(0);


    m_vertices2.endEdit();
    m_vnormals.endEdit();
    m_vtexcoords.endEdit();
    m_positions.endEdit();
    m_restPositions.endEdit();
    m_vertPosIdx.endEdit();
    m_vertNormIdx.endEdit();

    // Then we create the triangles and quads
    VecVisualEdge& edges = *(m_edges.beginEdit());
    VecVisualTriangle& triangles = *(m_triangles.beginEdit());
    VecVisualQuad& quads = *(m_quads.beginEdit());

    for (std::size_t i = 0; i < facetsImport.size(); i++)
    {
        const auto& vertNormTexIndex = facetsImport[i];
        const auto& verts = vertNormTexIndex[0];
        const auto& texs = vertNormTexIndex[1];
        const auto& norms = vertNormTexIndex[2];
        vector<visual_index_type> idxs;
        idxs.resize(verts.size());
        for (std::size_t j = 0; j < verts.size(); j++)
        {
            idxs[j] = vertTexNormMap[verts[j]][std::make_pair((tex?texs[j]:-1), (m_useNormals.getValue() ? norms[j] : 0))];
            if (idxs[j] >= nbVOut)
            {
                msg_error() << this->getPathName()<<" index "<<idxs[j]<<" out of range";
                idxs[j] = 0;
            }
        }

        if (verts.size() == 2)
        {
            edges.push_back({idxs[0], idxs[1]});
        }
        else if (verts.size() == 4)
        {
            quads.push_back({ idxs[0],idxs[1],idxs[2],idxs[3] });
        }
        else
        {
            for (std::size_t j = 2; j < verts.size(); j++)
            {
                triangles.push_back({ idxs[0],idxs[j - 1],idxs[j] });
            }
        }
    }

    m_edges.endEdit();
    m_triangles.endEdit();
    m_quads.endEdit();

    computeNormals();
    computeTangents();

}

bool VisualModelImpl::load(const std::string& filename, const std::string& loader, const std::string& textureName)
{
    using sofa::helper::io::Mesh;

    //      bool tex = !textureName.empty() || putOnlyTexCoords.getValue();
    if (!textureName.empty())
    {
        std::string textureFilename(textureName);
        if (sofa::helper::system::DataRepository.findFile(textureFilename))
        {
            msg_info() << "loading file " << textureName;
            bool textureLoaded = loadTexture(textureName);
            if(!textureLoaded)
            {
                msg_error()<<"Texture "<<textureName<<" cannot be loaded";
            }
        }
        else
        {
            msg_error() << "Texture \"" << textureName << "\" not found";
        }
    }

    // Make sure all Data are up-to-date
    m_vertices2.updateIfDirty();
    m_vnormals.updateIfDirty();
    m_vtexcoords.updateIfDirty();
    m_vtangents.updateIfDirty();
    m_vbitangents.updateIfDirty();
    m_edges.updateIfDirty();
    m_triangles.updateIfDirty();
    m_quads.updateIfDirty();

    if (!filename.empty() && (m_positions.getValue()).size() == 0 && (m_vertices2.getValue()).size() == 0)
    {
        std::string meshFilename(filename);
        if (sofa::helper::system::DataRepository.findFile(meshFilename))
        {
            //name = filename;
            std::unique_ptr<Mesh> objLoader;
            if (loader.empty())
            {
                objLoader.reset(Mesh::Create(filename));
            }
            else
            {
                objLoader.reset(Mesh::Create(loader, filename));
            }

            if (objLoader.get() == 0)
            {
                msg_error() << "Mesh creation failed. Loading mesh file directly inside the VisualModel is not maintained anymore. Use a MeshLoader and link the Data to the VisualModel. E.g:" << msgendl
                    << "<MeshObjLoader name='myLoader' filename='myFilePath.obj'/>" << msgendl
                    << "<OglModel src='@myLoader'/>";
                return false;
            }
            else
            {				
                if(objLoader.get()->loaderType == "obj")
                {
                    //Modified: previously, the texture coordinates were not loaded correctly if no texture name was specified.
                    //setMesh(*objLoader,tex);
                    msg_warning() << "Loading obj mesh file directly inside the VisualModel will be deprecated soon. Use a MeshObjLoader and link the Data to the VisualModel. E.g:" << msgendl
                        << "<MeshObjLoader name='myLoader' filename='myFilePath.obj'/>" << msgendl
                        << "<OglModel src='@myLoader'/>";
                    
                    setMesh(*objLoader, true); 
                }
                else
                {
                    msg_error() << "Loading mesh file directly inside the VisualModel is not anymore supported since release 18.06. Use a MeshLoader and link the Data to the VisualModel. E.g:" << msgendl
                        << "<MeshObjLoader name='myLoader' filename='myFilePath.obj'/>" << msgendl
                        << "<OglModel src='@myLoader'/>";
                    return false;
                }
            }

            if(textureName.empty())
            {
                //we check how many textures are linked with a material (only if a texture name is not defined in the scn file)
                bool isATextureLinked = false;
                for (std::size_t i = 0 ; i < this->materials.getValue().size() ; i++)
                {
                    //we count only the texture with an activated material
                    if (this->materials.getValue()[i].useTexture && this->materials.getValue()[i].activated)
                    {
                        isATextureLinked=true;
                        break;
                    }
                }
                if (isATextureLinked)
                {
                    loadTextures();
                }
            }
        }
        else
        {
            msg_error() << "Mesh \"" << filename << "\" not found";
        }
    }
    else
    {
        if ((m_positions.getValue()).size() == 0 && (m_vertices2.getValue()).size() == 0)
        {
            msg_info() << "will use Topology.";
            useTopology = true;
        }

        modified = true;
    }

    if (!xformsModified)
    {
        // add one identity matrix
        xforms.resize(1);
    }
    applyUVTransformation();
    return true;
}

void VisualModelImpl::applyUVTransformation()
{
    applyUVScale(m_scaleTex.getValue()[0], m_scaleTex.getValue()[1]);
    applyUVTranslation(m_translationTex.getValue()[0], m_translationTex.getValue()[1]);
    m_scaleTex.setValue(TexCoord(1,1));
    m_translationTex.setValue(TexCoord(0,0));
}

void VisualModelImpl::applyUVWTransformation()
{
    SReal dU = m_translationTex3.getValue()[0];
    SReal dV = m_translationTex3.getValue()[1];
    SReal dW = m_translationTex3.getValue()[2];
    
        SReal sU = m_scaleTex3.getValue()[0];
    SReal sV = m_scaleTex3.getValue()[1];
    SReal sW = m_scaleTex3.getValue()[2];
    
        VecTexCoord3 & vtexcoords3 = *(m_vtexcoords3.beginEdit());
    
        for (unsigned int i = 0; i < vtexcoords3.size(); i++)
        {
        vtexcoords3[i][0] *= sU;
        vtexcoords3[i][1] *= sV;
        vtexcoords3[i][2] *= sW;
        }
    
        for (unsigned int i = 0; i < vtexcoords3.size(); i++)
         {
        vtexcoords3[i][0] += dU;
        vtexcoords3[i][1] += dV;
        vtexcoords3[i][2] += dW;
        }
     m_vtexcoords3.endEdit();
    
        m_scaleTex3.setValue(TexCoord3(1, 1, 1));
    m_translationTex3.setValue(TexCoord3(0, 0, 0));
    }

void VisualModelImpl::applyTranslation(const SReal dx, const SReal dy, const SReal dz)
{
    Coord d((Real)dx,(Real)dy,(Real)dz);

    Data< VecCoord >* d_x = this->write(core::VecCoordId::position());
    VecCoord &x = *d_x->beginEdit();

    for (std::size_t i = 0; i < x.size(); i++)
    {
        x[i] += d;
    }

    d_x->endEdit();

    if(m_initRestPositions.getValue())
    {
        VecCoord& restPositions = *(m_restPositions.beginEdit());

        for (std::size_t i = 0; i < restPositions.size(); i++)
        {
            restPositions[i] += d;
        }

        m_restPositions.endEdit();
    }


    updateVisual();
}

void VisualModelImpl::applyRotation(const SReal rx, const SReal ry, const SReal rz)
{
    Quaternion q = helper::Quater<SReal>::createQuaterFromEuler( Vec<3,SReal>(rx,ry,rz)*M_PI/180.0);
    applyRotation(q);
}

void VisualModelImpl::applyRotation(const Quat q)
{
    Data< VecCoord >* d_x = this->write(core::VecCoordId::position());
    VecCoord &x = *d_x->beginEdit();

    for (std::size_t i = 0; i < x.size(); i++)
    {
        x[i] = q.rotate(x[i]);
    }

    d_x->endEdit();

    if(m_initRestPositions.getValue())
    {
        VecCoord& restPositions = *(m_restPositions.beginEdit());

        for (std::size_t i = 0; i < restPositions.size(); i++)
        {
            restPositions[i] = q.rotate(restPositions[i]);
        }

        m_restPositions.endEdit();
    }

    updateVisual();
}

void VisualModelImpl::applyScale(const SReal sx, const SReal sy, const SReal sz)
{
    Data< VecCoord >* d_x = this->write(core::VecCoordId::position());
    VecCoord &x = *d_x->beginEdit();

    for (std::size_t i = 0; i < x.size(); i++)
    {
        x[i][0] *= (Real)sx;
        x[i][1] *= (Real)sy;
        x[i][2] *= (Real)sz;
    }

    d_x->endEdit();

    if(m_initRestPositions.getValue())
    {
        VecCoord& restPositions = *(m_restPositions.beginEdit());

        for (std::size_t i = 0; i < restPositions.size(); i++)
        {
            restPositions[i][0] *= (Real)sx;
            restPositions[i][1] *= (Real)sy;
            restPositions[i][2] *= (Real)sz;
        }

        m_restPositions.endEdit();
    }

    updateVisual();
}

void VisualModelImpl::applyUVTranslation(const Real dU, const Real dV)
{
    float dUf = float(dU);
    float dVf = float(dV);
    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    for (std::size_t i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] += dUf;
        vtexcoords[i][1] += dVf;
    }
    m_vtexcoords.endEdit();
}

void VisualModelImpl::applyUVScale(const Real scaleU, const Real scaleV)
{
    float scaleUf = float(scaleU);
    float scaleVf = float(scaleV);
    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    for (std::size_t i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] *= scaleUf;
        vtexcoords[i][1] *= scaleVf;
    }
    m_vtexcoords.endEdit();
}


template<class VecCoord>
class VisualModelPointHandler : public sofa::component::topology::TopologyDataHandler<sofa::core::topology::Point,VecCoord >
{
public:
    typedef typename VecCoord::value_type Coord;
    typedef typename Coord::value_type Real;
    VisualModelPointHandler(VisualModelImpl* obj, sofa::component::topology::PointData<VecCoord>* data, int algo)
        : sofa::component::topology::TopologyDataHandler<sofa::core::topology::Point, VecCoord >(data), obj(obj), algo(algo) {}

    void applyCreateFunction(Index /*pointIndex*/, Coord& dest, const sofa::core::topology::Point &,
                             const sofa::helper::vector< Index > &ancestors,
                             const sofa::helper::vector< double > &coefs)
    {
        const VecCoord& x = this->m_topologyData->getValue();
        if (!ancestors.empty())
        {
            if (algo == 1 && ancestors.size() > 1) //fixMergedUVSeams
            {
                Coord c0 = x[ancestors[0]];
                dest = c0*coefs[0];
                for (Index i=1; i<ancestors.size(); ++i)
                {
                    Coord ci = x[ancestors[i]];
                    for (Index j=0; j<ci.size(); ++j)
                        ci[j] += helper::rnear(c0[j]-ci[j]);
                    dest += ci*coefs[i];
                }
            }
            else
            {
                dest = x[ancestors[0]]*coefs[0];
                for (std::size_t i=1; i<ancestors.size(); ++i)
                    dest += x[ancestors[i]]*coefs[i];
            }
        }
        // BUGFIX: remove link to the Data as it is now specific to this instance
        this->m_topologyData->setParent(nullptr);
    }

    void applyDestroyFunction(Index, Coord& )
    {
    }

protected:
    VisualModelImpl* obj;
    int algo;
};

template<class VecType>
void VisualModelImpl::addTopoHandler(topology::PointData<VecType>* data, int algo)
{
    data->createTopologicalEngine(m_topology, new VisualModelPointHandler<VecType>(this, data, algo), true);
    data->registerTopologicalData();
}

void VisualModelImpl::init()
{
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";


    if (m_vertPosIdx.getValue().size() > 0 && m_vertices2.getValue().empty())
    { // handle case where vertPosIdx was initialized through a loader
        m_vertices2.setValue(m_positions.getValue());
        if (m_positions.getParent())
        {
            m_positions.delInput(m_positions.getParent()); // remove any link to positions, as we need to recompute it
        }
        helper::WriteAccessor<Data<VecCoord>> vIn = m_positions;
        helper::ReadAccessor<Data<VecCoord>> vOut = m_vertices2;
        helper::ReadAccessor<Data<helper::vector<visual_index_type>>> vertPosIdx = m_vertPosIdx;
        std::size_t nbVIn = 0;
        for (std::size_t i = 0; i < vertPosIdx.size(); ++i)
        {
            if (vertPosIdx[i] >= nbVIn)
            {
                nbVIn = vertPosIdx[i]+1;
            }
        }
        vIn.resize(nbVIn);
        for (std::size_t i = 0; i < vertPosIdx.size(); ++i)
        {
            vIn[vertPosIdx[i]] = vOut[i];
        }
        m_topology = nullptr; // make sure we don't use the topology
    }

    load(fileMesh.getFullPath(), "", texturename.getFullPath());

    if (m_topology == nullptr || (m_positions.getValue().size()!=0 && m_positions.getValue().size() != m_topology->getNbPoints()))
    {
        // Fixes bug when neither an .obj file nor a topology is present in the VisualModel Node.
        // Thus nothing will be displayed.
        useTopology = false;
    }
    else
    {
        msg_info() << "Use topology " << m_topology->getName();
        // add the functions to handle topology changes.
        if (m_handleDynamicTopology.getValue())
        {
            //addTopoHandler(&m_positions);
            //addTopoHandler(&m_restPositions);
            //addTopoHandler(&m_vnormals);
            addTopoHandler(&m_vtexcoords,(m_fixMergedUVSeams.getValue()?1:0));
            addTopoHandler(&m_vtexcoords3, (m_fixMergedUVSeams.getValue() ? 1 : 0));
            //addTopoHandler(&m_vtangents);
            //addTopoHandler(&m_vbitangents);
        }
    }

    applyScale(m_scale.getValue()[0], m_scale.getValue()[1], m_scale.getValue()[2]);
    applyRotation(m_rotation.getValue()[0], m_rotation.getValue()[1], m_rotation.getValue()[2]);
    applyTranslation(m_translation.getValue()[0], m_translation.getValue()[1], m_translation.getValue()[2]);


    m_translation.setValue(Vec3Real());
    m_rotation.setValue(Vec3Real());
    m_scale.setValue(Vec3Real(1,1,1));

    VisualModel::init();
    updateVisual();
}

void VisualModelImpl::computeNormals()
{
    const VecCoord& vertices = getVertices();
    //const VecCoord& vertices = m_vertices2.getValue();
    if (vertices.empty() || (!m_updateNormals.getValue() && (m_vnormals.getValue()).size() == (vertices).size())) return;

    const VecVisualTriangle& triangles = m_triangles.getValue();
    const VecVisualQuad& quads = m_quads.getValue();
    const helper::vector<visual_index_type> &vertNormIdx = m_vertNormIdx.getValue();

    if (vertNormIdx.empty())
    {
        std::size_t nbn = vertices.size();

        VecDeriv& normals = *(m_vnormals.beginEdit());

        normals.resize(nbn);
        for (std::size_t i = 0; i < nbn; i++)
            normals[i].clear();

        for (std::size_t i = 0; i < triangles.size(); i++)
        {
            const Coord& v1 = vertices[triangles[i][0]];
            const Coord& v2 = vertices[triangles[i][1]];
            const Coord& v3 = vertices[triangles[i][2]];
            Coord n = cross(v2-v1, v3-v1);

            normals[triangles[i][0]] += n;
            normals[triangles[i][1]] += n;
            normals[triangles[i][2]] += n;
        }

        for (std::size_t i = 0; i < quads.size(); i++)
        {
            const Coord & v1 = vertices[quads[i][0]];
            const Coord & v2 = vertices[quads[i][1]];
            const Coord & v3 = vertices[quads[i][2]];
            const Coord & v4 = vertices[quads[i][3]];
            Coord n1 = cross(v2-v1, v4-v1);
            Coord n2 = cross(v3-v2, v1-v2);
            Coord n3 = cross(v4-v3, v2-v3);
            Coord n4 = cross(v1-v4, v3-v4);

            normals[quads[i][0]] += n1;
            normals[quads[i][1]] += n2;
            normals[quads[i][2]] += n3;
            normals[quads[i][3]] += n4;
        }

        for (std::size_t i = 0; i < normals.size(); i++)
            normals[i].normalize();

        m_vnormals.endEdit();
    }
    else
    {
        vector<Coord> normals;
        std::size_t nbn = 0;
        for (std::size_t i = 0; i < vertNormIdx.size(); i++)
        {
            if (vertNormIdx[i] >= nbn)
                nbn = vertNormIdx[i]+1;
        }

        normals.resize(nbn);
        for (std::size_t i = 0; i < nbn; i++)
            normals[i].clear();

        for (std::size_t i = 0; i < triangles.size() ; i++)
        {
            const Coord & v1 = vertices[triangles[i][0]];
            const Coord & v2 = vertices[triangles[i][1]];
            const Coord & v3 = vertices[triangles[i][2]];
            Coord n = cross(v2-v1, v3-v1);

            normals[vertNormIdx[triangles[i][0]]] += n;
            normals[vertNormIdx[triangles[i][1]]] += n;
            normals[vertNormIdx[triangles[i][2]]] += n;
        }

        for (std::size_t i = 0; i < quads.size() ; i++)
        {
            const Coord & v1 = vertices[quads[i][0]];
            const Coord & v2 = vertices[quads[i][1]];
            const Coord & v3 = vertices[quads[i][2]];
            const Coord & v4 = vertices[quads[i][3]];
            Coord n1 = cross(v2-v1, v4-v1);
            Coord n2 = cross(v3-v2, v1-v2);
            Coord n3 = cross(v4-v3, v2-v3);
            Coord n4 = cross(v1-v4, v3-v4);

            normals[vertNormIdx[quads[i][0]]] += n1;
            normals[vertNormIdx[quads[i][1]]] += n2;
            normals[vertNormIdx[quads[i][2]]] += n3;
            normals[vertNormIdx[quads[i][3]]] += n4;
        }

        for (std::size_t i = 0; i < normals.size(); i++)
        {
            normals[i].normalize();
        }

        VecDeriv& vnormals = *(m_vnormals.beginEdit());
        vnormals.resize(vertices.size());
        for (std::size_t i = 0; i < vertices.size(); i++)
        {
            vnormals[i] = normals[vertNormIdx[i]];
        }
        m_vnormals.endEdit();
    }
}

VisualModelImpl::Coord VisualModelImpl::computeTangent(const Coord &v1, const Coord &v2, const Coord &v3,
                                                       const TexCoord &t1, const TexCoord &t2, const TexCoord &t3)
{
    Coord v = (v2 - v1) * (t3.y() - t1.y()) + (v3 - v1) * (t1.y() - t2.y());
    v.normalize();
    return v;
}

VisualModelImpl::Coord VisualModelImpl::computeBitangent(const Coord &v1, const Coord &v2, const Coord &v3,
                                                         const TexCoord &t1, const TexCoord &t2, const TexCoord &t3)
{
    Coord v = (v2 - v1) * (t3.x() - t1.x()) + (v3 - v1) * (t1.x() - t2.x());
    v.normalize();
    return v;
}

void VisualModelImpl::computeTangents()
{
    if (!m_computeTangents.getValue() || !m_vtexcoords.getValue().size()) return;

    const VecVisualTriangle& triangles = m_triangles.getValue();
    const VecVisualQuad& quads = m_quads.getValue();
    const VecCoord& vertices = getVertices();
    const VecTexCoord& texcoords = m_vtexcoords.getValue();
    VecCoord& normals = *(m_vnormals.beginEdit());
    VecCoord& tangents = *(m_vtangents.beginEdit());
    VecCoord& bitangents = *(m_vbitangents.beginEdit());

    tangents.resize(vertices.size());
    bitangents.resize(vertices.size());

    for (unsigned i = 0; i < vertices.size(); i++)
    {
        tangents[i].clear();
        bitangents[i].clear();
    }
    const bool fixMergedUVSeams = m_fixMergedUVSeams.getValue();
    for (std::size_t i = 0; i < triangles.size() ; i++)
    {
        const Coord v1 = vertices[triangles[i][0]];
        const Coord v2 = vertices[triangles[i][1]];
        const Coord v3 = vertices[triangles[i][2]];
        TexCoord t1 = texcoords[triangles[i][0]];
        TexCoord t2 = texcoords[triangles[i][1]];
        TexCoord t3 = texcoords[triangles[i][2]];
        if (fixMergedUVSeams)
        {
            for (Size j=0; j<t1.size(); ++j)
            {
                t2[j] += helper::rnear(t1[j]-t2[j]);
                t3[j] += helper::rnear(t1[j]-t3[j]);
            }
        }
        Coord t = computeTangent(v1, v2, v3, t1, t2, t3);
        Coord b = computeBitangent(v1, v2, v3, t1, t2, t3);

        tangents[triangles[i][0]] += t;
        tangents[triangles[i][1]] += t;
        tangents[triangles[i][2]] += t;
        bitangents[triangles[i][0]] += b;
        bitangents[triangles[i][1]] += b;
        bitangents[triangles[i][2]] += b;
    }

    for (std::size_t i = 0; i < quads.size() ; i++)
    {
        const Coord & v1 = vertices[quads[i][0]];
        const Coord & v2 = vertices[quads[i][1]];
        const Coord & v3 = vertices[quads[i][2]];
        const Coord & v4 = vertices[quads[i][3]];
        const TexCoord t1 = texcoords[quads[i][0]];
        const TexCoord t2 = texcoords[quads[i][1]];
        const TexCoord t3 = texcoords[quads[i][2]];
        const TexCoord t4 = texcoords[quads[i][3]];

        // Too many options how to split a quad into two triangles...
        Coord t123 = computeTangent  (v1, v2, v3, t1, t2, t3);
        Coord b123 = computeBitangent(v1, v2, v2, t1, t2, t3);

        Coord t234 = computeTangent  (v2, v3, v4, t2, t3, t4);
        Coord b234 = computeBitangent(v2, v3, v4, t2, t3, t4);

        Coord t341 = computeTangent  (v3, v4, v1, t3, t4, t1);
        Coord b341 = computeBitangent(v3, v4, v1, t3, t4, t1);

        Coord t412 = computeTangent  (v4, v1, v2, t4, t1, t2);
        Coord b412 = computeBitangent(v4, v1, v2, t4, t1, t2);

        tangents  [quads[i][0]] += t123        + t341 + t412;
        bitangents[quads[i][0]] += b123        + b341 + b412;
        tangents  [quads[i][1]] += t123 + t234        + t412;
        bitangents[quads[i][1]] += b123 + b234        + b412;
        tangents  [quads[i][2]] += t123 + t234 + t341;
        bitangents[quads[i][2]] += b123 + b234 + b341;
        tangents  [quads[i][3]] +=        t234 + t341 + t412;
        bitangents[quads[i][3]] +=        b234 + b341 + b412;
    }
    for (std::size_t i = 0; i < vertices.size(); i++)
    {
        Coord n = normals[i];
        Coord& t = tangents[i];
        Coord& b = bitangents[i];

        b = sofa::defaulttype::cross(n, t.normalized());
        t = sofa::defaulttype::cross(b, n);
    }
    m_vtangents.endEdit();
    m_vbitangents.endEdit();
}

void VisualModelImpl::computeBBox(const core::ExecParams*, bool)
{
    const VecCoord& x = getVertices(); //m_vertices.getValue();

    SReal minBBox[3] = {std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max()};
    SReal maxBBox[3] = {-std::numeric_limits<Real>::max(),-std::numeric_limits<Real>::max(),-std::numeric_limits<Real>::max()};
    for (std::size_t i = 0; i < x.size(); i++)
    {
        const Coord& p = x[i];
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    this->f_bbox.setValue(sofa::defaulttype::TBoundingBox<SReal>(minBBox,maxBBox));
}


void VisualModelImpl::computeUVSphereProjection()
{
    sofa::core::visual::VisualParams* vparams = sofa::core::visual::VisualParams::defaultInstance();
    this->computeBBox(vparams);

    Vector3 center = (this->f_bbox.getValue().minBBox() + this->f_bbox.getValue().maxBBox())*0.5f;
    
    // Map mesh vertices to sphere
    // transform cart to spherical coordinates (r, theta, phi) and sphere to cart back with radius = 1
    const VecCoord& coords = getVertices();

    std::size_t nbrV = coords.size();
    VecCoord m_sphereV;
    m_sphereV.resize(nbrV);

    VecTexCoord& vtexcoords = *(m_vtexcoords.beginEdit());
    vtexcoords.resize(nbrV);

    for (std::size_t i = 0; i < nbrV; ++i)
    {
        Coord Vcentered = coords[i] - center;
        SReal r = sqrt(Vcentered[0] * Vcentered[0] + Vcentered[1] * Vcentered[1] + Vcentered[2] * Vcentered[2]);
        SReal theta = acos(Vcentered[2] / r);
        SReal phi = atan2(Vcentered[1], Vcentered[0]);

        r = 1.0;
        m_sphereV[i][0] = r * sin(theta)*cos(phi) + center[0];
        m_sphereV[i][1] = r * sin(theta)*sin(phi) + center[1];
        m_sphereV[i][2] = r * cos(theta) + center[2];

        Coord pos = m_sphereV[i] - center;
        pos.normalize();
        vtexcoords[i][0] = float(0.5 + atan2(pos[1], pos[0]) / (2 * R_PI));
        vtexcoords[i][1] = float(0.5 - asin(pos[2]) / R_PI);
    }

    m_vtexcoords.endEdit();
}

void VisualModelImpl::flipFaces()
{
    VecDeriv& vnormals = *(m_vnormals.beginEdit());
    VecVisualEdge& edges = *(m_edges.beginEdit());
    VecVisualTriangle& triangles = *(m_triangles.beginEdit());
    VecVisualQuad& quads = *(m_quads.beginEdit());

    for (std::size_t i = 0; i < edges.size() ; i++)
    {
        Index temp = edges[i][1];
        edges[i][1] = visual_index_type(edges[i][0]);
        edges[i][0] = visual_index_type(temp);
    }

    for (std::size_t i = 0; i < triangles.size() ; i++)
    {
        Index temp = triangles[i][1];
        triangles[i][1] = visual_index_type(triangles[i][2]);
        triangles[i][2] = visual_index_type(temp);
    }

    for (std::size_t i = 0; i < quads.size() ; i++)
    {
        Index temp = quads[i][1];
        quads[i][1] = visual_index_type(quads[i][3]);
        quads[i][3] = visual_index_type(temp);
    }

    for (std::size_t i = 0; i < vnormals.size(); i++)
    {
        vnormals[i] = -vnormals[i];
    }

    m_vnormals.endEdit();
    m_edges.endEdit();
    m_triangles.endEdit();
    m_quads.endEdit();
}

void VisualModelImpl::setColor(float r, float g, float b, float a)
{
    Material M = material.getValue();
    M.setColor(r,g,b,a);
    material.setValue(M);
}

void VisualModelImpl::setColor(std::string color)
{
    if (color.empty())
        return;

    RGBAColor theColor;
    if( !RGBAColor::read(color, theColor) )
    {
        msg_info(this) << "Unable to decode color '"<< color <<"'." ;
    }
    setColor(theColor.r(),theColor.g(),theColor.b(),theColor.a());
}


void VisualModelImpl::updateVisual()
{
    /*
        static unsigned int last = 0;
        if (m_vtexcoords.getValue().size() != last)
        {
            std::cout << m_vtexcoords.getValue().size() << std::endl;
            last = m_vtexcoords.getValue().size();
        }
    */
    if (modified && (!getVertices().empty() || useTopology))
    {
        if (useTopology)
        {
            sofa::helper::ScopedAdvancedTimer timer("VisualModelImpl::updateMesh");
            /** HD : build also a Ogl description from main Topology. But it needs to be build only once since the topology update
            is taken care of by the handleTopologyChange() routine */

            sofa::core::topology::TopologyModifier* topoMod;
            this->getContext()->get(topoMod);

            if (topoMod)
            {
                useTopology = false; // dynamic topology
                computeMesh();
            }
            else if (topoMod == nullptr && (m_topology->getRevision() != lastMeshRev))  // static topology
            {
                computeMesh();
            }
        }
        sofa::helper::AdvancedTimer::stepBegin("VisualModelImpl::computePositions");
        computePositions();
        sofa::helper::AdvancedTimer::stepEnd("VisualModelImpl::computePositions");

        sofa::helper::AdvancedTimer::stepBegin("VisualModelImpl::updateBuffers");
        updateBuffers();
        sofa::helper::AdvancedTimer::stepEnd("VisualModelImpl::updateBuffers");

        sofa::helper::AdvancedTimer::stepBegin("VisualModelImpl::computeNormals");
        computeNormals();
        sofa::helper::AdvancedTimer::stepEnd("VisualModelImpl::computeNormals");
        
        if (m_updateTangents.getValue())
        {
            sofa::helper::AdvancedTimer::stepBegin("VisualModelImpl::computeTangents");
            computeTangents();
            sofa::helper::AdvancedTimer::stepEnd("VisualModelImpl::computeTangents");
        }
        modified = false;

        if (m_vtexcoords.getValue().size() == 0)
            computeUVSphereProjection();

    }

    m_positions.updateIfDirty();
    m_vertices2.updateIfDirty();
    m_vnormals.updateIfDirty();
    //m_vtexcoords.updateIfDirty();
    m_vtangents.updateIfDirty();
    m_vbitangents.updateIfDirty();
    m_edges.updateIfDirty();
    m_triangles.updateIfDirty();
    m_quads.updateIfDirty();

}

void VisualModelImpl::computeTextureCoords3()
{
    VecTexCoord3 & my_texCoords = *(m_vtexcoords3.beginEdit());
    const VecCoord & coords = this->m_positions.getValue();
    my_texCoords.clear();
    my_texCoords.resize(coords.size());
    
        SReal Cmin[3]; Cmin[0] = 100000, Cmin[1] = 100000, Cmin[2] = 100000;
    SReal Cmax[3]; Cmax[0] = -100000, Cmax[1] = -100000, Cmax[2] = -100000;
    
        
        	// creating BB
        for (unsigned int i = 0; i < coords.size(); ++i)
         {
        const Coord & p0 = coords[i];
        for (unsigned int j = 0; j < 3; ++j)
             {
            if (p0[j] < Cmin[j]) Cmin[j] = p0[j];
            if (p0[j] > Cmax[j]) Cmax[j] = p0[j];
            }
         }
    
        unsigned int axe1 = 0, axe2 = 1, axe3 = 2;
    SReal Uscale = 1 / (SReal)(Cmax[axe1] - Cmin[axe1]);
    SReal Vscale = 1 / (SReal)(Cmax[axe2] - Cmin[axe2]);
    SReal Wscale = 1 / (SReal)(Cmax[axe3] - Cmin[axe3]);
    
        for (unsigned int i = 0; i < coords.size(); ++i)
         {
        const Coord & p0 = coords[i];
        TexCoord3 & textC = my_texCoords[i];
        SReal x = (p0[axe1] - Cmin[axe1]) * Uscale;
        SReal y = (p0[axe2] - Cmin[axe2]) * Vscale;
        SReal z = (p0[axe3] - Cmin[axe3]) * Wscale;
        
            textC[0] = x;
        textC[1] = y;
        textC[2] = z;
        
            }
    
        m_vtexcoords3.endEdit();
    }

void VisualModelImpl::computePositions()
{
    const helper::vector<visual_index_type> &vertPosIdx = m_vertPosIdx.getValue();

    if (!vertPosIdx.empty())
    {
        // Need to transfer positions
        VecCoord& vertices = *(m_vertices2.beginEdit());
        const VecCoord& positions = this->m_positions.getValue();

        for (std::size_t i=0 ; i < vertices.size(); ++i)
            vertices[i] = positions[vertPosIdx[i]];

        m_vertices2.endEdit();
    }
}

void VisualModelImpl::computeMesh()
{
    using sofa::component::topology::SparseGridTopology;
    using sofa::core::behavior::BaseMechanicalState;

//	sofa::helper::vector<Coord> bezierControlPointsArray;

    if ((m_positions.getValue()).empty() && (m_vertices2.getValue()).empty())
    {
        VecCoord& vertices = *(m_positions.beginEdit());

        if (m_topology->hasPos())
        {
            if (SparseGridTopology *spTopo = dynamic_cast< SparseGridTopology *>(m_topology))
            {
                sofa::helper::io::Mesh m;
                spTopo->getMesh(m);
                setMesh(m, !texturename.getValue().empty());
                dmsg_info() << " getting marching cube mesh from topology, "
                            << m.getVertices().size() << " points, "
                            << m.getFacets().size()  << " triangles." ;

                useTopology = false; //visual model needs to be created only once at initial time
                return;
            }

            dmsg_info() << " copying " << m_topology->getNbPoints() << " points from topology." ;

            vertices.resize(m_topology->getNbPoints());

            for (std::size_t i=0; i<vertices.size(); i++)
            {
                vertices[i][0] = SReal(m_topology->getPX(Size(i)));
                vertices[i][1] = SReal(m_topology->getPY(Size(i)));
                vertices[i][2] = SReal(m_topology->getPZ(Size(i)));
            }

        }
        else
        {
            BaseMechanicalState* mstate = m_topology->getContext()->getMechanicalState();

            if (mstate)
            {
                dmsg_info() << " copying " << mstate->getSize() << " points from mechanical state" ;

                vertices.resize(mstate->getSize());

                for (std::size_t i=0; i<vertices.size(); i++)
                {
                    vertices[i][0] = (Real)mstate->getPX(Size(i));
                    vertices[i][1] = (Real)mstate->getPY(Size(i));
                    vertices[i][2] = (Real)mstate->getPZ(Size(i));
                }

            }
        }
        m_positions.endEdit();
    }
    if (m_genTex3d.getValue()) {
        this->computeTextureCoords3();
        this->applyUVWTransformation();
        
    }
    lastMeshRev = m_topology->getRevision();

    const vector< Triangle >& inputTriangles = m_topology->getTriangles();


    dmsg_info() << " copying " << inputTriangles.size() << " triangles from topology" ;

    VecVisualTriangle& triangles = *(m_triangles.beginEdit());
    triangles.resize(inputTriangles.size());

    for (std::size_t i=0; i<triangles.size(); ++i)
    {
        triangles[i][0] = visual_index_type(inputTriangles[i][0]);
        triangles[i][1] = visual_index_type(inputTriangles[i][1]);
        triangles[i][2] = visual_index_type(inputTriangles[i][2]);
    }
    m_triangles.endEdit();


    const vector< BaseMeshTopology::Quad >& inputQuads = m_topology->getQuads();

    dmsg_info() << " copying " << inputQuads.size()<< " quads from topology." ;

    VecVisualQuad& quads = *(m_quads.beginEdit());
    quads.resize(inputQuads.size());

    for (std::size_t i=0; i<quads.size(); ++i)
    {
        quads[i][0] = visual_index_type(inputQuads[i][0]);
        quads[i][1] = visual_index_type(inputQuads[i][1]);
        quads[i][2] = visual_index_type(inputQuads[i][2]);
        quads[i][3] = visual_index_type(inputQuads[i][3]);
    }
    m_quads.endEdit();
}

void VisualModelImpl::handleTopologyChange()
{
    if (!m_topology) return;

    bool debug_mode = false;

    VecVisualTriangle& triangles = *(m_triangles.beginEdit());
    VecVisualQuad& quads = *(m_quads.beginEdit());
    m_positions.beginEdit();

    std::list<const TopologyChange *>::const_iterator itBegin=m_topology->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=m_topology->endChange();

    while( itBegin != itEnd )
    {
        core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

        switch( changeType )
        {
        case core::topology::ENDING_EVENT:
        {
            updateVisual();
            break;
        }

        case core::topology::TRIANGLESADDED:
        {
            if (!groups.getValue().empty())
            {
                groups.beginEdit()->clear();
                groups.endEdit();
            }

            const sofa::core::topology::TrianglesAdded *ta = static_cast< const sofa::core::topology::TrianglesAdded * >( *itBegin );

            VisualTriangle t;
            const std::size_t nbAddedTriangles = ta->getNbAddedTriangles();
            const std::size_t nbTririangles = triangles.size();

            triangles.resize(nbTririangles + nbAddedTriangles);

            for (std::size_t i = 0; i < nbAddedTriangles; ++i)
            {
                t[0] = visual_index_type(ta->triangleArray[i][0]);
                t[1] = visual_index_type(ta->triangleArray[i][1]);
                t[2] = visual_index_type(ta->triangleArray[i][2]);
                triangles[nbTririangles + i] = t;
            }

            break;
        }

        case core::topology::QUADSADDED:
        {
            if (!groups.getValue().empty())
            {
                groups.beginEdit()->clear();
                groups.endEdit();
            }

            const sofa::core::topology::QuadsAdded *qa = static_cast< const sofa::core::topology::QuadsAdded * >( *itBegin );

            VisualQuad q;
            const std::size_t nbAddedQuads = qa->getNbAddedQuads();
            const std::size_t nbQuaduads = quads.size();

            quads.resize(nbQuaduads + nbAddedQuads);

            for (std::size_t i = 0; i < nbAddedQuads; ++i)
            {
                const auto& rQuad = qa->getQuad(Size(i));

                quads[nbQuaduads + i][0] = visual_index_type(rQuad[0]);
                quads[nbQuaduads + i][1] = visual_index_type(rQuad[1]);
                quads[nbQuaduads + i][2] = visual_index_type(rQuad[2]);
                quads[nbQuaduads + i][3] = visual_index_type(rQuad[3]);
            }

            break;
        }

        case core::topology::TRIANGLESREMOVED:
        {
            if (!groups.getValue().empty())
            {
                groups.beginEdit()->clear();
                groups.endEdit();
            }

            std::size_t last;

            last = m_topology->getNbTriangles() - 1;

            const auto &tab = ( static_cast< const sofa::core::topology::TrianglesRemoved *>( *itBegin ) )->getArray();

            VisualTriangle tmp;

            for (std::size_t i = 0; i <tab.size(); ++i)
            {
                visual_index_type ind_k = visual_index_type(tab[i]);

                tmp = triangles[ind_k];
                triangles[ind_k] = triangles[last];
                triangles[last] = tmp;

                std::size_t ind_last = triangles.size() - 1;

                if(last != ind_last)
                {
                    tmp = triangles[last];
                    triangles[last] = triangles[ind_last];
                    triangles[ind_last] = tmp;
                }

                triangles.resize( triangles.size() - 1 );

                --last;
            }

            break;
        }

        case core::topology::QUADSREMOVED:
        {
            if (!groups.getValue().empty())
            {
                groups.beginEdit()->clear();
                groups.endEdit();
            }

            std::size_t last;

            last = m_topology->getNbQuads() - 1;

            const auto &tab = ( static_cast< const sofa::core::topology::QuadsRemoved *>( *itBegin ) )->getArray();

            VisualQuad tmp;

            for (std::size_t i = 0; i <tab.size(); ++i)
            {
                visual_index_type ind_k = visual_index_type(tab[i]);

                tmp = quads[ind_k];
                quads[ind_k] = quads[last];
                quads[last] = tmp;

                std::size_t ind_last = quads.size() - 1;

                if(last != ind_last)
                {
                    tmp = quads[last];
                    quads[last] = quads[ind_last];
                    quads[ind_last] = tmp;
                }

                quads.resize( quads.size() - 1 );

                --last;
            }

            break;
        }

        case core::topology::POINTSREMOVED:
        {
            if (m_topology->getNbTriangles()>0)
            {
                auto last = m_topology->getNbPoints() -1;

                Size i,j;

                const auto& tab = ( static_cast< const sofa::core::topology::PointsRemoved * >( *itBegin ) )->getArray();

                sofa::helper::vector<Index> lastIndexVec;

                for(Size i_init = 0; i_init < tab.size(); ++i_init)
                {
                    lastIndexVec.push_back(last - i_init);
                }

                for ( i = 0; i < tab.size(); ++i)
                {
                    std::size_t i_next = i;
                    bool is_reached = false;

                    while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
                    {
                        i_next += 1 ;
                        is_reached = is_reached || (lastIndexVec[i_next] == tab[i]);
                    }

                    if(is_reached)
                    {
                        lastIndexVec[i_next] = lastIndexVec[i];
                    }

                    const auto &shell= m_topology->getTrianglesAroundVertex(lastIndexVec[i]);
                    for (j=0; j<shell.size(); ++j)
                    {
                        auto ind_j = shell[j];

                        if ((unsigned)triangles[ind_j][0]==last)
                            triangles[ind_j][0]= visual_index_type(tab[i]);
                        else if ((unsigned)triangles[ind_j][1]==last)
                            triangles[ind_j][1]= visual_index_type(tab[i]);
                        else if ((unsigned)triangles[ind_j][2]==last)
                            triangles[ind_j][2]= visual_index_type(tab[i]);
                    }

                    if (debug_mode)
                    {
                        for (std::size_t j_loc=0; j_loc<triangles.size(); ++j_loc)
                        {
                            bool is_forgotten = false;
                            if ((unsigned)triangles[j_loc][0]==last)
                            {
                                triangles[j_loc][0]= visual_index_type(tab[i]);
                                is_forgotten=true;
                            }
                            else
                            {
                                if ((unsigned)triangles[j_loc][1]==last)
                                {
                                    triangles[j_loc][1]= visual_index_type(tab[i]);
                                    is_forgotten=true;
                                }
                                else
                                {
                                    if ((unsigned)triangles[j_loc][2]==last)
                                    {
                                        triangles[j_loc][2]= visual_index_type(tab[i]);
                                        is_forgotten=true;
                                    }
                                }
                            }

                            if(is_forgotten)
                            {
                                Index ind_forgotten = Size(j_loc);

                                bool is_in_shell = false;
                                for (std::size_t j_glob=0; j_glob<shell.size(); ++j_glob)
                                {
                                    is_in_shell = is_in_shell || (shell[j_glob] == ind_forgotten);
                                }

                                if(!is_in_shell)
                                {
                                    msg_info() << "INFO_print : Vis - triangle is forgotten in SHELL !!! global indices (point, triangle) = ( "  << last << " , " << ind_forgotten  << " )";

                                    if(ind_forgotten<m_topology->getNbTriangles())
                                    {
                                        const auto& t_forgotten = m_topology->getTriangle(ind_forgotten);
                                        msg_info() << "Vis - last = " << last << msgendl
                                                   << "Vis - lastIndexVec[i] = " << lastIndexVec[i] << msgendl
                                                   << "Vis - tab.size() = " << tab.size() << " , tab = " << tab << msgendl
                                                   << "Vis - t_local rectified = " << triangles[j_loc] << msgendl
                                                   << "Vis - t_global = " << t_forgotten;
                                    }
                                }
                            }
                        }
                    }

                    --last;
                }
            }
            else if (m_topology->getNbQuads()>0)
            {
                sofa::Index last = m_topology->getNbPoints() -1;

                Index i,j;

                const auto& tab = ( static_cast< const sofa::core::topology::PointsRemoved * >( *itBegin ) )->getArray();

                sofa::helper::vector<Index> lastIndexVec;
                for(Index i_init = 0; i_init < tab.size(); ++i_init)
                {
                    lastIndexVec.push_back(last - i_init);
                }

                for ( i = 0; i < tab.size(); ++i)
                {
                    Index i_next = i;
                    bool is_reached = false;
                    while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
                    {
                        i_next += 1 ;
                        is_reached = is_reached || (lastIndexVec[i_next] == tab[i]);
                    }

                    if(is_reached)
                    {
                        lastIndexVec[i_next] = lastIndexVec[i];
                    }

                    const auto &shell= m_topology->getQuadsAroundVertex(lastIndexVec[i]);
                    for (j=0; j<shell.size(); ++j)
                    {
                        Index ind_j = shell[j];

                        if (quads[ind_j][0]==last)
                            quads[ind_j][0]=visual_index_type(tab[i]);
                        else if (quads[ind_j][1]==last)
                            quads[ind_j][1]= visual_index_type(tab[i]);
                        else if (quads[ind_j][2]==last)
                            quads[ind_j][2]= visual_index_type(tab[i]);
                        else if (quads[ind_j][3]==last)
                            quads[ind_j][3]= visual_index_type(tab[i]);
                    }

                    --last;
                }
            }

            break;
        }

        case core::topology::POINTSRENUMBERING:
        {
            if (m_topology->getNbTriangles()>0)
            {
                const auto& tab = ( static_cast< const sofa::core::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                for (std::size_t i = 0; i < triangles.size(); ++i)
                {
                    triangles[i][0]  = visual_index_type(tab[triangles[i][0]]);
                    triangles[i][1]  = visual_index_type(tab[triangles[i][1]]);
                    triangles[i][2]  = visual_index_type(tab[triangles[i][2]]);
                }

            }
            else if (m_topology->getNbQuads()>0)
            {
                const auto& tab = ( static_cast< const sofa::core::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                for (std::size_t i = 0; i < quads.size(); ++i)
                {
                    quads[i][0]  = visual_index_type(tab[quads[i][0]]);
                    quads[i][1]  = visual_index_type(tab[quads[i][1]]);
                    quads[i][2]  = visual_index_type(tab[quads[i][2]]);
                    quads[i][3]  = visual_index_type(tab[quads[i][3]]);
                }
            }

            break;
        }

        case core::topology::POINTSMOVED:
        {
            updateVisual();
            break;
        }

        case core::topology::POINTSADDED:
        {
#if 0
            using sofa::core::behavior::BaseMechanicalState;
            BaseMechanicalState* mstate;
            //const Index nbPoints = ( static_cast< const sofa::component::topology::PointsAdded * >( *itBegin ) )->getNbAddedVertices();
            m_topology->getContext()->get(mstate);
            /* fjourdes:
            ! THIS IS OBVIOUSLY NOT THE APPROPRIATE WAY TO DO IT !
            However : VisualModelImpl stores in two separates data the vertices
              - Data position in inherited Vec3State
              - Data vertices
            I don t know what is the purpose of the Data vertices (except at the init maybe ? )
            When doing topological operations on a graph like
            (removal points triangles / add of points triangles for instance)
            + Hexas
            ...
            + Triangles
            + - MechObj Triangles
            + - TriangleSetTopologyContainer Container
            + - Hexa2TriangleTopologycalMapping
            + + VisualModel
            + + - OglModel visual
            + + - IdentityMapping

            The IdentityMapping reflects the changes in topology by updating the Data position of the OglModel
            knowing the Data position of the MechObj named Triangles.
            However the Data vertices which is used to compute the normals is not updated, and the next computeNormals will
            fail. BTW this is odd that normals are computed using Data vertices since Data normals it belongs to Vec3State
            (like Data position) ...
            So my question is how the changes in the Data position of and OglModel are reflected to its Data vertices?
            It must be done somewhere since ultimately visual models are drawn correctly by OglModel::internalDraw !
            */

            if (mstate)
            {

                dmsg_info() << " changing size.  " << msgendl
                            << " oldsize    " << this->getSize() << msgendl
                            << " copying " << mstate->getSize() << " points from mechanical state.";

                vertices.resize(mstate->getSize());

                for (Index i=0; i<vertices.size(); i++)
                {
                    vertices[i][0] = (Real)mstate->getPX(i);
                    vertices[i][1] = (Real)mstate->getPY(i);
                    vertices[i][2] = (Real)mstate->getPZ(i);
                }
            }
            updateVisual();
#endif
            break;
        }

        default:
            // Ignore events that are not Triangle  related.
            break;
        }; // switch( changeType )

        ++itBegin;
    } // while( changeIt != last; )

    m_triangles.endEdit();
    m_quads.endEdit();
    m_positions.endEdit();
}

void VisualModelImpl::initVisual()
{
}

void VisualModelImpl::exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, Index& vindex, Index& nindex, Index& tindex, int& count)
{
    *out << "g "<<name<<"\n";

    if (mtl != nullptr) // && !material.name.empty())
    {
        std::string name; // = material.name;
        if (name.empty())
        {
            std::ostringstream o; o << "mat" << count;
            name = o.str();
        }
        *mtl << "newmtl "<<name<<"\n";
        *mtl << "illum 4\n";
        if (material.getValue().useAmbient)
            *mtl << "Ka "<<material.getValue().ambient[0]<<' '<<material.getValue().ambient[1]<<' '<<material.getValue().ambient[2]<<"\n";
        if (material.getValue().useDiffuse)
            *mtl << "Kd "<<material.getValue().diffuse[0]<<' '<<material.getValue().diffuse[1]<<' '<<material.getValue().diffuse[2]<<"\n";
        *mtl << "Tf 1.00 1.00 1.00\n";
        *mtl << "Ni 1.00\n";
        if (material.getValue().useSpecular)
            *mtl << "Ks "<<material.getValue().specular[0]<<' '<<material.getValue().specular[1]<<' '<<material.getValue().specular[2]<<"\n";
        if (material.getValue().useShininess)
            *mtl << "Ns "<<material.getValue().shininess<<"\n";
        if (material.getValue().useDiffuse && material.getValue().diffuse[3]<1.0)
            *mtl << "Tf "<<material.getValue().diffuse[3]<<' '<<material.getValue().diffuse[3]<<' '<<material.getValue().diffuse[3]<<"\n";

        *out << "usemtl "<<name<<'\n';
    }

    const VecCoord& x = m_positions.getValue();
    const VecDeriv& vnormals = m_vnormals.getValue();
    const VecTexCoord& vtexcoords = m_vtexcoords.getValue();
    const VecVisualEdge& edges = m_edges.getValue();
    const VecVisualTriangle& triangles = m_triangles.getValue();
    const VecVisualQuad& quads = m_quads.getValue();

    const helper::vector<visual_index_type> &vertPosIdx = m_vertPosIdx.getValue();
    const helper::vector<visual_index_type> &vertNormIdx = m_vertNormIdx.getValue();

    auto nbv = Size(x.size());

    for (std::size_t i=0; i<nbv; i++)
    {
        *out << "v "<< std::fixed << x[i][0]<<' '<< std::fixed <<x[i][1]<<' '<< std::fixed <<x[i][2]<<'\n';
    }

    Size nbn = 0;

    if (vertNormIdx.empty())
    {
        nbn = sofa::Size(vnormals.size());
        for (sofa::Index i=0; i<nbn; i++)
        {
            *out << "vn "<< std::fixed << vnormals[i][0]<<' '<< std::fixed <<vnormals[i][1]<<' '<< std::fixed <<vnormals[i][2]<<'\n';
        }
    }
    else
    {
        for (sofa::Index i = 0; i < vertNormIdx.size(); i++)
        {
            if (vertNormIdx[i] >= nbn)
                nbn = vertNormIdx[i]+1;
        }
        vector<Index> normVertIdx(nbn);
        for (sofa::Index i = 0; i < vertNormIdx.size(); i++)
        {
            normVertIdx[vertNormIdx[i]]=i;
        }
        for (sofa::Index i = 0; i < nbn; i++)
        {
            Index j = normVertIdx[i];
            *out << "vn "<< std::fixed << vnormals[j][0]<<' '<< std::fixed <<vnormals[j][1]<<' '<< std::fixed <<vnormals[j][2]<<'\n';
        }
    }

    Size nbt = 0;
    if (!vtexcoords.empty())
    {
        nbt = sofa::Size(vtexcoords.size());
        for (std::size_t i=0; i<nbt; i++)
        {
            *out << "vt "<< std::fixed << vtexcoords[i][0]<<' '<< std::fixed <<vtexcoords[i][1]<<'\n';
        }
    }

    for (std::size_t i = 0; i < edges.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<2; j++)
        {
            Index i0 = edges[i][j];
            Index i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            Index i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    for (std::size_t i = 0; i < triangles.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<3; j++)
        {
            Index i0 = triangles[i][j];
            Index i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            Index i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    for (std::size_t i = 0; i < quads.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<4; j++)
        {
            Index i0 = quads[i][j];
            Index i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            Index i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    *out << sendl;
    vindex+=nbv;
    nindex+=nbn;
    tindex+=nbt;
}

template class SOFA_SOFABASEVISUAL_API VisualModelPointHandler< VisualModelImpl::VecCoord>;
template class SOFA_SOFABASEVISUAL_API VisualModelPointHandler< VisualModelImpl::VecTexCoord>;

} // namespace sofa::component::visualmodel

namespace sofa::component::topology
{

template class PointData< sofa::defaulttype::Vec3fTypes::VecCoord >;
template class PointData< sofa::defaulttype::Vec2fTypes::VecCoord >;

} // namespace sofa::component::topology
