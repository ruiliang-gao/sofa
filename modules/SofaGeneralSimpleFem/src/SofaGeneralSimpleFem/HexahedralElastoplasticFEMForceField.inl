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
#pragma once
#include "HexahedralElastoplasticFEMForceField.h"
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/helper/decompose.h>
#include <cassert>
#include <iostream>
#include <set>

#include <SofaBaseTopology/TopologyData.inl>



namespace sofa::component::forcefield
{

template< class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::HexHandler::applyCreateFunction(Index hexahedronIndex,
    HexahedronInformation&,
        const core::topology::BaseMeshTopology::Hexahedron &,
        const sofa::type::vector<Index> &,
        const sofa::type::vector<double> &)
{
    if (ff)
    {
        switch(ff->method)
        {
        case LARGE :
            ff->initLarge(hexahedronIndex);

            break;
        case POLAR :
            ff->initPolar(hexahedronIndex);
            break;
        }
    }
}

template <class DataTypes>
HexahedralElastoplasticFEMForceField<DataTypes>::HexahedralElastoplasticFEMForceField()
    : //f_method(initData(&f_method, std::string("large"), "method", "\"large\" or \"polar\" displacements"))
    //, f_poissonRatio(initData(&f_poissonRatio, (Real)0.45f, "poissonRatio", ""))
    //, f_youngModulus(initData(&f_youngModulus, (Real)5000, "youngModulus", ""))
      f_plasticMaxThreshold(initData(&f_plasticMaxThreshold, (Real)0.f, "plasticMaxThreshold", "Plastic Max Threshold"))
    , f_plasticYieldThreshold(initData(&f_plasticYieldThreshold, (Real)0.f, "plasticYieldThreshold", "Plastic Yield Threshold"))
    , f_plasticMaxRotationThreshold(initData(&f_plasticMaxRotationThreshold, (Real)1.0f, "plasticMaxRotationThreshold", "Plastic Max Rotation Threshold"))
    , f_plasticCreep(initData(&f_plasticCreep, (Real)0.1f, "plasticCreep", "Plastic Creep Factor * dt [0,1]. Warning this factor depends on dt."))
    , f_plasticRotationCreep(initData(&f_plasticRotationCreep, (Real)0.1f, "plasticRotationCreep", "Plastic Creep Factor * dt [0,1]. Warning this factor depends on dt."))
    , f_plasticRotationYieldThreshold(initData(&f_plasticRotationYieldThreshold, (Real)0.f, "plasticRotationYieldThreshold", "Plastic Rotation Yield Threshold"))
    , f_hardeningParam(initData(&f_hardeningParam, (Real)0.1f, "hardeningParam", "Material work hardening parameter"))
    , f_useHigherOrderPlasticity(initData(&f_useHigherOrderPlasticity, false, "useHigherOrderPlasticity", "Use higher order scheme: mixed vertex + center deformation for plasticity if true"))
    , f_preserveElementVolume(initData(&f_preserveElementVolume, false, "preserveElementVolume", "Preserve element volume under plasticity deformation"))
    , f_updateElementStiffness(initData(&f_updateElementStiffness, false, "updateElementStiffness", "Element stiffness matrix will be updated when necessary"))
    , hexahedronInfo(initData(&hexahedronInfo, "hexahedronInfo", "Internal hexahedron data"))
    /*for debugging*/, f_debugPlasticMethod(initData(&f_debugPlasticMethod, (int)2, "debugPlasticMethod", "debug methods"))
    , debugData(initData(&debugData, "debugData", "HexFEM debugData"))
    , hexahedronHandler(nullptr)
{
    // coef of each vertices <Mat 8x3, int> for a hexa
    // 	      7---------6
    //       /	       /|
    //      /	      / |
    //     4---------5  |
    //     |    	 |  |
    //     |  3------|--2
    //     | / 	     | /
    //     |/	     |/
    //     0---------1
    /*_coef[0][0] = -1;		_coef[0][1] = -1;		_coef[0][2] = -1;
    _coef[1][0] = 1;		_coef[1][1] = -1;		_coef[1][2] = -1;
    _coef[2][0] = 1;		_coef[2][1] = 1;		_coef[2][2] = -1;
    _coef[3][0] = -1;		_coef[3][1] = 1;		_coef[3][2] = -1;
    _coef[4][0] = -1;		_coef[4][1] = -1;		_coef[4][2] = 1;
    _coef[5][0] = 1;		_coef[5][1] = -1;		_coef[5][2] = 1;
    _coef[6][0] = 1;		_coef[6][1] = 1;		_coef[6][2] = 1;
    _coef[7][0] = -1;		_coef[7][1] = 1;		_coef[7][2] = 1;*/

    hexahedronHandler = new HexHandler(this, &hexahedronInfo);

    //f_poissonRatio.setRequired(true);
    //f_youngModulus.setRequired(true);
    d_totalVolume = 0;
    d_currentVolume = 0;
    d_debugRendering = false;
}

template <class DataTypes>
HexahedralElastoplasticFEMForceField<DataTypes>::~HexahedralElastoplasticFEMForceField()
{
    if(hexahedronHandler) delete hexahedronHandler;
}

template <class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();

    this->getContext()->get(_topology);
    if (_topology == nullptr)
    {
        SingleLink<HexahedralElastoplasticFEMForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
        msg_info() << "(HexahedralElastoplasticFEMForceField) link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
        if (l_topology) _topology = l_topology.get();
        if (_topology == nullptr || _topology->getNbHexahedra() <= 0)
            serr << "ERROR(HexahedralElastoplasticFEMForceField): object must have hexahedron based topology." << sendl;
        return;

    }
    std::cout<<"HexahedralElastoplasticFEMForceField<DataTypes>::init() with "<< _topology->getNbHexahedra()<<" elements.\n";
    this->reinit(); // compute per-element stiffness matrices and other precomputed values
}


template <class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::reinit()
{
    std::cout << "HexahedralElastoplasticFEMForceField<DataTypes>::reinit() \n";
    if (f_method.getValue() == "large")
        this->setMethod(LARGE);
    else if (f_method.getValue() == "polar")
        this->setMethod(POLAR);

    type::vector<typename HexahedralElastoplasticFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());


    hexahedronInf.resize(_topology->getNbHexahedra());
    //TODO get fixed indices
    //sofa::component::projectiveconstraintset::FixedConstraint<DataTypes>* _fixedConstraint;
    //sofa::component::projectiveconstraintset::LinearMovementConstraint<DataTypes>* _movingConstraint;
    //this->getContext()->get(_fixedConstraint);
    //this->getContext()->get(_movingConstraint);
    //if (_fixedConstraint) _fixedIndices = _fixedConstraint->d_indices.getValue();

    for (size_t i = 0; i < _topology->getNbHexahedra(); ++i)
    {
        hexahedronHandler->applyCreateFunction(i, hexahedronInf[i],
            _topology->getHexahedron(i), (const std::vector< Index >)0,
            (const std::vector< double >)0);
    }
    hexahedronInfo.createTopologyHandler(_topology, hexahedronHandler);
    hexahedronInfo.registerTopologicalData();
    hexahedronInfo.endEdit();

    //initialize the plasticity offsets
    int nP = _topology->getNbPoints();
    restStateOffsets.resize(nP);//reset all offsets
    restStateOffsets.fill(Coord(0, 0, 0));
    d_currentVolume = 0;

    //Fracture initialize
    _enabledFracture = false;//false
    _elemsToBeFractured.resize(0);
    this->getContext()->get(_hexModifier);
    if (!_hexModifier) _enabledFracture = false;

    //Debug Rendering initialize
    type::vector<Real>& debugInfPoint = *(debugData.beginEdit());
    debugInfPoint.resize(_topology->getNbPoints());
    debugInfPoint.fill(0.0);
    debugData.endEdit();
    
}

template<class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::addForce (const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& p, const DataVecDeriv& /*v*/)
{
    WDataRefVecDeriv _f = f;
    RDataRefVecCoord _p = p;

    _f.resize(_p.size());
    //TODO this code should be added into updateRest()
    //if (needsToUpdateRestMesh) ///plasticity update is needed, reset them first.
    //{
    //    int nP = _topology->getNbPoints();
    //    restStateOffsets.resize(nP);//reset all offsets
    //    restStateOffsets.fill(Coord(0, 0, 0));
    //    d_currentVolume = 0;
    //}
    //int nP = _topology->getNbPoints();
    
    //DebugInfo needs to be reset per computation
    if (d_debugRendering)
    {
        type::vector<Real>& debugInfPoint = *(debugData.beginEdit());
        debugInfPoint.resize(_topology->getNbPoints());
        debugInfPoint.fill(0.0);
        debugData.endEdit();
    }

    switch(method)
    {
    case LARGE :
    {
        for(size_t i = 0 ; i<_topology->getNbHexahedra(); ++i)
        {
            accumulateForceLarge( _f, _p, i);
        }
        //if (needsToUpdateRestMesh) updateRestStateLarge();
        break;
    }
    case POLAR :
    {
        for(size_t i = 0 ; i<_topology->getNbHexahedra(); ++i)
        {
            accumulateForcePolar( _f, _p, i);
        }
        //if (needsToUpdateRestMesh) updateRestStatePolar();
        break;
    }
    }
    if (needsToUpdateRestMesh) updateRestStatePlasticity();

}

template<class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::addDForce (const core::MechanicalParams *mparams, DataVecDeriv& v, const DataVecDeriv& x)
{
    helper::WriteAccessor< DataVecDeriv > _v = v;
    const VecCoord& _x = x.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    if( _v.size()!=_x.size() ) _v.resize(_x.size());

    const type::vector<typename HexahedralElastoplasticFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = hexahedronInfo.getValue();

    for(Size i = 0 ; i<_topology->getNbHexahedra(); ++i)
    {
        Transformation R_0_2;
        R_0_2.transpose(hexahedronInf[i].rotation);

        Displacement X;

        for(int w=0; w<8; ++w)
        {
            Coord x_2;
            x_2 = R_0_2 * _x[_topology->getHexahedron(i)[w]] * kFactor;
            X[w*3] = x_2[0];
            X[w*3+1] = x_2[1];
            X[w*3+2] = x_2[2];
        }

        Displacement F;
        computeForce( F, X, hexahedronInf[i].stiffness );//computeForce( F, X, hexahedronInfo[i].stiffness );

        for(int w=0; w<8; ++w)
            _v[_topology->getHexahedron(i)[w]] -= hexahedronInf[i].rotation * Deriv(F[w*3], F[w*3+1], F[w*3+2]);
    }
}



/////////////////////////////////////////////////
/////////////Plasticity Helper Methods///////////
/////////////////////////////////////////////////

template<class DataTypes>
typename HexahedralElastoplasticFEMForceField<DataTypes>::Mat33 HexahedralElastoplasticFEMForceField<DataTypes>::computeJacobian(const type::fixed_array<Coord, 8>& coords, Real x, Real y, Real z)
{
    // Evaluate the tri-linear at reference location (x,y,z) in [-1,1]^3
    // X = coords[0] (1-x)(1-y)(1-z)/8 + coords[1] (1+x)(1-y)(1-z)/8 + coords[2] (1+x)(1+y)(1-z)/8 + coords[3] (1-x)(1+y)(1-z)/8 + coords[4] (1-x)(1-y)(1+z)/8 + coords[5] (1+x)(1-y)(1+z)/8 + coords[6] (1+x)(1+y)(1+z)/8 + coords[7] (1-x)(1+y)(1+z)/8
    // J = [ DX / x ] 
    Mat33 J;
    Coord DXDx = (coords[1] - coords[0]) * (1 - y) * (1 - z) / 8 + (coords[2] - coords[3]) * (1 + y) * (1 - z) / 8 + (coords[5] - coords[4]) * (1 - y) * (1 + z) / 8 + (coords[6] - coords[7]) * (1 + y) * (1 + z) / 8;
    Coord DXDy = (coords[3] - coords[0]) * (1 - x) * (1 - z) / 8 + (coords[2] - coords[1]) * (1 + x) * (1 - z) / 8 + (coords[7] - coords[4]) * (1 - x) * (1 + z) / 8 + (coords[6] - coords[5]) * (1 + x) * (1 + z) / 8;
    Coord DXDz = (coords[4] - coords[0]) * (1 - x) * (1 - y) / 8 + (coords[5] - coords[1]) * (1 + x) * (1 - y) / 8 + (coords[6] - coords[2]) * (1 + x) * (1 + y) / 8 + (coords[7] - coords[3]) * (1 - x) * (1 + y) / 8;
    for (int c = 0; c < 3; ++c)
    {
        J[0][c] = DXDx[c];
        J[1][c] = DXDy[c];
        J[2][c] = DXDz[c];
    }
    return J;
}

template<class DataTypes>
typename HexahedralElastoplasticFEMForceField<DataTypes>::Mat33 HexahedralElastoplasticFEMForceField<DataTypes>::computeCenterJacobian(const type::fixed_array<Coord, 8>& coords)
{
    Mat33 J;
    Coord horizontal = (coords[1] - coords[0] + coords[2] - coords[3] + coords[5] - coords[4] + coords[6] - coords[7]) * .25;
    Coord vertical = (coords[3] - coords[0] + coords[2] - coords[1] + coords[7] - coords[4] + coords[6] - coords[5]) * .25;
    Coord deep = (coords[4] - coords[0] + coords[7] - coords[3] + coords[5] - coords[1] + coords[6] - coords[2]) * .25;
    for (int c = 0; c < 3; ++c)
    {
        J[c][0] = horizontal[c] / 2;
        J[c][1] = vertical[c] / 2;
        J[c][2] = deep[c] / 2;
    }
    return J;
}

template<class DataTypes>
typename HexahedralElastoplasticFEMForceField<DataTypes>::Real HexahedralElastoplasticFEMForceField<DataTypes>::computeElementVolume(const type::fixed_array<Coord, 8>& coords)
{
    //000->0, 100->1, 110->2, 010->3, 001->4, 101->5, 111->6, 011->7
    Real volume;
    float t[336];
    t[1] = coords[0][2];
    t[2] = coords[4][0];
    t[3] = coords[0][0];
    t[4] = t[2] - t[3];
    t[5] = coords[1][1];
    t[6] = coords[0][1];
    t[7] = t[5] - t[6];
    t[9] = coords[1][0];
    t[10] = t[9] - t[3];
    t[11] = coords[4][1];
    t[12] = t[11] - t[6];
    t[14] = t[4] * t[7] - t[10] * t[12];
    t[17] = coords[1][2];
    t[18] = coords[5][0];
    t[19] = t[18] - t[9];
    t[21] = coords[5][1];
    t[22] = t[21] - t[5];
    t[24] = t[19] * t[7] - t[10] * t[22];
    t[27] = coords[4][2];
    t[28] = t[21] - t[11];
    t[30] = t[18] - t[2];
    t[32] = t[4] * t[28] - t[30] * t[12];
    t[35] = coords[5][2];
    t[38] = t[19] * t[28] - t[30] * t[22];
    t[41] = coords[3][0];
    t[42] = t[41] - t[3];
    t[44] = coords[3][1];
    t[45] = t[44] - t[6];
    t[47] = t[42] * t[12] - t[4] * t[45];
    t[50] = coords[7][0];
    t[51] = t[50] - t[2];
    t[53] = coords[7][1];
    t[54] = t[53] - t[11];
    t[56] = t[51] * t[12] - t[4] * t[54];
    t[59] = coords[3][2];
    t[60] = t[53] - t[44];
    t[62] = t[50] - t[41];
    t[64] = t[42] * t[60] - t[62] * t[45];
    t[67] = coords[7][2];
    t[70] = t[51] * t[60] - t[62] * t[54];
    t[81] = t[1] * t[14] / 0.9e1 + t[17] * t[24] / 0.9e1 + t[27] * t[32] / 0.9e1 + t[35] * t[38] / 0.9e1 + t[1] * t[47] / 0.9e1 + t[27] * t[56] / 0.9e1 + t[59] * t[64] / 0.9e1 + t[67] * t[70] / 0.9e1 + t[1] * t[56] / 0.18e2 + t[1] * t[64] / 0.18e2 + t[1] * t[70] / 0.36e2 + t[27] * t[47] / 0.18e2;
    t[98] = coords[2][0];
    t[99] = t[98] - t[41];
    t[101] = coords[2][1];
    t[102] = t[101] - t[44];
    t[104] = t[99] * t[45] - t[42] * t[102];
    t[107] = t[101] - t[5];
    t[109] = t[98] - t[9];
    t[111] = t[10] * t[107] - t[109] * t[7];
    t[116] = t[99] * t[107] - t[109] * t[102];
    t[121] = t[10] * t[45] - t[42] * t[7];
    t[124] = t[27] * t[64] / 0.36e2 + t[27] * t[70] / 0.18e2 + t[59] * t[47] / 0.18e2 + t[59] * t[56] / 0.36e2 + t[59] * t[70] / 0.18e2 + t[67] * t[47] / 0.36e2 + t[67] * t[56] / 0.18e2 + t[67] * t[64] / 0.18e2 + t[1] * t[104] / 0.18e2 + t[1] * t[111] / 0.18e2 + t[1] * t[116] / 0.36e2 + t[59] * t[121] / 0.18e2;
    t[136] = coords[2][2];
    t[143] = coords[6][2];
    t[144] = -t[99];
    t[145] = coords[6][1];
    t[146] = t[101] - t[145];
    t[148] = coords[6][0];
    t[149] = t[98] - t[148];
    t[150] = -t[102];
    t[152] = t[144] * t[146] - t[149] * t[150];
    t[155] = t[50] - t[148];
    t[156] = -t[60];
    t[158] = -t[62];
    t[159] = t[53] - t[145];
    t[161] = t[155] * t[156] - t[158] * t[159];
    t[166] = t[144] * t[156] - t[158] * t[150];
    t[171] = t[155] * t[146] - t[149] * t[159];
    t[174] = t[59] * t[111] / 0.36e2 + t[59] * t[116] / 0.18e2 + t[17] * t[121] / 0.18e2 + t[17] * t[104] / 0.36e2 + t[17] * t[116] / 0.18e2 + t[136] * t[121] / 0.36e2 + t[136] * t[104] / 0.18e2 + t[136] * t[111] / 0.18e2 + t[143] * t[152] / 0.18e2 + t[143] * t[161] / 0.18e2 + t[143] * t[166] / 0.36e2 + t[136] * t[171] / 0.18e2;
    t[177] = t[21] - t[145];
    t[179] = t[18] - t[148];
    t[181] = t[149] * t[177] - t[179] * t[146];
    t[184] = -t[19];
    t[186] = -t[22];
    t[188] = t[184] * t[177] - t[179] * t[186];
    t[191] = -t[107];
    t[193] = -t[109];
    t[195] = t[149] * t[191] - t[193] * t[146];
    t[200] = t[184] * t[191] - t[193] * t[186];
    t[217] = t[136] * t[161] / 0.36e2 + t[143] * t[181] / 0.9e1 + t[35] * t[188] / 0.9e1 + t[136] * t[195] / 0.9e1 + t[17] * t[200] / 0.9e1 + t[1] * t[121] / 0.9e1 + t[59] * t[104] / 0.9e1 + t[17] * t[111] / 0.9e1 + t[136] * t[116] / 0.9e1 + t[143] * t[171] / 0.9e1 + t[136] * t[152] / 0.9e1 + t[67] * t[161] / 0.9e1;
    t[244] = t[59] * t[166] / 0.9e1 + t[136] * t[166] / 0.18e2 + t[67] * t[171] / 0.18e2 + t[67] * t[152] / 0.36e2 + t[67] * t[166] / 0.18e2 + t[59] * t[171] / 0.36e2 + t[59] * t[152] / 0.18e2 + t[59] * t[161] / 0.18e2 + t[143] * t[188] / 0.18e2 + t[143] * t[195] / 0.18e2 + t[143] * t[200] / 0.36e2 + t[35] * t[181] / 0.18e2;
    t[261] = -t[51];
    t[263] = -t[54];
    t[265] = t[261] * t[159] - t[155] * t[263];
    t[268] = -t[28];
    t[270] = -t[30];
    t[272] = t[179] * t[268] - t[270] * t[177];
    t[277] = t[261] * t[268] - t[270] * t[263];
    t[282] = t[179] * t[159] - t[155] * t[177];
    t[285] = t[35] * t[195] / 0.36e2 + t[35] * t[200] / 0.18e2 + t[136] * t[181] / 0.18e2 + t[136] * t[188] / 0.36e2 + t[136] * t[200] / 0.18e2 + t[17] * t[181] / 0.36e2 + t[17] * t[188] / 0.18e2 + t[17] * t[195] / 0.18e2 + t[143] * t[265] / 0.18e2 + t[143] * t[272] / 0.18e2 + t[143] * t[277] / 0.36e2 + t[67] * t[282] / 0.18e2;
    t[311] = t[67] * t[272] / 0.36e2 + t[67] * t[277] / 0.18e2 + t[35] * t[282] / 0.18e2 + t[35] * t[265] / 0.36e2 + t[35] * t[277] / 0.18e2 + t[27] * t[282] / 0.36e2 + t[27] * t[265] / 0.18e2 + t[27] * t[272] / 0.18e2 + t[143] * t[282] / 0.9e1 + t[67] * t[265] / 0.9e1 + t[35] * t[272] / 0.9e1 + t[27] * t[277] / 0.9e1;
    t[336] = t[1] * t[24] / 0.18e2 + t[1] * t[32] / 0.18e2 + t[1] * t[38] / 0.36e2 + t[17] * t[14] / 0.18e2 + t[17] * t[32] / 0.36e2 + t[17] * t[38] / 0.18e2 + t[27] * t[14] / 0.18e2 + t[27] * t[24] / 0.36e2 + t[27] * t[38] / 0.18e2 + t[35] * t[14] / 0.36e2 + t[35] * t[24] / 0.18e2 + t[35] * t[32] / 0.18e2;
    volume = t[81] + t[124] + t[174] + t[217] + t[244] + t[285] + t[311] + t[336];

    //The above code computes hex defined on [0,1]^3 with different vert ordering
    //scaled by *-0.125 to match the element definition in SOFA
    return volume * -0.125;
}

template<class DataTypes>
typename HexahedralElastoplasticFEMForceField<DataTypes>::Real HexahedralElastoplasticFEMForceField<DataTypes>::rotDist(Mat33 R1, Mat33 R2)
{
    Mat33 Dist; Dist.identity();
    Dist -= R1 * R2.transposed();
    return sqrt(sofa::type::trace(Dist.multTranspose(Dist))) / 2.83; /// Frobenius norm scaled to range [0,1]
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// large displacements method


template<class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::initLarge(const int i)
{
    // Rotation matrix (initial Hexahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    const VecCoord& X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    type::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = (X0)[_topology->getHexahedron(i)[w]];


    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    Transformation R_0_1;
    computeRotationLarge( R_0_1, horizontal,vertical);


    type::vector<typename HexahedralElastoplasticFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    for(int w=0; w<8; ++w)
        hexahedronInf[i].rotatedInitialElements[w] = R_0_1*nodes[w];

    ///initialize the plastic parameters per element
    if (f_plasticMaxThreshold.getValue() > 0) 
    {
        hexahedronInf[i].plasticYieldThreshold = this->f_plasticYieldThreshold.getValue();
        hexahedronInf[i].plasticMaxThreshold = this->f_plasticMaxThreshold.getValue();
        hexahedronInf[i].needsToUpdateRestMesh = false;
        Mat33 J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);
        hexahedronInf[i].restVolume = sofa::type::determinant(J);
        this->d_totalVolume += hexahedronInf[i].restVolume;
        hexahedronInf[i].materialDeformationInverse = J.inverted(); ///initialize the per element material deformation
    }

    computeMaterialStiffness( hexahedronInf[i].materialMatrix, f_youngModulus.getValue(), f_poissonRatio.getValue() );
    computeElementStiffness( hexahedronInf[i].stiffness, hexahedronInf[i].materialMatrix, hexahedronInf[i].rotatedInitialElements);

    hexahedronInfo.endEdit();
}


//void HexahedralFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey)

template<class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::accumulateForceLarge( WDataRefVecDeriv& f, RDataRefVecCoord & p, const int i)
{
    type::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = p[_topology->getHexahedron(i)[w]];

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;

    Transformation R_0_2; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationLarge( R_0_2, horizontal,vertical);

    type::vector<typename HexahedralElastoplasticFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    hexahedronInf[i].rotation.transpose(R_0_2);

    // positions of the deformed and displaced Hexahedre in its frame
    type::Vec<8,Coord> deformed;
    for(int w=0; w<8; ++w)
        deformed[w] = R_0_2 * nodes[w];


    // Compute the plasticity
    Real plastic_ratio = .0; ///(refer to gamma in the paper) this ratio describes the mount of deformation is absorbed in a timestep
    type::vector<Real>& debugInfPoint = *(debugData.beginEdit());
    if (hexahedronInf[i].plasticYieldThreshold > 0 && f_useHigherOrderPlasticity.getValue())
    {
        Coord restElementCenter = hexahedronInf[i].rotatedInitialElements[0];
        for (int w = 1; w < 8; w++)
            restElementCenter += hexahedronInf[i].rotatedInitialElements[w];
        restElementCenter = restElementCenter / 8.0;

        hexahedronInf[i].F_C = computeCenterJacobian(deformed) * hexahedronInf[i].materialDeformationInverse;

        Real volume = computeElementVolume(hexahedronInf[i].rotatedInitialElements);
        d_currentVolume += volume;
        Real volumeRatio = volume / hexahedronInf[i].restVolume;
        if (volumeRatio < 0) {
            volumeRatio = 1; std::cout << "[HexahedralElastoplasticFEMForceField] inverted elem " << i << "...\n";
        }

        for (int w = 0; w < 8; ++w)
        {
            unsigned int id = _topology->getHexahedron(i)[w]; //vert id global
            unsigned int NNeighbors = _topology->getHexahedraAroundVertex(id).size();
            debugInfPoint[id] += volumeRatio / NNeighbors;

            //ignore fixed indices
            if (std::find(_fixedIndices.begin(), _fixedIndices.end(), id) != _fixedIndices.end()) {
                continue;
            }

            //Compute vertex deformation
            Mat33 U, V;
            type::Vec<3, Real> Diag;
            Mat33 FV = computeJacobian(deformed, _coef[w][0], _coef[w][1], _coef[w][2]) * hexahedronInf[i].materialDeformationInverse;
            Mat33 phongF = (FV + hexahedronInf[i].F_C) / 2.0; //blended vertex Deformation
            helper::Decompose<Real>::SVD(phongF, U, Diag, V);
            Real L2Norm = std::max(std::max(Diag[0], Diag[1]), Diag[2]);
            if (L2Norm > 1 + hexahedronInf[i].plasticYieldThreshold)
            {
                type::Vec<3, Real> plasticDiag = Diag / std::cbrt(Diag[0] * Diag[1] * Diag[2]);
                if (L2Norm > 1 + 10 * hexahedronInf[i].plasticMaxThreshold) {
                    //TODO fracture or tearing
                    //hexahedronInf[i].plasticYieldThreshold = 0;
                    //L2Norm = std::max(1 + hexahedronInf[i].plasticMaxThreshold, 1 + hexahedronInf[i].plasticYieldThreshold);
                }
                else if (L2Norm > 1 + hexahedronInf[i].plasticMaxThreshold) {
                    //L2Norm = 1 + hexahedronInf[i].plasticMaxThreshold;
                    L2Norm = std::max(1 + hexahedronInf[i].plasticMaxThreshold, 1 + hexahedronInf[i].plasticYieldThreshold);
                }

                plastic_ratio = f_plasticCreep.getValue() * (L2Norm - 1 - hexahedronInf[i].plasticYieldThreshold) / L2Norm;

                //work hardening
                hexahedronInf[i].plasticYieldThreshold += f_hardeningParam.getValue() * 0.005/*dt*/ / 8.0 /*8 vertices*/; //hardening

                plasticDiag[0] = std::pow(plasticDiag[0], plastic_ratio);
                plasticDiag[1] = std::pow(plasticDiag[1], plastic_ratio);
                plasticDiag[2] = std::pow(plasticDiag[2], plastic_ratio);
                //std::cout << "plasticDiag of E"<<i<<" "<< plasticDiag << "...   ";

                //0 : Centered transform approach:
                if (f_debugPlasticMethod.getValue() == 0)
                {
                    hexahedronInf[i].rotatedInitialElements[w] += 0.04 * (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).linearProduct(plasticDiag - Coord(1, 1, 1)) / NNeighbors;
                }

                
                //1: Centered transform + vertex Offset, scaled with displacement per vertex approach:
                if (f_debugPlasticMethod.getValue() == 1)
                {
                    hexahedronInf[i].rotatedInitialElements[w] += /*0.01*(hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).linearProduct(plasticDiag - Coord(1, 1, 1))*/
                        0.02 * plastic_ratio * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]);
                    /*if (id == 67)
                        std::cout << "d=" << 0.02 * plastic_ratio * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]) << "   ..";*/
                }

                //2 : Use deformation as offsets with direction correction Approach
                if (f_debugPlasticMethod.getValue() == 2)
                {
                    hexahedronInf[i].elementPlasticOffset[w] = (plasticDiag - Coord(1, 1, 1)) * 0.008/*dt*/;
                    for (int j = 0; j < 3; j++)
                        //if (plasticDiag[j] < 1)
                        hexahedronInf[i].elementPlasticOffset[w][j] *= _coef[w][j];
                }

                //3 : scaled with actual vertex displacement
                if (f_debugPlasticMethod.getValue() == 3)//Not being used
                {
                    hexahedronInf[i].elementPlasticOffset[w] = (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).linearProduct(plasticDiag - Coord(1, 1, 1));
                    hexahedronInf[i].elementPlasticOffset[w] *= std::min(1.0, (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]).norm()) * 0.02;
                }

                //Volume Constraint Projection
                if (f_preserveElementVolume.getValue() && (volumeRatio >= 1.005 || volumeRatio <= 0.995))
                {
                    //std::cout << "prj Vol on "<<id<<"..   ";
                    //hexahedronInf[i].elementPlasticOffset[w] += (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter)*(1 - volumeRatio)*0.02/*dt*/;
                    hexahedronInf[i].rotatedInitialElements[w] += (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter) * (1 - volumeRatio) * 0.02 / NNeighbors;
                }            

                Mat33 J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);//recompute element center deformation inverse based on rotated rest elements
                hexahedronInf[i].materialDeformationInverse = J.inverted();
            }
        }
    }
    else if (hexahedronInf[i].plasticYieldThreshold > 0) //constant cell center plasticity
    {
        Coord restElementCenter = hexahedronInf[i].rotatedInitialElements[0];
        for (int w = 1; w < 8; w++)
            restElementCenter += hexahedronInf[i].rotatedInitialElements[w];
        restElementCenter = restElementCenter / 8.0;

        Mat33 J = computeCenterJacobian(deformed); /// compute the jacobian (world over reference)
        Mat33 totalF = J * hexahedronInf[i].materialDeformationInverse; /// Total Deformation Gradient F = J (world) * J^-1 (material)
        //Real Fnorm = defaulttype::trace(totalF.multTranspose(totalF)); /// Frobenius norm of totalF
        Real volume = computeElementVolume(hexahedronInf[i].rotatedInitialElements);
        Real volumeRatio = volume / hexahedronInf[i].restVolume;

        if (d_debugRendering) {
            for (int k = 0; k < 8; ++k)
            {
                unsigned int id = _topology->getHexahedron(i)[k]; //vert id
                unsigned int NNeighbors = _topology->getHexahedraAroundVertex(id).size();
                debugInfPoint[id] += volumeRatio / NNeighbors;
            }
        }
        

        Mat33 U, V;
        type::Vec<3, Real> Diag;
        helper::Decompose<Real>::SVD(totalF, U, Diag, V);
        Real L2Norm = std::max(std::max(Diag[0], Diag[1]), Diag[2]);
        if (L2Norm > 1 + hexahedronInf[i].plasticYieldThreshold)
        {
            if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;

            type::Vec<3, Real> plasticDiag = Diag / std::cbrt(Diag[0] * Diag[1] * Diag[2]);
            if (L2Norm > 1 + hexahedronInf[i].plasticMaxThreshold)
                L2Norm = std::max(1 + hexahedronInf[i].plasticMaxThreshold, 1 + hexahedronInf[i].plasticYieldThreshold);
            plastic_ratio = f_plasticCreep.getValue() * (L2Norm - 1 - hexahedronInf[i].plasticYieldThreshold) / L2Norm;
            hexahedronInf[i].plasticYieldThreshold += f_hardeningParam.getValue() * 0.005 /*dt*/; //work hardening
            
            //Update rest state
            for (int k = 0; k < 8; ++k)
            {
                //hexahedronInf[i].rotatedInitialElements[k] += 0.02 * plastic_ratio * (deformed[k] - hexahedronInf[i].rotatedInitialElements[k]);
                hexahedronInf[i].elementPlasticOffset[k] += 0.02 * plastic_ratio * (deformed[k] - hexahedronInf[i].rotatedInitialElements[k]);

            }

            //TODO: Check Fracture based on maxYieldThreshold
            //J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);//recompute element center deformation inverse based on rotated rest elements
            //hexahedronInf[i].materialDeformationInverse = J.inverted();
        }
        if (f_preserveElementVolume.getValue() && (volumeRatio >= 1.001 || volumeRatio <= 0.999))
        {
            //std::cout << " r" << volumeRatio;
            if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
            for (int w = 0; w < 8; ++w)
                hexahedronInf[i].elementPlasticOffset[w] += (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter) * (1 - volumeRatio) * 0.02/*dt*/;
        }
    }
    debugData.endEdit();

    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        int indice = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[indice+j] = hexahedronInf[i].rotatedInitialElements[k][j] - deformed[k][j];
    }

    Displacement F; //forces
    computeForce( F, D, hexahedronInf[i].stiffness ); // computeForce( F, D, hexahedronInf[i].stiffness ); // compute force on element

    for(int w=0; w<8; ++w)
        f[_topology->getHexahedron(i)[w]] += hexahedronInf[i].rotation * Deriv( F[w*3],  F[w*3+1],   F[w*3+2]  );

    hexahedronInfo.endEdit();
}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// polar decomposition method


template<class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::initPolar(const int i)
{
    const VecCoord& X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    type::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = (X0)[_topology->getHexahedron(i)[j]];

    Transformation R_0_1; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationPolar( R_0_1, nodes );


    type::vector<typename HexahedralElastoplasticFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    for(int j=0; j<8; ++j)
    {
        hexahedronInf[i].rotatedInitialElements[j] = R_0_1 * nodes[j];
        //hexahedronInf[i].vertRotation[j] = R_0_1;
    }

    
    ///initialize the plastic parameters per element
    if (f_plasticMaxThreshold.getValue() > 0) 
    {
        hexahedronInf[i].plasticYieldThreshold = this->f_plasticYieldThreshold.getValue();
        hexahedronInf[i].plasticMaxThreshold = this->f_plasticMaxThreshold.getValue();
        hexahedronInf[i].plasticRotationYieldThreshold = this->f_plasticRotationYieldThreshold.getValue();
        hexahedronInf[i].plasticMaxRotationThreshold = this->f_plasticMaxRotationThreshold.getValue();
        hexahedronInf[i].needsToUpdateRestMesh = false;
        Mat33 J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);
        hexahedronInf[i].restVolume = type::determinant(J);
        this->d_totalVolume += hexahedronInf[i].restVolume;
        hexahedronInf[i].materialDeformationInverse = J.inverted(); ///initialize the per element material deformation

        hexahedronInfo[i].restRotation.transpose(R_0_1);
        hexahedronInf[i].lastRotation.transpose(R_0_1);
    }

    computeMaterialStiffness( hexahedronInf[i].materialMatrix, f_youngModulus.getValue(), f_poissonRatio.getValue() );
    computeElementStiffness( hexahedronInf[i].stiffness, hexahedronInf[i].materialMatrix, hexahedronInf[i].rotatedInitialElements);

    hexahedronInfo.endEdit();
}


template<class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::accumulateForcePolar(WDataRefVecDeriv& f, RDataRefVecCoord & p, const int i)
{
    type::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = p[_topology->getHexahedron(i)[j]];


    Transformation R_0_2; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationPolar( R_0_2, nodes );


    type::vector<typename HexahedralElastoplasticFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    hexahedronInf[i].rotation.transpose(R_0_2);
    Mat33 incrementalRotation = R_0_2 * hexahedronInf[i].restRotation;
    Real restAngleChange = std::acos((type::trace(incrementalRotation) - 1) / 2);

    // positions of the deformed and displaced Hexahedre in its frame
    type::Vec<8,Coord> deformed;
    for(int j=0; j<8; ++j)
        deformed[j] = R_0_2 * nodes[j];

    type::vector<Real>& debugInfPoint = *(debugData.beginEdit());

    // Compute the plasticity 
    Mat33 I3; I3.identity();
    Real plastic_ratio = .0; ///(refer to gamma in the paper) this ratio describes the mount of deformation is absorbed in a timestep

    //blended vertex method (higher order) for handling plasticity
    if (hexahedronInf[i].plasticYieldThreshold > 0 && f_useHigherOrderPlasticity.getValue())
    {
        Real dt = this->getContext()->getDt();
        Coord restElementCenter = hexahedronInf[i].rotatedInitialElements[0];
        for (int w = 1; w < 8; w++)
            restElementCenter += hexahedronInf[i].rotatedInitialElements[w];
        restElementCenter = restElementCenter / 8.0;
        Real totalDisp = 0;
        for (int j = 0; j < 8; j++)
            totalDisp += (deformed[j] - hexahedronInf[i].rotatedInitialElements[j]).norm();
        hexahedronInf[i].F_C = computeCenterJacobian(deformed) * hexahedronInf[i].materialDeformationInverse;
        
        Mat33 JC = computeCenterJacobian(deformed);
        //Compute center rotation
        Mat33 RC, RV;//center Rotation and vertex Rotation
        Mat33 rU, rV;//for SVD
        type::Vec<3, Real> rDiag;
        helper::Decompose<Real>::SVD(JC, rU, rDiag, rV);
        RC = rU * rV.transposed();

        //Compute maximal vertex distortion -- for the RBR plastic rotation approach
        Real ratioMaxDistortion = 0;//maximal vertex rotation distortion
        
        for (int w = 0; w < 8; ++w)
        {
            //unsigned int id = _topology->getHexahedron(i)[w];
            //defaulttype::Vec<3, Real> Diag;
            //Mat33 FV = 0.5 * hexahedronInf[i].F_C  + 0.5 * computeJacobian(deformed, _coef[w][0], _coef[w][1], _coef[w][2]) * hexahedronInf[i].materialDeformationInverse;
            Mat33 JV = 0.5 * JC + 0.5 * computeJacobian(deformed, _coef[w][0], _coef[w][1], _coef[w][2]); //blended vert jacobian
            helper::Decompose<Real>::SVD(JV, rU, rDiag, rV);
            Mat33 RV = rU * rV.transposed();//vert Rotation
            //ratioMaxDistortion = std::max(ratioMaxDistortion, rotDist(RV, RC));
            //hexahedronInf[i].vertRotation[w] = RV;//Phong:Blended vertex
            Mat33 vertRotDistort = RV * RC.transposed();
            Real vertAngleChange = std::acos((type::trace(vertRotDistort) - 1) / 2); //TODO maintain this value <0.xx or >3.yy?
            ratioMaxDistortion = std::max(ratioMaxDistortion, vertAngleChange);
            //Mat33 axisM = (vertRotDistort - vertRotDistort.transposed()) / 2;
            //Coord axis; axis[0] = axisM[2][1]; axis[1] = axisM[0][2]; axis[2] = axisM[1][0]; axis.normalize();                 
        }
        //std::cout <<std::endl;
        //if(i==0) std::cout << ratioMaxDistortion << " ";
        Real volume = computeElementVolume(hexahedronInf[i].rotatedInitialElements);
        d_currentVolume += volume;
        Real volumeRatio = volume / hexahedronInf[i].restVolume;

        if (volumeRatio < 0)//TODO handle inverted volume?
        {
            volumeRatio = 1; std::cout << "detected inverted elem " << i << "\n";
            hexahedronInf[i].plasticRotationYieldThreshold = 0;
            hexahedronInf[i].plasticYieldThreshold = 0;
        }

        //check material stiffness updates hardening
        if (hexahedronInf[i].plasticRotationYieldThreshold >= hexahedronInf[i].plasticMaxRotationThreshold ||
            hexahedronInf[i].plasticYieldThreshold >= hexahedronInf[i].plasticMaxThreshold)
        {
            hexahedronInf[i].plasticRotationYieldThreshold = 0; //set to pure elastic / or do fracture 
            std::cout << "Ele" << i << " reach max yield...\n";
            if (_enabledFracture) {
                _elemsToBeFractured.push_back(i);
                _hexModifier->removeItems(_elemsToBeFractured);
                _elemsToBeFractured.resize(0);
            }
            return;
        }

        //For each vertex, compute the plastic deformation and update rest space        
        for (int w = 0; w < 8; ++w)
        {
            unsigned int id = _topology->getHexahedron(i)[w];
            unsigned int NNeighbors = _topology->getHexahedraAroundVertex(id).size();
            if (d_debugRendering) debugInfPoint[id] += volumeRatio / NNeighbors;

            //ignore fixed indices	
            if (std::find(_fixedIndices.begin(), _fixedIndices.end(), id) != _fixedIndices.end()) {
                continue;
            }

            Mat33 U, V;
            type::Vec<3, Real> Diag;
            Mat33 FV = computeJacobian(deformed, _coef[w][0], _coef[w][1], _coef[w][2]) * hexahedronInf[i].materialDeformationInverse;

            Mat33 phongF = (FV + hexahedronInf[i].F_C) / 2.0; //Blended vertex Deformation
            helper::Decompose<Real>::SVD(phongF, U, Diag, V);
            //RV = U * V.transposed();
            //Diag = defaulttype::diagonal(DiagMat);

            //Compute 2-norm of the first piola-kirch stress \sigma:
            /*Real y = this->f_youngModulus.getValue();
            Real p = this->f_poissonRatio.getValue();
            Real lambda = y * p / (1 + p) / (1 - 2 * p);
            Real miu = y / 2 / (1 + p);
            defaulttype::Vec<3, Real> DiagSigma;
            DiagSigma[0] = 2 * miu*(Diag[0] - 1) + lambda * (Diag[0] + Diag[1] + Diag[2] - 3);
            DiagSigma[1] = 2 * miu*(Diag[1] - 1) + lambda * (Diag[0] + Diag[1] + Diag[2] - 3);
            DiagSigma[2] = 2 * miu*(Diag[2] - 1) + lambda * (Diag[0] + Diag[1] + Diag[2] - 3);
            L2Norm = std::max(std::max(DiagSigma[0], DiagSigma[1]), DiagSigma[2]);
            if(i==0) std::cout << "||sigma||=" << L2Norm;*/

            Real L2Norm = std::max(std::max(Diag[0] - 1, Diag[1] - 1), Diag[2] - 1); // max eigen value TODO use better criteria
            
            //Handle symmetrical plasticity
            if (L2Norm > hexahedronInf[i].plasticYieldThreshold)
            {
                type::Vec<3, Real> plasticDiag = Diag / std::cbrt(Diag[0] * Diag[1] * Diag[2]); //normalization
                if (L2Norm > hexahedronInf[i].plasticMaxThreshold) {
                    L2Norm = hexahedronInf[i].plasticMaxThreshold;
                    //std::cout << "Ele " << i << "reached plasticMaxThreshold...";
                }

                plastic_ratio = f_plasticCreep.getValue() * (L2Norm - hexahedronInf[i].plasticYieldThreshold) / L2Norm;
                if (plastic_ratio > 0 && !hexahedronInf[i].needsToUpdateRestMesh)
                    hexahedronInf[i].needsToUpdateRestMesh = true;
                if (plastic_ratio < 0) plastic_ratio = 0;
                if (hexahedronInf[i].plasticYieldThreshold >= hexahedronInf[i].plasticMaxThreshold) {
                    std::cout << "Ele " << i << "reached max stretching...";
                    //hexahedronInf[i].plasticYieldThreshold = 0;
                    break;
                }
                plasticDiag[0] = std::pow(plasticDiag[0], plastic_ratio);
                plasticDiag[1] = std::pow(plasticDiag[1], plastic_ratio);
                plasticDiag[2] = std::pow(plasticDiag[2], plastic_ratio);
                /*if (id == 67 && f_debugPlasticMethod == 0) {
                    Mat33 debugPlasticF = U.multDiagonal(plasticDiag)*V;
                    std::cout << "E"<<i<<std::endl;
                    std::cout << "Diag=" << Diag << std::endl;
                    std::cout << "plastic_ratio=" << plastic_ratio << std::endl;
                    std::cout << "debugDisp=" << (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]) << std::endl;
                    std::cout << "debugPlasticDisp=" << debugPlasticF*(deformed[w] - hexahedronInf[i].rotatedInitialElements[w]) << std::endl;

                    std::cout << std::endl;
                }*/

                //0 : Centered transform approach:
                if (f_debugPlasticMethod.getValue() == 0)
                {
                    Mat33 debugVDVt = V.multDiagonal(plasticDiag) * V.transposed();
                    //Mat33 debugUDVt = U.multDiagonal(plasticDiag)*V.transposed();
                    hexahedronInf[i].elementPlasticOffset[w] = plastic_ratio * (debugVDVt - I3) * (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter);
                }

                //1: Centered transform + vertex disp Offset, scaled with displacement per vertex approach:
                if (f_debugPlasticMethod.getValue() == 1)
                {
                    Mat33 debugUDVt = V.multDiagonal(plasticDiag) * V.transposed();
                    hexahedronInf[i].elementPlasticOffset[w] = 0.01 * plastic_ratio * debugUDVt * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]);
                    hexahedronInf[i].plasticYieldThreshold += f_hardeningParam.getValue() * 0.01 * plastic_ratio * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]).norm2() / (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).norm2();
                }


                //2 : adding dt to above
                if (f_debugPlasticMethod.getValue() == 2)
                {
                    Mat33 debugUDVt = V.multDiagonal(plasticDiag) * V.transposed();
                    hexahedronInf[i].elementPlasticOffset[w] = dt * plastic_ratio * debugUDVt * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]);
                    hexahedronInf[i].plasticYieldThreshold += f_hardeningParam.getValue() * dt * plastic_ratio * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]).norm2() / (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).norm2();

                    /*helper::Decompose<Real>::SVD(hexahedronInf[i].F_C, U, Diag, V);
                    Mat33 debugVDVt = V.multDiagonal(plasticDiag)*V.transposed();
                    Mat33 debugUVt = U * V.transposed();*/
                    //std::cout << "debugUVt " << debugUVt << std::endl;
                    //hexahedronInf[i].elementPlasticOffset[w] = 0.01*plastic_ratio*debugVDVt*(/*deformedCenter - restElementCenter +*/ deformed[w] - hexahedronInf[i].rotatedInitialElements[w]);
                    //hexahedronInf[i].elementPlasticOffset[w] = 0.02 * (debugUVt-I3)*(hexahedronInf[i].rotatedInitialElements[w] - restElementCenter);     
                }

                if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
                //Mat33 J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);//recompute element center deformation inverse based on rotated rest elements	
                //hexahedronInf[i].materialDeformationInverse = J.inverted();

            }
            
            //Handle rotational plasticity
            if (hexahedronInf[i].plasticRotationYieldThreshold > 0) {
                //Compute this vertex rotation distortion
                Mat33 JV = 0.5 * JC + 0.5 * computeJacobian(deformed, _coef[w][0], _coef[w][1], _coef[w][2]); //vert jacobian
                helper::Decompose<Real>::SVD(JV, rU, rDiag, rV);
                Mat33 RV = rU * rV.transposed();//vert Rotation
                //hexahedronInf[i].vertRotation[w] = RV;//Phong:Blended vertex
                Mat33 vertRotDistort = RV * RC.transposed();
                Real vertAngleChange = std::acos((type::trace(vertRotDistort) - 1) / 2); //TODO maintain this value <0.xx or >3.yy?
                //Mat33 axisM = (vertRotDistort - vertRotDistort.transposed()) / 2;
                //Coord axis; axis[0] = axisM[2][1]; axis[1] = axisM[0][2]; axis[2] = axisM[1][0]; axis.normalize();
                //if(i==300) std::cout << vertAngleChange << " ";

                //Rotation plasticity update
                if (f_debugPlasticMethod.getValue() == 2 && restAngleChange > 0.1 && vertAngleChange > hexahedronInf[i].plasticRotationYieldThreshold)
                {
                    Real vertRotPlasticRatio = (vertAngleChange - hexahedronInf[i].plasticRotationYieldThreshold) / (restAngleChange + vertAngleChange);
                    //hexahedronInf[i].rotation = (1 - 0.1*(ratioDistortion - 0.9)) * hexahedronInf[i].rotation + 0.1 * (ratioDistortion - 0.9)*I3;
                    //hexahedronInf[i].elementPlasticOffset[w] += 0.1*(ratioDistortion - 0.9)*(R_0_2-I3)*(hexahedronInf[i].rotatedInitialElements[w] - restElementCenter);
                    hexahedronInf[i].elementPlasticOffset[w] -= f_plasticRotationCreep.getValue() * 0.5 * vertRotPlasticRatio * (incrementalRotation - I3) * (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter);
                    //if (i == 390) std::cout << " " << vertAngleChange <<" "<< restAngleChange << std::endl;
                    //hexahedronInf[i].elementPlasticOffset[w] = 0.02*(ratioDistortion - 0.01) * (hexahedronInf[i].vertRotation[w].transposed() - I3)*(hexahedronInf[i].rotatedInitialElements[w] - restElementCenter);
                    //std::cout << "r="<< ratioDistortion<<".. "<< RC;
                    /* hexahedronInf[i].elementPlasticOffset[w] = 0.02 *(ratioDistortion-0.3) * ( hexahedronInf[i].vertRotation[w].transposed() * (R_0_2.transposed()*hexahedronInf[i].rotatedInitialElements[w])
                        - hexahedronInf[i].rotatedInitialElements[w]);*/

                    hexahedronInf[i].plasticRotationYieldThreshold += f_hardeningParam.getValue() * 0.01/*dt*/ * vertRotPlasticRatio;
                    if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
                }
                else if (ratioMaxDistortion > hexahedronInf[i].plasticRotationYieldThreshold && f_debugPlasticMethod.getValue() == 1)
                {
                    Real maxRotPlasticRatio = (ratioMaxDistortion - hexahedronInf[i].plasticRotationYieldThreshold) / (restAngleChange + vertAngleChange);
                    hexahedronInf[i].elementPlasticOffset[w] -= f_plasticRotationCreep.getValue() * 0.5 * maxRotPlasticRatio * (incrementalRotation - I3) * (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter);
                    hexahedronInf[i].plasticRotationYieldThreshold += f_hardeningParam.getValue() * 0.01/*dt*/ * maxRotPlasticRatio;
                    if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
                }
            }
            
            //Volume Constraint Projection
            if (f_preserveElementVolume.getValue() && (volumeRatio >= 1.005 || volumeRatio <= 0.995))
            {
                if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
                for (int w = 0; w < 8; ++w)
                    hexahedronInf[i].elementPlasticOffset[w] += (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter) * (1 - volumeRatio) * 0.02/*dt*/ / NNeighbors;
                //std::cout << "v";
                if (!hexahedronInf[i].needsToUpdateRestMesh) hexahedronInf[i].needsToUpdateRestMesh = true;
            }
        }

    }

    //lower order method: piecewise constant cell center plasticity
    else if (hexahedronInf[i].plasticYieldThreshold > 0) 
    {
        Coord restElementCenter = hexahedronInf[i].rotatedInitialElements[0];
        for (int w = 1; w < 8; w++)
            restElementCenter += hexahedronInf[i].rotatedInitialElements[w];
        restElementCenter = restElementCenter / 8.0;

        Mat33 J = computeCenterJacobian(deformed); /// compute the jacobian (world over reference)
        Mat33 totalF = J * hexahedronInf[i].materialDeformationInverse; /// Total Deformation Gradient F = J (world) * J^-1 (material)
        //Real Fnorm = defaulttype::trace(totalF.multTranspose(totalF)); /// Frobenius norm of totalF
        Real volume = computeElementVolume(hexahedronInf[i].rotatedInitialElements); //deformed element volume
        Real volumeRatio = volume / hexahedronInf[i].restVolume;
        for (int k = 0; k < 8; ++k)
        {
            unsigned int id = _topology->getHexahedron(i)[k]; //vert id	
            unsigned int NNeighbors = _topology->getHexahedraAroundVertex(id).size();
            if (d_debugRendering) debugInfPoint[id] += volumeRatio / NNeighbors;
        }

        if (hexahedronInf[i].plasticYieldThreshold >= hexahedronInf[i].plasticMaxThreshold) {
            std::cout << "Ele " << i << "reached max stretching...";
            hexahedronInf[i].plasticYieldThreshold = 0;
        }
        Mat33 U, V;
        type::Vec<3, Real> Diag;
        helper::Decompose<Real>::SVD(totalF, U, Diag, V);
        Real L2Norm = std::max(std::max(Diag[0] - 1, Diag[1] - 1), Diag[2] - 1);
        if (L2Norm > hexahedronInf[i].plasticYieldThreshold)
        {
            type::Vec<3, Real> plasticDiag = Diag / std::cbrt(Diag[0] * Diag[1] * Diag[2]);
            if (L2Norm > hexahedronInf[i].plasticMaxThreshold)
                L2Norm = hexahedronInf[i].plasticMaxThreshold;
            plastic_ratio = f_plasticCreep.getValue() * (L2Norm - hexahedronInf[i].plasticYieldThreshold) / L2Norm;

            if (plastic_ratio > 0 && !hexahedronInf[i].needsToUpdateRestMesh)
                hexahedronInf[i].needsToUpdateRestMesh = true;
            if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
            if (plastic_ratio < 0) plastic_ratio = 0;

            hexahedronInf[i].plasticYieldThreshold += f_hardeningParam.getValue() * 0.01; //work hardening
            //Update rest state
            //std::cout << "plas r" << plastic_ratio;
            //Mat33 debugUVt = U * V.transposed();
            Mat33 VDVt = V.multDiagonal(plasticDiag) * V.transposed();
            for (int k = 0; k < 8; ++k)
            {
                //hexahedronInf[i].elementPlasticOffset[k] = 0.01 * (debugRotation.transposed()-I3)*(hexahedronInf[i].rotatedInitialElements[k] - restElementCenter);
                hexahedronInf[i].elementPlasticOffset[k] += 0.01 * plastic_ratio * VDVt * (deformed[k] - hexahedronInf[i].rotatedInitialElements[k]);
            }

            //TODO: Check Fracture based on maxYieldThreshold
            //J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);
            //hexahedronInf[i].materialDeformationInverse = J.inverted();
        }
        if (f_preserveElementVolume.getValue() && (volumeRatio >= 1.005 || volumeRatio <= 0.999))
        {
            if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
            for (int k = 0; k < 8; ++k)
                hexahedronInf[i].elementPlasticOffset[k] += (hexahedronInf[i].rotatedInitialElements[k] - restElementCenter) * (1 - volumeRatio) * 0.02/*dt*/;
            if (!hexahedronInf[i].needsToUpdateRestMesh) hexahedronInf[i].needsToUpdateRestMesh = true;
        }
    }
    
    debugData.endEdit();

    // displacement
    Displacement D;
    for(int k=0 ; k<8 ; ++k )
    {
        int indice = k*3;
        for(int j=0 ; j<3 ; ++j )
            D[indice+j] = hexahedronInf[i].rotatedInitialElements[k][j] - deformed[k][j];
    }

    //forces
    Displacement F;

    // compute force on element
    computeForce( F, D, hexahedronInf[i].stiffness );

    for(int j=0; j<8; ++j)
        f[_topology->getHexahedron(i)[j]] += hexahedronInf[i].rotation * Deriv( F[j*3],  F[j*3+1],   F[j*3+2]  );

    hexahedronInfo.endEdit();
}

template<class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::updateRestStatePlasticity()
{
    //Fist do reset, notice that topology may have changed due to cutting/tearing in other threads
    int nP = _topology->getNbPoints();
    restStateOffsets.resize(nP);//reset all plastic offsets to [0,0,0]
    restStateOffsets.fill(Coord(0, 0, 0));
    d_currentVolume = 0;
    needsToUpdateRestMesh = false; //reset the flag to false
    
    helper::WriteAccessor    <Data<VecCoord> > X0w = this->mstate->write(core::VecCoordId::restPosition());
    type::vector<typename HexahedralElastoplasticFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());
    for (size_t i = 0; i < _topology->getNbHexahedra(); ++i)
    {
        for (int w = 0; w < 8; ++w)
        {
            unsigned int id = _topology->getHexahedron(i)[w];
            int numVertNeighbors = _topology->getHexahedraAroundVertex(id).size();
            restStateOffsets[id] += hexahedronInf[i].rotation * hexahedronInf[i].elementPlasticOffset[w] / numVertNeighbors;
            hexahedronInf[i].elementPlasticOffset[w] = Coord(0, 0, 0);//reset the per-element plastic offsets 
        }
    }

    //ignore all fixed points TODO TBA
    for (int i = 0; i < _fixedIndices.size(); i++)
        restStateOffsets[_fixedIndices[i]] = Coord(0, 0, 0);

    //Do the actual updates
    for (int i = 0; i < _topology->getNbPoints(); i++)
    {
        X0w[i] = X0w[i] + restStateOffsets[i];
    }

    const VecCoord& X0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    //update the rotated rest shape per element -- polar method    
    type::Vec<8, Coord> nodes; // coord of the 8 nodes for the ith hex
    for (size_t i = 0; i < _topology->getNbHexahedra(); ++i)
    {
        if (!hexahedronInf[i].needsToUpdateRestMesh)
            continue;

        for (int j = 0; j < 8; ++j)
            nodes[j] = (X0)[_topology->getHexahedron(i)[j]];
        Transformation R_0_1; // Rotation matrix (deformed and displaced Hexahedron/world)
        
        if(method==POLAR) //Polar Method
            computeRotationPolar(R_0_1, nodes);
        else//Large Method
        {
            //average of four horizontal sides
            Coord horizontal = (nodes[1] - nodes[0] + nodes[2] - nodes[3] + nodes[5] - nodes[4] + nodes[6] - nodes[7]) * .25;
            //average of four vertical sides
            Coord vertical = (nodes[3] - nodes[0] + nodes[2] - nodes[1] + nodes[7] - nodes[4] + nodes[6] - nodes[5]) * .25;
            computeRotationLarge(R_0_1, horizontal, vertical);
        }
        
        hexahedronInfo[i].restRotation.transpose(R_0_1);
        for (int j = 0; j < 8; ++j)
        {
            hexahedronInf[i].rotatedInitialElements[j] = R_0_1 * nodes[j];
        }
        Mat33 J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);
        hexahedronInf[i].materialDeformationInverse = J.inverted();

        //Update Element Stiffness
        if (f_updateElementStiffness.getValue())
            computeElementStiffness(hexahedronInf[i].stiffness, hexahedronInf[i].materialMatrix, hexahedronInf[i].rotatedInitialElements);
        //std::cout << "up stiff on E" << i << " . ";

        //update debug rendering data

        hexahedronInf[i].needsToUpdateRestMesh = false;
    }


    hexahedronInfo.endEdit();

}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
//
//template<class DataTypes>
//void HexahedralFEMForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
//{
//    // Build Matrix Block for this ForceField
//    int i,j,n1, n2;
//
//    Index node1, node2;
//
//    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
//    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
//    const type::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = hexahedronInfo.getValue();
//
//    for(Size e=0 ; e<_topology->getNbHexahedra() ; ++e)
//    {
//        const ElementStiffness &Ke = hexahedronInf[e].stiffness;
//
//        // find index of node 1
//        for (n1=0; n1<8; n1++)
//        {
//            node1 = _topology->getHexahedron(e)[n1];
//
//            // find index of node 2
//            for (n2=0; n2<8; n2++)
//            {
//                node2 = _topology->getHexahedron(e)[n2];
//                Mat33 tmp = hexahedronInf[e].rotation.multTranspose( Mat33(Coord(Ke[3*n1+0][3*n2+0],Ke[3*n1+0][3*n2+1],Ke[3*n1+0][3*n2+2]),
//                        Coord(Ke[3*n1+1][3*n2+0],Ke[3*n1+1][3*n2+1],Ke[3*n1+1][3*n2+2]),
//                        Coord(Ke[3*n1+2][3*n2+0],Ke[3*n1+2][3*n2+1],Ke[3*n1+2][3*n2+2])) ) * hexahedronInf[e].rotation;
//                for(i=0; i<3; i++)
//                    for (j=0; j<3; j++)
//                        r.matrix->add(r.offset+3*node1+i, r.offset+3*node2+j, - tmp[i][j]*kFactor);
//            }
//        }
//    }
//}
//
//
//
//
template<class DataTypes>
void HexahedralElastoplasticFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->disableLighting();
    std::vector<sofa::type::RGBAColor> colorVector;
    std::vector<sofa::type::Vector3> vertices;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    for(size_t i = 0 ; i<_topology->getNbHexahedra(); ++i)
    {
        const core::topology::BaseMeshTopology::Hexahedron &t=_topology->getHexahedron(i);

        Index a = t[0];
        Index b = t[1];
        Index d = t[2];
        Index c = t[3];
        Index e = t[4];
        Index f = t[5];
        Index h = t[6];
        Index g = t[7];

        Coord center = (x[a]+x[b]+x[c]+x[d]+x[e]+x[g]+x[f]+x[h])*0.125;
        Real percentage = (Real) 0.15;
        Coord p0 = x[a]-(x[a]-center)*percentage;
        Coord p1 = x[b]-(x[b]-center)*percentage;
        Coord p2 = x[c]-(x[c]-center)*percentage;
        Coord p3 = x[d]-(x[d]-center)*percentage;
        Coord p4 = x[e]-(x[e]-center)*percentage;
        Coord p5 = x[f]-(x[f]-center)*percentage;
        Coord p6 = x[g]-(x[g]-center)*percentage;
        Coord p7 = x[h]-(x[h]-center)*percentage;

        sofa::type::fixed_array<float, 4> color = sofa::type::fixed_array<float, 4>(0.7f, 0.7f, 0.1f, 1.0f);
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p5));
        vertices.push_back(DataTypes::getCPos(p1));
        vertices.push_back(DataTypes::getCPos(p3));
        vertices.push_back(DataTypes::getCPos(p7));

        color = sofa::type::fixed_array<float, 4>(0.7f, 0.0f, 0.0f, 1.0f);
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p1));
        vertices.push_back(DataTypes::getCPos(p0));
        vertices.push_back(DataTypes::getCPos(p2));
        vertices.push_back(DataTypes::getCPos(p3));

        color = sofa::type::fixed_array<float, 4>(0.0f, 0.7f, 0.0f, 1.0f);
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p0));
        vertices.push_back(DataTypes::getCPos(p4));
        vertices.push_back(DataTypes::getCPos(p6));
        vertices.push_back(DataTypes::getCPos(p2));

        color = sofa::type::fixed_array<float, 4>(0.0f, 0.0f, 0.7f, 1.0f);
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p4));
        vertices.push_back(DataTypes::getCPos(p5));
        vertices.push_back(DataTypes::getCPos(p7));
        vertices.push_back(DataTypes::getCPos(p6));

        color = sofa::type::fixed_array<float, 4>(0.1f, 0.7f, 0.7f, 1.0f);
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p7));
        vertices.push_back(DataTypes::getCPos(p3));
        vertices.push_back(DataTypes::getCPos(p2));
        vertices.push_back(DataTypes::getCPos(p6));

        color = sofa::type::fixed_array<float, 4>(0.7f, 0.1f, 0.7f, 1.0f);
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        colorVector.push_back(sofa::type::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p1));
        vertices.push_back(DataTypes::getCPos(p5));
        vertices.push_back(DataTypes::getCPos(p4));
        vertices.push_back(DataTypes::getCPos(p0));
    }
    vparams->drawTool()->drawQuads(vertices,colorVector);
    vparams->drawTool()->restoreLastState();
}

} // namespace sofa::component::forcefield
