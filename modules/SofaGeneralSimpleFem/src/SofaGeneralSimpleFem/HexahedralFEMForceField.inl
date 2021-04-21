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
#include "HexahedralFEMForceField.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/types/RGBAColor.h>
#include <sofa/helper/decompose.h>
#include <cassert>
#include <iostream>
#include <set>

#include <SofaBaseTopology/TopologyData.inl>



// indices ordering  (same as in HexahedronSetTopology):
//
// 	   Y  7---------6
//     ^ /	       /|
//     |/	 Z    / |
//     3----^----2  |
//     |   /	 |  |
//     |  4------|--5
//     | / 	     | /
//     |/	     |/
//     0---------1-->X



namespace sofa::component::forcefield
{

template< class DataTypes>
void HexahedralFEMForceField<DataTypes>::HFFHexahedronHandler::applyCreateFunction(Index hexahedronIndex,
        HexahedronInformation &,
        const core::topology::BaseMeshTopology::Hexahedron &,
        const sofa::helper::vector<Index> &,
        const sofa::helper::vector<double> &)
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
HexahedralFEMForceField<DataTypes>::HexahedralFEMForceField()
    : f_method(initData(&f_method,std::string("large"),"method","\"large\" or \"polar\" displacements"))
    , f_poissonRatio(initData(&f_poissonRatio,(Real)0.45f,"poissonRatio",""))
    , f_youngModulus(initData(&f_youngModulus,(Real)5000,"youngModulus",""))
    , f_plasticMaxThreshold(initData(&f_plasticMaxThreshold, (Real)0.f, "plasticMaxThreshold", "Plastic Max Threshold"))
    , f_plasticYieldThreshold(initData(&f_plasticYieldThreshold, (Real)0.f, "plasticYieldThreshold", "Plastic Yield Threshold"))
    , f_plasticCreep(initData(&f_plasticCreep, (Real)0.1f, "plasticCreep", "Plastic Creep Factor * dt [0,1]. Warning this factor depends on dt."))
    , f_hardeningParam(initData(&f_hardeningParam, (Real)0.1f, "hardeningParam", "Material work hardening parameter"))
    , f_useVertexPlasticity(initData(&f_useVertexPlasticity, false, "useVertexPlasticity", "Use vertex + center deformation for plasticity if true"))
    , f_preserveElementVolume(initData(&f_preserveElementVolume, false, "preserveElementVolume", "Preserve element volume under plasticity deformation if true"))
    , f_updateElementStiffness(initData(&f_updateElementStiffness, false, "updateElementStiffness", "Element stiffness matrix will be updated when necessary"))
    , hexahedronInfo(initData(&hexahedronInfo, "hexahedronInfo", "Internal hexahedron data"))
    /*for debugging*/, f_debugPlasticMethod(initData(&f_debugPlasticMethod, (int)0, "debugPlasticMethod", "debug methods"))
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

    _coef[0][0]= -1;		_coef[0][1]= -1;		_coef[0][2]= -1;
    _coef[1][0]=  1;		_coef[1][1]= -1;		_coef[1][2]= -1;
    _coef[2][0]=  1;		_coef[2][1]=  1;		_coef[2][2]= -1;
    _coef[3][0]= -1;		_coef[3][1]=  1;		_coef[3][2]= -1;
    _coef[4][0]= -1;		_coef[4][1]= -1;		_coef[4][2]=  1;
    _coef[5][0]=  1;		_coef[5][1]= -1;		_coef[5][2]=  1;
    _coef[6][0]=  1;		_coef[6][1]=  1;		_coef[6][2]=  1;
    _coef[7][0]= -1;		_coef[7][1]=  1;		_coef[7][2]=  1;

    hexahedronHandler = new HFFHexahedronHandler(this,&hexahedronInfo);

    f_poissonRatio.setRequired(true);
    f_youngModulus.setRequired(true);
    totalVolume = 0;
    d_currentVolume = 0;
}


template <class DataTypes>
HexahedralFEMForceField<DataTypes>::~HexahedralFEMForceField()
{
    if(hexahedronHandler) delete hexahedronHandler;
}


template <class DataTypes>
void HexahedralFEMForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();

    this->getContext()->get(_topology);
    std::cout << "_topology info" << _topology->getNbHexahedra() << " " << _topology->getNbPoints() << std::endl;
    if (_topology == nullptr)
    {
        //msg_error() << "Object must have a HexahedronSetTopology.";
        SingleLink<HexahedralFEMForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
        msg_info() << "(HexahedralFEMForceField) link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
        if (l_topology) _topology = l_topology.get();
        if (_topology == nullptr || _topology->getNbHexahedra() <= 0)
             serr << "ERROR(HexahedralFEMForceField): object must have hexahedron based topology." << sendl;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    this->reinit(); // compute per-element stiffness matrices and other precomputed values
    sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}


template <class DataTypes>
void HexahedralFEMForceField<DataTypes>::reinit()
{
    if (f_method.getValue() == "large")
        this->setMethod(LARGE);
    else if (f_method.getValue()  == "polar")
        this->setMethod(POLAR);

    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());


    hexahedronInf.resize(_topology->getNbHexahedra());

    for (size_t i=0; i<_topology->getNbHexahedra(); ++i)
    {
        hexahedronHandler->applyCreateFunction(i,hexahedronInf[i],
                _topology->getHexahedron(i),  (const std::vector< Index > )0,
                (const std::vector< double >)0);
    }
    hexahedronInfo.createTopologicalEngine(_topology,hexahedronHandler);
    hexahedronInfo.registerTopologicalData();
    hexahedronInfo.endEdit();

    helper::vector<Real>& debugInfPoint = *(debugData.beginEdit());
    debugInfPoint.resize(_topology->getNbPoints());
    debugData.endEdit();
}


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::addForce (const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& p, const DataVecDeriv& /*v*/)
{
    WDataRefVecDeriv _f = f;
    RDataRefVecCoord _p = p;

    _f.resize(_p.size());


    int nP = _topology->getNbPoints();
    restStateOffsets.resize(nP);//reset all offsets
    restStateOffsets.fill(Coord(0, 0, 0));
    d_currentVolume = 0;

    switch(method)
    {
    case LARGE :
    {

        /*if (f_useVertexPlasticity.getValue() && f_plasticMaxThreshold.getValue() > 0) {
            for (size_t i = 0; i < _topology->getNbHexahedra(); ++i)
            {
                computeF_C(_p, i);
            }
            computeAllF_V();
        }*/

        helper::vector<Real>& debugInfPoint = *(debugData.beginEdit());
        debugInfPoint.resize(_topology->getNbPoints());
        debugInfPoint.fill(0.0);
        debugData.endEdit();
        for(size_t i = 0 ; i<_topology->getNbHexahedra(); ++i)
        {
            accumulateForceLarge( _f, _p, i);
        }
        if (f_plasticMaxThreshold.getValue() > 0 && needsToUpdateRestMesh) updateRestStateLarge();
        break;
    }
    case POLAR :
    {
        helper::vector<Real>& debugInfPoint = *(debugData.beginEdit());
        debugInfPoint.resize(_topology->getNbPoints());
        debugInfPoint.fill(0.0);
        debugData.endEdit();

        for(size_t i = 0 ; i<_topology->getNbHexahedra(); ++i)
        {
            accumulateForcePolar( _f, _p, i);
        }
        if (f_plasticMaxThreshold.getValue() > 0 && needsToUpdateRestMesh) updateRestStatePolar();
        break;
    }
    }

}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::addDForce (const core::MechanicalParams *mparams, DataVecDeriv& v, const DataVecDeriv& x)
{
    helper::WriteAccessor< DataVecDeriv > _v = v;
    const VecCoord& _x = x.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    if( _v.size()!=_x.size() ) _v.resize(_x.size());

    const helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = hexahedronInfo.getValue();

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

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const defaulttype::Vec<8,Coord> &nodes)
{
    // X = n0 (1-x1)(1-x2)(1-x3)/8 + n1 (1+x1)(1-x2)(1-x3)/8 + n2 (1+x1)(1+x2)(1-x3)/8 + n3 (1-x1)(1+x2)(1-x3)/8 + n4 (1-x1)(1-x2)(1+x3)/8 + n5 (1+x1)(1-x2)(1+x3)/8 + n6 (1+x1)(1+x2)(1+x3)/8 + n7 (1-x1)(1+x2)(1+x3)/8
    // J = [ DXi / xj ] = [ (n1i-n0i)(1-x2)(1-x3)/8+(n2i-n3i)(1+x2)(1-x3)/8+(n5i-n4i)(1-x2)(1+x3)/8+(n6i-n7i)(1+x2)(1+x3)/8  (n3i-n0i)(1-x1)(1-x3)/8+(n2i-n1i)(1+x1)(1-x3)/8+(n7i-n4i)(1-x1)(1+x3)/8+(n6i-n5i)(1+x1)(1+x3)/8  (n4i-n0i)(1-x1)(1-x2)/8+(n5i-n1i)(1+x1)(1-x2)/8+(n6i-n2i)(1+x1)(1+x2)/8+(n7i-n3i)(1-x1)(1+x2)/8 ]
    // if it is an orthogonal regular hexahedra: J = [ [ l0/2 0 0 ] [ 0 l1/2 0 ] [ 0 0 l2/2 ] ] det(J) = l0l1l2/8 = vol/8
    //
    // Ke = integralV(BtEBdv) = integral(BtEB det(J) dx1 dx2 dx3)
    // B = DN = [ qx 0  0  ]
    //          [ 0  qy 0  ]
    //          [ 0  0  qz ]
    //          [ qy qx 0  ]
    //          [ 0  qz qy ]
    //          [ qz 0  qx ]
    // with qx = [ dN1/dx ... dN8/dx ] qy = [ dN1/dy ... dN8/dy ] qz = [ dN1/dz ... dN8/dz ]
    // The submatrix Kij of K linking nodes i and j can then be computed as: Kij = integralV(Bjt E Bi det(J) dx1 dx2 dx3)
    // with Bi = part of B related to node i: Bi = [ [ dNi/dx 0 0 ] [ 0 dNi/dy 0 ] [ 0 0 dNi/dz ] [ dNi/dy dNi/dx 0 ] [ 0 dNi/dz dNi/dy ] [ dNi/dz 0 dNi/dx ] ]
    // This integral can be estimated using 8 gauss quadrature points (x1,x2,x3)=(+-1/sqrt(3),+-1/sqrt(3),+-sqrt(3))
    K.fill(0.0);
    Mat33 J; // J[i][j] = dXi/dxj
    Mat33 J_1; // J_1[i][j] = dxi/dXj
    Mat33 J_1t;
    Real detJ = (Real)1.0;
    // check if the hexaedra is a parallelepiped
    Coord lx = nodes[1] - nodes[0];
    Coord ly = nodes[3] - nodes[0];
    Coord lz = nodes[4] - nodes[0];
    bool isParallel = false;
    if ((nodes[3] + lx - nodes[2]).norm() < lx.norm() * 0.001 && (nodes[0] + lz - nodes[4]).norm() < lz.norm() * 0.001 && (nodes[1] + lz - nodes[5]).norm() < lz.norm() * 0.001 && (nodes[2] + lz - nodes[6]).norm() < lz.norm() * 0.001 && (nodes[3] + lz - nodes[7]).norm() < lz.norm() * 0.001)
    {
        isParallel = true;
        for (int c = 0; c < 3; ++c)
        {
            J[c][0] = lx[c] / 2;
            J[c][1] = ly[c] / 2;
            J[c][2] = lz[c] / 2;
        }
        detJ = defaulttype::determinant(J);
        J_1.invert(J);
        J_1t.transpose(J_1);
    }
    const Real U = M[0][0];
    const Real V = M[0][1];
    const Real W = M[3][3];
    const double inv_sqrt3 = 1.0 / sqrt(3.0);
    for (int gx1 = -1; gx1 <= 1; gx1 += 2)
        for (int gx2 = -1; gx2 <= 1; gx2 += 2)
            for (int gx3 = -1; gx3 <= 1; gx3 += 2)
            {
                double x1 = gx1 * inv_sqrt3;
                double x2 = gx2 * inv_sqrt3;
                double x3 = gx3 * inv_sqrt3;
                // compute jacobian matrix
                //Mat33 J; // J[i][j] = dXi/dxj
                //Mat33 J_1; // J_1[i][j] = dxi/dXj
                if (!isParallel)
        
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        J[c][0] = (Real)((nodes[1][c] - nodes[0][c]) * (1 - x2) * (1 - x3) / 8 + (nodes[2][c] - nodes[3][c]) * (1 + x2) * (1 - x3) / 8 + (nodes[5][c] - nodes[4][c]) * (1 - x2) * (1 + x3) / 8 + (nodes[6][c] - nodes[7][c]) * (1 + x2) * (1 + x3) / 8);
                        J[c][1] = (Real)((nodes[3][c] - nodes[0][c]) * (1 - x1) * (1 - x3) / 8 + (nodes[2][c] - nodes[1][c]) * (1 + x1) * (1 - x3) / 8 + (nodes[7][c] - nodes[4][c]) * (1 - x1) * (1 + x3) / 8 + (nodes[6][c] - nodes[5][c]) * (1 + x1) * (1 + x3) / 8);
                        J[c][2] = (Real)((nodes[4][c] - nodes[0][c]) * (1 - x1) * (1 - x2) / 8 + (nodes[5][c] - nodes[1][c]) * (1 + x1) * (1 - x2) / 8 + (nodes[6][c] - nodes[2][c]) * (1 + x1) * (1 + x2) / 8 + (nodes[7][c] - nodes[3][c]) * (1 - x1) * (1 + x2) / 8);
                    }
                    detJ = defaulttype::determinant(J);
                    J_1.invert(J);
                    J_1t.transpose(J_1);
                }
                Real qx[8];
                Real qy[8];
                Real qz[8];
                for (int i = 0; i < 8; ++i)
                {
                    // Ni = 1/8 (1+_coef[i][0]x1)(1+_coef[i][1]x2)(1+_coef[i][2]x3)
                    // qxi = dNi/dx = dNi/dx1 dx1/dx + dNi/dx2 dx2/dx + dNi/dx3 dx3/dx
                    Real dNi_dx1 = (Real)((_coef[i][0]) * (1 + _coef[i][1] * x2) * (1 + _coef[i][2] * x3) / 8.0);
                    Real dNi_dx2 = (Real)((1 + _coef[i][0] * x1) * (_coef[i][1]) * (1 + _coef[i][2] * x3) / 8.0);
                    Real dNi_dx3 = (Real)((1 + _coef[i][0] * x1) * (1 + _coef[i][1] * x2) * (_coef[i][2]) / 8.0);
                    qx[i] = dNi_dx1 * J_1[0][0] + dNi_dx2 * J_1[1][0] + dNi_dx3 * J_1[2][0];
                    qy[i] = dNi_dx1 * J_1[0][1] + dNi_dx2 * J_1[1][1] + dNi_dx3 * J_1[2][1];
                    qz[i] = dNi_dx1 * J_1[0][2] + dNi_dx2 * J_1[1][2] + dNi_dx3 * J_1[2][2];
                }
                for (int i = 0; i < 8; ++i)
                {
                    defaulttype::Mat<6, 3, Real> MBi;
                    MBi[0][0] = U * qx[i]; MBi[0][1] = V * qy[i]; MBi[0][2] = V * qz[i];
                    MBi[1][0] = V * qx[i]; MBi[1][1] = U * qy[i]; MBi[1][2] = V * qz[i];
                    MBi[2][0] = V * qx[i]; MBi[2][1] = V * qy[i]; MBi[2][2] = U * qz[i];
                    MBi[3][0] = W * qy[i]; MBi[3][1] = W * qx[i]; MBi[3][2] = (Real)0;
                    MBi[4][0] = (Real)0;   MBi[4][1] = W * qz[i]; MBi[4][2] = W * qy[i];
                    MBi[5][0] = W * qz[i]; MBi[5][1] = (Real)0;   MBi[5][2] = W * qx[i];
                    for (int j = i; j < 8; ++j)
                    {
                        Mat33 k; // k = BjtMBi
                        k[0][0] = qx[j] * MBi[0][0] + qy[j] * MBi[3][0] + qz[j] * MBi[5][0];
                        k[0][1] = qx[j] * MBi[0][1] + qy[j] * MBi[3][1] /*+ qz[j]*MBi[5][1]*/;
                        k[0][2] = qx[j] * MBi[0][2] /*+ qy[j]*MBi[3][2]*/ + qz[j] * MBi[5][2];
                        k[1][0] = qy[j] * MBi[1][0] + qx[j] * MBi[3][0] /*+ qz[j]*MBi[4][0]*/;
                        k[1][1] = qy[j] * MBi[1][1] + qx[j] * MBi[3][1] + qz[j] * MBi[4][1];
                        k[1][2] = qy[j] * MBi[1][2] /*+ qx[j]*MBi[3][2]*/ + qz[j] * MBi[4][2];
                        k[2][0] = qz[j] * MBi[2][0] /*+ qy[j]*MBi[4][0]*/ + qx[j] * MBi[5][0];
                        k[2][1] = qz[j] * MBi[2][1] + qy[j] * MBi[4][1] /*+ qx[j]*MBi[5][1]*/;
                        k[2][2] = qz[j] * MBi[2][2] + qy[j] * MBi[4][2] + qx[j] * MBi[5][2];
                        k *= detJ;
                        for (int m = 0; m < 3; ++m)
                            for (int l = 0; l < 3; ++l)
                            {
                                K[i * 3 + m][j * 3 + l] += k[l][m];
                            }
                    }
                }
            }
    for (int i = 0; i < 24; ++i) {
        for (int j = i + 1; j < 24; ++j)
        {
            K[j][i] = K[i][j];
        }
    }

}




template<class DataTypes>
typename HexahedralFEMForceField<DataTypes>::Mat33 HexahedralFEMForceField<DataTypes>::integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  )
{
    Mat33 K;

    Real t1 = J_1[0][0]*J_1[0][0];
    Real t2 = t1*signx0;
    Real t3 = (Real)(signy0*signz0);
    Real t4 = t2*t3;
    Real t5 = w*signx1;
    Real t6 = (Real)(signy1*signz1);
    Real t7 = t5*t6;
    Real t10 = t1*signy0;
    Real t12 = w*signy1;
    Real t13 = t12*signz1;
    Real t16 = t2*signz0;
    Real t17 = u*signx1;
    Real t18 = t17*signz1;
    Real t21 = t17*t6;
    Real t24 = t2*signy0;
    Real t25 = t17*signy1;
    Real t28 = t5*signy1;
    Real t32 = w*signz1;
    Real t37 = t5*signz1;
    Real t43 = J_1[0][0]*signx0;
    Real t45 = v*J_1[1][1];
    Real t49 = J_1[0][0]*signy0;
    Real t50 = t49*signz0;
    Real t51 = w*J_1[1][1];
    Real t52 = (Real)(signx1*signz1);
    Real t53 = t51*t52;
    Real t56 = t45*signy1;
    Real t64 = v*J_1[2][2];
    Real t68 = w*J_1[2][2];
    Real t69 = (Real)(signx1*signy1);
    Real t70 = t68*t69;
    Real t73 = t64*signz1;
    Real t81 = J_1[1][1]*signy0;
    Real t83 = v*J_1[0][0];
    Real t87 = J_1[1][1]*signx0;
    Real t88 = t87*signz0;
    Real t89 = w*J_1[0][0];
    Real t90 = t89*t6;
    Real t93 = t83*signx1;
    Real t100 = J_1[1][1]*J_1[1][1];
    Real t101 = t100*signx0;
    Real t102 = t101*t3;
    Real t110 = t100*signy0;
    Real t111 = t110*signz0;
    Real t112 = u*signy1;
    Real t113 = t112*signz1;
    Real t116 = t101*signy0;
    Real t144 = J_1[2][2]*signy0;
    Real t149 = J_1[2][2]*signx0;
    Real t150 = t149*signy0;
    Real t153 = J_1[2][2]*signz0;
    Real t172 = J_1[2][2]*J_1[2][2];
    Real t173 = t172*signx0;
    Real t174 = t173*t3;
    Real t177 = t173*signz0;
    Real t180 = t172*signy0;
    Real t181 = t180*signz0;
    K[0][0] = (float)(t4*t7/36.0+t10*signz0*t13/12.0+t16*t18/24.0+t4*t21/72.0+
            t24*t25/24.0+t24*t28/24.0+t1*signz0*t32/8.0+t10*t12/8.0+t16*t37/24.0+t2*t17/8.0);
    K[0][1] = (float)(t43*signz0*t45*t6/24.0+t50*t53/24.0+t43*t56/8.0+t49*t51*
            signx1/8.0);
    K[0][2] = (float)(t43*signy0*t64*t6/24.0+t50*t70/24.0+t43*t73/8.0+J_1[0][0]*signz0
            *t68*signx1/8.0);
    K[1][0] = (float)(t81*signz0*t83*t52/24.0+t88*t90/24.0+t81*t93/8.0+t87*t89*
            signy1/8.0);
    K[1][1] = (float)(t102*t7/36.0+t102*t21/72.0+t101*signz0*t37/12.0+t111*t113
            /24.0+t116*t28/24.0+t100*signz0*t32/8.0+t111*t13/24.0+t116*t25/24.0+t110*t112/
            8.0+t101*t5/8.0);
    K[1][2] = (float)(t87*signy0*t64*t52/24.0+t88*t70/24.0+t81*t73/8.0+J_1[1][1]*
            signz0*t68*signy1/8.0);
    K[2][0] = (float)(t144*signz0*t83*t69/24.0+t150*t90/24.0+t153*t93/8.0+t149*
            t89*signz1/8.0);
    K[2][1] = (float)(t149*signz0*t45*t69/24.0+t150*t53/24.0+t153*t56/8.0+t144*
            t51*signz1/8.0);
    K[2][2] = (float)(t174*t7/36.0+t177*t37/24.0+t181*t13/24.0+t174*t21/72.0+
            t173*signy0*t28/12.0+t180*t12/8.0+t181*t113/24.0+t177*t18/24.0+t172*signz0*u*
            signz1/8.0+t173*t5/8.0);

    return K /*/(J_1[0][0]*J_1[1][1]*J_1[2][2])*/;
}


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio)
{
    // Hooke's Law in Stiffness Form; This is matrix is the "D" in the paper, sig = D * eps
    m[0][0] = m[1][1] = m[2][2] = 1;
    m[0][1] = m[0][2] = m[1][0]= m[1][2] = m[2][0] =  m[2][1] = (Real)(poissonRatio/(1-poissonRatio));
    m[0][3] = m[0][4] =	m[0][5] = 0;
    m[1][3] = m[1][4] =	m[1][5] = 0;
    m[2][3] = m[2][4] =	m[2][5] = 0;
    m[3][0] = m[3][1] = m[3][2] = m[3][4] =	m[3][5] = 0;
    m[4][0] = m[4][1] = m[4][2] = m[4][3] =	m[4][5] = 0;
    m[5][0] = m[5][1] = m[5][2] = m[5][3] =	m[5][4] = 0;
    m[3][3] = m[4][4] = m[5][5] = (Real)((1-2*poissonRatio)/(2*(1-poissonRatio)));
    m *= (Real)((youngModulus*(1-poissonRatio))/((1+poissonRatio)*(1-2*poissonRatio)));
}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K )
{
    F = K*Depl;
}
template<class DataTypes>
typename HexahedralFEMForceField<DataTypes>::Mat33 HexahedralFEMForceField<DataTypes>::computeJacobian(const helper::fixed_array<Coord, 8>& coords, Real x, Real y, Real z)
{
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
typename HexahedralFEMForceField<DataTypes>::Mat33 HexahedralFEMForceField<DataTypes>::computeCenterJacobian(const helper::fixed_array<Coord, 8>& coords)
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
typename HexahedralFEMForceField<DataTypes>::Real HexahedralFEMForceField<DataTypes>::computeElementVolume(const helper::fixed_array<Coord, 8>& coords)
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

    return volume * -0.125;
    /*verts[0][0][0][0] = coords[0][0]; verts[1][0][0][0] = coords[0][1]; verts[2][0][0][0] = coords[0][2];
    verts[0][1][0][0] = coords[1][0]; verts[1][1][0][0] = coords[1][1]; verts[2][1][0][0] = coords[1][2];
    verts[0][1][1][0] = coords[2][0]; verts[1][1][1][0] = coords[2][1]; verts[2][1][1][0] = coords[2][2];
    verts[0][1][1][1] = coords[2][0]; verts[1][1][1][0] = coords[2][1]; verts[2][1][1][0] = coords[2][2];*/
}
template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeF_C(RDataRefVecCoord& p, int i)
{
    defaulttype::Vec<8, Coord> nodes;
    for (int w = 0; w < 8; ++w)
        nodes[w] = p[_topology->getHexahedron(i)[w]];
    Coord horizontal;
    horizontal = (nodes[1] - nodes[0] + nodes[2] - nodes[3] + nodes[5] - nodes[4] + nodes[6] - nodes[7]) * .25;
    Coord vertical;
    vertical = (nodes[3] - nodes[0] + nodes[2] - nodes[1] + nodes[7] - nodes[4] + nodes[6] - nodes[5]) * .25;
    Transformation R_0_2; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationLarge(R_0_2, horizontal, vertical);
    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());
    //hexahedronInf[i].rotation.transpose(R_0_2);
    // positions of the deformed and displaced Hexahedre in its frame
    defaulttype::Vec<8, Coord> deformed;
    for (int w = 0; w < 8; ++w)
        deformed[w] = R_0_2 * nodes[w];
    hexahedronInf[i].F_C = computeCenterJacobian(deformed) * hexahedronInf[i].materialDeformationInverse;
    hexahedronInfo.endEdit();
}
template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::updateRestStateLarge()
{
    needsToUpdateRestMesh = false;
    helper::WriteAccessor    <Data<VecCoord> > X0w = this->mstate->write(core::VecCoordId::restPosition());
    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());
    for (size_t i = 0; i < _topology->getNbHexahedra(); ++i)
    {
        for (int w = 0; w < 8; ++w)
        {
            unsigned int id = _topology->getHexahedron(i)[w];
            int numVertNeighbors = _topology->getHexahedraAroundVertex(id).size();
            restStateOffsets[id] += hexahedronInf[i].rotation * hexahedronInf[i].elementPlasticOffset[w] / numVertNeighbors;
            hexahedronInf[i].elementPlasticOffset[w] = Coord(0, 0, 0);//reset plastic offsets after been used
        }
    }
    //ignore all fixed points
    for (int i = 0; i < _fixedIndices.size(); i++)
        restStateOffsets[_fixedIndices[i]] = Coord(0, 0, 0);
    //Do the actual updates
    for (int i = 0; i < _topology->getNbPoints(); i++)
    {
        X0w[i] = X0w[i] + restStateOffsets[i];
    }


    const VecCoord& X0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    //update the rotated rest shape per element -- large method    
    defaulttype::Vec<8, Coord> nodes; // coord of the 8 nodes for the ith hex
    for (size_t i = 0; i < _topology->getNbHexahedra(); ++i)
    {
        for (int w = 0; w < 8; ++w) {
            unsigned int id = _topology->getHexahedron(i)[w];
            nodes[w] = (X0)[id];
        }
        //average of four horizontal sides
        Coord horizontal = (nodes[1] - nodes[0] + nodes[2] - nodes[3] + nodes[5] - nodes[4] + nodes[6] - nodes[7]) * .25;
        //average of four vertical sides
        Coord vertical = (nodes[3] - nodes[0] + nodes[2] - nodes[1] + nodes[7] - nodes[4] + nodes[6] - nodes[5]) * .25;
        Transformation R_0_1; //3x3 matrix for rigid transformation
        computeRotationLarge(R_0_1, horizontal, vertical);
        for (int w = 0; w < 8; ++w) {
            hexahedronInf[i].rotatedInitialElements[w] = R_0_1 * nodes[w];
        }
        Mat33 J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);
        hexahedronInf[i].materialDeformationInverse = J.inverted();
    }
    hexahedronInfo.endEdit();
}
template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeAllF_V()
{

    int nP = _topology->getNbPoints();
    F_V.resize(0);//reset all F_V
    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());
    //////Shankradra;Phong approach
    //for (int i = 0; i < _topology->getNbPoints(); i++)
    //{
    //    sofa::helper::vector<Index> hexArr = _topology->getHexahedraAroundVertex(i);
    //    Mat33 F;
    //    int nH = hexArr.size();
    //    for (int j = 0; j < nH; j++)
    //    {
    //        F += hexahedronInf[hexArr[j]].F_C / nH /*TODO Rj*/;
    //    }
    //    F_V.push_back(F);
    //}

    ///////Tri-linear evaluation

}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
////////////// large displacements method


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::initLarge(const int i)
{
    // Rotation matrix (initial Hexahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    const VecCoord& X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    defaulttype::Vec<8,Coord> nodes; // coord of the 8 nodes for the ith hex
    for(int w=0; w<8; ++w)
        nodes[w] = (X0)[_topology->getHexahedron(i)[w]];


    Coord horizontal; //average of four horizontal sides
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical; //average of four vertical sides
    vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    Transformation R_0_1; //3x3 matrix for rigid transformations like rotations
    computeRotationLarge( R_0_1, horizontal,vertical);


    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    for (int w = 0; w < 8; ++w) {
        hexahedronInf[i].rotatedInitialElements[w] = R_0_1 * nodes[w];
        //std::cout << "Hex " << i << " vert " << w << " : "<<hexahedronInf[i].rotatedInitialElements[w] << "\n";
    }

    sofa::component::projectiveconstraintset::FixedConstraint<DataTypes>* _fixedConstraint;
    this->getContext()->get(_fixedConstraint);
    if (_fixedConstraint) _fixedIndices = _fixedConstraint->d_indices.getValue();
    if (f_plasticMaxThreshold.getValue() > 0) ///initialize the plastic params per element
    {
        hexahedronInf[i].plasticYieldThreshold = this->f_plasticYieldThreshold.getValue();
        hexahedronInf[i].plasticMaxThreshold = this->f_plasticMaxThreshold.getValue();
        Mat33 J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);
        hexahedronInf[i].restVolume = defaulttype::determinant(J);
        this->totalVolume += hexahedronInf[i].restVolume;
        hexahedronInf[i].materialDeformationInverse = J.inverted(); ///initialize the per element material deformation
    }

    computeMaterialStiffness( hexahedronInf[i].materialMatrix, f_youngModulus.getValue(), f_poissonRatio.getValue() );
    computeElementStiffness(hexahedronInf[i].stiffness, hexahedronInf[i].materialMatrix, hexahedronInf[i].rotatedInitialElements);

    hexahedronInfo.endEdit();
}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey)
{
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second
    edgex.normalize();
    edgey.normalize();

    Coord edgez = cross( edgex, edgey );
    edgez.normalize();

    edgey = cross( edgez, edgex );
    edgey.normalize();

    r[0][0] = edgex[0];
    r[0][1] = edgex[1];
    r[0][2] = edgex[2];
    r[1][0] = edgey[0];
    r[1][1] = edgey[1];
    r[1][2] = edgey[2];
    r[2][0] = edgez[0];
    r[2][1] = edgez[1];
    r[2][2] = edgez[2];
}

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::accumulateForceLarge( WDataRefVecDeriv& f, RDataRefVecCoord & p, const int i)
{
    defaulttype::Vec<8,Coord> nodes;
    for(int w=0; w<8; ++w)
        nodes[w] = p[_topology->getHexahedron(i)[w]];

    Coord horizontal;
    horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    Coord vertical;
    vertical = (nodes[3] - nodes[0] + nodes[2] - nodes[1] + nodes[7] - nodes[4] + nodes[6] - nodes[5]) * .25;

    Transformation R_0_2; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationLarge( R_0_2, horizontal,vertical);

    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    hexahedronInf[i].rotation.transpose(R_0_2);//which means hexahedronInf[i].rotation = transpose(R_0_2);

    // positions of the deformed and displaced Hexahedre in its frame
    defaulttype::Vec<8,Coord> deformed;
    Coord deformedCenter;
    for (int w = 0; w < 8; ++w) {
        deformed[w] = R_0_2 * nodes[w];
        deformedCenter += deformed[w];
    }
    deformedCenter = deformedCenter / 8.0;

    // Compute the plasticity
    Real plastic_ratio = .0; ///(refer to gamma in the paper) this ratio describes the mount of deformation is absorbed in a timestep
    helper::vector<Real>& debugInfPoint = *(debugData.beginEdit());
    if (hexahedronInf[i].plasticYieldThreshold > 0 && f_useVertexPlasticity.getValue())
    {
        Coord restElementCenter = hexahedronInf[i].rotatedInitialElements[0];
        for (int w = 1; w < 8; w++)
            restElementCenter += hexahedronInf[i].rotatedInitialElements[w];
        restElementCenter = restElementCenter / 8.0;
        hexahedronInf[i].F_C = computeCenterJacobian(deformed) * hexahedronInf[i].materialDeformationInverse;

        //Real volumeRatio = defaulttype::determinant(computeCenterJacobian(hexahedronInf[i].rotatedInitialElements)) / hexahedronInf[i].restVolume;//volumeRatio deformed over init
        Real volume = computeElementVolume(hexahedronInf[i].rotatedInitialElements);
        d_currentVolume += volume;
        Real volumeRatio = volume / hexahedronInf[i].restVolume;
        if (volumeRatio < 0) {
            volumeRatio = 1; std::cout << "inverted elem " << i << "...";
        }

        //Mat33 UC, VC, centerF = hexahedronInf[i].F_C;
        //defaulttype::Vec<3, Real> DiagC;
        //helper::Decompose<Real>::SVD(centerF, UC, DiagC, VC);
        //Real CenterNorm = std::max(std::max(DiagC[0], DiagC[1]), DiagC[2]);
        ////std::cout << "hexahedronInf[i].F_C after " << i << "  " << hexahedronInf[i].F_C << "\n";
        //if (CenterNorm > 1 + hexahedronInf[i].plasticYieldThreshold)
        //{  
        //    //std::cout << "CenterNorm plastic...";
        //    //plastic_ratio = f_plasticCreep.getValue() * (CenterNorm - 1 - hexahedronInf[i].plasticYieldThreshold) / CenterNorm;
        //    //defaulttype::Vec<3, Real> plasticDiag = DiagC / std::cbrt(DiagC[0] * DiagC[1] * DiagC[2]);
        //For each vertex, compute the plastic deformation and store plastic offsets
        //helper::WriteAccessor    <Data<VecCoord> > restStateWrite = this->mstate->write(core::VecCoordId::restPosition());

        for (int w = 0; w < 8; ++w)
        {
            unsigned int id = _topology->getHexahedron(i)[w]; //vert id
            unsigned int NNeighbors = _topology->getHexahedraAroundVertex(id).size();
            debugInfPoint[id] += volumeRatio / NNeighbors;

            //ignore fixed indices
            if (std::find(_fixedIndices.begin(), _fixedIndices.end(), id) != _fixedIndices.end()) {
                continue;
            }

            //Compute vertex deformation
            Mat33 U, V;
            defaulttype::Vec<3, Real> Diag;
            Mat33 FV = computeJacobian(deformed, _coef[w][0], _coef[w][1], _coef[w][2]) * hexahedronInf[i].materialDeformationInverse;
            Mat33 phongF = (FV + hexahedronInf[i].F_C) / 2.0; //Phong Deformation
            helper::Decompose<Real>::SVD(phongF, U, Diag, V);
            Real L2Norm = std::max(std::max(Diag[0], Diag[1]), Diag[2]);
            if (L2Norm > 1 + hexahedronInf[i].plasticYieldThreshold)
            {
                defaulttype::Vec<3, Real> plasticDiag = Diag / std::cbrt(Diag[0] * Diag[1] * Diag[2]);
                if (L2Norm > 1 + 10 * hexahedronInf[i].plasticMaxThreshold) {//TODO fracture or tearing
                    //hexahedronInf[i].plasticYieldThreshold = 0;
                    //L2Norm = std::max(1 + hexahedronInf[i].plasticMaxThreshold, 1 + hexahedronInf[i].plasticYieldThreshold);
                    //std::cout << "fractured on ele "<<i<<"... ";
                }
                else if (L2Norm > 1 + hexahedronInf[i].plasticMaxThreshold) {
                    //L2Norm = 1 + hexahedronInf[i].plasticMaxThreshold;
                    L2Norm = std::max(1 + hexahedronInf[i].plasticMaxThreshold, 1 + hexahedronInf[i].plasticYieldThreshold);
                }

                plastic_ratio = f_plasticCreep.getValue() * (L2Norm - 1 - hexahedronInf[i].plasticYieldThreshold) / L2Norm;

                if (id == 67) {
                    //std::cout << " d=" << defaulttype::determinant(hexahedronInf[i].F_C);
                }//if(plastic_ratio > f_plasticCreep.getValue()/2) std::cout << plastic_ratio;
                hexahedronInf[i].plasticYieldThreshold += f_hardeningParam.getValue() * 0.005 / 8.0 /*8 vertices*/; //hardening

                plasticDiag[0] = std::pow(plasticDiag[0], plastic_ratio);
                plasticDiag[1] = std::pow(plasticDiag[1], plastic_ratio);
                plasticDiag[2] = std::pow(plasticDiag[2], plastic_ratio);
                //std::cout << "plasticDiag of E"<<i<<" "<< plasticDiag << "...   ";
                //0 : Centered transform approach:
                if (f_debugPlasticMethod == 0)
                {
                    hexahedronInf[i].rotatedInitialElements[w] += 0.04 * (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).linearProduct(plasticDiag - Coord(1, 1, 1)) / NNeighbors;
                }
                //Centered offsets Approach
                //hexahedronInf[i].elementPlasticOffset[w] = (plasticDiag - Coord(1,1,1)).linearProduct(_coef[w]) * 0.02/*dt*/ /** std::min(1.0, (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]).norm())*/;
                // Scaled with respect to displacement
                //hexahedronInf[i].elementPlasticOffset[w] *=  std::min(1.0,  (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]).norm());
                //1: Centered transform + vertex Offset, scaled with displacement per vertex approach:
                if (f_debugPlasticMethod == 1)
                {
                    hexahedronInf[i].rotatedInitialElements[w] += /*0.01*(hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).linearProduct(plasticDiag - Coord(1, 1, 1))*/
                        0.02 * plastic_ratio * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]);
                    if (id == 67)
                        std::cout << "d=" << 0.02 * plastic_ratio * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]) << "   ..";
                }

                //2 : Use deformation as offsets with direction correction Approach
                if (f_debugPlasticMethod == 2)
                {
                    hexahedronInf[i].elementPlasticOffset[w] = (plasticDiag - Coord(1, 1, 1)) * 0.008/*dt*/;
                    for (int j = 0; j < 3; j++)
                        //if (plasticDiag[j] < 1)
                        hexahedronInf[i].elementPlasticOffset[w][j] *= _coef[w][j];
                }
                //3 : scaled with actual vertex displacement
                if (f_debugPlasticMethod == 3)//Not being used
                {
                    hexahedronInf[i].elementPlasticOffset[w] = (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).linearProduct(plasticDiag - Coord(1, 1, 1));
                    hexahedronInf[i].elementPlasticOffset[w] *= std::min(1.0, (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]).norm()) * 0.02;
                }
                //Volume Constraint Projection
                if (f_preserveElementVolume.getValue() && (volumeRatio >= 1.001 || volumeRatio <= 0.999))
                {
                    //std::cout << "prj Vol on "<<id<<"..   ";
                    //hexahedronInf[i].elementPlasticOffset[w] += (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter)*(1 - volumeRatio)*0.02/*dt*/;
                    hexahedronInf[i].rotatedInitialElements[w] += (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter) * (1 - volumeRatio) * 0.02 / NNeighbors;
                }

                //if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;               
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
        for (int k = 0; k < 8; ++k)
        {
            unsigned int id = _topology->getHexahedron(i)[k]; //vert id
            unsigned int NNeighbors = _topology->getHexahedraAroundVertex(id).size();
            debugInfPoint[id] += volumeRatio / NNeighbors;
        }
        Mat33 U, V;
        defaulttype::Vec<3, Real> Diag;
        helper::Decompose<Real>::SVD(totalF, U, Diag, V);
        Real L2Norm = std::max(std::max(Diag[0], Diag[1]), Diag[2]);
        if (L2Norm > 1 + hexahedronInf[i].plasticYieldThreshold)
        {
            if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
            defaulttype::Vec<3, Real> plasticDiag = Diag / std::cbrt(Diag[0] * Diag[1] * Diag[2]);
            if (L2Norm > 1 + hexahedronInf[i].plasticMaxThreshold)
                L2Norm = std::max(1 + hexahedronInf[i].plasticMaxThreshold, 1 + hexahedronInf[i].plasticYieldThreshold);
            plastic_ratio = f_plasticCreep.getValue() * (L2Norm - 1 - hexahedronInf[i].plasticYieldThreshold) / L2Norm;
            hexahedronInf[i].plasticYieldThreshold += f_hardeningParam.getValue() * 0.005 /*dt*/; //work hardening
            /*plasticDiag[0] = std::pow(plasticDiag[0], plastic_ratio)-1;
            plasticDiag[1] = std::pow(plasticDiag[1], plastic_ratio)-1;
            plasticDiag[2] = std::pow(plasticDiag[2], plastic_ratio)-1;*/
            //Update rest state
            for (int k = 0; k < 8; ++k)
            {
                //hexahedronInf[i].rotatedInitialElements[k] += 0.02 * plastic_ratio * (deformed[k] - hexahedronInf[i].rotatedInitialElements[k]);
                hexahedronInf[i].elementPlasticOffset[k] += 0.02 * plastic_ratio * (deformed[k] - hexahedronInf[i].rotatedInitialElements[k]);

            }
            //if (f_preserveElementVolume.getValue() && volumeRatio != 1)
            //{
            //    for (int w = 0; w < 8; ++w)
            //        hexahedronInf[i].rotatedInitialElements[w] += (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter)*(1 - volumeRatio)*0.02/*dt*/;
            //}
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
void HexahedralFEMForceField<DataTypes>::initPolar(const int i)
{
    const VecCoord& X0=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    defaulttype::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = (X0)[_topology->getHexahedron(i)[j]];

    Transformation R_0_1; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationPolar( R_0_1, nodes );


    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    for(int j=0; j<8; ++j)
    {
        hexahedronInf[i].rotatedInitialElements[j] = R_0_1 * nodes[j];
    }
    sofa::component::projectiveconstraintset::FixedConstraint<DataTypes>* _fixedConstraint;
    this->getContext()->get(_fixedConstraint);
    if (_fixedConstraint) _fixedIndices = _fixedConstraint->d_indices.getValue();
    if (f_plasticMaxThreshold.getValue() > 0) ///initialize the plastic params per element
    {
        hexahedronInf[i].plasticYieldThreshold = this->f_plasticYieldThreshold.getValue();
        hexahedronInf[i].plasticMaxThreshold = this->f_plasticMaxThreshold.getValue();
        hexahedronInf[i].needsToUpdateRestMesh = false;
        Mat33 J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);
        hexahedronInf[i].restVolume = defaulttype::determinant(J);
        this->totalVolume += hexahedronInf[i].restVolume;
        hexahedronInf[i].materialDeformationInverse = J.inverted(); ///initialize the per element material deformation
    }
    computeMaterialStiffness( hexahedronInf[i].materialMatrix, f_youngModulus.getValue(), f_poissonRatio.getValue() );
    computeElementStiffness(hexahedronInf[i].stiffness, hexahedronInf[i].materialMatrix, hexahedronInf[i].rotatedInitialElements);

    hexahedronInfo.endEdit();
}


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::computeRotationPolar( Transformation &r, defaulttype::Vec<8,Coord> &nodes)
{
    Transformation A;
    Coord Edge =(nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
    A[0][0] = Edge[0];
    A[0][1] = Edge[1];
    A[0][2] = Edge[2];
    Edge = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
    A[1][0] = Edge[0];
    A[1][1] = Edge[1];
    A[1][2] = Edge[2];
    Edge = (nodes[4]-nodes[0] + nodes[5]-nodes[1] + nodes[7]-nodes[3] + nodes[6]-nodes[2])*.25;
    A[2][0] = Edge[0];
    A[2][1] = Edge[1];
    A[2][2] = Edge[2];

    Mat33 HT;
    for(int k=0; k<3; ++k)
        for(int j=0; j<3; ++j)
            HT[k][j]=A[k][j];

    helper::Decompose<Real>::polarDecomposition(HT, r);
}


template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::accumulateForcePolar(WDataRefVecDeriv& f, RDataRefVecCoord & p, const int i)
{
    defaulttype::Vec<8,Coord> nodes;
    for(int j=0; j<8; ++j)
        nodes[j] = p[_topology->getHexahedron(i)[j]];


    Transformation R_0_2; // Rotation matrix (deformed and displaced Hexahedron/world)
    computeRotationPolar( R_0_2, nodes );

    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());

    hexahedronInf[i].rotation.transpose( R_0_2 );

    // positions of the deformed and displaced Hexahedre in its frame
    defaulttype::Vec<8,Coord> deformed;
    Coord deformedCenter;
    for(int j=0; j<8; ++j) {
        deformed[j] = R_0_2 * nodes[j];
    deformedCenter += deformed[j];
}
deformedCenter = deformedCenter / 8.0; //averaging the 8 verts position to get the center
// Compute the plasticity
helper::vector<Real>& debugInfPoint = *(debugData.beginEdit());
Real plastic_ratio = .0; ///(refer to gamma in the paper) this ratio describes the mount of deformation is absorbed in a timestep
if (hexahedronInf[i].plasticYieldThreshold > 0 && f_useVertexPlasticity.getValue())
{
    Coord restElementCenter = hexahedronInf[i].rotatedInitialElements[0];
    for (int w = 1; w < 8; w++)
        restElementCenter += hexahedronInf[i].rotatedInitialElements[w];
    restElementCenter = restElementCenter / 8.0;
    Real totalDisp = 0;
    for (int j = 0; j < 8; j++)
        totalDisp += (deformed[j] - hexahedronInf[i].rotatedInitialElements[j]).norm();
    hexahedronInf[i].F_C = computeCenterJacobian(deformed) * hexahedronInf[i].materialDeformationInverse;
    /*if (i == 0) {
        Mat33 U, V;
        defaulttype::Vec<3, Real> Diag;
        helper::Decompose<Real>::SVD(hexahedronInf[i].F_C, U, Diag, V);
        std::cout << "FC @ E" << i << "=" << Diag << std::endl;
    }*/
    Real volume = computeElementVolume(hexahedronInf[i].rotatedInitialElements);
    d_currentVolume += volume;
    Real volumeRatio = volume / hexahedronInf[i].restVolume;
    //std::cout << "P_v_R: " << volumeRatio << "... ";
    if (volumeRatio < 0) { volumeRatio = 1; std::cout << "inverted elem " << i << "..."; }//TODO handle inverted volume?
    //For each vertex, compute the plastic deformation and update rest space
    for (int w = 0; w < 8; ++w)
    {
        unsigned int id = _topology->getHexahedron(i)[w];
        unsigned int NNeighbors = _topology->getHexahedraAroundVertex(id).size();
        debugInfPoint[id] += volumeRatio / NNeighbors;
        //ignore fixed indices	
        if (std::find(_fixedIndices.begin(), _fixedIndices.end(), id) != _fixedIndices.end()) {
            continue;
        }
        Mat33 U, V;
        defaulttype::Vec<3, Real> Diag;
        Mat33 FV = computeJacobian(deformed, _coef[w][0], _coef[w][1], _coef[w][2]) * hexahedronInf[i].materialDeformationInverse;

        Mat33 phongF = (FV + hexahedronInf[i].F_C) / 2.0; //Phong Deformation
        helper::Decompose<Real>::SVD(phongF, U, Diag, V);
        Mat33 debugUV = U * V;
        Mat33 debugVtDV = (V.transposed()).multDiagonal(Diag) * V;
        //Diag = defaulttype::diagonal(DiagMat);
        /*if (id == 63 && f_debugPlasticMethod == 0) {
            std::cout << "debugVtDV=" << debugVtDV << std::endl;
        }*/
        //helper::Decompose<Real>::symmetricDiagonalization(phongF, U, Diag);
        Real L2Norm = std::max(std::max(Diag[0], Diag[1]), Diag[2]); // max eigen value
        if (L2Norm > 1 + hexahedronInf[i].plasticYieldThreshold)
        {
            /* if (i == 0) {
                 std::cout << "FV @ v" << id << "=" << Diag << std::endl;
             }*/
            defaulttype::Vec<3, Real> plasticDiag = Diag / std::cbrt(Diag[0] * Diag[1] * Diag[2]); //normalization
            if (L2Norm > 1 + hexahedronInf[i].plasticMaxThreshold)
                L2Norm = std::max(1 + hexahedronInf[i].plasticMaxThreshold, 1 + hexahedronInf[i].plasticYieldThreshold);
            plastic_ratio = f_plasticCreep.getValue() * (L2Norm - 1 - hexahedronInf[i].plasticYieldThreshold) / L2Norm;
            if (plastic_ratio > 0 && !hexahedronInf[i].needsToUpdateRestMesh)
                hexahedronInf[i].needsToUpdateRestMesh = true;
            hexahedronInf[i].plasticYieldThreshold += f_hardeningParam.getValue() * 0.005 / 8.0 /*8 vertices*/; //hardening
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
            if (f_debugPlasticMethod == 0)
            {
                //hexahedronInf[i].rotatedInitialElements[w] += 0.04*(hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).linearProduct(plasticDiag - Coord(1, 1, 1)) / NNeighbors;
                hexahedronInf[i].elementPlasticOffset[w] = 0.04 * (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).linearProduct(plasticDiag - Coord(1, 1, 1));
            }
            //1: Centered transform + vertex disp Offset, scaled with displacement per vertex approach:
            if (f_debugPlasticMethod == 1)
            {
                //Mat33 debugPlasticF = U.multDiagonal(plasticDiag)*V;
                //hexahedronInf[i].rotatedInitialElements[w] += 0.05 * debugPlasticF.inverted() * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]);
                /*hexahedronInf[i].elementPlasticOffset[w] = 0.04*(hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).linearProduct(plasticDiag - Coord(1, 1, 1))
                    + 0.0002*(deformedCenter - restElementCenter);*/
                hexahedronInf[i].elementPlasticOffset[w] = 0.02 * plastic_ratio * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]);
                //hexahedronInf[i].rotatedInitialElements[w] += plastic_ratio * (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]).linearProduct(plasticDiag - Coord(1, 1, 1));
            }
            //2 : Use deformation as offsets with direction correction Approach
            if (f_debugPlasticMethod == 2)
            {
                hexahedronInf[i].elementPlasticOffset[w] = (plasticDiag - Coord(1, 1, 1)) * 0.008/*dt*/;
                for (int j = 0; j < 3; j++)
                    //if (plasticDiag[j] < 1)
                    hexahedronInf[i].elementPlasticOffset[w][j] *= _coef[w][j];
            }
            //3 : scaled with actual vertex displacement
            if (f_debugPlasticMethod == 3)
            {
                hexahedronInf[i].elementPlasticOffset[w] = (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter).linearProduct(plasticDiag - Coord(1, 1, 1));
                hexahedronInf[i].elementPlasticOffset[w] *= std::min(1.0, (deformed[w] - hexahedronInf[i].rotatedInitialElements[w]).norm()) * 0.02;
            }
            if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
            //Mat33 J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);//recompute element center deformation inverse based on rotated rest elements	
            //hexahedronInf[i].materialDeformationInverse = J.inverted();
        }
        //Volume Constraint
        if (f_preserveElementVolume.getValue() && (volumeRatio >= 1.001 || volumeRatio <= 0.999))
        {
            if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
            for (int w = 0; w < 8; ++w)
                if (!hexahedronInf[i].needsToUpdateRestMesh) hexahedronInf[i].needsToUpdateRestMesh = true;
            hexahedronInf[i].elementPlasticOffset[w] += (hexahedronInf[i].rotatedInitialElements[w] - restElementCenter) * (1 - volumeRatio) * 0.02/*dt*/ / NNeighbors;
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
    for (int k = 0; k < 8; ++k)
    {
        unsigned int id = _topology->getHexahedron(i)[k]; //vert id	
        unsigned int NNeighbors = _topology->getHexahedraAroundVertex(id).size();
        debugInfPoint[id] += volumeRatio / NNeighbors;
    }

    Mat33 U, V;
    defaulttype::Vec<3, Real> Diag;
    helper::Decompose<Real>::SVD(totalF, U, Diag, V);
    Real L2Norm = std::max(std::max(Diag[0], Diag[1]), Diag[2]);
    if (L2Norm > 1 + hexahedronInf[i].plasticYieldThreshold)
    {
        defaulttype::Vec<3, Real> plasticDiag = Diag / std::cbrt(Diag[0] * Diag[1] * Diag[2]);
        if (L2Norm > 1 + hexahedronInf[i].plasticMaxThreshold)
            L2Norm = std::max(1 + hexahedronInf[i].plasticMaxThreshold, 1 + hexahedronInf[i].plasticYieldThreshold);
        plastic_ratio = f_plasticCreep.getValue() * (L2Norm - 1 - hexahedronInf[i].plasticYieldThreshold) / L2Norm;
        if (plastic_ratio > 0 && !hexahedronInf[i].needsToUpdateRestMesh)
            hexahedronInf[i].needsToUpdateRestMesh = true;
        if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
        hexahedronInf[i].plasticYieldThreshold += f_hardeningParam.getValue() * 0.01; //work hardening
        //Update rest state
        std::cout << "plas r" << plastic_ratio;
        for (int k = 0; k < 8; ++k)
        {
            //hexahedronInf[i].rotatedInitialElements[k] += 0.02 * plastic_ratio * (deformed[k] - hexahedronInf[i].rotatedInitialElements[k]);
            hexahedronInf[i].elementPlasticOffset[k] += 0.02 * plastic_ratio * (deformed[k] - hexahedronInf[i].rotatedInitialElements[k]);
        }
        //TODO: Check Fracture based on maxYieldThreshold
        //J = computeCenterJacobian(hexahedronInf[i].rotatedInitialElements);
        //hexahedronInf[i].materialDeformationInverse = J.inverted();
    }
    if (f_preserveElementVolume.getValue() && (volumeRatio >= 1.001 || volumeRatio <= 0.999))
    {
        if (!needsToUpdateRestMesh) needsToUpdateRestMesh = true;
        //std::cout << "vol r" << volumeRatio;	
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
void HexahedralFEMForceField<DataTypes>::updateRestStatePolar()
{
    needsToUpdateRestMesh = false;
    helper::WriteAccessor    <Data<VecCoord> > X0w = this->mstate->write(core::VecCoordId::restPosition());
    helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = *(hexahedronInfo.beginEdit());
    for (size_t i = 0; i < _topology->getNbHexahedra(); ++i)
    {
        for (int w = 0; w < 8; ++w)
        {
            unsigned int id = _topology->getHexahedron(i)[w];
            int numVertNeighbors = _topology->getHexahedraAroundVertex(id).size();
            restStateOffsets[id] += hexahedronInf[i].rotation * hexahedronInf[i].elementPlasticOffset[w] / numVertNeighbors;
            hexahedronInf[i].elementPlasticOffset[w] = Coord(0, 0, 0);//reset plastic offsets after been used
        }
    }
    //ignore all fixed points
    for (int i = 0; i < _fixedIndices.size(); i++)
        restStateOffsets[_fixedIndices[i]] = Coord(0, 0, 0);
    //Do the actual updates
    for (int i = 0; i < _topology->getNbPoints(); i++)
    {
        X0w[i] = X0w[i] + restStateOffsets[i];
    }
    const VecCoord& X0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    //update the rotated rest shape per element -- polar method    
    defaulttype::Vec<8, Coord> nodes; // coord of the 8 nodes for the ith hex
    for (size_t i = 0; i < _topology->getNbHexahedra(); ++i)
    {
        if (!hexahedronInf[i].needsToUpdateRestMesh)
            continue;
        for (int j = 0; j < 8; ++j)
            nodes[j] = (X0)[_topology->getHexahedron(i)[j]];
        Transformation R_0_1; // Rotation matrix (deformed and displaced Hexahedron/world)
        computeRotationPolar(R_0_1, nodes);
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
        hexahedronInf[i].needsToUpdateRestMesh = false;
    }
    hexahedronInfo.endEdit();
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    // Build Matrix Block for this ForceField
    int i,j,n1, n2;

    Index node1, node2;

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    const Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    const helper::vector<typename HexahedralFEMForceField<DataTypes>::HexahedronInformation>& hexahedronInf = hexahedronInfo.getValue();

    for(Size e=0 ; e<_topology->getNbHexahedra() ; ++e)
    {
        const ElementStiffness &Ke = hexahedronInf[e].stiffness;  //Mat<24, 24, Real>

        // find index of node 1
        for (n1=0; n1<8; n1++)
        {
            node1 = _topology->getHexahedron(e)[n1];

            // find index of node 2
            for (n2=0; n2<8; n2++)
            {
                node2 = _topology->getHexahedron(e)[n2];
                Mat33 tmp = hexahedronInf[e].rotation.multTranspose( Mat33(Coord(Ke[3*n1+0][3*n2+0],Ke[3*n1+0][3*n2+1],Ke[3*n1+0][3*n2+2]),
                        Coord(Ke[3*n1+1][3*n2+0],Ke[3*n1+1][3*n2+1],Ke[3*n1+1][3*n2+2]),
                        Coord(Ke[3*n1+2][3*n2+0],Ke[3*n1+2][3*n2+1],Ke[3*n1+2][3*n2+2])) ) * hexahedronInf[e].rotation;
                for(i=0; i<3; i++) 
                    for (j=0; j<3; j++) // Add value to the element (using 0-based indices)
                        r.matrix->add(r.offset+3*node1+i, r.offset+3*node2+j, - tmp[i][j]*kFactor);
            }
        }
    }
}




template<class DataTypes>
void HexahedralFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->disableLighting();
    std::vector<sofa::helper::types::RGBAColor> colorVector;
    std::vector<sofa::defaulttype::Vector3> vertices;

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

        sofa::helper::fixed_array<float, 4> color = sofa::helper::fixed_array<float, 4>(0.7f, 0.7f, 0.1f, 1.0f);
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p5));
        vertices.push_back(DataTypes::getCPos(p1));
        vertices.push_back(DataTypes::getCPos(p3));
        vertices.push_back(DataTypes::getCPos(p7));

        color = sofa::helper::fixed_array<float, 4>(0.7f, 0.0f, 0.0f, 1.0f);
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p1));
        vertices.push_back(DataTypes::getCPos(p0));
        vertices.push_back(DataTypes::getCPos(p2));
        vertices.push_back(DataTypes::getCPos(p3));

        color = sofa::helper::fixed_array<float, 4>(0.0f, 0.7f, 0.0f, 1.0f);
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p0));
        vertices.push_back(DataTypes::getCPos(p4));
        vertices.push_back(DataTypes::getCPos(p6));
        vertices.push_back(DataTypes::getCPos(p2));

        color = sofa::helper::fixed_array<float, 4>(0.0f, 0.0f, 0.7f, 1.0f);
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p4));
        vertices.push_back(DataTypes::getCPos(p5));
        vertices.push_back(DataTypes::getCPos(p7));
        vertices.push_back(DataTypes::getCPos(p6));

        color = sofa::helper::fixed_array<float, 4>(0.1f, 0.7f, 0.7f, 1.0f);
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p7));
        vertices.push_back(DataTypes::getCPos(p3));
        vertices.push_back(DataTypes::getCPos(p2));
        vertices.push_back(DataTypes::getCPos(p6));

        color = sofa::helper::fixed_array<float, 4>(0.7f, 0.1f, 0.7f, 1.0f);
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        colorVector.push_back(sofa::helper::types::RGBAColor(color));
        vertices.push_back(DataTypes::getCPos(p1));
        vertices.push_back(DataTypes::getCPos(p5));
        vertices.push_back(DataTypes::getCPos(p4));
        vertices.push_back(DataTypes::getCPos(p0));
    }
    vparams->drawTool()->drawQuads(vertices,colorVector);
    vparams->drawTool()->restoreLastState();
}

} // namespace sofa::component::forcefield
