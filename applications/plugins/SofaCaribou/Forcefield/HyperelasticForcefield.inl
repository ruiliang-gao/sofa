#pragma once

#include <SofaCaribou/config.h>
#include <SofaCaribou/Forcefield/HyperelasticForcefield.h>
#include <SofaCaribou/Forcefield/CaribouForcefield.inl>
#include <SofaCaribou/Topology/CaribouTopology.h>

#include <Eigen/SVD> 

DISABLE_ALL_WARNINGS_BEGIN
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/helper/decompose.h>
DISABLE_ALL_WARNINGS_END

#include <Caribou/Mechanics/Elasticity/Strain.h>
#ifdef CARIBOU_WITH_OPENMP
#include <omp.h>
#endif

namespace SofaCaribou::forcefield {

template <typename Element>
HyperelasticForcefield<Element>::HyperelasticForcefield()
: d_material(initLink(
    "material",
    "Material used to compute the hyperelastic force field."))
, d_enable_multithreading(initData(&d_enable_multithreading,
    false,
    "enable_multithreading",
    "Enable the multithreading computation of the stiffness matrix. Only use this if you have a "
    "very large number of elements, otherwise performance might be worse than single threading."
    "When enabled, use the environment variable OMP_NUM_THREADS=N to use N threads."))
, d_plasticMaxThreshold(initData(&d_plasticMaxThreshold,
    (Real)0.f,
    "plasticMaxThreshold",
    "Maximal yield stress"))
, d_plasticYieldThreshold(initData(&d_plasticYieldThreshold,
    (Real)0.f,
    "plasticYieldThreshold",
    "Plastic yield stress (tensile test)"))
, d_plasticCreep(initData(&d_plasticCreep,
    (Real)0.f,
    "plasticCreep",
    "Plastic Creep"))
, d_hardeningParam(initData(&d_hardeningParam,
    (Real)0.f,
    "hardeningParam",
    "Hardening Parameter"))
, d_preserveVolume(initData(&d_preserveVolume,
    false,
    "preserveVolume",
    "Preserve element Volume"))
, d_fixedIndices(initData(&d_fixedIndices, "fixedIndices", "Indices of the fixed points"))
{
}

template <typename Element>
void HyperelasticForcefield<Element>::init()
{
    using sofa::core::topology::BaseMeshTopology;
    using sofa::core::objectmodel::BaseContext;
    Inherit::init();

    // No material set, try to find one in the current context
    if (not d_material.get()) {
        auto materials = this->getContext()->template getObjects<material::HyperelasticMaterial<DataTypes>>(BaseContext::Local);
        if (materials.empty()) {
            msg_warning() << "Could not find an hyperelastic material in the current context.";
        } else if (materials.size() > 1) {
            msg_warning() << "Multiple materials were found in the context node. "   <<
                             "Please specify which one should be use by explicitly " <<
                             "setting the material's path in the '" << d_material.getName() << "' parameter.";
        } else {
            d_material.set(materials[0]);
            msg_info() << "Automatically found the material '" << d_material.get()->getPathName() << "'.";
        }
    }

    // Compute and store the shape functions and their derivatives for every integration points
    initialize_elements();

    // Assemble the initial stiffness matrix
    assemble_stiffness();
}

template<typename Element>
void HyperelasticForcefield<Element>::addForce(const sofa::core::MechanicalParams *mparams, sofa::core::MultiVecDerivId fId) {
    if (mparams) {
        // Stores the identifier of the x position vector for later use in the stiffness matrix assembly.
        p_X_id = mparams->x();
    }
    Inherit::addForce(mparams, fId);
}

template <typename Element>
void HyperelasticForcefield<Element>::addForce(
    const sofa::core::MechanicalParams* mparams,
    sofa::core::objectmodel::Data<VecDeriv>& d_f,
    const sofa::core::objectmodel::Data<VecCoord>& d_x,
    const sofa::core::objectmodel::Data<VecDeriv>& d_v)
{
    using namespace sofa::core::objectmodel;
    using namespace sofa::helper;

    SOFA_UNUSED(mparams);
    SOFA_UNUSED(d_v);

    if (!this->mstate)
        return;

    const auto material = d_material.get();
    if (!material) {
        return;
    }

    // Update material parameters in case the user changed it
    material->before_update();

    ReadAccessor<Data<VecCoord>> sofa_x = d_x;
    WriteAccessor<Data<VecDeriv>> sofa_f = d_f;

    if (sofa_x.size() != sofa_f.size())
        return;
    const auto nb_nodes = sofa_x.size();
    const auto nb_elements = this->number_of_elements();

    if (nb_nodes == 0 || nb_elements == 0)
        return;

    if (p_elements_quadrature_nodes.size() != nb_elements)
        return;

    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Dimension, Eigen::RowMajor>>    X       (sofa_x.ref().data()->data(),  nb_nodes, Dimension);
    Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, Dimension, Eigen::RowMajor>> forces  (&(sofa_f[0][0]),  nb_nodes, Dimension);

    sofa::helper::WriteAccessor    <Data<VecCoord> > X0w = this->mstate->write(sofa::core::VecCoordId::restPosition());
    //const VecCoord& Xt = this->mstate->read(sofa::core::ConstVecCoordId::position())->getValue();
    
    sofa::helper::AdvancedTimer::stepBegin("HyperelasticForcefield::addForce");

    for (std::size_t element_id = 0; element_id < nb_elements; ++element_id) {

        // Fetch the node indices of the element
        auto node_indices = this->topology()->domain()->element_indices(element_id);

        // Fetch the initial and current positions of the element's nodes
        Matrix<NumberOfNodesPerElement, Dimension> current_nodes_position;
        Matrix<NumberOfNodesPerElement, Dimension> rest_nodes_position;

        //Compute element volume and ratio over its initial volume
        for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            for (int d = 0; d < Dimension; d++)
            {
                rest_nodes_position.row(i)(d) = X0w[node_indices[i]][d] + vertPlasticOffsets[node_indices[i]][d];
            }
        }
        Real volumeRatio = 1.0;
        if (d_preserveVolume.getValue())
        {
            Real currentVolume = computeHexVolume(rest_nodes_position);
            volumeRatio = currentVolume > 0 ? currentVolume / elemInitialVolume[element_id] : 1.0;
            //std::cout << " volumeRatio " << volumeRatio;
        }

        //Compute rest state cell-center position X0c;
        Coord X0c(0, 0, 0);

        //Update the vertices positions with plastic offsets
        for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            current_nodes_position.row(i).noalias() = X.row(node_indices[i]);
            if (std::find(d_fixedIndices.getValue().begin(), d_fixedIndices.getValue().end(), node_indices[i]) != d_fixedIndices.getValue().end()) {
                continue;
            }
            for (int d = 0; d < Dimension; d++)
                current_nodes_position.row(i)(d) -= vertPlasticOffsets[node_indices[i]][d];
            X0c += X0w[node_indices[i]];
        }
        X0c /= (Real)NumberOfNodesPerElement;
        sofa::type::Mat<3, 3, Real> currentRotation;
        if (NumberOfNodesPerElement == 8 && Dimension == 3) //TODO: should check element type -> hex8
        {
            currentRotation = extractHexRotation(current_nodes_position);
            /*if (element_id == 88)
                std::cout << "R " << currentRotation << std::endl;*/
        }
        else
            currentRotation.identity();

        // Compute the nodal forces
        Matrix<NumberOfNodesPerElement, Dimension> nodal_forces;
        nodal_forces.fill(0);
        int gauss_index = 0;
        for (GaussNode &gauss_node : p_elements_quadrature_nodes[element_id]) {

            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto & detJ = gauss_node.jacobian_determinant;

            // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
            const auto & dN_dx = gauss_node.dN_dx;

            // Gauss quadrature node weight
            const auto & w = gauss_node.weight;
            
            // Deformation tensor at gauss node
            Mat33 F = current_nodes_position.transpose() * dN_dx * gauss_node.inversePlasticStrain;
    
            auto J = F.determinant();

            // Right Cauchy-Green strain tensor at gauss node
            Mat33 C = F.transpose() * F ;

            // Second Piola-Kirchhoff stress tensor at gauss node
            Mat33 S;

            Coord Center2Vert = X0w[node_indices[this->H8G2V[gauss_index]]] - X0c;
            
            if (this->d_plasticMaxThreshold.getValue() > 1e-6)
            {               
                //auto& C_p = gauss_node.inversePlasticStrain;
                
                //update stress
                S = material->PK2_stress(J, C);
                Mat33 shiftedS = S;
                shiftedS(0, 0) -= gauss_node.backStress(0);
                shiftedS(1, 1) -= gauss_node.backStress(1);
                shiftedS(2, 2) -= gauss_node.backStress(2);

                //squared shiftedVonMisesStress (3 * J2)
                Real shiftedVonMisesStress =  0.5 * (shiftedS(0, 0) - shiftedS(1, 1)) * (shiftedS(0, 0) - shiftedS(1, 1)) +
                    0.5 * (shiftedS(1, 1) - shiftedS(2, 2)) * (shiftedS(1, 1) - shiftedS(2, 2)) +
                    0.5 * (shiftedS(2, 2) - shiftedS(0, 0)) * (shiftedS(2, 2) - shiftedS(0, 0)) +
                    3 * (shiftedS(0, 1) * shiftedS(0, 1) + shiftedS(1, 2) * shiftedS(1, 2) + shiftedS(0, 2) * shiftedS(0, 2));
                /*Real vonMisesStress = 0.5 * (S(0, 0) - S(1, 1)) * (S(0, 0) - S(1, 1)) +
                    0.5 * (S(1, 1) - S(2, 2)) * (S(1, 1) - S(2, 2)) +
                    0.5 * (S(2, 2) - S(0, 0)) * (S(2, 2) - S(0, 0)) +
                    3 * (S(0, 1) * S(0, 1) + S(1, 2) * S(1, 2) + S(0, 2) * S(0, 2));*/

                //if(node_indices[this->H8G2V[gauss_index]] == 49) std::cout << "vonMises " << vonMisesStress << std::endl;
                if (shiftedVonMisesStress > gauss_node.yieldStress * gauss_node.yieldStress)
                {
                    ///TODO update plasticity using return mapping algorithm   
                    Vec3 D;
                    sofa::type::Mat<3, 3, Real> sF;//Deformation gradient F in SOFA matrix type
                    for (int r = 0; r < 3; ++r)
                    {
                        for (int c = 0; c < 3; ++c)
                        {
                            sF[r][c] = F(r, c);
                        }
                    }
                    //using SVD method from SOFA
                    //Compute vertex deformation
                    sofa::type::Mat<3, 3, Real> U, V;
                    sofa::type::Vec<3, Real> Diag;
                    sofa::helper::Decompose<Real>::SVD_stable(sF, U, Diag, V); 
                    //Diag = currentRotation * Diag;
                    if (element_id == 88) std::cout << " currentDiag " << Diag<<std::endl;
                    //Eigen::JacobiSVD<Mat33> svd(F);
                    
                    ///Note the eigenvalues() method from Eigen is not stable as it will suddenly switch the orders!!

                    /*std::cout << "eigvalD : "  << Diag[0] << ", " << Diag[1] << ", " << Diag[2] << std::endl;*/
                    Diag = Diag / std::cbrt(Diag[0] * Diag[1] * Diag[2]); // normalization to gurantee the det(Dp) == 1
                    
                    Real gamma = this->d_plasticCreep.getValue() * 0.01/*this->getContext()->getDt()*/ * ( std::sqrt(shiftedVonMisesStress) - gauss_node.yieldStress) / std::sqrt(shiftedVonMisesStress);

                    Vec3 F_p;
                    
                    //gauss_node.inversePlasticStrain = gauss_node.inversePlasticStrain * D_p; //matrix-vector multplication, but triggers C2338 YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES???
                    //Eigen::Matrix<Real, 3, 1> Mat_D_p;
                    F_p(0) = std::pow(Diag[0], -1 * gamma);
                    F_p(1) = std::pow(Diag[1], -1 * gamma);
                    F_p(2) = std::pow(Diag[2], -1 * gamma);

                    //gauss_node.inversePlasticStrain *= Mat_D_p;
                    if (!_doNotUpdateRestMesh)//if method -> relax the plastic strain to rest mesh
                        gauss_node.inversePlasticStrain = Mat33::Identity(); //reset inversePlastic matrix

                    gauss_node.inversePlasticStrain.row(0) *= F_p(0);
                    gauss_node.inversePlasticStrain.row(1) *= F_p(1);
                    gauss_node.inversePlasticStrain.row(2) *= F_p(2);
                    
                    //std::cout<<"element_id="<<element_id;
                    //std::cout << "inversePlasticStrain : "  << gauss_node.inversePlasticStrain << "...\n ";
                    //F *= F_p;
                    //std::cout << "invPlasticStrain"  << gauss_node.inversePlasticStrain(0,0) << ", " << gauss_node.inversePlasticStrain(1,1) << ", " << gauss_node.inversePlasticStrain(2,2) << "  ";
                    //Ce = F.transpose() * F * gauss_node.inversePlasticStrain;
                    //F *= gauss_node.inversePlasticStrain;

                    //isotropic hardening
                    //gauss_node.yieldStress *= d_hardeningParam.getValue() * 0.01 /*dt*/; 

                    //kinematic hardening: update back stress
                    gauss_node.backStress += S * Vec3(1 - F_p(0), 1 - F_p(1), 1 - F_p(2)) * d_hardeningParam.getValue();
            
                    //if (element_id == 88)
                        //std::cout << "vecC2N" << vecC2N << std::endl;
                    
                    //if (element_id == 19) std::cout << "plsOff" << " " << Coord(plasOffset[0] * vecC2N[0], plasOffset[1] * vecC2N[1], plasOffset[2] * vecC2N[2]) << std::endl;
                    /*if (element_id > 3)*/ //X0w[node_indices[this->H8G2V[gauss_index]]] += Coord(plasOffset[0]*vecC2N[0], plasOffset[1] * vecC2N[1], plasOffset[2] * vecC2N[2]);
                    if (!_doNotUpdateRestMesh) //relaxation on the rest mesh vertex nearest to this gauss node 
                    {
                        //displacement vector
                        Coord vecC2N  = elemInitialRotation[element_id].transposed() * (currentRotation * sofa_x[node_indices[this->H8G2V[gauss_index]]]) - X0w[node_indices[this->H8G2V[gauss_index]]];
                        //Coord vecC2N2 = sofa_x[node_indices[this->H8G2V[gauss_index]]] - X0w[node_indices[this->H8G2V[gauss_index]]];   
                        Coord plasOffset(1 - F_p(0), 1 - F_p(1), 1 - F_p(2));
                        /*if (element_id == 88)
                        {
                            std::cout << "wrongC2N " << vecC2N2 << "\n" << "C2N " << vecC2N << "\n Diag" << Diag 
                                << std::endl;
                        }*/
                        vertPlasticOffsets[node_indices[this->H8G2V[gauss_index]]] += Coord(plasOffset[0] * vecC2N[0], plasOffset[1] * vecC2N[1], plasOffset[2] * vecC2N[2]);
                    }
                    // Fetch the vert indices of the element
                    //auto vert_indices = this->topology()->domain()->element_indices(element_id);
                }

                //Volume preservation
                if (d_preserveVolume.getValue() && (volumeRatio >= 1.01 || volumeRatio <= 0.99))
                {
                    vertPlasticOffsets[node_indices[this->H8G2V[gauss_index]]] += (1 - volumeRatio) * 0.01 * Center2Vert;
                    //std::cout << "vol " << volumeRatio << ".. ";
                }

                gauss_index++;
            
            }


            //compute the PK2_stress after plasticity update
            //Ce = F.transpose() * F;
            S = material->PK2_stress(J, C);

            // Elastic forces w.r.t the gauss node applied on each nodes
            for (size_t i = 0; i < NumberOfNodesPerElement; ++i) {
                const auto dx = dN_dx.row(i).transpose();
                const Vector<Dimension> f_ = (detJ * w) * F*S*dx;
                for (size_t j = 0; j < Dimension; ++j) {
                    nodal_forces(i, j) += f_[j];
                }
            }
        }

        for (size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            for (size_t j = 0; j < Dimension; ++j) {
                sofa_f[node_indices[i]][j] -= nodal_forces(i,j);
            }
        }
    }

    sofa::helper::AdvancedTimer::stepEnd("HyperelasticForcefield::addForce");

    // This is the only I found to detect when a stiffness matrix reassembly is needed for calls to addDForce
    K_is_up_to_date = false;
    eigenvalues_are_up_to_date = false;
}

template <typename Element>
void HyperelasticForcefield<Element>::addDForce(
    const sofa::core::MechanicalParams* mparams,
    sofa::core::objectmodel::Data<VecDeriv>& d_df,
    const sofa::core::objectmodel::Data<VecDeriv>& d_dx)
{
    using namespace sofa::core::objectmodel;

    if (not K_is_up_to_date) {
        assemble_stiffness();
    }

    auto kFactor = static_cast<Real> (mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()));
    sofa::helper::ReadAccessor<Data<VecDeriv>> sofa_dx = d_dx;
    sofa::helper::WriteAccessor<Data<VecDeriv>> sofa_df = d_df;

    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> DX   (&(sofa_dx[0][0]), sofa_dx.size()*3);
    Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1>>       DF   (&(sofa_df[0][0]), sofa_df.size()*3);

    sofa::helper::AdvancedTimer::stepBegin("HyperelasticForcefield::addDForce");

    for (int k = 0; k < p_K.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<Real>::InnerIterator it(p_K, k); it; ++it) {
            const auto i = it.row();
            const auto j = it.col();
            const auto v = -1 * it.value() * kFactor;
            if (i != j) {
                DF[i] += v*DX[j];
                DF[j] += v*DX[i];
            } else {
                DF[i] += v*DX[i];
            }
        }
    }

    sofa::helper::AdvancedTimer::stepEnd("HyperelasticForcefield::addDForce");
}

template <typename Element>
void HyperelasticForcefield<Element>::addKToMatrix(
    sofa::defaulttype::BaseMatrix * matrix,
    SReal kFact, unsigned int & offset)
{
    if (not K_is_up_to_date) {
        assemble_stiffness();
    }

    sofa::helper::AdvancedTimer::stepBegin("HyperelasticForcefield::addKToMatrix");

    // K is symmetric, so we only stored "one side" of the matrix.
    // But to accelerate the computation, coefficients were not
    // stored only in the upper or lower triangular part, but instead
    // in whatever triangular part (upper or lower) the first node
    // index of the element was. This means that a coefficient (i,j)
    // might be in the lower triangular part, while (k,l) is in the
    // upper triangular part. But no coefficient will be both in the
    // lower AND the upper part.

    for (int k = 0; k < p_K.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<Real>::InnerIterator it(p_K, k); it; ++it) {
            const auto i = it.row();
            const auto j = it.col();
            const auto v = -1 * it.value() * kFact;
            if (i != j) {
                matrix->add(offset+i, offset+j, v);
                matrix->add(offset+j, offset+i, v);
            } else {
                matrix->add(offset+i, offset+i, v);
            }
        }
    }

    sofa::helper::AdvancedTimer::stepEnd("HyperelasticForcefield::addKToMatrix");
}

template <typename Element>
SReal HyperelasticForcefield<Element>::getPotentialEnergy (
    const sofa::core::MechanicalParams* mparams,
    const sofa::core::objectmodel::Data<VecCoord>& d_x) const {
    using namespace sofa::core::objectmodel;

    SOFA_UNUSED(mparams);

    if (!this->mstate)
        return 0.;

    const auto material = d_material.get();
    if (!material) {
        return 0;
    }

    sofa::helper::ReadAccessor<Data<VecCoord>> sofa_x = d_x;
    sofa::helper::ReadAccessor<Data<VecCoord>> sofa_x0 = this->mstate->readRestPositions();

    if (sofa_x.size() != sofa_x0.size() )
        return 0.;

    const auto nb_nodes = sofa_x.size();
    const auto nb_elements = this->number_of_elements();

    if (nb_nodes == 0 || nb_elements == 0)
        return 0;

    if (p_elements_quadrature_nodes.size() != nb_elements)
        return 0;

    const Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Dimension, Eigen::RowMajor>>    X       (sofa_x.ref().data()->data(),  nb_nodes, Dimension);
    const Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Dimension, Eigen::RowMajor>>    X0      (sofa_x0.ref().data()->data(), nb_nodes, Dimension);

    SReal Psi = 0.;

    sofa::helper::AdvancedTimer::stepBegin("HyperelasticForcefield::getPotentialEnergy");

    for (std::size_t element_id = 0; element_id < nb_elements; ++element_id) {
        // Fetch the node indices of the element
        auto node_indices = this->topology()->domain()->element_indices(element_id);

        // Fetch the initial and current positions of the element's nodes
        Matrix<NumberOfNodesPerElement, Dimension> initial_nodes_position;
        Matrix<NumberOfNodesPerElement, Dimension> current_nodes_position;

        for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            initial_nodes_position.row(i).noalias() = X0.row(node_indices[i]);
            current_nodes_position.row(i).noalias() = X.row(node_indices[i]);
        }

        // Compute the nodal displacement
        Matrix<NumberOfNodesPerElement, Dimension> U {};
        for (size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            const auto u = sofa_x[node_indices[i]] - sofa_x0[node_indices[i]];
            for (size_t j = 0; j < Dimension; ++j) {
                U(i, j) = u[j];
            }
        }

        // Compute the nodal forces

        for (const GaussNode & gauss_node : p_elements_quadrature_nodes[element_id]) {

            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto & detJ = gauss_node.jacobian_determinant;

            // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
            const auto & dN_dx = gauss_node.dN_dx;

            // Gauss quadrature node weight
            const auto & w = gauss_node.weight;

            // Deformation tensor at gauss node
            const auto & F = caribou::mechanics::elasticity::strain::F(dN_dx, U);
            const auto J = F.determinant();

            // Strain tensor at gauss node
            const Mat33 C = F.transpose() * F;

            // Add the potential energy at gauss node
            Psi += (detJ * w) *  material->strain_energy_density(J, C);
        }
    }

    sofa::helper::AdvancedTimer::stepEnd("HyperelasticForcefield::getPotentialEnergy");

    return Psi;
}

template <typename Element>
void HyperelasticForcefield<Element>::initialize_elements()
{
    using namespace sofa::core::objectmodel;

    sofa::helper::AdvancedTimer::stepBegin("HyperelasticForcefield::initialize_elements");

    if (!this->mstate)
        return;

    // Resize the container of elements'quadrature nodes
    const auto nb_elements = this->number_of_elements();
    if (p_elements_quadrature_nodes.size() != nb_elements) {
        p_elements_quadrature_nodes.resize(nb_elements);
    }

    // Translate the Sofa's mechanical state vector to Eigen vector type
    sofa::helper::ReadAccessor<Data<VecCoord>> sofa_x0 = this->mstate->readRestPositions();
    const Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Dimension, Eigen::RowMajor>>    X0      (sofa_x0.ref().data()->data(), sofa_x0.size(), Dimension);

    // Loop on each element and compute the shape functions and their derivatives for every of their integration points
    for (std::size_t element_id = 0; element_id < nb_elements; ++element_id) {

        // Get an Element instance from the Domain
        const auto initial_element = this->topology()->element(element_id);

        // Fill in the Gauss integration nodes for this element
        p_elements_quadrature_nodes[element_id] = get_gauss_nodes(element_id, initial_element);
    }

    vertPlasticOffsets.resize(this->topology()->domain()->mesh()->number_of_nodes());
    vertPlasticOffsets.fill(Coord(0, 0, 0));
    elemInitialVolume.resize(nb_elements);
    elemInitialRotation.resize(nb_elements);
    
    //Compute the inital rotation per element -- TODO for hex8 element only
    if (NumberOfNodesPerElement == 8 && Dimension == 3) {
        for (int element_id = 0; element_id < static_cast<int>(nb_elements); ++element_id) {
            // Fetch the node indices of the element
            auto node_indices = this->topology()->domain()->element_indices(element_id);

            // Fetch the initial positions of the element's nodes
            Matrix<NumberOfNodesPerElement, Dimension> init_nodes_position;
            for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
                init_nodes_position.row(i).noalias() = X0.row(node_indices[i]).template cast<Real>();
            }
            elemInitialRotation[element_id] = extractHexRotation(init_nodes_position);
        }
    }
    

    // Compute the element initial volume
    Real v = 0.;
    Real elemVolume;
    for (std::size_t element_id = 0; element_id < nb_elements; ++element_id) {
        elemVolume = 0.;
        for (const auto & gauss_node : gauss_nodes_of(element_id)) {
            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto detJ = gauss_node.jacobian_determinant;

            // Gauss quadrature node weight
            const auto w = gauss_node.weight;

            elemVolume += detJ * w;
        }
        v += elemVolume;
        elemInitialVolume[element_id] = elemVolume;
    }
    msg_info() << "Total volume of the geometry is " << v;

    sofa::helper::AdvancedTimer::stepEnd("HyperelasticForcefield::initialize_elements");
}

template <typename Element>
void HyperelasticForcefield<Element>::assemble_stiffness()
{
    assemble_stiffness(*this->mstate->read (p_X_id.getId(this->mstate)));
}

template<typename Element>
void HyperelasticForcefield<Element>::assemble_stiffness(const sofa::core::objectmodel::Data<VecCoord> & x) {
    using namespace sofa::core::objectmodel;

    const sofa::helper::ReadAccessor<Data<VecCoord>> sofa_x= x;
    const auto nb_nodes = sofa_x.size();
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Dimension, Eigen::RowMajor>>    X       (sofa_x.ref().data()->data(),  nb_nodes, Dimension);

    assemble_stiffness(X);
}

template<typename Element>
template<typename Derived>
void HyperelasticForcefield<Element>::assemble_stiffness(const Eigen::MatrixBase<Derived> & x) {
    const auto material = d_material.get();

    [[maybe_unused]]
    const auto enable_multithreading = d_enable_multithreading.getValue();
    if (!material) {
        return;
    }

    // Update material parameters in case the user changed it
    material->before_update();

    static const auto Id = Mat33::Identity();
    const auto nb_elements = this->number_of_elements();
    const auto nb_nodes = x.rows();
    const auto nDofs = nb_nodes*Dimension;
    p_K.resize(nDofs, nDofs);

    ///< Triplets are used to store matrix entries before the call to 'compress'.
    /// Duplicates entries are summed up.
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(nDofs*24*2);

    sofa::helper::AdvancedTimer::stepBegin("HyperelasticForcefield::update_stiffness");
#pragma omp parallel for if (enable_multithreading)
    for (int element_id = 0; element_id < static_cast<int>(nb_elements); ++element_id) {
        // Fetch the node indices of the element
        auto node_indices = this->topology()->domain()->element_indices(element_id);

        // Fetch the current positions of the element's nodes
        Matrix<NumberOfNodesPerElement, Dimension> current_nodes_position;

        for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            current_nodes_position.row(i).noalias() = x.row(node_indices[i]).template cast<Real>();
            for (int d = 0; d < Dimension; d++)
                current_nodes_position.row(i)(d) -= vertPlasticOffsets[node_indices[i]][d];
        }

        using Stiffness = Eigen::Matrix<FLOATING_POINT_TYPE, NumberOfNodesPerElement*Dimension, NumberOfNodesPerElement*Dimension, Eigen::RowMajor>;
        Stiffness Ke = Stiffness::Zero();

        for (const auto & gauss_node : gauss_nodes_of(element_id)) {
            // Jacobian of the gauss node's transformation mapping from the elementary space to the world space
            const auto detJ = gauss_node.jacobian_determinant;

            // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
            const auto dN_dx = gauss_node.dN_dx;

            // Gauss quadrature node weight
            const auto w = gauss_node.weight;

            // Deformation tensor at gauss node
            const Mat33 F = current_nodes_position.transpose() * dN_dx * gauss_node.inversePlasticStrain;
            const auto J = F.determinant();

            // Right Cauchy-Green strain tensor at gauss node
            const Mat33 C = F.transpose() * F ;

            // Second Piola-Kirchhoff stress tensor at gauss node
            const auto S = material->PK2_stress(J, C);

            // Jacobian of the Second Piola-Kirchhoff stress tensor at gauss node
            const auto D = material->PK2_stress_jacobian(J, C);

            // Computation of the tangent-stiffness matrix
            for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
                // Derivatives of the ith shape function at the gauss node with respect to global coordinates x,y and z
                const Vec3 dxi = dN_dx.row(i).transpose();

                Matrix<6,3> Bi;
                Bi <<
                   F(0,0)*dxi[0],                 F(1,0)*dxi[0],                 F(2,0)*dxi[0],
                        F(0,1)*dxi[1],                 F(1,1)*dxi[1],                 F(2,1)*dxi[1],
                        F(0,2)*dxi[2],                 F(1,2)*dxi[2],                 F(2,2)*dxi[2],
                        F(0,0)*dxi[1] + F(0,1)*dxi[0], F(1,0)*dxi[1] + F(1,1)*dxi[0], F(2,0)*dxi[1] + F(2,1)*dxi[0],
                        F(0,1)*dxi[2] + F(0,2)*dxi[1], F(1,1)*dxi[2] + F(1,2)*dxi[1], F(2,1)*dxi[2] + F(2,2)*dxi[1],
                        F(0,0)*dxi[2] + F(0,2)*dxi[0], F(1,0)*dxi[2] + F(1,2)*dxi[0], F(2,0)*dxi[2] + F(2,2)*dxi[0];

                // The 3x3 sub-matrix Kii is symmetric, we only store its upper triangular part
                Mat33 Kii = (dxi.dot(S*dxi)*Id + Bi.transpose()*D*Bi) * detJ * w;
                Ke.template block<Dimension, Dimension>(i*Dimension, i*Dimension)
                        .template triangularView<Eigen::Upper>()
                        += Kii;

                // We now loop only on the upper triangular part of the
                // element stiffness matrix Ke since it is symmetric
                for (std::size_t j = i+1; j < NumberOfNodesPerElement; ++j) {
                    // Derivatives of the jth shape function at the gauss node with respect to global coordinates x,y and z
                    const Vec3 dxj = dN_dx.row(j).transpose();

                    Matrix<6,3> Bj;
                    Bj <<
                       F(0,0)*dxj[0],                 F(1,0)*dxj[0],                 F(2,0)*dxj[0],
                            F(0,1)*dxj[1],                 F(1,1)*dxj[1],                 F(2,1)*dxj[1],
                            F(0,2)*dxj[2],                 F(1,2)*dxj[2],                 F(2,2)*dxj[2],
                            F(0,0)*dxj[1] + F(0,1)*dxj[0], F(1,0)*dxj[1] + F(1,1)*dxj[0], F(2,0)*dxj[1] + F(2,1)*dxj[0],
                            F(0,1)*dxj[2] + F(0,2)*dxj[1], F(1,1)*dxj[2] + F(1,2)*dxj[1], F(2,1)*dxj[2] + F(2,2)*dxj[1],
                            F(0,0)*dxj[2] + F(0,2)*dxj[0], F(1,0)*dxj[2] + F(1,2)*dxj[0], F(2,0)*dxj[2] + F(2,2)*dxj[0];

                    // The 3x3 sub-matrix Kij is NOT symmetric, we store its full part
                    Mat33 Kij = (dxi.dot(S*dxj)*Id + Bi.transpose()*D*Bj) * detJ * w;
                    Ke.template block<Dimension, Dimension>(i*Dimension, j*Dimension)
                            .noalias() += Kij;
                }
            }
        }

#pragma omp critical
        for (std::size_t i = 0; i < NumberOfNodesPerElement; ++i) {
            // Node index of the ith node in the global stiffness matrix
            const auto x = static_cast<int>(node_indices[i]*Dimension);
            for (int m = 0; m < Dimension; ++m) {
                for (int n = m; n < Dimension; ++n) {
                    triplets.emplace_back(x+m, x+n, Ke(i*Dimension+m,i*Dimension+n));
                }
            }

            for (std::size_t j = i+1; j < NumberOfNodesPerElement; ++j) {
                // Node index of the jth node in the global stiffness matrix
                const auto y = static_cast<int>(node_indices[j]*Dimension);
                for (int m = 0; m < Dimension; ++m) {
                    for (int n = 0; n < Dimension; ++n) {
                        triplets.emplace_back(x+m, y+n, Ke(i*Dimension+m,j*Dimension+n));
                    }
                }
            }
        }
    }
    p_K.setFromTriplets(triplets.begin(), triplets.end());
    sofa::helper::AdvancedTimer::stepEnd("HyperelasticForcefield::update_stiffness");

    K_is_up_to_date = true;
    eigenvalues_are_up_to_date = false;
}

template <typename Element>
auto HyperelasticForcefield<Element>::get_gauss_nodes(const std::size_t & /*element_id*/, const Element & element) const -> GaussContainer {
    GaussContainer gauss_nodes {};
    if constexpr (NumberOfGaussNodesPerElement == caribou::Dynamic) {
        gauss_nodes.resize(element.number_of_gauss_nodes());
    }

    const auto nb_of_gauss_nodes = gauss_nodes.size();
    for (std::size_t gauss_node_id = 0; gauss_node_id < nb_of_gauss_nodes; ++gauss_node_id) {
        const auto & g = element.gauss_node(gauss_node_id);

        const auto J = element.jacobian(g.position);
        const Mat33 Jinv = J.inverse();
        const auto detJ = std::abs(J.determinant());

        // Derivatives of the shape functions at the gauss node with respect to global coordinates x,y and z
        const Matrix<NumberOfNodesPerElement, Dimension> dN_dx =
            (Jinv.transpose() * element.dL(g.position).transpose()).transpose();


        GaussNode & gauss_node = gauss_nodes[gauss_node_id];
        gauss_node.weight               = g.weight;
        gauss_node.jacobian_determinant = detJ;
        gauss_node.dN_dx                = dN_dx;
        
        ///init 
        gauss_node.inversePlasticStrain = Mat33::Identity();
        gauss_node.yieldStress = this->d_plasticYieldThreshold.getValue();
        gauss_node.backStress = Vec3(0, 0, 0);
    }

    return gauss_nodes;
}

template <typename Element>
auto HyperelasticForcefield<Element>::eigenvalues() -> const Vector<Eigen::Dynamic> & {
    if (not eigenvalues_are_up_to_date) {
#ifdef EIGEN_USE_LAPACKE
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> k (K());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> eigensolver(k, Eigen::EigenvaluesOnly);
#else
        Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<Real>> eigensolver(K(), Eigen::EigenvaluesOnly);
#endif
        if (eigensolver.info() != Eigen::Success) {
            msg_error() << "Unable to find the eigen values of K.";
        }

        p_eigenvalues = eigensolver.eigenvalues();
        eigenvalues_are_up_to_date = true;
    }

    return p_eigenvalues;
}

template <typename Element>
auto HyperelasticForcefield<Element>::cond() -> Real {
    const auto & values = eigenvalues();
    const auto min = values.minCoeff();
    const auto max = values.maxCoeff();

    return min/max;
}

template <typename Element>
auto HyperelasticForcefield<Element>::extractHexRotation(Matrix<NumberOfNodesPerElement, Dimension>& nodes) -> sofa::type::Mat<3, 3, Real>
{
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    Matrix<1, Dimension> horizontal;
    Matrix<1, Dimension> vertical;
    if (NumberOfNodesPerElement == 8 && Dimension == 3) {
        horizontal = (nodes.row(1) - nodes.row(0) + nodes.row(2) - nodes.row(3) + nodes.row(5) - nodes.row(4) + nodes.row(6) - nodes.row(7));
        vertical = (nodes.row(3) - nodes.row(0) + nodes.row(2) - nodes.row(1) + nodes.row(7) - nodes.row(4) + nodes.row(6) - nodes.row(5));
    }
    
    horizontal.normalize();
    Matrix<1, Dimension> edgez = horizontal.cross(vertical);
    edgez.normalize();

    vertical = edgez.cross(horizontal);
    vertical.normalize();

    sofa::type::Mat<3, 3, Real> r;
    r[0][0] = horizontal[0];
    r[0][1] = horizontal[1];
    r[0][2] = horizontal[2];
    r[1][0] = vertical[0];
    r[1][1] = vertical[1];
    r[1][2] = vertical[2];
    r[2][0] = edgez[0];
    r[2][1] = edgez[1];
    r[2][2] = edgez[2];

    return r;
}


template <typename Element>
auto HyperelasticForcefield<Element>::computeHexVolume(Matrix<NumberOfNodesPerElement, Dimension >& M) -> Real
{
    if (NumberOfNodesPerElement != 8 || Dimension != 3)
        return -1.;

    sofa::type::fixed_array<Coord, 8> coords;
    
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 3; j++)
            coords[i][j] = M(i, j);

    //000->0, 100->1, 110->2, 010->3, 001->4, 101->5, 111->6, 011->7
    Real volume;
    Real t[337];
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

    //The above code computes hex defined on [0,1]^3 with different vertex ordering
    //scaled by *-1 to get the correct result
    return volume * -1.0;
}


} // namespace SofaCaribou::forcefield
