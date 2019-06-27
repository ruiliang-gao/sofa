#ifndef PBDSIMULATIONMODEL_H
#define PBDSIMULATIONMODEL_H

#include <vector>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>
#include "initZyPositionBasedDynamicsPlugin.h"

#include "PBDCommon/PBDCommon.h"
#include "PBDSimulation/PBDRigidBody.h"
#include "PBDSimulation/PBDParticleData.h"
#include "PBDModels/PBDTriangleModel.h"
#include "PBDModels/PBDTetrahedronModel.h"
#include "PBDModels/PBDLineModel.h"

// Replaceable with SOFA's Data mechanism
// #include "ParameterObject.h"

#include "PBDConstraints/PBDConstraints.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            using namespace sofa::core::objectmodel;

            class PBDConstraintBase;

            class SOFA_ZY_POSITION_BASED_DYNAMICS_PLUGIN_API PBDSimulationModel: public sofa::core::objectmodel::BaseObject
            {
                public:
                    PBDSimulationModel();
                    virtual ~PBDSimulationModel();

                    void init();
                    void reset();
                    void cleanup();

                    typedef std::vector<PBDConstraintBase*> ConstraintVector;
                    typedef std::vector<RigidBodyContactConstraint> RigidBodyContactConstraintVector;
                    typedef std::vector<ParticleRigidBodyContactConstraint> ParticleRigidBodyContactConstraintVector;
                    typedef std::vector<ParticleTetContactConstraint> ParticleSolidContactConstraintVector;
                    typedef std::vector<PBDRigidBody*> RigidBodyVector;
                    typedef std::vector<PBDTriangleModel*> TriangleModelVector;
                    typedef std::vector<PBDTetrahedronModel*> TetModelVector;
                    typedef std::vector<PBDLineModel*> LineModelVector;
                    typedef std::vector<unsigned int> ConstraintGroup;
                    typedef std::vector<ConstraintGroup> ConstraintGroupVector;

                public:
                    Data<Real> CLOTH_STIFFNESS;
                    Data<Real> CLOTH_BENDING_STIFFNESS;
                    Data<Real> CLOTH_STIFFNESS_XX;
                    Data<Real> CLOTH_STIFFNESS_YY;
                    Data<Real> CLOTH_STIFFNESS_XY;
                    Data<Real> CLOTH_POISSON_RATIO_XY;
                    Data<Real> CLOTH_POISSON_RATIO_YX;
                    Data<bool> CLOTH_NORMALIZE_STRETCH;
                    Data<bool> CLOTH_NORMALIZE_SHEAR;

                    Data<Real> SOLID_STIFFNESS;
                    Data<Real> SOLID_POISSON_RATIO;
                    Data<bool> SOLID_NORMALIZE_STRETCH;
                    Data<bool> SOLID_NORMALIZE_SHEAR;

                protected:
                    RigidBodyVector m_rigidBodies;
                    TriangleModelVector m_triangleModels;
                    TetModelVector m_tetModels;
                    LineModelVector m_lineModels;
                    ParticleData m_particles;
                    OrientationData m_orientations;
                    ConstraintVector m_constraints;
                    RigidBodyContactConstraintVector m_rigidBodyContactConstraints;
                    ParticleRigidBodyContactConstraintVector m_particleRigidBodyContactConstraints;
                    ParticleSolidContactConstraintVector m_particleSolidContactConstraints;
                    ConstraintGroupVector m_constraintGroups;

                    Real m_contactStiffnessRigidBody;
                    Real m_contactStiffnessParticleRigidBody;

                    Real m_rod_stretchingStiffness;
                    Real m_rod_shearingStiffness1;
                    Real m_rod_shearingStiffness2;
                    Real m_rod_bendingStiffness1;
                    Real m_rod_bendingStiffness2;
                    Real m_rod_twistingStiffness;

                    virtual void initParameters();

            public:

                    RigidBodyVector &getRigidBodies();
                    ParticleData &getParticles();
                    OrientationData &getOrientations();
                    TriangleModelVector &getTriangleModels();
                    TetModelVector &getTetModels();
                    LineModelVector &getLineModels();
                    ConstraintVector &getConstraints();
                    RigidBodyContactConstraintVector &getRigidBodyContactConstraints();
                    ParticleRigidBodyContactConstraintVector &getParticleRigidBodyContactConstraints();
                    ParticleSolidContactConstraintVector &getParticleSolidContactConstraints();
                    ConstraintGroupVector &getConstraintGroups();
                    bool m_groupsInitialized;

                    void resetContacts();

                    void addTriangleModel(
                        const unsigned int nPoints,
                        const unsigned int nFaces,
                        Vector3r *points,
                        unsigned int* indices,
                        const PBDTriangleModel::ParticleMesh::UVIndices& uvIndices,
                        const PBDTriangleModel::ParticleMesh::UVs& uvs);

                    void addTetModel(
                        const unsigned int nPoints,
                        const unsigned int nTets,
                        Vector3r *points,
                        unsigned int* indices);

                    void addLineModel(
                        const unsigned int nPoints,
                        const unsigned int nQuaternions,
                        Vector3r *points,
                        Quaternionr *quaternions,
                        unsigned int *indices,
                        unsigned int *indicesQuaternions);

                    void updateConstraints();
                    void initConstraintGroups();

                    bool addBallJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos);
                    bool addBallOnLineJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &dir);
                    bool addHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    bool addTargetAngleMotorHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    bool addTargetVelocityMotorHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    bool addUniversalJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis1, const Vector3r &axis2);
                    bool addSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    bool addTargetPositionMotorSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    bool addTargetVelocityMotorSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    bool addRigidBodyParticleBallJoint(const unsigned int rbIndex, const unsigned int particleIndex);
                    bool addRigidBodySpring(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos1, const Vector3r &pos2, const Real stiffness);
                    bool addRigidBodyContactConstraint(const unsigned int rbIndex1, const unsigned int rbIndex2,
                            const Vector3r &cp1, const Vector3r &cp2,
                            const Vector3r &normal, const Real dist,
                            const Real restitutionCoeff, const Real frictionCoeff);
                    bool addParticleRigidBodyContactConstraint(const unsigned int particleIndex, const unsigned int rbIndex,
                            const Vector3r &cp1, const Vector3r &cp2,
                            const Vector3r &normal, const Real dist,
                            const Real restitutionCoeff, const Real frictionCoeff);

                    bool addParticleSolidContactConstraint(const unsigned int particleIndex, const unsigned int solidIndex,
                        const unsigned int tetIndex, const Vector3r &bary,
                        const Vector3r &cp1, const Vector3r &cp2,
                        const Vector3r &normal, const Real dist,
                        const Real restitutionCoeff, const Real frictionCoeff);

                    bool addDistanceConstraint(const unsigned int particle1, const unsigned int particle2);
                    bool addDihedralConstraint(	const unsigned int particle1, const unsigned int particle2,
                                                const unsigned int particle3, const unsigned int particle4);
                    bool addIsometricBendingConstraint(const unsigned int particle1, const unsigned int particle2,
                                                const unsigned int particle3, const unsigned int particle4);
                    bool addFEMTriangleConstraint(const unsigned int particle1, const unsigned int particle2, const unsigned int particle3);
                    bool addStrainTriangleConstraint(const unsigned int particle1, const unsigned int particle2, const unsigned int particle3);
                    bool addVolumeConstraint(const unsigned int particle1, const unsigned int particle2,
                                            const unsigned int particle3, const unsigned int particle4);
                    bool addFEMTetConstraint(const unsigned int particle1, const unsigned int particle2,
                                            const unsigned int particle3, const unsigned int particle4);
                    bool addStrainTetConstraint(const unsigned int particle1, const unsigned int particle2,
                                            const unsigned int particle3, const unsigned int particle4);
                    bool addShapeMatchingConstraint(const unsigned int numberOfParticles, const unsigned int particleIndices[], const unsigned int numClusters[]);
                    bool addStretchShearConstraint(const unsigned int particle1, const unsigned int particle2, const unsigned int quaternion1);
                    bool addBendTwistConstraint(const unsigned int quaternion1, const unsigned int quaternion2);
                    bool addStretchBendingTwistingConstraint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Real averageRadius, const Real averageSegmentLength, const Real youngsModulus, const Real torsionModulus);
                    bool addDirectPositionBasedSolverForStiffRodsConstraint(const std::vector<std::pair<unsigned int, unsigned int>> & jointSegmentIndices, const std::vector<Vector3r> &jointPositions, const std::vector<Real> &averageRadii, const std::vector<Real> &averageSegmentLengths, const std::vector<Real> &youngsModuli, const std::vector<Real> &torsionModuli);

                    Real getContactStiffnessRigidBody() const { return m_contactStiffnessRigidBody; }
                    void setContactStiffnessRigidBody(Real val) { m_contactStiffnessRigidBody = val; }
                    Real getContactStiffnessParticleRigidBody() const { return m_contactStiffnessParticleRigidBody; }
                    void setContactStiffnessParticleRigidBody(Real val) { m_contactStiffnessParticleRigidBody = val; }

                    Real getRodStretchingStiffness() const { return m_rod_stretchingStiffness;  }
                    void setRodStretchingStiffness(Real val) { m_rod_stretchingStiffness = val; }
                    Real getRodShearingStiffness1() const { return m_rod_shearingStiffness1; }
                    void setRodShearingStiffness1(Real val) { m_rod_shearingStiffness1 = val; }
                    Real getRodShearingStiffness2() const { return m_rod_shearingStiffness2; }
                    void setRodShearingStiffness2(Real val) { m_rod_shearingStiffness2 = val; }
                    Real getRodBendingStiffness1() const { return m_rod_bendingStiffness1; }
                    void setRodBendingStiffness1(Real val) { m_rod_bendingStiffness1 = val; }
                    Real getRodBendingStiffness2() const { return m_rod_bendingStiffness2; }
                    void setRodBendingStiffness2(Real val) { m_rod_bendingStiffness2 = val; }
                    Real getRodTwistingStiffness() const { return m_rod_twistingStiffness; }
                    void setRodTwistingStiffness(Real val) { m_rod_twistingStiffness = val; }
            };
        }
    }
}

#endif // PBDSIMULATIONMODEL_H
