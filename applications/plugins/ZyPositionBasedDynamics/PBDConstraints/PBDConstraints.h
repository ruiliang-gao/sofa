#ifndef PBDCONSTRAINTS_H
#define PBDCONSTRAINTS_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <list>
#include <memory>

#include "PBDCommon/PBDCommon.h"
#include "PBDConstraintBase.h"
#include "PBDynamics/DirectPositionBasedSolverForStiffRodsInterface.h"

namespace sofa
{
    namespace simulation
    {
        namespace PBDSimulation
        {
            class PBDSimulationModel;

            class BallJoint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 4> m_jointInfo;

                    BallJoint() : PBDConstraintBase(2) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class BallOnLineJoint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 10> m_jointInfo;

                    BallOnLineJoint() : PBDConstraintBase(2) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &dir);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class HingeJoint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 12> m_jointInfo;

                    HingeJoint() : PBDConstraintBase(2) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class UniversalJoint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 8> m_jointInfo;

                    UniversalJoint() : PBDConstraintBase(2) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis1, const Vector3r &axis2);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class SliderJoint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 14> m_jointInfo;

                    SliderJoint() : PBDConstraintBase(2) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class MotorJoint: public PBDConstraintBase
            {
                public:
                    Real m_target;
                    std::vector<Real> m_targetSequence;
                    MotorJoint() : PBDConstraintBase(2) { m_target = 0.0; }

                    virtual Real getTarget() const { return m_target; }
                    virtual void setTarget(const Real val) { m_target = val; }

                    virtual std::vector<Real> &getTargetSequence() { return m_targetSequence; }
                    virtual void setTargetSequence(const std::vector<Real> &val) { m_targetSequence = val; }

                    bool getRepeatSequence() const { return m_repeatSequence; }
                    void setRepeatSequence(bool val) { m_repeatSequence = val; }

                private:
                    bool m_repeatSequence;
            };

            class TargetPositionMotorSliderJoint : public MotorJoint
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 14> m_jointInfo;

                    TargetPositionMotorSliderJoint() : MotorJoint() {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class TargetVelocityMotorSliderJoint : public MotorJoint
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 14> m_jointInfo;

                    TargetVelocityMotorSliderJoint() : MotorJoint() {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
                    virtual bool solveVelocityConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class TargetAngleMotorHingeJoint : public MotorJoint
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 14> m_jointInfo;
                    TargetAngleMotorHingeJoint() : MotorJoint() {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual void setTarget(const Real val)
                    {
                            const Real pi = (Real)M_PI;
                            m_target = std::max(val, -pi);
                            m_target = std::min(m_target, pi);
                    }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
                private:
                    std::vector<Real> m_targetSequence;
            };

            class TargetVelocityMotorHingeJoint : public MotorJoint
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 14> m_jointInfo;
                    TargetVelocityMotorHingeJoint() : MotorJoint() {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
                    virtual bool solveVelocityConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class RigidBodyParticleBallJoint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 2> m_jointInfo;

                    RigidBodyParticleBallJoint() : PBDConstraintBase(2) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex, const unsigned int particleIndex);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class RigidBodySpring : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 4> m_jointInfo;
                    Real m_restLength;
                    Real m_stiffness;

                    RigidBodySpring() : PBDConstraintBase(2) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos1, const Vector3r &pos2, const Real stiffness);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class DistanceConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Real m_restLength;

                    DistanceConstraint() : PBDConstraintBase(2) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int particle1, const unsigned int particle2);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class DihedralConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Real m_restAngle;

                    DihedralConstraint() : PBDConstraintBase(4) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int particle1, const unsigned int particle2,
                                                                            const unsigned int particle3, const unsigned int particle4);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class IsometricBendingConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Matrix4r m_Q;

                    IsometricBendingConstraint() : PBDConstraintBase(4) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int particle1, const unsigned int particle2,
                                                                            const unsigned int particle3, const unsigned int particle4);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class FEMTriangleConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Real m_area;
                    Matrix2r m_invRestMat;

                    FEMTriangleConstraint() : PBDConstraintBase(3) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int particle1, const unsigned int particle2,
                            const unsigned int particle3);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class StrainTriangleConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Matrix2r m_invRestMat;

                    StrainTriangleConstraint() : PBDConstraintBase(3) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int particle1, const unsigned int particle2,
                            const unsigned int particle3);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class VolumeConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Real m_restVolume;

                    VolumeConstraint() : PBDConstraintBase(4) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int particle1, const unsigned int particle2,
                                                                    const unsigned int particle3, const unsigned int particle4);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class FEMTetConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Real m_volume;
                    Matrix3r m_invRestMat;

                    FEMTetConstraint() : PBDConstraintBase(4) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int particle1, const unsigned int particle2,
                                                                            const unsigned int particle3, const unsigned int particle4);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class StrainTetConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Matrix3r m_invRestMat;

                    StrainTetConstraint() : PBDConstraintBase(4) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int particle1, const unsigned int particle2,
                            const unsigned int particle3, const unsigned int particle4);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class ShapeMatchingConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Vector3r m_restCm;
                    Matrix3r m_invRestMat;
                    Real *m_w;
                    Vector3r *m_x0;
                    Vector3r *m_x;
                    Vector3r *m_corr;
                    unsigned int *m_numClusters;

                    ShapeMatchingConstraint(const unsigned int numberOfParticles) : PBDConstraintBase(numberOfParticles)
                    {
                            m_x = new Vector3r[numberOfParticles];
                            m_x0 = new Vector3r[numberOfParticles];
                            m_corr = new Vector3r[numberOfParticles];
                            m_w = new Real[numberOfParticles];
                            m_numClusters = new unsigned int[numberOfParticles];
                    }
                    virtual ~ShapeMatchingConstraint()
                    {
                            delete[] m_x;
                            delete[] m_x0;
                            delete[] m_corr;
                            delete[] m_w;
                            delete[] m_numClusters;
                    }
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int particleIndices[], const unsigned int numClusters[]);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class RigidBodyContactConstraint
            {
                public:
                    static int TYPE_ID;
                    /** indices of the linked bodies */
                    unsigned int m_bodies[2];
                    Real m_stiffness;
                    Real m_frictionCoeff;
                    Real m_sum_impulses;
                    Eigen::Matrix<Real, 3, 5> m_constraintInfo;

                    RigidBodyContactConstraint() {}
                    ~RigidBodyContactConstraint() {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int rbIndex1, const unsigned int rbIndex2,
                            const Vector3r &cp1, const Vector3r &cp2,
                            const Vector3r &normal, const Real dist,
                            const Real restitutionCoeff, const Real stiffness, const Real frictionCoeff);
                    virtual bool solveVelocityConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class ParticleRigidBodyContactConstraint
            {
                public:
                    static int TYPE_ID;
                    /** indices of the linked bodies */
                    unsigned int m_bodies[2];
                    Real m_stiffness;
                    Real m_frictionCoeff;
                    Real m_sum_impulses;
                    Eigen::Matrix<Real, 3, 5> m_constraintInfo;

                    ParticleRigidBodyContactConstraint() {}
                    ~ParticleRigidBodyContactConstraint() {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int particleIndex, const unsigned int rbIndex,
                            const Vector3r &cp1, const Vector3r &cp2,
                            const Vector3r &normal, const Real dist,
                            const Real restitutionCoeff, const Real stiffness, const Real frictionCoeff);
                    virtual bool solveVelocityConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class ParticleTetContactConstraint
            {
                public:
                    static int TYPE_ID;
                    /** indices of the linked bodies */
                    unsigned int m_bodies[2];
                    unsigned int m_solidIndex;
                    unsigned int m_tetIndex;
                    Vector3r m_bary;
                    Real m_lambda;
                    Real m_frictionCoeff;
                    Eigen::Matrix<Real, 3, 3> m_constraintInfo;
                    Real m_invMasses[4];
                    Vector3r m_x[4];
                    Vector3r m_v[4];

                    ParticleTetContactConstraint() { }
                    ~ParticleTetContactConstraint() {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int particleIndex, const unsigned int solidIndex,
                            const unsigned int tetindex, const Vector3r &bary,
                            const Vector3r &cp1, const Vector3r &cp2,
                            const Vector3r &normal, const Real dist,
                            const Real frictionCoeff);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
                    virtual bool solveVelocityConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class StretchShearConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Real m_restLength;

                    StretchShearConstraint() : PBDConstraintBase(3) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int particle1, const unsigned int particle2, const unsigned int quaternion1);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class BendTwistConstraint : public PBDConstraintBase
            {
                public:
                    static int TYPE_ID;
                    Quaternionr m_restDarbouxVector;

                    BendTwistConstraint() : PBDConstraintBase(2) {}
                    virtual int &getTypeId() const { return TYPE_ID; }

                    virtual bool initConstraint(PBDSimulationModel &model, const unsigned int quaternion1, const unsigned int quaternion2);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            class StretchBendingTwistingConstraint : public PBDConstraintBase
            {
                    using Matrix6r = Eigen::Matrix<Real, 6, 6>;
                    using Vector6r = Eigen::Matrix<Real, 6, 1>;
                public:
                    static int TYPE_ID;
                    Eigen::Matrix<Real, 3, 4> m_constraintInfo;

                    Real m_averageRadius;
                    Real m_averageSegmentLength;
                    Vector3r m_restDarbouxVector;
                    Vector3r m_stiffnessCoefficientK;
                    Vector3r m_stretchCompliance;
                    Vector3r m_bendingAndTorsionCompliance;
                    Vector6r m_lambdaSum;

                    StretchBendingTwistingConstraint() : PBDConstraintBase(2){}

                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model, const unsigned int segmentIndex1, const unsigned int segmentIndex2, const Vector3r &pos,
                            const Real averageRadius, const Real averageSegmentLength, Real youngsModulus, Real torsionModulus);
                    virtual bool initConstraintBeforeProjection(PBDSimulationModel &model);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);
            };

            struct Node;
            struct Interval;
            class PBDSimulationModel;
            using Vector6r = Eigen::Matrix<Real, 6, 1>;

            class DirectPositionBasedSolverForStiffRodsConstraint : public PBDConstraintBase
            {
                    class RodSegmentImpl : public RodSegment
                    {
                        public:
                            RodSegmentImpl(PBDSimulationModel &model, unsigned int idx) :
                                    m_model(model), m_segmentIdx(idx) {}

                            virtual bool isDynamic();
                            virtual Real Mass();
                            virtual const Vector3r & InertiaTensor();
                            virtual const Vector3r & Position();
                            virtual const Quaternionr & Rotation();

                            PBDSimulationModel &m_model;
                            unsigned int m_segmentIdx;
                    };

                    class RodConstraintImpl : public RodConstraint
                    {
                        public:
                            std::vector<unsigned int> m_segments;
                            Eigen::Matrix<Real, 3, 4> m_constraintInfo;

                            Real m_averageRadius;
                            Real m_averageSegmentLength;
                            Vector3r m_restDarbouxVector;
                            Vector3r m_stiffnessCoefficientK;
                            Vector3r m_stretchCompliance;
                            Vector3r m_bendingAndTorsionCompliance;

                            virtual unsigned int segmentIndex(unsigned int i){
                                    if (i < static_cast<unsigned int>(m_segments.size()))
                                            return m_segments[i];
                                    return 0u;
                            };

                            virtual Eigen::Matrix<Real, 3, 4> & getConstraintInfo(){ return m_constraintInfo; }
                            virtual Real getAverageSegmentLength(){ return m_averageSegmentLength; }
                            virtual Vector3r &getRestDarbouxVector(){ return m_restDarbouxVector; }
                            virtual Vector3r &getStiffnessCoefficientK() { return m_stiffnessCoefficientK; }
                            virtual Vector3r & getStretchCompliance(){ return m_stretchCompliance; }
                            virtual Vector3r & getBendingAndTorsionCompliance(){ return m_bendingAndTorsionCompliance; }
                    };

                public:
                    static int TYPE_ID;

                    DirectPositionBasedSolverForStiffRodsConstraint() :  PBDConstraintBase(2),
                            root(NULL), numberOfIntervals(0), intervals(NULL), forward(NULL), backward(NULL){}
                    ~DirectPositionBasedSolverForStiffRodsConstraint();

                    virtual int &getTypeId() const { return TYPE_ID; }

                    bool initConstraint(PBDSimulationModel &model,
                            const std::vector<std::pair<unsigned int, unsigned int>> & constraintSegmentIndices,
                            const std::vector<Vector3r> &constraintPositions,
                            const std::vector<Real> &averageRadii,
                            const std::vector<Real> &averageSegmentLengths,
                            const std::vector<Real> &youngsModuli,
                            const std::vector<Real> &torsionModuli);

                    virtual bool initConstraintBeforeProjection(PBDSimulationModel &model);
                    virtual bool updateConstraint(PBDSimulationModel &model);
                    virtual bool solvePositionConstraint(PBDSimulationModel &model, const unsigned int iter);

                protected:

                    /** root node */
                    Node *root;
                    /** intervals of constraints */
                    Interval *intervals;
                    /** number of intervals */
                    int numberOfIntervals;
                    /** list to process nodes with increasing row index in the system matrix H (from the leaves to the root) */
                    std::list <Node*> *forward;
                    /** list to process nodes starting with the highest row index to row index zero in the matrix H (from the root to the leaves) */
                    std::list <Node*> *backward;

                    std::vector<RodConstraintImpl> m_Constraints;
                    std::vector<RodConstraint*> m_rodConstraints;

                    std::vector<RodSegmentImpl> m_Segments;
                    std::vector<RodSegment*> m_rodSegments;

                    std::vector<Vector6r> m_rightHandSide;
                    std::vector<Vector6r> m_lambdaSums;
                    std::vector<std::vector<Matrix3r>> m_bendingAndTorsionJacobians;
                    std::vector<Vector3r> m_corr_x;
                    std::vector<Quaternionr> m_corr_q;

                    void deleteNodes();
            };
        }
    }
}

#endif // PBDCONSTRAINTS_H
