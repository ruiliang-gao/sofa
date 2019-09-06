#include "PBDSimulationModel.h"

using namespace sofa::simulation::PBDSimulation;

PBDSimulationModel::PBDSimulationModel(): sofa::core::objectmodel::BaseObject()
{
    m_contactStiffnessRigidBody = 1.0;
    m_contactStiffnessParticleRigidBody = 100.0;

    m_rod_shearingStiffness1 = 1.0;
    m_rod_shearingStiffness2 = 1.0;
    m_rod_stretchingStiffness = 1.0;
    m_rod_bendingStiffness1 = 0.5;
    m_rod_bendingStiffness2 = 0.5;
    m_rod_twistingStiffness = 0.5;

    m_groupsInitialized = false;

    m_contactsCleared = false;

    m_rigidBodyContactConstraints.reserve(10000);
    m_particleRigidBodyContactConstraints.reserve(10000);
    m_particleSolidContactConstraints.reserve(10000);
}

PBDSimulationModel::~PBDSimulationModel()
{
    if (m_rigidBodies.size() > 0)
    {
        for (size_t k = 0; k < m_rigidBodies.size(); k++)
        {
            if (m_rigidBodies[k] != NULL)
            {
                delete m_rigidBodies[k];
                m_rigidBodies[k] = NULL;
            }
        }
        m_rigidBodies.clear();
    }
}

bool PBDSimulationModel::getContactsCleared() const
{
    return m_contactsCleared;
}

void PBDSimulationModel::setContactsCleared(bool val)
{
    m_contactsCleared = val;
}

void PBDSimulationModel::init()
{
    initParameters();
}

void PBDSimulationModel::initParameters()
{
    initData(&CLOTH_STIFFNESS, 1.0f, "Cloth stiffness", "Stiffness of cloth models.");
    CLOTH_STIFFNESS.setGroup("Cloth");
    initData(&CLOTH_STIFFNESS_XX, 1.0f, "Youngs modulus XX", "XX stiffness of orthotropic cloth models.");
    CLOTH_STIFFNESS_XX.setGroup("Cloth");
    initData(&CLOTH_STIFFNESS_YY, 1.0f, "Youngs modulus YY", "YY stiffness of orthotropic cloth models.");
    CLOTH_STIFFNESS_YY.setGroup("Cloth");
    initData(&CLOTH_STIFFNESS_XY, 1.0f, "Youngs modulus XY", "XY stiffness of orthotropic cloth models.");
    CLOTH_STIFFNESS_XY.setGroup("Cloth");
    initData(&CLOTH_POISSON_RATIO_XY, 0.3f, "Poisson ratio XY", "XY Poisson ratio of orthotropic cloth models.");
    CLOTH_POISSON_RATIO_XY.setGroup("Cloth");
    initData(&CLOTH_POISSON_RATIO_YX, 0.3f, "Poisson ratio YX", "YX Poisson ratio of orthotropic cloth models.");
    CLOTH_POISSON_RATIO_YX.setGroup("Cloth");
    initData(&CLOTH_BENDING_STIFFNESS, 0.01f, "Bending stiffness", "Bending stiffness of cloth models.");
    CLOTH_BENDING_STIFFNESS.setGroup("Cloth");
    initData(&CLOTH_NORMALIZE_STRETCH, false, "Normalize stretch", "Normalize stretch (strain based dynamics)");
    CLOTH_NORMALIZE_STRETCH.setGroup("Cloth");
    initData(&CLOTH_NORMALIZE_SHEAR, false, "Normalize shear", "Normalize shear (strain based dynamics)");
    CLOTH_NORMALIZE_SHEAR.setGroup("Cloth");

    initData(&SOLID_STIFFNESS, 1.0f, "Stiffness", "Stiffness of solid models.");
    SOLID_STIFFNESS.setGroup("Solids");
    initData(&SOLID_POISSON_RATIO, 0.3f, "Poisson ratio", "XY Poisson ratio of solid models.");
    SOLID_POISSON_RATIO.setGroup("Solids");
    initData(&SOLID_NORMALIZE_STRETCH, false, "Normalize stretch", "Normalize stretch (strain based dynamics)");
    SOLID_NORMALIZE_STRETCH.setGroup("Solids");
    initData(&SOLID_NORMALIZE_SHEAR, false, "Normalize shear", "Normalize shear (strain based dynamics)");
}

void PBDSimulationModel::reset()
{
    resetContacts();

    // rigid bodies
    for (size_t i = 0; i < m_rigidBodies.size(); i++)
    {
        m_rigidBodies[i]->reset();
        m_rigidBodies[i]->getGeometry().updateMeshTransformation(m_rigidBodies[i]->getPosition(), m_rigidBodies[i]->getRotationMatrix());
    }

    // particles
    for (unsigned int i = 0; i < m_particles.size(); i++)
    {
        const Vector3r& x0 = m_particles.getPosition0(i);
        m_particles.getPosition(i) = x0;
        m_particles.getLastPosition(i) = m_particles.getPosition(i);
        m_particles.getOldPosition(i) = m_particles.getPosition(i);
        m_particles.getVelocity(i).setZero();
        m_particles.getAcceleration(i).setZero();
    }

    // orientations
    for(unsigned int i = 0; i < m_orientations.size(); i++)
    {
        const Quaternionr& q0 = m_orientations.getQuaternion0(i);
        m_orientations.getQuaternion(i) = q0;
        m_orientations.getLastQuaternion(i) = q0;
        m_orientations.getOldQuaternion(i) = q0;
        m_orientations.getVelocity(i).setZero();
        m_orientations.getAcceleration(i).setZero();
    }

    updateConstraints();
}

void PBDSimulationModel::cleanup()
{
    resetContacts();

    // Rigid body etc. instances are managed by the SOFA wrapper classes
    /*for (unsigned int i = 0; i < m_rigidBodies.size(); i++)
        delete m_rigidBodies[i];*/

    m_rigidBodies.clear();

    // Rigid body etc. instances are managed by the SOFA wrapper classes
    /*for (unsigned int i = 0; i < m_triangleModels.size(); i++)
        delete m_triangleModels[i];*/

    m_triangleModels.clear();

    // Rigid body etc. instances are managed by the SOFA wrapper classes
    /*for (unsigned int i = 0; i < m_tetModels.size(); i++)
        delete m_tetModels[i];*/

    m_tetModels.clear();

    // Rigid body etc. instances are managed by the SOFA wrapper classes
    /*for (unsigned int i = 0; i < m_lineModels.size(); i++)
        delete m_lineModels[i];*/

    m_lineModels.clear();

    for (unsigned int i = 0; i < m_constraints.size(); i++)
        delete m_constraints[i];

    m_constraints.clear();
    m_particles.release();
    m_orientations.release();
    m_groupsInitialized = false;
}

PBDSimulationModel::RigidBodyVector & PBDSimulationModel::getRigidBodies()
{
    return m_rigidBodies;
}

PBDParticleData & PBDSimulationModel::getParticles()
{
    return m_particles;
}

PBDOrientationData & PBDSimulationModel::getOrientations()
{
    return m_orientations;
}

PBDSimulationModel::TriangleModelVector & PBDSimulationModel::getTriangleModels()
{
    return m_triangleModels;
}

PBDSimulationModel::TetModelVector & PBDSimulationModel::getTetModels()
{
    return m_tetModels;
}

PBDSimulationModel::LineModelVector & PBDSimulationModel::getLineModels()
{
    return m_lineModels;
}

PBDSimulationModel::ConstraintVector & PBDSimulationModel::getConstraints()
{
    return m_constraints;
}

PBDSimulationModel::RigidBodyContactConstraintVector & PBDSimulationModel::getRigidBodyContactConstraints()
{
    return m_rigidBodyContactConstraints;
}

PBDSimulationModel::ParticleRigidBodyContactConstraintVector & PBDSimulationModel::getParticleRigidBodyContactConstraints()
{
    return m_particleRigidBodyContactConstraints;
}

PBDSimulationModel::ParticleSolidContactConstraintVector & PBDSimulationModel::getParticleSolidContactConstraints()
{
    return m_particleSolidContactConstraints;
}

PBDSimulationModel::ConstraintGroupVector & PBDSimulationModel::getConstraintGroups()
{
    return m_constraintGroups;
}

void PBDSimulationModel::updateConstraints()
{
    for (unsigned int i = 0; i < m_constraints.size(); i++)
        m_constraints[i]->updateConstraint(*this);
}


bool PBDSimulationModel::addBallJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos)
{
    BallJoint *bj = new BallJoint();
    const bool res = bj->initConstraint(*this, rbIndex1, rbIndex2, pos);
    if (res)
    {
        m_constraints.push_back(bj);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addBallOnLineJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &dir)
{
    BallOnLineJoint *bj = new BallOnLineJoint();
    const bool res = bj->initConstraint(*this, rbIndex1, rbIndex2, pos, dir);
    if (res)
    {
        m_constraints.push_back(bj);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis)
{
    HingeJoint *hj = new HingeJoint();
    const bool res = hj->initConstraint(*this, rbIndex1, rbIndex2, pos, axis);
    if (res)
    {
        m_constraints.push_back(hj);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addUniversalJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis1, const Vector3r &axis2)
{
    UniversalJoint *uj = new UniversalJoint();
    const bool res = uj->initConstraint(*this, rbIndex1, rbIndex2, pos, axis1, axis2);
    if (res)
    {
        m_constraints.push_back(uj);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis)
{
    SliderJoint *joint = new SliderJoint();
    const bool res = joint->initConstraint(*this, rbIndex1, rbIndex2, pos, axis);
    if (res)
    {
        m_constraints.push_back(joint);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addTargetPositionMotorSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis)
{
    TargetPositionMotorSliderJoint *joint = new TargetPositionMotorSliderJoint();
    const bool res = joint->initConstraint(*this, rbIndex1, rbIndex2, pos, axis);
    if (res)
    {
        m_constraints.push_back(joint);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addTargetVelocityMotorSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis)
{
    TargetVelocityMotorSliderJoint *joint = new TargetVelocityMotorSliderJoint();
    const bool res = joint->initConstraint(*this, rbIndex1, rbIndex2, pos, axis);
    if (res)
    {
        m_constraints.push_back(joint);
        m_groupsInitialized = false;
    }
    return res;
}


bool PBDSimulationModel::addTargetAngleMotorHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis)
{
    TargetAngleMotorHingeJoint *hj = new TargetAngleMotorHingeJoint();
    const bool res = hj->initConstraint(*this, rbIndex1, rbIndex2, pos, axis);
    if (res)
    {
        m_constraints.push_back(hj);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addTargetVelocityMotorHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis)
{
    TargetVelocityMotorHingeJoint *hj = new TargetVelocityMotorHingeJoint();
    const bool res = hj->initConstraint(*this, rbIndex1, rbIndex2, pos, axis);
    if (res)
    {
        m_constraints.push_back(hj);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addRigidBodyParticleBallJoint(const unsigned int rbIndex, const unsigned int particleIndex)
{
    RigidBodyParticleBallJoint *bj = new RigidBodyParticleBallJoint();
    const bool res = bj->initConstraint(*this, rbIndex, particleIndex);
    if (res)
    {
        m_constraints.push_back(bj);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addRigidBodySpring(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos1, const Vector3r &pos2, const Real stiffness)
{
    RigidBodySpring *s = new RigidBodySpring();
    const bool res = s->initConstraint(*this, rbIndex1, rbIndex2, pos1, pos2, stiffness);
    if (res)
    {
        m_constraints.push_back(s);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addRigidBodyContactConstraint(const unsigned int rbIndex1, const unsigned int rbIndex2,
    const Vector3r &cp1, const Vector3r &cp2,
    const Vector3r &normal, const Real dist,
    const Real restitutionCoeff, const Real frictionCoeff)
{
    msg_info("PBDSimulationModel") << "addRigidBodyContactConstraint(" << rbIndex1 << ", " << rbIndex2 << ")";
    m_rigidBodyContactConstraints.emplace_back(RigidBodyContactConstraint());
    RigidBodyContactConstraint &cc = m_rigidBodyContactConstraints.back();
    const bool res = cc.initConstraint(*this, rbIndex1, rbIndex2, cp1, cp2, normal, dist, restitutionCoeff, m_contactStiffnessRigidBody, frictionCoeff);

    if (!res)
    {
        msg_warning("PBDSimulationModel") << "Constraint init failed!";
        m_rigidBodyContactConstraints.pop_back();
    }

    if (m_contactsCleared)
        setContactsCleared(false);

    msg_info("PBDSimulationModel") << "Constraint init successful: " << res << ", total rigid body constraints count = " << m_rigidBodyContactConstraints.size();
    return res;
}

 bool PBDSimulationModel::addParticleRigidBodyContactConstraint(const unsigned int particleIndex, const unsigned int rbIndex,
    const Vector3r &cp1, const Vector3r &cp2,
    const Vector3r &normal, const Real dist,
    const Real restitutionCoeff, const Real frictionCoeff)
{
    m_particleRigidBodyContactConstraints.emplace_back(ParticleRigidBodyContactConstraint());
    ParticleRigidBodyContactConstraint &cc = m_particleRigidBodyContactConstraints.back();
    const bool res = cc.initConstraint(*this, particleIndex, rbIndex, cp1, cp2, normal, dist, restitutionCoeff, m_contactStiffnessParticleRigidBody, frictionCoeff);
    if (!res)
        m_particleRigidBodyContactConstraints.pop_back();

    if (m_contactsCleared)
        setContactsCleared(false);

    return res;
}

bool PBDSimulationModel::addParticleSolidContactConstraint(const unsigned int particleIndex, const unsigned int solidIndex,
    const unsigned int tetIndex, const Vector3r &bary,
    const Vector3r &cp1, const Vector3r &cp2,
    const Vector3r &normal, const Real dist,
    const Real restitutionCoeff, const Real frictionCoeff)
{
    m_particleSolidContactConstraints.emplace_back(ParticleTetContactConstraint());
    ParticleTetContactConstraint &cc = m_particleSolidContactConstraints.back();
    const bool res = cc.initConstraint(*this, particleIndex, solidIndex, tetIndex, bary, cp1, cp2, normal, dist, frictionCoeff);
    if (!res)
        m_particleSolidContactConstraints.pop_back();

    if (m_contactsCleared)
        setContactsCleared(false);

    return res;
}

bool PBDSimulationModel::addDistanceConstraint(const unsigned int particle1, const unsigned int particle2)
{
    DistanceConstraint *c = new DistanceConstraint();
    const bool res = c->initConstraint(*this, particle1, particle2);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addDihedralConstraint(const unsigned int particle1, const unsigned int particle2,
                                            const unsigned int particle3, const unsigned int particle4)
{
    DihedralConstraint *c = new DihedralConstraint();
    const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addIsometricBendingConstraint(const unsigned int particle1, const unsigned int particle2,
                                                    const unsigned int particle3, const unsigned int particle4)
{
    IsometricBendingConstraint *c = new IsometricBendingConstraint();
    const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addFEMTriangleConstraint(const unsigned int particle1, const unsigned int particle2,
            const unsigned int particle3)
{
    FEMTriangleConstraint *c = new FEMTriangleConstraint();
    const bool res = c->initConstraint(*this, particle1, particle2, particle3);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addStrainTriangleConstraint(const unsigned int particle1, const unsigned int particle2,
    const unsigned int particle3)
{
    StrainTriangleConstraint *c = new StrainTriangleConstraint();
    const bool res = c->initConstraint(*this, particle1, particle2, particle3);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addVolumeConstraint(const unsigned int particle1, const unsigned int particle2,
                                        const unsigned int particle3, const unsigned int particle4)
{
    VolumeConstraint *c = new VolumeConstraint();
    const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addFEMTetConstraint(const unsigned int particle1, const unsigned int particle2,
                                        const unsigned int particle3, const unsigned int particle4)
{
    FEMTetConstraint *c = new FEMTetConstraint();
    const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addStrainTetConstraint(const unsigned int particle1, const unsigned int particle2,
                                        const unsigned int particle3, const unsigned int particle4)
{
    StrainTetConstraint *c = new StrainTetConstraint();
    const bool res = c->initConstraint(*this, particle1, particle2, particle3, particle4);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addShapeMatchingConstraint(const unsigned int numberOfParticles, const unsigned int particleIndices[], const unsigned int numClusters[])
{
    ShapeMatchingConstraint *c = new ShapeMatchingConstraint(numberOfParticles);
    const bool res = c->initConstraint(*this, particleIndices, numClusters);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addStretchShearConstraint(const unsigned int particle1, const unsigned int particle2, const unsigned int quaternion1)
{
    StretchShearConstraint *c = new StretchShearConstraint();
    const bool res = c->initConstraint(*this, particle1, particle2, quaternion1);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addBendTwistConstraint(const unsigned int quaternion1, const unsigned int quaternion2)
{
    BendTwistConstraint *c = new BendTwistConstraint();
    const bool res = c->initConstraint(*this, quaternion1, quaternion2);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addStretchBendingTwistingConstraint(
    const unsigned int rbIndex1,
    const unsigned int rbIndex2,
    const Vector3r &pos,
    const Real averageRadius,
    const Real averageSegmentLength,
    const Real youngsModulus,
    const Real torsionModulus)
{
    StretchBendingTwistingConstraint *c = new StretchBendingTwistingConstraint();
    const bool res = c->initConstraint(*this, rbIndex1, rbIndex2, pos,
        averageRadius, averageSegmentLength, youngsModulus, torsionModulus);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

bool PBDSimulationModel::addDirectPositionBasedSolverForStiffRodsConstraint(
    const std::vector<std::pair<unsigned int, unsigned int>> & jointSegmentIndices,
    const std::vector<Vector3r> &jointPositions,
    const std::vector<Real> &averageRadii,
    const std::vector<Real> &averageSegmentLengths,
    const std::vector<Real> &youngsModuli,
    const std::vector<Real> &torsionModuli
    )
{
    DirectPositionBasedSolverForStiffRodsConstraint *c = new DirectPositionBasedSolverForStiffRodsConstraint();
    const bool res = c->initConstraint(*this, jointSegmentIndices, jointPositions,
        averageRadii, averageSegmentLengths, youngsModuli, torsionModuli);
    if (res)
    {
        m_constraints.push_back(c);
        m_groupsInitialized = false;
    }
    return res;
}

void PBDSimulationModel::addTriangleModel(
    const unsigned int nPoints,
    const unsigned int nFaces,
    Vector3r *points,
    unsigned int* indices,
    const PBDTriangleModel::ParticleMesh::UVIndices& uvIndices,
    const PBDTriangleModel::ParticleMesh::UVs& uvs)
{
    PBDTriangleModel *triModel = new PBDTriangleModel();
    m_triangleModels.push_back(triModel);

    unsigned int startIndex = m_particles.size();
    m_particles.reserve(startIndex + nPoints);

    for (unsigned int i = 0; i < nPoints; i++)
        m_particles.addVertex(points[i]);

    triModel->initMesh(nPoints, nFaces, startIndex, indices, uvIndices, uvs);

    // Update normals
    triModel->updateMeshNormals(m_particles);
}

void PBDSimulationModel::addTetModel(
    const unsigned int nPoints,
    const unsigned int nTets,
    Vector3r *points,
    unsigned int* indices)
{
    PBDTetrahedronModel *tetModel = new PBDTetrahedronModel();
    m_tetModels.push_back(tetModel);

    unsigned int startIndex = m_particles.size();
    m_particles.reserve(startIndex + nPoints);

    for (unsigned int i = 0; i < nPoints; i++)
        m_particles.addVertex(points[i]);

    tetModel->initMesh(nPoints, nTets, startIndex, indices);
}

void PBDSimulationModel::addLineModel(
    PBDLineModel* lineModel,
    const unsigned int nPoints,
    const unsigned int nQuaternions,
    Vector3r *points,
    Quaternionr *quaternions,
    unsigned int *indices,
    unsigned int *indicesQuaternions)
{
    msg_info("PBDSimulationModel") << "addLineModel() - nPoints = " << nPoints << ", nQuaternions = " << nQuaternions;

    m_lineModels.push_back(lineModel);

    msg_info("PBDSimulationModel") << "Particles vector size before adding new particles: " << m_particles.size();

    unsigned int startIndex = m_particles.size();
    m_particles.reserve(startIndex + nPoints);

    msg_info("PBDSimulationModel") << "Adding particles starting from index " << startIndex;

    for (unsigned int i = 0; i < nPoints; i++)
        m_particles.addVertex(points[i]);

    msg_info("PBDSimulationModel") << "Particles vector size after adding new particles: " << m_particles.size();

    msg_info("PBDSimulationModel") << "Particles orientations vector size before adding new particles: " << m_orientations.size();

    unsigned int startIndexOrientations = m_orientations.size();
    m_orientations.reserve(startIndexOrientations + nQuaternions);

    msg_info("PBDSimulationModel") << "Adding quaternions starting from index " << startIndexOrientations;

    for (unsigned int i = 0; i < nQuaternions; i++)
        m_orientations.addQuaternion(quaternions[i]);

    msg_info("PBDSimulationModel") << "Particles orientations vector size after adding new particles: " << m_orientations.size();

    lineModel->initMesh(nPoints, nQuaternions, startIndex, startIndexOrientations, indices, indicesQuaternions);
}

void PBDSimulationModel::initConstraintGroups()
{
    if (m_groupsInitialized)
        return;

    const unsigned int numConstraints = (unsigned int) m_constraints.size();
    const unsigned int numParticles = (unsigned int) m_particles.size();
    const unsigned int numRigidBodies = (unsigned int) m_rigidBodies.size();
    const unsigned int numBodies = numParticles + numRigidBodies;
    m_constraintGroups.clear();

    // Maps in which group a particle is or 0 if not yet mapped
    std::vector<unsigned char*> mapping;

    for (unsigned int i = 0; i < numConstraints; i++)
    {
        PBDConstraintBase *constraint = m_constraints[i];

        bool addToNewGroup = true;
        for (unsigned int j = 0; j < m_constraintGroups.size(); j++)
        {
            bool addToThisGroup = true;

            for (unsigned int k = 0; k < constraint->m_numberOfBodies; k++)
            {
                if (mapping[j][constraint->m_bodies[k]] != 0)
                {
                    addToThisGroup = false;
                    break;
                }
            }

            if (addToThisGroup)
            {
                m_constraintGroups[j].push_back(i);

                for (unsigned int k = 0; k < constraint->m_numberOfBodies; k++)
                    mapping[j][constraint->m_bodies[k]] = 1;

                addToNewGroup = false;
                break;
            }
        }
        if (addToNewGroup)
        {
            mapping.push_back(new unsigned char[numBodies]);
            memset(mapping[mapping.size() - 1], 0, sizeof(unsigned char)*numBodies);
            m_constraintGroups.resize(m_constraintGroups.size() + 1);
            m_constraintGroups[m_constraintGroups.size()-1].push_back(i);
            for (unsigned int k = 0; k < constraint->m_numberOfBodies; k++)
                mapping[m_constraintGroups.size() - 1][constraint->m_bodies[k]] = 1;
        }
    }

    for (unsigned int i = 0; i < mapping.size(); i++)
    {
        delete[] mapping[i];
    }
    mapping.clear();

    m_groupsInitialized = true;
}

void PBDSimulationModel::resetContacts()
{
    msg_info("PBDSimulationModel") << "resetContacts(); m_contactsCleared = " << m_contactsCleared;
    if (m_contactsCleared)
    {
        msg_info("PBDSimulationModel") << "Now clearing contact constraint lists.";
        m_rigidBodyContactConstraints.clear();
        m_particleRigidBodyContactConstraints.clear();
        m_particleSolidContactConstraints.clear();

        setContactsCleared(true);
    }
}
