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
#include <SofaConstraint/config.h>


#include <sofa/helper/map.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/fwd.h>
#include <sofa/linearalgebra/FullMatrix.h>

#include <sofa/simulation/CollisionAnimationLoop.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/simulation/fwd.h>

#include <vector>

namespace sofa::component::animationloop
{

class SOFA_SOFACONSTRAINT_API MechanicalGetConstraintResolutionVisitor : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalGetConstraintResolutionVisitor(const core::ConstraintParams* params, std::vector<core::behavior::ConstraintResolution*>& res, unsigned int offset)
        : simulation::BaseMechanicalVisitor(params), _res(res),_offset(offset), _cparams(params)
    {}

    Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet) override;
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/) override;

private:
    std::vector<core::behavior::ConstraintResolution*>& _res;
    unsigned int _offset;
    const sofa::core::ConstraintParams *_cparams;
};


class SOFA_SOFACONSTRAINT_API MechanicalSetConstraint : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalSetConstraint(const core::ConstraintParams* _cparams, core::MultiMatrixDerivId _res, unsigned int &_contactId)
        : simulation::BaseMechanicalVisitor(_cparams)
        , res(_res)
        , contactId(_contactId)
        , cparams(_cparams)
    {}

    Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* c) override;
    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override;
    bool isThreadSafe() const override;
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/) override;

protected:

    sofa::core::MultiMatrixDerivId res;
    unsigned int &contactId;
    const sofa::core::ConstraintParams *cparams;
};


class SOFA_SOFACONSTRAINT_API MechanicalAccumulateConstraint2 : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalAccumulateConstraint2(const core::ConstraintParams* _cparams, core::MultiMatrixDerivId _res)
        : simulation::BaseMechanicalVisitor(_cparams)
        , res(_res)
        , cparams(_cparams)
    {}

    void bwdMechanicalMapping(simulation::Node* node, core::BaseMapping* map) override;
    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override;

    bool isThreadSafe() const override;
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/) override;

protected:
    core::MultiMatrixDerivId res;
    const sofa::core::ConstraintParams *cparams;
};


class SOFA_SOFACONSTRAINT_API ConstraintProblem
{
protected:
    sofa::linearalgebra::LPtrFullMatrix<double> _W;
    sofa::linearalgebra::FullVector<double> _dFree, _force, _d, _df;// cf. These Duriez + _df for scheme correction
    std::vector<core::behavior::ConstraintResolution*> _constraintsResolutions;
    double _tol;
    int _dim;
    sofa::helper::system::thread::CTime *_timer;

public:
    ConstraintProblem(bool printLog=false);
    virtual ~ConstraintProblem();
    virtual void clear(int dim, const double &tol);

    inline int getSize(void) {return _dim;}
    inline sofa::linearalgebra::LPtrFullMatrix<double>* getW(void) {return &_W;}
    inline sofa::linearalgebra::FullVector<double>* getDfree(void) {return &_dFree;}
    inline sofa::linearalgebra::FullVector<double>* getD(void) {return &_d;}
    inline sofa::linearalgebra::FullVector<double>* getF(void) {return &_force;}
    inline sofa::linearalgebra::FullVector<double>* getdF(void) {return &_df;}
    inline std::vector<core::behavior::ConstraintResolution*>& getConstraintResolutions(void) {return _constraintsResolutions;}
    inline double *getTolerance(void) {return &_tol;}

    void gaussSeidelConstraintTimed(double &timeout, int numItMax);
};




class SOFA_SOFACONSTRAINT_API ConstraintAnimationLoop : public sofa::simulation::CollisionAnimationLoop
{
public:
    typedef sofa::simulation::CollisionAnimationLoop Inherit;

    SOFA_CLASS(ConstraintAnimationLoop, sofa::simulation::CollisionAnimationLoop);
protected:
    ConstraintAnimationLoop(simulation::Node* gnode = nullptr);
    ~ConstraintAnimationLoop() override;
public:

    void step(const core::ExecParams* params, SReal dt) override;
    void init() override;

    Data<bool> d_displayTime; ///< Display time for each important step of ConstraintAnimationLoop.
    Data<double> d_tol; ///< Tolerance of the Gauss-Seidel
    Data<int> d_maxIt; ///< Maximum number of iterations of the Gauss-Seidel
    Data<bool> d_doCollisionsFirst; ///< Compute the collisions first (to support penality-based contacts)
    Data<bool> d_doubleBuffer; ///< Buffer the constraint problem in a double buffer to be accessible with an other thread
    Data<bool> d_scaleTolerance; ///< Scale the error tolerance with the number of constraints
    Data<bool> d_allVerified; ///< All contraints must be verified (each constraint's error < tolerance)
    Data<double> d_sor; ///< Successive Over Relaxation parameter (0-2)
    Data<bool> d_schemeCorrection; ///< Apply new scheme where compliance is progressively corrected
    Data<bool> d_realTimeCompensation; ///< If the total computational time T < dt, sleep(dt-T)

    Data<bool> d_activateSubGraph;

    Data<std::map < std::string, sofa::type::vector<double> > > d_graphErrors; ///< Sum of the constraints' errors at each iteration
    Data<std::map < std::string, sofa::type::vector<double> > > d_graphConstraints; ///< Graph of each constraint's error at the end of the resolution
    Data<std::map < std::string, sofa::type::vector<double> > > d_graphForces; ///< Graph of each constraint's force at each step of the resolution

    ConstraintProblem *getConstraintProblem() {return bufCP1 ? &CP1 : &CP2;}

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T*, BaseContext* context, BaseObjectDescription* arg)
    {
        simulation::Node* gnode = dynamic_cast<simulation::Node*>(context);
        typename T::SPtr obj = sofa::core::objectmodel::New<T>(gnode);
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
        return obj;
    }

protected:
    void launchCollisionDetection(const core::ExecParams* params);
    void freeMotion(const core::ExecParams* params, simulation::Node *context, SReal &dt);
    void setConstraintEquations(const core::ExecParams* params, simulation::Node *context);
    void correctiveMotion(const core::ExecParams* params, simulation::Node *context);
    void debugWithContact(int numConstraints);

    ///  Specific procedures that are called for setting the constraints:

    /// 1.calling resetConstraint & setConstraint & accumulateConstraint visitors
    /// and resize the constraint problem that will be solved
    void writeAndAccumulateAndCountConstraintDirections(const core::ExecParams* params, simulation::Node *context, unsigned int &numConstraints);

    /// 2.calling GetConstraintViolationVisitor: each constraint provides its present violation
    /// for a given state (by default: free_position TODO: add VecId to make this method more generic)
    void getIndividualConstraintViolations(const core::ExecParams* params, simulation::Node *context);

    /// 3.calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
    void getIndividualConstraintSolvingProcess(const core::ExecParams* params, simulation::Node *context);

    /// 4.calling addComplianceInConstraintSpace projected in the contact space => getDelassusOperator(_W) = H*C*Ht
    virtual void computeComplianceInConstraintSpace();


    /// method for predictive scheme:
    void computePredictiveForce(int dim, double* force, std::vector<core::behavior::ConstraintResolution*>& res);



    void gaussSeidelConstraint(int dim, double* dfree, double** w, double* force, double* d, std::vector<core::behavior::ConstraintResolution*>& res, double* df);

    std::vector<core::behavior::BaseConstraintCorrection*> constraintCorrections;


    virtual ConstraintProblem* getCP();

    sofa::helper::system::thread::CTime *timer;
    SReal timeScale, time ;


    unsigned int numConstraints;

    bool bufCP1;
    SReal compTime, iterationTime;

private:
    ConstraintProblem CP1, CP2;
};

} //namespace sofa::component::animationloop
