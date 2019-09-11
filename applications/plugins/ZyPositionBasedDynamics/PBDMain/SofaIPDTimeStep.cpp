#include "SofaIPDTimeStep.h"

#include <sofa/core/visual/VisualParams.h>

using namespace sofa::simulation::PBDSimulation;

SofaIPDTimeStep::SofaIPDTimeStep(): sofa::core::objectmodel::BaseObject()
{
    m_simulator.reset(new IPS::Simulator());
}

SofaIPDTimeStep::~SofaIPDTimeStep()
{

}

double SofaIPDTimeStep::getTime()
{
    return 0.0;
}

void SofaIPDTimeStep::init()
{

}

void SofaIPDTimeStep::bwdInit()
{

}

void SofaIPDTimeStep::reset()
{

}

void SofaIPDTimeStep::cleanup()
{

}

void SofaIPDTimeStep::draw(const core::visual::VisualParams* vparams)
{

}

void SofaIPDTimeStep::step()
{

}
