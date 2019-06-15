#ifndef OBBTREEGPUINTERSECTION_H
#define OBBTREEGPUINTERSECTION_H

#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>

#include <ZyObbTreeGPU/config.h>
#include "ObbTreeGPUCollisionModel.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <SofaVolumetricData/DistanceGridCollisionModel.h>
#include <SofaBaseCollision/MinProximityIntersection.h>

#include <SofaConstraint/LocalMinDistance.h>

namespace sofa
{
    namespace component
    {
        namespace collision
        {
			class SOFA_OBBTREEGPUPLUGIN_API OBBTreeGPUDiscreteIntersection :  public sofa::core::objectmodel::BaseObject, public core::collision::BaseIntersector  /*, public component::collision::DiscreteIntersection */
            {
                typedef DiscreteIntersection::OutputVector OutputVector;

                public:
                    //SOFA_CLASS(OBBTreeGPUDiscreteIntersection,sofa::component::collision::DiscreteIntersection);
					SOFA_CLASS(OBBTreeGPUDiscreteIntersection, sofa::core::objectmodel::BaseObject);

                    OBBTreeGPUDiscreteIntersection();
                    OBBTreeGPUDiscreteIntersection(DiscreteIntersection* object, bool addSelf=true);

                    bool testIntersection(ObbTreeGPUCollisionModelNode&, ObbTreeGPUCollisionModelNode&);

                    int computeIntersection(ObbTreeGPUCollisionModelNode& e1, ObbTreeGPUCollisionModelNode& e2, OutputVector* contacts);

                protected:
                    DiscreteIntersection* intersection;
            };


            class SOFA_OBBTREEGPUPLUGIN_API ObbTreeGPULocalMinDistance: public LocalMinDistance
            {
				std::ofstream testOutput;
				sofa::core::objectmodel::DataFileName testOutputFilename;

				std::vector<std::pair<std::string, std::string> > checkedCollisionModels;
				std::vector<std::pair<std::string, std::string> > overlappingCollisionModels;

                public:
                    SOFA_CLASS(ObbTreeGPULocalMinDistance,sofa::component::collision::LocalMinDistance);

                    ObbTreeGPULocalMinDistance();

                    virtual void init();

					virtual void beginBroadPhase();
                    virtual void endBroadPhase();

                    bool testIntersection(ObbTreeGPUCollisionModelNode&, ObbTreeGPUCollisionModelNode&);

                    int computeIntersection(ObbTreeGPUCollisionModelNode& e1, ObbTreeGPUCollisionModelNode& e2, OutputVector* contacts);

                    // ObbTreeGPU and Cube models don't collide, currently
                    // (implementing these is necessary to add the corresponding intersector in ObbTreeGPULocalMinDistance::init())
                    // (without an intersector, SOFA complains if both ObbTreeGPU and Point/Line/Triangle models appear in the same scene)
                    bool testIntersection(ObbTreeGPUCollisionModelNode&, Cube&) { return false; }
                    int computeIntersection(ObbTreeGPUCollisionModelNode& e1, Cube& e2, OutputVector* contacts) { return 0; }

					bool testIntersection(Cube& c1, Cube& c2);
					bool testIntersection(Point&, Point&);
					bool testIntersection(Line&, Point&);
					bool testIntersection(Line&, Line&);
					bool testIntersection(Triangle&, Point&);

                    int computeIntersection(Cube& e1, Cube& e2, OutputVector* contacts);
					int computeIntersection(Point&, Point&, OutputVector*);
					int computeIntersection(Line&, Point&, OutputVector*);
					int computeIntersection(Line&, Line&, OutputVector*);
					int computeIntersection(Triangle&, Point&, OutputVector*);

            };
        } // namespace collision
    } // namespace component
} // namespace sofa


#endif // OBBTREEGPUINTERSECTION_H
