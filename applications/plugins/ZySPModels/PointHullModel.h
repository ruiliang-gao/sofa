#ifndef POINTHULLMODEL_H
#define POINTHULLMODEL_H

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/visual/VisualParams.h>

#include <SofaBaseCollision/CubeModel.h>

#include "initSPModels.h"

#include "PointHull_Container.h"
#include "PointHull_Computation.h"

#include <GL/gl.h>

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            class PointHullModel;
			class SOFA_SPMODELSPLUGIN_API HullPoint : public core::TCollisionElementIterator<PointHullModel>
            {
                public:
					HullPoint(PointHullModel* model, int index);
					explicit HullPoint(core::CollisionElementIterator& i);

                    inline const Vector3& p() const { return _point; }

                private:
                    Vector3 _point;
            };


			class SOFA_SPMODELSPLUGIN_API PointHullModel : public sofa::core::CollisionModel
            {
                public:
					SOFA_CLASS(PointHullModel, sofa::core::CollisionModel);

                    typedef HullPoint Element;
                    typedef Vec3Types InDataTypes;

					PointHullModel();
					virtual ~PointHullModel();

                    void init();
                    void cleanup();

                    void draw(const sofa::core::visual::VisualParams*);
					void draw(const sofa::core::visual::VisualParams*, int);

                    virtual void computeBoundingTree(int maxDepth);

				private:
                    Data<int> m_edgePointsPerEdge;                    
                    CubeModel* m_cubeModel;

					sofa::component::container::MechanicalObject<Vec3Types>* m_mechanicalObject;

					PointHullStorage_Full* m_storage;
					PointHullComputation_CPU* m_computation;
					
					GLuint m_pointHullList;

					void initDrawList();
            };
        }
    }
}

#endif // POINTHULLMODEL_H
