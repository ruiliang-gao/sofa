#ifndef OBBTREEGPUObbTreeGPUFrictionContact_INL
#define OBBTREEGPUObbTreeGPUFrictionContact_INL

#include "ObbTreeGPUFrictionContact.h"

#include <limits>
#include <cfloat>
#include <iostream>
#include <math.h>

#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/DefaultContactManager.h>
#include <sofa/simulation/Node.h>

#include <SofaSimulationTree/GNode.h>

// for intersection detection begin
#include "Primitives/Triangle3.h"
#include <Primitives/Segment3.h>
// for intersection detection end

#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include <SofaMeshCollision/RigidContactMapper.inl>

#ifdef _WIN32
#include <gl/glut.h>
#else
#include <GL/glut.h>
#endif

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace sofa::defaulttype;
            using namespace core::collision;
            using simulation::Node;

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::ObbTreeGPUFrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
                : model1(model1)
                , model2(model2)
                , mmodel1(NULL)
                , mmodel2(NULL)
                , intersectionMethod(intersectionMethod)
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                , m_constraintGPU_ll(NULL)
    #endif
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                , m_constraintGPU_lv(NULL)
    #endif
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                , m_constraintGPU_vf(NULL)
    #endif
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                , m_numLineVertexContacts(0)
    #endif
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                , m_numLineLineContacts(0)
    #endif
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                , m_numVertexFaceContacts(0)
    #endif
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                , m_numOtherContacts(0)
    #endif
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                , m_defaultConstraint(NULL)
    #endif
                , parent(NULL)
                , mu (initData(&mu, 0.8, "mu", "friction coefficient (0 for frictionless contacts)"))
                , tol (initData(&tol, 0.0, "tol", "tolerance for the constraints resolution (0 for default tolerance)"))
            {
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG
                std::cout << "ObbTreeGPUFrictionContact<" << model1->getTypeName() << "," << model2->getTypeName() << "," << typeid(ResponseDataTypes).name() << ">::ObbTreeGPUFrictionContact(" << model1->getName() << "," << model1->getName() << ", " << intersectionMethod->getTypeName() << ")" << std::endl;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                selfCollision = ((core::CollisionModel*)model1 == (core::CollisionModel*)model2);
                mapper1_default.setCollisionModel(model1);
#else
                selfCollision = false;
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                if (!selfCollision)
                    mapper2_default.setCollisionModel(model2);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                mapper1_gpu_ll.setCollisionModel(model1);
                mapper2_gpu_ll.setCollisionModel(model2);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                mapper1_gpu_lv.setCollisionModel(model1);
                mapper2_gpu_lv.setCollisionModel(model2);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                mapper1_gpu_vf.setCollisionModel(model1);
                mapper2_gpu_vf.setCollisionModel(model2);
#endif
                contacts.clear();

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                mappedContacts.clear();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                mappedContacts_VertexFace.clear();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                mappedContacts_LineLine.clear();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                mappedContacts_LineVertex.clear();
#endif

                this->setName(model1->getName() + "_" + model2->getName() + "_OBBTreeGPUFrictionContact");
				testOutputFilename.setValue(sofa::helper::system::DataRepository.getFirstPath() + "/" + this->getName() + ".log");
				std::cout << "testOutputFilename = " << testOutputFilename.getValue() << std::endl;

                if (testOutputFilename.getValue() != "")
                {
            #ifndef _WIN32
                    //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::trunc);
            #else
                    //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::trunc);
            #endif
                    //testOutput << std::endl << "========================================================================================================" << std::endl;
                    //testOutput << "Constructing OBBTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>: " << this->getName() << std::endl;
                    //testOutput.close();
                }

            }

            // code for intersection detection begin
            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::~ObbTreeGPUFrictionContact()
            {
            }

            //template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            //void ObbTreeGPUFrictionContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes>::setIntersectionOccurred(bool newSCRValue)
            //{
                //
                // TODO: Delete this if it's no longer needed
                // This method is no longer used for anything as far as I can tell. I want to delete it due to its general awfulness, 
                // but I will leave it commented out for now, in case I am wrong and it is still needed somewhere.
                // If it's still needed, it should be implemented differently, because currently it requires linking against the 
                // Zyklio plugin.
                //

                //// TD: is mapper1_gpu_vf.mapper always there ?
                //BaseContext* rootContext = mapper1_gpu_vf.mapper->getContext()->getRootContext();

                //// starts in getRootContext, because the animation loop should always be there
                //// (This is pretty crude and might be better done some other way - but it works.)
                //// TD: Nowadays I know a better way to do this (but this function is not needed right now and I don't have time to fix it).
                //for (sofa::helper::vector<BaseLink*>::const_iterator it = rootContext->getLinks().begin(); it!= rootContext->getLinks().end(); it++)
                //{
                //    //std::cout << (*it)->getName() << " " << std::endl;
                //    if ((*it)->getName() == "animationLoop")
                //    {
                //        std::cout << "Setting intersectionOccurred in animation loop to " << newSCRValue << std::endl;
                //        static_cast<sofa::core::objectmodel::SingleLink<sofa::simulation::Node,sofa::component::animationloop::ZyklioAnimationLoop,0>*>((*it))->get()->setIntersectionOccurred(newSCRValue);
                //    }
                //}

            //}
            // code for intersection detection end

            // code for contact manifold begin
            static inline SReal calcArea4Points(const Vector3 &p0,const Vector3 &p1,const Vector3 &p2,const Vector3 &p3)
            {
                // It calculates possible 3 area constructed from random 4 points and returns the biggest one.

                Vector3 a[3],b[3];
                a[0] = p0 - p1;
                a[1] = p0 - p2;
                a[2] = p0 - p3;
                b[0] = p2 - p3;
                b[1] = p1 - p3;
                b[2] = p1 - p2;

                //todo: Following 3 cross production can be easily optimized by SIMD.
                Vector3 tmp0 = a[0].cross(b[0]);
                Vector3 tmp1 = a[1].cross(b[1]);
                Vector3 tmp2 = a[2].cross(b[2]);

                return std::max(std::max(tmp0.norm2(),tmp1.norm2()),tmp2.norm2());
            }

            static inline int maxAxis4(Vector4 vec)
            {
                int maxIndex = -1;
                SReal maxVal = SReal(-DBL_MAX);
                if (vec[0] > maxVal)
                {
                    maxIndex = 0;
                    maxVal = vec[0];
                }
                if (vec[1] > maxVal)
                {
                    maxIndex = 1;
                    maxVal = vec[1];
                }
                if (vec[2] > maxVal)
                {
                    maxIndex = 2;
                    maxVal =vec[2];
                }
                if (vec[3] > maxVal)
                {
                    maxIndex = 3;
                    maxVal = vec[3];
                }

                return maxIndex;
            }

            // puts the given DetectionOutput into the given contact manifold vector, according to the manifold rules
            // WARNING: does not check contact types, so adding more than one type of
            //          contact to the same vector will probably break contact type counts.
            static void addDetectionToSingleTypeManifold (std::vector< sofa::core::collision::DetectionOutput* > &contactManifoldVector, DetectionOutput* o)
            {
                if (contactManifoldVector.size() < 4)
                {
                    contactManifoldVector.push_back(o);
                }
                else
                {
                    // new point is added according to the following rules:
                    // 1. the point with the smallest contact distance is always kept.
                    // 2. the new point overwrites one of the old points so that
                    //      the remaining three points and the new point cover
                    //      the biggest possible area.

                    int smallestDistanceIndex = -1;
                    SReal smallestDistance = (o->point[0] - o->point[1]).norm2();
                    for (int i=0;i<4;i++)
                    {
                        DetectionOutput* p = contactManifoldVector.at(i);
                        double pointDistance = (p->point[0] - p->point[1]).norm2();
                        if (pointDistance < smallestDistance)
                        {
                            smallestDistanceIndex = i;
                            smallestDistance = pointDistance;
                        }
                    }

                    SReal res0(SReal(0.)),res1(SReal(0.)),res2(SReal(0.)),res3(SReal(0.));

                    if(smallestDistanceIndex != 0) {
                        res0 = calcArea4Points(o->point[0],contactManifoldVector.at(1)->point[0],contactManifoldVector.at(2)->point[0],contactManifoldVector.at(3)->point[0]);
                    }

                    if(smallestDistanceIndex != 1) {
                        res1 = calcArea4Points(o->point[0],contactManifoldVector.at(0)->point[0],contactManifoldVector.at(2)->point[0],contactManifoldVector.at(3)->point[0]);
                    }

                    if(smallestDistanceIndex != 2) {
                        res2 = calcArea4Points(o->point[0],contactManifoldVector.at(0)->point[0],contactManifoldVector.at(1)->point[0],contactManifoldVector.at(3)->point[0]);
                    }

                    if(smallestDistanceIndex != 3) {
                        res3 = calcArea4Points(o->point[0],contactManifoldVector.at(0)->point[0],contactManifoldVector.at(1)->point[0],contactManifoldVector.at(2)->point[0]);
                    }

                    Vector4 maxvec(fabs(res0),fabs(res1),fabs(res2),fabs(res3));
                    int biggestarea = maxAxis4(maxvec);

                    /*std::cout << "inserting DetectionOutput of type " << o->contactType << std::endl;
                    std::cout << "into vector with types ";
                    for (int bla = 0; bla < contactManifoldVector.size(); bla++)
                    {
                        std::cout << contactManifoldVector.at(bla)->contactType << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "at position " << biggestarea << std::endl;*/
                    contactManifoldVector.at(biggestarea) = o;
                }
            }

            void addDetectionToClusterManifold (std::vector< std::pair< std::vector< sofa::core::collision::DetectionOutput* >,Vector3 > > &cmVector, DetectionOutput* o)
            {
                Vector3 newPoint = o->point[0];

                double closestDistance = (cmVector.at(0).second - newPoint).norm2();
                unsigned int closestManifoldIndex = 0;

                double currentDistance;
                if (cmVector.at(0).first.size() > 0)
                {
                    for (unsigned int z = 1; z<cmVector.size(); z++)
                    {
                        if (cmVector.at(z).first.size() == 0) {closestManifoldIndex = z; break;}
                        currentDistance = (cmVector.at(z).second - newPoint).norm2();
                        if ( currentDistance  < closestDistance )
                        {
                            closestDistance = currentDistance;
                            closestManifoldIndex = z;
                        }
                    }
                }

                std::vector< sofa::core::collision::DetectionOutput* > &closestManifold = cmVector.at(closestManifoldIndex).first;
                addDetectionToSingleTypeManifold(closestManifold,o);

                // determine new cluster
                Vector3 cluster(0,0,0);
                for (unsigned int w = 0; w < closestManifold.size(); w++)
                {
                    cluster = cluster + closestManifold.at(w)->point[0];
                }
                cluster = cluster / closestManifold.size();

                cmVector.at(closestManifoldIndex).second = cluster;
            }

            // code for contact manifold end

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::cleanup()
            {
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                if (m_constraintGPU_ll)
                {
                    m_constraintGPU_ll->cleanup();

                    if (parent != NULL)
                        parent->removeObject(m_constraintGPU_ll);

                    m_constraintGPU_ll.reset();
                }
                m_numLineLineContacts = 0;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                if (m_constraintGPU_lv)
                {
                    m_constraintGPU_lv->cleanup();

                    if (parent != NULL)
                        parent->removeObject(m_constraintGPU_lv);

                    m_constraintGPU_lv.reset();
                }
                m_numLineVertexContacts = 0;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                if (m_constraintGPU_vf)
                {
                    m_constraintGPU_vf->cleanup();

                    if (parent != NULL)
                        parent->removeObject(m_constraintGPU_vf);

                    m_constraintGPU_vf.reset();
                }
                m_numVertexFaceContacts = 0;
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                if (m_defaultConstraint)
                {
                    m_defaultConstraint->cleanup();

                    if (parent != NULL)
                        parent->removeObject(m_defaultConstraint);

                    m_defaultConstraint.reset();
                }
                m_numOtherContacts = 0;
#endif

                parent = NULL;

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                mapper1_default.cleanup();

                if (!selfCollision)
                    mapper2_default.cleanup();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                mapper1_gpu_ll.cleanup();
                mapper2_gpu_ll.cleanup();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                mapper1_gpu_lv.cleanup();
                mapper2_gpu_lv.cleanup();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                mapper1_gpu_vf.cleanup();
                mapper2_gpu_vf.cleanup();
#endif

                contacts.clear();

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                mappedContacts.clear();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                mappedContacts_VertexFace.clear();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                mappedContacts_LineLine.clear();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                mappedContacts_LineVertex.clear();
#endif
            }

//#define FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setDetectionOutputs(OutputVector* o)
            {

#ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
                if (testOutputFilename.getValue() != "")
                {
            #ifndef _WIN32
                    //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
            #else
                    //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
            #endif
                    //testOutput << std::endl << "========================================================================================================" << std::endl;
                    //testOutput << "Entering OBBTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setDetectionOutputs(" << o << ")" << std::endl;
                    //testOutput.close();
                }
#endif

#ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
                std::cout << "=== ObbTreeFrictionContact::setDetectionOutputs(" << this->getName() << ") ===" << std::endl;
#endif
                TOutputVector& outputs = *static_cast<TOutputVector*>(o);
                // We need to remove duplicate contacts
				const double minDist2 = 0.0001f /*0.00000001f*/;

                contacts.clear();

                if (model1->getContactStiffness(0) == 0 || model2->getContactStiffness(0) == 0)
                {
                    serr << "Disabled ObbTreeGPUFrictionContact with " << (outputs.size()) << " collision points." << sendl;
#ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
                    std::cerr << "Disabled FrictionContact with " << (outputs.size()) << " collision points." << std::endl;
#endif
                    return;
                }

                // check if contact manifolds should be used.
                // (manifold creation rules are explained below)
                bool createContactManifold = (model1->getUseContactManifolds() && model2->getUseContactManifolds());

                bool useDynamicManifolds = false; // experimental
                unsigned int dynamicManifoldCount = 0;

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                unsigned int vertexFaceCount = 0;
                unsigned int currentVFManifold = 0; // 'current**Manifold' variables track which manifold should be filled right now
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                unsigned int lineLineCount = 0;
                unsigned int currentLLManifold = 0;
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                unsigned int lineVertexCount = 0;
                unsigned int currentLVManifold = 0;
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                unsigned int defaultCount = 0;
                unsigned int currentDManifold = 0;
#endif
                if (createContactManifold)
                {
                    int maximumContactNumber = 0;
                    if (!useDynamicManifolds)
                    {
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                        //lineLineCount = std::max<int>(1,  // needs to be at least 1
                        //                    std::min(model1->getMaxNumberOfLineLineManifolds(),
                        //                             model2->getMaxNumberOfLineLineManifolds())
                        //                    );
                        lineLineCount = std::max<int>(1,  // needs to be at least 1
                            std::min(model1->getMaxNumberOfManifolds(),
                            model2->getMaxNumberOfManifolds())
                            );
                        lineLineManifoldVector.clear();
                        lineLineManifoldVector.resize(lineLineCount);
                        for (unsigned int u = 0; u < lineLineCount; u++)
                        {
                            lineLineManifoldVector.at(u).clear();
                            lineLineManifoldVector.at(u).reserve(4);
                            maximumContactNumber += 4;
                        }
    #endif
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                        //vertexFaceCount = std::max<int>(1,  // needs to be at least 1
                        //                      std::min(model1->getMaxNumberOfFaceVertexManifolds(),
                        //                               model2->getMaxNumberOfFaceVertexManifolds())
                        //                      );
                        vertexFaceCount = std::max<int>(1,  // needs to be at least 1
                            std::min(model1->getMaxNumberOfManifolds(),
                            model2->getMaxNumberOfManifolds())
                            );
                        faceVertexManifoldVector.clear();
                        faceVertexManifoldVector.resize(vertexFaceCount);
                        for (unsigned int u = 0; u < vertexFaceCount; u++)
                        {
                            faceVertexManifoldVector.at(u).clear();
                            faceVertexManifoldVector.at(u).reserve(4);
                            maximumContactNumber += 4;
                        }
    #endif
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                        lineVertexCount = 1;
                        lineVertexManifold.clear();
                        lineVertexManifold.resize(lineVertexCount);
                        for (unsigned int u = 0; u < lineVertexCount; u++)
                        {
                            lineVertexManifold.at(u).clear();
                            lineVertexManifold.at(u).reserve(4);
                            maximumContactNumber += 4;
                        }
    #endif
    #ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                        /*defaultCount = 1;*/
                        defaultCount = std::max<int>(1,  // needs to be at least 1
                            std::min(model1->getMaxNumberOfManifolds(),
                            model2->getMaxNumberOfManifolds())
                            );
                        defaultManifoldVector.clear();
                        defaultManifoldVector.resize(defaultCount);
                        for (unsigned int u = 0; u < defaultCount; u++)
                        {
                            defaultManifoldVector.at(u).clear();
                            defaultManifoldVector.at(u).reserve(4);
                            maximumContactNumber += 4;
                        }
    #endif
                        contacts.reserve(maximumContactNumber);
                    }
                    else
                    {
                        dynamicManifoldCount = 4; // TODO, replace with configurable value
                        dynamicManifoldVector.clear();
                        dynamicManifoldVector.resize(dynamicManifoldCount);
                        for (unsigned int u = 0; u < dynamicManifoldCount; u++)
                        {
                            dynamicManifoldVector.at(u).first.clear();
                            dynamicManifoldVector.at(u).first.reserve(4);
                            dynamicManifoldVector.at(u).second = Vector3(0,0,0);
                            maximumContactNumber += 4;
                        }
                        contacts.reserve(maximumContactNumber);
                    }
                }
                else
                {
                    // if contact manifolds are switched off, standard contact creation is done
                    contacts.reserve(outputs.size());
                }

                int SIZE = outputs.size();
#ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
                std::cout << " contacts output size = " << SIZE << std::endl;
#endif

#ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
				if (testOutputFilename.getValue() != "")
				{
#ifndef _WIN32
                    //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
#else
                    //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
#endif
                    //testOutput << " contacts output size = " << SIZE << std::endl;
                    //testOutput.close();
				}
#endif

				int contactsAdded = 0, contactsIgnored = 0;
                // the following procedure cancels the duplicated detection outputs
                for (int cpt=0; cpt<SIZE; cpt++)
                {
					// Max. 10 contacts, testing!
					//if (contacts.size() > 10)
					//	break;

                    DetectionOutput* o = &outputs[cpt];

                    bool found = false;
                    for (unsigned int i=0; i<contacts.size() && !found; i++)
                    {
                        DetectionOutput* p = contacts[i];

                        Vector3 op0 = o->point[0] - p->point[0];
                        Vector3 op1 = o->point[1] - p->point[1];

                        Vector3 po0 = o->point[0] - p->point[1];
                        Vector3 po1 = o->point[1] - p->point[0];

#ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
                        if (testOutputFilename.getValue() != "")
                        {
            #ifndef _WIN32
                            //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
            #else
                            //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
            #endif
                            //testOutput << "      -> contact " << o->point[0] << " - " << o->point[1] << " vs. " << p->point[0] << " - " << p->point[1] << ": op0 = " << op0 << ", op1 = " << op1;
                            //testOutput << "; po0 = " << po0 << ", po1 = " << po1;
                            //testOutput << "; op0.norm2() = " << op0.norm2() << ", op1.norm2() = " << op1.norm2();
                            //testOutput << "; po0.norm2() = " << po0.norm2() << ", po1.norm2() = " << po1.norm2();
                            //testOutput << "; op0.norm2() + op1.norm2() = " << op0.norm2() + op1.norm2();
                            //testOutput << "; po0.norm2() + po1.norm2() = " << po0.norm2() + po1.norm2();
                            //testOutput << "; minDist2 = " << minDist2 << std::endl;

                            //testOutput.close();
                        }
#endif
                        if (op0.norm2() + op1.norm2() < minDist2 ||
                            po0.norm2() + po1.norm2() < minDist2 /*||
                            po0.norm2() < minDist2 ||
                            po1.norm2() < minDist2*/)
                        {
            #ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
							std::cout << " - ignore " << cpt << ": " << *o << "; " << op0.norm2() + op1.norm2() << " <= " << minDist2 << ", " << po0.norm2() + po1.norm2()  << " <= " << minDist2 << std::endl;
            #endif
							contactsIgnored++;
                            found = true;
#ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
                            if (testOutputFilename.getValue() != "")
                            {
                #ifndef _WIN32
                                //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
                #else
                                //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
                #endif
                                if (o->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                                    //testOutput << " - IGNORED contact: FACE_VERTEX -- " << *o << std::endl;
                                else if (o->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                                    //testOutput << " - IGNORED contact: LINE_LINE   -- " << *o << std::endl;
                                else if (o->contactType == sofa::core::collision::CONTACT_LINE_VERTEX)
                                    //testOutput << " - IGNORED contact: LINE_VERTEX   -- " << *o << std::endl;
                                else
                                    //testOutput << " - IGNORED contact: OTHER       -- " << *o << std::endl;


                                //testOutput.close();
                            }
#endif
                        }
                    }

                    if (!found)
                    {
            #ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
                            std::cout << " - add " << cpt << ": " << *o << std::endl;
            #endif
                            if (createContactManifold)
                            {
                                if (!useDynamicManifolds)
                                {
            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                                    if (o->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                                    {
                                        addDetectionToSingleTypeManifold(faceVertexManifoldVector.at(currentVFManifold),o);
                                        currentVFManifold++;
                                        currentVFManifold = currentVFManifold % vertexFaceCount;
                                    }
            #endif

            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                                    if (o->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                                    {
                                        addDetectionToSingleTypeManifold(lineLineManifoldVector.at(currentLLManifold),o);
                                        currentLLManifold++;
                                        currentLLManifold = currentLLManifold % lineLineCount;
                                    }
            #endif

            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                                    if (o->contactType == sofa::core::collision::CONTACT_LINE_VERTEX)
                                    {
                                        addDetectionToSingleTypeManifold(lineVertexManifoldVector.at(currentLVManifold),o);
                                        currentLVManifold++;
                                        currentLVManifold = currentLVManifold % lineVertexCount;
                                    }
            #endif

            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                                    if (o->contactType == sofa::core::collision::CONTACT_INVALID)
                                    {
                                        addDetectionToSingleTypeManifold(defaultManifoldVector.at(currentDManifold),o);
                                        currentDManifold++;
                                        currentDManifold = currentDManifold % defaultCount;
                                    }
            #endif
                                }
                                else
                                {
                                    addDetectionToClusterManifold(dynamicManifoldVector,o);
                                }
                            }
                            else
                            {
                                contacts.push_back(o);
                                contactsAdded++;

        #ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                                if (o->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                                    m_numVertexFaceContacts++;
        #endif

        #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                                if (o->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                                    m_numLineLineContacts++;
        #endif

        #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                                if (o->contactType == sofa::core::collision::CONTACT_LINE_VERTEX)
                                    m_numLineVertexContacts++;
        #endif

        #ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                                if (o->contactType == sofa::core::collision::CONTACT_INVALID)
                                    m_numOtherContacts++;
        #endif
                            }
/*
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                        if (o->contactType == sofa::core::collision::CONTACT_LINE_VERTEX)
                            m_numLineVertexContacts++;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                        if (o->contactType == sofa::core::collision::CONTACT_INVALID)
                            m_numOtherContacts++;
#endif
*/

#ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
                        if (testOutputFilename.getValue() != "")
                        {
            #ifndef _WIN32
                            //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
            #else
                            //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
            #endif
                            if (o->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                                //testOutput << " + added contact: FACE_VERTEX -- " << *o << std::endl;
                            else if (o->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                                //testOutput << " + added contact: LINE_LINE   -- " << *o << std::endl;
                            else if (o->contactType == sofa::core::collision::CONTACT_LINE_VERTEX)
                                //testOutput << " + added contact: LINE_VERTEX   -- " << *o << std::endl;
                            else
                                //testOutput << " + added contact: OTHER       -- " << *o << std::endl;


                            //testOutput.close();
                        }
#endif
                    }
                }
                if (createContactManifold)
                {
                    if (!useDynamicManifolds)
                    {
            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                        m_numVertexFaceContacts = 0;
                        for (unsigned int u = 0; u < vertexFaceCount; u++)
                        {
                            for (unsigned int k=0; k<faceVertexManifoldVector.at(u).size(); k++)
                            {
                                contacts.push_back(faceVertexManifoldVector.at(u).at(k));
                            }
                            m_numVertexFaceContacts += faceVertexManifoldVector.at(u).size();
                        }
                        contactsAdded += m_numVertexFaceContacts;
            #endif

            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                        m_numLineLineContacts = 0;
                        for (unsigned int u = 0; u < lineLineCount; u++)
                        {
                            for (unsigned int k=0; k<lineLineManifoldVector.at(u).size(); k++)
                            {
                                contacts.push_back(lineLineManifoldVector.at(u).at(k));
                            }
                            m_numLineLineContacts += lineLineManifoldVector.at(u).size();
                        }
                        contactsAdded += m_numLineLineContacts;
            #endif

            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                        m_numLineVertexContacts = 0;
                        for (unsigned int u = 0; u < lineVertexCount; u++)
                        {
                            for (unsigned int k=0; k<lineVertexManifoldVector.at(u).size(); k++)
                            {
                                contacts.push_back(lineVertexManifoldVector.at(u).at(k));
                            }
                            m_numLineVertexContacts += lineVertexManifoldVector.at(u).size();
                        }
                        contactsAdded += m_numLineVertexContacts;
            #endif

            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                        m_numOtherContacts = 0;
                        for (unsigned int u = 0; u < defaultCount; u++)
                        {
                            for (unsigned int k=0; k<defaultManifoldVector.at(u).size(); k++)
                            {
                                contacts.push_back(defaultManifoldVector.at(u).at(k));
                            }
                            m_numOtherContacts += defaultManifoldVector.at(u).size();
                        }
                        contactsAdded += m_numOtherContacts;
            #endif
                    }
                    else
                    {
                        for (unsigned int u = 0; u < dynamicManifoldCount; u++)
                        {
                            for (unsigned int k=0; k<dynamicManifoldVector.at(u).first.size(); k++)
                            {
                                DetectionOutput* currentO = dynamicManifoldVector.at(u).first.at(k);
        #ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                                if (currentO->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                                    m_numVertexFaceContacts++;
        #endif

        #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                                if (currentO->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                                    m_numLineLineContacts++;
        #endif

        #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                                if (currentO->contactType == sofa::core::collision::CONTACT_LINE_VERTEX)
                                    m_numLineVertexContacts++;
        #endif

        #ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                                if (currentO->contactType == sofa::core::collision::CONTACT_INVALID)
                                    m_numOtherContacts++;
        #endif
                                contacts.push_back(currentO);
                                contactsAdded++;
                            }
                        }
                    }
                }
                //std::cout << "contactsAdded=" << contactsAdded << std::endl;
                //std::cout << "contacts.size()=" << contacts.size() << std::endl;

                if (contacts.size() < outputs.size() && this->f_printLog.getValue())
                {
                    // DUPLICATED CONTACTS FOUND
                    sout << "Removed " << (outputs.size()-contacts.size()) <<" / " << outputs.size() << " collision points." << sendl;
#ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
                    std::cout << "Removed " << (outputs.size()-contacts.size()) <<" / " << outputs.size() << " collision points." << std::endl;
#endif
                }

#ifdef FRICTIONCONTACT_DEBUG_SETDETECTIONOUTPUTS
                if (testOutputFilename.getValue() != "")
                {
            #ifndef _WIN32
                    //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
            #else
                    //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
            #endif
                    //testOutput << "Contacts added = " << contactsAdded << ", contacts ignored = " << contactsIgnored << std::endl;

					for (unsigned int i = 0; i < contacts.size(); i++)
					{
                        //testOutput << " -> contact " << *(contacts[i]) << std::endl;
					}

                    //testOutput << "Leaving OBBTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setDetectionOutputs(" << o << ")" << std::endl;
                    //testOutput << "========================================================================================================" << std::endl;
                    //testOutput.close();
                }
#endif
            }

//#define OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::activateMappers()
            {
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                if (testOutputFilename.getValue() != "")
                {
            #ifndef _WIN32
                    //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
            #else
                    //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
            #endif
                    //testOutput << std::endl << "========================================================================================================" << std::endl;
                    //testOutput << "Entering OBBTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::activateMappers()" << std::endl;
                    //testOutput.close();
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                std::cout << "=== " << this->getName() << ": begin activateMappers call ===" << std::endl;
#endif
                mmodel1 = NULL;
                mmodel2 = NULL;
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                if (!m_defaultConstraint)
                {
                    // Get the mechanical model from mapper1 to fill the constraint vector
                    std::string mapping_name1(this->getName() + "_Mapping1_Default");
                    // Get the mechanical model from mapper1 to fill the constraint vector
                    mmodel1 = mapper1_default.createMapping(mapping_name1.c_str());
                    // Get the mechanical model from mapper2 to fill the constraints vector
                    std::string mapping_name2(this->getName() + "_Mapping2_Default");
                    mmodel2 = selfCollision ? mmodel1 : mapper2_default.createMapping(mapping_name2.c_str());

                    m_defaultConstraint = sofa::core::objectmodel::New<constraintset::UnilateralInteractionConstraint<ResponseDataTypes> >(mmodel1, mmodel2);
                    m_defaultConstraint->setName( getName() + "_defaultConstraint" );
                    setInteractionTags(mmodel1, mmodel2, m_defaultConstraint.get());
                    m_defaultConstraint->setCustomTolerance( tol.getValue() );
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                if (!m_constraintGPU_ll)
                {
                    std::string mapping_name1(this->getName() + "_Mapping1_GPU_LineLine");
                    mmodel1 = mapper1_gpu_ll.createMapping(mapping_name1.c_str());
                    std::string mapping_name2(this->getName() + "_Mapping2_GPU_LineLine");
                    mmodel2 = mapper2_gpu_ll.createMapping(mapping_name2.c_str());
                    m_constraintGPU_ll = sofa::core::objectmodel::New<constraintset::UnilateralInteractionConstraint<ResponseDataTypes> >(mmodel1, mmodel2);
                    m_constraintGPU_ll->setName( getName() + "_constraintGPU_LineLine" );
                    setInteractionTags(mmodel1, mmodel2, m_constraintGPU_ll.get());
                    m_constraintGPU_ll->setCustomTolerance( tol.getValue() );
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                if (!m_constraintGPU_lv)
                {
                    std::string mapping_name1(this->getName() + "_Mapping1_GPU_LineVertex");
                    mmodel1 = mapper1_gpu_lv.createMapping(mapping_name1.c_str());
                    std::string mapping_name2(this->getName() + "_Mapping2_GPU_LineVertex");
                    mmodel2 = mapper2_gpu_lv.createMapping(mapping_name2.c_str());
                    m_constraintGPU_lv = sofa::core::objectmodel::New<constraintset::UnilateralInteractionConstraint<ResponseDataTypes> >(mmodel1, mmodel2);
                    m_constraintGPU_lv->setName( getName() + "_constraintGPU_LineVertex" );
                    setInteractionTags(mmodel1, mmodel2, m_constraintGPU_lv.get());
                    m_constraintGPU_lv->setCustomTolerance( tol.getValue() );
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                if (!m_constraintGPU_vf)
                {
                    std::string mapping_name1(this->getName() + "_Mapping1_GPU_VertexFace");
                    mmodel1 = mapper1_gpu_vf.createMapping(mapping_name1.c_str());
                    std::string mapping_name2(this->getName() + "_Mapping2_GPU_VertexFace");
                    mmodel2 = mapper2_gpu_vf.createMapping(mapping_name2.c_str());
                    m_constraintGPU_vf = sofa::core::objectmodel::New<constraintset::UnilateralInteractionConstraint<ResponseDataTypes> >(mmodel1, mmodel2);
                    m_constraintGPU_vf->setName( getName() + "_constraintGPU_VertexFace" );
                    setInteractionTags(mmodel1, mmodel2, m_constraintGPU_vf.get());
                    m_constraintGPU_vf->setCustomTolerance( tol.getValue() );
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                if (testOutputFilename.getValue() != "")
                {
            #ifndef _WIN32
                    //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
            #else
                    //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
            #endif
                    //testOutput << "Contact type counts: ";
            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                    //testOutput << "LINE_LINE = " << m_numLineLineContacts << ";";
            #endif
            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                    //testOutput << "LINE_VERTEX = " << m_numLineVertexContacts << ";";
            #endif
            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                    //testOutput << "FACE_VERTEX = " << m_numVertexFaceContacts << ";";
            #endif
            #ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                    //testOutput << "OTHER = " << m_numOtherContacts << ";";
            #endif
                    //testOutput << std::endl;
                    //testOutput.close();
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                std::cout << "==> Contact type counts <==" << std::endl;
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                std::cout << " other = " << m_numOtherContacts << std::endl;
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                std::cout << " line_line = " << m_numLineLineContacts << std::endl;
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                std::cout << " line_line = " << m_numLineLineContacts << std::endl;
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                std::cout << " vertex_face = " << m_numVertexFaceContacts << std::endl;
#endif
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                if (selfCollision)
                    mapper1_default.resize(2*m_numOtherContacts);
                else
                {
                    mapper1_default.resize(m_numOtherContacts);
                    mapper2_default.resize(m_numOtherContacts);
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                mapper1_gpu_ll.resize(m_numLineLineContacts);
                mapper2_gpu_ll.resize(m_numLineLineContacts);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                mapper1_gpu_lv.resize(m_numLineVertexContacts);
                mapper2_gpu_lv.resize(m_numLineVertexContacts);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                mapper1_gpu_vf.resize(m_numVertexFaceContacts);
                mapper2_gpu_vf.resize(m_numVertexFaceContacts);
#endif
                int i = 0;
                const double d0 = intersectionMethod->getContactDistance() + model1->getProximity() + model2->getProximity(); // - 0.001;

                /*
                std::cout<<"m_numLineLineContacts = "<<m_numLineLineContacts<<std::endl;
                std::cout<<"m_numVertexFaceContacts = "<<m_numVertexFaceContacts<<std::endl;
                std::cout<<"m_numOtherContacts = "<<m_numOtherContacts<<std::endl;
                std::cout<<" d0 = "<<d0<<std::endl;
                */

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                m_constraintGPU_ll->clear(m_numLineLineContacts);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                m_constraintGPU_lv->clear(m_numLineVertexContacts);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                m_constraintGPU_vf->clear(m_numVertexFaceContacts);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                m_defaultConstraint->clear(m_numOtherContacts);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                mappedContacts.resize(m_numOtherContacts);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                mappedContacts_VertexFace.resize(m_numVertexFaceContacts);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                mappedContacts_LineLine.resize(m_numLineLineContacts);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                mappedContacts_LineVertex.resize(m_numLineVertexContacts);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                std::cout << "Add contacts: " << contacts.size() << std::endl;
#endif
				m_numContacts = 0;
                int lineLineIdx = 0, vertexFaceIdx = 0, lineVertexIdx = 0;
                for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
                {
                    DetectionOutput* o = *it;
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                    std::cout << " * type: ";
                    if (o->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                        std::cout << "LINE_LINE";
                    else if (o->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                        std::cout << "FACE_VERTEX";
                    else
                        std::cout << "INVALID";

                    std::cout <<  ", collisionElements: " << o->elem.first.getIndex() << " - " << o->elem.second.getIndex() << std::endl;
#endif
                    int index1, index2;
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                    if (o->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                    {
                        int triIdx1 = o->elem.first.getIndex() / 3;
                        int triIdx2 = o->elem.second.getIndex() / 3;

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                        int vertexIdx1 = o->elem.first.getIndex() % 3;
                        int vertexIdx2 = o->elem.second.getIndex() % 3;
                        std::cout << "   indices: triangle " << triIdx1 << ", vertex " << vertexIdx1 << " - triangle " << triIdx2 << ", vertex " << vertexIdx2 << std::endl;
#endif
                        typename DataTypes1::Real r1 = 0.;
                        typename DataTypes2::Real r2 = 0.;
                        // Create mapping for first point
                        index1 = mapper1_gpu_vf.addPointB(o->point[0], triIdx2, r1
                    #ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                    , o->baryCoords[0]
                    #endif
                                    , o->contactType
                                                      );
                        // Create mapping for second point
                        index2 = mapper2_gpu_vf.addPointB(o->point[1], triIdx1, r2
                #ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                    , o->baryCoords[1]
                #endif
                                    , o->contactType
                                                      );

                        double distance = d0 + r1 + r2;

                        mappedContacts_VertexFace[vertexFaceIdx].first.first = index1;
                        mappedContacts_VertexFace[vertexFaceIdx].first.second = index2;
                        mappedContacts_VertexFace[vertexFaceIdx].second = distance;

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                        if (testOutputFilename.getValue() != "")
                        {
                    #ifndef _WIN32
                            //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
                    #else
                            //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
                    #endif
                            //testOutput << " * FACE_VERTEX contact stored as mappedContacts_VertexFace[" << vertexFaceIdx << "]: triangles " << triIdx1 << " - " << triIdx2 << "; index1 = " << index1 << ", index2 = " << index2 << ", distance = " << distance << " (" << d0 << " + " << r1 << " + " << r2 << ")" << std::endl;
                            //testOutput.close();
                        }
#endif

                        vertexFaceIdx++;
                    }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                    if (o->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                    {
                        int triIdx1 = o->elem.first.getIndex() / 3;
                        int triIdx2 = o->elem.second.getIndex() / 3;

                        int edgeIdx1 = o->elem.first.getIndex() % 3;
                        int edgeIdx2 = o->elem.second.getIndex() % 3;
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                        std::cout << "   indices: triangle1 " << triIdx1 << ", EDGE1 " << edgeIdx1 << " - triangle2 " << triIdx2 << ", EDGE2 " << edgeIdx2 << std::endl;
#endif
                        if (triIdx1 < this->model1->getMeshTopology()->getNbTriangles() &&
                            triIdx2 < this->model2->getMeshTopology()->getNbTriangles())
                        {
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                            std::cout << "   triangle1 = " << triIdx1 << " AND triangle2 = " << triIdx2 << " valid." << std::endl;
#endif
                            const sofa::core::topology::BaseMeshTopology::EdgesInTriangle& e1 = this->model1->getMeshTopology()->getEdgesInTriangle(triIdx1);
                            const sofa::core::topology::BaseMeshTopology::EdgesInTriangle& e2 = this->model2->getMeshTopology()->getEdgesInTriangle(triIdx2);
                            int edgeId1 = e1[edgeIdx1];
                            int edgeId2 = e2[edgeIdx2];

                            if (edgeId1 < this->model1->getMeshTopology()->getNbEdges() &&
                                edgeId2 < this->model2->getMeshTopology()->getNbEdges())
                            {
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                                std::cout << "   edgeIdx1 = " << edgeIdx1 << " AND edgeIdx2 = " << edgeIdx2 << " valid: Correspond to edgeID1 = " << edgeId1 << " and edgeID2 = " << edgeId2 << std::endl;
#endif
                                typename DataTypes1::Real r1 = 0.;
                                typename DataTypes2::Real r2 = 0.;
                                // Create mapping for first point
                                index1 = mapper1_gpu_ll.addPointB(o->point[0], edgeId1, r1
                            #ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                            , o->baryCoords[0]
                            #endif
                                            , o->contactType
                                                              );
                                // Create mapping for second point
                                index2 = mapper2_gpu_ll.addPointB(o->point[1], edgeId2, r2
                        #ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                            , o->baryCoords[1]
                        #endif
                                            , o->contactType
                                                              );

                                double distance = d0 + r1 + r2;

                                mappedContacts_LineLine[lineLineIdx].first.first = index1;
                                mappedContacts_LineLine[lineLineIdx].first.second = index2;
                                mappedContacts_LineLine[lineLineIdx].second = distance;

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                                if (testOutputFilename.getValue() != "")
                                {
                            #ifndef _WIN32
                                    //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
                            #else
                                    //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
                            #endif
                                    //testOutput << " * LINE_LINE contact stored as mappedContacts_LineLine[" << lineLineIdx << "]: triangles " << triIdx1 << " - " << triIdx2 << ", edges " << edgeId1 << " - " << edgeId2 << "; index1 = " << index1 << ", index2 = " << index2 << ", distance = " << distance << " (" << d0 << " + " << r1 << " + " << r2 << ")" << std::endl;
                                    //testOutput.close();
                                }
#endif

                                lineLineIdx++;
                            }
                        }
                    }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                    if (o->contactType == sofa::core::collision::CONTACT_LINE_VERTEX)
                    {
                        typename DataTypes2::Real r2 = 0.;
                        typename DataTypes1::Real r1 = 0.;

                        /// TODO: LINE_VERTEX case!?!
                        /*int triIdx1 = o->elem.first.getIndex() / 3;
                        int triIdx2 = o->elem.second.getIndex() / 3;

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                        int vertexIdx1 = o->elem.first.getIndex() % 3;
                        int vertexIdx2 = o->elem.second.getIndex() % 3;
                        std::cout << "   indices: triangle " << triIdx1 << ", vertex " << vertexIdx1 << " - triangle " << triIdx2 << ", vertex " << vertexIdx2 << std::endl;
#endif
                        // Create mapping for first point
                        index1 = mapper1_gpu_lv.addPointB(o->point[0], triIdx2, r1
                    #ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                    , o->baryCoords[0]
                    #endif
                                    , o->contactType
                                                      );
                        // Create mapping for second point
                        index2 = mapper2_gpu_lv.addPointB(o->point[1], triIdx1, r2
                #ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                    , o->baryCoords[1]
                #endif
                                    , o->contactType
                                                      );*/

                        double distance = d0 + r1 + r2;

                        mappedContacts_LineVertex[lineLineIdx].first.first = index1;
                        mappedContacts_LineVertex[lineLineIdx].first.second = index2;
                        mappedContacts_LineVertex[lineLineIdx].second = distance;

                        lineVertexIdx++;
                    }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                    if (o->contactType == sofa::core::collision::CONTACT_INVALID)
                    {
                        CollisionElement1 elem1(o->elem.first);
                        CollisionElement2 elem2(o->elem.second);

                        index1 = elem1.getIndex();
                        index2 = elem2.getIndex();
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                        std::cout << ", indices: " << index1 << " - " << index2 << std::endl;
#endif

                        typename DataTypes1::Real r1 = 0.;
                        typename DataTypes2::Real r2 = 0.;
                        //double constraintValue = ((o->point[1] - o->point[0]) * o->normal) - intersectionMethod->getContactDistance();

                        // Create mapping for first point
                        {
                            index1 = mapper1_default.addPointB(o->point[0], index1, r1
                    #ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                    , o->baryCoords[0]
                    #endif
                                    //, o->contactType // a mapper for SOFA CPU-contact models is used, which does not take a contactType
                                                      );
                        }
                        // Create mapping for second point
                        if (selfCollision)
                        {
                            {
                                index2 = mapper1_default.addPointB(o->point[1], index2, r2
                    #ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                        , o->baryCoords[1]
                    #endif
                                        //, o->contactType // a mapper for SOFA CPU-contact models is used, which does not take a contactType
                                                          );
                            }
                        }
                        else
                        {
                            {
                                index2 = mapper2_default.addPointB(o->point[1], index2, r2
                    #ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                                        , o->baryCoords[1]
                    #endif
                                        //, o->contactType // a mapper for SOFA CPU-contact models is used, which does not take a contactType
                                                          );
                            }
                        }
                        double distance = d0 + r1 + r2;

                        
                        mappedContacts[i].first.first = index1;
                        mappedContacts[i].first.second = index2;
                        mappedContacts[i].second = distance;
                    }
#endif
					m_numContacts++;
                }

				std::cout << "======================================================" << std::endl;
				std::cout << this->getName() << ": " << m_numContacts << " contacts." << std::endl;
				std::cout << "======================================================" << std::endl;

                // Update mappings
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                mapper1_default.update();
                mapper1_default.updateXfree();

                if (!selfCollision) mapper2_default.update();
                if (!selfCollision) mapper2_default.updateXfree();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                mapper1_gpu_ll.update();
                mapper1_gpu_ll.updateXfree();

                mapper2_gpu_ll.update();
                mapper2_gpu_ll.updateXfree();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                mapper1_gpu_lv.update();
                mapper1_gpu_lv.updateXfree();

                mapper2_gpu_lv.update();
                mapper2_gpu_lv.updateXfree();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                mapper1_gpu_vf.update();
                mapper1_gpu_vf.updateXfree();

                mapper2_gpu_vf.update();
                mapper2_gpu_vf.updateXfree();
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                std::cout << "=== " << this->getName() << ": end activateMappers call ===" << std::endl;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_ACTIVATEMAPPERS
                if (testOutputFilename.getValue() != "")
                {
            #ifndef _WIN32
                    //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
            #else
                    //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
            #endif
                    //testOutput << "Leaving OBBTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::activateMappers()" << std::endl;
                    //testOutput << "========================================================================================================" << std::endl;
                    //testOutput.close();
                }
#endif
#endif
            }

// #define OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::createResponse(core::objectmodel::BaseContext* group)
            {
#ifndef OBBTREEGPUFRICTIONCONTACT_SUPPRESS_CONTACT_RESPONSE
                activateMappers();
                const double mu_ = this->mu.getValue();
                // Checks if friction is considered
                if ( mu_ < 0.0 )
                    serr << sendl << "Error: mu has to take positive values" << sendl;

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                if (testOutputFilename.getValue() != "")
                {
            #ifndef _WIN32
                    //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
            #else
                    //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
            #endif
                    //testOutput << std::endl << "========================================================================================================" << std::endl;
                    //testOutput << "Entering ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::createResponse()" << std::endl;
                    //testOutput.close();
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                std::cout << "=== begin ObbTreeGPUFrictionContact::createResponse(" << this->getName() << ") call ===" << std::endl;
#endif

                int i = 0;
                int index1, index2;
                double distance;

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                std::cout << "=== process contacts: " << this->getName() << "; size = " << contacts.size() << " ===" << std::endl;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                if (testOutputFilename.getValue() != "")
                {
            #ifndef _WIN32
                    //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
            #else
                    //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
            #endif
                    //testOutput << "Contacts to add: " << contacts.size() << std::endl;
                    //testOutput.close();
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                if (m_constraintGPU_ll)
                {
                    for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++)
                    {
                        DetectionOutput* o = *it;

                        if (o->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                        {
                            index1 = mappedContacts_LineLine[i].first.first;
                            index2 = mappedContacts_LineLine[i].first.second;
                            distance = mappedContacts_LineLine[i].second;

                            //==================================================================================
                            //std::cout << "Banane! BANANE!" << std::endl;

                            int triIdx1 = o->elem.first.getIndex() / 3;
                            int triIdx2 = o->elem.second.getIndex() / 3;

                            /*
                            int edgeIdx1 = o->elem.first.getIndex() % 3;
                            int edgeIdx2 = o->elem.second.getIndex() % 3;

                            int edgeId1;
                            int edgeId2;
                            */
#ifdef DO_INTERSECTION_TEST_FOR_TRIANGLES
                            if (triIdx1 < this->model1->getMeshTopology()->getNbTriangles() &&
                                triIdx2 < this->model2->getMeshTopology()->getNbTriangles())

                            {
                                if (triIdx1 >= this->model1->getMeshTopology()->getNbTriangles())
                                {
                                    std::cout << "ERROR: triIdx1 >= this->model1->getMeshTopology()->getNbTriangles(); triIdx1 = " << triIdx1 << "this->model1->getMeshTopology()->getNbTriangles() = " << this->model1->getMeshTopology()->getNbTriangles() << std::endl;
                                }
                                if (triIdx2 >= this->model2->getMeshTopology()->getNbTriangles())
                                {
                                    std::cout << "ERROR: triIdx2 >= this->model2->getMeshTopology()->getNbTriangles(); triIdx2 = " << triIdx2 << "this->model2->getMeshTopology()->getNbTriangles() = " << this->model2->getMeshTopology()->getNbTriangles() << std::endl;
                                }
                            }
                            else
                            {
                                sofa::core::topology::Triangle contactTriangle1, contactTriangle2;                                

                                contactTriangle1 = model1->getContext()->getMeshTopology()->getTriangle(triIdx1);
                                contactTriangle2 = model2->getContext()->getMeshTopology()->getTriangle(triIdx2);

                                sofa::component::container::MechanicalObject<Vec3Types>* model1MechanicalState = dynamic_cast<sofa::component::container::MechanicalObject<Vec3Types>*>(model1->getContext()->getMechanicalState()) ;
                                sofa::component::container::MechanicalObject<Vec3Types>* model2MechanicalState = dynamic_cast<sofa::component::container::MechanicalObject<Vec3Types>*>(model2->getContext()->getMechanicalState()) ;

                                if ((!model1MechanicalState) || (!model2MechanicalState))
                                {
                                    if (!model1MechanicalState)
                                    {
                                        std::cout << "ERROR: dynamic_cast from 'model1->getContext()->getMechanicalState()' failed.";
                                    }
                                    if (!model2MechanicalState)
                                    {
                                        std::cout << "ERROR: dynamic_cast from 'model2->getContext()->getMechanicalState()' failed.";
                                    }
                                }
                                else
                                {
                                    // duplicate code? begin
                                    const Vec3Types::VecCoord& model1PositionVector = model1MechanicalState->read(core::ConstVecCoordId::position())->getValue();
                                    const Vec3Types::VecCoord& model2PositionVector = model2MechanicalState->read(core::ConstVecCoordId::position())->getValue();
                                    /*BVHModels::Triangle3<SReal> firstTriangle(model1PositionVector.at(contactTriangle2.at(0)),
                                                                              model1PositionVector.at(contactTriangle2.at(1)),
                                                                              model1PositionVector.at(contactTriangle2.at(2)));
                                    BVHModels::Triangle3<SReal> secondTriangle(model2PositionVector.at(contactTriangle1.at(0)),
                                                                               model2PositionVector.at(contactTriangle1.at(1)),
                                                                               model2PositionVector.at(contactTriangle1.at(2)));*/
                                    BVHModels::Triangle3<SReal> firstTriangle(model1PositionVector.at(contactTriangle1.at(0)),
                                                                              model1PositionVector.at(contactTriangle1.at(1)),
                                                                              model1PositionVector.at(contactTriangle1.at(2)));
                                    BVHModels::Triangle3<SReal> secondTriangle(model2PositionVector.at(contactTriangle2.at(0)),
                                                                               model2PositionVector.at(contactTriangle2.at(1)),
                                                                               model2PositionVector.at(contactTriangle2.at(2)));
                                                                //always check against the triangle of the same model;

                                    //std::cout << "firstTriangle " << firstTriangle.V[0] << ", " << firstTriangle.V[1] << ", " << firstTriangle.V[2] <<  std::endl;
                                    //std::cout << "secondTriangle " << secondTriangle.V[0] << ", " << secondTriangle.V[1] << ", " << secondTriangle.V[2] <<  std::endl;

                                    BVHModels::Triangle3IntersectionResult<SReal> intersectionResult;

                                    if (firstTriangle.Find(secondTriangle,intersectionResult))
                                    {
                                        // TODO
                                        std::cout << "Edge intersetion." << std::endl;
                                        setIntersectionOccurred(true);
                                        /*
                                        std::cout << "intersectionResult.mQuantity: " << intersectionResult.mQuantity << std::endl;
                                        for (int i=0; i<intersectionResult.mQuantity; i++)
                                        {
                                            std::cout << "intersectionResult.mPoint[" << i << "]: ";
                                            std::cout << intersectionResult.mPoint[i];
                                            std::cout << std::endl;
                                        }
                                        std::cout << "intersectionResult.GetIntersectionType(): " << intersectionResult.GetIntersectionType() << std::endl;
                                        */
                                    }
                                    else
                                    {
                                        //setIntersectionOccurred(false);
                                        //std::cout << "Triangles do not intersect." << std::endl;
                                    }

                                    /*for (int i=0; i<10; i++)
                                    {
                                        std::cout << "BananeEnde" << std::endl;
                                    }*/
                                    // duplicate code? end
                                }
                            }

                            //std::cout << "BANAAAAAAAAANE!" << std::endl;
                            //==================================================================================
#endif
                            typename ResponseDataTypes::Deriv normalVec(o->normal.x(), o->normal.y(), o->normal.z());

                            // Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
                            long index = cantorPolynomia(o->id /*cantorPolynomia(index1, index2)*/,id);
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                            std::cout << " * contact " << i << ": index1 = " << index1 << ", index2 = " << index2 <<  " id = " << o->id << ", elem1 = " << o->elem.first.getIndex() << ", elem2 = " << o->elem.second.getIndex() << ", distance = " << o->value;
                            std::cout << ", point0 = " << o->point[0] << ", point1 = " << o->point[1] << ", normal = " << o->normal << std::endl;

                            std::cout << "    add LINE_LINE contact to m_constraintGPU_ll" << std::endl;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                            std::string contactType("LINE_LINE");
                            if (testOutputFilename.getValue() != "")
                            {
                                #ifndef _WIN32
                                        //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
                                #else
                                        //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
                                #endif
                                //testOutput << " * contact " << i << ", type = " << contactType << ": index1 = " << index1 << ", index2 = " << index2 <<  " id = " << o->id << ", elem1 = " << o->elem.first.getIndex() << ", elem2 = " << o->elem.second.getIndex() << ", distance = " << o->value;
                                //testOutput << ", point0 = " << o->point[0] << ", point1 = " << o->point[1] << ", normal = " << o->normal << std::endl;
                                //testOutput.close();
                            }
#endif

                            m_constraintGPU_ll->addContact(mu_, normalVec, distance, index1, index2, index, o->id);
                            i++;
                        }
                    }
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                if (m_constraintGPU_lv)
                {

                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                if (m_constraintGPU_vf)
                {
                    i = 0;
                    for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++)
                    {
                        DetectionOutput* o = *it;

                        if (o->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                        {
                            index1 = mappedContacts_VertexFace[i].first.first;
                            index2 = mappedContacts_VertexFace[i].first.second;
                            distance = mappedContacts_VertexFace[i].second;

#ifdef DO_INTERSECTION_TEST_FOR_TRIANGLES
                            //==================================================================================
                            //std::cout << "Banane2! BANANE2!" << std::endl;

                            int triIdx1 = o->elem.first.getIndex() / 3;
                            int triIdx2 = o->elem.second.getIndex() / 3;

                            if (!(triIdx1 < this->model1->getMeshTopology()->getNbTriangles() &&
                                  triIdx2 < this->model2->getMeshTopology()->getNbTriangles()   ))
                            {
                                if (triIdx1 >= this->model1->getMeshTopology()->getNbTriangles())
                                {
                                    std::cout << "ERROR: triIdx1 >= this->model1->getMeshTopology()->getNbTriangles(); triIdx1 = " << triIdx1 << "this->model1->getMeshTopology()->getNbTriangles() = " << this->model1->getMeshTopology()->getNbTriangles() << std::endl;
                                }
                                if (triIdx2 >= this->model2->getMeshTopology()->getNbTriangles())
                                {
                                    std::cout << "ERROR: triIdx2 >= this->model2->getMeshTopology()->getNbTriangles(); triIdx2 = " << triIdx2 << "this->model2->getMeshTopology()->getNbTriangles() = " << this->model2->getMeshTopology()->getNbTriangles() << std::endl;
                                }
                            }
                            else
                            {
                                sofa::core::topology::Triangle contactTriangle1, contactTriangle2;

                                contactTriangle1 = model1->getContext()->getMeshTopology()->getTriangle(triIdx1);
                                contactTriangle2 = model2->getContext()->getMeshTopology()->getTriangle(triIdx2);

                                sofa::component::container::MechanicalObject<Vec3Types>* model1MechanicalState = dynamic_cast<sofa::component::container::MechanicalObject<Vec3Types>*>(model1->getContext()->getMechanicalState()) ;
                                sofa::component::container::MechanicalObject<Vec3Types>* model2MechanicalState = dynamic_cast<sofa::component::container::MechanicalObject<Vec3Types>*>(model2->getContext()->getMechanicalState()) ;

                                if ((!model1MechanicalState) || (!model2MechanicalState))
                                {
                                    if (!model1MechanicalState)
                                    {
                                        std::cout << "ERROR: dynamic_cast from 'model1->getContext()->getMechanicalState()' failed.";
                                    }
                                    if (!model2MechanicalState)
                                    {
                                        std::cout << "ERROR: dynamic_cast from 'model2->getContext()->getMechanicalState()' failed.";
                                    }
                                }
                                else
                                {
                                    const Vec3Types::VecCoord& model1PositionVector = model1MechanicalState->read(core::ConstVecCoordId::position())->getValue();
                                    const Vec3Types::VecCoord& model2PositionVector = model2MechanicalState->read(core::ConstVecCoordId::position())->getValue();
                                    BVHModels::Triangle3<SReal> firstTriangle(model1PositionVector.at(contactTriangle2.at(0)),
                                                                              model1PositionVector.at(contactTriangle2.at(1)),
                                                                              model1PositionVector.at(contactTriangle2.at(2)));
                                    BVHModels::Triangle3<SReal> secondTriangle(model2PositionVector.at(contactTriangle1.at(0)),
                                                                               model2PositionVector.at(contactTriangle1.at(1)),
                                                                               model2PositionVector.at(contactTriangle1.at(2)));
                                    /*BVHModels::Triangle3<SReal> firstTriangle(model1PositionVector.at(contactTriangle1.at(0)),
                                                                              model1PositionVector.at(contactTriangle1.at(1)),
                                                                              model1PositionVector.at(contactTriangle1.at(2)));
                                    BVHModels::Triangle3<SReal> secondTriangle(model2PositionVector.at(contactTriangle2.at(0)),
                                                                               model2PositionVector.at(contactTriangle2.at(1)),
                                                                               model2PositionVector.at(contactTriangle2.at(2)));*/
                                                                //always check against the triangle of the other model;

                                    //std::cout << "firstTriangle " << firstTriangle.V[0] << ", " << firstTriangle.V[1] << ", " << firstTriangle.V[2] <<  std::endl;
                                    //std::cout << "secondTriangle " << secondTriangle.V[0] << ", " << secondTriangle.V[1] << ", " << secondTriangle.V[2] <<  std::endl;

                                    BVHModels::Triangle3IntersectionResult<SReal> intersectionResult;

                                    if (firstTriangle.Find(secondTriangle,intersectionResult))
                                    {
                                        // TODO
                                        std::cout << "Triangle intersetion." << std::endl;
                                        setIntersectionOccurred(true);
                                        /*
                                        std::cout << "intersectionResult.mQuantity: " << intersectionResult.mQuantity << std::endl;
                                        for (int i=0; i<intersectionResult.mQuantity; i++)
                                        {
                                            std::cout << "intersectionResult.mPoint[" << i << "]: ";
                                            std::cout << intersectionResult.mPoint[i];
                                            std::cout << std::endl;
                                        }
                                        std::cout << "intersectionResult.GetIntersectionType(): " << intersectionResult.GetIntersectionType() << std::endl;
                                        */
                                    }
                                    else
                                    {
                                        //setIntersectionOccurred(false);
                                        //std::cout << "Triangles do not intersect." << std::endl;
                                    }

                                    /*for (int i=0; i<10; i++)
                                    {
                                        std::cout << "BananeEnde" << std::endl;
                                    }*/
                                }
                            }
                            //std::cout << "BANAAAAAAAAANE2!" << std::endl;
                            //==================================================================================
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                            std::cout << " * contact " << i << ": index1 = " << index1 << ", index2 = " << index2 <<  ", id = " << o->id << ", elem1 = " << o->elem.first.getIndex() << ", elem2 = " << o->elem.second.getIndex() << ", feature1 = " << o->elemFeatures.first << ", feature2 = " << o->elemFeatures.second << ", distance = " << o->value;
                            std::cout << ", point0 = " << o->point[0] << ", point1 = " << o->point[1] << ", normal = " << o->normal << std::endl;
                            std::cout << "   add FACE_VERTEX contact to m_constraintGPU_vf" << std::endl;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                            std::string contactType("VERTEX_FACE");
                            if (testOutputFilename.getValue() != "")
                            {
                        #ifndef _WIN32
                                //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
                        #else
                                //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
                        #endif
                                //testOutput << " * contact " << i << ", type = " << contactType << ": index1 = " << index1 << ", index2 = " << index2 <<  " id = " << o->id << ", elem1 = " << o->elem.first.getIndex() << ", elem2 = " << o->elem.second.getIndex() << ", distance = " << o->value;
                                //testOutput << ", point0 = " << o->point[0] << ", point1 = " << o->point[1] << ", normal = " << o->normal << std::endl;
                                //testOutput.close();
                            }
#endif

                            typename ResponseDataTypes::Deriv normalVec(o->normal.x(), o->normal.y(), o->normal.z());
                            // Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
                            long index = cantorPolynomia(o->id /*cantorPolynomia(index1, index2)*/,id);

                            m_constraintGPU_vf->addContact(mu_, normalVec, o->point[0], o->point[1], distance, index1, index2, o->point[0], o->point[1], index, o->id);
                            i++;
                        }
                    }
                }
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                if (m_defaultConstraint)
                {
                    i = 0;
                    for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++)
                    {
                        DetectionOutput* o = *it;
                        if (o->contactType != sofa::core::collision::CONTACT_FACE_VERTEX &&
                            o->contactType != sofa::core::collision::CONTACT_LINE_LINE &&
                            o->contactType != sofa::core::collision::CONTACT_LINE_VERTEX)
                        {
                            index1 = mappedContacts[i].first.first;
                            index2 = mappedContacts[i].first.second;
                            distance = mappedContacts[i].second;

                            typename ResponseDataTypes::Deriv normalVec(o->normal.x(), o->normal.y(), o->normal.z());
                            // Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
                            long index = cantorPolynomia(o->id /*cantorPolynomia(index1, index2)*/,id);

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                            std::cout << " * contact " << i << ": index1 = " << index1 << ", index2 = " << index2 <<  " id = " << o->id << ", elem1 = " << o->elem.first.getIndex() << ", elem2 = " << o->elem.second.getIndex() << ", distance = " << o->value;
                            std::cout << ", point0 = " << o->point[0] << ", point1 = " << o->point[1] << ", normal = " << o->normal << std::endl;
                            std::cout << "   add OTHER contact to m_constraint1" << std::endl;
#endif
                            
                            m_defaultConstraint->addContact(mu_, normalVec, distance, index1, index2, index, o->id);

                            i++;
                        }
                    }
                }
#endif
                if (parent!=NULL)
                {
                    parent->removeObject(this);

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                    parent->removeObject(m_constraintGPU_ll);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                    parent->removeObject(m_constraintGPU_vf);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                    parent->removeObject(m_defaultConstraint);
#endif
                }

                parent = group;

                if (parent!=NULL)
                {
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                    sout << "ObbTreeGPUFrictionContact " << this->getName() << ": Attaching contact response to " << parent->getName() << sendl;
                    std::cout << "ObbTreeGPUFrictionContact " << this->getName() << ": Attaching contact response to " << parent->getName() << std::endl;
#endif
                    parent->addObject(this);

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                    parent->addObject(m_constraintGPU_ll);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                    parent->addObject(m_constraintGPU_lv);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                    parent->addObject(m_constraintGPU_vf);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                    parent->addObject(m_defaultConstraint);
#endif
                }
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
            std::cout << "=== end   ObbTreeGPUFrictionContact::createResponse(" << this->getName() << ") call ===" << std::endl;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
            if (testOutputFilename.getValue() != "")
            {
        #ifndef _WIN32
                //testOutput.open(testOutputFilename.getValue().c_str(), std::ofstream::out | std::ofstream::app);
        #else
                //testOutput.open(testOutputFilename.getValue(), std::ofstream::out | std::ofstream::app);
        #endif
                //testOutput << "Leaving ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::createResponse()" << std::endl;
                //testOutput << "========================================================================================================" << std::endl;
                //testOutput.close();
            }
#endif

#else
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_CREATERESPONSE
                std::cout << "ObbTreeGPUFrictionContact " << this->getName() << ": Would create " << contacts.size() << " contacts." << std::endl;
                int i=0;
                for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it != contacts.end(); it++, i++)
                {
                    DetectionOutput* o = *it;
                    //int index1 = mappedContacts[i].first.first;
                    //int index2 = mappedContacts[i].first.second;
                    //double distance = mappedContacts[i].second;
                    std::cout << " * contact " << i << ": id = " << o->id << ", elem1 = " << o->elem.first.getIndex() << ", elem2 = " << o->elem.second.getIndex() << ", distance = " << o->value;
                    std::cout << ", point0 = " << o->point[0] << ", point1 = " << o->point[1] << ", normal = " << o->normal << std::endl;
                }
#endif
#endif

            }

//#define OBBTREEGPUFRICTIONCONTACT_DEBUG_REMOVERESPONSE
            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::removeResponse()
            {
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                if (m_constraintGPU_ll)
                {
                    mapper1_gpu_ll.resize(0);
                    mapper2_gpu_ll.resize(0);
                }
                m_numLineLineContacts = 0;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                if (m_constraintGPU_lv)
                {
                    mapper1_gpu_lv.resize(0);
                    mapper2_gpu_lv.resize(0);
                }
                m_numLineVertexContacts = 0;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                if (m_constraintGPU_vf)
                {
                    mapper1_gpu_vf.resize(0);
                    mapper2_gpu_vf.resize(0);
                }
                m_numVertexFaceContacts = 0;
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                if (m_defaultConstraint)
                {
                    mapper1_default.resize(0);
                    mapper2_default.resize(0);
                }
                m_numOtherContacts = 0;
#endif

                if (parent!=NULL)
                {
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_REMOVERESPONSE
                    sout << "ObbTreeGPUFrictionContact " << this->getName() << ": Removing contact response from " << parent->getName() << sendl;
                    std::cout << "ObbTreeGPUFrictionContact " << this->getName() << ": Removing contact response from " << parent->getName() << std::endl;
#endif
                    parent->removeObject(this);

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                    if (m_constraintGPU_ll)
                        parent->removeObject(m_constraintGPU_ll);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINEVERTEX_CONSTRAINT
                    if (m_constraintGPU_lv)
                        parent->removeObject(m_constraintGPU_lv);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                    if (m_constraintGPU_vf)
                        parent->removeObject(m_constraintGPU_vf);
#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                    if (m_defaultConstraint)
                        parent->removeObject(m_defaultConstraint);
#endif
                }
                parent = NULL;
            }

            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setInteractionTags(MechanicalState1* mstate1, MechanicalState2* mstate2,
                                                                                                                    typename sofa::component::constraintset::UnilateralInteractionConstraint<ResponseDataTypes>* constraint)
            {
				sofa::core::objectmodel::TagSet tagsm1 = mstate1->getTags();
				sofa::core::objectmodel::TagSet tagsm2 = mstate2->getTags();
				sofa::core::objectmodel::TagSet::iterator it;
                for(it=tagsm1.begin(); it != tagsm1.end(); it++)
                    constraint->addTag(*it);
                for(it=tagsm2.begin(); it!=tagsm2.end(); it++)
                    constraint->addTag(*it);
            }

#define OBBTREEGPUFRICTIONCONTACT_DEBUG_DRAW
#define OBBTREEGPUFRICTIONCONTACT_DRAW_CONTACT_POINTS
//#define OBBTREEGPUFRICTIONCONTACT_LABEL_CONTACT_POINTS
            template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
            void ObbTreeGPUFrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::draw(const core::visual::VisualParams* vparams)
            {
                typename core::behavior::MechanicalState<Vec3Types>::ReadVecCoord pos1 = this->model1->getMechanicalState()->readPositions();
                typename core::behavior::MechanicalState<Vec3Types>::ReadVecCoord pos2 = this->model2->getMechanicalState()->readPositions();

#ifdef OBBTREEGPUFRICTIONCONTACT_DRAW_CONTACT_POINTS
                if (contacts.size() > 0)
                {
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_DRAW
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_LINELINE_CONSTRAINT
                    this->m_constraintGPU_ll->draw(vparams);
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_VERTEXFACE_CONSTRAINT
                    this->m_constraintGPU_vf->draw(vparams);
#endif
#ifdef OBBTREEGPUFRICTIONCONTACT_USE_DEFAULT_CONSTRAINT
                    this->m_defaultConstraint->draw(vparams);
#endif

#endif

#ifdef OBBTREEGPUFRICTIONCONTACT_LABEL_CONTACT_POINTS
                    std::stringstream labelStream;
#endif
                    int contactCount = 0;
                    for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++)
                    {
                        DetectionOutput* o = *it;
                        contactCount++;
                        sofa::defaulttype::Vec3f color;

                        if (o->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                            color = Vec3f(0,0.8,0);
                        else if (o->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                            color = Vec3f(0.8,0,0);


                        int idx1 = o->elem.first.getIndex();
                        int idx2 = o->elem.second.getIndex();

                        glPushMatrix();

                        glPushAttrib(GL_ENABLE_BIT);
                        glEnable(GL_LIGHTING);
                        glEnable(GL_COLOR_MATERIAL);

                        glPointSize(12.0f);
                        glBegin(GL_POINTS);
                        glColor4d(color[0], color[1], color[2], 0.75f);
                        glVertex3d(o->point[0].x(), o->point[0].y(), o->point[0].z());
                        glColor4d(color[0], color[1], color[2], 0.75f);
                        glVertex3d(o->point[1].x(), o->point[1].y(), o->point[1].z());
                        glEnd();
                        glPointSize(1.0f);

                        glLineWidth(8.0f);
                        glBegin(GL_LINES);
                        glColor4d(1, 1, 1, 0.75f);
                        glVertex3d(o->point[0].x(), o->point[0].y(), o->point[0].z());
                        glColor4d(color[0], color[1], color[2], 0.75f);
						glVertex3d(o->point[0].x() + o->normal.x(), o->point[0].y() + o->normal.y(), o->point[0].z() + o->normal.z());
                        glEnd();
                        glLineWidth(1.0f);

#ifdef OBBTREEGPUFRICTIONCONTACT_LABEL_CONTACT_POINTS

                        Vec3d ptOffset(0.25 * contactCount, 0.25 * contactCount, 0.25 * contactCount);

                        glTranslated(o->point[1].x() + ptOffset.x(), o->point[1].y() + ptOffset.y(), o->point[1].z() + ptOffset.z());

                        Mat<4,4, GLfloat> modelviewM;

                        glColor3f(color[0], color[1], color[2]);
                        glDisable(GL_LIGHTING);

                        //const sofa::defaulttype::BoundingBox& bbox = this->getContext()->f_bbox.getValue();
                        float scale = 0.15f * 0.01f; //(float)((bbox.maxBBox() - bbox.minBBox()).norm() * 0.0001);

                        //std::cout << "  minBBox = " << bbox.minBBox() << ", maxBBox = " << bbox.maxBBox() << ", scale = " << scale << std::endl;

                        labelStream << this->getName();
                        if (o->contactType == sofa::core::collision::CONTACT_FACE_VERTEX)
                        {
                            labelStream << ": FACE_VERTEX " << idx1 << " -- " << idx2;
                        }
                        else if (o->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                        {
                            int triIdx1 = idx1 / 3;
                            int triIdx2 = idx2 / 3;

                            int edgeIdx1 = idx1 % 3;
                            int edgeIdx2 = idx2 % 3;

                            labelStream << ": LINE_LINE " << triIdx1 << "/" << edgeIdx1 << " (" << idx1 << ") -- " << triIdx2 << "/" << edgeIdx2 << " (" << idx2 << ")";
                        }

                        labelStream << " points: " << o->point[0] << " - " << o->point[1];
                        labelStream << ", normal: " << o->normal << ", value: " << o->value;
                        labelStream << ", deltaT: " << o->deltaT << ", elements: " << o->elem.first.getIndex() << " - " << o->elem.second.getIndex();

                        std::string tmp = labelStream.str();
                        const char* s = tmp.c_str();
                        glPushMatrix();

                        //glTranslatef(center[0], center[1], center[2]);
                        glScalef(scale,scale,scale);

                        // Makes text always face the viewer by removing the scene rotation
                        // get the current modelview matrix
                        glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
                        modelviewM.transpose();

                        sofa::defaulttype::Vec3f temp = modelviewM.transform(o->point[1] + ptOffset);

                        //glLoadMatrixf(modelview);
                        glLoadIdentity();

                        glTranslatef(temp[0], temp[1], temp[2]);
                        glScalef(scale,scale,scale);

                        labelStream.str("");

                        while(*s)
                        {
                            glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                            s++;
                        }

                        glPopMatrix();
#endif

                        glPopAttrib();
                        glPopMatrix();

                        glPushMatrix();
                        glPushAttrib(GL_ENABLE_BIT);
                        glEnable(GL_LIGHTING);
                        glEnable(GL_COLOR_MATERIAL);

                        int triIdx1 = idx1 / 3;
                        int triIdx2 = idx2 / 3;

                        int edgeIdx1 = idx1 % 3;
                        int edgeIdx2 = idx2 % 3;

                        if (o->contactType == sofa::core::collision::CONTACT_LINE_LINE)
                        {
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_DRAW
                            // std::cout << " * LINE_LINE contact " << contactCount << ": idx1 = " << idx1 << ", idx2 = " << idx2 << "; triIdx1 = " << triIdx1 << ", triIdx2 = " << triIdx2 << "; edgeIdx1 = " << edgeIdx1 << ", edgeIdx2 = " << edgeIdx2 << std::endl;
#endif
                            if (triIdx1 >= this->model1->getMeshTopology()->getNbTriangles() ||
                                triIdx2 >= this->model2->getMeshTopology()->getNbTriangles())
                                continue;

                            const sofa::core::topology::BaseMeshTopology::EdgesInTriangle& e1 = this->model1->getMeshTopology()->getEdgesInTriangle(triIdx1);
                            const sofa::core::topology::BaseMeshTopology::EdgesInTriangle& e2 = this->model2->getMeshTopology()->getEdgesInTriangle(triIdx2);
#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_DRAW
                            // std::cout << "   edges e1 = " << e1 << "; edges e2 = " << e2 << std::endl;
#endif
                            if (e1[edgeIdx1] >= this->model1->getMeshTopology()->getNbEdges() ||
                                e2[edgeIdx2] >= this->model2->getMeshTopology()->getNbEdges())
                                continue;

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_DRAW
                            // std::cout << "   GETEDGE model1: " << e1[edgeIdx1] << ", numEdges = " << this->model1->getMeshTopology()->getNbEdges() << std::endl;
#endif
                            const sofa::core::topology::BaseMeshTopology::Edge& edge1 = this->model1->getMeshTopology()->getEdge(e1[edgeIdx1]);

#ifdef OBBTREEGPUFRICTIONCONTACT_DEBUG_DRAW
                            // std::cout << "   GETEDGE model2: " << e1[edgeIdx2] << ", numEdges = " << this->model2->getMeshTopology()->getNbEdges() << std::endl;
#endif
                            const sofa::core::topology::BaseMeshTopology::Edge& edge2 = this->model2->getMeshTopology()->getEdge(e2[edgeIdx2]);

                            int ptIdx1 = edge1[0];
                            int ptIdx2 = edge1[1];

                            if (ptIdx1 < pos1.size() &&
                                ptIdx2 < pos1.size())
                            {
                                Vector3 pt1 = pos1[ptIdx1];
                                Vector3 pt2 = pos1[ptIdx2];

                                ptIdx1 = edge2[0];
                                ptIdx2 = edge2[1];
                                if (ptIdx1 < pos2.size() &&
                                    ptIdx2 < pos2.size())
                                {
                                    Vector3 pt3 = pos2[ptIdx1];
                                    Vector3 pt4 = pos2[ptIdx2];

                                    glLineWidth(32.0f);
                                    glBegin(GL_LINES);

                                    glColor4f(color[0], color[1], color[2], 0.75);
                                    glVertex3d(pt1.x(), pt1.y(), pt1.z());
                                    glColor4f(0.9, 0.9, 0, 0.75);
                                    glVertex3d(pt2.x(), pt2.y(), pt2.z());

                                    glColor4f(color[0], color[1], color[2], 0.75);
                                    glVertex3d(pt3.x(), pt3.y(), pt3.z());
                                    glColor4f(0, 0.9, 0.9, 0.75);
                                    glVertex3d(pt4.x(), pt4.y(), pt4.z());

                                    glEnd();
                                    glLineWidth(1.0f);
                                }
                            }
                        }
                        glPopAttrib();
                        glPopMatrix();

                    }
                }
#endif
            }
        }
    }
}
