#include "ObbTreeGPUFrictionContact.inl"

#include "ObbTreeGPUCollisionModel.h"

#include <SofaBaseCollision/BaseContactMapper.h>

namespace sofa
{
    namespace component
    {
        namespace collision
        {
            using namespace defaulttype;
            using namespace sofa::helper;

            sofa::core::collision::DetectionOutput::ContactId ObbTreeGPUContactIdentifier::cpt=0;
            std::list<sofa::core::collision::DetectionOutput::ContactId> ObbTreeGPUContactIdentifier::availableId;


            SOFA_DECL_CLASS(ObbTreeGPUFrictionContact)

            //Creator<Contact::Factory, ObbTreeGPUFrictionContact<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> > > ObbTreeGPUFrictionContactVec3Class("default",true);
            //Creator<Contact::Factory, ObbTreeGPUFrictionContact<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> > > ObbTreeGPUFrictionContactRigid3Class("FrictionContact",true);
            //Creator<Contact::Factory, ObbTreeGPUFrictionContact<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> > > ObbTreeGPUFrictionContactRigid3Class2("ObbTreeGPUFrictionContact",true);
            //ContactMapperCreator< ContactMapper<ObbTreeGPUCollisionModel<Rigid3Types> > > ObbTreeGPUContactMapperClass("default",true);
            
            Creator<Contact::Factory, ObbTreeGPUFrictionContact<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> > > ObbTreeGPUFrictionContactVec3Class("FrictionContact", true);
            Creator<Contact::Factory, ObbTreeGPUFrictionContact<ObbTreeGPUCollisionModel<Vec3Types>, ObbTreeGPUCollisionModel<Vec3Types> > > ObbTreeGPUFrictionContactVec3Class2("ObbTreeGPUFrictionContact", true);
            
            // Contact classes for SOFA CPU-contact models
            Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<PointModel, PointModel> > PointPointObbTreeGPUFrictionContactClass("ObbTreeGPUFrictionContact", true);
            Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<LineModel, SphereModel> > LineSphereObbTreeGPUFrictionContactClass("ObbTreeGPUFrictionContact", true);
            Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<LineModel, PointModel> > LinePointObbTreeGPUFrictionContactClass("ObbTreeGPUFrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<PointModel, LineModel> > LinePointObbTreeGPUFrictionContactClass4("ObbTreeGPUFrictionContact", true);
            Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<LineModel, LineModel> > LineLineObbTreeGPUFrictionContactClass("ObbTreeGPUFrictionContact", true);
            Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<TriangleModel, SphereModel> > TriangleSphereObbTreeGPUFrictionContactClass("ObbTreeGPUFrictionContact", true);
            Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<TriangleModel, PointModel> > TrianglePointObbTreeGPUFrictionContactClass("ObbTreeGPUFrictionContact", true);
            Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<TriangleModel, LineModel> > TriangleLineObbTreeGPUFrictionContactClass("ObbTreeGPUFrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<PointModel, TriangleModel> > TrianglePointObbTreeGPUFrictionContactClass4("ObbTreeGPUFrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<LineModel, TriangleModel> > TriangleLineObbTreeGPUFrictionContactClass4("ObbTreeGPUFrictionContact", true);
            Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<TriangleModel, TriangleModel> > TriangleTriangleObbTreeGPUFrictionContactClass("ObbTreeGPUFrictionContact", true);

			// 2nd time CPU models, with derived FrictionContact, but super-class id/alias
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<PointModel, PointModel> > PointPointObbTreeGPUFrictionContactClass2("FrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<LineModel, PointModel> > LinePointObbTreeGPUFrictionContactClass2("FrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<PointModel, LineModel> > LinePointObbTreeGPUFrictionContactClass3("FrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<LineModel, LineModel> > LineLineObbTreeGPUFrictionContactClass2("FrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<TriangleModel, PointModel> > TrianglePointObbTreeGPUFrictionContactClass2("FrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<TriangleModel, LineModel> > TriangleLineObbTreeGPUFrictionContactClass2("FrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<PointModel, TriangleModel> > TrianglePointObbTreeGPUFrictionContactClass3("FrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<LineModel, TriangleModel> > TriangleLineObbTreeGPUFrictionContactClass3("FrictionContact", true);
			Creator<sofa::core::collision::Contact::Factory, ObbTreeGPUFrictionContact<TriangleModel, TriangleModel> > TriangleTriangleObbTreeGPUFrictionContactClass2("FrictionContact", true);
            
        }
    }
}
