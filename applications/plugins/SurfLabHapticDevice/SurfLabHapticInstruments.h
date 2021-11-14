/******************************************************************************

******************************************************************************/
#pragma once 

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/VecTypes.h>//Vec3Types.h not found
#include <SofaUserInteraction/Controller.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>

#include <SofaGeneralDeformable/VectorSpringForceField.h>

#include <SofaSimulationGraph/DAGNode.h>
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <SofaBaseCollision/BaseContactMapper.h>
#include <SofaConstraint/StickContactConstraint.h>

#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/gl/template.h>
#include <SofaUserInteraction/TopologicalChangeManager.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>
#include <sofa/helper/AdvancedTimer.h>
#include <SofaLoader/MeshObjLoader.h>
#include <SofaOpenglVisual/OglShader.h>
#include <SofaOpenglVisual/OglTexture.h>

#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>

#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaDeformable/SpringForceField.h>

#include <sofa/gl/Capture.h>
//add the 2 libs below to support v12.16
#include <boost/scoped_ptr.hpp>
#include <sofa/core/topology/Topology.h>

//Changes for force feedback safety
#include <SofaHaptics/ForceFeedback.h>

#include <SofaPython/ScriptController.h>

namespace SurfLab
{
	//using namespace std;
	using namespace sofa;
	using namespace sofa::component::controller;
	using namespace sofa::defaulttype;
	using namespace sofa::simulation;

	class SurfLabHapticDevice;

	enum HapticInstrument_Images
	{
		HM_IMG_SUTURE,
		HM_IMG_CARVE,
		HM_IMG_DISSECT,
		HM_IMG_CLAMP,
		HM_IMG_ENDOSTAPLE,
		HM_IMG_GRASP,
		HM_IMG_NEEDLE,
		HM_IMG_RETRACT,
		HM_IMG_CONTAIN,
		HM_IMG_CAMERA,
		HM_IMG_SELECTION,
		HM_IMG_BACKGROUND,
		HM_IMG_BACKGROUND_HOVER,
		HM_IMG_BACKGROUND_SELECTED,
		HM_IMG_QUESTION_SELECTED,
		HM_IMG_MAX
	};

	/**
	* Omni driver
	*/
	class SurfLabHapticInstruments : public Controller, sofa::core::visual::VisualModel
	{

	public:
		SOFA_CLASS2(SurfLabHapticInstruments, sofa::component::controller::Controller, sofa::core::visual::VisualModel);

		typedef defaulttype::Vec3Types DataTypes;
		typedef defaulttype::Vec3f Vec3f;
		typedef defaulttype::Vec4f Vec4f;
		typedef defaulttype::RigidTypes RigidTypes;
		typedef defaulttype::Vector3 Vector3;
		typedef DataTypes::Coord Coord;
		typedef DataTypes::VecCoord VecCoord;
		typedef DataTypes::Real Real;
		typedef DataTypes::Deriv Deriv;

		SurfLabHapticInstruments();
		SurfLabHapticInstruments(SurfLabHapticDevice* InDevice);
		virtual ~SurfLabHapticInstruments();

		virtual void init();
		virtual void drawTransparent(const core::visual::VisualParams* vparams) override;

		bool IsReadyToSwap();
		void DoHoverSwap();
		///
		void DisableActiveTool();
		void EnableActiveTool();
		void SwitchToLastActiveTool(); /// switch the tool to the last active one
        void SetLastActiveTool();/// set LastActiveTool as the current one
		bool isActiveToolDisabled;
		Real timeActiveToolDisabled;
		sofa::simulation::Node::SPtr mActiveTool;
	private:

		sofa::component::controller::ScriptController* m_ScriptController;
		std::unique_ptr<sofa::gl::Texture> glToolTextures[HM_IMG_MAX];
		Real HoverTime;
		std::uint8_t SwapIDX;
		SurfLabHapticDevice* HapticDevice;
		int mLastActiveToolIDX;
		sofa::type::vector<sofa::simulation::Node::SPtr> Instruments;
		sofa::type::vector<HapticInstrument_Images> Instrument_TextureTypes;
	};

} // namespace surflab

