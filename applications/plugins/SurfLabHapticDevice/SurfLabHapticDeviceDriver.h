/******************************************************************************

******************************************************************************/
#pragma once 

#define SURFLAB_DRIVER_NAME_S "SurfLabHapticDevice"


//Sensable include
#include <HD/hd.h>
#include <HDU/hdu.h>
#include <HDU/hduError.h>
#include <HDU/hduVector.h>
#include <sofa/helper/LCPcalc.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/Quater.h>


#include <sofa/core/behavior/BaseController.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaRigid/RigidMapping.h>
#include <SofaUserInteraction/Controller.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <SofaBaseMechanics/MechanicalObject.h>


//force feedback
#include <SofaHaptics/ForceFeedback.h>
#include <SofaHaptics/MechanicalStateForceFeedback.h>
#include <SofaHaptics/LCPForceFeedback.h>
#include <SofaHaptics/NullForceFeedbackT.h>

#include <sofa/simulation/Node.h>
#include <cstring>

#include <SofaOpenglVisual/OglModel.h>
#include <SofaSimulationTree/GNode.h>
#include <SofaBaseTopology/TopologyData.h>
#include <SofaBaseVisual/InteractiveCamera.h>

#include <math.h>

#include "SurfLabHapticInstruments.h"
//TODO add UDP server
//#include <winsock2.h>
namespace SurfLab
{
	using std::vector;
	using namespace sofa;
	using namespace sofa::component::controller;
	using namespace sofa::defaulttype;
	using namespace sofa::simulation;

	/** Holds data retrieved from HDAPI. */
	struct HapticDeviceData
	{
		HHD id;
		int nupdates;
		int m_buttonState;					/* Has the device button has been pressed. */
		hduVector3Dd m_devicePosition;	/* Current device coordinates. */
		HDErrorInfo m_error;
		Vec3d pos;
		Vec3d fulcrumOffset;
		Quat quat;
		bool ready;
		bool stop;

		float DistBaseToStylus;

		HapticDeviceData()
		{
			id = 0;
			ready = false;
			stop = false;
			nupdates = 0;
			DistBaseToStylus = 0.0f;
			m_buttonState = 0;
			fulcrumOffset = Vec3d(0, 0, 0);
		}
	};

	struct NewOmniData
	{
		ForceFeedback::SPtr forceFeedback;
		simulation::Node::SPtr* context;

		sofa::defaulttype::SolidTypes<double>::Transform endOmni_H_virtualTool;
		//Transform baseOmni_H_endOmni;
		sofa::defaulttype::SolidTypes<double>::Transform world_H_baseOmni;
		double forceScale;
		double scale;
		bool permanent_feedback;
		Vec3d desirePosition;
		bool move2Pos; //if true, move the tool to the specified position when scene starts
		double stiffness;

		// API OMNI //
		HapticDeviceData servoDeviceData;   // for the haptic loop
		HapticDeviceData deviceData;		// for the simulation loop
		int LastButtonState;				// Has the device button been pressed in last simulation loop, we need it 
											// since the buttonStates in deviceData get overwrite each cycle from servoDeviceData
		double currentForce[3];

		/*NewOmniData()
		{
			forceScale = 0;
			scale = 0;
			permanent_feedback = false;
			 move2Pos = false;
			memset(currentForce, 0, sizeof(double) * 3);
		}*/
	};

	struct AllNewOmniData
	{
		std::vector<NewOmniData> omniData;
	};

	/**
	* Omni driver
	*/
	class SurfLabHapticDevice : public Controller
	{

	public:
		SOFA_CLASS(SurfLabHapticDevice, Controller);
		static std::string getName() {
			return SURFLAB_DRIVER_NAME_S;
		}

		typedef RigidTypes::Coord Coord;
		typedef RigidTypes::VecCoord VecCoord;
		typedef sofa::defaulttype::Vec3Types DataTypes;
		typedef component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> MMechanicalObject;

		struct VisualComponent
		{
			simulation::Node::SPtr node;
			sofa::component::visualmodel::OglModel::SPtr visu;
			sofa::component::mapping::RigidMapping< Rigid3dTypes, Vec3Types  >::SPtr mapping;
		};

		Data<double> forceScale;
		Data<double> scale;
		Data<double> distanceButtonTwoToggle;
		Data<double> timeDistanceToggle;

		Data<Vec3d> positionBase;
		Data<Quat> orientationBase;
		Data<Vec3d> positionTool;
		Data<Quat> orientationTool;
		Data<bool> permanent;
		Data<bool> omniVisu;
		Data< VecCoord > posDevice;
		Data< VecCoord > posStylus;
		Data< std::string > locDOF;
		Data< std::string > deviceName;
		Data< int > deviceIndex;
		Data<Vec1d> openTool;
		Data<double> maxTool;
		Data<double> minTool;
		Data<double> openSpeedTool;
		Data<double> closeSpeedTool;
		Data<bool> setRestShape;
		Data<bool> applyMappings;
		Data<bool> alignOmniWithCamera;
		Data<bool> stateButton1;
		Data<bool> stateButton2;
		Data<bool> setDesirePosition;
		Data<bool> isDominantHand;
		Data<Vec3d> desirePosition;

		sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes>::SPtr DOF;
		sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes>::SPtr toolDOF;
		sofa::component::visualmodel::BaseCamera::SPtr camera;

		bool initVisu;

		//UF - DS
		float CurrentTimeToToggle;

		NewOmniData data;
		AllNewOmniData allData;
		bool key_Q_down, key_W_down;//TIPS bluetooth map
		int key_Q_count, key_W_count;//TIPS bluetooth filter

		SurfLabHapticDevice();
		virtual ~SurfLabHapticDevice();

		virtual void init();
		virtual void bwdInit();
		virtual void reset();
		void reinit();

		int initDevice();

		void cleanup();
		virtual void draw(const core::visual::VisualParams*) override;
		virtual void draw();

		void setForceFeedback(ForceFeedback* ff);

		void onKeyPressedEvent(core::objectmodel::KeypressedEvent*);
		void onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent*);
		void onAnimateBeginEvent();

		void setDataValue();

		/////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////TIPS UDP phone controller//////////////////////////////
		Data<bool> enableUDPServer;
		Data<std::string> inServerIPAddr;
		Data<int> inServerPortNum;
		Data<double> motionScaleFactor;
		static bool streamActive;
		static std::string serverIPAddr;
		static int serverPortNum;
		void addUDPServerThread();
		static void runServerLoop();
		static float recvOriDataDev1[4];
		static float recvOriDataDev2[4];
		static int recvButtonStateDev1;
		static int recvButtonStateDev2;
		static std::string messageToReplyDev1; //message to reply to cell phone client, should be a queue actually
		static std::string messageToReplyDev2; //message to reply to cell phone client, should be a queue actually
		std::string messageFromHapticManagerDev1;
		std::string messageFromHapticManagerDev2;
		//void setMessageToReply(std::string s);
		static float recvMotionStateYDev1;
		static float recvMotionStateXDev1;
		static float recvMotionStateYDev2;
		static float recvMotionStateXDev2;
		static Quat startCorrectionQuat; // for calibration
		static Quat startViewQuat; // for calibration
		Vec3d startTool1Pos; // for calibration -> reset position
		Vec3d startTool2Pos; // for calibration -> reset position
		static bool needToRecalibrateTool1;
		static bool needToRecalibrateTool2;
		//variable pour affichage graphique
		simulation::Node* parent;
		enum
		{
			VN_stylus = 0,
			VN_joint2 = 1,
			VN_joint1 = 2,
			VN_arm2 = 3,
			VN_arm1 = 4,
			VN_joint0 = 5,
			VN_base = 6,
			VN_X = 7,
			VN_Y = 8,
			VN_Z = 9,
			NVISUALNODE = 10
		};
		VisualComponent visualNode[NVISUALNODE];
		static const char* visualNodeNames[NVISUALNODE];
		static const char* visualNodeFiles[NVISUALNODE];
		simulation::Node::SPtr nodePrincipal;
		//UF - DS
		simulation::Node::SPtr InstrumentNode;
		SurfLabHapticInstruments::SPtr m_Instruments;

		MMechanicalObject::SPtr rigidDOF;
		bool changeScale;
		bool firstInit;
		float oldScale;
		bool visuActif;
		bool isInitialized;
		Vec3d positionBase_buf;
		bool modX;
		bool modY;
		bool modZ;
		bool modS;
		bool axesActif;
		HDfloat angle1[3];
		HDfloat angle2[3];
		bool firstDevice;
		//vector<SurfLabHapticDevice*> autreOmniDriver;
		std::atomic<int> ServoAheadCount;
	private:
		void handleEvent(core::objectmodel::Event*);
		bool noDevice;

	};

} // namespace surflab

