/******************************************************************************
******************************************************************************/

#include "SurfLabHapticDeviceDriver.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>

#include <sofa/helper/system/thread/CTime.h>

#include <sofa/core/visual/DrawToolGL.h>
#include <sofa/core/objectmodel/GUIEvent.h>

#include <sofa/helper/system/FileRepository.h>

using namespace std;

namespace SurfLab
{

	using namespace sofa::defaulttype;

	SOFA_DECL_CLASS(SurfLabHapticInstruments)

		int SurfLabHapticInstrumentsClass = core::RegisterObject("Haptic Instruments Drawing and Selection.").add< SurfLabHapticInstruments >();
	const char* TF_TO_TEXTURE_FILENAME_Conversion[HM_IMG_MAX] = { "tool_suture","tool_scissor", "tool_cut", "tool_clamp", "tool_endostapler","tool_grasp", "tool_needle",
		"tool_retractor","tool_contain", "tool_camera", "tool_selection", "tool_background",	"tool_background_hover", "tool_background_selected", "tool_unknown" };

	static std::map<std::string, HapticInstrument_Images > INSTRUMENT_TOOL_TO_IMAGE_MAP =
	{
		{ "tRight_angled_Hook_Suction_Electrode_Coagulators_", HM_IMG_DISSECT },
		{ "tBlunt_grasper", HM_IMG_GRASP },
		{"CameraTool", HM_IMG_CAMERA },
		{"tCurved Maryland Dissector close", HM_IMG_SUTURE},
		{ "tCurved Maryland Dissector open", HM_IMG_SUTURE },
		{"tClipApplier", HM_IMG_CLAMP },
		{"tCurved Scissor open", HM_IMG_CARVE},
		{"tDuet TRS close", HM_IMG_ENDOSTAPLE},
		{"tRetractor_grasp", HM_IMG_RETRACT },
		{"tTIPS_Pouch_for_Gallbladder", HM_IMG_CONTAIN},
		{ "tTIPS_Pouch", HM_IMG_CONTAIN },
		{ "tTIPS_Pouch_small", HM_IMG_CONTAIN },
		{"tBlunt_grasper_wNeedle", HM_IMG_NEEDLE}

	};

	SurfLabHapticInstruments::SurfLabHapticInstruments() : SurfLabHapticInstruments(nullptr) {}
	SurfLabHapticInstruments::SurfLabHapticInstruments(SurfLabHapticDevice* InDevice)
	{
		HapticDevice = InDevice;
		HoverTime = 0;
		isActiveToolDisabled = false;
		timeActiveToolDisabled = 0.0;
	}

	SurfLabHapticInstruments::~SurfLabHapticInstruments() { }

	bool ContainsNode(sofa::core::objectmodel::BaseNode* CurrentNode, sofa::core::objectmodel::BaseNode* SearchNode)
	{
		if (CurrentNode == SearchNode)
		{
			return true;
		}

		sofa::helper::vector< sofa::core::objectmodel::BaseNode* > children = CurrentNode->getChildren();
		for (size_t i = 0; i < children.size(); i++)
		{
			if (ContainsNode(children[i], SearchNode))
			{
				return true;
			}
		}

		return false;
	}

	void FindTaggedNodes(sofa::core::objectmodel::BaseNode* CurrentNode, sofa::helper::vector< sofa::core::objectmodel::BaseNode* >& OutTaggedNodes, const sofa::core::objectmodel::TagSet& tags)
	{
		sofa::helper::vector< sofa::core::objectmodel::BaseNode* > children = CurrentNode->getChildren();
		for (size_t i = 0; i < children.size(); i++)
		{
			if (children[i]->getTags().includes(tags))
			{
				OutTaggedNodes.push_back(children[i]);
			}
			FindTaggedNodes(children[i], OutTaggedNodes, tags);
		}
	}

	void ParseNodesAndFindInstruments(sofa::helper::vector< sofa::core::objectmodel::BaseNode* >& oChildren, sofa::core::objectmodel::BaseNode* n, const sofa::core::objectmodel::TagSet& tags)
	{
		sofa::helper::vector< sofa::core::objectmodel::BaseNode* > children = n->getChildren();
		for (size_t i = 0; i < children.size(); i++)
		{
			if (children[i]->getTags().includes(tags))
			{
				oChildren.push_back(children[i]);
			}
			ParseNodesAndFindInstruments(oChildren, children[i], tags);
		}
	}

	void SurfLabHapticInstruments::init()
	{
		// grab script controller
		core::objectmodel::BaseContext* c = this->getContext();
		c->get(m_ScriptController);

		// load tool textures
		for (int Iter = 0; Iter < HM_IMG_MAX; Iter++)
		{
			std::string TexturePath = std::string(TF_TO_TEXTURE_FILENAME_Conversion[Iter]) + ".png";

			sofa::core::objectmodel::DataFileName TexturePath_DFN(TexturePath, "", true, false);

			std::string filename = TexturePath_DFN.getFullPath();
			std::cout << "fname: " << filename << std::endl;
			if (sofa::helper::system::DataRepository.findFile(filename))
			{
				// Ordinary texture.
				helper::io::Image* NewImage = helper::io::Image::Create(filename.c_str());
				std::unique_ptr<helper::gl::Texture> NewGLTexture(new helper::gl::Texture(NewImage, false, true, false));
				glToolTextures[Iter] = std::move(NewGLTexture);
				glToolTextures[Iter]->init();
			}
		}

		//Parase our various tools...
		// a visitor executed from top but only run for this' parents will enforce the selected object unicity due even with diamond graph setups	

		sofa::simulation::Node* groot = dynamic_cast<simulation::Node*>(this->getContext()->getRootContext());
		sofa::simulation::Node* HapticDeviceAsNode = dynamic_cast<simulation::Node*>(HapticDevice->getContext());

		sofa::core::objectmodel::TagSet HapticTag;
		HapticTag.insert(sofa::core::objectmodel::Tag(std::string("haptic")));

		sofa::core::objectmodel::TagSet InstrumentTag;
		InstrumentTag.insert(sofa::core::objectmodel::Tag(std::string("instrument")));

		sofa::helper::vector< sofa::core::objectmodel::BaseNode* > TopInstrumentLayers;
		FindTaggedNodes(groot, TopInstrumentLayers, HapticTag);

		printf("---------------------------------------------\n");
		printf("SurfLabHapticInstruments for %s parsing nodes \n", HapticDevice->getName().c_str());

		for (int Iter = 0; Iter < TopInstrumentLayers.size(); Iter++)
		{
			printf("- FOUND HAPTIC TOOL %s\n", TopInstrumentLayers[Iter]->getName().c_str());

			if (ContainsNode(TopInstrumentLayers[Iter], HapticDeviceAsNode))
			{
				printf("- Contains Desired Device\n");

				sofa::helper::vector< sofa::core::objectmodel::BaseNode* > ChildInstruments;
				FindTaggedNodes(TopInstrumentLayers[Iter], ChildInstruments, InstrumentTag);

				for (int Iter = 0; Iter < ChildInstruments.size(); Iter++)
				{
					printf("-- FOUND INSTRUMENT %s\n", ChildInstruments[Iter]->getName().c_str());
					Instruments.push_back(dynamic_cast<simulation::Node*>(ChildInstruments[Iter]));

					std::map<std::string, HapticInstrument_Images>::iterator FindInstrument = INSTRUMENT_TOOL_TO_IMAGE_MAP.find(ChildInstruments[Iter]->getName());

					if (FindInstrument != INSTRUMENT_TOOL_TO_IMAGE_MAP.end())
					{
						Instrument_TextureTypes.push_back(FindInstrument->second);
					}
					else
					{
						Instrument_TextureTypes.push_back(HM_IMG_QUESTION_SELECTED);
					}
				}
			}
		}

		printf("---------------------------------------------\n");
	}

	// TODO: move to helper
	// https://www.opengl.org/wiki/GluProject_and_gluUnProject_code
	template<class Real>
	int glhProjectf(Real objx, Real objy, Real objz, Real* modelview, Real* projection, int* viewport, Real* windowCoordinate)
	{
		// Transformation vectors
		float fTempo[8];
		// Modelview transform
		fTempo[0] = modelview[0] * objx + modelview[4] * objy + modelview[8] * objz + modelview[12]; // w is always 1
		fTempo[1] = modelview[1] * objx + modelview[5] * objy + modelview[9] * objz + modelview[13];
		fTempo[2] = modelview[2] * objx + modelview[6] * objy + modelview[10] * objz + modelview[14];
		fTempo[3] = modelview[3] * objx + modelview[7] * objy + modelview[11] * objz + modelview[15];
		// Projection transform, the final row of projection matrix is always [0 0 -1 0]
		// so we optimize for that.
		fTempo[4] = projection[0] * fTempo[0] + projection[4] * fTempo[1] + projection[8] * fTempo[2] + projection[12] * fTempo[3];
		fTempo[5] = projection[1] * fTempo[0] + projection[5] * fTempo[1] + projection[9] * fTempo[2] + projection[13] * fTempo[3];
		fTempo[6] = projection[2] * fTempo[0] + projection[6] * fTempo[1] + projection[10] * fTempo[2] + projection[14] * fTempo[3];
		fTempo[7] = -fTempo[2];
		// The result normalizes between -1 and 1
		if (fTempo[7] == 0.0) // The w value
			return 0;
		fTempo[7] = 1.0 / fTempo[7];
		// Perspective division
		fTempo[4] *= fTempo[7];
		fTempo[5] *= fTempo[7];
		fTempo[6] *= fTempo[7];
		// Window coordinates
		// Map x, y to range 0-1
		windowCoordinate[0] = (fTempo[4] * 0.5 + 0.5) * viewport[2] + viewport[0];
		windowCoordinate[1] = (fTempo[5] * 0.5 + 0.5) * viewport[3] + viewport[1];
		// This is only correct when glDepthRange(0.0, 1.0)
		windowCoordinate[2] = (1.0 + fTempo[6]) * 0.5;	// Between 0 and 1
		return 1;
	}

	void SurfLabHapticInstruments::drawTransparent(const core::visual::VisualParams* vparams)
	{
		Vec4f textcolor(255.0f, 255.0f, 255.0f, 1.0f);
		if (vparams && vparams->drawTool())
		{
			sofa::simulation::Node::SPtr ActiveTool;
			for (int Iter = 0; Iter < Instruments.size(); Iter++)
			{
				if (Instruments[Iter]->isActive())
				{
					ActiveTool = Instruments[Iter];
					mActiveTool = Instruments[Iter];
					break;
				}
			}

			HapticInstrument_Images ActiveToolImage = HM_IMG_QUESTION_SELECTED;

			if (ActiveTool != nullptr)
			{
				std::map<std::string, HapticInstrument_Images>::iterator FindInstrument = INSTRUMENT_TOOL_TO_IMAGE_MAP.find(ActiveTool->getName());
				if (FindInstrument != INSTRUMENT_TOOL_TO_IMAGE_MAP.end())
				{
					ActiveToolImage = FindInstrument->second;
				}
			}

			vparams->drawTool()->saveLastState();
			vparams->drawTool()->disableStencilTest();

			int viewport[4];
			viewport[0] = vparams->viewport()[0];
			viewport[1] = vparams->viewport()[1];
			viewport[2] = vparams->viewport()[2];
			viewport[3] = vparams->viewport()[3];

			Vector3 CenterPosV(viewport[2] / 2, viewport[3] / 2, 0);
			double CircleRadius = min(CenterPosV[0], CenterPosV[1]);

			int IconSize = 80;
			int HalfIconSize = IconSize / 2;

			double DegreePerIter = ceil(asin(IconSize / CircleRadius) * 57.2958);
			double CurrentAngle = (Instruments.size() * DegreePerIter / 2) / 2;

			bool bValidOffScreenIcon = false;
			Vector3 OffScreenDrawPos;
			double OffScreenAngle;

			// draw off screen icon
			if (ActiveTool != nullptr)
			{
				double screenpos[3];
				double modelview[16];
				double projection[16];

				vparams->getModelViewMatrix(modelview);
				vparams->getProjectionMatrix(projection);

				const sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes>::VecCoord& WorldPos = HapticDevice->toolDOF->read(core::ConstVecCoordId::position())->getValue();
				const sofa::defaulttype::Vec<3, double>& TCenter = WorldPos[0].getCenter();
				glhProjectf<double>(TCenter[0], TCenter[1], TCenter[2], modelview, projection, viewport, screenpos);

				Vector3 ScreenPosV(screenpos[0], screenpos[1], 0);

				double DistanceToPoint = (ScreenPosV - CenterPosV).norm();
				if (DistanceToPoint > CircleRadius)
				{
					OffScreenDrawPos = (ScreenPosV - CenterPosV).normalized() * CircleRadius - Vector3(HalfIconSize, HalfIconSize, 0);
					OffScreenAngle = atan2(OffScreenDrawPos[1], OffScreenDrawPos[0]) * 57.2958;
					while (OffScreenAngle < 180)
					{
						OffScreenAngle += 360;
					}
					while (OffScreenAngle > 180)
					{
						OffScreenAngle -= 360;
					}
					OffScreenDrawPos += CenterPosV;

					bValidOffScreenIcon = true;
				}
			}

			//x,y,width,height,texture,bIsRightSide
			bool bAnySelected = false;
			for (int Iter = 0; Iter < Instruments.size(); Iter++)
			{
				Vector3 RadialPos(cos((double)CurrentAngle * 0.0174533), sin((double)CurrentAngle * 0.0174533), 0);

				if (HapticDevice->isDominantHand.getValue() == false)
				{
					RadialPos[0] = -RadialPos[0];
				}

				Vector3 DrawPos = CenterPosV + RadialPos * (CircleRadius + IconSize);
				DrawPos[0] -= HalfIconSize;
				DrawPos[1] -= HalfIconSize;

				bool bDoHover = false;

				if (bValidOffScreenIcon)
				{
					int TestActiveAngle = HapticDevice->isDominantHand.getValue() ? CurrentAngle : 180 - CurrentAngle;
					while (TestActiveAngle < 180)
					{
						TestActiveAngle += 360;
					}
					while (TestActiveAngle > 180)
					{
						TestActiveAngle -= 360;
					}
					if (abs(TestActiveAngle - OffScreenAngle) < DegreePerIter / 2)
					{
						bDoHover = true;
					}
				}

				bAnySelected |= bDoHover;

				//selected
				if (Instruments[Iter] == ActiveTool)
				{
					((core::visual::DrawToolGL*)vparams->drawTool())->DrawTextureQuad(DrawPos[0], DrawPos[1], IconSize, IconSize, glToolTextures[HM_IMG_BACKGROUND_SELECTED].get());
				}
				else if (bDoHover)
				{
					((core::visual::DrawToolGL*)vparams->drawTool())->DrawTextureQuad(DrawPos[0], DrawPos[1], IconSize, IconSize, glToolTextures[HM_IMG_BACKGROUND_HOVER].get());

					HoverTime += this->getContext()->getDt();
					SwapIDX = Iter;

					// done via device now
					//if (IsReadyToSwap())
					//{
					//	DoHoverSwap();
					//}
				}
				else
				{
					((core::visual::DrawToolGL*)vparams->drawTool())->DrawTextureQuad(DrawPos[0], DrawPos[1], IconSize, IconSize, glToolTextures[HM_IMG_BACKGROUND].get());
				}


				((core::visual::DrawToolGL*)vparams->drawTool())->DrawTextureQuad(DrawPos[0], DrawPos[1], IconSize, IconSize, glToolTextures[Instrument_TextureTypes[Iter]].get());
				CurrentAngle -= DegreePerIter;
			}

			if (bAnySelected == false)
			{
				HoverTime = 0;
			}

			// draw off screen icon
			if (bValidOffScreenIcon)
			{
				((core::visual::DrawToolGL*)vparams->drawTool())->DrawTextureQuad(OffScreenDrawPos[0], OffScreenDrawPos[1], 64, 64, glToolTextures[ActiveToolImage].get());
			}

			vparams->drawTool()->restoreLastState();
		}
	}

	bool SurfLabHapticInstruments::IsReadyToSwap()
	{
		return (HoverTime > 0.1);
	}

	void SurfLabHapticInstruments::DoHoverSwap()
	{
		if (SwapIDX < Instruments.size())
		{
			sofa::core::objectmodel::GUIEvent newGUIEvent(HapticDevice->getPathName().c_str(), "setTool", Instruments[SwapIDX]->getPathName().c_str());
			m_ScriptController->handleEvent(&newGUIEvent);
		}

		HoverTime = 0;
	}

	void SurfLabHapticInstruments::DisableActiveTool()
	{
		if (!isActiveToolDisabled && mActiveTool) {
			std::cout << "diable ActiveTool: " << mActiveTool->getName() << std::endl;
			isActiveToolDisabled = true;
			mActiveTool->setActive(false);
		}
	}

	void SurfLabHapticInstruments::EnableActiveTool()
	{
		if (isActiveToolDisabled && mActiveTool) {
			std::cout << "enable ActiveTool: " << mActiveTool->getName() << std::endl;
			isActiveToolDisabled = false;
			mActiveTool->setActive(true);
		}
	}

} // namespace SurfLab
