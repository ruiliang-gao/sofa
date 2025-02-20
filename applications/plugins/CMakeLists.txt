cmake_minimum_required(VERSION 3.12)

find_package(SofaFramework)

# SofaTest needs to be after SofaMiscCollision, as it depends on it
if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
    # Library used to write high level tests involving many components.
    add_subdirectory(SofaTest)
endif()

sofa_add_subdirectory_external(SofaHighOrder SofaHighOrder)

sofa_add_plugin(CImgPlugin CImgPlugin ON) # ON by default and first as it is used by other plugins.
sofa_add_plugin(ArticulatedSystemPlugin ArticulatedSystemPlugin ON)
sofa_add_plugin(SofaEulerianFluid SofaEulerianFluid)
sofa_add_plugin(SofaSphFluid SofaSphFluid)
sofa_add_plugin(SofaDistanceGrid SofaDistanceGrid) # Depends on SofaMiscCollision
sofa_add_plugin(SofaImplicitField SofaImplicitField)
sofa_add_plugin(MultiThreading MultiThreading)
sofa_add_plugin(DiffusionSolver DiffusionSolver) # Depends on CImgPlugin
sofa_add_plugin_external(SofaPython SofaPython)
sofa_add_plugin(image image) # Depends on CImgPlugin, DiffusionSolver, MultiThreading (soft)
sofa_add_plugin(SofaNewmat SofaNewmat)

sofa_add_plugin_external(Compliant Compliant)
sofa_add_subdirectory_external(SofaPython3 SofaPython3)
sofa_add_plugin_external(CGALPlugin CGALPlugin) # Depends on image
sofa_add_plugin_external(Flexible Flexible)     # Depends on image, CImgPlugin, SofaHighOrderTopology (soft)
sofa_add_plugin_external(Registration Registration) # Depends on image, SofaPython, SofaGui and SofaDistanceGrid
sofa_add_plugin(BulletCollisionDetection BulletCollisionDetection) # Depends on Compliant and LMConstraint
sofa_add_plugin(PreassembledMass PreassembledMass) # Depends on Flexible and Compliant
sofa_add_plugin(ExternalBehaviorModel ExternalBehaviorModel)
sofa_add_plugin_external(InvertibleFVM InvertibleFVM)
sofa_add_plugin(MeshSTEPLoader MeshSTEPLoader)
sofa_add_plugin_external(PluginExample PluginExample)
sofa_add_plugin_external(ManifoldTopologies ManifoldTopologies)
sofa_add_plugin_external(OptiTrackNatNet OptiTrackNatNet)
sofa_add_plugin(SixenseHydra SixenseHydra)
sofa_add_plugin(SofaOpenCL SofaOpenCL)
sofa_add_plugin(Xitact Xitact)
sofa_add_plugin(Haption Haption)
sofa_add_plugin(ManualMapping ManualMapping)
sofa_add_plugin(PersistentContact PersistentContact)
sofa_add_plugin(Sensable Sensable)
sofa_add_plugin(SensableEmulation SensableEmulation)
sofa_add_plugin(SofaHAPI SofaHAPI)
sofa_add_plugin_external(THMPGSpatialHashing THMPGSpatialHashing)
sofa_add_plugin(SofaCarving SofaCarving)
sofa_add_plugin_external(RigidScale RigidScale)
sofa_add_plugin(LeapMotion LeapMotion)
sofa_add_plugin(Geomagic Geomagic)
sofa_add_plugin(SofaAssimp SofaAssimp) # ColladaSceneLoader Depends on Flexible and image
sofa_add_plugin(SofaMatrix SofaMatrix ON) # Depends on image, CImgPlugin
sofa_add_plugin_external(OpenCTMPlugin OpenCTMPlugin)

sofa_add_plugin_experimental(PSL PSL)

if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    sofa_add_plugin(SofaPardisoSolver SofaPardisoSolver) # SofaPardisoSolver is only available under linux with gcc
endif()

find_package(Sofa.GL QUIET)
if(Sofa.GL_FOUND)
    sofa_add_plugin(SofaCUDA SofaCUDA)           # SofaCUDA plugin can't work without OPENGL
    sofa_add_plugin(SofaSimpleGUI SofaSimpleGUI) # SofaSimpleGUI plugin can't work without OPENGL
    sofa_add_plugin(VolumetricRendering VolumetricRendering) # VolumetricRendering plugin can't work without OPENGL
else()
    message("Sofa.GL not found; disabling SofaCUDA, SofaSimpleGUI and VolumetricRendering plugins")
endif()
