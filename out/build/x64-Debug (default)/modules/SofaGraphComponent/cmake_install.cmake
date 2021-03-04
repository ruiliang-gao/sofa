# Install script for directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/install/x64-Debug (default)")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaGraphComponent/SofaGraphComponentTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaGraphComponent/SofaGraphComponentTargets.cmake"
         "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaGraphComponent/CMakeFiles/Export/lib/cmake/SofaGraphComponent/SofaGraphComponentTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaGraphComponent/SofaGraphComponentTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaGraphComponent/SofaGraphComponentTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaGraphComponent/CMakeFiles/Export/lib/cmake/SofaGraphComponent/SofaGraphComponentTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaGraphComponent/CMakeFiles/Export/lib/cmake/SofaGraphComponent/SofaGraphComponentTargets-debug.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaGraphComponent/SofaGraphComponentConfigVersion.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/cmake/SofaGraphComponentConfig.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/lib/SofaGraphComponent_d.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/bin/SofaGraphComponent_d.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/include/SofaGraphComponent/SofaGraphComponent/config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/initSofaGraphComponent.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/AddFrameButtonSetting.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/AddRecordedCameraButtonSetting.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/AttachBodyButtonSetting.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/FixPickedParticleButtonSetting.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/Gravity.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/InteractingBehaviorModel.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/MouseButtonSetting.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/PauseAnimation.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/PauseAnimationOnEvent.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/SofaDefaultPathSetting.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/StatsSetting.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/ViewerSetting.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/SceneCheck.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/SceneCheckDuplicatedName.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/SceneCheckMissingRequiredPlugin.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/SceneCheckAPIChange.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/SceneCheckUsingAlias.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/SceneCheckerVisitor.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/SceneCheckerListener.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaGraphComponent" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/src/SofaGraphComponent/APIVersion.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGraphComponent/README.md")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaGraphComponent/SofaGraphComponent_test/cmake_install.cmake")

endif()

