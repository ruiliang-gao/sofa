# Install script for directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaUserInteraction/SofaUserInteractionTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaUserInteraction/SofaUserInteractionTargets.cmake"
         "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaUserInteraction/CMakeFiles/Export/lib/cmake/SofaUserInteraction/SofaUserInteractionTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaUserInteraction/SofaUserInteractionTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaUserInteraction/SofaUserInteractionTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaUserInteraction/CMakeFiles/Export/lib/cmake/SofaUserInteraction/SofaUserInteractionTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaUserInteraction/CMakeFiles/Export/lib/cmake/SofaUserInteraction/SofaUserInteractionTargets-debug.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaUserInteraction/SofaUserInteractionConfigVersion.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/cmake/SofaUserInteractionConfig.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/lib/SofaUserInteraction_d.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/bin/SofaUserInteraction_d.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/include/SofaUserInteraction/SofaUserInteraction/config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/initSofaUserInteraction.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/InteractionPerformer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/MouseInteractor.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/MouseInteractor.inl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/AddRecordedCameraPerformer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/AttachBodyPerformer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/AttachBodyPerformer.inl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/ComponentMouseInteraction.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/ComponentMouseInteraction.inl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/Controller.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/FixParticlePerformer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/FixParticlePerformer.inl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/InciseAlongPathPerformer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/MechanicalStateController.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/MechanicalStateController.inl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/Ray.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/RayContact.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/RayDiscreteIntersection.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/RayDiscreteIntersection.inl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/RayModel.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/RayNewProximityIntersection.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/RayTraceDetection.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/SleepController.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/TopologicalChangeManager.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/RemovePrimitivePerformer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/RemovePrimitivePerformer.inl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/StartNavigationPerformer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/SuturePointPerformer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaUserInteraction" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/src/SofaUserInteraction/SuturePointPerformer.inl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaUserInteraction/README.md")
endif()

