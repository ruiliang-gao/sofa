# Install script for directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationGraph

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/lib/SofaSimulationGraph_d.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/bin/SofaSimulationGraph_d.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaSimulation/SofaSimulationGraph" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/include/SofaSimulation/SofaSimulationGraph/config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaSimulation/SofaSimulationGraph" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationGraph/DAGNode.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaSimulation/SofaSimulationGraph" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationGraph/DAGNodeMultiMappingElement.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaSimulation/SofaSimulationGraph" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationGraph/DAGSimulation.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaSimulation/SofaSimulationGraph" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationGraph/SimpleApi.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaSimulation/SofaSimulationGraph" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationGraph/init.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaSimulation/SofaSimulationGraph" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationGraph/graph.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaSimulation/SofaSimulationGraph/testing" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/SofaKernel/modules/SofaSimulationGraph/testing/BaseSimulationTest.h")
endif()

