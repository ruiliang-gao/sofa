# Install script for directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaMiscExtra

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaMiscExtra/SofaMiscExtraTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaMiscExtra/SofaMiscExtraTargets.cmake"
         "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaMiscExtra/CMakeFiles/Export/lib/cmake/SofaMiscExtra/SofaMiscExtraTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaMiscExtra/SofaMiscExtraTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaMiscExtra/SofaMiscExtraTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaMiscExtra" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaMiscExtra/CMakeFiles/Export/lib/cmake/SofaMiscExtra/SofaMiscExtraTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaMiscExtra" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaMiscExtra/CMakeFiles/Export/lib/cmake/SofaMiscExtra/SofaMiscExtraTargets-debug.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaMiscExtra" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaMiscExtra/SofaMiscExtraConfigVersion.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaMiscExtra" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/cmake/SofaMiscExtraConfig.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/lib/SofaMiscExtra_d.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/bin/SofaMiscExtra_d.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaMiscExtra" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaMiscExtra/src/SofaMiscExtra/initSofaMiscExtra.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaMiscExtra" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/include/SofaMiscExtra/SofaMiscExtra/config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaMiscExtra" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaMiscExtra/src/SofaMiscExtra/MeshTetraStuffing.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaMiscExtra/README.md")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/modules/SofaMiscExtra/SofaMiscExtra_test/cmake_install.cmake")

endif()

