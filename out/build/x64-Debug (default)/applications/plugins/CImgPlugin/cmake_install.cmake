# Install script for directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/applications/plugins/CImgPlugin

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/lib/win64/libtiff.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/lib/win64/libtiff.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/lib/win64/libjpeg.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/lib/win64/libpng16.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/lib/win64/libpng16.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/lib/win64/zlib.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/lib/win64/zlib.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/CImgPlugin/CImgPluginTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/CImgPlugin/CImgPluginTargets.cmake"
         "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/applications/plugins/CImgPlugin/CMakeFiles/Export/lib/cmake/CImgPlugin/CImgPluginTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/CImgPlugin/CImgPluginTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/CImgPlugin/CImgPluginTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/CImgPlugin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/applications/plugins/CImgPlugin/CMakeFiles/Export/lib/cmake/CImgPlugin/CImgPluginTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/CImgPlugin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/applications/plugins/CImgPlugin/CMakeFiles/Export/lib/cmake/CImgPlugin/CImgPluginTargets-debug.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/CImgPlugin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/applications/plugins/CImgPlugin/CImgPluginConfigVersion.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/CImgPlugin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/cmake/CImgPluginConfig.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/lib/CImgPlugin_d.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/bin/CImgPlugin_d.dll")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CImgPlugin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/applications/plugins/CImgPlugin/src/CImgPlugin/ImageCImg.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CImgPlugin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/include/CImgPlugin/CImgPlugin/CImgPlugin.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CImgPlugin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/applications/plugins/CImgPlugin/src/CImgPlugin/SOFACImg.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CImgPlugin" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/applications/plugins/CImgPlugin/src/CImgPlugin/CImgData.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xresourcesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sofa/examples/CImgPlugin" TYPE DIRECTORY FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/applications/plugins/CImgPlugin/examples/")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/applications/plugins/CImgPlugin/extlibs/CImg/cmake_install.cmake")
  include("C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Debug (default)/applications/plugins/CImgPlugin/CImgPlugin_test/cmake_install.cmake")

endif()

