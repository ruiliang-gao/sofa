# Install script for directory: C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/install/x64-Release2")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/QGLViewer/QGLViewerTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/QGLViewer/QGLViewerTargets.cmake"
         "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/SofaGui/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/CMakeFiles/Export/lib/cmake/QGLViewer/QGLViewerTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/QGLViewer/QGLViewerTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/QGLViewer/QGLViewerTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/SofaGui/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/CMakeFiles/Export/lib/cmake/QGLViewer/QGLViewerTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/SofaGui/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/CMakeFiles/Export/lib/cmake/QGLViewer/QGLViewerTargets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/SofaGui/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/QGLViewerConfigVersion.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/cmake/QGLViewerConfig.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/lib/Release/QGLViewer.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/bin/Release/QGLViewer.dll")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/AxisAlignedBox.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/Exporter.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/NVector3.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/Optimizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/ParserGL.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/Primitive.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/PrimitivePositioning.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/SortMethod.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/Types.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/VRender.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/Vector2.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/Vector3.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/VRender" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/VRender/gpc.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/constraint.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/domUtils.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/mouseGrabber.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/quaternion.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/modules/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/vec.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/SofaGui/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/ui_ImageInterface.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/extlibs/QGLViewer/QGLViewer/QGLViewer" TYPE FILE FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/SofaGui/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/ui_VRenderInterface.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE DIRECTORY FILES "C:/Files/Sheamus/School/Senior/SeniorProject/sofa_1906_tips/src/out/build/x64-Release2/SofaGui/SofaGuiQt/libQGLViewer-2.7.1/QGLViewer/windeployqt/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xapplicationsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Redist/MSVC/14.28.29910/x64/Microsoft.VC142.CRT/msvcp140.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Redist/MSVC/14.28.29910/x64/Microsoft.VC142.CRT/msvcp140_1.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Redist/MSVC/14.28.29910/x64/Microsoft.VC142.CRT/msvcp140_2.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Redist/MSVC/14.28.29910/x64/Microsoft.VC142.CRT/msvcp140_codecvt_ids.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Redist/MSVC/14.28.29910/x64/Microsoft.VC142.CRT/vcruntime140_1.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Redist/MSVC/14.28.29910/x64/Microsoft.VC142.CRT/vcruntime140.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Redist/MSVC/14.28.29910/x64/Microsoft.VC142.CRT/concrt140.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-console-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-console-l1-2-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-datetime-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-debug-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-errorhandling-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-file-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-file-l1-2-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-file-l2-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-handle-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-heap-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-interlocked-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-libraryloader-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-localization-l1-2-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-memory-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-namedpipe-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-processenvironment-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-processthreads-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-processthreads-l1-1-1.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-profile-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-rtlsupport-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-string-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-synch-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-synch-l1-2-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-sysinfo-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-timezone-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-core-util-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-conio-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-convert-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-environment-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-filesystem-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-heap-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-locale-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-math-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-multibyte-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-private-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-process-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-runtime-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-stdio-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-string-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-time-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/api-ms-win-crt-utility-l1-1-0.dll"
    "C:/Program Files (x86)/Windows Kits/10/Redist/10.0.18362.0/ucrt/DLLs/x64/ucrtbase.dll"
    )
endif()

