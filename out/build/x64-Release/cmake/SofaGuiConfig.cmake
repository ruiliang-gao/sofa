# CMake package configuration file for SofaGui

### Expanded from @PACKAGE_GUARD@ by SofaMacrosInstall.cmake ###
include_guard()
################################################################

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was SofaGuiConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(SOFAGUI_TARGETS SofaGuiCommon;SofaGuiQt;SofaGuiMain)

set(SOFAGUI_HAVE_SOFAHEADLESSRECORDER 0)
set(SOFAGUI_HAVE_SOFAGUIQT 1)
set(SOFAGUIQT_HAVE_QTVIEWER 1)
set(SOFAGUIQT_HAVE_QGLVIEWER 1)
set(SOFAGUIQT_HAVE_QT5_CHARTS 1)
set(SOFAGUIQT_HAVE_QT5_WEBENGINE 1)
set(SOFAGUIQT_HAVE_NODEEDITOR )
set(SOFAGUIQT_HAVE_TINYXML )

# Find dependencies
find_package(SofaFramework QUIET REQUIRED)
find_package(SofaUserInteraction QUIET REQUIRED)
find_package(SofaGraphComponent QUIET REQUIRED)
find_package(SofaMiscForceField QUIET REQUIRED) # SofaGuiQt
find_package(SofaLoader QUIET REQUIRED)

if(SOFAGUI_HAVE_SOFAGUIQT)
    if(SOFAGUIQT_HAVE_QTVIEWER)
        find_package(Qt5 QUIET REQUIRED Core Gui OpenGL)
        if(SOFAGUIQT_HAVE_QT5_CHARTS)
            find_package(Qt5 QUIET REQUIRED Charts)
        endif()
        if(SOFAGUIQT_HAVE_QT5_WEBENGINE)
            find_package(Qt5 QUIET REQUIRED WebEngine WebEngineWidgets)
        endif()
    endif()
    if(SOFAGUIQT_HAVE_QGLVIEWER)
        find_package(QGLViewer QUIET REQUIRED)
    endif()
    if(SOFAGUIQT_HAVE_NODEEDITOR)
        find_package(NodeEditor QUIET REQUIRED)
    endif()
    if(SOFAGUIQT_HAVE_TINYXML)
        find_package(TinyXML QUIET REQUIRED)
    endif()
endif()

foreach(target ${SOFAGUI_TARGETS})
    if(NOT TARGET ${target})
        include("${CMAKE_CURRENT_LIST_DIR}/SofaGuiTargets.cmake")
        break()
    endif()
endforeach()
if(NOT TARGET SofaGui)
    include("${CMAKE_CURRENT_LIST_DIR}/SofaGuiTargets.cmake")
endif()
check_required_components(SofaGui)
