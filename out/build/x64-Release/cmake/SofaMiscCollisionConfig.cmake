# CMake package configuration file for the SofaMiscCollision plugin

### Expanded from @PACKAGE_GUARD@ by SofaMacrosInstall.cmake ###
include_guard()
list(APPEND CMAKE_LIBRARY_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../bin")
list(APPEND CMAKE_LIBRARY_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../lib")
################################################################

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was SofaMiscCollisionConfig.cmake.in                            ########

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

set(SOFAMISCCOLLISION_HAVE_SOFASPHFLUID 0)
set(SOFAMISCCOLLISION_HAVE_SOFADISTANCEGRID 0)

if(SOFAMISCCOLLISION_HAVE_SOFASPHFLUID)
    find_package(SofaSphFluid QUIET REQUIRED)
endif()
if(SOFAMISCCOLLISION_HAVE_SOFADISTANCEGRID)
    find_package(SofaDistanceGrid QUIET REQUIRED)
endif()

if(NOT TARGET SofaMiscCollision)
    include("${CMAKE_CURRENT_LIST_DIR}/SofaMiscCollisionTargets.cmake")
endif()

check_required_components(SofaMiscCollision)
set(SofaMiscCollision_LIBRARIES SOFAMISCCOLLISION)
set(SofaMiscCollision_INCLUDE_DIRS  ${SOFAMISCCOLLISION_INCLUDE_DIR})
