# CMake package configuration file for SofaMisc

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was SofaMiscConfig.cmake.in                            ########

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

set(SOFAMISC_TARGETS SofaMisc;SofaMiscEngine;SofaMiscFem;SofaMiscForceField;SofaMiscMapping;SofaMiscSolver;SofaMiscTopology)

set(SOFAMISCTOPOLOGY_HAVE_ZLIB 1)
set(SOFAMISCFEM_HAVE_NEWMAT 1)

find_package(SofaAdvanced REQUIRED)
find_package(SofaGeneral REQUIRED)

if(SOFAMISCTOPOLOGY_HAVE_ZLIB)
    find_package(ZLIB QUIET REQUIRED)
endif()
if(SOFAMISCFEM_HAVE_NEWMAT)
    find_package(Newmat QUIET REQUIRED)
endif()

foreach(target ${SOFAMISC_TARGETS})
    if(NOT TARGET ${target})
        include("${CMAKE_CURRENT_LIST_DIR}/SofaMiscTargets.cmake")
        break()
    endif()
endforeach()

check_required_components(${SOFAMISC_TARGETS})
