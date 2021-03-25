# CMake package configuration file for SofaCommon


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was SofaCommonConfig.cmake.in                            ########

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

set(SOFACOMMON_TARGETS SofaDeformable;SofaEngine;SofaExplicitOdeSolver;SofaImplicitOdeSolver;SofaLoader;SofaMeshCollision;SofaObjectInteraction;SofaRigid;SofaSimpleFem;SofaEigen2Solver)

find_package(SofaBase REQUIRED SofaComponentBase)

foreach(target ${SOFACOMMON_TARGETS})
    if(NOT TARGET ${target})
        include("${CMAKE_CURRENT_LIST_DIR}/SofaCommonTargets.cmake")
        break()
    endif()
endforeach()

check_required_components(${SOFACOMMON_TARGETS})
