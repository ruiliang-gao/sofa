#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaSimulationCommon" for configuration "Debug"
set_property(TARGET SofaSimulationCommon APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaSimulationCommon PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaSimulationCommon_d.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "tinyxml"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaSimulationCommon_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaSimulationCommon )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaSimulationCommon "${_IMPORT_PREFIX}/lib/SofaSimulationCommon_d.lib" "${_IMPORT_PREFIX}/bin/SofaSimulationCommon_d.dll" )

# Import target "SofaSimulationGraph" for configuration "Debug"
set_property(TARGET SofaSimulationGraph APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaSimulationGraph PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaSimulationGraph_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaSimulationGraph_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaSimulationGraph )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaSimulationGraph "${_IMPORT_PREFIX}/lib/SofaSimulationGraph_d.lib" "${_IMPORT_PREFIX}/bin/SofaSimulationGraph_d.dll" )

# Import target "SofaSimulationTree" for configuration "Debug"
set_property(TARGET SofaSimulationTree APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaSimulationTree PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaSimulationTree_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaSimulationTree_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaSimulationTree )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaSimulationTree "${_IMPORT_PREFIX}/lib/SofaSimulationTree_d.lib" "${_IMPORT_PREFIX}/bin/SofaSimulationTree_d.dll" )

# Import target "SofaSimulation" for configuration "Debug"
set_property(TARGET SofaSimulation APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaSimulation PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaSimulation_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaSimulation_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaSimulation )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaSimulation "${_IMPORT_PREFIX}/lib/SofaSimulation_d.lib" "${_IMPORT_PREFIX}/bin/SofaSimulation_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
