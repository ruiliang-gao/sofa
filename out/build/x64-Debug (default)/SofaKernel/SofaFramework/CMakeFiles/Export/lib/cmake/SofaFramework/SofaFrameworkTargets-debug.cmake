#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaCore" for configuration "Debug"
set_property(TARGET SofaCore APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaCore PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaCore_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaCore_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaCore )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaCore "${_IMPORT_PREFIX}/lib/SofaCore_d.lib" "${_IMPORT_PREFIX}/bin/SofaCore_d.dll" )

# Import target "SofaDefaultType" for configuration "Debug"
set_property(TARGET SofaDefaultType APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaDefaultType PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaDefaultType_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaDefaultType_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaDefaultType )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaDefaultType "${_IMPORT_PREFIX}/lib/SofaDefaultType_d.lib" "${_IMPORT_PREFIX}/bin/SofaDefaultType_d.dll" )

# Import target "SofaHelper" for configuration "Debug"
set_property(TARGET SofaHelper APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaHelper PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaHelper_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaHelper_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaHelper )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaHelper "${_IMPORT_PREFIX}/lib/SofaHelper_d.lib" "${_IMPORT_PREFIX}/bin/SofaHelper_d.dll" )

# Import target "SofaSimulationCore" for configuration "Debug"
set_property(TARGET SofaSimulationCore APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaSimulationCore PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaSimulationCore_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaSimulationCore_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaSimulationCore )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaSimulationCore "${_IMPORT_PREFIX}/lib/SofaSimulationCore_d.lib" "${_IMPORT_PREFIX}/bin/SofaSimulationCore_d.dll" )

# Import target "SofaFramework" for configuration "Debug"
set_property(TARGET SofaFramework APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaFramework PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaFramework_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaFramework_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaFramework )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaFramework "${_IMPORT_PREFIX}/lib/SofaFramework_d.lib" "${_IMPORT_PREFIX}/bin/SofaFramework_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
