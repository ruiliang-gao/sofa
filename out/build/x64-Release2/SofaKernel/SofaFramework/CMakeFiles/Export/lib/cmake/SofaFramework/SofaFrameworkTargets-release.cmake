#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaCore" for configuration "Release"
set_property(TARGET SofaCore APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaCore PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaCore.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaCore.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaCore )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaCore "${_IMPORT_PREFIX}/lib/SofaCore.lib" "${_IMPORT_PREFIX}/bin/SofaCore.dll" )

# Import target "SofaDefaultType" for configuration "Release"
set_property(TARGET SofaDefaultType APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaDefaultType PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaDefaultType.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaDefaultType.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaDefaultType )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaDefaultType "${_IMPORT_PREFIX}/lib/SofaDefaultType.lib" "${_IMPORT_PREFIX}/bin/SofaDefaultType.dll" )

# Import target "SofaHelper" for configuration "Release"
set_property(TARGET SofaHelper APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaHelper PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaHelper.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaHelper.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaHelper )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaHelper "${_IMPORT_PREFIX}/lib/SofaHelper.lib" "${_IMPORT_PREFIX}/bin/SofaHelper.dll" )

# Import target "SofaSimulationCore" for configuration "Release"
set_property(TARGET SofaSimulationCore APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaSimulationCore PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaSimulationCore.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaSimulationCore.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaSimulationCore )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaSimulationCore "${_IMPORT_PREFIX}/lib/SofaSimulationCore.lib" "${_IMPORT_PREFIX}/bin/SofaSimulationCore.dll" )

# Import target "SofaFramework" for configuration "Release"
set_property(TARGET SofaFramework APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaFramework PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaFramework.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaFramework.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaFramework )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaFramework "${_IMPORT_PREFIX}/lib/SofaFramework.lib" "${_IMPORT_PREFIX}/bin/SofaFramework.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
