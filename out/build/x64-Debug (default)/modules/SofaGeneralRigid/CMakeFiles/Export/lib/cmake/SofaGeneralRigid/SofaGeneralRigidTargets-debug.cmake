#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralRigid" for configuration "Debug"
set_property(TARGET SofaGeneralRigid APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGeneralRigid PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGeneralRigid_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGeneralRigid_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralRigid )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralRigid "${_IMPORT_PREFIX}/lib/SofaGeneralRigid_d.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralRigid_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
