#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaRigid" for configuration "Debug"
set_property(TARGET SofaRigid APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaRigid PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaRigid_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaRigid_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaRigid )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaRigid "${_IMPORT_PREFIX}/lib/SofaRigid_d.lib" "${_IMPORT_PREFIX}/bin/SofaRigid_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
