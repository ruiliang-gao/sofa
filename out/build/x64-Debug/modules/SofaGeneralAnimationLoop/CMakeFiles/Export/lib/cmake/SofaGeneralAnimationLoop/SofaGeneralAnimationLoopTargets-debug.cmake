#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGeneralAnimationLoop" for configuration "Debug"
set_property(TARGET SofaGeneralAnimationLoop APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGeneralAnimationLoop PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGeneralAnimationLoop_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGeneralAnimationLoop_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGeneralAnimationLoop )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGeneralAnimationLoop "${_IMPORT_PREFIX}/lib/SofaGeneralAnimationLoop_d.lib" "${_IMPORT_PREFIX}/bin/SofaGeneralAnimationLoop_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
