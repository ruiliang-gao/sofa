#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaBoundaryCondition" for configuration "Release"
set_property(TARGET SofaBoundaryCondition APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaBoundaryCondition PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaBoundaryCondition.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaBoundaryCondition.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaBoundaryCondition )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaBoundaryCondition "${_IMPORT_PREFIX}/lib/SofaBoundaryCondition.lib" "${_IMPORT_PREFIX}/bin/SofaBoundaryCondition.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
