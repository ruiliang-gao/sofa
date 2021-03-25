#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaDeformable" for configuration "Release"
set_property(TARGET SofaDeformable APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaDeformable PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaDeformable.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaDeformable.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaDeformable )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaDeformable "${_IMPORT_PREFIX}/lib/SofaDeformable.lib" "${_IMPORT_PREFIX}/bin/SofaDeformable.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
