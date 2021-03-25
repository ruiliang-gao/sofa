#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SceneCreator" for configuration "Debug"
set_property(TARGET SceneCreator APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SceneCreator PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SceneCreator_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SceneCreator_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SceneCreator )
list(APPEND _IMPORT_CHECK_FILES_FOR_SceneCreator "${_IMPORT_PREFIX}/lib/SceneCreator_d.lib" "${_IMPORT_PREFIX}/bin/SceneCreator_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
