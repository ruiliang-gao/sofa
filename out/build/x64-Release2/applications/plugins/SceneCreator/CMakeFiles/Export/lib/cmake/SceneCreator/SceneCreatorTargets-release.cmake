#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SceneCreator" for configuration "Release"
set_property(TARGET SceneCreator APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SceneCreator PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SceneCreator.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SceneCreator.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SceneCreator )
list(APPEND _IMPORT_CHECK_FILES_FOR_SceneCreator "${_IMPORT_PREFIX}/lib/SceneCreator.lib" "${_IMPORT_PREFIX}/bin/SceneCreator.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
