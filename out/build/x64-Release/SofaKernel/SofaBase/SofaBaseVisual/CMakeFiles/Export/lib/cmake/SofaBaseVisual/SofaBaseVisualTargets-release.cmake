#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaBaseVisual" for configuration "Release"
set_property(TARGET SofaBaseVisual APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaBaseVisual PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/SofaBaseVisual.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "tinyxml"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/SofaBaseVisual.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaBaseVisual )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaBaseVisual "${_IMPORT_PREFIX}/lib/SofaBaseVisual.lib" "${_IMPORT_PREFIX}/bin/SofaBaseVisual.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
