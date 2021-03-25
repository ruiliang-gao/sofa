#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGTestMain" for configuration "Release"
set_property(TARGET SofaGTestMain APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaGTestMain PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/SofaGTestMain.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGTestMain )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGTestMain "${_IMPORT_PREFIX}/lib/SofaGTestMain.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
