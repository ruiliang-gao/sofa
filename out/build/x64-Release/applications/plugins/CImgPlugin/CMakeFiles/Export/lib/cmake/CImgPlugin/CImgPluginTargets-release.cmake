#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CImgPlugin" for configuration "Release"
set_property(TARGET CImgPlugin APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(CImgPlugin PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/CImgPlugin.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/CImgPlugin.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS CImgPlugin )
list(APPEND _IMPORT_CHECK_FILES_FOR_CImgPlugin "${_IMPORT_PREFIX}/lib/CImgPlugin.lib" "${_IMPORT_PREFIX}/bin/CImgPlugin.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
