#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "QGLViewer" for configuration "Debug"
set_property(TARGET QGLViewer APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(QGLViewer PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/QGLViewer_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/QGLViewer_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS QGLViewer )
list(APPEND _IMPORT_CHECK_FILES_FOR_QGLViewer "${_IMPORT_PREFIX}/lib/QGLViewer_d.lib" "${_IMPORT_PREFIX}/bin/QGLViewer_d.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
