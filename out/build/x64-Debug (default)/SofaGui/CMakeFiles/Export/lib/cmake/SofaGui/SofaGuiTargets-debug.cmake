#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaGuiCommon" for configuration "Debug"
set_property(TARGET SofaGuiCommon APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGuiCommon PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGuiCommon_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGuiCommon_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGuiCommon )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGuiCommon "${_IMPORT_PREFIX}/lib/SofaGuiCommon_d.lib" "${_IMPORT_PREFIX}/bin/SofaGuiCommon_d.dll" )

# Import target "SofaGuiQt" for configuration "Debug"
set_property(TARGET SofaGuiQt APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGuiQt PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGuiQt_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGuiQt_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGuiQt )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGuiQt "${_IMPORT_PREFIX}/lib/SofaGuiQt_d.lib" "${_IMPORT_PREFIX}/bin/SofaGuiQt_d.dll" )

# Import target "SofaGuiMain" for configuration "Debug"
set_property(TARGET SofaGuiMain APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGuiMain PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGuiMain_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGuiMain_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGuiMain )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGuiMain "${_IMPORT_PREFIX}/lib/SofaGuiMain_d.lib" "${_IMPORT_PREFIX}/bin/SofaGuiMain_d.dll" )

# Import target "SofaGui" for configuration "Debug"
set_property(TARGET SofaGui APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SofaGui PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/SofaGui_d.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/SofaGui_d.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS SofaGui )
list(APPEND _IMPORT_CHECK_FILES_FOR_SofaGui "${_IMPORT_PREFIX}/lib/SofaGui_d.lib" "${_IMPORT_PREFIX}/bin/SofaGui_d.dll" )

# Import target "runSofa" for configuration "Debug"
set_property(TARGET runSofa APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(runSofa PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/runSofa_d.exe"
  )

list(APPEND _IMPORT_CHECK_TARGETS runSofa )
list(APPEND _IMPORT_CHECK_FILES_FOR_runSofa "${_IMPORT_PREFIX}/bin/runSofa_d.exe" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
