include(CMakePackageConfigHelpers)
include(CMakeParseLibraryList)


# - Create an imported target from a library path and an include dir path.
#   Handle the special case where LIBRARY_PATH is in fact an existing target.
#   Handle the case where LIBRARY_PATH contains the following syntax supported by cmake:
#                                      "optimized /usr/lib/foo.lib debug /usr/lib/foo_d.lib"
#
# sofa_create_imported_target(TARGETNAME LIBRARY_PATH INCLUDE_DIRS)
#  TARGETNAME_Target  - (output) variable which contains the name of the created target.
#                       It is usually contains TARGETNAME with one notable exception.
#                       If LIBRARY_PATH is an existing target, TARGETNAME_Target
#                       contains LIBRARY_PATH instead.
#  TARGETNAME         - (input) the name of the target to create.
#  NAMESPACE          - (input) the namespace where the target is put.
#  LIBRARY_PATH       - (input) the path to the library ( .so or .lib depending on the platform)
#  INCLUDE_DIRS       - (input) include directories associated with the library,
#                       which are added as INTERFACE_INCLUDE_DIRECTORIES for the target.
#
# The typical usage scenario is to convert the absolute paths to a system library that cmake return
# after a find_package call into an imported target. By using the cmake target mechanism, it is
# easier to redistribute a software that depends on system libraries, whose locations are not
# known before hand on the consumer system.
#
# For further reference about this subject :
# http://public.kitware.com/pipermail/cmake-developers/2014-March/009983.html
# Quoted from https://github.com/Kitware/CMake/blob/master/Help/manual/cmake-packages.7.rst
# "Note that it is not advisable to populate any properties which may contain paths,
#  such as :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` and :prop_tgt:`INTERFACE_LINK_LIBRARIES`,
#  with paths relevnt to dependencies. That would hard-code into installed packages the
#  include directory or library paths for dependencies as found on the machine the package
#  was made on."
#
# Example:
#
# add_library( SHARED myLib )
# find_package(PNG REQUIRED)
# sofa_create_target( PNG MyNamespace "${PNG_LIBRARY}" "${PNG_INCLUDE_DIRS}" )
# target_link_libraries( myLib PUBLIC ${PNG_Target} )
#
macro(sofa_create_target TARGETNAME NAMESPACE LIBRARY_PATH INCLUDE_DIRS)
    # message("TARGETNAME ${TARGETNAME}")
    set(NAMESPACE_TARGETNAME "${NAMESPACE}::${TARGETNAME}")
    # message("LIBRARY_PATH ${LIBRARY_PATH}")
    parse_library_list( "${LIBRARY_PATH}" FOUND LIB_FOUND DEBUG LIB_DEBUG OPT LIB_OPT GENERAL LIB_GEN )

    # message("FOUND ${LIB_FOUND} DEBUG: ${LIB_DEBUG} OPT: ${LIB_OPT} GEN: ${LIB_GEN}")
    if(LIB_FOUND)
        if(NOT TARGET ${TARGETNAME} )
            set(${TARGETNAME}_Target ${NAMESPACE_TARGETNAME} )
            if(NOT TARGET ${NAMESPACE_TARGETNAME} )
                add_library( ${NAMESPACE_TARGETNAME} UNKNOWN IMPORTED )
                set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}" )
                if( NOT ${LIB_DEBUG} STREQUAL "")
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION_DEBUG "${LIB_DEBUG}" )
                endif()
                if( NOT ${LIB_OPT} STREQUAL "")
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION "${LIB_OPT}" )
                elseif( NOT ${LIB_GEN} STREQUAL "" )
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION "${LIB_GEN}" )
                endif()
            endif()
        else()
            message( SEND_ERROR "sofa_create_target error. ${TARGETNAME} is an already an existing TARGET.\
                                 Choose a different name.")
        endif()
    else()
        if(NOT TARGET "${LIBRARY_PATH}" )
            # message("${LIBRARY_PATH} is not a TARGET")
            if(NOT TARGET ${TARGETNAME} )
                # message("${TARGETNAME} is not a TARGET")
                set(${TARGETNAME}_Target ${NAMESPACE_TARGETNAME} )
                if(NOT TARGET ${NAMESPACE_TARGETNAME} )
                    # message("${NAMESPACE_TARGETNAME} is not a TARGET")
                    add_library( ${NAMESPACE_TARGETNAME} UNKNOWN IMPORTED )
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION "${LIBRARY_PATH}" )
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}" )
                endif()
            else()
                message( SEND_ERROR "sofa_create_target error. ${TARGETNAME} is an already an existing TARGET.\
                                     Choose a different name.")
            endif()
        else()
            # message("${LIBRARY_PATH} is a TARGET")
            set(${TARGETNAME}_Target ${LIBRARY_PATH} )
        endif()

    endif()
endmacro()



macro(sofa_add_generic directory name type)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${directory}" AND IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${directory}")
        string(TOUPPER ${type}_${name} option)
        string(TOLOWER ${type} type_lower)

        # optional parameter to activate/desactivate the option
        #  e.g.  sofa_add_application( path/MYAPP MYAPP APPLICATION ON)
        set(active OFF)
        if(${ARGV3})
            if( ${ARGV3} STREQUAL ON )
                set(active ON)
            endif()
        endif()

        option(${option} "Build the ${name} ${type_lower}." ${active})
        if(${option})
            message("Adding ${type_lower} ${name}")
            add_subdirectory(${directory})
            #Check if the target has been successfully added
            if(TARGET ${name})
                set_target_properties(${name} PROPERTIES FOLDER ${type}s) # IDE folder
                set_target_properties(${name} PROPERTIES DEBUG_POSTFIX "_d")
            endif()
        endif()

        # Add current target in the internal list only if not present already
        get_property(_allTargets GLOBAL PROPERTY __GlobalTargetList__)
        get_property(_allTargetNames GLOBAL PROPERTY __GlobalTargetNameList__)

        # if(NOT ${name} IN_LIST _allTargets) # ONLY CMAKE >= 3.3 and policy to NEW
        list (FIND _allTargets ${name} _index)
        if(NOT ${_index} GREATER -1)
            set_property(GLOBAL APPEND PROPERTY __GlobalTargetList__ ${name})
        endif()

        #if(NOT ${option} IN_LIST _allTargetNames)# ONLY CMAKE >= 3.3 and policy to NEW
        list (FIND _allTargetNames ${option} _index)
        if(NOT ${_index} GREATER -1)
            set_property(GLOBAL APPEND PROPERTY __GlobalTargetNameList__ ${option})
        endif()
    else()
        message("The ${type_lower} ${name} (${CMAKE_CURRENT_LIST_DIR}/${directory}) does not exist and will be ignored.")
    endif()
endmacro()

macro(sofa_add_collection directory name)
    sofa_add_generic( ${directory} ${name} "Collection" ${ARGV2} )
endmacro()

macro(sofa_add_plugin directory plugin_name)
    sofa_add_generic( ${directory} ${plugin_name} "Plugin" ${ARGV2} )
endmacro()

macro(sofa_add_plugin_experimental directory plugin_name)
    sofa_add_generic( ${directory} ${plugin_name} "Plugin" ${ARGV2} )
    message("-- ${plugin_name} is an experimental feature, use it at your own risk.")
endmacro()

macro(sofa_add_module directory module_name)
    sofa_add_generic( ${directory} ${module_name} "Module" ${ARGV2} )
endmacro()

macro(sofa_add_module_experimental directory module_name)
    sofa_add_generic( ${directory} ${module_name} "Module" ${ARGV2} )
    message("-- ${module_name} is an experimental feature, use it at your own risk.")
endmacro()

macro(sofa_add_application directory app_name)
    sofa_add_generic( ${directory} ${app_name} "Application" ${ARGV2} )
endmacro()


### External projects management
# Thanks to http://crascit.com/2015/07/25/cmake-gtest/
#
# Use this macro (subdirectory or plugin version) to add out-of-repository projects.
# Usage:
# 1. Add repository configuration in MyProjectDir/ExternalProjectConfig.cmake.in
# 2. Call sofa_add_subdirectory_external(MyProjectDir MyProjectName [ON,OFF] [FETCH_ONLY])
#      or sofa_add_plugin_external(MyProjectDir MyProjectName [ON,OFF] [FETCH_ONLY])
# ON,OFF = execute the fetch by default + enable the fetched plugin (if calling sofa_add_plugin_external)
# FETCH_ONLY = do not "add_subdirectory" the fetched repository
# See plugins/SofaHighOrder for example
#
function(sofa_add_generic_external directory name type)
    set(optionArgs FETCH_ONLY)
    cmake_parse_arguments("ARG" "${optionArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Make directory absolute
    if(NOT IS_ABSOLUTE "${directory}")
        set(directory "${CMAKE_CURRENT_LIST_DIR}/${directory}")
    endif()
    if(NOT EXISTS "${directory}")
        message("${directory} does not exist and will be ignored.")
        return()
    endif()

    string(TOLOWER ${type} type_lower)

    # Default value for fetch activation and for plugin activation (if adding a plugin)
    set(active OFF)
    set(optional_argv3 "${ARGV3}")
    if(optional_argv3)
        set(active ${optional_argv3})
    endif()

    # Create option
    string(TOUPPER ${PROJECT_NAME}_FETCH_${name} fetch_enabled)
    option(${fetch_enabled} "Fetch/update ${name} repository." ${active})

    # Setup fetch directory
    set(fetched_dir "${CMAKE_BINARY_DIR}/external_directories/fetched/${name}" )

    # Fetch
    if(${fetch_enabled})
        message("Fetching ${type_lower} ${name}")

        if(NOT EXISTS ${fetched_dir})
            file(MAKE_DIRECTORY ${fetched_dir})
        endif()

        # Download and unpack at configure time
        configure_file(${directory}/ExternalProjectConfig.cmake.in ${fetched_dir}/CMakeLists.txt)
        # Copy ExternalProjectConfig.cmake.in in build dir for post-pull recovery in src dir
        file(COPY ${directory}/ExternalProjectConfig.cmake.in DESTINATION ${fetched_dir})

        # Execute commands to fetch content
        message("  Pulling ...")
        file(WRITE "${fetched_dir}/logs.txt" "") # Empty log file
        execute_process(COMMAND "${CMAKE_COMMAND}" -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY "${fetched_dir}"
            RESULT_VARIABLE generate_exitcode
            OUTPUT_VARIABLE generate_logs ERROR_VARIABLE generate_logs)
        file(APPEND "${fetched_dir}/logs.txt" "${generate_logs}")
        execute_process(COMMAND "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY "${fetched_dir}"
            RESULT_VARIABLE build_exitcode
            OUTPUT_VARIABLE build_logs ERROR_VARIABLE build_logs)
        file(APPEND "${fetched_dir}/logs.txt" "${build_logs}")

        if(generate_exitcode EQUAL 0 AND build_exitcode EQUAL 0 AND EXISTS "${directory}/.git")
            message("  Sucess.")
            # Add .gitignore for Sofa
            file(WRITE "${directory}/.gitignore" "*")
            # Recover ExternalProjectConfig.cmake.in from build dir (erased by pull)
            file(COPY ${fetched_dir}/ExternalProjectConfig.cmake.in DESTINATION ${directory})
            # Disable fetching for next configure
            set(${fetch_enabled} OFF CACHE BOOL "Fetch/update ${name} repository." FORCE)
            message("  ${fetch_enabled} is now OFF. Set it back to ON to trigger a new fetch.")
        else()
            message(SEND_ERROR "Failed to add external repository ${name}."
                               "\nSee logs in ${fetched_dir}/logs.txt")
        endif()
    endif()

    # Add
    if(EXISTS "${directory}/.git" AND IS_DIRECTORY "${directory}/.git")
        configure_file(${directory}/ExternalProjectConfig.cmake.in ${fetched_dir}/CMakeLists.txt)
        if(NOT ARG_FETCH_ONLY AND "${type}" STREQUAL "Subdirectory")
            add_subdirectory("${directory}")
        elseif(NOT ARG_FETCH_ONLY AND "${type}" STREQUAL "Plugin")
            sofa_add_plugin("${name}" "${name}" ${active})
        endif()
    endif()
endfunction()

function(sofa_add_subdirectory_external directory name)
    sofa_add_generic_external(${directory} ${name} "Subdirectory" ${ARGN})
endfunction()

function(sofa_add_plugin_external directory name)
    sofa_add_generic_external(${directory} ${name} "Plugin" ${ARGN})
endfunction()



# Declare a (unique, TODO?) directory containing the python scripts of
# a plugin.  This macro:
# - creates rules to install all the .py scripts in ${directory} to
#   lib/python2.7/site-packages/${plugin_name}
# - creates a etc/sofa/python.d/${plugin_name} file in the build tree
#   pointing to the source tree
# - creates a etc/sofa/python.d/${plugin_name} file in the install
#   tree, containing a relative path to the installed script directory
#
# Assumes relative path.
macro(sofa_set_python_directory plugin_name directory)
    message(WARNING "sofa_set_python_directory is deprecated. Use sofa_install_pythonscripts instead.")
    sofa_install_pythonscripts(PLUGIN_NAME "${plugin_name}" PYTHONSCRIPTS_SOURCE_DIR "${directory}")
endmacro()

macro(sofa_install_pythonscripts)
    set(oneValueArgs PLUGIN_NAME PYTHONSCRIPTS_SOURCE_DIR PYTHONSCRIPTS_INSTALL_DIR)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments("ARG" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Required arguments
    foreach(arg ARG_PLUGIN_NAME ARG_PYTHONSCRIPTS_SOURCE_DIR)
        if("${${arg}}" STREQUAL "")
            string(SUBSTRING "${arg}" 4 -1 arg_name) # arg name without "ARG_"
            message(SEND_ERROR "Missing parameter ${arg_name}.")
        endif()
    endforeach()

    set(include_install_dir "lib/python2.7/site-packages")
    if(ARG_PYTHONSCRIPTS_INSTALL_DIR)
        set(include_install_dir "${ARG_PYTHONSCRIPTS_INSTALL_DIR}")
    endif()

    ## Install python scripts, preserving the file tree
    file(GLOB_RECURSE ALL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_PYTHONSCRIPTS_SOURCE_DIR}/*")
    file(GLOB_RECURSE PYC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_PYTHONSCRIPTS_SOURCE_DIR}/*.pyc")
    if(PYC_FILES)
        list(REMOVE_ITEM ALL_FILES ${PYC_FILES})
    endif()
    foreach(python_file ${ALL_FILES})
        file(RELATIVE_PATH script "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_PYTHONSCRIPTS_SOURCE_DIR}" "${python_file}")
        get_filename_component(path ${script} DIRECTORY)
        install(FILES ${ARG_PYTHONSCRIPTS_SOURCE_DIR}/${script}
                DESTINATION "${include_install_dir}/${path}"
                COMPONENT applications)
    endforeach()

    ## Python configuration file (build tree)
    file(WRITE "${CMAKE_BINARY_DIR}/etc/sofa/python.d/${ARG_PLUGIN_NAME}"
         "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_PYTHONSCRIPTS_SOURCE_DIR}")
    ## Python configuration file (install tree)
     file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaPython-config"
         "${include_install_dir}")
     install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaPython-config"
             DESTINATION "etc/sofa/python.d"
             RENAME "${ARG_PLUGIN_NAME}"
             COMPONENT applications)
endmacro()


# - Create a target for a python binding module relying on pybind11
#
# sofa_add_pybind11_module(TARGET OUTPUT SOURCES DEPENDS CYTHONIZE)
#  TARGET             - (input) the name of the generated target.
#  OUTPUT             - (input) the output location.
#  SOURCES            - (input) list of input files. It can be .cpp, .h ...
#  DEPENDS            - (input) set of target the generated target will depends on.
#  NAME               - (input) The actual name of the generated .so file
#                       (most commonly equals to TARGET, without the "python" prefix)
#
# The typical usage scenario is to build a python module out of cython binding.
#
# Example:
# find_package(pybind11)
# set(SOURCES_FILES
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/initbindings.cpp
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/binding1.cpp
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/binding2.cpp
#       [...]
#    )
# sofa_add_pybind11_module( TARGET MyModule SOURCES ${SOURCE_FILES}
#                           DEPENDS Deps1 Deps2  OUTPUT ${CMAKE_CURRENT_BIN_DIR}
#                           NAME python_module_name)
function(sofa_add_pybind11_module)
    set(options)
    set(oneValueArgs TARGET OUTPUT NAME)
    set(multiValueArgs SOURCES DEPENDS)
    cmake_parse_arguments("" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    set(PYBIND11_CPP_STANDARD -std=c++11)
    pybind11_add_module(${_TARGET} SHARED ${_SOURCES} NO_EXTRAS)
    target_link_libraries(${_TARGET} PRIVATE ${_DEPENDS} ${PYTHON_LIBRARIES} pybind11::module)
    set_target_properties(${_TARGET} PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${_OUTPUT}
      LIBRARY_OUTPUT_DIRECTORY ${_OUTPUT}
      RUNTIME_OUTPUT_DIRECTORY ${_OUTPUT}
      OUTPUT_NAME ${_NAME})
endfunction()


# - Create a target for a mixed python module composed of .py and binding code (in .cpp or .pyx)
#
# sofa_add_python_module(TARGET OUTPUT SOURCES DEPENDS CYTHONIZE)
#  TARGET             - (input) the name of the generated target.
#  OUTPUT             - (input) the output location, if not provided ${CMAKE_CURRENT_SOURCE_DIR} will be used. 
#  SOURCES            - (input) list of input files. It can be .py, .pyx, .pxd, .cpp
#                               .cpp are compiled, .pyx can generate .cpp if CYTHONIZE param is set to true
#  DEPENDS            - (input) set of target the generated target will depends on.
#  CYTHONIZE          - (input) boolean indicating wether or not
#                               we need to call cython on the .pyx file to re-generate the .cpp file.
#
# The typical usage scenario is to build a python module out of cython binding.
#
# Example:
# find_package(Cython QUIET)
# set(SOURCES_FILES
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/__init__.py
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/purepython.py
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/binding_withCython.pyx
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/binding_withCython.pxd
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/binding_withCPython.cpp
#    )
# sofa_add_python_module( TARGET MyModule SOURCES ${SOURCE_FILES} DEPENDS Deps1 Deps2 CYTHONIZE True OUTPUT ${CMAKE_CURRENT_BIN_DIR})
function(sofa_add_python_module)
    set(options)
    set(oneValueArgs TARGET OUTPUT CYTHONIZE DIRECTORY)
    set(multiValueArgs SOURCES DEPENDS)
    cmake_parse_arguments("" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    set(INCLUDE_DIRS)
    set(LIB_DIRS)

    add_custom_target(${_TARGET}
                       ALL
                       SOURCES ${_SOURCES}
                       DEPENDS ${_DEPENDS})

    if(NOT PYTHON_BINDING_VERSION)
        set(PYTHON_BINDING_VERSION 3)
    endif()

    set(_DIRECTORY ${_OUTPUT})

    foreach( source ${_SOURCES} )
        unset(cppfile)
        get_filename_component(pathdir ${source} DIRECTORY)
        get_filename_component(filename ${source} NAME_WE)
        get_filename_component(ext ${source} EXT)

        if((${ext} STREQUAL ".cpp"))
            set(cppfile "${pathdir}/${filename}.cpp")
        endif()

        if(_CYTHONIZE AND (${ext} STREQUAL ".pyx"))
            set(pyxfile "${pathdir}/${filename}.pyx")
            set(cppfile "${pathdir}/${filename}.cpp")

            # Build the .cpp out of the .pyx
            add_custom_command(
                COMMAND cython ${pathdir}/${filename}${ext} --cplus -${PYTHON_BINDING_VERSION} --fast-fail --force # Execute this command,
                DEPENDS ${_SOURCES} ${_DEPENDS}                                                     # The target depens on these files...
                WORKING_DIRECTORY ${_DIRECTORY}                                   # In this working directory
                OUTPUT ${cppfile}
            )

            message("-- ${_TARGET} cython generated '${cppfile}' from '${filename}${ext}'" )
        endif()

        if(cppfile)
            set(pyxtarget "${_TARGET}_${filename}")
            add_library(${pyxtarget} SHARED ${cppfile})

            # The implementation of Python deliberately breaks strict-aliasing rules, so we
            # compile with -fno-strict-aliasing to prevent the compiler from relying on
            # those rules to optimize the code.
            if(${CMAKE_COMPILER_IS_GNUCC})
                set(SOFACYTHON_COMPILER_FLAGS "-fno-strict-aliasing")
            endif()

            target_link_libraries(${pyxtarget} ${_DEPENDS} ${PYTHON_LIBRARIES})
            target_include_directories(${pyxtarget} PRIVATE ${PYTHON_INCLUDE_DIRS})
            target_compile_options(${pyxtarget} PRIVATE ${SOFACYTHON_COMPILER_FLAGS})
            set_target_properties(${pyxtarget}
                PROPERTIES
                ARCHIVE_OUTPUT_DIRECTORY "${_OUTPUT}"
                LIBRARY_OUTPUT_DIRECTORY "${_OUTPUT}"
                RUNTIME_OUTPUT_DIRECTORY "${_OUTPUT}"
                )

            set_target_properties(${pyxtarget} PROPERTIES PREFIX "")
            set_target_properties(${pyxtarget} PROPERTIES OUTPUT_NAME "${filename}")

            add_dependencies(${_TARGET} ${pyxtarget})
        endif()
    endforeach()
endfunction()


# sofa_set_01
#
# Defines a variable to
#   - 1 if VALUE is 1, ON, YES, TRUE, Y, or a non-zero number.
#   - 0 if VALUE is 0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, the empty string, or ends in the suffix -NOTFOUND.
# This macro is used to quickly define variables for "#define SOMETHING ${SOMETHING}" in config.h.in files.
# PARENT_SCOPE (option): set the variable only in parent scope
# BOTH_SCOPES (option): set the variable in current AND parent scopes
macro(sofa_set_01 name)
    set(optionArgs PARENT_SCOPE BOTH_SCOPES)
    set(oneValueArgs VALUE)
    set(multiValueArgs)
    cmake_parse_arguments("ARG" "${optionArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    if(ARG_VALUE)
        if(ARG_BOTH_SCOPES OR NOT ARG_PARENT_SCOPE)
            set(${name} 1)
        endif()
        if(ARG_BOTH_SCOPES OR ARG_PARENT_SCOPE)
            set(${name} 1 PARENT_SCOPE)
        endif()
    else()
        if(ARG_BOTH_SCOPES OR NOT ARG_PARENT_SCOPE)
            set(${name} 0)
        endif()
        if(ARG_BOTH_SCOPES OR ARG_PARENT_SCOPE)
            set(${name} 0 PARENT_SCOPE)
        endif()
    endif()
endmacro()


# sofa_find_package
#
# Defines a PROJECTNAME_HAVE_PACKAGENAME variable to be used in:
#  - XXXConfig.cmake.in to decide if find_dependency must be done
#  - config.h.in as a #cmakedefine
#  - config.h.in as a #define SOMETHING ${SOMETHING}
# BOTH_SCOPES (option): set the variable in current AND parent scopes
macro(sofa_find_package name)
    set(optionArgs QUIET REQUIRED BOTH_SCOPES)
    set(oneValueArgs)
    set(multiValueArgs COMPONENTS OPTIONAL_COMPONENTS)
    cmake_parse_arguments("ARG" "${optionArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    set(find_package_args ${ARGN})
    if(find_package_args)
        list(REMOVE_ITEM find_package_args "BOTH_SCOPES")
    endif()
    find_package(${name} ${find_package_args})
    string(TOUPPER ${name} name_upper)
    string(TOUPPER ${PROJECT_NAME} project_upper)
    set(scopes "") # nothing = current scope only
    if(ARG_BOTH_SCOPES)
        set(scopes "BOTH_SCOPES")
    endif()
    if(ARG_COMPONENTS OR ARG_OPTIONAL_COMPONENTS)
        foreach(component ${ARG_COMPONENTS} ${ARG_OPTIONAL_COMPONENTS})
            string(TOUPPER ${component} component_upper)
            if(TARGET ${name}::${component})
                sofa_set_01(${project_upper}_HAVE_${name_upper}_${component_upper} VALUE TRUE ${scopes})
            else()
                sofa_set_01(${project_upper}_HAVE_${name_upper}_${component_upper} VALUE FALSE ${scopes})
            endif()
        endforeach()
    else()
        if(${name}_FOUND OR ${name_upper}_FOUND)
            sofa_set_01(${project_upper}_HAVE_${name_upper} VALUE TRUE ${scopes})
        else()
            sofa_set_01(${project_upper}_HAVE_${name_upper} VALUE FALSE ${scopes})
        endif()
    endif()
endmacro()



##########################################################
#################### INSTALL MACROS ######################
##########################################################

# sofa_install_targets
#
# package_name: Name of the package. One package can contain multiple targets. All the targets will be exported in ${package_name}Targets.
# the_targets: The targets to add to this package
# include_install_dir: Name of the INSTALLED directory that will contain headers
# (ARGV3) include_source_dir: Directory from which include tree will start (default: ${CMAKE_CURRENT_SOURCE_DIR})
# (ARGV4) example_install_dir: Name of the INSTALLED directory that will contain examples (default: share/sofa/${package_name}/examples)
macro(sofa_install_targets package_name the_targets include_install_dir)
    install(TARGETS ${the_targets}
            EXPORT ${package_name}Targets
            RUNTIME DESTINATION "bin" COMPONENT applications
            LIBRARY DESTINATION "lib" COMPONENT libraries
            ARCHIVE DESTINATION "lib" COMPONENT libraries
            PUBLIC_HEADER DESTINATION "include/${include_install_dir}" COMPONENT headers

            # [MacOS] install runSofa above the already populated runSofa.app (see CMAKE_INSTALL_PREFIX)
            BUNDLE DESTINATION "../../.." COMPONENT applications
            )

    # non-flat headers install (if no PUBLIC_HEADER and include_install_dir specified)
    foreach(target ${the_targets})
        set(version ${${target}_VERSION})
        string(TOUPPER "${package_name}" package_name_upper)
        if(version VERSION_GREATER "0.0")
            set_target_properties(${target} PROPERTIES VERSION "${version}")
        elseif(target MATCHES "^Sofa" AND NOT PLUGIN_${package_name_upper} AND Sofa_VERSION)
            # Default to Sofa_VERSION for all SOFA modules
            set_target_properties(${target} PROPERTIES VERSION "${Sofa_VERSION}")
        endif()

        get_target_property(target_sources ${target} SOURCES)
        #list(FILTER ${target_sources} INCLUDE REGEX ".*\.h\.in$") # CMake >= 3.6
        foreach(filepath ${target_sources})
            if("${filepath}" MATCHES "\\.*\\.h\\.in$")
                get_filename_component(filename ${filepath} NAME_WE)

                set(configure_dir "${CMAKE_BINARY_DIR}/include/${include_install_dir}")
                if("${package_name}" STREQUAL "${target}")
                    # target is a plugin
                    string(REPLACE "${target}/${target}" "${target}" configure_dir "${configure_dir}")
                else()
                    # target is an old module
                    string(REPLACE "include/${package_name}" "include" configure_dir "${configure_dir}")
                endif()

                configure_file("${filepath}" "${configure_dir}/${filename}.h")
                install(FILES "${configure_dir}/${filename}.h" DESTINATION "include/${include_install_dir}")
            endif()
        endforeach()

        get_target_property(public_header ${target} PUBLIC_HEADER)
        if("${public_header}" STREQUAL "public_header-NOTFOUND" AND NOT "${include_install_dir}" STREQUAL "")
            set(optional_argv3 "${ARGV3}")
            if(optional_argv3)
                # ARGV3 is a non-breaking additional argument to handle INCLUDE_SOURCE_DIR (see sofa_generate_package)
                # TODO: add a real argument "include_source_dir" to this macro
                set(include_source_dir "${ARGV3}")
            endif()
            if(NOT include_source_dir)
                set(include_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
            elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${include_source_dir}")
                # will be true if include_source_dir is empty
                set(include_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/${include_source_dir}")
            endif()
            #message("${target}: ${include_source_dir} -> include/${include_install_dir}")
            file(GLOB_RECURSE header_files "${include_source_dir}/*.h" "${include_source_dir}/*.inl")
            foreach(header ${header_files})
                file(RELATIVE_PATH path_from_package "${include_source_dir}" "${header}")
                get_filename_component(dir_from_package ${path_from_package} DIRECTORY)
                install(FILES ${header}
                        DESTINATION "include/${include_install_dir}/${dir_from_package}"
                        COMPONENT headers)
            endforeach()
        endif()
    endforeach()

    ## Default install rules for resources
    set(example_install_dir "share/sofa/examples/${package_name}")
    set(optional_argv4 "${ARGV4}")
    if(optional_argv4)
        # ARGV3 is a non-breaking additional argument to handle EXAMPLE_INSTALL_DIR (see sofa_generate_package)
        # TODO: add a real argument "example_install_dir" to this macro
        set(example_install_dir "${optional_argv4}")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/" DESTINATION "${example_install_dir}" COMPONENT resources)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scenes")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scenes/" DESTINATION "${example_install_dir}" COMPONENT resources)
    endif()

    # RELOCATABLE optional arg
    set(optional_argv5 "${ARGV5}")
    if(optional_argv5)
        sofa_set_install_relocatable(${package_name} ${optional_argv5})
    endif()
endmacro()



# sofa_set_install_relocatable
#   TARGET MUST EXIST, TO BE CALLED AFTER add_library
# Content:
#   If building out of SOFA: does nothing.
#   If building through SOFA: call add_custom_target with custom commands to obtain a self-contained relocatable install.
#   Self-contained plugins are useful to build modular binaries: they do not "pollute" SOFA install
#   with self-contained plugins SOFA install will always look the same, no matter how many plugins are included.
# Effect:
#   add_custom_target will add the line 'set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${install_dir}/${name}")' at the top of the
#   plugin's cmake_install.cmake to force the plugin to be installed in it's own directory instead of in SOFA's install directory
#   (look at the build directory of any plugin to find an example of cmake_install.cmake).
function(sofa_set_install_relocatable target install_dir)
    if(NOT SOFA_KERNEL_SOURCE_DIR)
        # not building through SOFA
        return()
    endif()
    if(NOT TARGET ${target})
        message(WARNING "sofa_set_install_relocatable: \"${target}\" is not an existing target.")
        return()
    endif()

    get_target_property(target_binary_dir ${target} BINARY_DIR)

    # Remove cmakepatch file at each configure
    file(REMOVE "${target_binary_dir}/cmake_install.cmakepatch")

    # Hack to make installed plugin independant and keep the add_subdirectory mechanism
    # Does not fail if cmakepatch file already exists thanks to "|| true"
    if(WIN32)
        set(escaped_dollar "\$")
        if(CMAKE_SYSTEM_VERSION VERSION_LESS 10 ) # before Windows 10
            set(escaped_dollar "\$\$")
        endif()
        string(REGEX REPLACE "/" "\\\\" target_binary_dir_windows "${target_binary_dir}")
        add_custom_target(${target}_relocatable_install ALL
            COMMENT "${target}: Patching cmake_install.cmake"
            COMMAND
                if not exist \"${target_binary_dir}/cmake_install.cmakepatch\"
                echo set ( CMAKE_INSTALL_PREFIX_BACK \"${escaped_dollar}\{CMAKE_INSTALL_PREFIX\}\" )
                    > \"${target_binary_dir}/cmake_install.cmakepatch\"
                && echo set ( CMAKE_INSTALL_PREFIX \"${escaped_dollar}\{CMAKE_INSTALL_PREFIX\}/${install_dir}/${target}\" )
                    >> \"${target_binary_dir}/cmake_install.cmakepatch\"
                && type \"${target_binary_dir_windows}\\cmake_install.cmake\" >> \"${target_binary_dir_windows}\\cmake_install.cmakepatch\"
                && echo set ( CMAKE_INSTALL_PREFIX \"${escaped_dollar}\{CMAKE_INSTALL_PREFIX_BACK\}\" )
                    >> \"${target_binary_dir}/cmake_install.cmakepatch\"
                && ${CMAKE_COMMAND} -E copy \"${target_binary_dir}/cmake_install.cmakepatch\" \"${target_binary_dir}/cmake_install.cmake\"
            )
    else()
        add_custom_target(${target}_relocatable_install ALL
            COMMENT "${target}: Patching cmake_install.cmake"
            COMMAND
                test ! -e ${target_binary_dir}/cmake_install.cmakepatch
                && echo \" set ( CMAKE_INSTALL_PREFIX_BACK \\"\\$$\{CMAKE_INSTALL_PREFIX\}\\" ) \"
                    > "${target_binary_dir}/cmake_install.cmakepatch"
                && echo \" set ( CMAKE_INSTALL_PREFIX \\"\\$$\{CMAKE_INSTALL_PREFIX\}/${install_dir}/${target}\\" ) \"
                    >> "${target_binary_dir}/cmake_install.cmakepatch"
                && cat ${target_binary_dir}/cmake_install.cmake >> ${target_binary_dir}/cmake_install.cmakepatch
                && echo \" set ( CMAKE_INSTALL_PREFIX \\"\\$$\{CMAKE_INSTALL_PREFIX_BACK\}\\" ) \"
                    >> "${target_binary_dir}/cmake_install.cmakepatch"
                && ${CMAKE_COMMAND} -E copy ${target_binary_dir}/cmake_install.cmakepatch ${target_binary_dir}/cmake_install.cmake
                || true
            )
    endif()
endfunction()



# sofa_write_package_config_files(Foo <version> <build-include-dirs>)
#
# Create CMake package configuration files
# - In the build tree:
#   - ${CMAKE_BINARY_DIR}/cmake/FooConfig.cmake
#   - ${CMAKE_BINARY_DIR}/cmake/FooConfigVersion.cmake
# - In the install tree:
#   - lib/cmake/Foo/FooConfigVersion.cmake
#   - lib/cmake/Foo/FooConfig.cmake
#   - lib/cmake/Foo/FooTargets.cmake
#
# This macro factorizes boilerplate CMake code for the different
# packages in Sofa.  It assumes that there is a FooConfig.cmake.in
# file template in the same directory.  For example, if a package Foo
# depends on Bar and Baz, and creates the targets Foo and Qux, here is
# a typical FooConfig.cmake.in:
#
# @PACKAGE_INIT@
#
# find_package(Bar REQUIRED)
# find_package(Baz REQUIRED)
#
# if(NOT TARGET Qux)
# 	include("${CMAKE_CURRENT_LIST_DIR}/FooTargets.cmake")
# endif()
#
# check_required_components(Foo Qux)
macro(sofa_write_package_config_files package_name version)
    ## <package_name>Targets.cmake
    install(EXPORT ${package_name}Targets DESTINATION "lib/cmake/${package_name}" COMPONENT headers)

    ## <package_name>ConfigVersion.cmake
    set(filename ${package_name}ConfigVersion.cmake)
    write_basic_package_version_file(${filename} VERSION ${version} COMPATIBILITY ExactVersion)
    configure_file("${CMAKE_CURRENT_BINARY_DIR}/${filename}" "${CMAKE_BINARY_DIR}/cmake/${filename}" COPYONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${filename}" DESTINATION "lib/cmake/${package_name}" COMPONENT headers)

    ### <package_name>Config.cmake
    configure_package_config_file(
        ${package_name}Config.cmake.in
        "${CMAKE_BINARY_DIR}/cmake/${package_name}Config.cmake"
        INSTALL_DESTINATION "lib/cmake/${package_name}"
        )
    install(FILES "${CMAKE_BINARY_DIR}/cmake/${package_name}Config.cmake" DESTINATION "lib/cmake/${package_name}" COMPONENT headers)
endmacro()


# - Create a target for SOFA plugin or module
# - write the package Config, Version & Target files
# - Deploy the headers, resources, scenes & examples
# - Replaces the now deprecated sofa_create_package macro
#
# sofa_generate_package(NAME VERSION TARGETS INCLUDE_INSTALL_DIR INCLUDE_SOURCE_DIR)
#  NAME                - (input) the name of the generated package (usually ${PROJECT_NAME}).
#  VERSION             - (input) the package version (usually ${PROJECT_VERSION}).
#  TARGETS             - (input) list of targets to install. For standard plugins & modules, ${PROJECT_NAME}
#  INCLUDE_INSTALL_DIR - (input) [OPTIONAL] include directory (for Multi-dir install of header files).
#  INCLUDE_SOURCE_DIR  - (input) [OPTIONAL] install headers with same tree structure as source starting from this dir (defaults to ${CMAKE_CURRENT_SOURCE_DIR})
#
# Example:
# project(ExamplePlugin VERSION 1.0)
# find_package(SofaFramework)
# set(SOURCES_FILES  initExamplePlugin.cpp myComponent.cpp )
# set(HEADER_FILES   initExamplePlugin.h myComponent.h )
# add_library( ${PROJECT_NAME} SHARED ${SOURCE_FILES})
# target_link_libraries(${PROJECT_NAME} SofaCore)
# sofa_generate_package(NAME ${PROJECT_NAME} VERSION ${PROJECT_VERSION} TARGETS ${PROJECT_NAME} INCLUDE_INSTALL_DIR "sofa/custom/install/dir" INCLUDE_SOURCE_DIR src/${PROJECT_NAME} )
#
function(sofa_generate_package)
    set(oneValueArgs NAME VERSION INCLUDE_ROOT_DIR INCLUDE_INSTALL_DIR INCLUDE_SOURCE_DIR EXAMPLE_INSTALL_DIR RELOCATABLE)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments("ARG" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # Required arguments
    foreach(arg ARG_NAME ARG_VERSION ARG_TARGETS)
        if("${${arg}}" STREQUAL "")
            string(SUBSTRING "${arg}" 4 -1 arg_name)
            message(SEND_ERROR "Missing parameter ${arg_name}.")
        endif()
    endforeach()

    set(include_install_dir "${ARG_INCLUDE_INSTALL_DIR}")
    if(NOT ARG_INCLUDE_INSTALL_DIR)
        if(ARG_INCLUDE_ROOT_DIR)
            set(include_install_dir "${ARG_INCLUDE_ROOT_DIR}")
            message(WARNING "sofa_generate_package(${ARG_NAME}): INCLUDE_ROOT_DIR is deprecated. Please use INCLUDE_INSTALL_DIR instead.")
        else()
            set(include_install_dir "${ARG_NAME}")
        endif()
    endif()

    sofa_install_targets("${ARG_NAME}" "${ARG_TARGETS}" "${include_install_dir}" "${ARG_INCLUDE_SOURCE_DIR}" "${ARG_EXAMPLE_INSTALL_DIR}" "${ARG_RELOCATABLE}")
    sofa_write_package_config_files("${ARG_NAME}" "${ARG_VERSION}")
endfunction()

macro(sofa_create_package package_name version the_targets include_install_dir)
    message(WARNING "Deprecated macro. Use the keyword argument function 'sofa_generate_package' instead")
    # ARGV4 is a non-breaking additional argument to handle INCLUDE_SOURCE_DIR (see sofa_generate_package)
    # TODO: add a real argument "include_source_dir" to this macro
    sofa_generate_package(
        NAME "${package_name}" VERSION "${version}"
        TARGETS "${the_targets}"
        INCLUDE_INSTALL_DIR "${include_install_dir}" INCLUDE_SOURCE_DIR "${ARGV4}"
        EXAMPLE_INSTALL_DIR "${ARGV5}"
        )
endmacro()


# Get path of all library versions (involving symbolic links) for a specified library
function(sofa_install_libraries)
    set(options NO_COPY)
    set(multiValueArgs TARGETS PATHS)
    cmake_parse_arguments("sofa_install_libraries" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    set(no_copy ${sofa_install_libraries_NO_COPY})
    set(targets ${sofa_install_libraries_TARGETS})
    set(lib_paths ${sofa_install_libraries_PATHS})

    foreach(target ${targets})
        get_target_property(target_location ${target} LOCATION_${CMAKE_BUILD_TYPE})
        get_target_property(is_framework ${target} FRAMEWORK)
        if(APPLE AND is_framework)
            get_filename_component(target_location ${target_location} DIRECTORY) # parent dir
            install(DIRECTORY ${target_location} DESTINATION "lib" COMPONENT applications)
        else()
            list(APPEND lib_paths "${target_location}")
        endif()
    endforeach()

    PARSE_LIBRARY_LIST(${lib_paths}
        FOUND   parseOk
        DEBUG   LIBRARIES_DEBUG
        OPT     LIBRARIES_RELEASE
        GENERAL LIBRARIES_GENERAL)

    if(parseOk)
        if(CMAKE_BUILD_TYPE MATCHES DEBUG)
            set(lib_paths ${LIBRARIES_DEBUG})
        else()
            set(lib_paths ${LIBRARIES_RELEASE})
        endif()
    endif()

    foreach(lib_path ${lib_paths})
        if(EXISTS ${lib_path})
            get_filename_component(LIBREAL ${lib_path} REALPATH)
            get_filename_component(LIBREAL_NAME ${LIBREAL} NAME_WE)
            get_filename_component(LIBREAL_PATH ${LIBREAL} PATH)

            # In "${LIBREAL_NAME}." the dot is a real dot, not a regex symbol
            # CMAKE_*_LIBRARY_SUFFIX also start with a dot
            # So regex is:
            # <lib_path> <slash> <library_name> <dot> <dll/so/dylib/...>
            # or:
            # <lib_path> <slash> <library_name> <dot> <anything> <dot> <dll/so/dylib/...>
            file(GLOB_RECURSE SHARED_LIBS
                "${LIBREAL_PATH}/${LIBREAL_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}*" # libtiff.dll
                "${LIBREAL_PATH}/${LIBREAL_NAME}[0-9]${CMAKE_SHARED_LIBRARY_SUFFIX}*"
                "${LIBREAL_PATH}/${LIBREAL_NAME}[0-9][0-9]${CMAKE_SHARED_LIBRARY_SUFFIX}*" # libpng16.dll
                "${LIBREAL_PATH}/${LIBREAL_NAME}.*${CMAKE_SHARED_LIBRARY_SUFFIX}*" # libpng.16.dylib
                )
            file(GLOB_RECURSE STATIC_LIBS
                "${LIBREAL_PATH}/${LIBREAL_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}*"
                "${LIBREAL_PATH}/${LIBREAL_NAME}[0-9]${CMAKE_STATIC_LIBRARY_SUFFIX}*"
                "${LIBREAL_PATH}/${LIBREAL_NAME}[0-9][0-9]${CMAKE_STATIC_LIBRARY_SUFFIX}*"
                "${LIBREAL_PATH}/${LIBREAL_NAME}.*${CMAKE_STATIC_LIBRARY_SUFFIX}*"
                )

            if(WIN32)
                install(FILES ${SHARED_LIBS} DESTINATION "bin" COMPONENT applications)
            else()
                install(FILES ${SHARED_LIBS} DESTINATION "lib" COMPONENT applications)
            endif()
            # install(FILES ${STATIC_LIBS} DESTINATION "lib" COMPONENT libraries)
        endif()
    endforeach()

    if(WIN32 AND NOT no_copy)
        sofa_copy_libraries(PATHS ${lib_paths})
    endif()
endfunction()

function(sofa_install_get_libraries library)
    message(WARNING "sofa_install_get_libraries() is deprecated. Please use sofa_install_libraries() instead.")
    sofa_install_libraries(PATHS ${library})
endfunction()


function(sofa_copy_libraries)
    set(multiValueArgs TARGETS PATHS)
    cmake_parse_arguments("sofa_copy_libraries" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    set(targets ${sofa_copy_libraries_TARGETS})
    set(lib_paths ${sofa_copy_libraries_PATHS})

    foreach(target ${targets})
        if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (MSVC)
            foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
                get_target_property(target_location ${target} LOCATION_${CONFIG})
                list(APPEND lib_paths "${target_location}")
            endforeach()
        else() # Single-config generator (nmake)
            get_target_property(target_location ${target} LOCATION_${CMAKE_BUILD_TYPE})
            list(APPEND lib_paths "${target_location}")
        endif()
    endforeach()

    foreach(lib_path ${lib_paths})
        if(EXISTS ${lib_path})
            get_filename_component(LIB_NAME ${lib_path} NAME_WE)
            get_filename_component(LIB_PATH ${lib_path} PATH)

            file(GLOB SHARED_LIB
                "${LIB_PATH}/${LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}"
                "${LIB_PATH}/${LIB_NAME}[0-9]${CMAKE_SHARED_LIBRARY_SUFFIX}"
                "${LIB_PATH}/${LIB_NAME}[0-9][0-9]${CMAKE_SHARED_LIBRARY_SUFFIX}")

            set(runtime_output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
            if(NOT runtime_output_dir)
                set(runtime_output_dir ${CMAKE_BINARY_DIR}) # fallback
            endif()
            if(NOT EXISTS runtime_output_dir)
                # make sure runtime_output_dir exists before calling configure_file COPYONLY
                # otherwise it will not be treated as a directory
                file(MAKE_DIRECTORY ${runtime_output_dir})
            endif()

            if(EXISTS ${SHARED_LIB})
                if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (Visual Studio)
                    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
                        configure_file(${SHARED_LIB} "${runtime_output_dir}/${CONFIG}" COPYONLY)
                    endforeach()
                else()                        # Single-config generator (nmake, ninja)
                    configure_file(${SHARED_LIB} "${runtime_output_dir}" COPYONLY)
                endif()
            endif()
        endif()
    endforeach()
endfunction()



## to store which sources have been used for installed binaries
## these should be internal files and not delivered, but this is definitively useful
## when storing backups / demos across several repositories (e.g. sofa + plugins)
macro( sofa_install_git_version name sourcedir )
INSTALL( CODE
"
    find_package(Git REQUIRED)

    # the current commit hash should be enough
    # except if the git history changes...
    # so adding more stuff to be sure

    # get the current working branch
    execute_process(
      COMMAND \${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${sourcedir}
      OUTPUT_VARIABLE SOFA_GIT_BRANCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # get the current commit info (hash, author, date, comment)
    execute_process(
      COMMAND \${GIT_EXECUTABLE} log --format=medium -n 1
      WORKING_DIRECTORY ${sourcedir}
      OUTPUT_VARIABLE SOFA_GIT_INFO
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    string( TOLOWER \"${name}\" name_lower )
    if( name_lower STREQUAL \"sofa\" )
        file(WRITE  \"${CMAKE_INSTALL_PREFIX}/git.version\" \"######## ${name} ########\nBranch: \${SOFA_GIT_BRANCH}\n\${SOFA_GIT_INFO}\n############\n\n\" )
    else()
        file(APPEND \"${CMAKE_INSTALL_PREFIX}/git.version\" \"######## ${name} ########\nBranch: \${SOFA_GIT_BRANCH}\n\${SOFA_GIT_INFO}\n############\n\n\" )
    endif()
"
)
endmacro()


function(debug_print_target_properties tgt)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)

    # Convert command output into a CMake list
    STRING(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    STRING(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")

    if(NOT TARGET ${tgt})
      message("There is no target named '${tgt}'")
      return()
    endif()

    foreach(prop ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" prop ${prop})
        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(prop STREQUAL "LOCATION" OR prop MATCHES "^LOCATION_" OR prop MATCHES "_LOCATION$")
            continue()
        endif()
        # message ("Checking ${prop}")
        get_property(propval TARGET ${tgt} PROPERTY ${prop} SET)
        if (propval)
            get_target_property(propval ${tgt} ${prop})
            message ("${tgt} ${prop} = ${propval}")
        endif()
    endforeach(prop)
endfunction()
