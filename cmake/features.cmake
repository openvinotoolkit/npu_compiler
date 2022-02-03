# Copyright (C) 2019-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

if(NOT ENABLE_TESTS)
    set(ENABLE_TESTS OFF)
endif()
ie_dependent_option(ENABLE_TESTS "Unit, behavior and functional tests" ${ENABLE_TESTS} "ENABLE_TESTS" OFF)

if(NOT ENABLE_LTO)
    set(ENABLE_LTO OFF)
endif()
ie_dependent_option(ENABLE_LTO "Enable Link Time Optimization" ${ENABLE_LTO} "LINUX OR WIN32;NOT CMAKE_CROSSCOMPILING" OFF)

if(NOT ENABLE_FASTER_BUILD)
    set(ENABLE_FASTER_BUILD OFF)
endif()
ie_dependent_option(ENABLE_FASTER_BUILD "Enable build features (PCH, UNITY) to speed up build time" ${ENABLE_FASTER_BUILD} "CMAKE_VERSION VERSION_GREATER_EQUAL 3.16" OFF)

if(NOT ENABLE_CPPLINT)
    set(ENABLE_CPPLINT OFF)
endif()
ie_dependent_option(ENABLE_CPPLINT "Enable cpplint checks during the build" ${ENABLE_CPPLINT} "UNIX;NOT ANDROID" OFF)

if(NOT ENABLE_CLANG_FORMAT)
    set(ENABLE_CLANG_FORMAT OFF)
endif()
ie_option(ENABLE_CLANG_FORMAT "Enable clang-format checks during the build" ${ENABLE_CLANG_FORMAT})

ie_option(ENABLE_KMB_SAMPLES "Enable KMB samples" OFF)

set(HAVE_HDDL_UNITE_PACKAGE FALSE)
if(X86_64 AND LINUX AND LINUX_OS_NAME STREQUAL "Ubuntu 20.04")
    set(HAVE_HDDL_UNITE_PACKAGE TRUE)
endif()

# TODO: [Track number: S#46168] Investigate hddlunite build issue for ARM
ie_dependent_option(ENABLE_CUSTOM_HDDLUNITE "Use custom build hddlunite" OFF "NOT AARCH64" OFF)

# HDDL2 is not supported for static lib case
ie_dependent_option(ENABLE_HDDL2 "Enable HDDL2 Plugin" ON "HAVE_HDDL_UNITE_PACKAGE OR ENABLE_CUSTOM_HDDLUNITE;BUILD_SHARED_LIBS" OFF)
ie_dependent_option(ENABLE_HDDL2_TESTS "Enable Unit and Functional tests for HDDL2 Plugin" ON "ENABLE_HDDL2;ENABLE_TESTS" OFF)
if(ENABLE_HDDL2)
    add_definitions(-DENABLE_HDDL2)
endif()

ie_dependent_option(ENABLE_MODELS "download all models required for functional testing" ON "ENABLE_FUNCTIONAL_TESTS" OFF)
ie_dependent_option(ENABLE_VALIDATION_SET "download validation_set required for functional testing" ON "ENABLE_FUNCTIONAL_TESTS" OFF)

ie_option(ENABLE_EXPORT_SYMBOLS "Enable compiler -fvisibility=default and linker -export-dynamic options" OFF)

ie_option(ENABLE_MCM_COMPILER_PACKAGE "Enable build of separate mcmCompiler package" OFF)
# MCM compiler is not supported for static lib case
ie_dependent_option(ENABLE_MCM_COMPILER "Enable compilation of mcmCompiler libraries" ON "BUILD_SHARED_LIBS" OFF)

ie_option(ENABLE_MLIR_COMPILER "Enable compilation of vpux_mlir_compiler libraries" ON)

ie_option(BUILD_COMPILER_FOR_DRIVER "Enable build of VPUXCompilerL0" OFF)

ie_dependent_option(ENABLE_DRIVER_COMPILER_ADAPTER "Enable VPUX Compiler inside driver" ON "WIN32;NOT BUILD_COMPILER_FOR_DRIVER" OFF)

# if ENABLE_ZEROAPI_BACKEND=ON, it adds the ze_loader dependency for VPUXCompilerL0
ie_dependent_option(ENABLE_ZEROAPI_BACKEND "Enable zero-api as a plugin backend" ON "NOT AARCH64;NOT BUILD_COMPILER_FOR_DRIVER" OFF)
if(ENABLE_ZEROAPI_BACKEND)
    add_definitions(-DENABLE_ZEROAPI_BACKEND)
endif()

ie_option(ENABLE_DEVELOPER_BUILD "Enable developer build with extra validation/logging functionality" OFF)

if(NOT DEFINED MV_TOOLS_PATH AND DEFINED ENV{MV_TOOLS_DIR} AND DEFINED ENV{MV_TOOLS_VERSION})
    set(MV_TOOLS_PATH $ENV{MV_TOOLS_DIR}/$ENV{MV_TOOLS_VERSION})
endif()

ie_dependent_option(ENABLE_EMULATOR "Enable emulator as a plugin backend" OFF "MV_TOOLS_PATH" OFF)
if(ENABLE_EMULATOR)
    add_definitions(-DENABLE_EMULATOR)
endif()

ie_dependent_option(ENABLE_IMD_BACKEND "Enable InferenceManagerDemo based VPUX AL backend" OFF "NOT WIN32;NOT CMAKE_CROSSCOMPILING" OFF)
if(ENABLE_IMD_BACKEND)
    add_definitions(-DENABLE_IMD_BACKEND)
endif()

ie_option(ENABLE_VPUX_DOCS "Documentation for VPUX plugin" OFF)

if(ENABLE_VPUX_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/docs/VPUX_DG/Doxyfile.in)
        set(DOXYGEN_OUT ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/docs/VPUX_DG/generated/Doxyfile)

        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message("Doxygen build started")

        add_custom_target(vpux_plugin_docs ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
    else()
        message(WARNING "Doxygen need to be installed to generate the doxygen documentation")
    endif()
endif()

function (print_enabled_kmb_features)
    message(STATUS "KMB Plugin enabled features: ")
    message(STATUS "")
    foreach(var IN LISTS IE_OPTIONS)
        message(STATUS "    ${var} = ${${var}}")
    endforeach()
    message(STATUS "")
endfunction()
