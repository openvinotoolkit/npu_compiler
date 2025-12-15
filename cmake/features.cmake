#
# Copyright (C) 2022-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

# private tests should be disabled when building in OV CI
if(NOT ENABLE_TESTS)
    set(ENABLE_PRIVATE_TESTS OFF)
else()
    if (NOT ENABLE_PRIVATE_TESTS AND ${PROJECT_BINARY_DIR} MATCHES "build-modules")
        set(ENABLE_PRIVATE_TESTS OFF)
    else()
        set(ENABLE_PRIVATE_TESTS ON)
    endif()
endif()
ov_option(ENABLE_PRIVATE_TESTS "NPU private unit, behavior and functional tests" OFF)

ov_dependent_option(ENABLE_NPU_FUZZ_TESTS "NPU Fuzz tests" OFF "ENABLE_TESTS" OFF)

if(NOT ENABLE_LTO)
    set(ENABLE_LTO OFF)
endif()
ov_dependent_option(ENABLE_LTO "Enable Link Time Optimization" ${ENABLE_LTO} "LINUX OR WIN32" OFF)

if(NOT ENABLE_FASTER_BUILD)
    set(ENABLE_FASTER_BUILD OFF)
endif()
ov_dependent_option(ENABLE_FASTER_BUILD "Enable build features (PCH, UNITY) to speed up build time" ${ENABLE_FASTER_BUILD} "CMAKE_VERSION VERSION_GREATER_EQUAL 3.16" OFF)

if(NOT ENABLE_CPPLINT)
    set(ENABLE_CPPLINT OFF)
endif()
ov_dependent_option(ENABLE_CPPLINT "Enable cpplint checks during the build" ${ENABLE_CPPLINT} "UNIX;NOT ANDROID" OFF)

ov_option(ENABLE_EXPORT_SYMBOLS "Enable compiler -fvisibility=default and linker -export-dynamic options" OFF)

ov_option(ENABLE_MLIR_COMPILER "Enable compilation of npu_mlir_compiler libraries" ON)

ov_option(BUILD_COMPILER_FOR_DRIVER "Enable build of VPUXCompilerL0" OFF)

ov_option(ENABLE_PRODUCTION_BUILD "Enable production build with extra control" OFF)

ov_dependent_option(ENABLE_DRIVER_COMPILER_ADAPTER "Enable VPUX Compiler inside driver" ON "NOT BUILD_COMPILER_FOR_DRIVER" OFF)

if(NOT BUILD_SHARED_LIBS AND NOT ENABLE_MLIR_COMPILER AND NOT ENABLE_DRIVER_COMPILER_ADAPTER)
    message(FATAL_ERROR "No compiler found for static build!")
endif()

ov_option(ENABLE_DEVELOPER_BUILD "Enable developer build with extra validation/logging functionality" OFF)

if(NOT DEFINED ENABLE_PRIVATE_COMPILER_OPTIONS)
    set(ENABLE_PRIVATE_COMPILER_OPTIONS ${ENABLE_DEVELOPER_BUILD})
endif()
ov_option(ENABLE_PRIVATE_COMPILER_OPTIONS "Enable the private compiler options, which are accessible via the NPU_COMPILATION_MODE_PARAMS property" ${ENABLE_PRIVATE_COMPILER_OPTIONS})
if(ENABLE_PRIVATE_COMPILER_OPTIONS)
    add_definitions(-DPRIVATE_COMPILER_OPTIONS_ENABLED)
endif()

ov_option(ENABLE_NPU_LOADER "Enable npu-loader" OFF)
ov_option(ENABLE_NPU_LSP_SERVER "Enable npu-lsp-server" OFF)

get_target_property(ov_linked_libs openvino::runtime IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE)
if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO" OR "TBB::tbb" IN_LIST ov_linked_libs)
    set(TBB_AVAILABLE ON)
endif()
ov_dependent_option(ENABLE_BACKGROUND_FOLDING "Enable constant folding in background" ON "TBB_AVAILABLE" OFF)
if(ENABLE_BACKGROUND_FOLDING)
    add_definitions(-DBACKGROUND_FOLDING_ENABLED)
endif()

ov_option(ENABLE_SOURCE_PACKAGE "Enable generation of source code package" OFF)

ov_option(ENABLE_VPUX_DOCS "Documentation for VPUX plugin" OFF)

ov_option(ENABLE_DIALECT_SHARED_LIBRARIES "Enable exporting vpux dialects as shared libraries" OFF)

ov_option(ENABLE_SPLIT_DWARF "Use -gsplit-dwarf when compiling the project and --gdb-index when linking" OFF)

ov_option(LIT_TESTS_USE_LINKS "Create symlink to lit-tests in the binary directory instead of copying them" OFF)

if(NOT DEFINED ENABLE_NPU_MICRO_BENCHMARKS)
    set(ENABLE_NPU_MICRO_BENCHMARKS ${ENABLE_DEVELOPER_BUILD})
endif()
ov_option(ENABLE_NPU_MICRO_BENCHMARKS "NPU micro benchmarks" ${ENABLE_NPU_MICRO_BENCHMARKS})

if(ENABLE_VPUX_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs/VPUX_DG/Doxyfile.in)
        set(DOXYGEN_OUT ${PROJECT_SOURCE_DIR}/docs/VPUX_DG/generated/Doxyfile)

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

function (print_enabled_npu_features)
    message(STATUS "NPU Plugin enabled features: ")
    message(STATUS "")
    foreach(var IN LISTS OV_OPTIONS)
        message(STATUS "    ${var} = ${${var}}")
    endforeach()
    message(STATUS "")
endfunction()
