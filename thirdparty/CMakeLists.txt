#
# Copyright (C) 2022-2026 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

#
# Replace compiler flags
#

foreach(flag IN ITEMS "-Werror" "-fvisibility=hidden" "-fvisibility-inlines-hidden")
    string(REGEX REPLACE "${flag}=?[A-Za-z0-9-]*" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE "${flag}=?[A-Za-z0-9-]*" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "${flag}=?[A-Za-z0-9-]*" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REGEX REPLACE "${flag}=?[A-Za-z0-9-]*" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endforeach()


# TODO: Remove once the thirdparties respect these flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(REPLACE "/sdl" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/sdl" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

set(BUILD_SHARED_LIBS OFF)

if(FORCE_CCACHE_FOR_NPU_THIRDPARTY)
    find_program(CCACHE_PROGRAM NAMES ccache REQUIRED)
    message(STATUS "FORCE_CCACHE_FOR_NPU_THIRDPARTY is ON, using ccache for thirdparty dependencies builds")
    set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

#
# npu_elf
#

if (ENABLE_NPU_MONO)
    if (NOT TARGET npu_elf)
        message(FATAL_ERROR "elf target must exist since ENABLE_NPU_MONO is ON")
    endif()
    # TODO remove:
    # `src/vpux_compiler/include/vpux/compiler/dialect/NPUReg37XX/api`
    # `src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/api`
    # after integration finishes
else()
    # Legacy no-monorepo scenario
    add_subdirectory(elf EXCLUDE_FROM_ALL)
    # elf lib is distributed exernally, therefore place it together with other artifacts in OV bin directory
    if(OV_ARCHIVE_OUTPUT_DIRECTORY)
        set_target_properties(npu_elf PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY ${OV_ARCHIVE_OUTPUT_DIRECTORY})
    endif()
    target_include_directories(npu_elf PRIVATE
        "${PROJECT_SOURCE_DIR}/src/vpux_compiler/include/vpux/compiler/NPU37XX/dialect/NPUReg37XX/firmware_headers/details"
        "${PROJECT_SOURCE_DIR}/src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details")
endif()

#
# nn cost model
#

set(VPUNN_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(VPUNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VPUNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(VPUNN_OPT_LEGACY_ZTILING ON CACHE BOOL "" FORCE)
set(VPUNN_OPT_LEGACY_DMA_TH_4 ON CACHE BOOL "" FORCE)
set(VPUNN_OPT_LEGACY_DMA_TH_5 OFF CACHE BOOL "" FORCE)
add_subdirectory(vpucostmodel EXCLUDE_FROM_ALL)


# remove Intel specific global compile definitions to prevent them affecting thirdparty targets below (especially LLVM/MLIR)
get_directory_property(dir_defs COMPILE_DEFINITIONS)
list(FILTER dir_defs EXCLUDE REGEX "^INTEL_") # ex.: NPU INTEL_EMBARGO compile definitions
list(FILTER dir_defs EXCLUDE REGEX "OV_NATIVE_PARENT_PROJECT_ROOT_DIR") # OV global compile definition
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "${dir_defs}")


#
# LLVM/MLIR
#

if(NOT ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    set_llvm_flags()
    add_subdirectory(llvm-project/llvm EXCLUDE_FROM_ALL)
    set(MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/mlir/include")
    set(MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/tools/mlir/include")
    include_directories(SYSTEM ${MLIR_SOURCE_DIR} ${MLIR_BINARY_DIR})
endif()

#
# flatbuffers
#

include(flatbuffers.cmake)
