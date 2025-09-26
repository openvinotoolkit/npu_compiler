#
# Copyright (C) 2022-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

#
# Replace compiler flags
#

foreach(flag IN ITEMS "-Werror" "-fvisibility=hidden" "-fvisibility-inlines-hidden")
    string(REPLACE ${flag} "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE ${flag} "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE ${flag} "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE ${flag} "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endforeach()


# TODO: Remove once the thirdparties respect these flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(REPLACE "/sdl" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/sdl" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    if(NOT BUILD_COMPILER_FOR_DRIVER)
        set(ie_c_cxx_deprecated_no_errors "${ie_c_cxx_deprecated_no_errors} /wd4146 /wd4703")
    endif()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ie_c_cxx_deprecated_no_errors}")
endif()

set(BUILD_SHARED_LIBS OFF)

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

if(TARGET flatbuffers OR TARGET flatc)
    # we are building NPU plugin via -DOPENVINO_EXTRA_MODULES
    # and flatbuffers is already built as part of OpenVINO in case of
    # building in a single tree
    message(WARNING "Flatbuffers target present. Possible version mismatch.")
else()
    set(FLATBUFFERS_BUILD_TESTS OFF)
    set(FLATBUFFERS_INSTALL OFF)
    set(FLATBUFFERS_BUILD_FLATC ON)

    add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)

    # FIXME: this flag is forced while building in a single tree with OV
    # but code cannot be compiled with the flag
    if(NOT MSVC)
        target_compile_options(flatbuffers PRIVATE -Wno-suggest-override)
        target_compile_options(flatc PRIVATE -Wno-suggest-override)
        if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13))
            target_compile_options(flatbuffers PRIVATE -Wno-unused-but-set-variable)
            target_compile_options(flatc PRIVATE -Wno-unused-but-set-variable)
        endif()
    endif()
endif()
set(flatc_TARGET flatc)
set(flatc_COMMAND $<TARGET_FILE:flatc>)
set(flatc_TARGET "${flatc_TARGET}" PARENT_SCOPE)
set(flatc_COMMAND "${flatc_COMMAND}" PARENT_SCOPE)

#
# npu_elf
#

if (ENABLE_NPU_MONO)
    if (NOT TARGET npu_elf)
        message(FATAL_ERROR "elf/vpux_elf target must exist since ENABLE_NPU_MONO is ON")
    endif()
    # TODO remove:
    # `src/vpux_compiler/include/vpux/compiler/dialect/NPUReg37XX/api`
    # `src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/api`
    # after integration finishes
else()
    # Legacy no-monorepo scenario
    add_subdirectory(elf/vpux_elf EXCLUDE_FROM_ALL)
    target_include_directories(npu_elf PRIVATE
        "${PROJECT_SOURCE_DIR}/src/vpux_compiler/include/vpux/compiler/NPU37XX/dialect/NPUReg37XX/firmware_headers/details"
        "${PROJECT_SOURCE_DIR}/src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details")

    target_include_directories(vpux_elf PRIVATE
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
