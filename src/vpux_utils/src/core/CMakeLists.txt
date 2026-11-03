#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()

find_package(ZLIB)

set(TARGET_NAME npu_core_utils)

list (APPEND SOURCES
            common_string_utils.cpp
            developer_build_utils.cpp
            type/float8_e4m3.cpp
            type/float8_e5m2.cpp
            type/float8_e8m0.cpp
            type/float16.cpp
            type/bfloat16.cpp
            type/float4_e2m1.cpp
            error.cpp
            mask.cpp
            mem_size.cpp
            simple_math.cpp
            developer_path_utils.cpp
)
if(WIN32)
  list (APPEND SOURCES
            win32/env.cpp
            win32/memory_usage.cpp)
else()
  list (APPEND SOURCES
            unix/env.cpp
            unix/memory_usage.cpp)
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

enable_lto_if_possible(${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

ov_build_target_faster(${TARGET_NAME} PCH PUBLIC "precomp.hpp")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
)

target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        $<BUILD_INTERFACE:${LLVM_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${LLVM_BINARY_DIR}/include>
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        LLVMSupport
    PRIVATE
        npu_logger
)
