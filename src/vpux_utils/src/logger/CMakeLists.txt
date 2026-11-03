#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()

set(TARGET_NAME npu_logger)

list(APPEND SOURCES
    logger.cpp
    common_logger.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

enable_lto_if_possible(${TARGET_NAME})
target_link_libraries(${TARGET_NAME}
    PUBLIC
        npu_core_utils
    PRIVATE
        LLVMSupport)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
)

target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${LLVM_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${LLVM_BINARY_DIR}/include>
)
