#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME npu_compiler_loader_utils)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

target_include_directories(${TARGET_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${VCL_LOADER_SOURCE_DIR}/include>
    )

target_link_libraries(
    ${TARGET_NAME}
    PUBLIC
        npu_logger
    PRIVATE
        npu_ov_utils # OV core function
        openvino::util
)
