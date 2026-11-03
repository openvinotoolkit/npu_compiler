#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME prof_parser)

set(GFLAGS_USE_TARGET_NAMESPACE ON)
find_package(gflags QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        $<TARGET_PROPERTY:flatbuffers,INTERFACE_INCLUDE_DIRECTORIES>
    LINK_LIBRARIES
        flatbuffers
        openvino::gflags_nothreads_static
        npu_profiling_utils
        openvino::runtime::dev
        npu_profiling_schema
)
