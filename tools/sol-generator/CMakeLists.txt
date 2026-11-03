#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME sol-generator)

set(GFLAGS_USE_TARGET_NAMESPACE ON)
find_package(gflags QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        openvino::runtime
        openvino::gflags_nothreads_static
)
