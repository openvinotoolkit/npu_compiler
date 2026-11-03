#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME query_model)

set(GFLAGS_USE_TARGET_NAMESPACE ON)
find_package(gflags QUIET)

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        openvino::runtime
        openvino::runtime::dev
        openvino::gflags_nothreads_static
    )

# Additional linkage required: https://en.cppreference.com/w/cpp/filesystem#:~:text=GNU%20implementation%20prior,lstdc%2B%2Bfs
if ((${CMAKE_CXX_COMPILER_ID} STREQUAL GNU) AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1))
    add_compile_definitions(GNU_LESS_9_1)
    target_link_libraries(${TARGET_NAME} PRIVATE stdc++fs)
endif()
