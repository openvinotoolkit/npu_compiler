#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(ZLIB)

set(TARGET_NAME npu_algo_utils)

list(APPEND SOURCES
                simulated_annealing.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

enable_lto_if_possible(${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PUBLIC npu_core_utils)
