#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()

#
# profiling schema target
#

vpux_add_flatc_target(npu_profiling_schema
    SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/schema"
    DST_DIR "${PROJECT_BINARY_DIR}/profiling/generated"
    ARGS "--scoped-enums" "--gen-object-api" "--reflect-names")
set_target_properties(npu_profiling_schema PROPERTIES FOLDER "src")

#
# profiling utils
#

set(TARGET_NAME npu_profiling_utils)

list(APPEND SOURCES
                common.cpp
                metadata.cpp
                taskinfo.cpp
                tasknames.cpp
                utils.cpp
                parser/debug.cpp
                parser/freq.cpp
                parser/parser.cpp
                parser/sync.cpp
                parser/records.cpp
                reports/hooks.cpp
                reports/json.cpp
                reports/layers.cpp
                reports/stats.cpp
                reports/tasklist.cpp
                reports/text.cpp
)

add_library(${TARGET_NAME} STATIC ${SOURCES})

enable_lto_if_possible(${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
    PRIVATE
        "${PROJECT_SOURCE_DIR}/src/vpux_compiler/include"
)

ov_link_system_libraries(${TARGET_NAME}
    PUBLIC
        flatbuffers
        npu_profiling_schema
        npu_logger
        npu_core_utils
    PRIVATE
        npu_elf
        LLVMSupport
        npu_compiler_npu37xx
        npu_compiler_npu40xx
)
