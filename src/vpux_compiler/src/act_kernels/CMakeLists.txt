#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_act_kernels
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        npu_core_utils
        npu_compiler_dialect_config
      PRIVATE
        npu_elf
        npu_compiler_core
        npu_compiler_dialect_core
        MLIRIR
        MLIRFuncDialect
    TARGET_SOURCES PRIVATE
        $<TARGET_OBJECTS:act_shave_kernels_lib>
        $<TARGET_OBJECTS:rt_kernel_lib>)

target_include_directories(npu_compiler_act_kernels
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>
)
