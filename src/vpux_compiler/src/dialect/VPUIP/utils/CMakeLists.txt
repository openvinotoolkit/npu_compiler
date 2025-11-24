#
# Copyright (C) 2024-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_dialect_vpuip_utils")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_compiler_act_kernels
        npu_compiler_core
        npu_compiler_utils
        npu_llvm_utils
        npu_ov_utils
        npu_profiling_utils
        npu_elf
        openvino::npu_al
        npu_compiler_dialect_vpurt
        npu_compiler_dialect_vpuip
        npu_costmodel)

ov_add_version_defines(${VPUX_COMPILER_VERSION_FILE} ${TARGET_NAME})
