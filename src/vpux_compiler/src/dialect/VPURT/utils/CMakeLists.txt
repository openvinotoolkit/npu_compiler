#
# Copyright (C) 2024-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpurt_utils
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_compiler_core
        npu_llvm_utils
        npu_ov_utils
        npu_profiling_utils
        flatbuffers
        npu_elf
        npu_compiler_dialect_vpurt_interfaces)
