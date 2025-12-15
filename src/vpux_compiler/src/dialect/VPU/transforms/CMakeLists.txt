#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpu_transforms
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        MLIRSCFTransformOps
        npu_compiler_utils
        npu_llvm_utils
        npu_ov_utils
        npu_profiling_utils
        npu_elf
        npu_compiler_dialect_config
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_net
        npu_compiler_npu37xx
        npu_compiler_npu40xx
        npu_compiler_npu50xx
        npu_costmodel)
