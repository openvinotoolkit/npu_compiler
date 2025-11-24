#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpu_utils
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_compiler_utils
        npu_llvm_utils
        npu_ov_utils
        npu_profiling_utils
        npu_elf
        npu_compiler_dialect_vpu
        #TODO:#-102154 remove this dependency
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_dialect_vpuip_utils
        npu_costmodel
    DEPENDS
        npu_costmodel)
