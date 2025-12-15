#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_npureg40xx
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_compiler_utils
        npu_llvm_utils
        npu_ov_utils
        npu_elf
        npu_40xx_firmware_headers
        npu_compiler_dialect_elf_utils
        npu_compiler_dialect_vpurt
        npu_compiler_dialect_vpu_utils)
