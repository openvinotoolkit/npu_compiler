#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpuipdpu
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_llvm_utils
        npu_ov_utils
        npu_compiler_dialect_elf_utils
        npu_compiler_dialect_vpuasm
        npu_compiler_vpuipdpu_37XX # TODO: E120294 - remove arch specific references
        npu_compiler_vpuipdpu_40XX
        npu_compiler_vpuipdpu_50XX
)
