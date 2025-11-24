#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_elf_transforms
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_llvm_utils
        npu_elf
        MLIRFuncDialect
    )
