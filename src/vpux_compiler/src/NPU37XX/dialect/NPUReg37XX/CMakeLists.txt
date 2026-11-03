#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_npureg37xx
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_vpureg
        npu_core_utils
        MLIRFuncDialect
      PRIVATE
        npu_elf
        MLIRIR
)
enable_lto_if_possible(npu_compiler_dialect_npureg37xx)

