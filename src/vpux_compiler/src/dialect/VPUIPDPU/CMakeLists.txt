#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpuipdpu
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_core
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpureg
        npu_compiler_dialect_elf_utils
        npu_compiler_dialect_elf
        npu_compiler_dialect_vpuasm
        npu_logger
      PRIVATE
        npu_compiler_vpuipdpu_37XX # TODO: E120294 - remove arch specific references
        npu_compiler_vpuipdpu_40XX
        npu_compiler_vpuipdpu_50XX
        npu_compiler_dialect_config
        npu_compiler_utils
        npu_compiler_core
        npu_core_utils
)
