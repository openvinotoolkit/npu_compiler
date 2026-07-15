#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_elf_transforms
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_utils
        npu_logger
      PRIVATE
        npu_compiler_dialect_elf
        npu_compiler_dialect_elf_utils
        npu_elf
        npu_compiler_dialect_config
        npu_compiler_dialect_core
        npu_compiler_dialect_net
        npu_compiler_dialect_vpuasm
        npu_compiler_conversion
        npu_core_utils
        MLIRFuncDialect
)
