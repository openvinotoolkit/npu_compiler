#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_elf_utils
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_elf
        npu_compiler_dialect_vpuasm
        npu_compiler_dialect_vpureg
    PRIVATE
        npu_core_utils
        npu_compiler_dialect_config
        npu_compiler_dialect_net
        npu_compiler_dialect_vpu
        npu_compiler_act_kernels
)
enable_lto_if_possible(npu_compiler_dialect_elf_utils)

