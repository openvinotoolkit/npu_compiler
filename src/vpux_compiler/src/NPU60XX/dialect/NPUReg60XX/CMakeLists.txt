#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_npureg60xx
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_config
        npu_compiler_dialect_elf
        npu_compiler_dialect_vpuasm
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpureg
        npu_compiler_utils
        npu_core_utils
        npu_compiler_dialect_elf_utils
      PRIVATE
        npu_elf
        npu_60xx_firmware_headers
        npu_compiler_dialect_core
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_vpumi40xx
        npu_compiler_act_kernels
)
enable_lto_if_possible(npu_compiler_dialect_npureg60xx)

