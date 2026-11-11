#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpuasm
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_const
        npu_compiler_dialect_core
        npu_compiler_dialect_elf
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpuipdpu
        npu_compiler_dialect_vpureg
        npu_compiler_dialect_vpurt
        npu_logger
        npu_compiler_dialect_elf_utils
        npu_compiler_dialect_vpumi40xx
      PRIVATE
        compiler_commit_hash
        npu_compiler_utils
        npu_elf
        # E#98178: remove dependency on firmware headers
        npu_40xx_firmware_headers
        npu_60xx_firmware_headers
        npu_compiler_dialect_config
        npu_compiler_dialect_elfnpu37xx
        npu_compiler_dialect_net
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_vpumi37xx
        npu_compiler_act_kernels
        npu_core_utils
)
enable_lto_if_possible(npu_compiler_dialect_vpuasm)

