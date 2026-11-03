#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpumi37xx
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_core
        npu_compiler_dialect_core
        npu_compiler_dialect_elfnpu37xx
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpureg
        npu_compiler_dialect_vpurt
        npu_core_utils
        npu_ov_utils
        npu_logger
        npu_compiler_dialect_vpuip
      PRIVATE
        npu_compiler_utils
        npu_elf
        # E#98178: remove dependency on firmware headers
        npu_37xx_firmware_headers
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_elf_utils
        npu_profiling_schema
        npu_compiler_interface
        npu_compiler_dialect_config
        npu_compiler_dialect_const
        npu_compiler_dialect_vpu_utils
        npu_compiler_act_kernels
)
enable_lto_if_possible(npu_compiler_dialect_vpumi37xx)

