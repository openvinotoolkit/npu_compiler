#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpurt_interfaces
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_config
        npu_core_utils
        npu_compiler_core
        npu_compiler_dialect_vpurt
        npu_logger
      PRIVATE
        npu_elf
        npu_costmodel
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_vpurt_utils
        npu_compiler_utils
)
enable_lto_if_possible(npu_compiler_dialect_vpurt_interfaces)

