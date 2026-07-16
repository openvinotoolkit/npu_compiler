#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpurt_utils
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_vpurt
        npu_compiler_utils
        npu_compiler_core
        npu_compiler_dialect_vpurt_interfaces
      PRIVATE
        npu_elf
        npu_compiler_dialect_config
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_utils
)
