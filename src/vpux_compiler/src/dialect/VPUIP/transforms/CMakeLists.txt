#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_dialect_vpuip_transforms")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_const
        npu_compiler_dialect_vpu_utils
        npu_compiler_dynamic_rewriter
        npu_logger
        npu_compiler_core
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpurt
        npu_costmodel
      PRIVATE
        npu_compiler_utils
        npu_profiling_utils
        npu_elf
        npu_compiler_dialect_elf_utils
        npu_compiler_dialect_config
        npu_compiler_dialect_core
        npu_compiler_dialect_ie
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_net
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_vpureg
        npu_compiler_dialect_vpurt_interfaces
        npu_compiler_dialect_vpurt_utils
        npu_core_utils
)
enable_lto_if_possible(${TARGET_NAME})


ov_add_version_defines(${VPUX_COMPILER_VERSION_FILE} ${TARGET_NAME})
