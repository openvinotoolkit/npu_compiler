#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpu_transforms
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_core
        npu_compiler_dialect_config
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_utils
        npu_compiler_utils
        npu_core_utils
        npu_logger
        npu_costmodel
      PRIVATE
        MLIRSCFTransformOps
        npu_profiling_utils
        npu_elf
        npu_compiler_dialect_net
        npu_compiler_npu37xx
        npu_compiler_npu40xx
        npu_compiler_npu50xx
        npu_compiler_conversion
        npu_compiler_dialect_const
        npu_compiler_dialect_core
        npu_compiler_dialect_elf
        npu_compiler_dialect_hostexec
        npu_compiler_dialect_ie
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_npureg37xx
        npu_compiler_dialect_npureg40xx
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_shave
)
