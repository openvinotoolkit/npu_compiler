#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_core
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_config
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_dialect_vpureg
        npu_compiler_dialect_vpurt_utils
        npu_core_utils
        npu_logger
        npu_compiler_dialect_core
        npu_compiler_dialect_net
        npu_compiler_dialect_vpurt
        npu_compiler_utils
        npu_profiling_utils
        npu_costmodel
      PRIVATE
        flatbuffers
        MLIRAsyncDialect
        MLIRControlFlowInterfaces
        MLIRViewLikeInterface
        npu_elf
        npu_compiler_dialect_vpu_utils
        npu_profiling_schema
        npu_compiler_dialect_const
        npu_compiler_dialect_ie
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpuip_transforms
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_vpumi37xx
)
