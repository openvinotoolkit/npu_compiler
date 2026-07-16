#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_conversion
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        mlir-dependencies
        npu_compiler_dialect_config
        npu_compiler_dialect_const
        npu_compiler_dialect_core
        npu_compiler_dialect_ie
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpuasm
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpumi40xx
        npu_compiler_dialect_vpureg
        npu_compiler_vpuipdpu_40XX
        npu_compiler_vpuipdpu_50XX
        npu_logger
        npu_compiler_dialect_npureg40xx
        npu_compiler_dialect_npureg50xx
        npu_compiler_dialect_vpurt
        npu_compiler_utils
        npu_compiler_dialect_shave
      PRIVATE
        npu_compiler_core
        npu_elf
        # E#98178: remove dependency on firmware headers
        npu_37xx_firmware_headers
        npu_40xx_firmware_headers
        npu_compiler_dialect_elfnpu37xx
        npu_compiler_dialect_elf_utils
        MLIRMathTransforms
        npu_compiler_dialect_bytecode_utils
        npu_compiler_dialect_elf
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_net
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_vpuip_transforms
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_vpumi37xx
        npu_compiler_npu40xx
        npu_compiler_shave_code_gen
        npu_compiler_act_kernels
        npu_core_utils
        npu_compiler_dialect_bytecode
        npu_compiler_dialect_vpuipdpu
)
