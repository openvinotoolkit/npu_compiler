#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_elf
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_core
        npu_compiler_dialect_vpurt
        npu_compiler_dialect_config
        npu_core_utils
        npu_logger
      PRIVATE
        npu_compiler_utils
        npu_compiler_dialect_vpuasm
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpureg
        npu_elf
        # E#98178: remove dependency on firmware headers
        npu_40xx_firmware_headers
        npu_compiler_dialect_elf_utils
        npu_compiler_dialect_elfnpu37xx
        npu_compiler_dialect_npureg40xx
        MLIRFuncDialect
        openvino::npu_al
)
