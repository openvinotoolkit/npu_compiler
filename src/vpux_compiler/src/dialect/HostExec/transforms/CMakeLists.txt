#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_dialect_hostexec_transforms")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_core_utils
        npu_logger
        npu_compiler_core
        npu_compiler_utils
        npu_compiler_dialect_elf
        npu_compiler_dialect_config
      PRIVATE
        npu_elf
        openvino::npu_al
        npu_compiler_dialect_hostexec
        npu_compiler_dialect_bytecode_transforms
        npu_compiler_dialect_elfnpu37xx
        npu_compiler_dialect_core
        npu_compiler_dialect_net
        npu_compiler_dialect_vpuasm
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_ie
)
