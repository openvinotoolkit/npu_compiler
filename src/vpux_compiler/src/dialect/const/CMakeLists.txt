#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_const
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_core
        npu_compiler_dialect_config
        npu_compiler_dialect_core
        npu_compiler_dialect_elfnpu37xx
        npu_compiler_dialect_vpu_utils
        npu_core_utils
        npu_logger
        npu_compiler_utils
        MLIRTransformUtils
    PRIVATE
        # npu_compiler_dialect_const should not depend on other dialects (IE, VPU, VPUIP...),
        # to avoid circular dependencies.
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_utils
        npu_elf
)
