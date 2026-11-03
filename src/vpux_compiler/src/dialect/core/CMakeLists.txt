#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_core
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_core
        npu_core_utils
        npu_logger
        npu_compiler_utils
        MLIRTransformUtils
        openvino::core::dev
      PRIVATE
        # npu_compiler_dialect_core should not depend on other dialects (IE, VPU, VPUIP...),
        # to avoid circular dependencies.
        npu_compiler_dialect_config
        npu_compiler_dialect_const
        npu_compiler_dialect_ie
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_net
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_utils
        npu_compiler_act_kernels
        npu_elf
)
enable_lto_if_possible(npu_compiler_dialect_core)

