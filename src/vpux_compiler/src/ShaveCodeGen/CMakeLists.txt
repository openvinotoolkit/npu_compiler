#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_shave_code_gen
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        mlir-dependencies
        npu_compiler_dialect_ie
        npu_core_utils
        npu_logger
        npu_compiler_utils
      PRIVATE
        MLIRQuantTransforms
        MLIRTensorInferTypeOpInterfaceImpl
        npu_compiler_dialect_net
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_utils
    )
