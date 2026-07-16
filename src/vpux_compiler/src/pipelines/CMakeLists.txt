#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_pipelines")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        mlir-dependencies
        npu_logger
        npu_compiler_dialect_config
        npu_compiler_dialect_vpu_transforms
        npu_core_utils
        npu_ov_utils
      PRIVATE
        # TODO: see discussion 16934
        npu_compiler_dialect_vpu
        npu_compiler_dialect_hostexec_transforms
        npu_compiler_locverif
        npu_compiler_utils
        npu_compiler_npu37xx
        npu_compiler_npu40xx
        npu_compiler_npu50xx
        npu_compiler_core
        npu_compiler_shave_code_gen
        npu_compiler_dialect_vpuasm
        npu_compiler_dialect_const
        npu_compiler_dialect_core
        npu_compiler_dialect_vpuip_transforms
        npu_compiler_dialect_ie_transforms
        openvino::npu_al
)
