#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_dialect_vpuip_utils")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_config
        npu_compiler_dialect_const
        npu_compiler_dialect_core
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_utils
        npu_core_utils
        npu_logger
        npu_compiler_core
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_dialect_vpurt
        npu_compiler_utils
        npu_costmodel
      PRIVATE
        npu_elf
        openvino::npu_al
        npu_compiler_dialect_ie
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_net
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_shave
)

ov_add_version_defines(${VPUX_COMPILER_VERSION_FILE} ${TARGET_NAME})
