#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpu_interfaces
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_core
        npu_compiler_dialect_config
        npu_compiler_dialect_core
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_vpu_utils
        npu_core_utils
        npu_logger
        npu_compiler_dialect_vpu
        npu_compiler_utils
        npu_costmodel
      PRIVATE
        npu_elf
        #TODO:#-95278 remove dependency on VPUIP
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_dialect_ie
        npu_compiler_dialect_vpuip_utils
)
