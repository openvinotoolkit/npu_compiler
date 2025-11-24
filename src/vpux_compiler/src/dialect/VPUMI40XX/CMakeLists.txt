#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_vpumi40xx
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_llvm_utils
        npu_ov_utils
        npu_elf
        # E#98178: remove dependency on firmware headers
        npu_40xx_firmware_headers
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_vpu_utils
    SYSTEM_INCLUDES
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>)
