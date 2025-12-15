#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_conversion
    ${SRC_FILES}
    LINK_LIBS
    PRIVATE
        npu_compiler_core
        npu_compiler_utils
        npu_ov_utils
        npu_llvm_utils
        npu_profiling_utils
        npu_elf
        # E#98178: remove dependency on firmware headers
        npu_37xx_firmware_headers
        npu_40xx_firmware_headers
        npu_compiler_dialect_npureg40xx
        npu_compiler_dialect_npureg50xx
        npu_compiler_dialect_elfnpu37xx
        npu_compiler_dialect_elf_utils
        npu_compiler_dialect_vpurt
        MLIRMathTransforms
    PUBLIC
        mlir-dependencies
    SYSTEM_INCLUDES
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>)
