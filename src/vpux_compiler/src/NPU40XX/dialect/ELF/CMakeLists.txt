#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_elf
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        compiler_commit_hash
        npu_llvm_utils
        npu_ov_utils
        flatbuffers
        npu_elf
        openvino::npu_al
        inferenceStatic
        # E#98178: remove dependency on firmware headers
        npu_40xx_firmware_headers
        MLIRFuncDialect
    SYSTEM_INCLUDES
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>)
