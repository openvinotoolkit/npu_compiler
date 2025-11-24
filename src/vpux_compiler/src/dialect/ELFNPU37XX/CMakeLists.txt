#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_elfnpu37xx
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_llvm_utils
        npu_ov_utils
        npu_elf
        openvino::npu_al
        # E#98178: remove dependency on firmware headers
        npu_37xx_firmware_headers
        npu_compiler_dialect_vpurt
        npu_profiling_schema)
