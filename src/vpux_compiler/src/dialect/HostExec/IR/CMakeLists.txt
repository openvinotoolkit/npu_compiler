#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_dialect_hostexec")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_compiler_core
        npu_compiler_utils
        npu_llvm_utils
        npu_ov_utils
        npu_elf
        openvino::npu_al)
