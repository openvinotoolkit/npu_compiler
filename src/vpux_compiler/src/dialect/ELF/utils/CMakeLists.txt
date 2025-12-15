#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_elf_utils
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_core_utils
        npu_compiler_act_kernels)
