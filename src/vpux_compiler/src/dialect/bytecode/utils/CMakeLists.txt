#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_bytecode_utils
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
        PRIVATE
            npu_compiler_utils
        PUBLIC
            npu_core_utils
            npu_compiler_dialect_bytecode
            npu_bytecode_utils
    )
