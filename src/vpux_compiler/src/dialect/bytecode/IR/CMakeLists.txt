#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_bytecode
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
        PRIVATE
            npu_compiler_dialect_core
        PUBLIC
            npu_compiler_dialect_bytecode_utils
            npu_core_utils
            npu_bytecode_utils
    )
