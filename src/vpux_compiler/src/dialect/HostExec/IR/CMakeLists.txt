#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_dialect_hostexec")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_core
        npu_core_utils
        npu_compiler_core
        npu_compiler_utils
      PRIVATE
        npu_elf
        npu_compiler_dialect_config
        npu_compiler_dialect_const
)
