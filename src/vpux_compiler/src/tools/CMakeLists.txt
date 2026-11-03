#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_tools
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_config
      PRIVATE
        npu_core_utils
)
enable_lto_if_possible(npu_compiler_tools)

