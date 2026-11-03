#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dynamic_rewriter
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        mlir-dependencies
        npu_compiler_utils
        npu_core_utils
        npu_logger
      PRIVATE
        npu_compiler_dialect_vpuip
)
enable_lto_if_possible(npu_compiler_dynamic_rewriter)

