#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_net
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_core
        npu_compiler_utils
        npu_logger
      PRIVATE
        npu_compiler_dialect_config
        npu_compiler_dialect_core
        npu_compiler_dialect_hostexec_transforms
        npu_core_utils
)
enable_lto_if_possible(npu_compiler_dialect_net)

